"""云端单次运行(GitHub Actions 每 15 分钟调用一次)。

无数据库:每次都从 OKX 重新拉取约 37 天的小时数据来计算 30 天 z-score。
跨次运行需要记住的东西(规则冷却时间、价位提醒状态)放在 state.json 里,
由工作流从 data 分支取回、运行完再推回去。

输出(写到 --out 目录,工作流会强制推送到 data 分支):
  signals.json  当前快照:大盘、自选、正在触发的全部规则、独立行情榜
  events.json   最近 7 天的"新事件"(过了冷却期的新规则触发、价位穿越),Claude 定时任务读这个来推送
  state.json    冷却与价位状态
  status.md     方便在 GitHub 网页上直接查看的中文摘要
  archive.csv.gz   全市场小时特征样本库(最近 180 天),历史概率用,随时间增长
  predictions.json 每条预警当时的历史概率和到期后的真实结果(记分卡)
  ledger.csv    信号后验记录表:每条推送的信号一行(规则、得分、24h/72h/1 周/2 周真实收益与回撤),永久累积
  opp_log.csv.gz          机会榜预测留档(每 6 小时一次,含方向分与截面排名,保留 120 天)
  opp_outcomes.csv.gz     机会榜预测的实盘结果:72h / 1 周 / 2 周超额收益与不利变动,永久累积
  opp_direction_daily.csv 按天汇总的实盘方向成绩(IC、偏涨减偏跌、命中率 vs 同期基准)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

from cryptoradar import foresight as fs
from cryptoradar import opportunity as opp
from cryptoradar.config import load_config
from cryptoradar.features import build_features
from cryptoradar.okx_api import HOUR_MS, OKX, OKXBlockedError
from cryptoradar.okx_collect import collect
from cryptoradar.signals import apply_weights, describe, evaluate_last, merged_thresholds
from cryptoradar.universe import DEFAULT_EXCLUDE, _looks_like_stable, fetch_coingecko_top

log = logging.getLogger("cloud")

FEATURE_KEYS = ["close", "ret_24h", "resid_24h_z", "ret_1h_z", "oi_chg_24h", "oi_z", "funding",
                "funding_z", "vol_z", "top_ls", "top_ls_z", "taker_z", "adr_14d", "beta"]
LS_LABEL = "多空账户比"
MODEL_PATH = Path(__file__).with_name("models") / "opportunity_price.joblib"
_BUNDLE: dict = {}


def compute_opportunity(cfg: dict, frames: dict, uni: list, prev_log, combined: dict,
                        rules: dict | None = None):
    """72 小时机会模型:波动/回撤/概率/方向分。任何一步出错都不能影响主扫描,调用方会兜底。
    返回 (机会榜, 预测留档, 72h 实盘核对, 已到期留档的 72h/1 周/2 周结果)。"""
    oc = cfg.get("opportunity") or {}
    if oc.get("enabled", True) is False:
        return None, prev_log, None, None
    path = Path(oc.get("model") or MODEL_PATH)
    if not path.is_absolute():
        path = Path(__file__).with_name(str(path))
    if str(path) not in _BUNDLE:
        _BUNDLE[str(path)] = opp.load_bundle(path)
    bundle = _BUNDLE[str(path)]
    # 训练用的全是收盘完整的 K 线;OKX 返回的最后一根可能还没收盘(几十分钟的成交额、波动都偏小),去掉它再打分
    now = now_ms()
    done = {k: (v.iloc[:-1] if len(v) and int(v.index[-1]) + HOUR_MS > now else v) for k, v in frames.items()}
    block, R = opp.cloud_opportunity(bundle, done, uni, int(oc.get("topk", 8)), now_ms=now,
                                     min_age_days=float(oc.get("min_age_days", 30)),
                                     young_age_days=float(oc.get("young_age_days", 60)),
                                     young_dd_mult=float(oc.get("young_dd_mult", 1.3)), rules=rules)
    log_df = opp.log_snapshot(prev_log, R)
    resolved = opp.resolve_log(log_df, combined, opp.HORIZONS)
    live = opp.live_summary(resolved, bundle["models"].thr)
    return block, log_df, live, resolved


def now_ms() -> int:
    return int(time.time() * 1000)


def _num(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) or pd.isna(x) else float(x)


def load_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


PAPRIKA = "https://api.coinpaprika.com/v1/tickers"


def ranked_coins(ucfg: dict) -> tuple[list[dict], str]:
    """市值排名列表 [{symbol, rank, current_price, market_cap}]。CoinGecko 失败时换 CoinPaprika。"""
    for attempt in range(3):
        try:
            coins = fetch_coingecko_top(250, ucfg.get("coingecko_api_key") or None)
            return [{"symbol": (c.get("symbol") or "").upper(), "rank": c.get("market_cap_rank"),
                     "current_price": c.get("current_price"), "market_cap": c.get("market_cap")}
                    for c in coins], "coingecko"
        except Exception as e:
            log.warning("CoinGecko 第 %d 次失败:%s", attempt + 1, e)
            time.sleep(10 * (attempt + 1))
    try:
        r = requests.get(PAPRIKA, params={"quotes": "USD"}, timeout=40)
        r.raise_for_status()
        rows = sorted((x for x in r.json() if x.get("rank")), key=lambda x: x["rank"])[:400]
        return [{"symbol": (x.get("symbol") or "").upper(), "rank": x["rank"],
                 "current_price": (x.get("quotes") or {}).get("USD", {}).get("price"),
                 "market_cap": (x.get("quotes") or {}).get("USD", {}).get("market_cap")} for x in rows], "coinpaprika"
    except Exception as e:
        log.warning("CoinPaprika 也失败:%s", e)
    return [], "none"


def build_universe(okx: OKX, cfg: dict, cached: dict | None = None) -> tuple[list[dict], dict]:
    ucfg = cfg["universe"]
    top_n = int(ucfg.get("top_n", 150))
    swaps = okx.usdt_swaps()
    exclude = DEFAULT_EXCLUDE | {s.upper() for s in ucfg.get("exclude", [])}
    watch = [s.upper() for s in ucfg.get("watchlist", [])]

    coins, source = ranked_coins(ucfg)
    if coins:
        cache = {"ts": now_ms(), "source": source, "coins": coins}
    elif cached and cached.get("coins"):
        log.warning("排名数据源都失败,沿用 %s 的缓存", cached.get("source"))
        cache = dict(cached)
        cache["source"] = f"cache({cached.get('source')})"
        coins = cached["coins"]
    else:
        cache = {"ts": 0, "source": "none", "coins": []}

    rank_of, mcap_of = {}, {}
    for c in coins:
        rank_of.setdefault(c["symbol"], c.get("rank"))
        mcap_of.setdefault(c["symbol"], c.get("market_cap"))
    rows, seen = [], set()
    for c in coins:
        sym = c["symbol"]
        if not c.get("rank") or c["rank"] > top_n:
            continue
        if not sym or sym in exclude or _looks_like_stable(c) or sym in seen or sym not in swaps:
            continue
        seen.add(sym)
        rows.append({"ccy": sym, "inst": swaps[sym], "rank": c["rank"], "watch": sym in watch,
                     "mcap": c.get("market_cap"), "list_ms": getattr(okx, "list_ms", {}).get(sym)})
    for sym in watch + ["BTC", "ETH"]:
        if sym in seen:
            continue
        if sym not in swaps:
            log.warning("%s 在 OKX 没有 USDT 永续,跳过", sym)
            continue
        seen.add(sym)
        rows.append({"ccy": sym, "inst": swaps[sym], "rank": rank_of.get(sym), "watch": sym in watch,
                     "mcap": mcap_of.get(sym), "list_ms": getattr(okx, "list_ms", {}).get(sym)})
    log.info("监控名单 %d 个(排名来源:%s)", len(rows), cache["source"])
    return rows, cache


def run(cfg: dict, okx: OKX, prev_state: dict, prev_events: list, tz,
        prev_archive: pd.DataFrame | None = None, prev_preds: list | None = None,
        prev_opp_log: pd.DataFrame | None = None, extras: dict | None = None):
    t0 = time.time()
    th = merged_thresholds(cfg["signals"].get("thresholds"))
    sc = cfg["signals"]
    cooldown = float(sc.get("cooldown_hours", 6)) * HOUR_MS
    now = now_ms()

    uni, uni_cache = build_universe(okx, cfg, prev_state.get("universe"))
    order = sorted(uni, key=lambda u: (u["ccy"] not in ("BTC", "ETH"),))  # 先取 BTC/ETH
    data, failed = {}, []
    for i, u in enumerate(order, 1):
        try:
            data[u["ccy"]] = collect(okx, u["ccy"], u["inst"])
        except OKXBlockedError:
            raise
        except Exception as e:
            failed.append(u["ccy"])
            log.warning("%s 采集失败:%s", u["ccy"], e)
        if i % 20 == 0:
            log.info("采集进度 %d/%d(%.0f 秒)", i, len(order), time.time() - t0)

    if "BTC" not in data or data["BTC"][0].empty:
        raise RuntimeError("BTC 数据获取失败,无法计算残差收益")
    btc, eth = data["BTC"][0], data.get("ETH", (pd.DataFrame(),))[0]

    rules = apply_weights(sc.get("rule_weights"))
    results, frames = [], {}
    for u in uni:
        if u["ccy"] not in data:
            continue
        df, fund, live = data[u["ccy"]]
        if len(df) < 200:
            continue
        try:
            f = build_features(df, btc, fund, live.get("funding_interval_h") or 8.0, eth, live)
        except Exception as e:
            failed.append(u["ccy"])
            log.warning("%s 特征计算失败:%s", u["ccy"], e)
            continue
        results.append((u, f.iloc[-1], evaluate_last(f, th, rules)))
        frames[u["ccy"]] = f

    # 历史概率与市场状态(样本 = 本轮拉到的约 37 天 + data 分支里积累的样本库)
    combined = fs.merge_archive(prev_archive, frames)
    labeled = fs.label_frames(combined, th)
    watch_syms = [u["ccy"] for u in uni if u["watch"]]
    rates = fs.base_rates(labeled, watch_syms)
    mframe = fs.market_frame(combined)
    mrates = fs.market_rates(mframe) if not mframe.empty else {}
    mstate = fs.market_summary(mframe, mrates)
    preds = list(prev_preds or [])

    def outlook(sym: str, fired) -> tuple[str, str | None, dict | None, str]:
        cid, st, scope = fs.pick_outlook([r.id for r in fired], sym, rates)
        if not cid:
            return "", None, None, ""
        return "- 历史:" + fs.outlook_line(cid, st, rates["baseline"], scope), cid, st, scope

    last_fire = dict(prev_state.get("last_fire", {}))
    new_events, firing = [], []
    for u, row, fired in results:
        if not fired:
            continue
        score = sum(r.weight for r in fired)
        text = describe(u["ccy"], row, fired, u["rank"], u["watch"], LS_LABEL)
        ol, cid, st, scope = outlook(u["ccy"], fired)
        if ol:
            text += "\n" + ol
        firing.append({"symbol": u["ccy"], "rank": u["rank"], "watch": u["watch"],
                       "rules": [r.id for r in fired], "score": score, "text": text})
        fresh = [r for r in fired if now - int(last_fire.get(f"{u['ccy']}|{r.id}", 0)) > cooldown]
        need = sc.get("watchlist_min_score", 1.0) if u["watch"] else sc.get("min_score_to_push", 2.5)
        if fresh and score >= need:
            new_events.append({
                "id": f"{now}-{u['ccy']}", "ts": now, "type": "signal", "symbol": u["ccy"],
                "rank": u["rank"], "watch": u["watch"], "rules": [r.id for r in fired],
                "rule_names": [r.name for r in fired], "new_rules": [r.id for r in fresh],
                "score": score, "price": _num(row.get("close")), "text": text,
                "features": {k: _num(row.get(k)) for k in FEATURE_KEYS},
                "outlook": {"cond": cid, "scope": scope, **(st or {})} if cid else None,
                "call": fs.call_from(st, rates.get("baseline"), cid) if cid else "none",
            })
            c0 = cid or fired[0].id
            preds.append(fs.new_prediction(
                now, "signal", u["ccy"], c0, int(row.name), _num(row.get("close")),
                fs.call_from(st, rates.get("baseline"), cid) if cid else "none",
                (st or {}).get("up72"), (rates.get("baseline") or {}).get("up72"),
                (st or {}).get("med72"), (st or {}).get("n", 0), scope,
                rules=[r.id for r in fired], score=score, watch=u["watch"]))
            for r in fired:
                last_fire[f"{u['ccy']}|{r.id}"] = now

    # 价位提醒(穿越时触发一次)
    prices = {u["ccy"]: _num(row.get("close")) for u, row, _ in results}
    pa_state = dict(prev_state.get("price_alerts", {}))
    levels = []
    for pa in cfg.get("price_alerts") or []:
        sym = str(pa.get("symbol", "")).upper().replace("USDT", "").replace("-SWAP", "").strip("-")
        p = prices.get(sym)
        for side in ("below", "above"):
            if side not in pa or p is None:
                continue
            level = float(pa[side])
            hit = p <= level if side == "below" else p >= level
            key = f"{sym}|{side}|{level}"
            levels.append({"symbol": sym, "side": side, "level": level, "price": p, "hit": hit,
                           "note": pa.get("note", "")})
            if hit and not pa_state.get(key, False):
                word = "跌破" if side == "below" else "突破"
                new_events.append({"id": f"{now}-{key}", "ts": now, "type": "price", "symbol": sym,
                                   "price": p, "level": level, "side": side,
                                   "text": f"{sym} 现价 {p:.6g} 已{word} {level:g}:{pa.get('note', '')}"})
            pa_state[key] = bool(hit)

    # 资金费率提醒(8h 口径;越过阈值触发一次,回到阈值内后重新生效,同一阈值受冷却期限制)
    funds = {u["ccy"]: _num(row.get("funding")) for u, row, _ in results}
    fa_state = dict(prev_state.get("funding_alerts", {}))
    fund_levels = []
    for fa in cfg.get("funding_alerts") or []:
        sym = str(fa.get("symbol", "")).upper().replace("USDT", "").replace("-SWAP", "").strip("-")
        r = funds.get(sym)
        if r is None:
            continue
        pos = float(fa.get("position_usdt") or 0)
        daily = r * 3 * pos if pos else None  # 正数 = 多头付出
        for side in ("above", "below"):
            if side not in fa:
                continue
            level = float(fa[side])
            hit = r >= level if side == "above" else r <= level
            key = f"{sym}|{side}|{level}"
            fund_levels.append({"symbol": sym, "side": side, "level": level, "funding": r, "hit": hit,
                                "position_usdt": pos or None, "daily_cost": daily, "note": fa.get("note", "")})
            cool_key = f"{sym}|FUNDALERT|{side}|{level}"
            if hit and not fa_state.get(key, False) and now - int(last_fire.get(cool_key, 0)) > cooldown:
                word = "升至" if side == "above" else "降至"
                text = f"{sym} 资金费率{word} {r * 100:.4f}%/8h(阈值 {level * 100:.3f}%)"
                if daily is not None:
                    text += (f",按 {pos:,.0f} USDT 多仓每天约付 {daily:.2f} USDT" if daily >= 0
                             else f",按 {pos:,.0f} USDT 多仓每天约收 {-daily:.2f} USDT")
                if fa.get("note"):
                    text += f":{fa['note']}"
                new_events.append({"id": f"{now}-fund-{key}", "ts": now, "type": "funding", "symbol": sym,
                                   "funding": r, "level": level, "side": side, "daily_cost": daily,
                                   "text": text})
                last_fire[cool_key] = now
            fa_state[key] = bool(hit)

    # 回购收益率 = 年化回购额 ÷ 当前流通市值(回购额是手填的估算,见 cloud_config.yaml)
    mcaps = {u["ccy"]: u.get("mcap") for u in uni}
    eth_px = prices.get("ETH")
    buybacks = []
    for b in cfg.get("buybacks") or []:
        sym = str(b.get("symbol", "")).upper().replace("USDT", "").replace("-SWAP", "").strip("-")
        annual = b.get("annual_usd")
        if annual is None and b.get("annual_eth") is not None and eth_px:
            annual = float(b["annual_eth"]) * eth_px
        mc = _num(mcaps.get(sym))
        buybacks.append({"symbol": sym, "annual_usd": _num(annual), "market_cap": mc,
                         "yield": _num(annual / mc) if annual and mc else None,
                         "price": prices.get(sym), "basis": b.get("basis", "")})
    buybacks.sort(key=lambda x: -(x["yield"] if x["yield"] is not None else -1))

    # 市场红绿灯:状态切换时生成事件,并记入记分卡
    prev_ms = prev_state.get("market_state")
    if mstate and mstate["state"] != prev_ms:
        new_events.append({"id": f"{now}-market-{mstate['state']}", "ts": now, "type": "market",
                           "symbol": "MARKET", "state": mstate["state"], "prev_state": prev_ms,
                           "text": fs.market_text(mstate)})
        mst, mbase = mrates.get(mstate["state"]), mrates.get("ALL")
        t_bar = int(mframe.dropna(subset=["state"]).index[-1])
        preds.append(fs.new_prediction(now, "market", "ALT", mstate["state"], t_bar, None,
                                       fs.market_call(mst, mbase), (mst or {}).get("alt_up72"),
                                       (mbase or {}).get("alt_up72"), (mst or {}).get("alt_med72"),
                                       (mst or {}).get("n", 0)))
    preds = fs.resolve(preds, combined, mframe, now)
    sc = fs.scorecard(preds, now)
    opp_block, opp_log, opp_live, opp_resolved = None, prev_opp_log, None, None
    try:
        fired_ids = {u["ccy"]: [r.id for r in fired] for u, _, fired in results}
        opp_block, opp_log, opp_live, opp_resolved = compute_opportunity(cfg, frames, uni, prev_opp_log, combined,
                                                                         fired_ids)
    except Exception as e:      # 模型出错不能拖垮主扫描
        log.warning("机会模型失败:%s", traceback.format_exc())
        opp_block = {"error": f"{type(e).__name__}: {e}"}
    if extras is not None:
        extras["opp_log"] = opp_log
        extras["opp_resolved"] = opp_resolved
    archive = fs.archive_table(combined, now)

    btc_row = next((row for u, row, _ in results if u["ccy"] == "BTC"), None)
    market = {}
    if btc_row is not None:
        market = {"btc_price": _num(btc_row["close"]), "btc_ret_24h": _num(np.expm1(btc_row["ret_24h"])),
                  "ethbtc_ret_24h": _num(np.expm1(btc_row.get("ethbtc_ret_24h", np.nan)))}
    movers = sorted([(u, row) for u, row, _ in results if pd.notna(row.get("resid_24h_z"))],
                    key=lambda x: -abs(x[1]["resid_24h_z"]))[:8]
    local = datetime.fromtimestamp(now / 1000, tz=timezone.utc).astimezone(tz)

    signals = {
        "generated_at": now,
        "generated_at_local": local.strftime("%Y-%m-%d %H:%M %Z"),
        "runtime_sec": round(time.time() - t0, 1),
        "scanned": len(results), "universe": len(uni), "universe_source": uni_cache["source"], "failed": failed,
        "market": market,
        "market_state": mstate,
        "watchlist": [{"symbol": u["ccy"], "rank": u["rank"], "rules": [r.id for r in fired],
                       "text": describe(u["ccy"], row, fired, u["rank"], True, LS_LABEL)
                       + ("\n" + outlook(u["ccy"], fired)[0] if fired and outlook(u["ccy"], fired)[0] else ""),
                       "features": {k: _num(row.get(k)) for k in FEATURE_KEYS}}
                      for u, row, fired in results if u["watch"]],
        "firing": sorted(firing, key=lambda x: (-x["watch"], -x["score"])),
        "top_movers": [{"symbol": u["ccy"], "rank": u["rank"], "ret_24h": _num(np.expm1(row["ret_24h"])),
                        "resid_24h_z": _num(row["resid_24h_z"]), "oi_z": _num(row.get("oi_z")),
                        "funding": _num(row.get("funding"))} for u, row in movers],
        "price_levels": levels,
        "funding_levels": fund_levels,
        "buybacks": buybacks,
        "new_events": len(new_events),
        "base_rates": {"span_days": rates.get("span_days"), "coins": len(labeled),
                       "baseline": rates.get("baseline"),
                       "conditions": {cid: {"name": fs.RULES_BY_ID[cid].name, **st}
                                      for cid, st in rates.get("conditions", {}).items()},
                       "market": mrates},
        "scorecard": sc,
        "scorecard_text": fs.scorecard_text(sc),
        "opportunity": opp_block,
        "opportunity_live": opp_live,
    }
    keep_after = now - 7 * 24 * HOUR_MS
    events = [e for e in prev_events if int(e.get("ts", 0)) >= keep_after] + new_events
    state = {"last_fire": {k: v for k, v in last_fire.items() if v >= keep_after},
             "price_alerts": pa_state, "funding_alerts": fa_state, "universe": uni_cache,
             "market_state": mstate.get("state") if mstate else prev_ms}
    return signals, events, state, archive, preds


def opportunity_md(o: dict, live: dict | None) -> list[str]:
    by = {c["symbol"]: c for c in o["coins"]}
    age = lambda c: "" if c.get("age_days") is None else (f"{c['age_days']}天" + ("⚠️" if c.get("status") == "上市较短" else ""))
    row = lambda c: (f"| {c['symbol']} | {age(c)} | {c['vol_range'] * 100:.0f}% | {c['mae_q10'] * 100:+.0f}% | {c['safe_lev']:.1f}x | "
                     f"{c['p_up'] * 100:.0f}% | {c['p_dn'] * 100:.0f}% | 波动#{c['rank_vol_range']} |")
    head = ["| 币 | 上市 | 预测波动 | 回撤 q10 | 杠杆上限 | P上 | P下 | 排名 |", "|---|---|---|---|---|---|---|---|"]
    lines = []
    if o.get("watch_highlights"):
        lines += ["**自选里值得留意的**:" + ";".join(f"{h['symbol']}({'、'.join(h['reasons'])})" for h in o["watch_highlights"]), ""]
    for title, key in [("波动最大(两头概率同步偏高)", "top_vol"), ("自选", "watchlist")]:
        syms = o.get(key) or []
        if syms:
            lines += [f"**{title}**"] + head + [row(by[s]) for s in syms if s in by] + [""]
    if o.get("abstained"):
        lines += ["**暂不判断的新币**:" + ";".join(f"{c['symbol']}({c['note']})" for c in o["abstained"]), ""]
    lines += ["> " + n for n in o.get("notes", [])]
    if live:
        lines.append(f"> 实盘核对:{live.get('status', '')}" if live.get("status") != "ok" else
                     f"> 实盘核对({live['resolved']} 条):上涨概率预测均值 {live['p_up']['预测均值']:.1%} / 实际 {live['p_up']['实际频率']:.1%};"
                     f"下跌 {live['p_dn']['预测均值']:.1%} / {live['p_dn']['实际频率']:.1%};回撤越界率 {live['mae_breach_rate']:.1%}")
    return lines


def direction_md(ds: dict) -> list[str]:
    """机会榜方向分(P上-P下)的实盘成绩:每 6 小时留档一次,到期后用真实价格核对。"""
    names = {"72h": "72h", "168h": "1 周", "336h": "2 周"}
    pc = lambda v, d=1: "—" if v is None else f"{v * 100:+.{d}f}%"
    pp = lambda v: "—" if v is None else f"{v * 100:.0f}%"
    tt = lambda v: "—" if v is None else f"{v:+.1f}"
    lines = ["| 持有期 | 截面数 | 独立区间≈ | IC 均值(t) | 偏涨−偏跌 超额(t) | 偏涨组跑赢比例 / 同期全体 | 偏跌组跑输比例 / 同期全体 |",
             "|---|---|---|---|---|---|---|"]
    notes = []
    for k, st in ds.items():
        nm = names.get(k, k)
        if st.get("status") != "ok":
            notes.append(f"{nm}:{st.get('status')}")
            continue
        flag = "(仅供参考)" if (st.get("indep") or 0) < 10 else ""
        lines.append(f"| {nm}{flag} | {st['n_cross']} | {st.get('indep')} | {st['ic_mean']:+.3f}({tt(st['ic_t'])}) | "
                     f"{pc(st['spread_mean'])}({tt(st['spread_t'])}) | {pp(st['top_hit'])} / {pp(st['base_up'])} | "
                     f"{pp(st['bottom_hit'])} / {pp(st['base_dn'])} |")
    if len(lines) == 2:
        lines = []
    lines += [f"- {n}" for n in notes]
    lines.append("> 方向分 = P上−P下 的截面排名,偏涨/偏跌 = 每次排名的前/后 k 名;收益为剔除 BTC beta 后的超额。"
                 "t 值已按持有期做 Newey-West 校正;独立区间 = 覆盖时长 / 持有期,小于 10 时结论很不稳。"
                 "明细见 data 分支 opp_outcomes.csv.gz / opp_direction_daily.csv")
    return lines


def status_md(sig: dict, events: list) -> str:
    m = sig.get("market", {})
    lines = [f"# CryptoRadar 状态 · {sig.get('generated_at_local', '')}", ""]
    if sig.get("error"):
        lines += [f"**运行出错:** {sig['error']}", ""]
    if m:
        lines.append(f"BTC ${m['btc_price']:,.0f} (24h {m['btc_ret_24h'] * 100:+.1f}%) · "
                     f"扫描 {sig.get('scanned')} 个合约 · 用时 {sig.get('runtime_sec')} 秒")
    if sig.get("market_state"):
        lines += ["", "## 市场状态", fs.market_text(sig["market_state"])]
    if sig.get("scorecard_text"):
        lines += ["", "## 预警记分卡", sig["scorecard_text"]]
    if sig.get("opportunity") and not sig["opportunity"].get("error"):
        lines += ["", "## 72 小时机会榜(波动 / 回撤 / 概率)"] + opportunity_md(sig["opportunity"], sig.get("opportunity_live"))
    if sig.get("opportunity_direction"):
        lines += ["", "## 实盘方向核对(72h / 1 周 / 2 周)"] + direction_md(sig["opportunity_direction"])
    if sig.get("ledger_summary"):
        lines += ["", "## 实盘信号后验表(按规则)"] + fs.ledger_text(sig["ledger_summary"])
    br = sig.get("base_rates") or {}
    if br.get("baseline"):
        b = br["baseline"]
        lines += ["", f"## 历史概率(样本 {br['coins']} 个币 · {br['span_days']} 天,72h)",
                  "| 情形 | 次数 | 上涨概率 | 中位收益 | 最差 10% | 期间回撤中位 |", "|---|---|---|---|---|---|",
                  f"| 任意时点(基准) | {b['n']} | {b['up72'] * 100:.0f}% | {b['med72'] * 100:+.1f}% | "
                  f"{b['p10_72'] * 100:+.1f}% | {b['mae72_med'] * 100:+.1f}% |"]
        for cid, c in sorted(br.get("conditions", {}).items(), key=lambda kv: -kv[1]["up72"]):
            lines.append(f"| {c['name']} | {c['n']} | {c['up72'] * 100:.0f}% | {c['med72'] * 100:+.1f}% | "
                         f"{c['p10_72'] * 100:+.1f}% | {c['mae72_med'] * 100:+.1f}% |")
    lines += ["", "## 自选"] + [w["text"] + "\n" for w in sig.get("watchlist", [])]
    seen = set()
    for fl in sig.get("funding_levels", []):
        if fl["symbol"] in seen:
            continue
        seen.add(fl["symbol"])
        s = f"- {fl['symbol']} 资金费率 {fl['funding'] * 100:.4f}%/8h"
        if fl.get("daily_cost") is not None:
            d = fl["daily_cost"]
            s += f" · {fl['position_usdt']:,.0f} USDT 多仓每天{'付' if d >= 0 else '收'} {abs(d):.2f} USDT"
        lines.append(s)
    if seen:
        lines.append("")
    if sig.get("buybacks"):
        lines += ["## 回购收益率(年化回购 ÷ 流通市值)"]
        for b in sig["buybacks"]:
            y = f"{b['yield'] * 100:.1f}%" if b.get("yield") is not None else "—"
            amt = f"${b['annual_usd'] / 1e6:,.0f}M/年" if b.get("annual_usd") else "—"
            mc = f"${b['market_cap'] / 1e6:,.0f}M" if b.get("market_cap") else "—"
            lines.append(f"- {b['symbol']} {y} · 回购 {amt} · 市值 {mc} · {b.get('basis', '')}")
        lines.append("")
    lines += ["## 正在触发"] + [f["text"] + "\n" for f in sig.get("firing", [])[:20]]
    lines += ["## 最近 24 小时新事件"]
    cut = sig.get("generated_at", 0) - 24 * HOUR_MS
    for e in reversed([e for e in events if e["ts"] >= cut]):
        t = datetime.fromtimestamp(e["ts"] / 1000, tz=timezone.utc).strftime("%m-%d %H:%M UTC")
        lines.append(f"- {t} {e['text'].splitlines()[0]}")
    lines += ["", "> 信号只描述仓位结构异动,不代表方向。"]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="cloud_config.yaml")
    ap.add_argument("--prev", default="prev", help="上一次的 data 分支内容所在目录")
    ap.add_argument("--out", default="out")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass

    cfg = load_config(args.config)
    tz = ZoneInfo(cfg.get("display_timezone", "Asia/Singapore"))
    prev, out = Path(args.prev), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    prev_state = load_json(prev / "state.json", {})
    prev_events = load_json(prev / "events.json", {}).get("events", [])
    prev_signals = load_json(prev / "signals.json", {})
    prev_archive = fs.load_archive(prev / "archive.csv.gz")
    prev_preds = fs.load_predictions(prev / "predictions.json")
    prev_ledger = fs.load_ledger(prev / "ledger.csv")
    try:
        prev_opp_log = pd.read_csv(prev / "opp_log.csv.gz")
    except Exception:
        prev_opp_log = None
    try:
        prev_outcomes = pd.read_csv(prev / "opp_outcomes.csv.gz")
    except Exception:
        prev_outcomes = None
    extras: dict = {}

    code = 0
    try:
        signals, events, state, archive, preds = run(cfg, OKX(), prev_state, prev_events, tz,
                                                     prev_archive, prev_preds, prev_opp_log, extras)
    except Exception as e:
        log.error("运行失败:%s", traceback.format_exc())
        code = 1
        now = now_ms()
        signals = dict(prev_signals)  # 保留上一次的快照,只标记错误
        signals.update({"error": f"{type(e).__name__}: {e}", "error_at": now,
                        "consecutive_errors": int(prev_signals.get("consecutive_errors", 0)) + 1})
        signals.setdefault("generated_at", 0)
        events, state = prev_events, prev_state
        archive, preds = prev_archive, prev_preds
    else:
        signals["consecutive_errors"] = 0

    ledger = fs.update_ledger(prev_ledger, preds)
    signals["ledger_summary"] = fs.ledger_summary(ledger)
    outcomes = prev_outcomes
    try:
        if extras.get("opp_resolved") is not None:
            outcomes = opp.merge_outcomes(prev_outcomes, extras["opp_resolved"])
        if outcomes is not None and len(outcomes):
            signals["opportunity_direction"] = opp.direction_summary(outcomes)
    except Exception:           # 核对出错不能影响其他输出;保留上一份结果表
        log.warning("方向核对失败:%s", traceback.format_exc())
        outcomes = prev_outcomes
    (out / "signals.json").write_text(json.dumps(signals, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "events.json").write_text(json.dumps({"events": events}, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "state.json").write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
    (out / "status.md").write_text(status_md(signals, events), encoding="utf-8")
    if archive is not None and len(archive):
        fs.save_archive(archive, out / "archive.csv.gz")
    (out / "predictions.json").write_text(json.dumps({"predictions": preds}, ensure_ascii=False),
                                          encoding="utf-8")
    ledger.to_csv(out / "ledger.csv", index=False)
    opp_log = extras.get("opp_log", prev_opp_log)
    if opp_log is not None and len(opp_log):
        opp_log.to_csv(out / "opp_log.csv.gz", index=False)
    if outcomes is not None and len(outcomes):
        outcomes.to_csv(out / "opp_outcomes.csv.gz", index=False)
        opp.direction_daily(outcomes).to_csv(out / "opp_direction_daily.csv", index=False)
    # 出错也以 0 退出:错误写进 signals.json 由 Claude 转告,避免 GitHub 每 15 分钟发一封失败邮件
    log.info("完成:新事件 %s 个%s", signals.get("new_events"), "(本轮出错)" if code else "")


if __name__ == "__main__":
    main()
