"""云端单次运行(GitHub Actions 每 15 分钟调用一次)。

无数据库:每次都从 OKX 重新拉取约 37 天的小时数据来计算 30 天 z-score。
跨次运行需要记住的东西(规则冷却时间、价位提醒状态)放在 state.json 里,
由工作流从 data 分支取回、运行完再推回去。

输出(写到 --out 目录,工作流会强制推送到 data 分支):
  signals.json  当前快照:大盘、自选、正在触发的全部规则、独立行情榜
  events.json   最近 7 天的"新事件"(过了冷却期的新规则触发、价位穿越),Claude 定时任务读这个来推送
  state.json    冷却与价位状态
  status.md     方便在 GitHub 网页上直接查看的中文摘要
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

from cryptoradar.config import load_config
from cryptoradar.features import build_features
from cryptoradar.okx_api import HOUR_MS, OKX, OKXBlockedError
from cryptoradar.okx_collect import collect
from cryptoradar.signals import describe, evaluate_last, merged_thresholds
from cryptoradar.universe import DEFAULT_EXCLUDE, _looks_like_stable, fetch_coingecko_top

log = logging.getLogger("cloud")

FEATURE_KEYS = ["close", "ret_24h", "resid_24h_z", "ret_1h_z", "oi_chg_24h", "oi_z", "funding",
                "funding_z", "vol_z", "top_ls", "top_ls_z", "taker_z", "adr_14d", "beta"]
LS_LABEL = "多空账户比"


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
    """市值排名列表 [{symbol, rank, current_price}]。CoinGecko 失败时换 CoinPaprika。"""
    for attempt in range(3):
        try:
            coins = fetch_coingecko_top(250, ucfg.get("coingecko_api_key") or None)
            return [{"symbol": (c.get("symbol") or "").upper(), "rank": c.get("market_cap_rank"),
                     "current_price": c.get("current_price")} for c in coins], "coingecko"
        except Exception as e:
            log.warning("CoinGecko 第 %d 次失败:%s", attempt + 1, e)
            time.sleep(10 * (attempt + 1))
    try:
        r = requests.get(PAPRIKA, params={"quotes": "USD"}, timeout=40)
        r.raise_for_status()
        rows = sorted((x for x in r.json() if x.get("rank")), key=lambda x: x["rank"])[:400]
        return [{"symbol": (x.get("symbol") or "").upper(), "rank": x["rank"],
                 "current_price": (x.get("quotes") or {}).get("USD", {}).get("price")} for x in rows], "coinpaprika"
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

    rank_of = {}
    for c in coins:
        rank_of.setdefault(c["symbol"], c.get("rank"))
    rows, seen = [], set()
    for c in coins:
        sym = c["symbol"]
        if not c.get("rank") or c["rank"] > top_n:
            continue
        if not sym or sym in exclude or _looks_like_stable(c) or sym in seen or sym not in swaps:
            continue
        seen.add(sym)
        rows.append({"ccy": sym, "inst": swaps[sym], "rank": c["rank"], "watch": sym in watch})
    for sym in watch + ["BTC", "ETH"]:
        if sym in seen:
            continue
        if sym not in swaps:
            log.warning("%s 在 OKX 没有 USDT 永续,跳过", sym)
            continue
        seen.add(sym)
        rows.append({"ccy": sym, "inst": swaps[sym], "rank": rank_of.get(sym), "watch": sym in watch})
    log.info("监控名单 %d 个(排名来源:%s)", len(rows), cache["source"])
    return rows, cache


def run(cfg: dict, okx: OKX, prev_state: dict, prev_events: list, tz) -> tuple[dict, list, dict]:
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

    results = []
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
        results.append((u, f.iloc[-1], evaluate_last(f, th)))

    last_fire = dict(prev_state.get("last_fire", {}))
    new_events, firing = [], []
    for u, row, fired in results:
        if not fired:
            continue
        score = sum(r.weight for r in fired)
        text = describe(u["ccy"], row, fired, u["rank"], u["watch"], LS_LABEL)
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
            })
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
        "watchlist": [{"symbol": u["ccy"], "rank": u["rank"], "rules": [r.id for r in fired],
                       "text": describe(u["ccy"], row, fired, u["rank"], True, LS_LABEL),
                       "features": {k: _num(row.get(k)) for k in FEATURE_KEYS}}
                      for u, row, fired in results if u["watch"]],
        "firing": sorted(firing, key=lambda x: (-x["watch"], -x["score"])),
        "top_movers": [{"symbol": u["ccy"], "rank": u["rank"], "ret_24h": _num(np.expm1(row["ret_24h"])),
                        "resid_24h_z": _num(row["resid_24h_z"]), "oi_z": _num(row.get("oi_z")),
                        "funding": _num(row.get("funding"))} for u, row in movers],
        "price_levels": levels,
        "funding_levels": fund_levels,
        "new_events": len(new_events),
    }
    keep_after = now - 7 * 24 * HOUR_MS
    events = [e for e in prev_events if int(e.get("ts", 0)) >= keep_after] + new_events
    state = {"last_fire": {k: v for k, v in last_fire.items() if v >= keep_after},
             "price_alerts": pa_state, "funding_alerts": fa_state, "universe": uni_cache}
    return signals, events, state


def status_md(sig: dict, events: list) -> str:
    m = sig.get("market", {})
    lines = [f"# CryptoRadar 状态 · {sig.get('generated_at_local', '')}", ""]
    if sig.get("error"):
        lines += [f"**运行出错:** {sig['error']}", ""]
    if m:
        lines.append(f"BTC ${m['btc_price']:,.0f} (24h {m['btc_ret_24h'] * 100:+.1f}%) · "
                     f"扫描 {sig.get('scanned')} 个合约 · 用时 {sig.get('runtime_sec')} 秒")
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

    code = 0
    try:
        signals, events, state = run(cfg, OKX(), prev_state, prev_events, tz)
    except Exception as e:
        log.error("运行失败:%s", traceback.format_exc())
        code = 1
        now = now_ms()
        signals = dict(prev_signals)  # 保留上一次的快照,只标记错误
        signals.update({"error": f"{type(e).__name__}: {e}", "error_at": now,
                        "consecutive_errors": int(prev_signals.get("consecutive_errors", 0)) + 1})
        signals.setdefault("generated_at", 0)
        events, state = prev_events, prev_state
    else:
        signals["consecutive_errors"] = 0

    (out / "signals.json").write_text(json.dumps(signals, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "events.json").write_text(json.dumps({"events": events}, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "state.json").write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
    (out / "status.md").write_text(status_md(signals, events), encoding="utf-8")
    # 出错也以 0 退出:错误写进 signals.json 由 Claude 转告,避免 GitHub 每 15 分钟发一封失败邮件
    log.info("完成:新事件 %s 个%s", signals.get("new_events"), "(本轮出错)" if code else "")


if __name__ == "__main__":
    main()
