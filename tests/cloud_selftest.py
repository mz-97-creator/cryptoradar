"""云端版离线自检:python -m tests.cloud_selftest

用模拟的 OKX 接口跑 cloud_run 两轮,检查事件生成、冷却、价位提醒和出错时的行为。
"""
from __future__ import annotations

import json
import logging
import shutil
import sys
import tempfile
from pathlib import Path
from zoneinfo import ZoneInfo

import cloud_run
import numpy as np
import pandas as pd

from cryptoradar import foresight as fs
from cryptoradar import opportunity as opp_mod
from cryptoradar import universe
from cryptoradar.config import DEFAULTS, _merge
from cryptoradar.okx_api import HOUR_MS, OKX, OKXBlockedError
from tests.mock_api import Market, fake_coingecko

CCY = {"BTC": "BTCUSDT", "ETH": "ETHUSDT", "OP": "OPUSDT", "SOL": "SOLUSDT",
       "ARB": "ARBUSDT", "PEPE": "1000PEPEUSDT", "DOGE": "DOGEUSDT"}


class FakeOKX(OKX):
    def __init__(self, market: Market, blocked: bool = False):
        super().__init__()
        self.m, self.blocked = market, blocked
        for lim in self.lim.values():
            lim.max_calls = 10**9

    def _get(self, path, params=None, group="default"):
        if self.blocked:
            raise OKXBlockedError("OKX 拒绝访问 HTTP 451")
        p, m = params or {}, self.m
        if path.endswith("/instruments"):
            import time as _t
            # SOL 设成 5 天前上市(新币),ARB 设成 45 天前(上市较短),其余 400 天前
            lt = {"SOL": 5, "ARB": 45}
            return [{"instId": f"{c}-USDT-SWAP", "settleCcy": "USDT", "ctType": "linear", "state": "live", "ctValCcy": c,
                     "listTime": str(int((_t.time() - lt.get(c, 400) * 86400) * 1000))} for c in CCY]
        ccy = p.get("ccy") or (p.get("instId", "").split("-")[0])
        d = m.data[CCY[ccy]]
        snaps = [t for t in d.index if t >= m.end - 30 * 24 * HOUR_MS]
        if path.endswith("/candles"):
            sel = d[d.index < int(p["after"])] if "after" in p else d
            sel = sel.iloc[::-1].iloc[: p["limit"]]
            return [[str(t), str(r.open), str(r.high), str(r.low), str(r.close), "1", "1",
                     str(r.volume * r.close), "1"] for t, r in sel.iterrows()]
        prev = lambda t: d.loc[t - HOUR_MS] if t - HOUR_MS in d.index else d.loc[t]
        if path.endswith("open-interest-volume"):
            return [[str(t), str(prev(t).oi * prev(t).close), "1"] for t in reversed(snaps)]
        if path.endswith("long-short-account-ratio"):
            return [[str(t), str(prev(t).top_ls)] for t in reversed(snaps)]
        if path.endswith("taker-volume"):
            return [[str(t), "100", str(100 * d.loc[t].taker)] for t in reversed(snaps)]
        if path.endswith("/funding-rate"):
            return [{"fundingRate": str(d.funding.iloc[-1]), "fundingTime": str(m.end + HOUR_MS),
                     "nextFundingTime": str(m.end + 9 * HOUR_MS)}]
        if path.endswith("funding-rate-history"):
            sel = d.iloc[::8].iloc[::-1].iloc[:100]
            return [{"fundingTime": str(t), "fundingRate": str(r.funding), "realizedRate": str(r.funding)}
                    for t, r in sel.iterrows()]
        raise AssertionError(path)


def _op_frame(m):
    """OP 与 BTC 的特征表(只用价格列),给记分卡测试用。"""
    import pandas as pd
    def hourly(sym):
        d = m.data[sym]
        return pd.DataFrame({"open": d.open, "high": d.high, "low": d.low, "close": d.close,
                             "quote_volume": d.volume * d.close, "oi": d.oi, "top_ls": d.top_ls,
                             "taker_ratio": d.taker})
    return hourly("OPUSDT"), hourly("BTCUSDT")


def check(cond, msg):
    print(("  ✔ " if cond else "  ✘ ") + msg)
    if not cond:
        raise SystemExit(1)


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    logging.basicConfig(level=logging.WARNING)
    universe.fetch_coingecko_top = fake_coingecko
    cloud_run.fetch_coingecko_top = fake_coingecko
    cfg = _merge(DEFAULTS, {"universe": {"top_n": 150, "watchlist": ["OP"]},
                            "price_alerts": [{"symbol": "OP", "below": 1.0, "note": "测试"}],
                            "funding_alerts": [{"symbol": "OP", "above": -1.0, "position_usdt": 3000},
                                               {"symbol": "OP", "below": -1.0}],
                            "buybacks": [{"symbol": "OP", "annual_eth": 100}, {"symbol": "SOL", "annual_usd": 5e6}]})
    tz = ZoneInfo("Asia/Singapore")
    # 基本面不联网:OP 的持币人收入最近 7 天翻倍(回购加速),其余币没有对应协议
    import time as _t
    day0 = int(_t.time()) // 86400 * 86400
    hrev = {str(day0 - k * 86400): (200_000.0 if k <= 7 else 100_000.0) for k in range(60)}
    fake_store = {"mapping": {"OP": {"chain": "Optimism", "protocol": None}}, "mapping_at": _t.time(),
                  "coins": {"OP": {"at": _t.time(), "fees": hrev, "rev": hrev, "hrev": hrev, "tvl": None}}}
    cloud_run.fd.refresh = lambda store, coins, budget_s=60: fake_store
    # 事件库不联网:每轮固定"看到"一条 OP 现货上新
    cloud_run.evt.refresh = lambda prev, st, now, budget_s=20: (
        cloud_run.evt.merge(prev, [{"id": "upbit|OP", "kind": "spot_list", "source": "upbit", "symbol": "OP",
                                    "event_ts": now, "title": "upbit 新增 OP"}], now), {"listings": {"upbit": ["OP"]}})
    cfg["early"] = {"push": "watchlist", "push_detectors": ["F_REV_UP"], "push_cooldown_hours": 168}
    m = Market(hours=1000)

    print("\n[1] 第一轮")
    sig, events, state, archive, preds = cloud_run.run(cfg, FakeOKX(m), {}, [], tz)
    syms = {e["symbol"]: e for e in events if e["type"] == "signal"}
    check(sig["scanned"] >= 6, f"扫描 {sig['scanned']} 个合约")
    check("OI_DIV" in syms.get("OP", {}).get("rules", []), f"OP 产生 OI 背离事件:{syms.get('OP', {}).get('rules')}")
    check(any(e["type"] == "price" for e in events), "价位提醒事件生成")
    fe = [e for e in events if e["type"] == "funding"]
    check(len(fe) == 1 and fe[0]["daily_cost"] is not None, f"资金费率提醒事件生成:{fe[0]['text'] if fe else None}")
    check(len(sig["funding_levels"]) == 2, "资金费率阈值写入快照")
    bb = {b["symbol"]: b for b in sig["buybacks"]}
    check(bb["OP"]["yield"] and bb["SOL"]["yield"], f"回购收益率:{ {k: round(v['yield'], 4) for k, v in bb.items()} }")
    check(abs(bb["OP"]["live_annual_usd"] - (7 * 200_000 + 23 * 100_000) * 365 / 30) < 1
          and abs(bb["OP"]["live_ratio_7d"] - 2.0) < 1e-9, f"实时回购:年化 {bb['OP']['live_annual_usd']:,.0f},7 日加速 {bb['OP']['live_ratio_7d']}")
    ef = {x["symbol"]: x for x in sig["early"]["firing"]}
    check("OP" in ef and {"F_REV_UP", "F_FEES_UP"} <= set(ef["OP"]["detectors"]), f"基本面检测触发:{ef.get('OP')}")
    ee = [e for e in events if e["type"] == "early"]
    check(len(ee) == 1 and ee[0]["symbol"] == "OP" and ee[0]["detectors"] == ["F_REV_UP"] and "2.00 倍" in ee[0]["text"],
          f"自选币推送回购加速(只推 F_REV_UP):{ee[0]['text'] if ee else None}")
    check(state.get("early_push_last", {}).get("OP|F_REV_UP"), "推送冷却已记录")
    check(any("OP⭐ 现货上新(upbit)" in x for x in sig["events_recent"]) and "事件(近 48 小时" in cloud_run.status_md(sig, events)
          and not any(e["type"] == "spot_list" for e in events), "事件库只展示不推送")
    check(state.get("event_state", {}).get("listings"), "交易对清单写进状态")
    check(sig["watchlist"] and sig["watchlist"][0]["symbol"] == "OP", "自选 OP 在快照里")
    check("USDC" not in {f["symbol"] for f in sig["firing"]}, "稳定币已排除")
    print("\n" + sig["watchlist"][0]["text"] + "\n")

    print("[2] 第二轮(冷却期内)")
    sig2, events2, state2, archive2, preds2 = cloud_run.run(cfg, FakeOKX(m), state, events, tz, archive, preds)
    check(sig2["new_events"] == 0, "没有重复事件")
    check(len(events2) == len(events), "事件日志保留")

    print("\n[2b] 前瞻模块:市场灯、历史概率、样本库、记分卡")
    ms = sig.get("market_state") or {}
    check(ms.get("state") in fs.STATES, f"市场状态:{ms.get('icon')} {ms.get('title')}")
    check(any(e["type"] == "market" for e in events), "市场灯首次判定生成事件")
    check(not any(e["type"] == "market" for e in events2[len(events):]), "状态未变不重复推送")
    br = sig["base_rates"]
    check(br["baseline"] and br["baseline"]["n"] > 50, f"基准样本 {br['baseline']['n']} · 条件 {len(br['conditions'])} 个")
    op_ev = next(e for e in events if e["type"] == "signal" and e["symbol"] == "OP")
    check(op_ev.get("outlook") and "历史:" in op_ev["text"], "事件附带历史概率")
    check(any(p["kind"] == "signal" for p in preds) and any(p["kind"] == "market" for p in preds), "预警写入记分卡")

    # 72 小时机会模型:云端打分结构完整,概率在 0~1,回撤为负,杠杆上限为正;出错不影响主扫描
    ob = sig["opportunity"]
    check(ob and not ob.get("error"), f"机会模型输出:{ob.get('error') if ob else None}")
    cs_ = ob["coins"]
    check(len(cs_) >= 6 and all(0 <= c["p_up"] <= 1 and 0 <= c["p_dn"] <= 1 for c in cs_), "概率在 0~1 之间")
    check(all(c["mae_q10"] < 0 and c["safe_lev"] > 0 and c["vol_range"] > 0 for c in cs_), "回撤为负、杠杆上限与波动为正")
    check(ob["top_vol"] and set(ob["top_vol"]) <= {c["symbol"] for c in cs_}, "波动榜在名单内")
    check(all(c["direction"] in ("无明确方向", "中性", "偏涨", "偏跌") for c in cs_), "方向标签合法(证据弱时不下结论)")
    ab = {c["symbol"]: c for c in ob["abstained"]}
    check("SOL" in ab and ab["SOL"]["p_up"] is None and ab["SOL"]["direction"] == "暂不判断", "新币(上市5天)暂不判断且不给数字")
    check("SOL" not in {c["symbol"] for c in cs_} and "SOL" not in ob["top_vol"], "新币不进榜单和排名")
    arb = next((c for c in cs_ if c["symbol"] == "ARB"), None)
    check(arb is not None and arb["status"] == "上市较短" and arb["p_up"] is not None, "上市较短的币给结果但带标记")
    check(ob["n_coins"] + ob["n_abstained"] == len(set(c["symbol"] for c in cs_) | set(ab)), "可评估与暂不判断合计等于全部")
    check(all({"rank_vol_range", "rank_p_up", "rank_p_dn"} <= set(c) for c in cs_), "每个币带三项排名")
    check(all(set(h) == {"symbol", "reasons"} and h["reasons"] for h in ob["watch_highlights"]), "自选关注项含原因")
    check("机会榜" in cloud_run.status_md(sig, events), "status.md 含机会榜")
    check(sig["opportunity_live"]["resolved"] == 0, "实盘核对:刚上线时无已结算记录")
    check(sorted(c["dir_rank"] for c in cs_) == list(range(1, len(cs_) + 1))
          and all(isinstance(c["dir_score"], float) for c in cs_), "每个可评估币带方向分和截面排名(只在可评估币之间排)")
    ex: dict = {}
    cloud_run.run(cfg, FakeOKX(m), {}, [], tz, None, None, None, ex)
    lg0 = ex.get("opp_log")
    check(lg0 is not None and list(lg0.columns) == opp_mod.LOG_COLS, "预测留档含方向分、排名、档位和触发规则列")
    check(ex.get("opp_resolved") is not None, "留档核对结果交给 main 累积")

    # 样本库:读写一致,且比实时窗口更早的样本会被保留
    tmpa = Path(tempfile.mkdtemp()) / "archive.csv.gz"
    fs.save_archive(archive, tmpa)
    back = fs.load_archive(tmpa)
    check(len(back) == len(archive) and set(back["symbol"]) == set(archive["symbol"]), f"样本库读写 {len(back)} 行")
    old = back[back.symbol == "OP"].head(1).copy()
    old["ts"] = int(old["ts"].iloc[0]) - 500 * HOUR_MS
    sig3, _, _, archive3, _ = cloud_run.run(cfg, FakeOKX(m), state2, events2, tz,
                                            pd.concat([back, old]), preds2)
    check(int(archive3[archive3.symbol == "OP"]["ts"].min()) == int(old["ts"].iloc[0]), "更早的样本被保留,样本库会增长")

    # 记分卡:构造一条 100 小时前的预测,核对 24h/72h 收益与直接计算一致
    f = fs.merge_archive(None, {"OP": cloud_run.build_features(*_op_frame(m))})["OP"]
    t0 = int(f.index[-100])
    pr = fs.new_prediction(t0, "signal", "OP", "OI_DIV", t0, None, "up", 0.6, 0.5, 0.01, 30)
    res = fs.resolve([pr], {"OP": f, "BTC": f}, pd.DataFrame(), t0 + 100 * HOUR_MS)[0]["res"]
    want = f.at[t0 + 72 * HOUR_MS, "close"] / f.at[t0, "close"] - 1
    check(abs(res["72h"]["ret"] - want) < 1e-12 and "24h" in res, f"到期核对正确:72h {res['72h']['ret'] * 100:+.2f}%")
    future = fs.new_prediction(t0, "signal", "OP", "OI_DIV", int(f.index[-10]), None, "up", .6, .5, .01, 30)
    check(fs.resolve([future], {"OP": f}, pd.DataFrame(), int(f.index[-1]))[0]["res"] == {}, "未到期的不提前核对")
    print("  " + fs.scorecard_text(fs.scorecard([dict(pr, res=res, ts=t0 + 100 * HOUR_MS)], t0 + 100 * HOUR_MS)))

    # 实盘记录:每 6 小时记一次(同一时刻只保留第一次),满 72h 后用真实价格结算
    from cryptoradar import opportunity as opp
    t6 = (int(f.index[-200]) // HOUR_MS // 6) * 6 * HOUR_MS
    t6 = t6 if t6 in f.index else int(f.index[-200])
    snap = pd.DataFrame({"t_bar": [t6, t6], "symbol": ["OP", "OP"], "p_up": [0.3, 0.9], "p_dn": [0.1, 0.1],
                         "vol_range": [0.2, 0.2], "mae_q10": [-0.1, -0.1], "t": [0, 0]})
    lg = opp.log_snapshot(None, snap.drop(columns="t").iloc[[0]].assign(t_bar=(t6 // HOUR_MS // 6) * 6 * HOUR_MS))
    lg2 = opp.log_snapshot(lg, snap.drop(columns="t").iloc[[1]].assign(t_bar=(t6 // HOUR_MS // 6) * 6 * HOUR_MS))
    check(len(lg2) == 1 and lg2["p_up"].iloc[0] == 0.3, "同一时刻只保留第一次预测")
    rs = opp.resolve_log(pd.DataFrame({"t_bar": [int(f.index[-150])], "symbol": ["OP"], "p_up": [0.3], "p_dn": [0.1],
                                      "vol_range": [0.2], "mae_q10": [-0.1]}), {"OP": f})
    check(len(rs) == 1 and rs["mae72"].iloc[0] <= 0 and rs["range72"].iloc[0] > 0, "满 72h 的记录被结算")
    check(opp.live_summary(rs, 0.05)["resolved"] == 1 and "积累中" in opp.live_summary(rs, 0.05)["status"], "样本不足时只报条数")

    # 信号后验记录表:含全部触发规则,幂等更新,预测被清理后记录仍在
    pr2 = fs.new_prediction(t0, "signal", "OP", "OI_DIV", t0, 0.5, "up", 0.6, 0.5, 0.01, 30,
                            rules=["OI_DIV", "RESID"], score=3.0, watch=True)
    pr2 = fs.resolve([pr2], {"OP": f, "BTC": f}, pd.DataFrame(), t0 + 100 * HOUR_MS)[0]
    led = fs.update_ledger(None, [pr2])
    check(len(led) == 1 and led.at[0, "rules"] == "OI_DIV;RESID" and pd.notna(led.at[0, "resid72"]), "记录表写入全部规则和 72h 结果")
    led2 = fs.update_ledger(led, [pr2])
    check(len(led2) == 1, "重复更新不产生重复行")
    check(len(fs.update_ledger(led2, [])) == 1, "预测被清理后记录表仍保留")
    ls = fs.ledger_summary(led2)
    check(set(ls["by_rule"]) == {"OI_DIV", "RESID"} and ls["resolved"] == 1, "按规则汇总(多规则信号各算一次)")
    check(any("OI 激增但价格未涨" in ln for ln in fs.ledger_text(ls)), "记录表文字可读")
    unresolved = fs.new_prediction(t0, "signal", "OP", "VOL", int(f.index[-10]), None, "none", None, None, None, 0)
    check(fs.ledger_summary(fs.update_ledger(None, [unresolved]))["resolved"] == 0, "未到期的信号不计入汇总")

    # 规则权重可由配置覆盖:权重 ≤ 0 即停用,未知规则名报错
    from cryptoradar.signals import RULES, RULES_BY_ID, apply_weights
    check([r.weight for r in apply_weights(None)] == [r.weight for r in RULES], "不配置时和内置权重完全一致")
    rw = {r.id: r.weight for r in apply_weights({"OI_DIV": 0, "RESID": 3.0})}
    check("OI_DIV" not in rw and rw["RESID"] == 3.0 and rw["VOL"] == RULES_BY_ID["VOL"].weight, "权重覆盖与停用生效")
    try:
        apply_weights({"NOPE": 1})
        check(False, "未知规则名应报错")
    except ValueError:
        check(True, "未知规则名报错")
    print("\n" + fs.market_text(ms))

    print("\n[3] 完整命令行流程 + OKX 被拒绝时")
    tmp = Path(tempfile.mkdtemp())
    try:
        (tmp / "prev").mkdir()
        for name, obj in [("signals.json", sig2), ("events.json", {"events": events2}), ("state.json", state2)]:
            (tmp / "prev" / name).write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
        orig = cloud_run.OKX
        cloud_run.OKX = lambda: FakeOKX(m, blocked=True)
        sys.argv = ["cloud_run.py", "--config", "cloud_config.yaml", "--prev", str(tmp / "prev"), "--out", str(tmp / "out")]
        cloud_run.main()
        cloud_run.OKX = orig
        s3 = json.loads((tmp / "out" / "signals.json").read_text(encoding="utf-8"))
        e3 = json.loads((tmp / "out" / "events.json").read_text(encoding="utf-8"))["events"]
        check("OKX" in s3.get("error", ""), f"错误写入 signals.json:{s3.get('error')}")
        check(s3.get("consecutive_errors") == 1, "连续错误计数")
        check(len(e3) == len(events2), "出错时事件和状态原样保留")
        check((tmp / "out" / "status.md").exists(), "status.md 已生成")
        check((tmp / "out" / "ledger.csv").exists(), "ledger.csv 已生成(出错时也保留)")
        check("资金费率" in cloud_run.status_md(sig2, events2), "status.md 显示资金费率和每日成本")
        check("回购收益率" in cloud_run.status_md(sig2, events2), "status.md 显示回购收益率")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("\n全部通过 ✅")


if __name__ == "__main__":
    main()
