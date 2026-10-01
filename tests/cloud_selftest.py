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
            return [{"instId": f"{c}-USDT-SWAP", "settleCcy": "USDT", "ctType": "linear",
                     "state": "live", "ctValCcy": c} for c in CCY]
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
