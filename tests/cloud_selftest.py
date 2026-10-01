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
    sig, events, state = cloud_run.run(cfg, FakeOKX(m), {}, [], tz)
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
    sig2, events2, state2 = cloud_run.run(cfg, FakeOKX(m), state, events, tz)
    check(sig2["new_events"] == 0, "没有重复事件")
    check(len(events2) == len(events), "事件日志保留")

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
        check("资金费率" in cloud_run.status_md(sig2, events2), "status.md 显示资金费率和每日成本")
        check("回购收益率" in cloud_run.status_md(sig2, events2), "status.md 显示回购收益率")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("\n全部通过 ✅")


if __name__ == "__main__":
    main()
