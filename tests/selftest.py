"""离线自检:python -m tests.selftest

依次跑通:监控名单 → 采集 → 特征 → 规则 → 推送(打印到屏幕)→ 冷却 → 历史回填 → 事件研究。
全部使用合成数据,不联网、不推送、不动你的真实数据库。
"""
from __future__ import annotations

import logging
import shutil
import sys
import tempfile
from pathlib import Path

import pandas as pd

import backfill
import research
from cryptoradar import universe
from cryptoradar.config import DEFAULTS, _merge
from cryptoradar.monitor import Monitor
from cryptoradar.storage import Store
from tests.mock_api import FakeBinance, Market, fake_coingecko


def check(cond: bool, msg: str) -> None:
    print(("  ✔ " if cond else "  ✘ ") + msg)
    if not cond:
        raise SystemExit(1)


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    logging.basicConfig(level=logging.WARNING)
    tmp = Path(tempfile.mkdtemp(prefix="cryptoradar_selftest_"))
    try:
        market = Market()
        api = FakeBinance(market)
        universe.fetch_coingecko_top = fake_coingecko

        cfg = _merge(DEFAULTS, {
            "storage": {"db_path": str(tmp / "t.db")},
            "notify": {"channel": "console", "heartbeat_hour": None},
            "price_alerts": [{"symbol": "OPUSDT", "below": 1.0, "note": "测试价位"}],
        })
        cfg["_base_dir"] = str(tmp)

        print("\n[1] 实时监控一轮(首次运行会自动补 30 天数据)")
        mon = Monitor(cfg, api=api, dry_run=True)
        mon.run_once()
        uni = {u["symbol"] for u in mon.store.load_universe()}
        check("OPUSDT" in uni and "1000PEPEUSDT" in uni, "名单映射正确(含 1000PEPE 前缀)")
        check(not any(s.startswith(("USDC", "USDT", "WBTC")) for s in uni), "稳定币/包装币已排除")
        rows = mon.store.conn.execute("SELECT symbol, rules, pushed FROM alerts").fetchall()
        fired = {s: r for s, r, _ in rows}
        check("OI_DIV" in fired.get("OPUSDT", ""), f"OP 触发 OI 背离规则:{fired.get('OPUSDT')}")
        check("FUND_HOT" in fired.get("SOLUSDT", ""), f"SOL 触发资金费率偏高:{fired.get('SOLUSDT')}")
        pushes1 = mon.store.pushes_since(0)
        check(pushes1 == 1, "本轮合并为 1 条推送")

        print("\n[2] 立即再跑一轮:冷却期内不应重复推送")
        mon.run_once()
        check(mon.store.pushes_since(0) == pushes1, "冷却生效,没有重复推送")

        print("\n[3] 历史回填(模拟 data.binance.vision)")
        store = Store(tmp / "t.db")
        backfill._download = lambda sym, day: (day, market.metrics_zip(sym, day))
        start = pd.Timestamp(market.ts[0], unit="ms", tz="UTC").date()
        end = pd.Timestamp(market.end, unit="ms", tz="UTC").date()
        for s in ["OPUSDT", "BTCUSDT", "ETHUSDT", "SOLUSDT", "ARBUSDT"]:
            backfill.backfill_klines(store, api, s, int(market.ts[0]))
            backfill.backfill_funding(store, api, s, int(market.ts[0]))
            backfill.backfill_metrics(store, s, start, end, workers=4)
        op = store.load_hourly("OPUSDT")
        src = market.data["OPUSDT"]
        common = op.index.intersection(src.index)[:-2]
        err = (op.loc[common, "oi"] / src.loc[common, "oi"] - 1).abs().max()
        check(len(op) >= 5900, f"OP 回填 {len(op)} 小时")
        check(err < 1e-6, f"回填的 OI 与源数据逐小时对齐(最大误差 {err:.1e})")

        print("\n[4] 事件研究:应能找回埋入的规律(OI 背离后 72h 约 +7%)")
        th = research.merged_thresholds({})
        btc, eth = store.load_hourly("BTCUSDT"), store.load_hourly("ETHUSDT")
        frames = [research.symbol_frame(store, "OPUSDT", btc, eth, th)]
        res = research.run(frames, min_n=5).set_index("condition")
        base = res.iloc[0]
        div = res.loc["OI_DIV"]
        print(f"     基准 72h 残差收益 {base.resid72_mean * 100:+.2f}%(n={base.n})")
        print(f"     OI_DIV 72h 残差收益 {div.resid72_mean * 100:+.2f}%(n={div.n}, t={div.t72:+.1f})")
        # 机制检查:检测到的事件时间应落在埋点前 24 小时内(说明没有前视、对齐正确)
        f = frames[0]
        ev = research.decluster(f["C_OI_DIV"])
        H = 3_600_000
        hit = sum(any(e - 24 * H <= t <= e for t in ev) for e in market.events)
        check(hit >= 0.8 * len(market.events), f"埋入的 {len(market.events)} 次事件识别出 {hit} 次,且都在事件发生之前")
        check(div.resid72_mean > base.resid72_mean + 0.03, "信号后的 72h 收益明显高于基准")

        print("\n[5] 调参、学习权重与评分模型的样本外检验(tune.py):流程跑通,且训练期始终早于检验期")
        import numpy as np
        import tune
        pool = tune.load_frames(store, th, ["OPUSDT", "SOLUSDT", "ARBUSDT"])
        tune.GRID = {"oi_z": [2.0, 3.0], "resid_z": [2.5], "funding_z": [2.5], "min_score": [1.5, 2.5]}
        res = tune.run_tune(pool, th, folds=2, top_pct=5.0, min_n=3, min_lev=1.0)
        per_fold, total, chosen = res["per_fold"], res["total"], res["chosen"]
        print(tune.fmt(total).to_string(index=False))
        want = {"基准", "当前规则", "调参规则", "学习权重", "逻辑回归"} | ({"梯度提升"} if tune.have_sklearn() else set())
        check(set(total["方法"]) == want, f"各方法都有结果:{sorted(want)}")
        check(len(per_fold) == 2 * len(want) and len(chosen) == 2, "每轮每种方法各一行")
        check(all(np.isfinite(v) for v in res["models"]["逻辑回归"].coefs().values()), "模型系数有限")
        wb = res["weights_by_fold"][tune.RULE_IDS]
        check((wb.to_numpy() >= 0).all() and (wb.max(axis=1) <= 2.0 + 1e-9).all(), "学出的权重非负且不超过 2.0")
        # 约束:safe_lev 要求设得高到不可能满足时,调参规则/学习权重这一轮不触发,而不是勉强给结果
        strict = tune.run_tune(pool, th, folds=2, top_pct=5.0, min_n=3, min_lev=1e6)
        s_total = strict["total"].set_index("方法")
        check(all(s_total.loc[m, "n"] == 0 for m in ("调参规则", "学习权重", "逻辑回归")),
              "safe_lev 约束不可满足时不触发")
        split = tune.run_tune(pool, th, folds=1, first_train=0.7, top_pct=5.0, min_n=3, min_lev=1.0)
        check(len(split["per_fold"]) == len(want), "70/30 单次检验可运行")
        args = type("A", (), {"folds": 2, "top_pct": 5.0, "min_n": 3, "min_safe_lev": 1.0})()
        fw = tune.final_weights(pool, th, 3, 1.0)
        tune.write_report(tmp / "reports", {"滚动检验": res, "70/30": split}, args, fw)
        check((tmp / "reports" / "tune_report.md").exists(), "报告已生成")

        print("\n全部通过 ✅")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
