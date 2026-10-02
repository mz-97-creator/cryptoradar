"""实盘方向核对(72h / 1 周 / 2 周)与研究脚本去重修复的回归测试:python -m unittest tests.test_direction -v

全部用合成数据,不联网。
"""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

import cloud_run
import combo_study
from cryptoradar import foresight as fs
from cryptoradar import opportunity as opp

HOUR = opp.HOUR


def market(n_coins=30, hours=1200, seed=0):
    """BTC 不动、beta=1 的合成行情,这样超额收益就是币自身的对数收益,方便直接核对。"""
    rng = np.random.default_rng(seed)
    idx = np.arange(hours) * HOUR
    frames = {}
    for i in range(n_coins):
        lc = np.log(100) + np.cumsum(rng.normal(0, 0.01, hours))
        c = np.exp(lc)
        frames[f"C{i}"] = pd.DataFrame({"close": c, "_high": c * 1.002, "_low": c * 0.998,
                                         "_btc_lc": np.log(100.0), "beta": 1.0}, index=idx)
    return frames


def snapshots(frames, every_h=6, until_h=800, horizon=168, noise=0.0, seed=1, topk=3):
    """每 every_h 小时一次全部币的留档;方向分 = 未来 horizon 小时真实收益 + 噪声(noise=0 即完美预测)。"""
    rng = np.random.default_rng(seed)
    rows = []
    for t in range(0, until_h, every_h):
        t0 = t * HOUR
        fut = {s: np.log(f.at[t0 + horizon * HOUR, "close"] / f.at[t0, "close"]) for s, f in frames.items()}
        score = {s: v + rng.normal(0, noise) for s, v in fut.items()}
        order = sorted(score, key=lambda s: -score[s])
        for r, s in enumerate(order, 1):
            tier = "偏涨" if r <= topk else "偏跌" if r > len(order) - topk else "中性"
            rows.append({"t_bar": t0, "symbol": s, "p_up": .2, "p_dn": .2, "vol_range": .2, "mae_q10": -.1,
                         "dir_score": score[s], "dir_rank": r, "n_ranked": len(order), "dir_tier": tier,
                         "rules": "OI_DIV" if r == 1 else ""})
    return pd.DataFrame(rows, columns=opp.LOG_COLS)


class ResolveTests(unittest.TestCase):
    def test_horizons_match_direct_calculation(self):
        frames = market(n_coins=2, hours=600)
        log = snapshots(frames, until_h=60)
        rs = opp.resolve_log(log, frames, opp.HORIZONS)
        f = frames["C0"]
        r = rs[(rs.symbol == "C0") & (rs.t_bar == 12 * HOUR)].iloc[0]
        for h in opp.HORIZONS:
            want = np.log(f.at[(12 + h) * HOUR, "close"] / f.at[12 * HOUR, "close"])
            self.assertAlmostEqual(r[f"resid{h}"], want, places=12)
            low = f["_low"].loc[13 * HOUR:(12 + h) * HOUR].min() / f.at[12 * HOUR, "close"] - 1
            self.assertAlmostEqual(r[f"mae{h}"], low, places=12)

    def test_not_resolved_before_due_or_on_open_candle(self):
        frames = market(n_coins=1, hours=500)
        log = snapshots(frames, until_h=1, horizon=72)      # t=0 一条
        log = pd.concat([log, log.assign(t_bar=(499 - 72) * HOUR)])   # 72h 终点正好是最后一根(可能未收盘)
        rs = opp.resolve_log(log, frames, opp.HORIZONS)
        r0 = rs[rs.t_bar == 0].iloc[0]
        self.assertTrue(np.isfinite(r0["resid72"]) and np.isfinite(r0["resid168"]) and np.isfinite(r0["resid336"]))
        self.assertNotIn((499 - 72) * HOUR, set(rs.t_bar))
        rs2 = opp.resolve_log(log.assign(t_bar=300 * HOUR), frames, opp.HORIZONS)
        self.assertTrue(np.isfinite(rs2["resid168"]).all() and rs2["resid336"].isna().all(), "2 周未到期时留空")

    def test_gappy_window_not_resolved(self):
        frames = market(n_coins=1, hours=400)
        frames["C0"].loc[10 * HOUR:30 * HOUR, "_low"] = np.nan      # 72h 窗口缺 21 根(> 10%)
        rs = opp.resolve_log(snapshots(frames, until_h=1, horizon=72), frames, opp.HORIZONS)
        self.assertTrue(rs.empty or rs["resid72"].isna().all())

    def test_legacy_log_without_direction_columns(self):
        frames = market(n_coins=12, hours=600)
        old = snapshots(frames, until_h=60)[["t_bar", "symbol", "p_up", "p_dn", "vol_range", "mae_q10"]]
        log = opp.log_snapshot(old, None)
        self.assertEqual(list(log.columns), opp.LOG_COLS)
        out = opp.merge_outcomes(None, opp.resolve_log(log, frames, opp.HORIZONS))
        self.assertTrue(opp.direction_cross_sections(out, 72).empty, "旧日志没有方向分,不参与方向核对")


class DirectionSummaryTests(unittest.TestCase):
    def test_perfect_signal_scores_high_and_noise_scores_near_zero(self):
        frames = market()
        good = opp.merge_outcomes(None, opp.resolve_log(snapshots(frames, noise=0.0), frames, opp.HORIZONS))
        st = opp.direction_summary(good)["168h"]
        self.assertEqual(st["status"], "ok")
        self.assertGreater(st["ic_mean"], 0.99)
        self.assertGreater(st["spread_mean"], 0)
        self.assertEqual(st["top_hit"] is not None and st["top_hit"] >= st["base_up"], True)
        rand = opp.merge_outcomes(None, opp.resolve_log(snapshots(frames, noise=10.0), frames, opp.HORIZONS))
        st = opp.direction_summary(rand)["168h"]
        self.assertLess(abs(st["ic_mean"]), 0.1)
        self.assertLess(abs(st["ic_t"]), 3)

    def test_newey_west_shrinks_t_for_overlapping_series(self):
        rng = np.random.default_rng(3)
        x = pd.Series(np.convolve(rng.normal(0.02, 1, 600), np.ones(28), "valid"))   # 28 期重叠
        naive = x.mean() / (x.std() / np.sqrt(len(x)))
        self.assertLess(abs(opp.nw_t(x, 28)), abs(naive) / 2)

    def test_small_sample_only_reports_counts(self):
        frames = market(n_coins=12, hours=600)
        out = opp.merge_outcomes(None, opp.resolve_log(snapshots(frames, until_h=30), frames, opp.HORIZONS))
        st = opp.direction_summary(out)["72h"]
        self.assertIn("积累中", st["status"])
        self.assertNotIn("ic_mean", st)

    def test_outcomes_accumulate_beyond_log_and_fill_later_horizons(self):
        frames = market(n_coins=12, hours=1200)
        log = snapshots(frames, until_h=60)
        short = {s: f.iloc[:300] for s, f in frames.items()}        # 此时只到期了 72h / 1 周
        first = opp.merge_outcomes(None, opp.resolve_log(log, short, opp.HORIZONS))
        self.assertTrue(first["resid336"].isna().all() and first["resid168"].notna().all())
        later = opp.merge_outcomes(first, opp.resolve_log(log.iloc[:12], frames, opp.HORIZONS))   # 日志已删掉大部分
        self.assertEqual(len(later), len(first), "日志里删掉的行仍保留在结果表")
        self.assertTrue(later.set_index(["t_bar", "symbol"]).loc[(0, "C0"), "resid336"] == later.iloc[0]["resid336"])
        self.assertEqual(int(later["resid336"].notna().sum()), 12, "后到期的 2 周结果补进来")
        self.assertEqual(len(opp.merge_outcomes(later, opp.resolve_log(log, frames, opp.HORIZONS))), len(later), "幂等")

    def test_daily_csv_and_status_md(self):
        frames = market()
        out = opp.merge_outcomes(None, opp.resolve_log(snapshots(frames), frames, opp.HORIZONS))
        daily = opp.direction_daily(out)
        self.assertEqual(set(daily["horizon_h"]), set(opp.HORIZONS))
        self.assertTrue(daily["date"].str.match(r"\d{4}-\d{2}-\d{2}").all())
        md = "\n".join(cloud_run.direction_md(opp.direction_summary(out)))
        for name in ("72h", "1 周", "2 周", "Newey-West"):
            self.assertIn(name, md)


class LedgerLongHorizonTests(unittest.TestCase):
    def test_ledger_resolves_week_and_two_weeks(self):
        frames = market(n_coins=1, hours=600)
        f = frames["C0"]
        btc = pd.DataFrame({"close": 100.0, "_low": 100.0}, index=f.index)
        pr = fs.new_prediction(0, "signal", "C0", "OI_DIV", 0, 0.5, "up", .6, .5, .01, 30, rules=["OI_DIV"])
        pr = fs.resolve([pr], {"C0": f, "BTC": btc}, pd.DataFrame(), 0 + 400 * HOUR)[0]
        self.assertEqual(set(pr["res"]), {"24h", "72h", "168h", "336h"})
        led = fs.update_ledger(None, [pr])
        want = f.at[336 * HOUR, "close"] / f.at[0, "close"] - 1
        self.assertAlmostEqual(led.at[0, "resid336"], want, places=12)
        st = fs.ledger_summary(led)["by_rule"]["OI_DIV"]
        self.assertEqual((st["n168"], st["n336"]), (1, 1))
        self.assertIn("2周", "\n".join(fs.ledger_text(fs.ledger_summary(led))))


class StudyFixTests(unittest.TestCase):
    def test_h4_requires_negative_funding(self):
        n = 50
        g = pd.DataFrame({"close": np.arange(n) + 100., "ret_24h": .1, "funding": .0001, "funding_z": -2.,
                          "oi_z": 0., "spot_buy_z": 1., "spot_buy_dz12": .6})
        key = "H4 父:负费率+不再下跌"
        self.assertFalse(combo_study.conditions(g)[key].any(), "费率为正时不算负费率")
        g["funding"] = -.0001
        self.assertTrue(combo_study.conditions(g)[key][-1])

    def test_dedup_uses_real_time_gap(self):
        # 缺标签的行被删掉后,行号之差小于小时数:相隔 100 小时的两次触发只隔 2 行,按行号会被误并成一次
        ts = np.array([0, 1, 100, 101]) * HOUR
        D = pd.DataFrame({"ts": ts, "sym": "A", "r": 0.01, "C": [True, False, True, False]})
        res = combo_study.test_events(D.copy(), ["C"], pd.Series(0.0, index=ts), 72)
        self.assertEqual(int(res.iloc[0]["事件数"]), 2)
        # 真正相隔不足 72 小时的仍算一次
        D2 = D.assign(ts=np.array([0, 1, 30, 31]) * HOUR)
        res = combo_study.test_events(D2, ["C"], pd.Series(0.0, index=D2.ts), 72)
        self.assertEqual(int(res.iloc[0]["事件数"]), 1)

    def test_rule_validation_dedup_uses_real_time_gap(self):
        import rule_validation as rv
        idx = np.array([0, 1, 100, 101]) * HOUR
        f = pd.DataFrame({"symbol": "A", "fwd_resid_72h": 0.01}, index=idx)
        R = rv.validate([f], [np.array([[1], [0], [1], [0]])], ["X"])
        self.assertEqual(int(R.iloc[0]["n_events"]), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
