"""早期检测(cryptoradar/early.py)的回归测试,合成数据:python -m unittest tests.test_early -v"""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from cryptoradar import early

HOUR = early.HOUR
TH = early.thresholds(None)


def frame(n=900, drift=0.0, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.arange(n) * HOUR
    lc = np.log(100) + np.cumsum(rng.normal(drift, 0.004, n))
    f = pd.DataFrame({"close": np.exp(lc), "_high": np.exp(lc) * 1.002, "_low": np.exp(lc) * 0.998,
                      "_btc_lc": np.log(100.0), "beta": 1.0, "xs_lc": lc - np.log(100.0)}, index=idx)
    for c in ("resid_6h", "resid_6h_z", "resid_24h_z", "ret_24h", "funding", "oi_72h_z", "oi_up_days", "oi_z",
              "resid_72h"):
        f[c] = 0.0
    return f


class DetectorTests(unittest.TestCase):
    def last(self, f, **cols):
        for k, v in cols.items():
            f.loc[f.index[-1], k] = v
        return [d.id for d in early.evaluate_last(f, TH)]

    def test_each_detector_fires_on_its_pattern(self):
        self.assertEqual(self.last(frame(), resid_6h_z=2.5, resid_6h=0.03, resid_24h_z=1.0), ["E_RESID6"])
        self.assertEqual(self.last(frame(), resid_6h_z=2.5, resid_6h=0.03, resid_24h_z=3.0), [],
                         "24h 已经是 RESID 级别的大涨,不算早期")
        self.assertEqual(self.last(frame(), resid_6h_z=-2.5, resid_6h=-0.03), [], "只看向上")
        self.assertEqual(self.last(frame(), oi_72h_z=2.0, oi_up_days=3, oi_z=1.0, resid_72h=0.01), ["E_OI_BUILD"])
        self.assertEqual(self.last(frame(), oi_72h_z=2.0, oi_up_days=2, oi_z=1.0), [], "三段都要在增加")
        self.assertEqual(self.last(frame(), oi_72h_z=2.0, oi_up_days=3, oi_z=2.5), [], "单次激增归 OI 规则")
        self.assertEqual(self.last(frame(), funding=-0.0001, resid_24h_z=1.5, ret_24h=0.05), ["E_SQUEEZE"])
        self.assertEqual(self.last(frame(), funding=0.0001, resid_24h_z=1.5, ret_24h=0.05), [], "费率为正不算")

    def test_breakout_needs_new_high_and_rising_rank(self):
        frames = {f"C{i}": frame(seed=i) for i in range(10)}
        g = frames["C0"]
        n = len(g)
        # C0 先跑输、最后 100 小时稳步走强并创 20 日新高(每小时涨幅很小,不是急拉);3 天前它的 7 日排名还很低
        lc = np.log(100) + np.r_[np.linspace(0, -0.3, n - 100), np.linspace(-0.3, 0.05, 100)]
        g["close"], g["xs_lc"] = np.exp(lc), lc - np.log(100.0)
        out = early.add_cross_section(frames)
        fired = early.evaluate_frame(out["C0"], TH)["E_BREAKOUT"]
        self.assertTrue(fired.iloc[-100:].any(), "上涨段里创新高、排名上升时触发")
        self.assertFalse(fired.iloc[:-100].any(), "下跌段不触发")
        r = out["C0"][fired].iloc[0]
        self.assertTrue(r["xs_high"] > 0 and r["rs_rank"] >= 0.7 and r["rs_rank_chg"] >= 0.15)
        self.assertFalse(early.evaluate_frame(out["C3"], TH)["E_BREAKOUT"].iloc[-100:].any(), "随机游走的币不触发")


class CloudTests(unittest.TestCase):
    def frames(self):
        fr = {f"C{i}": frame(seed=i) for i in range(12)}
        fr["C0"].loc[fr["C0"].index[-1], ["resid_6h_z", "resid_6h", "resid_24h_z"]] = [2.5, 0.03, 1.0]
        return fr

    def test_scan_cooldown_and_log(self):
        fr = self.frames()
        now = int(fr["C0"].index[-1]) + 30 * 60_000
        uni = [{"ccy": "C0", "watch": True, "rank": 5}]
        firing, rows, last = early.cloud_scan(fr, uni, TH, now, {})
        self.assertEqual([x["symbol"] for x in firing], ["C0"])
        self.assertEqual(firing[0]["new"], ["E_RESID6"])
        self.assertEqual(list(rows.columns), early.LOG_COLS)
        self.assertEqual(len(rows), 1)
        f2, rows2, _ = early.cloud_scan(fr, uni, TH, now + 2 * HOUR, last)
        self.assertEqual(len(rows2), 0, "24 小时内不重复留档")
        self.assertNotIn("new", f2[0])
        _, rows3, _ = early.cloud_scan(fr, uni, TH, now + 25 * HOUR, last)
        self.assertEqual(len(rows3), 1)

    def test_resolve_against_market_and_summary(self):
        fr = {f"C{i}": frame(n=1200, seed=i) for i in range(12)}
        rows = [{"t_bar": t * HOUR, "ts": t * HOUR, "symbol": "C0", "watch": 0, "detector": "E_RESID6", "price": 1}
                for t in range(100, 700, 12)]
        log = early.resolve(pd.DataFrame(rows), fr)
        f = fr["C0"]
        t0 = 100 * HOUR
        want = np.log(f.at[t0 + 336 * HOUR, "close"] / f.at[t0, "close"])
        self.assertAlmostEqual(log.loc[0, "resid336"], want, places=10)
        mkt = np.mean([np.log(g.at[t0 + 336 * HOUR, "close"] / g.at[t0, "close"]) for g in fr.values()])
        self.assertAlmostEqual(log.loc[0, "mkt336"], mkt, places=10)
        late = log[log.t_bar >= (1199 - 72) * HOUR]
        self.assertTrue(late["resid72"].isna().all(), "未到期的不结算")
        again = early.resolve(log, fr)
        pd.testing.assert_frame_equal(again, log)
        st = early.summary(log)["E_RESID6"]
        self.assertEqual(st["total"], len(rows))
        self.assertGreaterEqual(st["72h"]["n"], 30)
        self.assertIn("t", st["72h"])
        self.assertTrue(any("E_RESID6" in x for x in early.text([], early.summary(log))))


if __name__ == "__main__":
    unittest.main(verbosity=2)


class FundamentalsTests(unittest.TestCase):
    def test_daily_features_and_two_day_lag(self):
        from cryptoradar import fundamentals as fd
        day0 = int(pd.Timestamp("2025-03-01").timestamp())
        hrev = {str(day0 + k * 86400): (200_000.0 if k >= 35 else 100_000.0) for k in range(43)}
        d = fd.daily_features({"hrev": hrev, "fees": hrev, "rev": hrev})
        last = d.iloc[-1]                                      # 最后一天(未收完)已丢掉:第 41 天
        self.assertEqual(d.index[-1], pd.Timestamp("2025-03-01") + pd.Timedelta(days=41))
        self.assertAlmostEqual(last["f_hrev_7d"], 7 * 200_000)
        self.assertAlmostEqual(last["f_hrev_ratio"], 2.0)
        t = int(pd.Timestamp("2025-04-05").timestamp() * 1000)     # 第 35 天 00:00 起只能看到第 33 天的数据
        f = pd.DataFrame({"close": 1.0}, index=np.arange(t, t + 3 * 86_400_000, HOUR))
        a = fd.attach_history(f, d)
        self.assertAlmostEqual(a["f_hrev_7d"].iloc[0], d.loc["2025-04-03", "f_hrev_7d"])
        self.assertAlmostEqual(a["f_hrev_7d"].iloc[-1], d.loc["2025-04-05", "f_hrev_7d"])
        self.assertLess(a["f_hrev_7d"].iloc[0], 7 * 200_000, "不能用到还没公布的数据")


class RevenueGuardTests(unittest.TestCase):
    def test_tiny_base_spike_is_not_acceleration(self):
        f = frame(n=2)
        f.loc[f.index[-1], ["f_hrev_7d", "f_hrev_ratio", "f_rev_7d", "f_rev_ratio"]] = [np.nan, np.nan, 250_000, 32.8]
        self.assertNotIn("F_REV_UP", [d.id for d in early.evaluate_last(f, TH)], "XRP 式:基数约 7.6k/周,一次突变 32 倍")
        f.loc[f.index[-1], ["f_rev_7d", "f_rev_ratio"]] = [250_000, 1.5]
        self.assertIn("F_REV_UP", [d.id for d in early.evaluate_last(f, TH)], "基数 16.7 万/周,1.5 倍")
