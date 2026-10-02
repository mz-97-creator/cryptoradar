"""lag_study.py 的回归测试(合成数据):python -m unittest tests.test_lag_study -v"""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

import lag_study as L
from cryptoradar.signals import RULES, merged_thresholds

HOUR = L.HOUR


class ZigzagTests(unittest.TestCase):
    def test_finds_trough_to_peak_swings(self):
        v = np.log([100, 95, 90, 100, 120, 130, 125, 110, 100, 105, 150, 160, 140])
        got = L.zigzag_rallies(pd.Series(v, index=np.arange(len(v)) * HOUR), 0.2, 0.1)
        self.assertEqual([(a // HOUR, b // HOUR) for a, b, _ in got], [(2, 5), (8, 11)])
        self.assertAlmostEqual(np.expm1(got[1][2]), 0.6)

    def test_small_swings_ignored(self):
        v = np.log([100, 110, 100, 112, 101, 113, 100])
        self.assertEqual(L.zigzag_rallies(pd.Series(v, index=np.arange(len(v)) * HOUR), 0.2, 0.05), [])


class ReplayTests(unittest.TestCase):
    def frame(self, n=48):
        f = pd.DataFrame(0.0, index=np.arange(n) * HOUR,
                         columns=["oi_z", "resid_24h_z", "ret_24h", "funding_z", "funding", "ret_1h_z", "vol_z",
                                  "top_ls_z", "range_z"])
        return f

    def test_cooldown_and_threshold_match_cloud(self):
        f = self.frame()
        f.loc[[0, 3 * HOUR, 7 * HOUR], "vol_z"] = 3.0              # VOL 权重 1.0:第 0 小时推送,第 3 小时在 6h 冷却内,第 7 小时再推
        th = merged_thresholds({})
        _, pushes = L.replay(f, th, RULES, need=1.0, cooldown_h=6)
        self.assertEqual(list(pushes.index // HOUR), [0, 7])
        _, none = L.replay(f, th, RULES, need=2.5, cooldown_h=6)   # 非自选门槛 2.5:单条 VOL 不够
        self.assertTrue(none.empty)

    def test_up_flavor_uses_sign(self):
        f = self.frame(4)
        f.loc[0, "resid_24h_z"], f.loc[HOUR, "resid_24h_z"] = -3.0, 3.0
        hits, _ = L.replay(f, merged_thresholds({}), RULES, 1.0, 6)
        self.assertEqual(list(L.up_flavored(f, hits)[:2]), [False, True])


class MeasureTests(unittest.TestCase):
    def test_lag_and_done_fraction(self):
        idx = np.arange(20) * HOUR
        x = pd.Series(np.log(np.r_[np.full(5, 100.), np.linspace(100, 150, 11), np.full(4, 150.)]), index=idx)
        rallies = L.zigzag_rallies(x, 0.2, 0.1) or [(4 * HOUR, 15 * HOUR, float(np.log(1.5)))]
        t0, t1, _ = rallies[0]
        hits = pd.DataFrame({"VOL": False}, index=idx)
        hits.loc[t0 + 5 * HOUR, "VOL"] = True
        pushes = pd.Series({int(t0 + 5 * HOUR): ["VOL"]}, dtype=object)
        up = pd.Series(False, index=idx)
        r = L.measure("A", x, rallies, hits, pushes, up, 48, False, {"push": 0, "uppush": 0, "fire": 0})[0]
        self.assertEqual(r["push_lag_h"], 5)
        want = (x[t0 + 5 * HOUR] - x[t0]) / (x[t1] - x[t0])
        self.assertAlmostEqual(r["push_done"], want)
        self.assertTrue(np.isnan(r["uppush_lag_h"]), "没有\"像在涨\"的推送时记为漏报")


if __name__ == "__main__":
    unittest.main(verbosity=2)
