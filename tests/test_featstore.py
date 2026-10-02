"""永久特征库(cryptoradar/featstore.py)回归测试:python -m unittest tests.test_featstore -v"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from cryptoradar import featstore as fst
from cryptoradar.features import build_features

HOUR = fst.HOUR
T0 = int(pd.Timestamp("2026-09-28").timestamp() * 1000)


def raw(n=200, start=T0, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.arange(start, start + n * HOUR, HOUR)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    return pd.DataFrame({"open": c, "high": c * 1.01, "low": c * 0.99, "close": c, "quote_volume": 1e6,
                         "oi": 1e5 * (1 + 0.001 * np.arange(n)), "top_ls": 1.1, "taker_ratio": 1.0,
                         "spot_buy": 500.0, "spot_sell": 480.0}, index=pd.Index(idx, name="ts"))


class FeatstoreTests(unittest.TestCase):
    def test_only_closed_bars_and_funding_from_features(self):
        df = raw(10)
        f = pd.DataFrame({"funding": 0.0001}, index=df.index)
        now = int(df.index[-1]) + 30 * 60_000                 # 最后一根还没收盘
        rows = fst.hourly_rows({"A": (df, None, {})}, {"A": f}, now)
        self.assertEqual(len(rows), 9)
        self.assertTrue((rows["funding"] == 0.0001).all())
        self.assertEqual(list(rows.columns), ["ts", "symbol"] + fst.RAW_COLS)

    def test_first_seen_kept_and_gaps_filled_across_months(self):
        with tempfile.TemporaryDirectory() as tmp:
            prev, out1, out2 = Path(tmp, "prev"), Path(tmp, "o1"), Path(tmp, "o2")
            prev.mkdir()
            df = raw(120)                                      # 跨 9 月 / 10 月
            df.loc[df.index[-1], "oi"] = np.nan                # 最后一小时持仓快照还没到
            now1 = int(df.index[-1]) + 2 * HOUR
            r1 = fst.hourly_rows({"A": (df, None, {})}, {}, now1)
            res = fst.save(prev, out1, r1, None, now1)
            self.assertEqual(set(res["features"]), {"2026-09", "2026-10"})
            # 第二轮:接口修订了收盘价(不应覆盖),补上了持仓(应补上),多了一小时
            df2 = raw(121)
            df2["close"] *= 1.5
            now2 = now1 + HOUR
            r2 = fst.hourly_rows({"A": (df2, None, {})}, {}, now2)
            fst.save(out1, out2, r2, None, now2)
            back = fst.load(out2).set_index("ts")
            self.assertEqual(len(back), 121)
            self.assertAlmostEqual(back.loc[int(df.index[5]), "close"], df["close"].iloc[5], places=4, msg="首次看到的值不被覆盖")
            self.assertEqual(back.loc[int(df.index[5]), "seen_at"], now1)
            self.assertFalse(np.isnan(back.loc[int(df.index[-1]), "oi"]), "缺失的持仓后补上")
            self.assertEqual(back.loc[int(df2.index[-1]), "seen_at"], now2)

    def test_untouched_months_are_carried_and_corrupt_file_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            prev, out = Path(tmp, "prev"), Path(tmp, "out")
            (prev / "features").mkdir(parents=True)
            old = pd.DataFrame({"ts": [1, 2], "symbol": "A", "close": [1.0, 2.0], "seen_at": 1})
            old.to_csv(prev / "features" / "2025-01.csv.gz", index=False)
            (prev / "features" / "2026-09.csv.gz").write_bytes(b"not a gzip file")
            df = raw(5, start=int(pd.Timestamp("2026-09-10").timestamp() * 1000))
            now = int(df.index[-1]) + 2 * HOUR
            fst.save(prev, out, fst.hourly_rows({"A": (df, None, {})}, {}, now), None, now)
            self.assertTrue((out / "features" / "2025-01.csv.gz").exists(), "旧月份原样带过来")
            self.assertEqual((out / "features" / "2026-09.csv.gz").read_bytes(), b"not a gzip file", "损坏的文件不覆盖")
            self.assertEqual(len(list((out / "features").glob("2026-09-*.csv.gz"))), 1, "新数据另存")

    def test_fundamentals_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = fst.fundamental_rows({"PUMP": {"day": "2026-10-01", "f_hrev_7d": 7e6, "f_hrev_ratio": 1.4}})
            fst.save(Path(tmp, "p"), Path(tmp, "o"), None, rows, 123)
            back = fst.load(Path(tmp, "o"), fst.FUND_DIR)
            self.assertEqual(back.loc[0, "symbol"], "PUMP")
            self.assertEqual(back.loc[0, "seen_at"], 123)

    def test_round_trip_recomputes_same_features(self):
        df, btc = raw(400, seed=1), raw(400, seed=2)
        with tempfile.TemporaryDirectory() as tmp:
            now = int(df.index[-1]) + 2 * HOUR
            rows = fst.hourly_rows({"A": (df, None, {}), "BTC": (btc, None, {})}, {}, now)
            fst.save(Path(tmp, "p"), Path(tmp, "o"), rows, None, now)
            fr = fst.to_frames(fst.load(Path(tmp, "o")))
        a = build_features(df, btc)
        b = build_features(fr["A"], fr["BTC"])
        for c in ("resid_24h_z", "oi_z", "vol_z", "spot_buy_z", "resid_6h_z"):
            # 存盘保留 8 位有效数字,z 分数会有 1e-6 量级的舍入差
            np.testing.assert_allclose(a[c].to_numpy(), b[c].to_numpy(), rtol=1e-4, atol=1e-5, equal_nan=True, err_msg=c)


if __name__ == "__main__":
    unittest.main(verbosity=2)
