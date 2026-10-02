"""Regression tests for price/time leakage, barrier ordering and missing data."""
from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from cryptoradar.barriers import BarrierSpec, HOUR, holdout_masks, label_bars
from cryptoradar.foresight import new_prediction, resolve, scorecard
from cryptoradar.quality import append_research, frame_quality


class ResearchIntegrity(unittest.TestCase):
    def bars(self, rows):
        return pd.DataFrame(rows, columns=["open", "high", "low", "close"],
                            index=np.arange(len(rows)) * HOUR)

    def test_next_open_and_costs(self):
        bars = self.bars([[50, 51, 49, 50], [100, 121, 99, 120], [120, 120, 119, 119]])
        r = label_bars(bars, BarrierSpec(horizon_h=2)).iloc[0]
        self.assertEqual(r.entry_price, 100)
        self.assertEqual(r.outcome, "profit")
        self.assertAlmostEqual(r.net_return, .20)
        self.assertGreater(r.exit_price, 120)  # fees must be earned back

    def test_both_barriers_unknown_not_success(self):
        bars = self.bars([[100, 101, 99, 100], [100, 125, 85, 110]])
        r = label_bars(bars, BarrierSpec(horizon_h=2)).iloc[0]
        self.assertEqual(r.outcome, "ambiguous")
        self.assertTrue(pd.isna(r.target))

    def test_opening_gap_has_known_order(self):
        bars = self.bars([[100, 101, 99, 100], [100, 110, 95, 100], [80, 125, 75, 110]])
        r = label_bars(bars, BarrierSpec(horizon_h=2)).iloc[0]
        self.assertEqual(r.outcome, "stop")
        self.assertEqual(r.exit_price, 80)
        self.assertLess(r.net_return, -.10)

    def test_incomplete_future_not_negative(self):
        bars = self.bars([[100, 101, 99, 100], [100, 105, 95, 100]])
        r = label_bars(bars, BarrierSpec(horizon_h=168)).iloc[0]
        self.assertEqual(r.outcome, "censored")
        self.assertTrue(pd.isna(r.target))

    def test_missing_hour_not_negative(self):
        bars = self.bars([[100, 101, 99, 100]] * 4).drop(index=2 * HOUR)
        r = label_bars(bars, BarrierSpec(horizon_h=3)).iloc[0]
        self.assertEqual(r.outcome, "censored")

    def test_no_training_label_crosses_holdout(self):
        labels = pd.DataFrame({"target": [1., 0., 1., np.nan],
                               "entry_ts": [0, HOUR, 4 * HOUR, 5 * HOUR],
                               "label_end_ts": [HOUR, 4 * HOUR, 6 * HOUR, None]})
        train, test = holdout_masks(labels, 4 * HOUR)
        self.assertEqual(train.tolist(), [True, False, False, False])
        self.assertEqual(test.tolist(), [False, False, True, False])

    def test_settlement_uses_saved_price_and_closed_endpoint(self):
        ts = np.arange(27) * HOUR
        f = pd.DataFrame({"close": [999.] + [110.] * 26, "_low": [1.] + [105.] * 26}, index=ts)
        p = new_prediction(HOUR // 2, "signal", "OP", "OI_DIV", 0, 100., "none", None, None, None, 0)
        p["reference_btc_price"] = 100.
        resolve([p], {"OP": f, "BTC": f}, pd.DataFrame(), 24 * HOUR + HOUR // 2)
        self.assertNotIn("24h", p["res"])  # endpoint bar not closed
        resolve([p], {"OP": f, "BTC": f}, pd.DataFrame(), 25 * HOUR)
        r = p["res"]["24h"]
        self.assertAlmostEqual(r["ret"], .10)
        self.assertEqual(r["actual_h"], 24.5)
        self.assertFalse(r["mae_complete"])
        self.assertAlmostEqual(r["mae"], .05)  # entry-hour low excluded
        self.assertAlmostEqual(r["btc"], .10)

    def test_legacy_predictions_keep_old_semantics(self):
        f = pd.DataFrame({"close": [100.] + [110.] * 25, "_low": [95.] * 26}, index=np.arange(26) * HOUR)
        p = new_prediction(0, "signal", "OP", "OI_DIV", 0, None, "none", None, None, None, 0)
        resolve([p], {"OP": f}, pd.DataFrame(), 25 * HOUR)
        self.assertAlmostEqual(p["res"]["24h"]["ret"], .10)

    def test_first_snapshot_immutable_and_no_partial_bars(self):
        f = self.bars([[100, 101, 99, 100]] * 3)
        old = append_research(None, {"OP": f}, 2 * HOUR + 1)
        self.assertEqual(len(old), 2)
        changed = f.copy()
        changed["close"] = 999
        new = append_research(old, {"OP": changed}, 3 * HOUR)
        self.assertEqual(len(new), 3)
        self.assertEqual(new.iloc[0]["close"], 100)
        self.assertEqual(new.iloc[0]["observed_at"], 2 * HOUR + 1)

    def test_freshness(self):
        f = self.bars([[100, 101, 99, 100]])
        self.assertFalse(frame_quality(f, 3 * HOUR)["usable"])
        self.assertTrue(frame_quality(f, HOUR // 2)["usable"])
        self.assertFalse(frame_quality(f, -HOUR)["usable"])

    def test_partial_universe_cannot_emit_market_light(self):
        import cloud_run
        from zoneinfo import ZoneInfo
        from cryptoradar.config import DEFAULTS, _merge
        from tests.cloud_selftest import FakeOKX
        from tests.mock_api import Market, fake_coingecko

        cfg = _merge(DEFAULTS, {"universe": {"watchlist": ["OP"]}, "opportunity": {"enabled": False}})
        api = FakeOKX(Market(hours=1000))
        collect = cloud_run.collect

        def some_stale(okx, ccy, inst):
            df, funding, live = collect(okx, ccy, inst)
            return (df.iloc[:-4] if ccy in {"OP", "SOL"} else df), funding, live

        with patch.object(cloud_run, "fetch_coingecko_top", fake_coingecko), patch.object(cloud_run, "collect", some_stale):
            sig, events, *_ = cloud_run.run(cfg, api, {}, [], ZoneInfo("UTC"))
        self.assertFalse(sig["quality"]["market_usable"])
        self.assertEqual(sig["market_state"], {})
        self.assertFalse(any(e["type"] == "market" for e in events))
        self.assertFalse(any(e["symbol"] in {"OP", "SOL"} for e in events))

    def test_stale_btc_blocks_entire_scan(self):
        import cloud_run
        from zoneinfo import ZoneInfo
        from cryptoradar.config import DEFAULTS
        from tests.cloud_selftest import FakeOKX
        from tests.mock_api import Market, fake_coingecko

        collect = cloud_run.collect

        def stale_btc(okx, ccy, inst):
            df, funding, live = collect(okx, ccy, inst)
            return (df.iloc[:-4] if ccy == "BTC" else df), funding, live

        with patch.object(cloud_run, "fetch_coingecko_top", fake_coingecko), patch.object(cloud_run, "collect", stale_btc):
            with self.assertRaisesRegex(RuntimeError, "BTC 数据陈旧"):
                cloud_run.run(DEFAULTS, FakeOKX(Market(hours=1000)), {}, [], ZoneInfo("UTC"))


if __name__ == "__main__":
    unittest.main()
