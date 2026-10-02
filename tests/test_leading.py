from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from cryptoradar.events import append_events, available_events
from cryptoradar.leading import HOUR, append_shadow, features
from cryptoradar.barriers import BarrierSpec, label_bars
from leading_study import alerts, block_interval, training_mask
from leading_collect import history


class LeadingIntegrity(unittest.TestCase):
    def bars(self, n=800):
        ts = np.arange(n, dtype=np.int64) * HOUR
        close = 100 * np.exp(np.sin(np.arange(n) / 20) * .02 + np.arange(n) * .00001)
        return pd.DataFrame({"open": close, "high": close * 1.01, "low": close * .99,
                             "close": close, "quote_volume": 1e6 + np.arange(n)}, index=pd.Index(ts, name="ts"))

    def test_future_price_and_flow_do_not_change_past_features(self):
        s = self.bars()
        flow = pd.DataFrame({"buy": 60., "sell": 40.}, index=s.index)
        funding = pd.DataFrame({"funding": [.001, .005], "interval_h": [8, 8]}, index=pd.Index([500 * HOUR, 750 * HOUR], name="ts"))
        before = features(s, s, s, flow, funding=funding)
        altered, future_flow = s.copy(), flow.copy()
        altered.loc[601 * HOUR:, ["open", "high", "low", "close", "quote_volume"]] *= 3
        future_flow.loc[601 * HOUR:, "buy"] = 999
        after = features(altered, altered, altered, future_flow, funding=funding)
        pd.testing.assert_frame_equal(before.loc[:600 * HOUR], after.loc[:600 * HOUR])
        self.assertTrue(pd.isna(before.loc[498 * HOUR, "funding"]))
        self.assertEqual(before.loc[500 * HOUR, "funding"], .001)

    def test_missing_flow_is_unknown_not_zero(self):
        s = self.bars()
        out = features(s, s, s)
        self.assertTrue(out.spot_buy_share_24h.isna().all())
        self.assertTrue(out.oi_change_24h.isna().all())

    def test_shadow_records_first_real_observation_not_bar_time(self):
        f = pd.DataFrame({"symbol": ["OP"], "available_ts": [HOUR], "spot_buy_share_24h": [.6]})
        first = append_shadow(None, f, 5 * HOUR)
        second = append_shadow(first, f.assign(spot_buy_share_24h=.9), 6 * HOUR)
        self.assertEqual(len(second), 1)
        self.assertEqual(second.recorded_ts.iloc[0], 5 * HOUR)
        self.assertEqual(second.spot_buy_share_24h.iloc[0], .6)

    def test_event_backfill_cannot_be_known_in_past(self):
        event = {"kind": "listing", "source_url": "https://www.okx.com/help/test", "symbols": ["OP"], "published_ts": HOUR, "title": "OP listed"}
        with tempfile.TemporaryDirectory() as t:
            path = Path(t) / "events.jsonl"
            first = append_events(path, [event], 10 * HOUR)
            self.assertEqual(available_events(first, "OP", 9 * HOUR), [])
            second = append_events(path, [event], 12 * HOUR)
            self.assertEqual(len(second), 1)
            revised = append_events(path, [dict(event, title="Updated OP listing")], 15 * HOUR)
            self.assertEqual(revised[-1]["first_seen_ts"], 10 * HOUR)
            self.assertEqual(available_events(revised, "OP", 14 * HOUR)[0]["title"], "OP listed")
            self.assertEqual(available_events(revised, "OP", 16 * HOUR)[0]["title"], "Updated OP listing")

    def test_estimated_buyback_not_accepted_as_execution(self):
        with tempfile.TemporaryDirectory() as t:
            with self.assertRaises(ValueError):
                append_events(Path(t) / "events.jsonl", [{"kind": "buyback_execution", "source_url": "https://example.com", "amount_usd": 100000}], 0)

    def test_quote_refresh_is_not_an_announcement_revision(self):
        event = {"kind": "listing", "source_url": "https://www.okx.com/help/test", "symbols": ["OP"], "title": "OP listed", "observed_quotes": {"OP": {"price": 1}}}
        with tempfile.TemporaryDirectory() as t:
            path = Path(t) / "events.jsonl"
            append_events(path, [event], 0)
            result = append_events(path, [dict(event, observed_quotes={"OP": {"price": 2}})], HOUR)
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0]["observed_quotes"]["OP"]["price"], 1)

    def test_collection_excludes_unconfirmed_and_future_candles(self):
        class API:
            def _get(self, path, params):
                return [[str(2 * HOUR), '100', '110', '90', '100', '1', '1', '1000', '0'],
                        [str(HOUR), '100', '110', '90', '100', '1', '1', '1000', '1'],
                        ['0', '100', '110', '90', '100', '1', '1', '1000', '1']]
        self.assertEqual(history(API(), "OP-USDT", 0, 2 * HOUR).ts.tolist(), [0, HOUR])

    def test_collection_stops_on_nonprogressing_page(self):
        class API:
            def _get(self, path, params):
                return [[str(params['after']), '100', '110', '90', '100', '1', '1', '1000', '1']]
        with self.assertRaisesRegex(RuntimeError, "Non-progressing"):
            history(API(), "OP-USDT", 0, 2 * HOUR)

    def test_purge_uses_full_horizon_not_just_early_success(self):
        data = pd.DataFrame({"candidate": [True, True], "target": [1., 1.],
                             "entry_ts": [HOUR, 9 * HOUR], "label_end_ts": [2 * HOUR, 10 * HOUR]})
        mask = training_mask(data, 12 * HOUR, {"embargo_h": 1, "horizon_h": 7})
        self.assertEqual(mask.tolist(), [True, False])

    def test_selection_cannot_skip_unknown_outcome(self):
        data = pd.DataFrame({"candidate": [True, True], "score": [.8, .7], "target": [np.nan, 1.],
                             "entry_ts": [0, 0], "symbol": ["OP", "AAVE"]})
        chosen = alerts(data, "tree_strength", {"model_probability_floor": .15, "max_alerts_per_decision": 1})
        self.assertEqual(chosen.symbol.tolist(), ["OP"])

    def test_startup_time_is_an_interval_and_stride_uses_entry_time(self):
        bars = pd.DataFrame({"open": [100, 100, 101], "high": [101, 110, 125], "low": [99, 99, 100], "close": [100, 101, 122]},
                            index=np.arange(3) * HOUR)
        r = label_bars(bars, BarrierSpec(horizon_h=2), startup_return=.05).iloc[0]
        self.assertEqual(r.startup_earliest_ts, HOUR)
        self.assertEqual(r.startup_latest_ts, 2 * HOUR)
        self.assertEqual(r.outcome, "profit")
        rows = label_bars(bars, BarrierSpec(horizon_h=2), decision_stride_h=2)
        self.assertTrue((rows.entry_ts % (2 * HOUR) == 0).all())

    def test_bootstrap_keeps_calendar_days_and_handles_unknowns(self):
        plan = {"bootstrap_block_days": 14, "bootstrap_repetitions": 50}
        days = np.arange(90) * 24 * HOUR
        b = pd.DataFrame({"entry_ts": days, "net_return": .01, "target": 1.})
        r = block_interval(b, b, days, plan)
        self.assertEqual(r["ci95"], [0., 0.])
        unknown = b.copy()
        unknown.loc[0, "target"] = np.nan
        self.assertIn("unknown", block_interval(unknown, b, days, plan)["status"])


if __name__ == "__main__":
    unittest.main()
