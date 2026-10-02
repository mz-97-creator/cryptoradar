"""事件库(cryptoradar/events.py)与事件检验(event_study.py)的回归测试:python -m unittest tests.test_events -v"""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

import event_study as ES
from cryptoradar import events as ev

DAY = ev.DAY
HOUR = 3_600_000


class ClassifyTests(unittest.TestCase):
    def test_titles(self):
        cases = {
            "Binance Futures Will Launch USDⓈ-Margined CTUSDT Perpetual Contract (2026-10-01)": [("perp_list", "CT")],
            "Binance Futures Will Launch USDⓈ-Margined 1000XUSDT and ABCUSDT Perpetual Contracts": [("perp_list", "X"), ("perp_list", "ABC")],
            "Binance Will List Plasma (XPL) with Seed Tag Applied": [("spot_list", "XPL")],
            "OKX to list CARDS/USDT (Collector Crypt) for spot trading": [("spot_list", "CARDS")],
            "Binance HODLer Airdrops: Kite (KITE) - Earn KITE": [("spot_list", "KITE")],
            "OKX to list perpetual futures for XYZ, ABC": [("perp_list", "XYZ"), ("perp_list", "ABC")],
            "Binance Futures Will Launch Multiple TradFi USDⓈ-Margined Perpetual Contracts (2026-09-29)": [],
            "Binance Will Add 7 bStocks Tokenized Securities as Collateral Asset": [],
            "Binance Will Delist ABC, DEF": [],
        }
        for t, want in cases.items():
            self.assertEqual(ev.classify(t), want, t)


class StoreTests(unittest.TestCase):
    def test_listing_diff_first_run_builds_set_only(self):
        rows, st = ev.diff_listings(None, {"upbit": {"BTC": None, "ETH": None}}, now=100)
        self.assertEqual(rows, [])
        rows, st = ev.diff_listings(st, {"upbit": {"BTC": None, "ETH": None, "OP": None},
                                         "okx_spot": {"A": 5}}, now=200)
        self.assertEqual([(r["symbol"], r["event_ts"]) for r in rows], [("OP", 200)], "OKX 第一次出现只建清单")
        self.assertEqual(st["upbit"], ["BTC", "ETH", "OP"])

    def test_merge_keeps_first_seen_and_updates_buyback_amount(self):
        a = ev.merge(None, [{"id": "x", "kind": "spot_list", "source": "upbit", "symbol": "OP", "event_ts": 1, "title": "t"},
                            {"id": "b", "kind": "buyback_day", "source": "hl", "symbol": "HYPE", "event_ts": DAY,
                             "amount_usd": 1.0, "title": "t"}], now=10)
        b = ev.merge(a, [{"id": "x", "kind": "spot_list", "source": "upbit", "symbol": "OP", "event_ts": 1, "title": "t"},
                         {"id": "b", "kind": "buyback_day", "source": "hl", "symbol": "HYPE", "event_ts": DAY,
                          "amount_usd": 5.0, "title": "t"}], now=20)
        self.assertEqual(len(b), 2)
        self.assertTrue((b["seen_at"] == 10).all(), "首次看到的时间不变")
        self.assertEqual(b.set_index("id").at["b", "amount_usd"], 5.0, "当天回购额更新为最新累计值")

    def test_buyback_days_and_accel(self):
        t0 = int(pd.Timestamp("2025-01-01").timestamp() * 1000)
        fills = pd.DataFrame({"time": [t0 + d * DAY + HOUR for d in range(60)], "px": 1.0,
                              "sz": [100.0 if d < 50 else 300.0 for d in range(60)], "tid": range(60)})
        fills["usd"] = fills["px"] * fills["sz"]
        rows = ev.buyback_days(fills)
        self.assertEqual(len(rows), 60)
        self.assertEqual(rows[0]["event_ts"], t0 + DAY, "当天结束才算已知")
        acc = ES.buyback_accel(pd.DataFrame(rows))
        first = pd.to_datetime(acc["event_ts"].min(), unit="ms")
        # 第 51 天(2-20)回购翻 3 倍:近 7 天 900 / 前 4 周周均 700 = 1.29 倍,还不够;
        # 第 52 天(2-21)1100 / 700 = 1.57 倍,当天结束(2-22 00:00)触发
        self.assertEqual(first, pd.Timestamp("2025-02-22"))


class WindowTests(unittest.TestCase):
    def test_cum_excess_and_window(self):
        idx = np.arange(0, 200) * HOUR
        base = pd.DataFrame({"_btc_lc": np.log(100.0), "beta": 1.0}, index=idx)
        a = base.assign(close=np.exp(np.log(100) + 0.01 * np.arange(200)))     # 每小时多涨 1%
        b = base.assign(close=100.0)
        C = ES.cum_excess({"A": a, "B": b})
        self.assertAlmostEqual(ES.window(C, "A", 100 * HOUR, 0, 24), 0.5 * 0.01 * 24, places=9)
        self.assertAlmostEqual(ES.window(C, "A", 100 * HOUR, -72, 0), 0.5 * 0.01 * 72, places=9)
        self.assertTrue(np.isnan(ES.window(C, "A", 190 * HOUR, 0, 24)), "超出数据范围")


if __name__ == "__main__":
    unittest.main(verbosity=2)


class RecentTextTests(unittest.TestCase):
    def test_uses_official_time_and_dedupes(self):
        now = 100 * DAY
        rows = [{"id": "a", "kind": "spot_list", "source": "okx_ann", "symbol": "OLD", "event_ts": now - 60 * DAY, "title": "old"},
                {"id": "b", "kind": "spot_list", "source": "okx_ann", "symbol": "NEW", "event_ts": now - HOUR, "title": "NEW/USD"},
                {"id": "c", "kind": "spot_list", "source": "okx_ann", "symbol": "NEW", "event_ts": now - HOUR, "title": "NEW/USDT"}]
        lines = ev.recent_text(ev.merge(None, rows, now), now, 48)
        self.assertEqual(len(lines), 1, "旧公告不算近期事件,同一天两个交易对只列一条")
        self.assertIn("NEW", lines[0])
