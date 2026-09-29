"""离线模拟的币安接口:用合成数据跑通整条流水线,不需要联网。

OPUSDT 里埋了一个已知规律:每次 "OI 在 24h 内 +40% 而价格不动" 之后,
接下来 72 小时价格额外上涨约 7%。事件研究如果能把它找出来,说明特征、
标签对齐和去重叠逻辑没有前视偏差之类的错误。
"""
from __future__ import annotations

import io
import time
import zipfile

import numpy as np
import pandas as pd

from cryptoradar.binance_api import HOUR_MS, BinanceFutures

SYMBOLS = {
    "BTCUSDT": ("BTC", 1.0, 60000.0),
    "ETHUSDT": ("ETH", 1.2, 2500.0),
    "OPUSDT": ("OP", 1.5, 0.13),
    "SOLUSDT": ("SOL", 1.3, 100.0),
    "ARBUSDT": ("ARB", 1.5, 0.3),
    "1000PEPEUSDT": ("1000PEPE", 2.0, 0.01),
    "DOGEUSDT": ("DOGE", 1.4, 0.2),
}
HISTORY_HOURS = 6000
PLANT_EVERY = 300


def _now_hour() -> int:
    ms = int(time.time() * 1000)
    return ms - ms % HOUR_MS


class Market:
    def __init__(self, seed: int = 7, hours: int = HISTORY_HOURS):
        rng = np.random.default_rng(seed)
        self.end = _now_hour()  # 当前小时(未收盘)的开盘时间
        self.ts = np.arange(self.end - (hours - 1) * HOUR_MS, self.end + HOUR_MS, HOUR_MS)
        n = len(self.ts)
        btc_r = rng.normal(0, 0.006, n)
        self.data: dict[str, pd.DataFrame] = {}
        self.events: list[int] = []
        for sym, (_, beta, p0) in SYMBOLS.items():
            idio = rng.normal(0, 0.004 if sym == "OPUSDT" else 0.012, n) if sym != "BTCUSDT" else np.zeros(n)
            r = beta * btc_r + idio if sym != "BTCUSDT" else btc_r
            oi_r = rng.normal(0, 0.01, n)
            if sym == "OPUSDT":
                for e in range(400, n - 100, PLANT_EVERY):
                    oi_r[e - 24:e] += np.log(1.4) / 24          # OI 24h +40%
                    r[e - 24:e] = beta * btc_r[e - 24:e]          # 价格只跟随 BTC
                    r[e + 1:e + 73] += 0.07 / 72                  # 之后 72h 额外 +7%
                    self.events.append(int(self.ts[e]))
                # 最后一天也埋一次,让实时监控能触发
                oi_r[-24:] += np.log(1.35) / 24
                r[-24:] = beta * btc_r[-24:] - 0.002
            close = p0 * np.exp(np.cumsum(r) - np.cumsum(r)[-1])
            op = np.r_[close[0], close[:-1]]
            hi = np.maximum(op, close) * (1 + np.abs(rng.normal(0, 0.004, n)))
            lo = np.minimum(op, close) * (1 - np.abs(rng.normal(0, 0.004, n)))
            vol = np.exp(rng.normal(12, 0.3, n)) / close
            oi = 1e8 / p0 * np.exp(np.cumsum(oi_r) - np.cumsum(oi_r)[-1])
            funding = np.full(n, 0.0001) + rng.normal(0, 0.00005, n)
            if sym == "SOLUSDT":
                funding[-10:] = 0.0015  # 多头拥挤
            self.data[sym] = pd.DataFrame({
                "open": op, "high": hi, "low": lo, "close": close, "volume": vol,
                "oi": oi, "top_ls": np.exp(rng.normal(0.3, 0.1, n)),
                "taker": np.exp(rng.normal(0, 0.05, n)), "funding": funding,
            }, index=self.ts)

    # data.binance.vision 的每日 metrics 文件
    def metrics_zip(self, sym: str, day: str) -> bytes | None:
        d = self.data.get(sym)
        if d is None:
            return None
        start = int(pd.Timestamp(day, tz="UTC").timestamp() * 1000)
        rows = []
        for t5 in range(start, start + 24 * HOUR_MS, 5 * 60_000):
            bar = ((t5 - 1) // HOUR_MS) * HOUR_MS
            if bar not in d.index or t5 > self.end:
                continue
            x = d.loc[bar]
            rows.append({
                "create_time": pd.Timestamp(t5, unit="ms", tz="UTC").strftime("%Y-%m-%d %H:%M:%S"),
                "symbol": sym, "sum_open_interest": x.oi, "sum_open_interest_value": x.oi * x.close,
                "count_toptrader_long_short_ratio": 1.1, "sum_toptrader_long_short_ratio": x.top_ls,
                "count_long_short_ratio": 1.0, "sum_taker_long_short_vol_ratio": x.taker,
            })
        if not rows:
            return None
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            z.writestr(f"{sym}-metrics-{day}.csv", pd.DataFrame(rows).to_csv(index=False))
        return buf.getvalue()


class FakeBinance(BinanceFutures):
    def __init__(self, market: Market):
        super().__init__(min_interval=0)
        self.m = market
        self.calls = 0

    def _get(self, path, params=None):
        self.calls += 1
        p = params or {}
        m = self.m
        if path == "/fapi/v1/exchangeInfo":
            return {"symbols": [
                {"symbol": s, "baseAsset": b, "contractType": "PERPETUAL",
                 "quoteAsset": "USDT", "status": "TRADING"} for s, (b, _, _) in SYMBOLS.items()]}
        d = m.data.get(p.get("symbol")) if p.get("symbol") else None
        if path == "/fapi/v1/klines":
            sel = d[d.index >= p.get("startTime", 0)].iloc[: p["limit"]]
            return [[int(t), str(r.open), str(r.high), str(r.low), str(r.close), str(r.volume),
                     int(t) + HOUR_MS - 1, str(r.volume * r.close), 1000,
                     str(r.volume * 0.5), str(r.volume * r.close * 0.5), "0"]
                    for t, r in sel.iterrows()]
        if path == "/fapi/v1/premiumIndex":
            return [{"symbol": s, "markPrice": str(x.close.iloc[-1]),
                     "lastFundingRate": str(x.funding.iloc[-1]),
                     "nextFundingTime": m.end + HOUR_MS} for s, x in m.data.items()]
        if path == "/fapi/v1/fundingInfo":
            return [{"symbol": "ARBUSDT", "fundingIntervalHours": 4}]
        if path == "/fapi/v1/fundingRate":
            sel = d.iloc[::8]
            sel = sel[sel.index >= p.get("startTime", 0)].iloc[: p["limit"]]
            return [{"symbol": p["symbol"], "fundingTime": int(t), "fundingRate": str(r.funding)}
                    for t, r in sel.iterrows()]
        if path == "/fapi/v1/openInterest":
            return {"symbol": p["symbol"], "openInterest": str(d.oi.iloc[-1]), "time": m.end}
        if path.startswith("/futures/data/"):
            # 快照时间 = 整点;只保留 30 天;值取上一根 K 线的状态
            end = min(p.get("endTime", m.end), m.end)
            snaps = [t for t in d.index if t <= end and t >= m.end - 30 * 24 * HOUR_MS][-p["limit"]:]
            out = []
            for t in snaps:
                x = d.loc[t - HOUR_MS] if t - HOUR_MS in d.index else d.loc[t]
                if path.endswith("openInterestHist"):
                    out.append({"symbol": p["symbol"], "sumOpenInterest": str(x.oi),
                                "sumOpenInterestValue": str(x.oi * x.close), "timestamp": int(t)})
                elif path.endswith("topLongShortPositionRatio"):
                    out.append({"symbol": p["symbol"], "longShortRatio": str(x.top_ls), "timestamp": int(t)})
                else:
                    out.append({"buySellRatio": str(x.taker), "timestamp": int(t)})
            return out
        raise AssertionError(f"未模拟的接口 {path}")


def fake_coingecko(top_n, api_key=None):
    rows = [
        ("bitcoin", "btc", 60000), ("ethereum", "eth", 2500), ("tether", "usdt", 1.0),
        ("usd-coin", "usdc", 1.0), ("solana", "sol", 100), ("dogecoin", "doge", 0.2),
        ("wrapped-bitcoin", "wbtc", 60000), ("pepe", "pepe", 0.00001),
        ("arbitrum", "arb", 0.3), ("optimism", "op", 0.13),
    ]
    return [{"id": i, "symbol": s, "name": i, "current_price": p,
             "market_cap_rank": k + 1, "market_cap": 1e9 / (k + 1)} for k, (i, s, p) in enumerate(rows)][:top_n]
