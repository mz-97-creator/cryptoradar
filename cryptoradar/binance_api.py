"""币安 U 本位永续合约公开接口(无需 API Key)。

只用只读的公开行情接口,不涉及下单,也不需要账户权限。
"""
from __future__ import annotations

import logging
import time
from collections import deque

import requests

log = logging.getLogger(__name__)

FAPI = "https://fapi.binance.com"
HOUR_MS = 3_600_000


class BinanceBlockedError(RuntimeError):
    """币安对当前 IP 所在地区返回 451/403。"""


class _SlidingWindowLimiter:
    """滑动窗口限速:window 秒内最多 max_calls 次。"""

    def __init__(self, max_calls: int, window: float):
        self.max_calls = max_calls
        self.window = window
        self.calls: deque[float] = deque()

    def wait(self) -> None:
        now = time.monotonic()
        while self.calls and now - self.calls[0] > self.window:
            self.calls.popleft()
        if len(self.calls) >= self.max_calls:
            sleep_for = self.window - (now - self.calls[0]) + 0.05
            log.info("触发 /futures/data 限速保护,等待 %.0f 秒", sleep_for)
            time.sleep(max(sleep_for, 0))
        self.calls.append(time.monotonic())


class BinanceFutures:
    def __init__(self, timeout: float = 15, max_retries: int = 4, min_interval: float = 0.08):
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": "CryptoRadar/1.0"})
        self.timeout = timeout
        self.max_retries = max_retries
        self.min_interval = min_interval
        self._last_call = 0.0
        # 官方限制:/futures/data/* 每 IP 每 5 分钟 1000 次,这里留余量
        self._data_limiter = _SlidingWindowLimiter(900, 300)

    # ------------------------------------------------------------------ core
    def _get(self, path: str, params: dict | None = None):
        if path.startswith("/futures/data"):
            self._data_limiter.wait()
        gap = time.monotonic() - self._last_call
        if gap < self.min_interval:
            time.sleep(self.min_interval - gap)

        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            self._last_call = time.monotonic()
            try:
                r = self.s.get(FAPI + path, params=params, timeout=self.timeout)
            except requests.RequestException as e:
                last_err = e
                time.sleep(2 ** attempt)
                continue
            if r.status_code in (451, 403):
                raise BinanceBlockedError(
                    f"币安拒绝访问(HTTP {r.status_code}),通常是服务器所在地区受限,"
                    "请换非美国地区的网络/VPS。"
                )
            if r.status_code in (418, 429):
                wait = int(r.headers.get("Retry-After", "60"))
                log.warning("币安限频(HTTP %s),等待 %s 秒", r.status_code, wait)
                time.sleep(wait)
                continue
            if r.status_code == 400:
                log.debug("400 %s %s: %s", path, params, r.text[:200])
                return None
            if r.status_code >= 500:
                last_err = RuntimeError(f"HTTP {r.status_code}")
                time.sleep(2 ** attempt)
                continue
            r.raise_for_status()
            return r.json()
        raise RuntimeError(f"请求失败 {path} {params}: {last_err}")

    # ------------------------------------------------------------ endpoints
    def perp_symbols(self) -> dict[str, str]:
        """返回 {币安 symbol: baseAsset},仅限 USDT 永续且在交易中的合约。"""
        info = self._get("/fapi/v1/exchangeInfo") or {}
        out = {}
        for s in info.get("symbols", []):
            if (
                s.get("contractType") == "PERPETUAL"
                and s.get("quoteAsset") == "USDT"
                and s.get("status") == "TRADING"
            ):
                out[s["symbol"]] = s["baseAsset"]
        return out

    def klines(self, symbol: str, interval: str = "1h", limit: int = 500,
               start_time: int | None = None, end_time: int | None = None):
        params = {"symbol": symbol, "interval": interval, "limit": min(limit, 1500)}
        if start_time is not None:
            params["startTime"] = int(start_time)
        if end_time is not None:
            params["endTime"] = int(end_time)
        return self._get("/fapi/v1/klines", params) or []

    def premium_index_all(self):
        """所有合约的标记价格与当期资金费率(一次请求)。"""
        return self._get("/fapi/v1/premiumIndex") or []

    def funding_info(self):
        """结算周期不是 8 小时的合约列表(fundingIntervalHours)。"""
        return self._get("/fapi/v1/fundingInfo") or []

    def funding_history(self, symbol: str, start_time: int | None = None, limit: int = 1000):
        params = {"symbol": symbol, "limit": limit}
        if start_time is not None:
            params["startTime"] = int(start_time)
        return self._get("/fapi/v1/fundingRate", params) or []

    def open_interest(self, symbol: str):
        return self._get("/fapi/v1/openInterest", {"symbol": symbol})

    def _data_hist(self, path: str, symbol: str, period: str, limit: int,
                   end_time: int | None = None):
        params = {"symbol": symbol, "period": period, "limit": min(limit, 500)}
        if end_time is not None:
            params["endTime"] = int(end_time)
        return self._get(path, params) or []

    def data_hist_paged(self, kind: str, symbol: str, hours: int, period: str = "1h"):
        """分页获取 /futures/data 历史(官方只保留最近 30 天)。"""
        path = {
            "oi": "/futures/data/openInterestHist",
            "top_ls": "/futures/data/topLongShortPositionRatio",
            "taker": "/futures/data/takerlongshortRatio",
        }[kind]
        hours = max(1, min(hours, 30 * 24))
        rows: list[dict] = []
        end_time = None
        remaining = hours
        while remaining > 0:
            batch = self._data_hist(path, symbol, period, min(remaining, 500), end_time)
            if not batch:
                break
            rows = batch + rows
            remaining -= len(batch)
            earliest = min(int(b["timestamp"]) for b in batch)
            end_time = earliest - 1
            if len(batch) < min(remaining + len(batch), 500):
                break
        return rows
