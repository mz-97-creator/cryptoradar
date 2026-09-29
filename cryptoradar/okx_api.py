"""OKX 公开行情接口(无需 API Key)。

GitHub Actions 的服务器在美国,币安会返回 451,所以云端版改用 OKX:
- K 线:/market/candles(1H,可向前翻页)
- 持仓量 + 成交额:/rubik/stat/contracts/open-interest-volume(按币种汇总,USD 计,约 30 天小时数据)
- 多空账户比:/rubik/stat/contracts/long-short-account-ratio
- 主动买卖量:/rubik/stat/taker-volume
- 资金费率:/public/funding-rate(当期)、/public/funding-rate-history
"""
from __future__ import annotations

import logging
import time
from collections import deque

import requests

log = logging.getLogger(__name__)

BASE = "https://www.okx.com"
HOUR_MS = 3_600_000


class OKXBlockedError(RuntimeError):
    pass


class _Limiter:
    def __init__(self, max_calls: int, window: float):
        self.max_calls, self.window = max_calls, window
        self.calls: deque[float] = deque()

    def wait(self) -> None:
        now = time.monotonic()
        while self.calls and now - self.calls[0] > self.window:
            self.calls.popleft()
        if len(self.calls) >= self.max_calls:
            time.sleep(max(self.window - (now - self.calls[0]) + 0.05, 0))
        self.calls.append(time.monotonic())


class OKX:
    def __init__(self, timeout: float = 15, max_retries: int = 4):
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": "CryptoRadar/1.0"})
        self.timeout = timeout
        self.max_retries = max_retries
        # 官方限速(按 IP):rubik 5 次/2 秒,funding-history 10 次/2 秒,candles 40 次/2 秒
        self.lim = {
            "rubik": _Limiter(4, 2.0),
            "funding_hist": _Limiter(8, 2.0),
            "default": _Limiter(15, 2.0),
        }

    def _get(self, path: str, params: dict | None = None, group: str = "default"):
        self.lim[group].wait()
        last_err = None
        for attempt in range(self.max_retries):
            try:
                r = self.s.get(BASE + path, params=params, timeout=self.timeout)
            except requests.RequestException as e:
                last_err = e
                time.sleep(2 ** attempt)
                continue
            if r.status_code in (403, 451):
                raise OKXBlockedError(f"OKX 拒绝访问 HTTP {r.status_code}")
            if r.status_code == 429:
                time.sleep(2 + 2 ** attempt)
                continue
            if r.status_code >= 500:
                last_err = RuntimeError(f"HTTP {r.status_code}")
                time.sleep(2 ** attempt)
                continue
            data = r.json()
            code = str(data.get("code", "0"))
            if code == "50011":  # 限频
                time.sleep(2 + 2 ** attempt)
                continue
            if code != "0":
                log.debug("OKX %s %s -> %s %s", path, params, code, data.get("msg"))
                return []
            return data.get("data", [])
        raise RuntimeError(f"OKX 请求失败 {path} {params}: {last_err}")

    # ------------------------------------------------------------ bulk
    def usdt_swaps(self) -> dict[str, str]:
        """{币种: instId},仅 USDT 本位线性永续。"""
        out = {}
        for i in self._get("/api/v5/public/instruments", {"instType": "SWAP"}):
            if i.get("settleCcy") == "USDT" and i.get("ctType") == "linear" and i.get("state") == "live":
                out[i["ctValCcy"].upper()] = i["instId"]
        return out

    def tickers(self) -> dict[str, dict]:
        return {t["instId"]: t for t in self._get("/api/v5/market/tickers", {"instType": "SWAP"})}

    # ------------------------------------------------------- per symbol
    def candles(self, inst_id: str, hours: int = 900) -> list[list]:
        """1H K 线,旧→新。含当前未收盘的一根(confirm=0)。"""
        rows: list[list] = []
        after = None
        while len(rows) < hours:
            params = {"instId": inst_id, "bar": "1H", "limit": 300}
            if after:
                params["after"] = after
            batch = self._get("/api/v5/market/candles", params)
            if not batch:
                break
            rows.extend(batch)
            after = batch[-1][0]
            if len(batch) < 300:
                break
        rows.sort(key=lambda r: int(r[0]))
        return rows

    def oi_volume(self, ccy: str) -> list[list]:
        return self._get("/api/v5/rubik/stat/contracts/open-interest-volume",
                         {"ccy": ccy, "period": "1H"}, "rubik")

    def long_short(self, ccy: str) -> list[list]:
        return self._get("/api/v5/rubik/stat/contracts/long-short-account-ratio",
                         {"ccy": ccy, "period": "1H"}, "rubik")

    def taker(self, ccy: str) -> list[list]:
        return self._get("/api/v5/rubik/stat/taker-volume",
                         {"ccy": ccy, "instType": "CONTRACTS", "period": "1H"}, "rubik")

    def funding_now(self, inst_id: str) -> dict:
        d = self._get("/api/v5/public/funding-rate", {"instId": inst_id})
        return d[0] if d else {}

    def funding_history(self, inst_id: str) -> list[dict]:
        return self._get("/api/v5/public/funding-rate-history",
                         {"instId": inst_id, "limit": 100}, "funding_hist")
