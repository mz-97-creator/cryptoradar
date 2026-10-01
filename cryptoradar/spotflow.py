"""现货资金流特征:币安现货 1 小时 K 线(data.binance.vision)里带"主动买入成交额",
用它和合约侧(主动买卖比、持仓、资金费率)对照,检验"是现货真金白银在买,还是合约杠杆在推"。

存进和其他历史数据同一个 SQLite(表 spot_hourly)。特征(都只用当时已知的信息,z 分数和 features.py 一样是和自己过去 30 天比):
  spot_buy_z      现货主动买入占比(24h 成交额加权)的 z 分数:正 = 现货买盘比平时强
  spot_buy_dz12   spot_buy_z 近 12 小时的变化:正 = 现货买盘在增强,负 = 在减弱
  lev_share_z     合约成交额 / 现货成交额(取对数)的 z 分数:高 = 这个币更多是杠杆合约在交易
  basis_z         合约相对现货的溢价(24h 均值)的 z 分数:高 = 合约比现货贵,杠杆多头更急
"""
from __future__ import annotations

import io
import logging
import sqlite3
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests

from .features import rolling_z

log = logging.getLogger("spotflow")
BASE = "https://data.binance.vision/data/spot"
HOUR = 3_600_000
SPOT_COLS = ["spot_buy_z", "spot_buy_dz12", "lev_share_z", "basis_z"]
SCHEMA = """CREATE TABLE IF NOT EXISTS spot_hourly(
    symbol TEXT NOT NULL, ts INTEGER NOT NULL, close REAL, quote_volume REAL, taker_buy_quote REAL,
    PRIMARY KEY(symbol, ts))"""


def spot_symbol(perp_symbol: str) -> str:
    s = perp_symbol.upper().removesuffix("USDT")
    return (s[4:] if s.startswith("1000") else s) + "USDT"


def _fetch(url: str) -> pd.DataFrame | None:
    for i in range(4):
        try:
            r = requests.get(url, timeout=40)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                with z.open(z.namelist()[0]) as fh:
                    df = pd.read_csv(fh, header=None)
            df = df[pd.to_numeric(df.iloc[:, 0], errors="coerce").notna()].astype(float)
            ts = df.iloc[:, 0]
            ts = np.where(ts > 1e14, ts // 1000, ts)             # 2025 年起现货文件的时间戳是微秒
            return pd.DataFrame({"ts": ts.astype("int64"), "close": df.iloc[:, 4].to_numpy(),
                                 "quote_volume": df.iloc[:, 7].to_numpy(), "taker_buy_quote": df.iloc[:, 10].to_numpy()})
        except requests.RequestException:
            if i == 3:
                raise
    return None


def _months(start: date, end: date):
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        yield y, m
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)


def backfill(conn: sqlite3.Connection, perp_symbols: list[str], start: date, workers: int = 16) -> dict[str, int]:
    conn.execute(SCHEMA)
    today = datetime.now(timezone.utc).date()
    out = {}
    for k, ps in enumerate(perp_symbols, 1):
        sym = spot_symbol(ps)
        urls, daily = [], []
        for y, m in _months(start, today):
            if (y, m) == (today.year, today.month):
                daily.append((y, m))
            else:
                urls.append((f"{BASE}/monthly/klines/{sym}/1h/{sym}-1h-{y}-{m:02d}.zip", (y, m)))
        n = 0
        with ThreadPoolExecutor(max_workers=workers) as ex:
            res = list(ex.map(_fetch, [u for u, _ in urls]))
            for df, (_, ym) in zip(res, urls):
                if df is None:
                    daily.append(ym)
                else:
                    n += _store(conn, sym, df)
            dl = []
            for y, m in daily:
                d = date(y, m, 1)
                while d.month == m and d < today:
                    dl.append(f"{BASE}/daily/klines/{sym}/1h/{sym}-1h-{d.isoformat()}.zip")
                    d += timedelta(days=1)
            for df in ex.map(_fetch, dl):
                if df is not None:
                    n += _store(conn, sym, df)
        conn.commit()
        out[ps] = n
        log.info("[%d/%d] %s 现货 %d 根", k, len(perp_symbols), sym, n)
    return out


def _store(conn: sqlite3.Connection, sym: str, df: pd.DataFrame) -> int:
    conn.executemany("INSERT OR REPLACE INTO spot_hourly VALUES(?,?,?,?,?)",
                     [(sym, int(r.ts), r.close, r.quote_volume, r.taker_buy_quote) for r in df.itertuples()])
    return len(df)


def features(conn: sqlite3.Connection, perp_symbol: str, perp: pd.DataFrame) -> pd.DataFrame:
    """perp: 该币合约的小时表(storage.load_hourly 的输出,index=ts,含 close/quote_volume)。
    返回与 perp.index 对齐的 SPOT_COLS;没有现货数据则返回空表。"""
    sym = spot_symbol(perp_symbol)
    s = pd.read_sql_query("SELECT ts, close, quote_volume, taker_buy_quote FROM spot_hourly WHERE symbol=? ORDER BY ts",
                          conn, params=[sym])
    if s.empty or perp.empty:
        return pd.DataFrame(index=perp.index, columns=SPOT_COLS, dtype=float)
    s = s.drop_duplicates("ts").set_index("ts").reindex(perp.index)
    qv24 = s["quote_volume"].rolling(24, min_periods=20).sum()
    buy24 = s["taker_buy_quote"].rolling(24, min_periods=20).sum()
    ratio = (buy24 / qv24.where(qv24 > 0))
    out = pd.DataFrame(index=perp.index)
    out["spot_buy_z"] = rolling_z(ratio)
    out["spot_buy_dz12"] = out["spot_buy_z"].diff(12)
    pq24 = perp["quote_volume"].rolling(24, min_periods=20).sum()
    out["lev_share_z"] = rolling_z(np.log((pq24 / qv24).where((pq24 > 0) & (qv24 > 0))))
    basis = (perp["close"] / s["close"] - 1).rolling(24, min_periods=20).mean()
    out["basis_z"] = rolling_z(basis)
    return out


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="回填币安现货 1 小时 K 线(主动买入成交额)到 SQLite")
    ap.add_argument("--db", default="data/cryptoradar.db")
    ap.add_argument("--start", default="2024-01-01")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    conn = sqlite3.connect(args.db)
    syms = sorted(r[0] for r in conn.execute("SELECT DISTINCT symbol FROM hourly"))
    res = backfill(conn, syms, datetime.strptime(args.start, "%Y-%m-%d").date())
    print("有现货数据的币:", sum(1 for v in res.values() if v > 0), "/", len(res))


if __name__ == "__main__":
    main()
