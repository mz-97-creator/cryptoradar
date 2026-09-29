"""增量采集:K 线、持仓量、大户多空比、主动买卖比、资金费率。

- 价格/实时 OI:每个周期都更新(当前小时的未收盘 K 线也会写入,之后被覆盖)
- 持仓结构(OI/多空比/主动买卖比,1h 粒度):每出现一根新的整点才补一次
- 电脑关机造成的空档:下次启动自动补。但 /futures/data 只保留 30 天,
  超过 30 天的空档无法从实时接口补回,只能用 backfill.py 从 data.binance.vision 回填
"""
from __future__ import annotations

import logging

from .binance_api import HOUR_MS, BinanceFutures
from .storage import Store, now_ms

log = logging.getLogger(__name__)

BOOTSTRAP_HOURS = 30 * 24 + 48


def _hour_floor(ms: int) -> int:
    return ms - ms % HOUR_MS


def update_klines(store: Store, api: BinanceFutures, symbol: str) -> None:
    last = store.last_ts(symbol, "close")
    if last is None:
        start = now_ms() - BOOTSTRAP_HOURS * HOUR_MS
        rows = []
        while True:
            batch = api.klines(symbol, "1h", limit=1500, start_time=start)
            if not batch:
                break
            rows.extend(batch)
            if len(batch) < 1500:
                break
            start = int(batch[-1][0]) + HOUR_MS
        store.upsert_klines(symbol, rows)
        return
    # 从最后一根(可能未收盘)开始重新拉,覆盖更新
    missing = (now_ms() - last) // HOUR_MS + 2
    start = last
    while True:
        batch = api.klines(symbol, "1h", limit=int(min(1500, missing + 2)), start_time=start)
        if not batch:
            break
        store.upsert_klines(symbol, batch)
        if len(batch) < 1500:
            break
        start = int(batch[-1][0]) + HOUR_MS


def update_positioning(store: Store, api: BinanceFutures, symbol: str) -> None:
    """OI / 大户持仓多空比 / 主动买卖比。时间戳 t 的快照记到 K 线 t-1h 上。"""
    current_bar = _hour_floor(now_ms())
    last = store.last_ts(symbol, "oi")
    # 最新可得的快照是整点 current_bar,对应 K 线 current_bar - 1h
    if last is not None and last >= current_bar - HOUR_MS:
        return
    hours = BOOTSTRAP_HOURS if last is None else (current_bar - last) // HOUR_MS + 2
    if last is not None and hours > 30 * 24:
        log.warning("%s 持仓数据空档超过 30 天,实时接口补不全,可运行 backfill.py", symbol)

    oi = api.data_hist_paged("oi", symbol, hours)
    store.upsert_hist_column(symbol, "oi", [(int(r["timestamp"]) - HOUR_MS, float(r["sumOpenInterest"])) for r in oi])
    store.upsert_hist_column(symbol, "oi_value", [(int(r["timestamp"]) - HOUR_MS, float(r["sumOpenInterestValue"])) for r in oi])

    ls = api.data_hist_paged("top_ls", symbol, hours)
    store.upsert_hist_column(symbol, "top_ls", [(int(r["timestamp"]) - HOUR_MS, float(r["longShortRatio"])) for r in ls])

    tk = api.data_hist_paged("taker", symbol, hours)
    store.upsert_hist_column(symbol, "taker_ratio", [(int(r["timestamp"]) - HOUR_MS, float(r["buySellRatio"])) for r in tk])


def update_funding(store: Store, api: BinanceFutures, symbol: str) -> None:
    last = store.last_funding_ts(symbol)
    start = now_ms() - 90 * 24 * HOUR_MS if last is None else last + 1
    while True:
        batch = api.funding_history(symbol, start_time=start, limit=1000)
        if not batch:
            break
        store.upsert_funding(symbol, [(int(b["fundingTime"]), float(b["fundingRate"])) for b in batch])
        if len(batch) < 1000:
            break
        start = int(batch[-1]["fundingTime"]) + 1


def update_live(store: Store, api: BinanceFutures, symbols: list[str],
                interval_map: dict[str, float]) -> None:
    wanted = set(symbols)
    for p in api.premium_index_all():
        s = p.get("symbol")
        if s not in wanted:
            continue
        store.upsert_live(
            s,
            mark_price=float(p["markPrice"]),
            funding_rate=float(p["lastFundingRate"]),
            next_funding_time=int(p.get("nextFundingTime") or 0),
            funding_interval_h=float(interval_map.get(s, 8)),
        )
    store.commit()


def update_live_oi(store: Store, api: BinanceFutures, symbol: str) -> None:
    r = api.open_interest(symbol)
    if r and "openInterest" in r:
        store.upsert_live(symbol, oi_now=float(r["openInterest"]))


def collect_symbol(store: Store, api: BinanceFutures, symbol: str) -> None:
    update_klines(store, api, symbol)
    update_positioning(store, api, symbol)
    update_funding(store, api, symbol)
    update_live_oi(store, api, symbol)
    store.commit()
