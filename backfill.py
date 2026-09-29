"""历史回填:让评分模型不必从零"逐渐积累",而是马上有 4 年多的样本。

数据来源:
- 1h K 线、资金费率:币安 API 分页,可取到合约上线以来的全部历史
- OI / 大户持仓多空比 / 主动买卖比:币安官方数据站 data.binance.vision 的每日 metrics 文件
  (5 分钟粒度,OPUSDT 从 2022-06-01 起)。实时接口只保留 30 天,这是唯一的长期来源

用法:
  python backfill.py                       回填 config.yaml 里 research.symbols(默认 OP/BTC/ETH)
  python backfill.py --symbols SOLUSDT,ARBUSDT
  python backfill.py --universe            回填整个监控名单(约 150 个合约,耗时较长,可中断续跑)
"""
from __future__ import annotations

import argparse
import io
import logging
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

from cryptoradar.binance_api import HOUR_MS, BinanceFutures
from cryptoradar.config import load_config
from cryptoradar.storage import Store
from monitor import setup_logging

log = logging.getLogger("backfill")

VISION = "https://data.binance.vision/data/futures/um/daily/metrics/{s}/{s}-metrics-{d}.zip"
EPOCH = pd.Timestamp("1970-01-01", tz="UTC")


def to_ms(d: str | date) -> int:
    if isinstance(d, str):
        d = datetime.strptime(d, "%Y-%m-%d").date()
    return int(datetime(d.year, d.month, d.day, tzinfo=timezone.utc).timestamp() * 1000)


def backfill_klines(store: Store, api: BinanceFutures, sym: str, start_ms: int) -> int:
    n, start = 0, start_ms
    while True:
        batch = api.klines(sym, "1h", limit=1500, start_time=start)
        if not batch:
            break
        n += store.upsert_klines(sym, batch)
        if len(batch) < 1500:
            break
        start = int(batch[-1][0]) + HOUR_MS
    return n


def backfill_funding(store: Store, api: BinanceFutures, sym: str, start_ms: int) -> int:
    n, start = 0, start_ms
    while True:
        batch = api.funding_history(sym, start_time=start, limit=1000)
        if not batch:
            break
        store.upsert_funding(sym, [(int(b["fundingTime"]), float(b["fundingRate"])) for b in batch])
        n += len(batch)
        if len(batch) < 1000:
            break
        start = int(batch[-1]["fundingTime"]) + 1
    return n


def parse_metrics(zbytes: bytes) -> pd.DataFrame:
    """5 分钟 metrics → 小时粒度,按 storage.py 的约定对齐到 K 线开盘时间。"""
    with zipfile.ZipFile(io.BytesIO(zbytes)) as z:
        with z.open(z.namelist()[0]) as fh:
            df = pd.read_csv(fh)
    if df.empty or "create_time" not in df:
        return pd.DataFrame()
    t = pd.to_datetime(df["create_time"], utc=True)
    ms = (t - EPOCH) // pd.Timedelta(milliseconds=1)
    df["bar"] = ((ms - 1) // HOUR_MS) * HOUR_MS  # (T, T+1h] 内的快照归到 K 线 T
    df = df.sort_values("bar")
    agg = {}
    if "sum_open_interest" in df:
        agg["oi"] = ("sum_open_interest", "last")
    if "sum_open_interest_value" in df:
        agg["oi_value"] = ("sum_open_interest_value", "last")
    if "sum_toptrader_long_short_ratio" in df:
        agg["top_ls"] = ("sum_toptrader_long_short_ratio", "last")
    if "sum_taker_long_short_vol_ratio" in df:
        agg["taker_ratio"] = ("sum_taker_long_short_vol_ratio", "mean")
    return df.groupby("bar").agg(**agg)


def _download(sym: str, day: str) -> tuple[str, bytes | None]:
    url = VISION.format(s=sym, d=day)
    for attempt in range(4):
        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 404:
                return day, None
            r.raise_for_status()
            return day, r.content
        except requests.RequestException:
            if attempt == 3:
                raise
    return day, None


def backfill_metrics(store: Store, sym: str, start: date, end: date, workers: int = 8) -> None:
    done = {d for (d,) in store.conn.execute(
        "SELECT day FROM backfill_done WHERE symbol=?", (sym,)).fetchall()}
    days = []
    d = start
    while d <= end:
        s = d.isoformat()
        if s not in done:
            days.append(s)
        d += timedelta(days=1)
    if not days:
        log.info("%s metrics 已是最新", sym)
        return
    log.info("%s 需下载 %d 天的 metrics 文件", sym, len(days))
    ok = missing = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_download, sym, day) for day in days]
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                day, content = fut.result()
            except Exception as e:
                log.warning("%s 下载失败:%s(下次运行会重试)", sym, e)
                continue
            if content is None:
                missing += 1
                store.conn.execute("INSERT OR REPLACE INTO backfill_done VALUES(?,?,?)", (sym, day, "missing"))
                continue
            h = parse_metrics(content)
            for col in ("oi", "oi_value", "top_ls", "taker_ratio"):
                if col in h:
                    s = h[col].dropna()
                    store.upsert_hist_column(sym, col, list(zip(s.index.astype(int), s.astype(float))))
            store.conn.execute("INSERT OR REPLACE INTO backfill_done VALUES(?,?,?)", (sym, day, "ok"))
            ok += 1
            if i % 100 == 0:
                store.commit()
                log.info("%s 进度 %d/%d", sym, i, len(days))
    store.commit()
    log.info("%s metrics 完成:成功 %d 天,无数据 %d 天(合约上线前或当日未发布)", sym, ok, missing)


def main() -> None:
    ap = argparse.ArgumentParser(description="回填历史数据")
    ap.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    ap.add_argument("--symbols", help="逗号分隔,如 OPUSDT,BTCUSDT")
    ap.add_argument("--universe", action="store_true", help="回填整个监控名单")
    ap.add_argument("--start", help="起始日期 YYYY-MM-DD")
    ap.add_argument("--skip-metrics", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg["_base_dir"])
    store = Store(cfg["storage"]["db_path"])
    api = BinanceFutures()

    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    elif args.universe:
        symbols = [u["symbol"] for u in store.load_universe()]
        if not symbols:
            raise SystemExit("监控名单为空,请先运行一次 python monitor.py --once")
    else:
        symbols = cfg["research"]["symbols"]
    for must in ("BTCUSDT", "ETHUSDT"):
        if must not in symbols:
            symbols.append(must)

    start = args.start or cfg["research"]["start"]
    start_ms = to_ms(start)
    end = datetime.now(timezone.utc).date() - timedelta(days=1)

    for sym in symbols:
        log.info("==== %s ====", sym)
        log.info("K 线 %d 根", backfill_klines(store, api, sym, start_ms))
        log.info("资金费率 %d 条", backfill_funding(store, api, sym, start_ms))
        if not args.skip_metrics:
            backfill_metrics(store, sym, datetime.strptime(start, "%Y-%m-%d").date(), end, args.workers)
    log.info("回填完成。下一步:python research.py")


if __name__ == "__main__":
    main()
