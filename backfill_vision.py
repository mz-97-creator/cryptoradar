"""历史回填(只用 data.binance.vision,不碰 fapi.binance.com)。

backfill.py 的 K 线和资金费率走币安 API,在币安限制的地区(HTTP 451)用不了。
本脚本把三类数据都改成读官方数据站的压缩包,写进同一个 SQLite,后面的 research.py / tune.py 不用改:
- 1h K 线:按月文件;当月、以及月度文件还没发布的上个月,用按日文件补齐
- 资金费率:按月文件(当月最后几天会缺,对应时段资金费率相关规则不触发)
- OI / 大户多空比 / 主动买卖比:按日 metrics 文件(沿用 backfill.py 的解析和续跑记录)

用法:
  python backfill_vision.py --symbols OP,ETH,SOL --start 2024-01-01
  python backfill_vision.py --bases-file syms.txt --start 2024-01-01   逗号分隔的币种简称
"""
from __future__ import annotations

import argparse
import io
import logging
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

from backfill import backfill_metrics, to_ms
from cryptoradar.config import load_config
from cryptoradar.storage import Store
from monitor import setup_logging

log = logging.getLogger("backfill_vision")

BASE = "https://data.binance.vision/data/futures/um"
THOUSAND = {"BONK", "PEPE", "SHIB"}  # 币安上叫 1000XXXUSDT,价格是原价的 1000 倍(收益率不受影响)


def to_symbol(base: str) -> str:
    base = base.upper().removesuffix("USDT")
    return f"1000{base}USDT" if base in THOUSAND else f"{base}USDT"


def fetch_zip_csv(url: str) -> pd.DataFrame | None:
    for attempt in range(4):
        try:
            r = requests.get(url, timeout=40)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                with z.open(z.namelist()[0]) as fh:
                    return pd.read_csv(fh)
        except requests.RequestException:
            if attempt == 3:
                raise
    return None


def months(start: date, end: date):
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        yield y, m
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)


def kline_rows(df: pd.DataFrame) -> list[list]:
    df = df[pd.to_numeric(df.iloc[:, 0], errors="coerce").notna()]  # 新文件带表头行
    return df.iloc[:, :11].astype(float).values.tolist()


def backfill_klines_vision(store: Store, sym: str, start: date, today: date, workers: int) -> int:
    """按月文件取 K 线;当月、以及上个月的月度文件还没发布(404)时,改用按日文件补齐。"""
    def daily_urls(y: int, m: int) -> list[str]:
        d, out = date(y, m, 1), []
        while d.month == m and d < today:
            out.append(f"{BASE}/daily/klines/{sym}/1h/{sym}-1h-{d.isoformat()}.zip")
            d += timedelta(days=1)
        return out

    urls, fallback = [], []
    for y, m in months(start, today):
        if (y, m) == (today.year, today.month):
            fallback.append((y, m))
        else:
            urls.append((f"{BASE}/monthly/klines/{sym}/1h/{sym}-1h-{y}-{m:02d}.zip", (y, m)))
    n = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        dfs = list(ex.map(fetch_zip_csv, [u for u, _ in urls]))
        for df, (_, ym) in zip(dfs, urls):
            if df is None:
                fallback.append(ym)          # 月度文件还没发布:用按日文件
            elif not df.empty:
                n += store.upsert_klines(sym, kline_rows(df))
        daily = [u for ym in fallback for u in daily_urls(*ym)]
        for df in ex.map(fetch_zip_csv, daily):
            if df is not None and not df.empty:
                n += store.upsert_klines(sym, kline_rows(df))
    return n


def backfill_funding_vision(store: Store, sym: str, start: date, today: date, workers: int) -> int:
    urls = [f"{BASE}/monthly/fundingRate/{sym}/{sym}-fundingRate-{y}-{m:02d}.zip"
            for y, m in months(start, today) if (y, m) != (today.year, today.month)]
    n = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for df in ex.map(fetch_zip_csv, urls):
            if df is None or df.empty:
                continue
            df = df[pd.to_numeric(df.iloc[:, 0], errors="coerce").notna()]
            rows = [(int(t), float(r)) for t, r in zip(df.iloc[:, 0], df.iloc[:, 2])]
            store.upsert_funding(sym, rows)
            n += len(rows)
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description="用 data.binance.vision 回填历史数据")
    ap.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    ap.add_argument("--symbols", help="逗号分隔的币种简称,如 OP,ETH,SOL")
    ap.add_argument("--bases-file", help="含逗号分隔币种简称的文本文件")
    ap.add_argument("--start", default="2024-01-01")
    ap.add_argument("--skip-metrics", action="store_true")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg["_base_dir"])
    store = Store(cfg["storage"]["db_path"])

    bases = []
    if args.symbols:
        bases += args.symbols.split(",")
    if args.bases_file:
        bases += Path(args.bases_file).read_text().replace("\n", ",").split(",")
    bases = [b.strip() for b in bases if b.strip()]
    for must in ("BTC", "ETH"):
        if must not in bases:
            bases.append(must)
    symbols = list(dict.fromkeys(to_symbol(b) for b in bases))

    start = datetime.strptime(args.start, "%Y-%m-%d").date()
    today = datetime.now(timezone.utc).date()
    skipped = []
    for i, sym in enumerate(symbols, 1):
        log.info("==== [%d/%d] %s ====", i, len(symbols), sym)
        k = backfill_klines_vision(store, sym, start, today, args.workers)
        if k == 0:
            log.warning("%s 在数据站没有 K 线(币安没上永续?),跳过", sym)
            skipped.append(sym)
            continue
        log.info("K 线 %d 根", k)
        log.info("资金费率 %d 条", backfill_funding_vision(store, sym, start, today, args.workers))
        if not args.skip_metrics:
            first = store.conn.execute("SELECT MIN(ts) FROM hourly WHERE symbol=?", (sym,)).fetchone()[0]
            m_start = max(start, datetime.fromtimestamp(first / 1000, timezone.utc).date())
            backfill_metrics(store, sym, m_start, today - timedelta(days=1), args.workers)
    if skipped:
        log.warning("数据站没有这些合约:%s", ",".join(skipped))
    log.info("回填完成。下一步:python research.py --pool")


if __name__ == "__main__":
    main()
