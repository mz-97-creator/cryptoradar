"""Public OKX collector for isolated leading-indicator research. No notifications."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import re

import pandas as pd

from cryptoradar.events import append_events
from cryptoradar.okx_api import OKX, HOUR_MS

log = logging.getLogger("leading.collect")


def history(api: OKX, inst: str, start: int, end: int) -> pd.DataFrame:
    """Completed 1H candles with exact timestamps, stop on non-progress."""
    after, rows = end, {}
    while True:
        batch = api._get("/api/v5/market/history-candles", {"instId": inst, "bar": "1H", "limit": 100, "after": after})
        if not batch:
            break
        oldest = min(int(r[0]) for r in batch)
        if oldest >= after:
            raise RuntimeError(f"Non-progressing candle pagination: {inst}")
        for r in batch:
            t = int(r[0])
            if start <= t and t + HOUR_MS <= end and len(r) > 8 and r[8] == "1":
                rows[t] = [t, *map(float, r[1:5]), float(r[7])]
        if oldest <= start:
            break
        after = oldest
    return pd.DataFrame(sorted(rows.values()), columns=["ts", "open", "high", "low", "close", "quote_volume"])


def series_rows(raw: list, names: list[str], offset: int = 0) -> pd.DataFrame:
    rows = [[int(r[0]) + offset, *map(float, r[1:len(names)])] for r in raw]
    return pd.DataFrame(rows, columns=names).drop_duplicates("ts").sort_values("ts")


def collect_coin(symbol: str, start: int, end: int, folder: Path) -> dict:
    api = OKX(timeout=25, max_retries=3)
    result = {"symbol": symbol, "errors": {}, "rows": {}}
    for tag, inst in (("spot", f"{symbol}-USDT"), ("perp", f"{symbol}-USDT-SWAP")):
        path = folder / f"{symbol}_{tag}.csv.gz"
        try:
            if path.exists():
                prior = pd.read_csv(path)
                # Keep downloaded historical bars, refresh the latest two days.
                lo = max(start, int(prior.ts.max()) - 48 * HOUR_MS) if len(prior) else start
                df = pd.concat([prior, history(api, inst, lo, end)], ignore_index=True).drop_duplicates("ts", keep="last").sort_values("ts")
            else:
                df = history(api, inst, start, end)
            df.to_csv(path, index=False)
            result["rows"][tag] = len(df)
            if not len(df):
                result["errors"][tag] = "No historical candles returned"
            if len(df):
                result[tag + "_start"] = int(df.ts.min())
                result[tag + "_end"] = int(df.ts.max())
        except Exception as e:
            result["errors"][tag] = str(e)
    feeds = {
        "spot_flow": ("/api/v5/rubik/stat/taker-volume", {"ccy": symbol, "instType": "SPOT", "period": "1H"}, ["ts", "sell", "buy"], 0),
        "perp_flow": ("/api/v5/rubik/stat/taker-volume", {"ccy": symbol, "instType": "CONTRACTS", "period": "1H"}, ["ts", "sell", "buy"], 0),
        "oi": ("/api/v5/rubik/stat/contracts/open-interest-volume", {"ccy": symbol, "period": "1H"}, ["ts", "oi_usd", "contract_volume"], -HOUR_MS),
    }
    for tag, (endpoint, params, names, offset) in feeds.items():
        try:
            df = series_rows(api._get(endpoint, params, "rubik"), names, offset)
            df = df[df.ts + HOUR_MS <= end]
            path = folder / f"{symbol}_{tag}.csv.gz"
            if path.exists():
                df = pd.concat([pd.read_csv(path), df], ignore_index=True).drop_duplicates("ts", keep="first").sort_values("ts")
            df.to_csv(path, index=False)
            result["rows"][tag] = len(df)
            if not len(df):
                result["errors"][tag] = "Feed unavailable; missing is not zero"
        except Exception as e:
            result["errors"][tag] = str(e)
    # Funding settlement has its own actual time, not an hourly future value.
    try:
        df = pd.DataFrame([{ "ts": int(r["fundingTime"]), "funding": float(r.get("realizedRate") or r["fundingRate"])}
                           for r in api.funding_history(f"{symbol}-USDT-SWAP")])
        path = folder / f"{symbol}_funding.csv.gz"
        if path.exists():
            df = pd.concat([pd.read_csv(path), df]).drop_duplicates("ts", keep="first")
        df.sort_values("ts").to_csv(path, index=False)
        result["rows"]["funding"] = len(df)
    except Exception as e:
        result["errors"]["funding"] = str(e)
    log.info("%s: %s%s", symbol, result["rows"], f" errors={result['errors']}" if result["errors"] else "")
    return result


def announcements(api: OKX, symbols: list[str]) -> list[dict]:
    events = []
    pages = 1
    page = 1
    while page <= min(pages, 20):
        raw = api._get("/api/v5/support/announcements", {"page": page})
        if not raw:
            break
        payload = raw[0]
        pages = int(payload.get("totalPage", 1))
        for a in payload.get("details", []):
            typ = a.get("annType", "")
            kind = "listing" if "new-listings" in typ else "delisting" if "delistings" in typ else "announcement"
            title = a.get("title", "")
            tokens = [s for s in symbols if re.search(r"(?<![A-Z0-9])" + re.escape(s) + r"(?![A-Z0-9])", title.upper())]
            events.append({"source": "OKX announcements", "source_url": a["url"], "kind": kind,
                           "symbols": tokens, "title": title, "published_ts": int(a["pTime"]),
                           "scheduled_ts": int(a.get("businessPTime") or a["pTime"])})
        page += 1
    return events


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default="leading_plan.json")
    ap.add_argument("--out", default="codex-data/leading")
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    plan = json.loads(Path(args.plan).read_text())
    folder = Path(args.out)
    folder.mkdir(parents=True, exist_ok=True)
    start = int(pd.to_datetime(plan["history_start_utc"], utc=True).timestamp() * 1000)
    api = OKX(timeout=25, max_retries=3)
    clock = api._get("/api/v5/public/time")
    end = int(clock[0]["ts"])
    # Persist the actual run cut-off before fetching results or reading labels.
    (folder / "run_cutoff.json").write_text(json.dumps({"asof_ts": end, "plan": plan}, indent=2), encoding="utf-8")
    results = []
    with ThreadPoolExecutor(max_workers=max(1, min(args.workers, 4))) as pool:
        futures = [pool.submit(collect_coin, s, start, end, folder) for s in plan["symbols"]]
        for future in as_completed(futures):
            results.append(future.result())
    try:
        raw = announcements(api, plan["symbols"])
        quotes = {r["instId"].removesuffix("-USDT"): r for r in api._get("/api/v5/market/tickers", {"instType": "SPOT"})
                  if r["instId"].endswith("-USDT")}
        for event in raw:
            event["observed_quotes"] = {s: {"price": float(quotes[s]["last"]), "quote_ts": int(quotes[s]["ts"]),
                                            "prior_return_24h": float(quotes[s]["last"]) / float(quotes[s]["open24h"]) - 1
                                            if float(quotes[s].get("open24h") or 0) > 0 else None}
                                        for s in event["symbols"] if s in quotes}
        events = append_events(folder / "events.jsonl", raw, int(datetime.now(timezone.utc).timestamp() * 1000))
        event_status = {"observed_revisions": len(events), "matched_revisions": sum(bool(e["symbols"]) for e in events),
                        "historical_use": "excluded before first_seen_ts"}
    except Exception as e:
        event_status = {"error": str(e)}
    manifest = {"asof_ts": end, "coins": sorted(results, key=lambda x: x["symbol"]), "announcements": event_status,
                "unavailable": {"buyback_execution": "No verified chain RPC/contracts configured; estimates are not executions",
                                "protocol_revenue": "Protocol data endpoints not in current environment allowlist",
                                "unlock": "No verified timed unlock source configured"}}
    (folder / "source_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
