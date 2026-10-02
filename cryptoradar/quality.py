"""Data freshness and append-only research snapshots. No network or notification."""
from __future__ import annotations

import numpy as np
import pandas as pd

HOUR = 3_600_000


def frame_quality(frame: pd.DataFrame, now: int, max_age_h: float = 2) -> dict:
    if frame.empty or "close" not in frame:
        return {"usable": False, "reason": "empty", "age_h": None}
    last = int(frame.index[-1])
    price = frame["close"].iloc[-1]
    age = (now - last) / HOUR
    usable = bool(np.isfinite(price) and price > 0 and 0 <= age <= max_age_h)
    return {"usable": usable, "reason": "ok" if usable else "stale_or_invalid",
            "age_h": age, "last_bar_open": last}


def append_research(prev: pd.DataFrame | None, frames: dict, observed_at: int) -> pd.DataFrame:
    """Save only closed bars, all feature columns, first observation wins.

    An observed_at column records when these features were first available to
    this scanner. Late backfilled rows must not be treated as past live signals.
    There is deliberately no retention cutoff. The caller must persist the file.
    """
    parts = [prev] if prev is not None and len(prev) else []
    for symbol, frame in frames.items():
        closed = frame.loc[frame.index + HOUR <= observed_at].copy()
        if closed.empty:
            continue
        closed.index.name = "ts"
        closed = closed.reset_index()
        closed["symbol"], closed["observed_at"] = symbol, observed_at
        closed["source_exchange"] = "OKX"
        closed["feature_schema"] = 1
        parts.append(closed)
    if not parts:
        return pd.DataFrame(columns=["symbol", "ts", "observed_at", "source_exchange", "feature_schema"])
    return pd.concat(parts, ignore_index=True).drop_duplicates(["symbol", "ts"], keep="first")
