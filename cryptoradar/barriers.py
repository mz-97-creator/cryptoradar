"""Long-only, cost-aware research labels; no model or trading instruction.

Features on bar T become available at T+1h. Entry is the NEXT bar open.
Success means the upper barrier is reached before the lower barrier or timeout.
If both barriers touch within one bar, the path is unknown and is excluded.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

HOUR = 3_600_000


@dataclass(frozen=True)
class BarrierSpec:
    horizon_h: int = 168
    take_profit: float = 0.20
    stop_loss: float = 0.10
    entry_cost: float = 0.001
    exit_cost: float = 0.001

    def __post_init__(self):
        if self.horizon_h < 1 or self.take_profit <= 0 or not 0 < self.stop_loss < 1:
            raise ValueError("Invalid horizon or barrier")
        if not 0 <= self.entry_cost < 1 or not 0 <= self.exit_cost < 1:
            raise ValueError("Invalid cost")


def label_bars(bars: pd.DataFrame, spec: BarrierSpec = BarrierSpec(), decision_stride_h: int = 1,
               startup_return: float = 0.05) -> pd.DataFrame:
    required = {"open", "high", "low", "close"}
    if not required <= set(bars):
        raise ValueError("OHLC columns are required")
    if not bars.index.is_monotonic_increasing or not bars.index.is_unique:
        raise ValueError("Bars must have sorted, unique timestamps")
    if decision_stride_h < 1 or startup_return <= 0:
        raise ValueError("Invalid decision stride or startup return")
    ts = bars.index.to_numpy(dtype=np.int64)
    ohlc = bars[["open", "high", "low", "close"]].to_numpy(float)
    rows = []
    for i, feature_ts in enumerate(ts):
        entry_ts = int(feature_ts + HOUR)
        if (entry_ts // HOUR) % decision_stride_h:
            continue
        r = {"feature_ts": int(feature_ts), "entry_ts": entry_ts,
             "label_end_ts": None, "outcome": "censored", "target": np.nan,
             "entry_price": np.nan, "exit_price": np.nan, "net_return": np.nan,
             "startup_earliest_ts": None, "startup_latest_ts": None}
        if i + 1 >= len(ts) or ts[i + 1] != entry_ts:
            rows.append(r)
            continue
        entry = ohlc[i + 1, 0]
        if not np.isfinite(entry) or entry <= 0:
            rows.append(r)
            continue
        r["entry_price"] = entry
        upper = entry * (1 + spec.entry_cost) * (1 + spec.take_profit) / (1 - spec.exit_cost)
        lower = entry * (1 + spec.entry_cost) * (1 - spec.stop_loss) / (1 - spec.exit_cost)
        for k in range(spec.horizon_h):
            j = i + 1 + k
            if j >= len(ts) or ts[j] != entry_ts + k * HOUR:
                break
            op, hi, lo, cl = ohlc[j]
            if not np.isfinite(ohlc[j]).all() or lo <= 0 or not lo <= min(op, cl) <= max(op, cl) <= hi:
                break
            if r["startup_earliest_ts"] is None and hi >= entry * (1 + startup_return):
                r["startup_earliest_ts"] = int(ts[j])
                r["startup_latest_ts"] = int(ts[j] if op >= entry * (1 + startup_return) else ts[j] + HOUR)
            outcome, exit_price = None, None
            # Opening gaps have known ordering and may execute worse than stop.
            if op <= lower:
                outcome, exit_price = "stop", op
            elif op >= upper:
                outcome, exit_price = "profit", op
            elif hi >= upper and lo <= lower:
                r.update(outcome="ambiguous", label_end_ts=int(ts[j] + HOUR))
                break
            elif lo <= lower:
                outcome, exit_price = "stop", lower
            elif hi >= upper:
                outcome, exit_price = "profit", upper
            elif k == spec.horizon_h - 1:
                outcome, exit_price = "timeout", cl
            if outcome:
                r.update(outcome=outcome, target=float(outcome == "profit"),
                         label_end_ts=int(ts[j] + HOUR), exit_price=float(exit_price),
                         net_return=float(exit_price * (1 - spec.exit_cost) / (entry * (1 + spec.entry_cost)) - 1))
                break
        rows.append(r)
    return pd.DataFrame(rows)


def holdout_masks(labels: pd.DataFrame, test_start: int, embargo_h: int = 0):
    """Purge training labels reaching into the fixed final test interval.

    test_start must be chosen before inspecting test outcomes. Only fully
    resolved, unambiguous labels are eligible. No tuning on the final holdout.
    """
    eligible = labels["target"].notna()
    train = eligible & (labels["entry_ts"] < test_start) & (labels["label_end_ts"] < test_start - embargo_h * HOUR)
    test = eligible & (labels["entry_ts"] >= test_start)
    return train, test
