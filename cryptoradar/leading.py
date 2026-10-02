"""Past-only leading features. NaN means unavailable, never no-buying."""
from __future__ import annotations

import numpy as np
import pandas as pd

HOUR = 3_600_000
STRENGTH = ["resid_6h", "resid_24h", "resid_7d", "down_resilience_7d", "recovery_change_6h",
            "range_compression", "volume_accel_6h", "rv_7d", "basis_24h", "perp_share_24h",
            "btc_ret_7d", "breadth", "rank_resid_24h", "ret_24h"]
SPOT = ["spot_buy_share_24h", "spot_buy_accel_6h", "spot_buy_consistency_6h", "oi_change_24h", "funding"]


def grid(bars: pd.DataFrame) -> pd.DataFrame:
    if bars.empty:
        return bars
    return bars.sort_index().reindex(np.arange(int(bars.index.min()), int(bars.index.max()) + HOUR, HOUR))


def features(spot: pd.DataFrame, perp: pd.DataFrame, btc: pd.DataFrame,
             spot_flow: pd.DataFrame | None = None, oi: pd.DataFrame | None = None,
             funding: pd.DataFrame | None = None) -> pd.DataFrame:
    s = grid(spot)
    p = perp.reindex(s.index)
    blc = np.log(btc["close"].where(btc["close"] > 0)).reindex(s.index)
    lc = np.log(s["close"].where(s["close"] > 0))
    r, br = lc.diff(), blc.diff()
    beta = (r.rolling(720, min_periods=168).cov(br) / br.rolling(720, min_periods=168).var()).shift(1)
    out = pd.DataFrame(index=s.index)
    out["ret_24h"], out["ret_7d"] = lc.diff(24), lc.diff(168)
    for h in (6, 24, 168):
        out["resid_7d" if h == 168 else f"resid_{h}h"] = lc.diff(h) - beta * blc.diff(h)
    out["down_resilience_7d"] = (r - br).where(br < 0).rolling(168, min_periods=30).mean()
    low, high = s["low"].rolling(24, min_periods=24).min(), s["high"].rolling(24, min_periods=24).max()
    recovery = (s["close"] - low) / (high - low).where(high > low)
    out["recovery_change_6h"] = recovery.diff(6)
    range24 = (high - low) / s["close"]
    out["range_compression"] = range24 / range24.rolling(168, min_periods=120).mean().shift(1)
    q6 = s["quote_volume"].rolling(6, min_periods=6).sum()
    out["volume_accel_6h"] = np.log((q6 / q6.shift(6)).where((q6 > 0) & (q6.shift(6) > 0)))
    out["rv_7d"] = r.rolling(168, min_periods=120).std()
    out["basis_24h"] = (p["close"] / s["close"] - 1).rolling(24, min_periods=24).mean()
    pq, sq = p["quote_volume"].rolling(24, min_periods=24).sum(), s["quote_volume"].rolling(24, min_periods=24).sum()
    out["perp_share_24h"] = np.log((pq / sq).where((pq > 0) & (sq > 0)))
    out["btc_ret_7d"] = blc.diff(168)
    out["above_sma7d"] = (s["close"] > s["close"].rolling(168, min_periods=168).mean()).where(s["close"].rolling(168, min_periods=168).count() == 168).astype(float)
    for c in SPOT:
        out[c] = np.nan
    if spot_flow is not None and len(spot_flow):
        flow = spot_flow.reindex(s.index)
        total = flow["buy"] + flow["sell"]
        share6 = flow["buy"].rolling(6, min_periods=6).sum() / total.rolling(6, min_periods=6).sum().where(lambda x: x > 0)
        out["spot_buy_share_24h"] = flow["buy"].rolling(24, min_periods=24).sum() / total.rolling(24, min_periods=24).sum().where(lambda x: x > 0)
        out["spot_buy_accel_6h"] = share6 - share6.shift(6)
        out["spot_buy_consistency_6h"] = (flow["buy"] > flow["sell"]).where(total.notna()).rolling(6, min_periods=6).mean()
    if oi is not None and len(oi):
        quantity = oi["oi_usd"].reindex(s.index) / p["close"]
        out["oi_change_24h"] = np.log(quantity.where(quantity > 0)).diff(24)
    if funding is not None and len(funding):
        # Exact settlement-time as-of lookup. At most 24h old; no future fill.
        time = pd.DataFrame({"available_ts": s.index + HOUR})
        source = funding.reset_index().rename(columns={funding.index.name or "index": "settled_ts"})
        interval = source.get("interval_h", source.settled_ts.diff() / HOUR)
        source["funding"] = source["funding"] * 8 / interval.where(interval.isin([1, 2, 4, 8]))
        aligned = pd.merge_asof(time, source.sort_values("settled_ts"), left_on="available_ts", right_on="settled_ts", direction="backward", tolerance=24 * HOUR)
        out["funding"] = aligned["funding"].to_numpy()
    out["close"] = s["close"]
    out["available_ts"] = out.index + HOUR
    return out.replace([np.inf, -np.inf], np.nan)


def cross_section(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    parts = [f.rename_axis("feature_ts").reset_index().assign(symbol=s) for s, f in frames.items() if s != "BTC"]
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    group = out.groupby("feature_ts")
    out["breadth"] = group["above_sma7d"].transform("mean")
    out["rank_resid_24h"] = group["resid_24h"].rank(pct=True)
    out["regime"] = np.where(out["btc_ret_7d"] > .03, "btc_up", np.where(out["btc_ret_7d"] < -.03, "btc_down", "btc_flat"))
    return out


def shadow_rows(table: pd.DataFrame, asof: int) -> pd.DataFrame:
    """Observation scores are research heuristics, not probabilities or buys."""
    if table.empty:
        return table
    recent = table[table.available_ts <= asof].sort_values("available_ts").groupby("symbol").tail(1).copy()
    recent = recent[recent.available_ts >= asof - 2 * HOUR]
    recent["quote_age_min"] = (asof - recent.available_ts) / 60_000
    recent["already_moved"] = recent.ret_24h > np.log1p(.05)
    recent["spot_accumulation"] = (recent.spot_buy_share_24h > .55) & (recent.spot_buy_accel_6h > .03) & (recent.spot_buy_consistency_6h >= 4 / 6)
    recent["independent_strength"] = (recent.down_resilience_7d > 0) & (recent.resid_6h > 0) & (recent.rank_resid_24h >= .7)
    recent["watch_only"] = ~recent.already_moved & (recent.spot_accumulation | recent.independent_strength)
    return recent


def append_shadow(prev: pd.DataFrame | None, snapshot: pd.DataFrame, recorded_ts: int) -> pd.DataFrame:
    """First real observation wins; bar completion is not acquisition time."""
    current = snapshot.copy()
    current["recorded_ts"] = recorded_ts
    parts = [prev, current] if prev is not None and len(prev) else [current]
    return pd.concat(parts, ignore_index=True).drop_duplicates(["symbol", "available_ts"], keep="first")
