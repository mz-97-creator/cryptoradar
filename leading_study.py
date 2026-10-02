"""Predeclared walk-forward/final-holdout study for early, accurate alerts.

All variants are recorded; no winner is chosen using the final holdout. Results
are historical signal outcomes on spot prices, not an executable portfolio.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from cryptoradar.barriers import BarrierSpec, label_bars
from cryptoradar.events import available_events
from cryptoradar.leading import HOUR, STRENGTH, SPOT, append_shadow, cross_section, features, shadow_rows

log = logging.getLogger("leading.study")
DAY = 24 * HOUR


def load(folder: Path, symbol: str, tag: str) -> pd.DataFrame:
    path = folder / f"{symbol}_{tag}.csv.gz"
    if not path.exists():
        return pd.DataFrame()
    f = pd.read_csv(path)
    return f.drop_duplicates("ts").set_index("ts").sort_index() if "ts" in f else pd.DataFrame()


def build(folder: Path, plan: dict, asof: int):
    btc = load(folder, "BTC", "spot")
    if btc.empty:
        raise ValueError("BTC spot history is required")
    feat, label_parts = {}, []
    spec = BarrierSpec(plan["horizon_h"], plan["take_profit"], plan["stop_loss"], plan["cost_per_side"], plan["cost_per_side"])
    for s in plan["symbols"]:
        spot, perp = load(folder, s, "spot"), load(folder, s, "perp")
        if spot.empty or perp.empty:
            log.warning("Skipping %s: spot or perp is missing", s)
            continue
        spot, perp = spot.loc[spot.index + HOUR <= asof], perp.loc[perp.index + HOUR <= asof]
        feat[s] = features(spot, perp, btc, load(folder, s, "spot_flow"), load(folder, s, "oi"), load(folder, s, "funding"))
        if s != "BTC":
            labels = label_bars(spot, spec, plan["decision_stride_h"], plan["startup_return"])
            label_parts.append(labels.assign(symbol=s))
    table = cross_section(feat)
    labels = pd.concat(label_parts, ignore_index=True)
    joined = table.merge(labels, on=["symbol", "feature_ts"], how="inner")
    # Same past-only candidate filter applies to every model/baseline.
    joined["candidate"] = (joined.ret_24h <= np.log1p(plan["max_prior_24h_return"])) & joined.resid_7d.notna()
    joined["weekday"] = pd.to_datetime(joined.entry_ts, unit="ms", utc=True).dt.weekday
    joined = joined[joined.weekday.isin(plan["decision_weekdays_utc"])].copy()
    joined["lead_h_lower"] = (joined.startup_earliest_ts - joined.entry_ts) / HOUR
    joined["lead_h_upper"] = (joined.startup_latest_ts - joined.entry_ts) / HOUR
    return table, joined


def training_mask(data: pd.DataFrame, cutoff: int, plan: dict) -> pd.Series:
    boundary = cutoff - plan["embargo_h"] * HOUR
    return (data.candidate & data.target.notna() & (data.label_end_ts < boundary)
            & (data.entry_ts + plan["horizon_h"] * HOUR < boundary))


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, arm: str):
    if arm == "momentum_baseline":
        return test.resid_7d.to_numpy(), None
    cols = STRENGTH + (SPOT if arm == "tree_strength_spot" else [])
    if len(train) < 200 or train.target.sum() < 20 or train.target.nunique() != 2:
        return None, "insufficient training rows or positive events"
    if arm == "tree_strength_spot":
        coverage = train[SPOT].notna().mean()
        if (coverage < .5).any():
            return None, "spot/derivatives history covers less than 50% of training rows: " + str(coverage.round(3).to_dict())
    if arm == "logistic_strength":
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        m = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(), LogisticRegression(C=.1, max_iter=1000, random_state=0))
    else:
        from sklearn.ensemble import HistGradientBoostingClassifier
        m = HistGradientBoostingClassifier(max_depth=3, max_iter=100, learning_rate=.05, l2_regularization=10,
                                           min_samples_leaf=50, random_state=0)
    m.fit(train[cols], train.target)
    return m.predict_proba(test[cols])[:, 1], None


def alerts(scored: pd.DataFrame, arm: str, plan: dict) -> pd.DataFrame:
    """Choose before seeing outcomes. Missing outcomes do not get replaced."""
    eligible = scored[scored.candidate & scored.score.notna()].copy()
    if arm != "momentum_baseline":
        eligible = eligible[eligible.score >= plan["model_probability_floor"]]
    if arm == "tree_strength_spot":
        eligible = eligible[eligible[SPOT].notna().all(axis=1)]
    return eligible.sort_values(["entry_ts", "score", "symbol"], ascending=[True, False, True]).groupby("entry_ts").head(plan["max_alerts_per_decision"])


def summary(chosen: pd.DataFrame, pool: pd.DataFrame) -> dict:
    observed = chosen[chosen.target.notna()]
    success = observed[observed.target == 1]
    base = pool[pool.candidate & pool.target.notna()]
    return {"alerts": int(len(chosen)), "resolved": int(len(observed)),
            "successes": int(len(success)),
            "positive_candidate_rows": int(base.target.sum()),
            "candidate_row_recall": float(len(success) / base.target.sum()) if base.target.sum() else None,
            "unknown_outcomes": int(chosen.target.isna().sum()), "decision_days": int(pool.entry_ts.nunique()),
            "precision": float(observed.target.mean()) if len(observed) else None,
            "precision_lower": float(success.shape[0] / len(chosen)) if len(chosen) else None,
            "precision_upper": float((success.shape[0] + chosen.target.isna().sum()) / len(chosen)) if len(chosen) else None,
            "candidate_base_rate": float(base.target.mean()) if len(base) else None,
            "net_return_mean": float(observed.net_return.mean()) if len(observed) else None,
            "success_lead_h_lower_median": float(success.lead_h_lower.median()) if len(success) else None,
            "success_lead_h_upper_median": float(success.lead_h_upper.median()) if len(success) else None,
            "success_at_least_6h_early": float((success.lead_h_lower >= 6).mean()) if len(success) else None,
            "prior_24h_return_median": float(np.expm1(observed.ret_24h.median())) if len(observed) else None,
            "by_regime": {r: {"n": int(len(g)), "precision": float(g.target.mean()), "net_return_mean": float(g.net_return.mean())}
                          for r, g in observed.groupby("regime")}}


def block_interval(chosen: pd.DataFrame, baseline: pd.DataFrame, days: np.ndarray, plan: dict) -> dict:
    """Paired 14-day moving-block bootstrap of daily signal-return totals.

    Coins from the same day stay together; overlaps of 7-day outcomes stay in
    blocks. Abstention contributes zero. Unknown outcomes make inference invalid.
    This is not portfolio PnL: overlapping hypothetical trades are not financed.
    """
    if chosen.target.isna().any() or baseline.target.isna().any():
        return {"status": "unknown selected outcomes; no confidence interval"}
    def daily(f):
        return f.groupby("entry_ts").net_return.sum().reindex(days, fill_value=0).to_numpy()
    delta = daily(chosen) - daily(baseline)
    n, block = len(days), plan["bootstrap_block_days"]
    if n < block * 3:
        return {"status": "too few time blocks"}
    rng = np.random.default_rng(0)
    values = []
    for _ in range(plan["bootstrap_repetitions"]):
        starts = rng.integers(0, n - block + 1, size=int(np.ceil(n / block)))
        pos = np.concatenate([np.arange(k, k + block) for k in starts])[:n]
        values.append(float(delta[pos].mean()))
    lo, hi = np.quantile(values, [.025, .975])
    return {"status": "exploratory; 4 registered arms, not adjusted for multiple comparisons",
            "mean_daily_signal_return_difference": float(delta.mean()), "ci95": [float(lo), float(hi)],
            "block_calendar_days": block, "rough_independent_blocks": n / block}


def evaluate(data: pd.DataFrame, plan: dict, asof: int):
    cutoff = int(pd.to_datetime(plan["final_test_start_utc"], utc=True).timestamp() * 1000)
    dev_edges = [int(pd.to_datetime(t, utc=True).timestamp() * 1000) for t in plan["development_fold_starts_utc"]] + [cutoff]
    result = {"development": {}, "final_holdout": {}}
    selections, final_scores = {}, []
    for stage, windows in (("development", list(zip(dev_edges[:-1], dev_edges[1:]))), ("final_holdout", [(cutoff, asof - plan["horizon_h"] * HOUR)])):
        for arm in plan["arms"]:
            batches, score_batches, skips = [], [], []
            for lo, hi in windows:
                train = data[training_mask(data, lo, plan)]
                end = min(hi, cutoff - plan["horizon_h"] * HOUR) if stage == "development" else hi
                test = data[(data.entry_ts >= lo) & (data.entry_ts < end)].copy()
                if not len(test):
                    skips.append("no test rows")
                    continue
                score, why = fit_predict(train, test, arm)
                if why:
                    skips.append(why)
                    continue
                test["score"], test["arm"] = score, arm
                batches.append(alerts(test, arm, plan))
                score_batches.append(test)
                log.info("%s %s train=%d test=%d", stage, arm, len(train), len(test))
            if not score_batches:
                result[stage][arm] = {"status": "skipped", "reasons": skips}
                continue
            scored, chosen = pd.concat(score_batches), pd.concat(batches)
            result[stage][arm] = {"status": "evaluated", **summary(chosen, scored), "skipped_windows": skips}
            if stage == "final_holdout":
                selections[arm] = chosen
                final_scores.append(scored)
    pool = data[(data.entry_ts >= cutoff) & (data.entry_ts < asof - plan["horizon_h"] * HOUR)]
    days = np.arange((cutoff // DAY) * DAY, ((asof - plan["horizon_h"] * HOUR) // DAY) * DAY + DAY, DAY)
    baseline = selections.get("momentum_baseline")
    if baseline is not None:
        for arm, chosen in selections.items():
            result["final_holdout"][arm]["vs_momentum_baseline"] = block_interval(chosen, baseline, days, plan)
    result["coverage"] = {"rows": len(data), "final_rows": len(pool), "final_candidate_days": int(pool[pool.candidate].entry_ts.nunique()),
                          "spot_train_coverage": data.loc[training_mask(data, cutoff, plan), SPOT].notna().mean().to_dict(),
                          "spot_final_coverage": pool[SPOT].notna().mean().to_dict()}
    return result, selections, pd.concat(final_scores, ignore_index=True) if final_scores else pd.DataFrame()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default="leading_plan.json")
    ap.add_argument("--data", default="codex-data/leading")
    ap.add_argument("--out", default="codex-reports/leading")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    plan_text = Path(args.plan).read_text()
    plan = json.loads(plan_text)
    folder, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    source = json.loads((folder / "source_manifest.json").read_text())
    asof = source["asof_ts"]
    table, data = build(folder, plan, asof)
    result, selections, scored = evaluate(data, plan, asof)
    result.update(plan_sha256=hashlib.sha256(plan_text.encode()).hexdigest(), plan=plan, source_manifest=source,
                  limitations=["Fixed current survivor universe, not all-market performance", "Historical reconstructed inputs, not original live forecasts",
                               "Cost proxy, no order book execution or capital curve", "Small number of independent 14-day blocks in final holdout",
                               "Spot taker feed is coin-level flow, not exchange deposits/withdrawals", "New economic/announcement sources are not validated yet"])
    (out / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    data.to_csv(out / "labeled_features.csv.gz", index=False)
    scored.to_csv(out / "final_scores.csv.gz", index=False)
    for arm, selected in selections.items():
        selected.to_csv(out / f"alerts_{arm}.csv", index=False)
    decision_ts = int(datetime.now(timezone.utc).timestamp() * 1000)
    shadow = shadow_rows(table, decision_ts)
    event_path = folder / "events.jsonl"
    events = [json.loads(line) for line in event_path.read_text().splitlines() if line.strip()] if event_path.exists() else []
    shadow["known_events"] = [json.dumps(available_events(events, s, decision_ts), ensure_ascii=False) for s in shadow.symbol]
    shadow.to_csv(out / "shadow_watch.csv", index=False)
    history_path = folder / "shadow_observations.csv.gz"
    prior = pd.read_csv(history_path) if history_path.exists() else None
    append_shadow(prior, shadow, decision_ts).to_csv(history_path, index=False)
    print(json.dumps({"final_holdout": result["final_holdout"], "coverage": result["coverage"],
                      "shadow_watch_only": shadow.loc[shadow.watch_only, "symbol"].tolist()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
