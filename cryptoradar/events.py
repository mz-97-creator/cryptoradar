"""Immutable event observations: publication time is not first-known time."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from urllib.parse import urlparse

KINDS = {"listing", "delisting", "announcement", "buyback_execution", "protocol_revenue", "unlock"}


def append_events(path: Path, events: list[dict], observed_at: int) -> list[dict]:
    """Record revisions without changing first_seen; require explicit evidence.

    Non-announcement events are accepted only as sourced observations, never as
    inferred execution amounts. On-chain buybacks require chain/tx/contract.
    Historical imports are available no earlier than their first import here.
    """
    old = [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []
    first = {e["event_key"]: e["first_seen_ts"] for e in old}
    seen = {(e["event_key"], e["revision_hash"]) for e in old}
    new = []
    for event in events:
        e = dict(event)
        kind = e.get("kind")
        url = e.get("source_url", "")
        if kind not in KINDS or urlparse(url).scheme != "https" or not urlparse(url).netloc:
            raise ValueError("Event requires a supported kind and HTTPS evidence URL")
        if kind == "buyback_execution" and not all(e.get(k) for k in ("chain", "tx_hash", "contract", "amount_usd")):
            raise ValueError("Executed buyback requires chain, tx_hash, contract and positive USD amount")
        if kind in {"buyback_execution", "protocol_revenue"}:
            if float(e.get("amount_usd", -1)) <= 0:
                raise ValueError("Observed economic amount must be positive")
        key = str(e.get("event_key") or (e.get("tx_hash") if kind == "buyback_execution" else url))
        content = {k: v for k, v in e.items() if k != "observed_quotes"}
        revision = hashlib.sha256(json.dumps(content, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        if (key, revision) in seen:
            continue
        published = int(e.get("published_ts") or observed_at)
        available = max(observed_at, published)
        e.update(event_key=key, revision_hash=revision, first_seen_ts=first.get(key, observed_at),
                 revision_seen_ts=observed_at, available_ts=available,
                 publication_to_observation_h=(observed_at - published) / 3_600_000)
        first.setdefault(key, observed_at)
        seen.add((key, revision))
        new.append(e)
    if new:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            for e in new:
                fh.write(json.dumps(e, ensure_ascii=False) + "\n")
    return old + new


def available_events(events: list[dict], symbol: str, decision_ts: int) -> list[dict]:
    """Use only revisions already observed at the decision, with matching token."""
    latest = {}
    for e in events:
        if e["available_ts"] <= decision_ts and symbol in e.get("symbols", []):
            if e["event_key"] not in latest or e["revision_seen_ts"] > latest[e["event_key"]]["revision_seen_ts"]:
                latest[e["event_key"]] = e
    return list(latest.values())
