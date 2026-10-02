"""Import sourced economic observations; availability starts at import time."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from cryptoradar.events import append_events


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="JSON list with kind/source_url/symbols and economic evidence")
    ap.add_argument("--ledger", default="codex-data/leading/events.jsonl")
    args = ap.parse_args()
    records = json.loads(Path(args.input).read_text())
    if not isinstance(records, list):
        ap.error("input must be a list")
    for e in records:
        e["verification_status"] = "sourced_import_not_independently_chain_verified"
    all_events = append_events(Path(args.ledger), records, int(datetime.now(timezone.utc).timestamp() * 1000))
    print(f"Ledger has {len(all_events)} revisions; imported publication dates cannot backdate first-seen time")


if __name__ == "__main__":
    main()
