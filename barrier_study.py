"""Generate research labels from an OHLC CSV. Never sends notifications."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import pandas as pd

from cryptoradar.barriers import BarrierSpec, holdout_masks, label_bars


def main():
    ap = argparse.ArgumentParser(description="成本调整的下一小时开盘入场研究标签")
    ap.add_argument("--input", required=True, help="CSV: symbol,ts,open,high,low,close; ts=毫秒开盘时间")
    ap.add_argument("--out", required=True)
    ap.add_argument("--test-start", required=True, help="预先固定的最终检验起点,ISO 日期/时间(UTC)")
    ap.add_argument("--horizon-h", type=int, default=168)
    ap.add_argument("--take-profit", type=float, default=0.20)
    ap.add_argument("--stop-loss", type=float, default=0.10)
    ap.add_argument("--cost-per-side", type=float, default=0.001)
    args = ap.parse_args()
    spec = BarrierSpec(args.horizon_h, args.take_profit, args.stop_loss, args.cost_per_side, args.cost_per_side)
    data = pd.read_csv(args.input)
    if not {"symbol", "ts"} <= set(data):
        ap.error("input requires symbol and ts")
    labels = []
    for symbol, bars in data.groupby("symbol"):
        labels.append(label_bars(bars.set_index("ts").sort_index(), spec).assign(symbol=symbol))
    if not labels:
        ap.error("input is empty")
    result = pd.concat(labels, ignore_index=True)
    test_start = int(pd.to_datetime(args.test_start, utc=True).timestamp() * 1000)
    train, test = holdout_masks(result, test_start)
    result["split"] = "excluded"
    result.loc[train, "split"], result.loc[test, "split"] = "train", "final_test"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    result.to_csv(out / "barrier_labels.csv", index=False)
    summary = {"spec": asdict(spec), "test_start_utc": args.test_start,
               "outcomes": result["outcome"].value_counts().to_dict(),
               "train": int(train.sum()), "final_test": int(test.sum()),
               "note": "研究标签,不是样本外模型效果。上/下障碍同小时触及无法确定先后,排除;缺失未来数据不记失败。"}
    (out / "barrier_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
