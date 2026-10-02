"""规则的长期验证:去掉"全市场共同因子"和"同一天多个币一起触发"这两个会让结果显得过于确定的因素。

做法(币安历史,约 2.7 年,全部币):
  1. 每个事件的 72h 超额收益 = 该币相对 BTC 的 72h 残差 - 同一时刻全部币残差的均值(扣掉共同因子)
  2. 同一天触发的多个币只算一天(按日取均值),再用 Newey-West 校正自相关算 t 值
  3. 通过标准:|t| ≥ 3(13 条规则同时检验,比 2 更严格)且前半段、后半段方向一致
结果写进 models/rule_validation.json;cloud_run 的"偏涨/偏跌"只对通过的规则出现,其余只给历史频率、不下方向结论。

为什么要这样:云端样本库只有约 27 天,多个币在同一轮行情里一起触发,实际独立的证据只有十几天;
再从 13 条条件里挑偏离最大的那条,会进一步夸大。旧口径下 RESID t=4.9、COIL t=-5.2,换成本口径后都不显著。

用法:  python rule_validation.py
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import tune
from cryptoradar.config import load_config
from cryptoradar.signals import RULES, merged_thresholds
from cryptoradar.storage import Store

log = logging.getLogger("rule_validation")
T_BAR = 3.0
OUT = Path(__file__).with_name("models") / "rule_validation.json"


def nw_t(x: pd.Series, lags: int) -> float:
    x = x.dropna().to_numpy()
    n = len(x)
    if n < 10:
        return float("nan")
    e = x - x.mean()
    var = np.mean(e * e)
    for l in range(1, min(lags, n - 1) + 1):
        var += 2 * (1 - l / (lags + 1)) * np.mean(e[l:] * e[:-l])
    return float(x.mean() / np.sqrt(max(var, 1e-18) / n))


def validate(frames, hits, ids) -> pd.DataFrame:
    parts = []
    for f, h in zip(frames, hits):
        g = pd.DataFrame({"ts": f.index.to_numpy(), "sym": f["symbol"].iloc[0], "r": f["fwd_resid_72h"].to_numpy()})
        for j, i in enumerate(ids):
            g[i] = h[:, j] > 0
        parts.append(g)
    D = pd.concat(parts, ignore_index=True)
    D = D[D.r.notna()].copy()
    D["ex"] = D["r"] - D.groupby("ts")["r"].transform("mean")
    D["day"] = D.ts // 86_400_000
    mid = D.ts.quantile(0.5)
    rows = []
    for i in ids:
        ev = []
        for _, g in D.groupby("sym"):
            g = g.sort_values("ts")
            # 按真实时间间隔去重:D 已去掉缺标签的行,行号之差不等于小时数
            ts, keep, last = g["ts"].to_numpy(), [], None
            for p in np.flatnonzero(g[i].to_numpy()):
                if last is None or ts[p] - last >= 72 * 3_600_000:
                    keep.append(p)
                    last = ts[p]
            if keep:
                ev.append(g.iloc[keep])
        if not ev:
            continue
        E = pd.concat(ev)
        daily = E.groupby("day")["ex"].mean()
        raw = E.groupby("day")["r"].mean()
        e1, e2 = E[E.ts < mid].groupby("day")["ex"].mean().mean(), E[E.ts >= mid].groupby("day")["ex"].mean().mean()
        t = nw_t(daily, 3)
        rows.append({"rule": i, "n_events": int(len(E)), "n_days": int(len(daily)), "excess_mean": float(daily.mean()),
                     "t": t, "t_raw_vs_btc": nw_t(raw, 3), "first_half": float(e1), "second_half": float(e2),
                     "validated": bool(abs(t) >= T_BAR and np.sign(e1) == np.sign(e2)),
                     "direction": "up" if daily.mean() > 0 else "down"})
    return pd.DataFrame(rows)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cfg = load_config(str(Path(__file__).with_name("config.yaml")))
    th = merged_thresholds(cfg["signals"].get("thresholds"))
    frames = tune.load_frames(Store(cfg["storage"]["db_path"]), th)
    ids = [r.id for r in RULES]
    R = validate(frames, tune.hit_matrix(frames, th), ids)
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps({"made_at": datetime.now(timezone.utc).isoformat(), "t_bar": T_BAR,
                               "rules": {r["rule"]: r for r in R.to_dict("records")}}, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    pd.set_option("display.width", 220)
    print(R.sort_values("t", ascending=False).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\n通过验证的规则:{[r for r in R[R.validated].rule] or '无'}  → 已写入 {OUT}")


if __name__ == "__main__":
    main()
