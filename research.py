"""事件研究:某个条件(或条件组合)出现后,未来 24h/72h 的收益分布是什么样。

这是"OP 多头机会评分模型"的第一步:先确认哪些信号在历史上真的有预测力,再谈加权打分。

几个刻意的设计:
1. 前瞻而不是回看:统计的是"条件出现之后"的所有情况,而不是"大涨之前有什么",
   否则只看到成功案例,看不到条件出现了但没涨的那些次
2. 去重叠:条件连续多个小时成立只算一次事件,两次事件至少间隔 72 小时,
   否则 72h 收益高度重叠,样本数和 t 值都会虚高
3. 同时报告收益和最大不利波动(MAE):对杠杆交易来说,"平均赚 3%"
   远不如"途中通常先跌多少"重要。safe_lev 是让 90% 的事件不被强平的最高杠杆近似值
4. 前 70% / 后 30% 分段:两段方向不一致的信号,大概率是噪音
5. 残差收益:剔除 BTC beta,避免把"那段时间 BTC 在涨"误当成信号有效

用法:
  python research.py                       OP 单币
  python research.py --symbol ARBUSDT
  python research.py --pool                合并数据库里所有已回填的币(样本多得多,推荐)
"""
from __future__ import annotations

import argparse
import itertools
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from cryptoradar.config import load_config
from cryptoradar.features import add_labels, build_features
from cryptoradar.signals import RULES, evaluate_frame, merged_thresholds
from cryptoradar.storage import Store
from monitor import setup_logging

log = logging.getLogger("research")
GAP = 72


def decluster(mask: pd.Series, gap: int = GAP) -> pd.Index:
    """条件成立的时点里,只保留与上一次事件间隔 ≥ gap 小时的第一个。"""
    events, last_pos = [], -10**9
    arr = mask.to_numpy()
    for pos in np.flatnonzero(arr):
        if pos - last_pos >= gap:
            events.append(pos)
            last_pos = pos
    return mask.index[events]


def symbol_frame(store: Store, sym: str, btc: pd.DataFrame, eth: pd.DataFrame, th: dict) -> pd.DataFrame:
    df = store.load_hourly(sym)
    if len(df) < 24 * 60:
        return pd.DataFrame()
    fund = store.load_funding(sym)
    f = add_labels(build_features(df, btc, fund, 8.0, eth))
    conds = evaluate_frame(f, th)
    for a, b in itertools.combinations([r.id for r in RULES], 2):
        conds[f"{a}+{b}"] = conds[a] & conds[b]
    f = f.join(conds.add_prefix("C_"))
    f["symbol"] = sym
    return f


def stats(sub: pd.DataFrame, split_ts: float) -> dict:
    r24, r72 = sub["fwd_ret_24h"], sub["fwd_ret_72h"]
    x72 = sub["fwd_resid_72h"]
    mae = sub["mae_72h"]
    n = len(sub)
    sd = x72.std()
    early = x72[sub.index.get_level_values("ts") < split_ts] if isinstance(sub.index, pd.MultiIndex) \
        else x72[sub.index < split_ts]
    late = x72.drop(early.index)
    mae_p10 = mae.quantile(0.10)
    return {
        "n": n,
        "ret24_mean": r24.mean(),
        "ret72_mean": r72.mean(),
        "resid72_mean": x72.mean(),
        "resid72_median": x72.median(),
        "hit72": (x72 > 0).mean(),
        "t72": x72.mean() / (sd / np.sqrt(n)) if n > 1 and sd > 0 else np.nan,
        "mae72_mean": mae.mean(),
        "mae72_p10": mae_p10,
        "safe_lev": 1 / abs(mae_p10) if mae_p10 < 0 else np.nan,
        "early_resid72": early.mean() if len(early) else np.nan,
        "late_resid72": late.mean() if len(late) else np.nan,
    }


def run(frames: list[pd.DataFrame], min_n: int) -> pd.DataFrame:
    data = pd.concat(frames)
    data.index.name = "ts"
    data = data.set_index("symbol", append=True)
    label_ok = data[["fwd_ret_72h", "fwd_resid_72h", "mae_72h"]].notna().all(axis=1)
    split_ts = data.index.get_level_values("ts").to_series().quantile(0.7)

    cond_cols = [c for c in data.columns if c.startswith("C_")]
    rows = []

    # 基准:每隔 72 小时抽一个点,代表"随便什么时候开仓"
    base_parts = []
    for sym, g in data[label_ok].groupby(level="symbol"):
        base_parts.append(g.iloc[::GAP])
    base = pd.concat(base_parts)
    rows.append({"condition": "基准(任意时点)", **stats(base, split_ts)})

    for c in cond_cols:
        parts = []
        for sym, g in data.groupby(level="symbol"):
            m = g[c] & label_ok.loc[g.index]
            ev = decluster(m.reset_index(level="symbol", drop=True))
            if len(ev):
                parts.append(g.loc[[(t, sym) for t in ev]])
        if not parts:
            continue
        sub = pd.concat(parts)
        if len(sub) < min_n:
            continue
        rows.append({"condition": c[2:], **stats(sub, split_ts)})

    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="信号事件研究")
    ap.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    ap.add_argument("--symbol", default="OPUSDT")
    ap.add_argument("--pool", action="store_true", help="合并所有已回填的币")
    ap.add_argument("--min-n", type=int, default=15, help="事件数少于该值的条件不显示")
    args = ap.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg["_base_dir"])
    store = Store(cfg["storage"]["db_path"])
    th = merged_thresholds(cfg["signals"].get("thresholds"))

    btc = store.load_hourly("BTCUSDT")
    eth = store.load_hourly("ETHUSDT")
    if len(btc) < 24 * 60:
        raise SystemExit("BTC 历史数据不足,请先运行 python backfill.py")

    if args.pool:
        syms = [r[0] for r in store.conn.execute(
            "SELECT symbol FROM hourly WHERE oi IS NOT NULL GROUP BY symbol HAVING COUNT(*) > 2000")]
        syms = [s for s in syms if s not in ("BTCUSDT",)]
    else:
        syms = [args.symbol.upper()]

    frames = []
    for s in syms:
        f = symbol_frame(store, s, btc, eth, th)
        if f.empty:
            log.warning("%s 数据不足,跳过(先用 backfill.py 回填)", s)
            continue
        frames.append(f)
        log.info("%s:%d 小时样本", s, len(f))
    if not frames:
        raise SystemExit("没有可用数据")

    res = run(frames, args.min_n)
    base = res.iloc[0]
    res = res.iloc[1:].sort_values("resid72_mean", ascending=False)
    res = pd.concat([base.to_frame().T, res])

    out_dir = Path(cfg["_base_dir"]) / "reports"
    out_dir.mkdir(exist_ok=True)
    tag = "pool" if args.pool else args.symbol.upper()
    out = out_dir / f"event_study_{tag}.csv"
    res.to_csv(out, index=False, encoding="utf-8-sig")

    show = res.copy()
    for c in ["ret24_mean", "ret72_mean", "resid72_mean", "resid72_median", "hit72",
              "mae72_mean", "mae72_p10", "early_resid72", "late_resid72"]:
        show[c] = (show[c].astype(float) * 100).map(lambda v: "" if pd.isna(v) else f"{v:+.2f}%")
    show["t72"] = show["t72"].astype(float).map(lambda v: "" if pd.isna(v) else f"{v:+.2f}")
    show["safe_lev"] = show["safe_lev"].astype(float).map(lambda v: "" if pd.isna(v) else f"{v:.1f}x")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 200)
    print(show.to_string(index=False))
    n_tests = len(res) - 1
    print(f"\n共检验 {n_tests} 个条件。多重检验下,|t|<2.5 的结果大概率是噪音;"
          "early/late 两段方向相反的也不可信。")
    print("safe_lev:按历史 10% 最差的持有期回撤估算的'不被强平'杠杆上限(未计手续费和维持保证金)。")
    print(f"完整结果已保存:{out}")


if __name__ == "__main__":
    main()
