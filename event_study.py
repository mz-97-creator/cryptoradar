"""事件前后的价格变化:信息出现时,价格是已经反应了,还是之后才反应?

对事件库(cryptoradar/events.py)里每个事件,以官方时间(event_ts,取整到小时)为 0 点,算该币相对全市场的超额
(剔除 BTC beta 后的累计收益,减去同一时刻全部币的平均):
  事件前:[-72h, 0]、[-24h, 0]         明显为正 = 价格提前反应了(消息泄露 / 大家都知道),信息到你手里已经晚了
  事件后:[0, +24h]、[0, +72h]、[0, +1 周]、[0, +2 周]   为正 = 看到信息后还有机会
同一币同一类事件 72 小时内只算一次;统计用 evalkit(同一天的事件先合并、Newey-West、开发期 / 保留集分开、Holm / BH)。

HYPE 链上回购(buyback_day):先把每天的回购额变成"加速"事件(最近 7 天 ≥ 之前 4 周周均的 1.3 倍,
和 F_REV_UP 同一个定义),再做同样的前后检验;另外报告回购额变化和之后收益的相关性。

用法:
  python event_study.py --events events_history.csv      (币安历史数据库,先用 backfill_vision.py 回填)
结果:reports/event_study.md、reports/event_windows.csv
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

import evalkit as K
import lag_study as L
from cryptoradar.config import load_config

log = logging.getLogger("event_study")
HOUR = K.HOUR
PRE = (72, 24)
POST = (24, 72, 168, 336)


def cum_excess(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """每币每小时的累计超额(对数):逐小时 (币收益 - beta × BTC 收益) 累加,再减去全部币的平均。"""
    inc = {}
    for s, f in frames.items():
        lc = np.log(f["close"].where(f["close"] > 0))
        inc[s] = lc.diff() - f["beta"] * f["_btc_lc"].diff()
    X = pd.DataFrame(inc)
    X = X.sub(X.mean(axis=1), axis=0)                 # 相对同一小时全市场平均
    return X.fillna(0).cumsum().where(X.notna())


def window(C: pd.DataFrame, sym: str, t0: int, a_h: int, b_h: int) -> float:
    """[t0 + a_h, t0 + b_h] 区间的累计超额;端点缺数据返回 NaN。"""
    if sym not in C:
        return np.nan
    c = C[sym]
    ta, tb = t0 + a_h * HOUR, t0 + b_h * HOUR
    if ta not in c.index or tb not in c.index:
        return np.nan
    return float(c.at[tb] - c.at[ta])


def buyback_accel(ev: pd.DataFrame, ratio: float = 1.3) -> pd.DataFrame:
    """回购日数据 -> 加速事件(最近 7 天 / 之前 28 天周均 ≥ ratio)。只用当天结束时已知的数据。"""
    out = []
    for sym, g in ev[ev["kind"] == "buyback_day"].groupby("symbol"):
        s = g.set_index(pd.to_datetime(g["event_ts"] - 1, unit="ms").dt.normalize())["amount_usd"].sort_index()
        s = s.asfreq("D").fillna(0)
        s7 = s.rolling(7, min_periods=7).sum()
        prior = s.shift(7).rolling(28, min_periods=28).sum() / 4
        r = s7 / prior.where(prior > 0)
        for d, v in r[r >= ratio].items():
            out.append({"id": f"accel|{sym}|{d.date()}", "kind": "buyback_accel", "source": g["source"].iloc[0],
                        "symbol": sym, "event_ts": int((d + pd.Timedelta(days=1)).timestamp() * 1000),
                        "amount_usd": float(s7.loc[d]), "title": f"7 日回购 {v:.2f} 倍"})
    return pd.DataFrame(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    ap.add_argument("--events", required=True, help="事件表 CSV(events.py 的 EVENT_COLS)")
    ap.add_argument("--start", default="2025-03-01")
    ap.add_argument("--holdout-start", default=K.HOLDOUT_START)
    ap.add_argument("--out", default=str(Path(__file__).with_name("reports")))
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    frames = L.frames_from_db(load_config(args.config), None)
    frames.pop("BTC", None)
    C = cum_excess(frames)
    ev = pd.read_csv(args.events)
    ev = pd.concat([ev[ev["kind"] != "buyback_day"], buyback_accel(ev)], ignore_index=True)
    start = int(pd.Timestamp(args.start, tz="UTC").timestamp() * 1000)
    ev = ev[(ev["event_ts"] >= start) & ev["symbol"].isin(C.columns)].copy()
    ev["t0"] = (ev["event_ts"] // HOUR) * HOUR
    ev = ev.sort_values("t0")
    # 同一币同一类事件 72 小时内只算一次(不同来源报同一次上新也算一次)
    keep, last = [], {}
    for i, r in ev.iterrows():
        k = (r.symbol, r.kind)
        if k not in last or r.t0 - last[k] >= 72 * HOUR:
            keep.append(i)
            last[k] = r.t0
    ev = ev.loc[keep]
    for a in PRE:
        ev[f"pre{a}"] = [window(C, s, t, -a, 0) for s, t in zip(ev.symbol, ev.t0)]
    for b in POST:
        ev[f"post{b}"] = [window(C, s, t, 0, b) for s, t in zip(ev.symbol, ev.t0)]
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    ev.to_csv(out / "event_windows.csv", index=False, encoding="utf-8-sig")

    sp = lambda v: "—" if v is None or pd.isna(v) else f"{v * 100:+.2f}%"
    tt = lambda v: "—" if v is None or pd.isna(v) else f"{v:+.1f}"
    names = {"spot_list": "现货上新", "perp_list": "合约上新", "buyback_accel": "链上回购加速"}
    rows = []
    for (kind, src), g in ev.groupby(["kind", "source"]):
        for col, h in [(f"pre{a}", a) for a in PRE] + [(f"post{b}", b) for b in POST]:
            dev, hold = K.split_holdout(g.t0.to_numpy(), args.holdout_start, h if col.startswith("post") else 0)
            sd = K.event_stats(g.t0[dev], g[col][dev], h)
            sh = K.event_stats(g.t0[hold], g[col][hold], h)
            rows.append((kind, src, col, sd, sh))
    holm, bh = K.adjust([r[3].get("p", np.nan) for r in rows])
    md = ["# 事件前后的价格变化(币安历史)\n",
          f"事件时间 = 官方发布时间;超额 = 该币剔除 BTC beta 后的累计收益减去同一时刻全市场平均;"
          f"开发期 {args.start} ~ {args.holdout_start},保留集之后。同一币同一类事件 72 小时内只算一次。"
          "只统计事件发生时已经有币安合约的币(价格数据来自币安合约)。\n",
          "**怎么读**:事件前(pre)明显为正 = 价格提前反应,信息到手时已经晚了;事件后(post)为正 = 看到后还有机会。\n",
          "| 事件 | 来源 | 窗口 | 开发期:事件数 | 平均 | 中位 | 跑赢比例 | t | Holm p | 保留集:事件数 | 平均 | t |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for (kind, src, col, sd, sh), hp in zip(rows, holm):
        win = f"前 {col[3:]}h" if col.startswith("pre") else f"后 {col[4:]}h"
        if sd["n"] < 10:
            md.append(f"| {names.get(kind, kind)} | {src} | {win} | {sd['n']} | — | — | — | — | — | {sh['n']} | "
                      f"{sp(sh.get('mean')) if sh['n'] >= 5 else '—'} | — |")
            continue
        md.append(f"| {names.get(kind, kind)} | {src} | {win} | {sd['n']} | {sp(sd['mean'])} | {sp(sd['median'])} | "
                  f"{sd['hit'] * 100:.0f}% | {tt(sd['t'])} | {hp:.3f} | {sh['n']} | {sp(sh.get('mean'))} | {tt(sh.get('t'))} |")
    bb = pd.read_csv(args.events)
    bb = bb[bb["kind"] == "buyback_day"]
    if len(bb):
        md += ["", "## HYPE 链上回购额与之后的收益\n"]
        s = bb.set_index(pd.to_datetime(bb["event_ts"] - 1, unit="ms").dt.normalize())["amount_usd"].sort_index()
        s = s.asfreq("D").fillna(0)
        r7 = np.log((s.rolling(7).sum() + 1) / (s.shift(7).rolling(28).sum() / 4 + 1))
        day_end = ((s.index + pd.Timedelta(days=1)).astype("int64") // 10**6).to_numpy()
        day_end = (pd.Index(s.index + pd.Timedelta(days=1)) - pd.Timestamp(0)) // pd.Timedelta(milliseconds=1)
        for b in (24, 168):
            fwd = pd.Series([window(C, "HYPE", int(t), 0, b) for t in day_end], index=s.index)
            ok = r7.notna() & fwd.notna()
            if ok.sum() > 30:
                ic = r7[ok].rank().corr(fwd[ok].rank())
                md.append(f"- 回购加速程度(近 7 天 / 前 4 周周均,对数)与之后 {b}h 超额的秩相关:{ic:+.3f}"
                          f"(样本 {int(ok.sum())} 天,相邻天的 {b}h 窗口重叠,不是独立样本)")
        md.append(f"- 回购成交覆盖 {s.index.min().date()} ~ {s.index.max().date()},日均 ${s.mean() / 1e6:,.2f}M")
    p = out / "event_study.md"
    p.write_text("\n".join(md), encoding="utf-8")
    print(p.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
