"""早期检测(cryptoradar/early.py,含现货 S_* 与基本面 F_*)的历史验证:能不能比现有推送更早抓到大涨,触发之后有没有超额收益。

三个问题:
  1. 触发之后:72h / 1 周 / 2 周相对全市场(同一时刻全部币的平均超额)还有没有超额收益?
     事件去重(同一币同一检测至少间隔 72 小时),按天聚合,t 值用 Newey-West 校正;前后两半时间段方向要一致。
  2. 时机:在 lag_study.py 划出的每段大涨里,它多常在"初段"(最低点到涨幅走完 30%)就响?
     各阶段的触发倾向 = 落在该阶段的比例 / 随机一个小时落在该阶段的比例。
  3. 合起来:把它和现有"读起来像在涨"的推送合在一起,第一次提醒能提前多少、多走完多少,代价是每币每天多几次提醒。

用法(先用 backfill_vision.py 回填币安历史):
  python early_study.py --start 2025-03-01
  python early_study.py --start 2025-03-01 --spot --fund-history data/fund_history.json   连同现货与基本面检测
结果:reports/early_study.md、reports/early_events.csv
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

import lag_study as L
from cryptoradar import early
from cryptoradar.config import load_config
from cryptoradar.signals import apply_weights, merged_thresholds

log = logging.getLogger("early_study")
HOUR = L.HOUR
HORIZONS = (72, 168, 336)


def nw_t(x: pd.Series, lags: int) -> float:
    x = pd.Series(x, dtype=float).dropna().to_numpy()
    n = len(x)
    if n < 10:
        return float("nan")
    e = x - x.mean()
    var = np.mean(e * e)
    for l in range(1, min(lags, n - 1) + 1):
        var += 2 * (1 - l / (lags + 1)) * np.mean(e[l:] * e[:-l])
    return float(x.mean() / np.sqrt(max(var, 1e-18) / n))


def decluster(ts: np.ndarray, gap_h: int = 72) -> np.ndarray:
    keep, last = [], None
    for i, t in enumerate(ts):
        if last is None or t - last >= gap_h * HOUR:
            keep.append(i)
            last = t
    return np.array(keep, dtype=int)


def forward_resid(f: pd.DataFrame, h: int) -> pd.Series:
    lc = np.log(f["close"].where(f["close"] > 0))
    return (lc.shift(-h) - lc) - f["beta"] * (f["_btc_lc"].shift(-h) - f["_btc_lc"])


def event_stats(E: pd.DataFrame, h: int) -> dict:
    """E:去重后的事件,含 ts 和 ex{h}(相对全市场的超额)。"""
    x = E[f"ex{h}"].dropna()
    if len(x) < 10:
        return {"n": len(x)}
    d = E.assign(day=E.ts // (24 * HOUR)).dropna(subset=[f"ex{h}"])
    daily = d.groupby("day")[f"ex{h}"].mean()
    mid = d.ts.quantile(0.5)
    e1, e2 = d[d.ts < mid][f"ex{h}"].mean(), d[d.ts >= mid][f"ex{h}"].mean()
    return {"n": len(x), "mean": x.mean(), "median": x.median(), "hit": (x > 0).mean(),
            "t": nw_t(daily, max(3, h // 24)), "first": e1, "second": e2}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    ap.add_argument("--rules-config", default=str(Path(__file__).with_name("cloud_config.yaml")))
    ap.add_argument("--symbols")
    ap.add_argument("--start", help="只评这天之后的事件和大涨,如 2025-03-01")
    ap.add_argument("--min-gain", type=float, default=0.20)
    ap.add_argument("--reversal", type=float, default=0.10)
    ap.add_argument("--pre-h", type=int, default=48)
    ap.add_argument("--spot", action="store_true", help="接上币安现货资金流(先运行 python -m cryptoradar.spotflow 回填)")
    ap.add_argument("--fund-history", help="DefiLlama 完整历史的 JSON({mapping, coins}),接上基本面特征")
    ap.add_argument("--out", default=str(Path(__file__).with_name("reports")))
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    rc = load_config(args.rules_config)
    sc = rc["signals"]
    th_rules = merged_thresholds(sc.get("thresholds"))
    rules = apply_weights(sc.get("rule_weights"))
    eth = early.thresholds(rc.get("early", {}).get("thresholds"))
    watch = {s.upper() for s in rc["universe"].get("watchlist", [])}
    syms = [s.strip().upper() for s in args.symbols.split(",")] if args.symbols else None
    cfg = load_config(args.config)
    frames = L.frames_from_db(cfg, syms)
    frames.pop("BTC", None)
    if args.spot:
        import sqlite3
        from cryptoradar import spotflow
        from backfill_vision import to_symbol
        from cryptoradar.storage import Store
        conn, store = sqlite3.connect(cfg["storage"]["db_path"]), Store(cfg["storage"]["db_path"])
        for sym, f in frames.items():
            ps = to_symbol(sym)                  # 合约名,如 PEPE -> 1000PEPEUSDT
            perp = store.load_hourly(ps).reindex(f.index)
            sp = spotflow.features(conn, ps, perp)
            for c in ("spot_buy_z", "spot_buy_dz12", "lev_share_z"):
                frames[sym][c] = sp[c].to_numpy()
        log.info("现货特征:%d 个币有数据", sum(f["spot_buy_z"].notna().any() for f in frames.values()))
    if args.fund_history:
        import json
        from cryptoradar import fundamentals as fd
        fh = json.loads(Path(args.fund_history).read_text(encoding="utf-8"))
        for sym in list(frames):
            s = fh["coins"].get(sym)
            if s:
                frames[sym] = fd.attach_history(frames[sym], fd.daily_features(s))
        log.info("基本面特征:%d 个币有数据", sum(1 for s in frames if s in fh["coins"]))
    frames = early.add_cross_section(frames, int(eth["breakout_days"]))
    start = int(pd.Timestamp(args.start, tz="UTC").timestamp() * 1000) if args.start else -np.inf

    # 全市场同一时刻的平均超额,用来把"币自己涨"和"那段时间山寨普涨"分开
    fwd = {h: pd.DataFrame({s: forward_resid(f, h) for s, f in frames.items()}) for h in HORIZONS}
    mkt = {h: fwd[h].mean(axis=1) for h in HORIZONS}

    det_ids = [d.id for d in early.DETECTORS]
    events, rally_rows, phase_parts = [], [], []
    coin_hours = 0
    for sym, f in sorted(frames.items()):
        if f["close"].notna().sum() < 24 * 30 or f["resid_24h_z"].notna().sum() == 0:
            continue
        valid = f["resid_24h_z"].notna().to_numpy()
        first_ok = int(f.index[valid][0])
        lo_ok = max(first_ok + max(args.pre_h, 480) * HOUR, start)    # 20 日新高、7 日排名都需要预热
        idx = f.index.to_numpy()
        in_range = (idx >= lo_ok) & valid
        coin_hours += int(in_range.sum())
        E = early.evaluate_frame(f, eth)
        need = sc.get("watchlist_min_score", 1.0) if sym in watch else sc.get("min_score_to_push", 2.5)
        hits, pushes = L.replay(f, th_rules, rules, need, float(sc.get("cooldown_hours", 6)))
        up = L.up_flavored(f, hits)
        up_push = pushes.index[up.reindex(pushes.index).fillna(False).to_numpy(dtype=bool)]
        # 2/3 要用的大涨分段(先算,事件表里标出触发时是否正处在某段大涨的初段)
        x = f["xs_lc"]
        rallies = [r for r in L.zigzag_rallies(x, args.min_gain, args.reversal) if r[0] >= lo_ok]
        ph = L.phase_masks(idx, x, rallies, args.pre_h)
        in_early = pd.Series(ph["early"], index=idx)
        # 1. 事件 + 前瞻超额
        for d in det_ids:
            ts = idx[E[d].to_numpy() & in_range]
            if len(ts) == 0:
                continue
            ts = ts[decluster(ts)]
            row = pd.DataFrame({"symbol": sym, "detector": d, "ts": ts, "in_early": in_early.reindex(ts).to_numpy()})
            for h in HORIZONS:
                row[f"ex{h}"] = fwd[h][sym].reindex(ts).to_numpy() - mkt[h].reindex(ts).to_numpy()
            events.append(row)
        # 2/3. 大涨时机
        events.append(pd.DataFrame({"symbol": sym, "detector": "__up_push__", "ts": up_push[up_push >= lo_ok],
                                    "in_early": in_early.reindex(up_push[up_push >= lo_ok]).to_numpy()}))
        for d in det_ids + ["__up_push__"]:
            on = E[d].to_numpy() if d in E else np.isin(idx, up_push.to_numpy())
            on = on & in_range
            phase_parts.append(pd.DataFrame({"detector": d, "n": int(on.sum()),
                                             **{k: [int((on & v).sum())] for k, v in ph.items()}}))
        phase_parts.append(pd.DataFrame({"detector": "__base__", "n": int(in_range.sum()),
                                         **{k: [int((in_range & v).sum())] for k, v in ph.items()}}))
        first_any_early = {}
        for t0, t1, g in rallies:
            x0, x1 = x.get(t0), x.get(t1)
            seg = x.loc[t0:t1]
            q30 = seg.index[(seg - x0).to_numpy() >= 0.3 * (x1 - x0)]
            t30 = int(q30[0]) if len(q30) else t1
            row = {"symbol": sym, "start": t0, "peak": t1, "gain": float(np.expm1(g))}
            firsts = {}
            for d in det_ids:
                tt = idx[E[d].to_numpy() & (idx >= t0) & (idx <= t1)]
                firsts[d] = int(tt[0]) if len(tt) else None
                row[f"{d}_early"] = bool(len(tt) and tt[0] <= t30)
            tt = up_push[(up_push >= t0) & (up_push <= t1)]
            firsts["up_push"] = int(tt[0]) if len(tt) else None
            cand = [t for t in firsts.values() if t is not None]
            firsts["combined"] = min(cand) if cand else None
            for k in ("up_push", "combined"):
                ta = firsts[k]
                row[f"{k}_lag_h"] = np.nan if ta is None else (ta - t0) / HOUR
                row[f"{k}_done"] = np.nan if ta is None else (x.get(ta) - x0) / (x1 - x0)
            rally_rows.append(row)

    out = Path(args.out)
    out.mkdir(exist_ok=True)
    EV = pd.concat(events, ignore_index=True) if events else pd.DataFrame()
    RR = pd.DataFrame(rally_rows)
    PH = pd.concat(phase_parts).groupby("detector").sum()
    base = PH.loc["__base__"]
    EV.assign(ts_utc=pd.to_datetime(EV.ts, unit="ms")).to_csv(out / "early_events.csv", index=False, encoding="utf-8-sig")
    RR.to_csv(out / "early_rallies.csv", index=False, encoding="utf-8-sig")

    pct = lambda v, d=0: "—" if v is None or pd.isna(v) else f"{v * 100:+.{d}f}%" if d else f"{v * 100:.0f}%"
    sp = lambda v: "—" if v is None or pd.isna(v) else f"{v * 100:+.2f}%"
    tt = lambda v: "—" if v is None or pd.isna(v) else f"{v:+.1f}"
    days = coin_hours / 24
    md = [f"# 早期检测的历史验证(币安,{RR.symbol.nunique()} 个币,{len(RR)} 段大涨)\n",
          f"大涨 = 相对 BTC 超额 zigzag 涨幅 ≥ {args.min_gain:.0%};事件同一币同一检测至少间隔 72 小时;"
          "超额 = 该币剔除 BTC beta 后的收益减去同一时刻全部币的平均值。\n",
          "## 1. 触发之后的超额收益\n",
          "| 检测 | 持有期 | 事件数 | 平均 | 中位 | 跑赢全市场比例 | t(NW) | 前半段 / 后半段 |", "|---|---|---|---|---|---|---|---|"]
    for d in early.DETECTORS:
        sub = EV[EV.detector == d.id] if len(EV) else EV
        for h in HORIZONS:
            s = event_stats(sub, h) if len(sub) else {"n": 0}
            if s["n"] < 10:
                md.append(f"| {d.name}({d.id}) | {h}h | {s['n']} | — | — | — | — | — |")
                continue
            md.append(f"| {d.name}({d.id}) | {h}h | {s['n']} | {sp(s['mean'])} | {sp(s['median'])} | {pct(s['hit'])} | "
                      f"{tt(s['t'])} | {sp(s['first'])} / {sp(s['second'])} |")
    md += ["", "## 2. 在大涨里的时机\n",
           "倾向 = 触发小时落在该阶段的比例 ÷ 随机一个小时落在该阶段的比例;初段命中 = 这段大涨的初段(最低点到涨幅走完 30%)里它至少响过一次。\n",
           "精确度 = 去重后的每次触发里,当时正处在某段大涨初段的比例(其余都是没有接着大涨的\"假启动\")。\n",
           "| 检测 | 触发频率(每币每天小时数) | 启动前 | 初段 | 后段 | 初段命中 | 精确度 | 去重后每币每月次数 |",
           "|---|---|---|---|---|---|---|---|"]
    for d in early.DETECTORS + [None]:
        k = d.id if d else "__up_push__"
        r = PH.loc[k]
        lift = lambda ph: (r[ph] / max(r["n"], 1)) / (base[ph] / base["n"]) if r["n"] else np.nan
        hit = RR[f"{k}_early"].mean() if d else (RR["up_push_done"] <= 0.3).mean()
        name = f"{d.name}({d.id})" if d else "对照:现有\"像在涨\"的推送"
        ev = EV[EV.detector == k]
        md.append(f"| {name} | {r['n'] / days:.2f} | {lift('pre'):.2f} | {lift('early'):.2f} | {lift('late'):.2f} | {pct(hit)} | "
                  f"{pct(ev.in_early.mean())} | {len(ev) / days * 30:.1f} |")
    md.append(f"\n参照:随机一个小时正处在某段大涨初段的比例 {base['early'] / base['n']:.0%}。")
    md += ["", "## 3. 和现有推送合起来\n",
           "比例都以全部大涨为分母(含漏报)。\n",
           "| 口径 | 第一次提醒(中位) | 已走完(中位,四分位) | 前 30% 内提醒 | 过半才提醒 | 整段漏报 |", "|---|---|---|---|---|---|"]
    for k, nm in (("up_push", "现有\"像在涨\"的推送"), ("combined", "现有推送 + 全部早期检测(任一先响)")):
        c = RR[f"{k}_done"]
        md.append(f"| {nm} | {RR[f'{k}_lag_h'].median():+.0f}h | {pct(c.median())}({pct(c.quantile(.25))}~{pct(c.quantile(.75))}) | "
                  f"{pct((c <= .3).mean())} | {pct((c > .5).mean())} | {pct(c.isna().mean())} |")
    extra = sum(PH.loc[d.id, "n"] for d in early.DETECTORS) / days
    md += ["", f"代价:全部早期检测合计每币每天约触发 {extra:.2f} 个小时(连续触发的小时去重前;实际推送还要再加冷却)。\n",
           "## 怎么读\n",
           "- 第 1 节看有没有\"预测力\":超额要为正、t ≥ 2.5 左右、前后两半方向一致才算站得住;多个检测 × 多个持有期一起看,单个 t≈2 很可能是碰巧。",
           "- 第 2 节看\"时机\":初段倾向明显 > 1、后段不高,才是真正偏早的检测。",
           "- 第 3 节看实际能提前多少;即使提前了,如果第 1 节没有超额,也只是\"早响\",不代表响了以后会涨。",
           "- 这些是历史检验(币安数据);云端实盘(OKX)会把每次触发留档,到期后用 72h / 1 周 / 2 周真实结果再核对一遍。"]
    p = out / "early_study.md"
    p.write_text("\n".join(md), encoding="utf-8")
    print(p.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
