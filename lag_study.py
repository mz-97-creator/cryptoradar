"""滞后量化:一段大涨开始之后,监控系统多久才第一次报警?报警时这段涨幅已经走完了多少?

做法:
  1. 找"大涨":对每个币的价格(默认相对 BTC 的超额,即 log(币价) - log(BTC 价))做 zigzag 拐点分段,
     从低点到高点涨幅 ≥ --min-gain(默认 20%)的上涨段就是一段大涨;回撤超过 --reversal(默认 10%)才算这段结束。
     低点时刻记为"启动",高点时刻记为"见顶"。
  2. 回放报警:逐小时用和云端完全相同的规则、阈值、权重、推送门槛(自选 ≥ watchlist_min_score,其余 ≥ min_score_to_push)
     和冷却期(同一币同一规则 cooldown_hours 内不重复)算出"推送"时点;另外单独记"任意规则触发"(不看门槛)。
  3. 每段大涨取启动(最低点)到见顶之间的第一次推送 / 触发,给出:
       滞后小时 = 报警时刻 - 启动时刻
       已走完比例 = (报警时价格 - 启动价) / (见顶价 - 启动价),0% = 在最低点报警,100% = 在最高点才报警
       剩余涨幅 = 从报警时刻到见顶还能涨多少
     以及最先响的是哪条规则。整段上涨里一次都没报警的记为"漏报"。
     启动前的报警单独统计,并和"随便挑一个 pre-h 小时窗口里有报警"的概率比较:系统报警频繁时,启动前碰巧有报警很常见。
  4. 每条规则在大涨各阶段(启动前 / 初段 / 后段)的触发倾向,和随机一个小时相比(提升倍数 > 1 才说明偏爱这个阶段)。

局限(报告里也会写):逐小时回放用的是收盘 K 线,云端每 15 分钟扫一次、会看到未收盘的 K 线,实际报警可能早几十分钟;
大涨是事后才能确定的(用到了未来的高点),这里只用来衡量报警时机,不能当成交易信号的回测。

用法:
  python lag_study.py --archive data/archive.csv.gz        云端 data 分支的样本库(OKX 实时特征,约 180 天)
  python lag_study.py --db                                本地数据库(币安历史,先用 backfill_vision.py 回填)
  python lag_study.py --db --start 2025-01-01 --min-gain 0.3 --symbols AAVE,PUMP
结果:reports/lag_study_<来源>.md、reports/lag_rallies_<来源>.csv、reports/lag_rules_<来源>.csv
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from cryptoradar.config import load_config
from cryptoradar.signals import RULES_BY_ID, apply_weights, evaluate_frame, merged_thresholds

log = logging.getLogger("lag")
HOUR = 3_600_000


# ------------------------------------------------------------------ 数据
def base_symbol(s: str) -> str:
    s = s.upper().removesuffix("USDT")
    return s[4:] if s.startswith("1000") and len(s) > 4 else s


def frames_from_archive(path: str) -> dict[str, pd.DataFrame]:
    a = pd.read_csv(path)
    out = {}
    for sym, g in a.groupby("symbol"):
        g = g.drop_duplicates("ts", keep="last").set_index("ts").sort_index()
        out[base_symbol(sym)] = g.reindex(np.arange(g.index.min(), g.index.max() + HOUR, HOUR))
    return out


def frames_from_db(cfg: dict, symbols: list[str] | None) -> dict[str, pd.DataFrame]:
    import tune
    from cryptoradar.storage import Store
    th = merged_thresholds(cfg["signals"].get("thresholds"))
    syms = [s if s.endswith("USDT") else f"{s}USDT" for s in symbols] if symbols else None
    out = {}
    for f in tune.load_frames(Store(cfg["storage"]["db_path"]), th, syms):
        f = f[~f.index.duplicated(keep="last")].sort_index()
        out[base_symbol(f["symbol"].iloc[0])] = f.reindex(np.arange(f.index.min(), f.index.max() + HOUR, HOUR))
    return out


# ------------------------------------------------------------------ 大涨识别
def zigzag_rallies(x: pd.Series, min_gain: float, reversal: float) -> list[tuple[int, int, float]]:
    """x:对数价格(按小时)。返回 [(启动时刻, 见顶时刻, 对数涨幅)],只要上涨段且涨幅 ≥ log(1+min_gain)。
    拐点规则:从当前极值反向走超过 log(1+reversal) 才确认一个拐点。"""
    x = x.dropna()
    if len(x) < 3:
        return []
    ts, v = x.index.to_numpy(), x.to_numpy()
    rev, need = np.log1p(reversal), np.log1p(min_gain)
    out, mode, lo_i, hi_i = [], None, 0, 0     # mode: None 未定 / "up" 上涨段里找高点 / "down" 下跌段里找低点
    for i in range(1, len(v)):
        if mode != "down" and v[i] > v[hi_i]:
            hi_i = i
        if mode != "up" and v[i] < v[lo_i]:
            lo_i = i
        if mode is None:
            if v[hi_i] - v[lo_i] >= rev:
                mode = "up" if hi_i > lo_i else "down"
        elif mode == "up" and v[hi_i] - v[i] >= rev:          # 从高点回撤够了:确认见顶
            if v[hi_i] - v[lo_i] >= need:
                out.append((int(ts[lo_i]), int(ts[hi_i]), float(v[hi_i] - v[lo_i])))
            mode, lo_i = "down", i
        elif mode == "down" and v[i] - v[lo_i] >= rev:        # 从低点反弹够了:确认见底,新的上涨段开始
            mode, hi_i = "up", i
    if mode == "up" and v[hi_i] - v[lo_i] >= need:            # 数据末尾还在涨:见顶时刻取目前最高点
        out.append((int(ts[lo_i]), int(ts[hi_i]), float(v[hi_i] - v[lo_i])))
    return out


# ------------------------------------------------------------------ 报警回放
def replay(f: pd.DataFrame, th: dict, rules, need: float, cooldown_h: float) -> tuple[pd.DataFrame, pd.Series]:
    """逐小时回放。返回 (每小时各规则是否触发的表, 推送时刻 -> 触发的规则列表)。
    推送条件与 cloud_run 一致:本小时触发的规则里至少一条不在冷却期,且权重合计 ≥ need;推送后这些规则全部进入冷却。"""
    hits = evaluate_frame(f, th)[[r.id for r in rules]]
    w = np.array([r.weight for r in rules])
    ids = np.array([r.id for r in rules])
    arr, ts = hits.to_numpy(), hits.index.to_numpy()
    last = {r: -np.inf for r in ids}
    cd = cooldown_h * HOUR
    pushes = {}
    for k in np.flatnonzero(arr.any(axis=1)):
        on = arr[k]
        t = ts[k]
        fired = ids[on]
        if w[on].sum() >= need and any(t - last[r] > cd for r in fired):
            pushes[int(t)] = list(fired)
            for r in fired:
                last[r] = t
    return hits, pd.Series(pushes, dtype=object)


def pre_base(alert: pd.Series, in_range: np.ndarray, pre_h: int) -> float:
    """随机挑一个小时,它之前 pre_h 小时内有报警的概率:用来判断"启动前有报警"是不是碰巧。"""
    prior = alert.astype(float).rolling(pre_h, min_periods=1).sum().shift(1).fillna(0).to_numpy() > 0
    return float(prior[in_range].mean()) if in_range.any() else np.nan


UP_RULES = ("OI_TREND", "SHORT_COVER", "LONG_CROWD")      # 规则本身就要求价格在涨


def up_flavored(f: pd.DataFrame, hits: pd.DataFrame) -> pd.Series:
    """这一小时是否有"读起来像在涨"的规则触发:上面三条,或正向的 RESID / SHOCK(这两条按绝对值触发,下跌也会响)。"""
    up = hits[[c for c in UP_RULES if c in hits]].any(axis=1)
    if "RESID" in hits:
        up |= hits["RESID"] & (f["resid_24h_z"] > 0)
    if "SHOCK" in hits:
        up |= hits["SHOCK"] & (f["ret_1h_z"] > 0)
    return up


def measure(sym, x, rallies, hits, pushes, up, pre_h, watch, bases):
    """每段大涨:启动前 pre_h 小时内有没有报警;启动(最低点)之后第一次报警的时间、已走完比例、剩余涨幅、规则。
    启动前的报警单独算:那时价格往往还在下跌,不能算"提前抓到"这段上涨。"""
    rows = []
    any_hit = hits.any(axis=1)
    fire_t = any_hit.index[any_hit.to_numpy()]
    up_push_t = pushes.index[up.reindex(pushes.index).fillna(False).to_numpy(dtype=bool)]
    for t0, t1, g in rallies:
        x0, x1 = x.get(t0), x.get(t1)
        row = {"symbol": sym, "watch": watch, "start": t0, "peak": t1, "gain": float(np.expm1(g)),
               "days": (t1 - t0) / HOUR / 24}
        for tag, times in (("push", pushes.index), ("uppush", up_push_t), ("fire", fire_t)):
            pre = times[(times >= t0 - pre_h * HOUR) & (times < t0)]
            row[f"{tag}_pre"] = bool(len(pre))
            row[f"{tag}_pre_base"] = bases[tag]
            row[f"{tag}_pre_rules"] = ";".join(pushes[int(pre[-1])]) if tag != "fire" and len(pre) else ""
            post = times[(times >= t0) & (times <= t1)]
            if len(post) == 0:
                row.update({f"{tag}_lag_h": np.nan, f"{tag}_done": np.nan, f"{tag}_left": np.nan, f"{tag}_rules": ""})
                continue
            ta = int(post[0])
            xa = x.get(ta, np.nan)
            fired = pushes[ta] if tag != "fire" else list(hits.columns[hits.loc[ta].to_numpy()])
            row.update({f"{tag}_lag_h": (ta - t0) / HOUR, f"{tag}_done": (xa - x0) / (x1 - x0),
                        f"{tag}_left": float(np.expm1(x1 - xa)), f"{tag}_rules": ";".join(fired)})
        seg = x.loc[t0:t1]               # 参考:涨幅走完 30% / 50% 用了多少小时
        for q in (0.3, 0.5):
            hit = seg.index[(seg - x0).to_numpy() >= q * (x1 - x0)]
            row[f"t{int(q * 100)}_h"] = (int(hit[0]) - t0) / HOUR if len(hit) else np.nan
        rows.append(row)
    return rows


def phase_masks(index: np.ndarray, x: pd.Series, rallies, pre_h: int) -> dict[str, np.ndarray]:
    """每个小时属于哪个阶段:pre 启动前 pre_h 小时 / early 启动到涨幅走完 30% / late 涨幅过半到见顶。"""
    m = {k: np.zeros(len(index), dtype=bool) for k in ("pre", "early", "late")}
    for t0, t1, _ in rallies:
        seg = x.loc[t0:t1]
        x0, x1 = seg.iloc[0], x.get(t1)
        q30 = seg.index[(seg - x0).to_numpy() >= 0.3 * (x1 - x0)]
        q50 = seg.index[(seg - x0).to_numpy() > 0.5 * (x1 - x0)]
        m["pre"] |= (index >= t0 - pre_h * HOUR) & (index < t0)
        m["early"] |= (index >= t0) & (index <= (int(q30[0]) if len(q30) else t1))
        if len(q50):
            m["late"] |= (index >= int(q50[0])) & (index <= t1)
    return m


# ------------------------------------------------------------------ 汇总与报告
def pct(v, d=0):
    return "—" if v is None or pd.isna(v) else f"{v * 100:.{d}f}%"


def hours(v):
    return "—" if v is None or pd.isna(v) else f"{v:+.0f}h"


def summarize(R: pd.DataFrame, tag: str) -> dict:
    caught = R[R[f"{tag}_lag_h"].notna()]
    q = lambda c, v: caught[c].quantile(v) if len(caught) else np.nan
    return {"n": len(R), "pre": R[f"{tag}_pre"].mean(), "pre_base": R[f"{tag}_pre_base"].mean(),
            "miss": 1 - len(caught) / len(R) if len(R) else np.nan,
            "lag_med": q(f"{tag}_lag_h", .5), "done_med": q(f"{tag}_done", .5),
            "done_q25": q(f"{tag}_done", .25), "done_q75": q(f"{tag}_done", .75), "left_med": q(f"{tag}_left", .5),
            "in_first30": (caught[f"{tag}_done"] <= 0.3).mean() if len(caught) else np.nan,
            "after_half": (caught[f"{tag}_done"] > 0.5).mean() if len(caught) else np.nan,
            "t30_med": R["t30_h"].median(), "t50_med": R["t50_h"].median()}


def write_report(out: Path, src: str, args, R: pd.DataFrame, rule_tab: pd.DataFrame, span: tuple, n_coins: int,
                 push_count: int, coin_days: float, focus: list[str]) -> Path:
    def sum_rows(df):
        rows = []
        for label, tag in (("推送(真正会发给你的)", "push"), ("推送且读起来像在涨", "uppush"),
                           ("任意规则触发(不看推送门槛)", "fire")):
            s = summarize(df, tag)
            rows.append(f"| {label} | {s['n']} | {pct(s['pre'])} / {pct(s['pre_base'])} | {pct(s['miss'])} | "
                        f"{hours(s['lag_med'])} | {pct(s['done_med'])}({pct(s['done_q25'])}~{pct(s['done_q75'])}) | "
                        f"{pct(s['in_first30'])} | {pct(s['after_half'])} | {pct(s['left_med'])} |")
        s = summarize(df, "push")
        rows.append(f"\n参考:这些大涨从最低点到涨幅走完 30% 中位用了 {hours(s['t30_med'])},走完一半用了 {hours(s['t50_med'])}。\n")
        return rows

    head = [f"| 口径 | 大涨段数 | 启动前 {args.pre_h}h 有报警 / 随机基准 | 启动后漏报 | 启动后第一次报警(中位) | "
            "报警时已走完(中位,四分位) | 前 30% 内报警 | 过半才报警 | 报警后剩余涨幅中位 |",
            "|---|---|---|---|---|---|---|---|---|"]
    md = [f"# 滞后量化:大涨启动后多久报警({src})\n",
          f"数据:{n_coins} 个币,{pd.to_datetime(span[0], unit='ms'):%Y-%m-%d} ~ {pd.to_datetime(span[1], unit='ms'):%Y-%m-%d};"
          f"价格口径:{'相对 BTC 超额' if not args.raw else '原始价格'};大涨 = zigzag 上涨段涨幅 ≥ {args.min_gain:.0%}"
          f"(回撤 ≥ {args.reversal:.0%} 才算结束)。\n",
          f"回放期间共推送 {push_count} 次(约每币每天 {push_count / max(coin_days, 1):.2f} 次)。\n",
          "## 全部大涨\n", *head, *sum_rows(R), ""]
    for label, sub in (("自选币", R[R.watch]), ("非自选币", R[~R.watch])):
        if len(sub):
            md += [f"## {label}\n", *head, *sum_rows(sub), ""]
    big = R[R.gain >= 2 * args.min_gain]
    if len(big):
        md += [f"## 特大涨幅(≥ {2 * args.min_gain:.0%})\n", *head, *sum_rows(big), ""]
    first = R["push_rules"].replace("", np.nan).dropna().str.split(";").explode().value_counts()
    if len(first):
        md += ["## 启动后第一次推送时触发的规则(按出现次数)\n", "| 规则 | 次数 |", "|---|---|"]
        md += [f"| {RULES_BY_ID[k].name}({k}) | {v} |" for k, v in first.items()] + [""]
    md += ["## 各规则在大涨各阶段的触发倾向\n",
           f"每列 = 该规则的触发小时里落在这一阶段的比例 ÷ 随机一个小时落在这一阶段的比例(提升倍数)。"
           f"启动前 = 最低点之前 {args.pre_h} 小时;初段 = 最低点到涨幅走完 30%;后段 = 涨幅过半到见顶。"
           "倍数 > 1 表示这条规则偏爱这个阶段;一条真正能提前的规则应该在\"启动前\"或\"初段\"明显 > 1。\n",
           "| 规则 | 触发小时数 | 启动前 | 初段 | 后段 |", "|---|---|---|---|---|"]
    for r in rule_tab.itertuples():
        md.append(f"| {RULES_BY_ID[r.rule].name}({r.rule}) | {r.n} | {r.lift_pre:.2f} | {r.lift_early:.2f} | {r.lift_late:.2f} |")
    md.append("")
    if focus:
        F = R[R.symbol.isin(focus)].sort_values("start")
        md += [f"## 个案:{'、'.join(focus)}\n",
               f"| 币 | 启动(UTC) | 见顶 | 涨幅 | 启动前 {args.pre_h}h 推送 | 启动后第一次推送 | 已走完 | 推送规则 | 第一次\"像在涨\"的推送 | 已走完 | 剩余 | 规则 |",
               "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in F.itertuples():
            md.append(f"| {r.symbol} | {pd.to_datetime(r.start, unit='ms'):%Y-%m-%d %H:%M} | {pd.to_datetime(r.peak, unit='ms'):%m-%d %H:%M} | "
                      f"{pct(r.gain)} | {r.push_pre_rules or '无'} | {hours(r.push_lag_h)} | {pct(r.push_done)} | {r.push_rules or '漏报'} | "
                      f"{hours(r.uppush_lag_h)} | {pct(r.uppush_done)} | {pct(r.uppush_left)} | {r.uppush_rules or '漏报'} |")
        md.append("")
    md += ["## 怎么读\n",
           "- 启动 = 这段上涨的最低点(事后才知道)。启动后第一次报警是 +0h 已经是最好的情况。",
           "- 启动前有报警 / 随机基准:系统报警很频繁,随便哪 48 小时里都常有报警。两者接近,说明启动前的报警只是碰巧,"
           "不是提前看出了这段上涨(那时价格往往还在跌)。",
           "- 已走完:启动后第一次报警时,这段涨幅已经涨掉的比例。中位数过半,说明多数时候是涨起来以后才报警。",
           "- 推送且读起来像在涨:推送里含 OI_TREND / SHORT_COVER / LONG_CROWD,或正向的 RESID / SHOCK。"
           "RESID、SHOCK、VOL 等按绝对值触发,在最低点响的往往是在报\"刚刚急跌\",不是在说\"要涨了\"。",
           "- 推送 vs 任意触发:差得多说明问题在推送门槛(权重合计、冷却);差得少说明规则本身就响得晚。",
           "- 局限:逐小时回放用收盘 K 线,云端每 15 分钟扫一次、看得到未收盘的 K 线,实际报警最多早 1 小时左右;"
           "大涨是事后划分的,只用于衡量报警时机,不是交易回测。"]
    p = out / f"lag_study_{src}.md"
    p.write_text("\n".join(md), encoding="utf-8")
    return p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")),
                    help="读 storage.db_path(--db 时)")
    ap.add_argument("--rules-config", default=str(Path(__file__).with_name("cloud_config.yaml")),
                    help="规则阈值、权重、推送门槛、冷却和自选名单从这里读(默认与云端一致)")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--archive", help="云端 data 分支的 archive.csv.gz")
    src.add_argument("--db", action="store_true", help="用本地数据库(币安历史)")
    ap.add_argument("--symbols", help="逗号分隔;默认全部")
    ap.add_argument("--start", help="只看这天之后启动的大涨,如 2025-03-01(前面的数据仍用于计算特征)")
    ap.add_argument("--min-gain", type=float, default=0.20)
    ap.add_argument("--reversal", type=float, default=0.10)
    ap.add_argument("--pre-h", type=int, default=48, help="启动前多少小时内的报警算提前")
    ap.add_argument("--raw", action="store_true", help="用原始价格而不是相对 BTC 的超额来划分大涨")
    ap.add_argument("--focus", default="AAVE,PUMP", help="报告里单独列出的币")
    ap.add_argument("--out", default=str(Path(__file__).with_name("reports")))
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    rc = load_config(args.rules_config)
    sc = rc["signals"]
    th = merged_thresholds(sc.get("thresholds"))
    rules = apply_weights(sc.get("rule_weights"))
    watch = {s.upper() for s in rc["universe"].get("watchlist", [])}
    syms = [s.strip().upper() for s in args.symbols.split(",")] if args.symbols else None
    if args.archive:
        frames, tag = frames_from_archive(args.archive), "live"
        if syms:
            frames = {k: v for k, v in frames.items() if k in syms}
    else:
        frames, tag = frames_from_db(load_config(args.config), syms), "binance"
    frames.pop("BTC", None)
    start = int(pd.Timestamp(args.start, tz="UTC").timestamp() * 1000) if args.start else None

    all_rows, rule_parts, push_count, coin_hours, span = [], [], 0, 0, [np.inf, -np.inf]
    for sym, f in sorted(frames.items()):
        if f["close"].notna().sum() < 24 * 14:
            continue
        lc = np.log(f["close"])
        x = lc if args.raw else lc - f["_btc_lc"]
        need = sc.get("watchlist_min_score", 1.0) if sym in watch else sc.get("min_score_to_push", 2.5)
        hits, pushes = replay(f, th, rules, need, float(sc.get("cooldown_hours", 6)))
        rallies = zigzag_rallies(x, args.min_gain, args.reversal)
        valid = f["resid_24h_z"].notna()          # 特征还没算出来(预热期)的时段不评
        first_ok = int(valid.index[valid.to_numpy()][0]) if valid.any() else None
        if first_ok is None:
            continue
        lo_ok = max(first_ok + args.pre_h * HOUR, start or -np.inf)
        rallies = [r for r in rallies if r[0] >= lo_ok]
        idx = hits.index.to_numpy()
        in_range = (idx >= lo_ok) & valid.to_numpy()
        push_ind = pd.Series(np.isin(idx, pushes.index.to_numpy()), index=hits.index)
        up = up_flavored(f, hits)
        uppush_ind = push_ind & up
        bases = {"push": pre_base(push_ind, in_range, args.pre_h), "uppush": pre_base(uppush_ind, in_range, args.pre_h),
                 "fire": pre_base(hits.any(axis=1), in_range, args.pre_h)}
        all_rows += measure(sym, x, rallies, hits, pushes, up, args.pre_h, sym in watch, bases)
        ph = phase_masks(idx, x, rallies, args.pre_h)
        H = hits.to_numpy()[in_range]
        cols = np.repeat(hits.columns.to_numpy()[None, :], H.shape[0], 0)[H]
        rule_parts.append(pd.DataFrame({"rule": cols, **{k: np.repeat(v[in_range][:, None], H.shape[1], 1)[H]
                                                         for k, v in ph.items()}}))
        rule_parts.append(pd.DataFrame({"rule": "__base__", **{k: v[in_range] for k, v in ph.items()}}))
        push_count += int(((pushes.index >= lo_ok)).sum())
        coin_hours += int(in_range.sum())
        span = [min(span[0], lo_ok), max(span[1], int(idx[-1]))]

    if not all_rows:
        raise SystemExit("没有找到满足条件的大涨,试试降低 --min-gain 或加长数据")
    R = pd.DataFrame(all_rows)
    RP = pd.concat(rule_parts, ignore_index=True)
    base = RP[RP.rule == "__base__"][["pre", "early", "late"]].mean()
    rule_tab = (RP[RP.rule != "__base__"].groupby("rule")
                .agg(n=("pre", "size"), pre=("pre", "mean"), early=("early", "mean"), late=("late", "mean")).reset_index())
    for k in ("pre", "early", "late"):
        rule_tab[f"lift_{k}"] = rule_tab[k] / base[k]
    rule_tab = rule_tab.sort_values("lift_early", ascending=False)

    out = Path(args.out)
    out.mkdir(exist_ok=True)
    R.assign(start_utc=pd.to_datetime(R.start, unit="ms"), peak_utc=pd.to_datetime(R.peak, unit="ms")) \
        .to_csv(out / f"lag_rallies_{tag}.csv", index=False, encoding="utf-8-sig")
    rule_tab.to_csv(out / f"lag_rules_{tag}.csv", index=False, encoding="utf-8-sig")
    focus = [s.strip().upper() for s in args.focus.split(",") if s.strip()]
    p = write_report(out, tag, args, R, rule_tab, span, R.symbol.nunique(), push_count, coin_hours / 24, focus)
    print(p.read_text(encoding="utf-8"))
    print(f"\n基准:随机一小时落在 启动前 {base['pre']:.1%} / 初段 {base['early']:.1%} / 后段 {base['late']:.1%}")


if __name__ == "__main__":
    main()
