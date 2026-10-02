"""早期检测(实验):专门抓行情启动阶段,和 signals.py 里偏"确认型"的规则互补。

lag_study.py 的结论:现有规则里 OI_TREND / RESID / VOL 多在涨幅过半之后才响,能早响的又多是按绝对值触发、方向不明的。
这里四类检测都只看"向上",并刻意避开已经被现有规则覆盖的大幅异动:

  E_RESID6    短窗口超额:6h 剔除 BTC beta 后的收益相对自身 30 天分布 z ≥ 2,且 24h 的 resid_24h_z 还没到 RESID 阈值
  E_BREAKOUT  慢涨突破:相对 BTC 的价格创 N 日新高,相对强弱(7 日超额)排名在全市场前 30%,且最近 3 天排名明显上升;
              用的是绝对位置和排名,不是相对自身波动的 z,慢慢涨上去的也能抓到;同时 24h 没有急拉(那是 RESID 的事)
  E_OI_BUILD  持仓逐步累积:72h 持仓变化相对自身 z ≥ 1.5,最近三个 24h 段持仓都在增加,但 24h 没有单次激增,价格还没大动
  E_SQUEEZE   空头挤压前兆:价格在涨(24h 超额 z ≥ 1 且 24h 收益为正),资金费率仍为负(空头在付费、还没认输)
  S_SPOT_LED  现货买盘主导:现货主动买入占比明显高于平时且还在上升,合约成交占比不高,价格还没大动
  F_REV_UP    回购/协议收入加速:持币人收入(没有则看协议收入)最近 7 日 ≥ 之前 4 周周均的 1.3 倍
  F_FEES_UP   协议费用(使用量)加速:同上,看总费用
  F_TVL_UP    TVL 7 日增长 ≥ 10%
  基本面(F_*)来自 cryptoradar/fundamentals.py(DefiLlama,日频),现货(S_*)来自 OKX 现货主动买卖量 / 币安现货 K 线。

这些检测在验证有效之前默认不推送,只写进 status.md / signals.json 并留档,到期用 72h / 1 周 / 2 周的实盘结果核对。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

HOUR = 3_600_000

DEFAULTS = {
    "resid6_z": 2.0,          # E_RESID6:6h 超额 z
    "resid24_cap": 2.5,       # E_RESID6 / E_BREAKOUT:24h 超额 z 低于它(还没被 RESID 抓到)
    "breakout_days": 20,      # E_BREAKOUT:创多少日新高
    "rs_min": 0.70,           # E_BREAKOUT:7 日超额收益的全市场排名(0~1)
    "rs_rise": 0.15,          # E_BREAKOUT:排名比 3 天前至少上升多少
    "oi72_z": 1.5,            # E_OI_BUILD:72h 持仓变化 z
    "oi24_cap": 2.0,          # E_OI_BUILD:24h 持仓 z 低于它(不是单次激增)
    "oi_build_ret_cap": 0.05, # E_OI_BUILD:72h 超额收益绝对值低于它(价格还没大动)
    "squeeze_resid_z": 1.0,   # E_SQUEEZE:24h 超额 z
    "spot_buy_z": 1.5,        # S_SPOT_LED:现货主动买入占比 z
    "spot_buy_rise": 0.5,     # S_SPOT_LED:比 12 小时前上升
    "lev_share_max": 0.0,     # S_SPOT_LED:合约/现货成交量 z 不高于它(现货占比高于平时)
    "f_ratio": 1.3,           # F_*:最近 7 日 / 之前 28 日周均值
    "f_min_usd_7d": 50_000,   # F_REV_UP / F_FEES_UP:最近 7 日至少这么多美元,太小的不看
    "f_tvl_7d": 0.10,         # F_TVL_UP:TVL 7 日对数变化
}


@dataclass(frozen=True)
class Detector:
    id: str
    name: str
    fn: Callable[[pd.DataFrame, dict], pd.Series]


def _c(f: pd.DataFrame, col: str) -> pd.Series:
    return f[col] if col in f else pd.Series(np.nan, index=f.index)


DETECTORS = [
    Detector("E_RESID6", "短窗口超额走强",
             lambda f, t: (_c(f, "resid_6h_z") >= t["resid6_z"]) & (_c(f, "resid_6h") > 0)
             & (_c(f, "resid_24h_z") < t["resid24_cap"])),
    Detector("E_BREAKOUT", "慢涨突破(创新高+相对强弱上升)",
             lambda f, t: (_c(f, "xs_high") > 0) & (_c(f, "rs_rank") >= t["rs_min"])
             & (_c(f, "rs_rank_chg") >= t["rs_rise"]) & (_c(f, "resid_24h_z") < t["resid24_cap"])),
    Detector("E_OI_BUILD", "持仓逐步累积",
             lambda f, t: (_c(f, "oi_72h_z") >= t["oi72_z"]) & (_c(f, "oi_up_days") >= 3)
             & (_c(f, "oi_z") < t["oi24_cap"]) & (_c(f, "resid_72h").abs() < t["oi_build_ret_cap"])),
    Detector("E_SQUEEZE", "价格上涨但费率仍为负",
             lambda f, t: (_c(f, "funding") < 0) & (_c(f, "resid_24h_z") >= t["squeeze_resid_z"])
             & (_c(f, "ret_24h") > 0)),
]


def _rev_up(f: pd.DataFrame, t: dict) -> pd.Series:
    """持币人收入(回购/分红)加速;没有持币人收入数据的协议看协议收入。"""
    h7, r7 = _c(f, "f_hrev_7d"), _c(f, "f_rev_7d")
    has_h = h7 >= t["f_min_usd_7d"]
    up_h = has_h & (_c(f, "f_hrev_ratio") >= t["f_ratio"])
    up_r = ~has_h & (r7 >= t["f_min_usd_7d"]) & (_c(f, "f_rev_ratio") >= t["f_ratio"])
    return up_h | up_r


DETECTORS += [
    Detector("S_SPOT_LED", "现货买盘主导走强",
             lambda f, t: (_c(f, "spot_buy_z") >= t["spot_buy_z"]) & (_c(f, "spot_buy_dz12") >= t["spot_buy_rise"])
             & (_c(f, "lev_share_z") <= t["lev_share_max"]) & (_c(f, "resid_24h_z") < t["resid24_cap"])
             & (_c(f, "resid_24h_z") > -1)),
    Detector("F_REV_UP", "回购/协议收入加速", _rev_up),
    Detector("F_FEES_UP", "协议费用(使用量)加速",
             lambda f, t: (_c(f, "f_fees_7d") >= t["f_min_usd_7d"]) & (_c(f, "f_fees_ratio") >= t["f_ratio"])),
    Detector("F_TVL_UP", "TVL 明显增长", lambda f, t: _c(f, "f_tvl_chg_7d") >= t["f_tvl_7d"]),
]
DETECTORS_BY_ID = {d.id: d for d in DETECTORS}


def thresholds(cfg: dict | None) -> dict:
    return {**DEFAULTS, **(cfg or {})}


def add_cross_section(frames: dict[str, pd.DataFrame], breakout_days: int = DEFAULTS["breakout_days"],
                      exclude: tuple = ("BTC", "BTCUSDT")) -> dict[str, pd.DataFrame]:
    """需要全市场一起算的列:
    rs_rank      7 日超额收益(resid 7d,对数)在同一小时全部币里的排名(0~1,1 = 最强)
    rs_rank_chg  rs_rank 与 72 小时前之差
    xs_high      相对 BTC 的对数价格比过去 breakout_days 天(不含当前)最高点高出多少,> 0 即创新高"""
    n = breakout_days * 24
    r7 = {}
    for sym, f in frames.items():
        if sym in exclude or f.empty or "xs_lc" not in f:
            continue
        lc = np.log(f["close"].where(f["close"] > 0))
        r7[sym] = lc.diff(168) - f["beta"] * f["_btc_lc"].diff(168)
    rank = pd.DataFrame(r7).rank(axis=1, pct=True) if r7 else pd.DataFrame()
    out = {}
    for sym, f in frames.items():
        f = f.copy()
        if sym in rank:
            f["rs_rank"] = rank[sym].reindex(f.index)
            f["rs_rank_chg"] = f["rs_rank"] - f["rs_rank"].shift(72)
        if "xs_lc" in f:
            prev_max = f["xs_lc"].rolling(n, min_periods=n // 2).max().shift(1)
            f["xs_high"] = f["xs_lc"] - prev_max
        out[sym] = f
    return out


def evaluate_frame(f: pd.DataFrame, th: dict) -> pd.DataFrame:
    """每列一个检测的 bool 表。"""
    return pd.DataFrame({d.id: d.fn(f, th).fillna(False).astype(bool) for d in DETECTORS}, index=f.index)


def evaluate_last(f: pd.DataFrame, th: dict) -> list[Detector]:
    if f.empty:
        return []
    last = f.iloc[[-1]]
    return [d for d in DETECTORS if bool(d.fn(last, th).fillna(False).iloc[0])]


# ------------------------------------------------------------------ 云端:检测、留档、到期核对
HORIZONS = (72, 168, 336)
LOG_COLS = (["t_bar", "ts", "symbol", "watch", "detector", "price"]
            + [f"{k}{h}" for h in HORIZONS for k in ("resid", "mkt")])


def cloud_scan(frames: dict[str, pd.DataFrame], uni: list[dict], th: dict, now: int,
               last_fire: dict | None, cooldown_h: float = 24) -> tuple[list[dict], pd.DataFrame, dict]:
    """对本轮每个币的最新一行跑四类检测。
    返回 (本轮正在触发的列表, 新留档的行, 更新后的冷却记录)。同一币同一检测 cooldown_h 小时内只留档一次。"""
    info = {u["ccy"]: u for u in uni}
    frames = add_cross_section(frames, int(th["breakout_days"]))
    last = dict(last_fire or {})
    firing, rows = [], []
    for sym, f in frames.items():
        if sym in ("BTC", "BTCUSDT") or f.empty:
            continue
        hit = evaluate_last(f, th)
        if not hit:
            continue
        u, r = info.get(sym, {}), f.iloc[-1]
        firing.append({"symbol": sym, "rank": u.get("rank"), "watch": bool(u.get("watch")),
                       "detectors": [d.id for d in hit], "names": [d.name for d in hit],
                       "price": float(r["close"]) if pd.notna(r["close"]) else None})
        for d in hit:
            key = f"{sym}|{d.id}"
            if now - int(last.get(key, 0)) > cooldown_h * HOUR:
                rows.append({"t_bar": int(f.index[-1]), "ts": now, "symbol": sym, "watch": int(bool(u.get("watch"))),
                             "detector": d.id, "price": firing[-1]["price"]})
                last[key] = now
                firing[-1].setdefault("new", []).append(d.id)
    keep = now - 30 * 24 * HOUR
    return (sorted(firing, key=lambda x: (-x["watch"], -len(x["detectors"]))),
            pd.DataFrame(rows, columns=LOG_COLS), {k: v for k, v in last.items() if v >= keep})


def resolve(log: pd.DataFrame, combined: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """补上已到期的结果:resid{h} 该币剔除 BTC beta 后的超额(对数),mkt{h} 同一时刻全部币的平均值(基准)。
    已经有值的不重算,所以表可以永久累积。"""
    from .opportunity import resolve_log
    log = log.reindex(columns=LOG_COLS).copy()
    if log.empty:
        return log
    pend = log[log[[f"resid{h}" for h in HORIZONS]].isna().any(axis=1)]
    if pend.empty:
        return log
    pairs = pend[["t_bar", "symbol"]].drop_duplicates()
    got = resolve_log(pairs, combined, HORIZONS).set_index(["t_bar", "symbol"])
    syms = [s for s in combined if s not in ("BTC", "BTCUSDT")]
    allp = pd.DataFrame([(t, s) for t in pairs["t_bar"].unique() for s in syms], columns=["t_bar", "symbol"])
    mk = resolve_log(allp, combined, HORIZONS).groupby("t_bar")[[f"resid{h}" for h in HORIZONS]].mean()
    for h in HORIZONS:
        key = pd.MultiIndex.from_frame(log[["t_bar", "symbol"]])
        r = got[f"resid{h}"].reindex(key).to_numpy() if f"resid{h}" in got else np.full(len(log), np.nan)
        m = mk[f"resid{h}"].reindex(log["t_bar"]).to_numpy() if f"resid{h}" in mk else np.full(len(log), np.nan)
        ok = log[f"resid{h}"].isna().to_numpy() & np.isfinite(r) & np.isfinite(m)
        log.loc[ok, f"resid{h}"] = r[ok]
        log.loc[ok, f"mkt{h}"] = m[ok]
    return log


def _nw_t(x: pd.Series, lags: int) -> float | None:
    x = x.dropna().to_numpy()
    n = len(x)
    if n < 10:
        return None
    e = x - x.mean()
    var = np.mean(e * e)
    for l in range(1, min(lags, n - 1) + 1):
        var += 2 * (1 - l / (lags + 1)) * np.mean(e[l:] * e[:-l])
    return float(x.mean() / np.sqrt(max(var, 1e-18) / n))


def summary(log: pd.DataFrame | None, min_n: int = 30) -> dict:
    """按检测、按持有期汇总实盘成绩:超额 = resid - mkt(相对同一时刻全部币的平均)。
    同一天的触发先取平均再算 t(Newey-West,滞后 = 持有期天数),避免同一波行情里多个币重复计数。"""
    out = {}
    if log is None or log.empty:
        return out
    for d in DETECTORS:
        sub = log[log["detector"] == d.id]
        st = {"name": d.name, "total": int(len(sub))}
        for h in HORIZONS:
            ex = (sub[f"resid{h}"] - sub[f"mkt{h}"]).dropna()
            s = {"n": int(len(ex))}
            if len(ex):
                s.update({"mean": float(ex.mean()), "median": float(ex.median()), "hit": float((ex > 0).mean())})
            if len(ex) >= min_n:
                daily = ex.groupby(sub.loc[ex.index, "t_bar"] // (24 * HOUR)).mean()
                s["t"] = _nw_t(daily, max(3, h // 24))
            st[f"{h}h"] = s
        out[d.id] = st
    return out


def text(firing: list[dict], summ: dict, min_n: int = 30) -> list[str]:
    lines = []
    if firing:
        lines.append("**本轮触发**:" + ";".join(
            f"{x['symbol']}{'⭐' if x['watch'] else ''}({'、'.join(x['names'])})" for x in firing[:20])
            + (f" 等 {len(firing)} 个" if len(firing) > 20 else ""))
    else:
        lines.append("本轮没有币触发早期检测")
    pc = lambda v: "—" if v is None else f"{v * 100:+.1f}%"
    pp = lambda v: "—" if v is None else f"{v * 100:.0f}%"
    if summ:
        lines += ["", "| 检测 | 累计触发 | 72h 超额中位(跑赢比例,到期数,t) | 1 周 | 2 周 |", "|---|---|---|---|---|"]
        for d in DETECTORS:
            st = summ.get(d.id)
            if not st:
                continue
            cell = lambda s: ("—" if not s.get("n") else
                              f"{pc(s.get('median'))}({pp(s.get('hit'))},{s['n']}"
                              + (f",t={s['t']:+.1f}" if s.get("t") is not None else "") + ")"
                              + ("(样本少)" if s["n"] < min_n else ""))
            lines.append(f"| {d.name}({d.id}) | {st['total']} | {cell(st['72h'])} | {cell(st['168h'])} | {cell(st['336h'])} |")
    lines.append("> 实验功能,默认不推送。超额 = 相对同一时刻全部币平均的超额;t 按天聚合并做 Newey-West 校正。"
                 "历史检验(币安 2025-03~2026-10,见 studies/early_study.md):四类都能比现有推送更早响,"
                 "但触发后平均没有显著超额,需要靠这里的实盘结果继续判断。明细见 data 分支 early_log.csv.gz")
    return lines
