"""统一的时间检验工具,所有研究脚本共用,防止"回测里好看、实盘不行"。

1. 清洗边界(purge + embargo):标签要看未来 h 小时,训练样本的标签窗口不能伸进检验期,
   所以训练集只用 ts < 检验期开始 - h 的样本;检验期结束后再空出 h 小时才能重新用于训练。h 必须等于标签的持有期
   (72h 标签隔 72 小时,1 周标签隔 168 小时,2 周隔 336 小时),见 purged_walk_forward。
2. 保留集:HOLDOUT_START 之后的数据不参与任何选参、选阈值、选特征,只在最后跑一次(split_holdout)。
   2025-03 ~ 2026-10 的币安历史已经被多轮研究看过,对旧检测已不算干净;新特征的阈值必须在只看开发期的情况下先定好。
   最干净的保留集是 data 分支特征库从 2026-10 起新攒的 OKX 数据。
3. 同一轮行情的相关性:同一时刻很多币一起涨,事件不是独立样本。先按天把当天所有事件的超额取平均(一天一个数),
   再用 Newey-West(滞后 = 持有期天数)算 t;另报"有效独立期数" = 覆盖天数 / 持有期天数。
4. 大盘环境:按 BTC 30 日趋势、市场广度(站上 7 日均线的币占比)、山寨相对 BTC 的 30 日表现分组,分别报告(regimes)。
5. 多重检验:一次报告里检验了多少个(检测 × 持有期 × 分组),就对全部 p 值做 Holm(控制"至少错一个")
   和 Benjamini-Hochberg(控制错误发现率)校正(adjust)。
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

HOUR = 3_600_000
DAY = 24 * HOUR
HOLDOUT_START = "2026-06-01"        # 新特征的保留集起点:之后的数据只在最后跑一次


# ------------------------------------------------------------------ 1. 清洗边界
def purged_walk_forward(ts: np.ndarray, horizon_h: int, folds: int = 4, first_train: float = 0.4,
                        embargo_h: int | None = None) -> list[tuple[np.ndarray, np.ndarray]]:
    """按时间滚动:第 k 段检验 [lo, hi),训练 = ts < lo - horizon_h(标签窗口不越过检验期开始);
    之后各段训练也不用 [lo, hi + embargo) 里的样本(这里是只往前滚,天然满足)。返回 [(训练掩码, 检验掩码)]。"""
    ts = np.asarray(ts)
    qs = [first_train + (1 - first_train) * i / folds for i in range(folds + 1)]
    edges = np.quantile(ts, qs).astype("int64")
    edges[-1] += 1
    gap = horizon_h * HOUR
    out = []
    for k in range(folds):
        lo, hi = edges[k], edges[k + 1]
        test = (ts >= lo) & (ts < hi - gap)          # 检验样本的标签也不能伸出检验期(伸进下一段 / 保留集)
        train = ts < lo - gap
        out.append((train, test))
    return out


def split_holdout(ts, holdout_start: str = HOLDOUT_START, horizon_h: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """(开发期掩码, 保留集掩码)。开发期样本的标签窗口不能伸进保留集:ts < 起点 - horizon_h。"""
    ts = np.asarray(ts)
    t0 = int(pd.Timestamp(holdout_start, tz="UTC").timestamp() * 1000)
    return ts < t0 - horizon_h * HOUR, ts >= t0


# ------------------------------------------------------------------ 3. 相关性与 t
def nw_t(x, lags: int) -> float:
    x = pd.Series(x, dtype=float).dropna().to_numpy()
    n = len(x)
    if n < max(10, 2 * lags):           # 样本少于滞后期两倍时 Newey-West 方差不可靠
        return float("nan")
    e = x - x.mean()
    var = np.mean(e * e)
    for l in range(1, min(lags, n - 1) + 1):
        var += 2 * (1 - l / (lags + 1)) * np.mean(e[l:] * e[:-l])
    if not var > 0:
        return float("nan")
    return float(x.mean() / np.sqrt(var / n))


def p_two_sided(t: float) -> float:
    return float("nan") if t is None or not np.isfinite(t) else math.erfc(abs(t) / math.sqrt(2))


def event_stats(ts: np.ndarray, ex: np.ndarray, horizon_h: int) -> dict:
    """事件的超额收益统计:先按天平均(同一天多个币算一个),再做 Newey-West。"""
    d = pd.DataFrame({"day": np.asarray(ts) // DAY, "ex": np.asarray(ex, dtype=float)}).dropna()
    if len(d) < 10:
        return {"n": int(len(d))}
    daily = d.groupby("day")["ex"].mean()
    t = nw_t(daily, max(3, horizon_h // 24))
    span_days = (daily.index.max() - daily.index.min() + 1)
    return {"n": int(len(d)), "days": int(len(daily)), "indep": round(span_days * 24 / horizon_h, 1),
            "mean": float(d["ex"].mean()), "median": float(d["ex"].median()), "hit": float((d["ex"] > 0).mean()),
            "daily_mean": float(daily.mean()), "t": t, "p": p_two_sided(t)}


# ------------------------------------------------------------------ 4. 大盘环境
def regimes(frames: dict[str, pd.DataFrame], btc_lc: pd.Series | None = None) -> pd.DataFrame:
    """每小时的大盘环境标签(只用当时已知的数据):
    btc_trend   BTC 30 日涨跌:上涨 / 下跌
    breadth     站上 7 日均线的币占比:≥ 50% 宽 / < 50% 窄
    alt_vs_btc  山寨等权指数相对 BTC 的 30 日表现:山寨强 / BTC 强"""
    closes = pd.DataFrame({s: f["close"] for s, f in frames.items() if "close" in f})
    if btc_lc is None:
        any_f = next(iter(frames.values()))
        btc_lc = any_f["_btc_lc"]
    btc_lc = btc_lc.reindex(closes.index)
    lc = np.log(closes.where(closes > 0))
    above = (closes > closes.rolling(168, min_periods=120).mean()).where(closes.notna())
    alt = lc.diff().mean(axis=1).cumsum()
    out = pd.DataFrame(index=closes.index)
    out["btc_trend"] = np.where(btc_lc.diff(720) > 0, "BTC 30 日上涨", "BTC 30 日下跌")
    out.loc[btc_lc.diff(720).isna(), "btc_trend"] = np.nan
    b = above.mean(axis=1)
    out["breadth"] = np.where(b >= 0.5, "广度宽(≥50% 在 7 日线上)", "广度窄")
    out.loc[b.isna(), "breadth"] = np.nan
    rel = (alt - btc_lc).diff(720)
    out["alt_vs_btc"] = np.where(rel > 0, "山寨相对 BTC 强", "BTC 相对山寨强")
    out.loc[rel.isna(), "alt_vs_btc"] = np.nan
    return out


# ------------------------------------------------------------------ 5. 多重检验
def adjust(pvals) -> tuple[np.ndarray, np.ndarray]:
    """(Holm 校正后的 p, Benjamini-Hochberg q)。NaN 保持 NaN,不计入检验个数。"""
    p = np.asarray(pvals, dtype=float)
    ok = np.isfinite(p)
    holm, bh = np.full(len(p), np.nan), np.full(len(p), np.nan)
    m = int(ok.sum())
    if m == 0:
        return holm, bh
    idx = np.flatnonzero(ok)
    order = idx[np.argsort(p[idx])]
    run = 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (m - k) * p[i]))
        holm[i] = run
    run = 1.0
    for k in range(m - 1, -1, -1):
        i = order[k]
        run = min(run, p[i] * m / (k + 1))
        bh[i] = min(run, 1.0)
    return holm, bh
