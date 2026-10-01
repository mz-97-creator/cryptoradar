"""前瞻模块:把"刚才发生了什么"变成"历史上接下来通常怎样"。

1. 特征样本库(archive):每轮把全市场的小时特征存进 data 分支,样本随时间增长,
   历史概率不再只靠 OKX 接口给的最近 30 天。
2. 历史概率(base rates):对每条规则/形态,统计全市场所有代币过去出现同样情形后,
   未来 24h/72h 的涨跌概率、中位收益、最差 10%、持有期最大回撤,并和"任意时点"基准对比。
3. 市场红绿灯:全市场杠杆热度(资金费率分位、OI 变化)× 广度(站上 7 日均线的比例)。
4. 预测记分卡:每条预警都记录当时的历史概率,到期后用真实价格核对,
   让你知道这些预警在样本外到底准不准。

所有统计只用当时已知的数据;未来收益只在事件过去之后才计算。
"""
from __future__ import annotations

import gzip
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .features import add_labels
from .signals import RULES, RULES_BY_ID, evaluate_frame

HOUR = 3_600_000
GAP = 72                 # 同一代币同一条件,两次事件至少间隔 72 小时
ARCHIVE_DAYS = 180
PRED_KEEP_DAYS = 120
MIN_N_CALL = 15          # 样本少于这个数,不给方向判断
EDGE_CALL = 0.08         # 上涨概率比基准高/低 8 个百分点以上,才算有方向
ARCHIVE_COLS = ["close", "_high", "_low", "_btc_lc", "beta", "ret_24h", "resid_24h_z", "ret_1h_z",
                "oi", "oi_chg_24h", "oi_z", "funding", "funding_z", "vol_z", "top_ls_z",
                "range_24h", "range_z"]


# ------------------------------------------------------------------ 样本库
def load_archive(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path, compression="gzip")
    except Exception:
        return None


def merge_archive(archive: pd.DataFrame | None, frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """实时特征优先;实时缺失(预热期)或更早的小时用样本库补上。返回按代币的小时表。"""
    out = {}
    groups = dict(tuple(archive.groupby("symbol"))) if archive is not None and len(archive) else {}
    for sym, f in frames.items():
        live = f[[c for c in ARCHIVE_COLS if c in f.columns]]
        old = groups.get(sym)
        if old is not None:
            old = old.set_index("ts")[[c for c in ARCHIVE_COLS if c in old.columns]]
            old = old[~old.index.duplicated(keep="last")]
            comb = live.combine_first(old)
        else:
            comb = live.copy()
        comb = comb.sort_index()
        if len(comb):
            grid = np.arange(int(comb.index.min()), int(comb.index.max()) + HOUR, HOUR)
            comb = comb.reindex(grid)
        out[sym] = comb
    return out


def archive_table(combined: dict[str, pd.DataFrame], now: int) -> pd.DataFrame:
    cut = now - ARCHIVE_DAYS * 24 * HOUR
    parts = []
    for sym, f in combined.items():
        g = f[f.index >= cut].dropna(subset=["close"])
        if len(g):
            g = g.copy()
            g.insert(0, "symbol", sym)
            g.index.name = "ts"
            parts.append(g.reset_index())
    if not parts:
        return pd.DataFrame(columns=["symbol", "ts"] + ARCHIVE_COLS)
    return pd.concat(parts, ignore_index=True)


def save_archive(table: pd.DataFrame, path: Path) -> None:
    buf = io.StringIO()
    table.to_csv(buf, index=False, float_format="%.6g")
    with gzip.open(path, "wt", encoding="utf-8", compresslevel=9) as fh:
        fh.write(buf.getvalue())


# ------------------------------------------------------------------ 历史概率
def label_frames(combined: dict[str, pd.DataFrame], th: dict) -> dict[str, pd.DataFrame]:
    out = {}
    for sym, f in combined.items():
        if len(f) < 200:
            continue
        g = add_labels(f)
        out[sym] = g.join(evaluate_frame(g, th).add_prefix("C_"))
    return out


def decluster(mask: np.ndarray, gap: int = GAP) -> np.ndarray:
    keep, last = [], -10**9
    for pos in np.flatnonzero(mask):
        if pos - last >= gap:
            keep.append(pos)
            last = pos
    return np.array(keep, dtype=int)


def _stats(sub: pd.DataFrame) -> dict:
    r24 = np.expm1(sub["fwd_ret_24h"])
    r72 = np.expm1(sub["fwd_ret_72h"])
    x72 = np.expm1(sub["fwd_resid_72h"])
    mae = sub["mae_72h"]
    n = int(len(sub))
    sd = x72.std()
    return {
        "n": n,
        "up24": float((r24 > 0).mean()), "up72": float((r72 > 0).mean()),
        "med24": float(r24.median()), "med72": float(r72.median()), "mean72": float(r72.mean()),
        "p10_72": float(r72.quantile(0.10)), "p90_72": float(r72.quantile(0.90)),
        "resid72_med": float(x72.median()),
        "t_resid72": float(x72.mean() / (sd / np.sqrt(n))) if n > 2 and sd > 0 else None,
        "mae72_med": float(mae.median()), "mae72_p10": float(mae.quantile(0.10)),
        "mfe72_med": float(sub["mfe_72h"].median()),
    }


def base_rates(labeled: dict[str, pd.DataFrame], watch: list[str], min_n: int = 5) -> dict:
    need = ["fwd_ret_24h", "fwd_ret_72h", "fwd_resid_72h", "mae_72h", "mfe_72h"]
    base_parts, by_cond, by_sym = [], {r.id: [] for r in RULES}, {}
    t_min, t_max = None, None
    for sym, g in labeled.items():
        ok = g[need].notna().all(axis=1).to_numpy()
        if not ok.any():
            continue
        idx = g.index[ok]
        t_min = idx.min() if t_min is None else min(t_min, idx.min())
        t_max = idx.max() if t_max is None else max(t_max, idx.max())
        base_parts.append(g.loc[ok].iloc[::24][need])
        for r in RULES:
            pos = decluster(g[f"C_{r.id}"].to_numpy() & ok)
            if len(pos):
                ev = g.iloc[pos][need]
                by_cond[r.id].append(ev)
                if sym in watch:
                    by_sym.setdefault(sym, {})[r.id] = ev
    if not base_parts:
        return {"baseline": None, "conditions": {}, "by_symbol": {}, "span_days": 0}
    baseline = _stats(pd.concat(base_parts))
    conds = {}
    for cid, parts in by_cond.items():
        if parts:
            sub = pd.concat(parts)
            if len(sub) >= min_n:
                conds[cid] = _stats(sub)
    syms = {}
    for sym, d in by_sym.items():
        syms[sym] = {cid: _stats(ev) for cid, ev in d.items() if len(ev) >= 4}
    return {"baseline": baseline, "conditions": conds, "by_symbol": syms,
            "span_days": round((t_max - t_min) / 86_400_000, 1) if t_min is not None else 0}


def call_from(stats: dict | None, baseline: dict | None) -> str:
    """有足够样本且明显偏离基准时,给出方向:up / down;否则 none。"""
    if not stats or not baseline or stats["n"] < MIN_N_CALL:
        return "none"
    edge = stats["up72"] - baseline["up72"]
    if edge >= EDGE_CALL:
        return "up"
    if edge <= -EDGE_CALL:
        return "down"
    return "none"


def _p(x, d=0) -> str:
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x * 100:+.{d}f}%"


def _pp(x) -> str:
    return "—" if x is None else f"{x * 100:.0f}%"


def outlook_line(cid: str, st: dict, baseline: dict, scope: str) -> str:
    call = call_from(st, baseline)
    word = {"up": "偏涨", "down": "偏跌", "none": "无明显方向"}[call]
    small = "(样本少,仅供参考)" if st["n"] < MIN_N_CALL else ""
    return (f"{RULES_BY_ID[cid].name} → {scope}同类 {st['n']} 次:72h 上涨概率 {_pp(st['up72'])}"
            f"(基准 {_pp(baseline['up72'])}),中位 {_p(st['med72'], 1)},最差 10% {_p(st['p10_72'], 1)},"
            f"持有期最大回撤中位 {_p(st['mae72_med'], 1)} · {word}{small}")


def pick_outlook(fired_ids: list[str], sym: str, rates: dict) -> tuple[str | None, dict | None, str]:
    """在触发的条件里,挑历史上最有信息量的一条(偏离基准最大且样本够)。优先用该币自己的历史。"""
    base = rates.get("baseline")
    if not base:
        return None, None, ""
    best, best_score, best_scope = None, -1.0, ""
    for cid in fired_ids:
        for scope, st in (("本币", rates.get("by_symbol", {}).get(sym, {}).get(cid)),
                          ("全市场", rates.get("conditions", {}).get(cid))):
            if not st:
                continue
            score = abs(st["up72"] - base["up72"]) * np.sqrt(st["n"])
            if scope == "本币" and st["n"] < 8:
                continue
            if score > best_score:
                best, best_score, best_scope = (cid, st), score, scope
    if not best:
        return None, None, ""
    return best[0], best[1], best_scope


# ------------------------------------------------------------------ 市场红绿灯
STATES = {
    "RED": ("🔴", "去杠杆风险高",
            "资金费率在近 30 天高位、持仓还在增加,但上涨的币越来越少。杠杆在加,参与面在缩,一旦下跌容易连环爆仓。",
            "费率回到中位以下,或站上 7 日均线的币回到 50% 以上"),
    "ORANGE": ("🟠", "杠杆过热",
               "资金费率在近 30 天高位且持仓增加,多头在用杠杆追。趋势还在,但回调时的跌幅会被放大。",
               "广度开始下滑(站上 7 日均线的币比例 24h 内掉 20 个百分点以上)就升级为红灯"),
    "YELLOW": ("🟡", "市场转弱",
               "站上 7 日均线的币不到 35%,或 24h 内大幅减少,但杠杆不算高。偏弱整理,爆仓连锁的风险不大。",
               "广度回到 50% 以上;若同时费率升高则转橙灯"),
    "BLUE": ("🔵", "杠杆已出清",
             "全市场持仓明显下降、费率回落,多头杠杆刚被清洗。历史上常是短线反弹窗口,但不保证见底。",
             "持仓重新增加而价格不涨,说明有人在抄底失败"),
    "GREEN": ("🟢", "正常",
              "杠杆和广度都在正常区间,没有系统性风险信号。",
              "费率升到近 30 天 80% 分位以上且持仓增加"),
}


def _past_pct(s: pd.Series, window: int = 720, min_periods: int = 168) -> pd.Series:
    """当前值在过去 window 小时(不含当前)里的分位。"""
    def f(x):
        past = x[:-1]
        past = past[~np.isnan(past)]
        if len(past) < min_periods or np.isnan(x[-1]):
            return np.nan
        return float((past <= x[-1]).mean())
    return s.rolling(window + 1, min_periods=min_periods + 1).apply(f, raw=True)


def market_frame(combined: dict[str, pd.DataFrame], btc: str = "BTC") -> pd.DataFrame:
    alts = {s: f for s, f in combined.items() if s != btc and len(f) > 200}
    if len(alts) < 5 or btc not in combined:
        return pd.DataFrame()
    close = pd.DataFrame({s: f["close"] for s, f in alts.items()}).sort_index()
    close = close[close.index >= close.index.max() - ARCHIVE_DAYS * 24 * HOUR]
    ma = close.rolling(168, min_periods=120).mean()
    valid = ma.notna() & close.notna()
    cnt = valid.sum(axis=1)
    m = pd.DataFrame(index=close.index)
    m["breadth"] = ((close > ma) & valid).sum(axis=1) / cnt.where(cnt >= 5)
    m["breadth_chg24"] = m["breadth"].diff(24)
    m["fund_med"] = pd.DataFrame({s: f["funding"] for s, f in alts.items()}).reindex(close.index).median(axis=1)
    m["fund_pct"] = _past_pct(m["fund_med"])
    m["oi_z_med"] = pd.DataFrame({s: f["oi_z"] for s, f in alts.items()}).reindex(close.index).median(axis=1)
    lr = np.log(close).diff()
    m["alt_idx"] = lr.median(axis=1).fillna(0).cumsum()
    m["alt_ret24"] = np.expm1(m["alt_idx"].diff(24))
    b = combined[btc]["close"].reindex(close.index)
    m["btc_close"] = b
    m["btc_ret24"] = b.pct_change(24)
    # 未来:山寨指数(等权中位)与 BTC 的 72h 收益、山寨指数持有期最大回撤
    m["alt_fwd72"] = np.expm1(m["alt_idx"].shift(-72) - m["alt_idx"])
    fut_min = m["alt_idx"].iloc[::-1].rolling(72, min_periods=72).min().iloc[::-1].shift(-1)
    m["alt_mae72"] = np.expm1(fut_min - m["alt_idx"])
    m["btc_fwd72"] = b.shift(-72) / b - 1

    heat = (m["fund_pct"] >= 0.8) & (m["oi_z_med"] >= 0.3)
    weak = (m["breadth"] <= 0.35) | (m["breadth_chg24"] <= -0.20)
    flush = (m["oi_z_med"] <= -1.0) & (m["fund_pct"] <= 0.5)
    st = pd.Series("GREEN", index=m.index)
    st[weak] = "YELLOW"
    st[flush] = "BLUE"
    st[heat] = "ORANGE"
    st[heat & weak] = "RED"
    st[m["breadth"].isna() | m["fund_pct"].isna()] = None
    m["state"] = st
    return m


def market_rates(m: pd.DataFrame) -> dict:
    out = {}
    ok = m[["alt_fwd72", "btc_fwd72", "alt_mae72"]].notna().all(axis=1).to_numpy()
    for key in STATES:
        pos = decluster((m["state"] == key).to_numpy() & ok)
        if not len(pos):
            continue
        sub = m.iloc[pos]
        out[key] = {"n": int(len(sub)),
                    "alt_up72": float((sub["alt_fwd72"] > 0).mean()),
                    "alt_med72": float(sub["alt_fwd72"].median()),
                    "alt_p10_72": float(sub["alt_fwd72"].quantile(0.1)),
                    "alt_mae72_med": float(sub["alt_mae72"].median()),
                    "btc_med72": float(sub["btc_fwd72"].median()),
                    "btc_up72": float((sub["btc_fwd72"] > 0).mean())}
    base = m[ok].iloc[::24]
    if len(base):
        out["ALL"] = {"n": int(len(base)), "alt_up72": float((base["alt_fwd72"] > 0).mean()),
                      "alt_med72": float(base["alt_fwd72"].median()),
                      "alt_p10_72": float(base["alt_fwd72"].quantile(0.1)),
                      "alt_mae72_med": float(base["alt_mae72"].median()),
                      "btc_med72": float(base["btc_fwd72"].median()),
                      "btc_up72": float((base["btc_fwd72"] > 0).mean())}
    return out


def market_summary(m: pd.DataFrame, rates: dict) -> dict:
    if m.empty or m["state"].dropna().empty:
        return {}
    last = m.dropna(subset=["state"]).iloc[-1]
    key = last["state"]
    since = m["state"][m["state"] != key].index
    since_ts = int(m.index[m.index > since.max()].min()) if len(since) else int(m.index.min())
    icon, title, meaning, watch = STATES[key]
    facts = (f"全市场中位资金费率 {last['fund_med'] * 100:.4f}%/8h(高于近 30 天 {last['fund_pct'] * 100:.0f}% 的时间)"
             f" · 持仓变化中位 z={last['oi_z_med']:+.1f}"
             f" · {last['breadth'] * 100:.0f}% 的币站上 7 日均线(24h {last['breadth_chg24'] * 100:+.0f} 个百分点)"
             f" · 山寨 24h {_p(last['alt_ret24'], 1)} / BTC {_p(last['btc_ret24'], 1)}")
    r, base = rates.get(key), rates.get("ALL")
    if r:
        hist = (f"历史上进入此状态 {r['n']} 次,之后 72h:山寨中位 {_p(r['alt_med72'], 1)}"
                f"(最差 10% {_p(r['alt_p10_72'], 1)},期间回撤中位 {_p(r['alt_mae72_med'], 1)}),"
                f"BTC 中位 {_p(r['btc_med72'], 1)}、上涨概率 {_pp(r['btc_up72'])}")
        if base:
            hist += f";任意时点基准:山寨中位 {_p(base['alt_med72'], 1)},BTC 上涨概率 {_pp(base['btc_up72'])}"
        if r["n"] < 5:
            hist += "(样本少,仅供参考)"
    else:
        hist = "样本库里还没有足够的同类状态,随数据积累会给出"
    return {"state": key, "icon": icon, "title": title, "since": since_ts,
            "meaning": meaning, "facts": facts, "history": hist, "watch": watch,
            "metrics": {k: (None if pd.isna(last[k]) else float(last[k]))
                        for k in ["fund_med", "fund_pct", "oi_z_med", "breadth", "breadth_chg24",
                                  "alt_ret24", "btc_ret24"]}}


def market_text(ms: dict) -> str:
    if not ms:
        return "市场状态:数据不足"
    return (f"{ms['icon']} 市场:{ms['title']}。{ms['meaning']}\n"
            f"- 数据:{ms['facts']}\n- {ms['history']}\n- 何时解除/升级:{ms['watch']}")


# ------------------------------------------------------------------ 预测记分卡
def load_predictions(path: Path) -> list[dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("predictions", [])
    except Exception:
        return []


def new_prediction(now: int, kind: str, symbol: str, cid: str, t_bar: int, price: float | None,
                   call: str, pred_up72: float | None, base_up72: float | None, pred_med72: float | None,
                   n: int, scope: str = "") -> dict:
    return {"id": f"{now}-{kind}-{symbol}-{cid}", "ts": now, "t_bar": int(t_bar), "kind": kind,
            "symbol": symbol, "cond": cid, "price": price, "scope": scope, "call": call,
            "pred_up72": pred_up72, "base_up72": base_up72, "pred_med72": pred_med72, "n": int(n), "res": {}}


def market_call(st: dict | None, base: dict | None) -> str:
    """市场灯的方向同样来自历史:该状态之后山寨中位收益比基准低/高 2 个百分点以上才给方向。"""
    if not st or not base or st["n"] < 5:
        return "none"
    d = st["alt_med72"] - base["alt_med72"]
    return "up" if d >= 0.02 else "down" if d <= -0.02 else "none"


def resolve(preds: list[dict], combined: dict[str, pd.DataFrame], m: pd.DataFrame, now: int) -> list[dict]:
    """到期的预测用真实价格核对:收益、相对 BTC 的超额、持有期最大回撤。"""
    btc = combined.get("BTC")
    for p in preds:
        for h in (24, 72):
            key = f"{h}h"
            if key in p["res"] or now < p["t_bar"] + (h + 1) * HOUR:
                continue
            t0, t1 = p["t_bar"], p["t_bar"] + h * HOUR
            if p["kind"] == "market":
                if m.empty or t0 not in m.index or t1 not in m.index:
                    continue
                a0, a1 = m.at[t0, "alt_idx"], m.at[t1, "alt_idx"]
                win = m["alt_idx"].loc[t0 + HOUR:t1]
                b0, b1 = m.at[t0, "btc_close"], m.at[t1, "btc_close"]
                if pd.isna(b0) or pd.isna(b1):
                    continue
                p["res"][key] = {"ret": float(np.expm1(a1 - a0)), "btc": float(b1 / b0 - 1),
                                 "mae": float(np.expm1(win.min() - a0)) if len(win) else None}
            else:
                f = combined.get(p["symbol"])
                if f is None or t0 not in f.index or t1 not in f.index:
                    continue
                c0, c1 = f.at[t0, "close"], f.at[t1, "close"]
                lows = f["_low"].loc[t0 + HOUR:t1]
                if pd.isna(c0) or pd.isna(c1):
                    continue
                bret = None
                if btc is not None and t0 in btc.index and t1 in btc.index:
                    bret = float(btc.at[t1, "close"] / btc.at[t0, "close"] - 1)
                p["res"][key] = {"ret": float(c1 / c0 - 1), "btc": bret,
                                 "mae": float(lows.min() / c0 - 1) if lows.notna().any() else None}
    cut = now - PRED_KEEP_DAYS * 24 * HOUR
    return [p for p in preds if p["ts"] >= cut]


def scorecard(preds: list[dict], now: int, days: int = 30) -> dict:
    cut = now - days * 24 * HOUR
    done = [p for p in preds if p["ts"] >= cut and "72h" in p["res"]]
    out = {"days": days, "pending": sum(1 for p in preds if "72h" not in p["res"]), "groups": {}}

    def summarize(items: list[dict]) -> dict | None:
        if not items:
            return None
        rets = np.array([p["res"]["72h"]["ret"] for p in items])
        maes = np.array([p["res"]["72h"]["mae"] for p in items if p["res"]["72h"].get("mae") is not None])
        calls = [p for p in items if p["call"] in ("up", "down")]
        hits = [(p["res"]["72h"]["ret"] > 0) == (p["call"] == "up") for p in calls]
        preds_up = [p["pred_up72"] for p in items if p.get("pred_up72") is not None]
        return {"n": len(items), "ret72_med": float(np.median(rets)), "ret72_mean": float(rets.mean()),
                "up72_real": float((rets > 0).mean()),
                "up72_pred": float(np.mean(preds_up)) if preds_up else None,
                "mae72_med": float(np.median(maes)) if len(maes) else None,
                "n_calls": len(calls), "hit": float(np.mean(hits)) if hits else None}

    for kind in ("signal", "market"):
        items = [p for p in done if p["kind"] == kind]
        out["groups"][kind] = {
            "all": summarize(items),
            "up": summarize([p for p in items if p["call"] == "up"]),
            "down": summarize([p for p in items if p["call"] == "down"]),
            "by_cond": {c: summarize([p for p in items if p["cond"] == c])
                        for c in sorted({p["cond"] for p in items})},
        }
    return out


def scorecard_text(sc: dict) -> str:
    g = sc.get("groups", {}).get("signal", {})
    a = g.get("all")
    if not a:
        return f"记分卡:过去 {sc.get('days', 30)} 天还没有到期的预警(待核对 {sc.get('pending', 0)} 条),72 小时后开始出结果"
    parts = [f"记分卡(近 {sc['days']} 天到期 {a['n']} 条):72h 中位 {_p(a['ret72_med'], 1)},"
             f"实际上涨比例 {_pp(a['up72_real'])} vs 当时预计 {_pp(a['up72_pred'])}"]
    if a["n_calls"]:
        parts.append(f"有方向判断的 {a['n_calls']} 条命中 {_pp(a['hit'])}(随机约 50%)")
    for k, w in (("up", "偏涨"), ("down", "偏跌")):
        s = g.get(k)
        if s:
            parts.append(f"{w}预警 {s['n']} 条实际 72h 中位 {_p(s['ret72_med'], 1)}")
    mk = sc["groups"].get("market", {}).get("all")
    if mk:
        parts.append(f"市场灯切换 {mk['n']} 次,之后山寨 72h 中位 {_p(mk['ret72_med'], 1)}")
    parts.append(f"待核对 {sc['pending']} 条")
    return ";".join(parts)
