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
RESOLVE_HORIZONS = (24, 72, 168, 336)    # 预警到期核对:24h / 72h / 1 周 / 2 周
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


_VALIDATED: dict | None = None


def validated_rules() -> dict[str, str]:
    """通过长期验证(rule_validation.py:扣掉全市场共同因子、按日聚合、|t|≥3 且前后半段同向)的规则 → 方向。
    文件缺失或没有规则通过时为空,此时任何规则都不会给出偏涨/偏跌。"""
    global _VALIDATED
    if _VALIDATED is None:
        try:
            d = json.loads((Path(__file__).resolve().parent.parent / "models" / "rule_validation.json").read_text(encoding="utf-8"))
            _VALIDATED = {k: v["direction"] for k, v in d.get("rules", {}).items() if v.get("validated")}
        except Exception:
            _VALIDATED = {}
    return _VALIDATED


def call_from(stats: dict | None, baseline: dict | None, cid: str | None = None) -> str:
    """方向判断:up / down / none。
    之前只看"近期同类次数 ≥ 15 且偏离基准 ≥ 8 个百分点",两个问题:同一轮行情里多个币一起触发并不是独立证据
    (云端样本库只有约 27 天),并且是从多个条件里挑偏离最大的那个(选择偏差)。
    现在要求该规则先通过长期验证(见 validated_rules),且近期偏离方向与验证方向一致,才给方向。"""
    if not stats or not baseline or stats["n"] < MIN_N_CALL or not cid:
        return "none"
    want = validated_rules().get(cid)
    if not want:
        return "none"
    edge = stats["up72"] - baseline["up72"]
    if edge >= EDGE_CALL and want == "up":
        return "up"
    if edge <= -EDGE_CALL and want == "down":
        return "down"
    return "none"


def _p(x, d=0) -> str:
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x * 100:+.{d}f}%"


def _pp(x) -> str:
    return "—" if x is None else f"{x * 100:.0f}%"


def outlook_line(cid: str, st: dict, baseline: dict, scope: str) -> str:
    call = call_from(st, baseline, cid)
    word = {"up": "偏涨", "down": "偏跌", "none": "无明确方向"}[call]
    small = "(样本少,仅供参考)" if st["n"] < MIN_N_CALL else ""
    if call == "none" and cid not in validated_rules():
        small += "(该情形未通过长期检验,只是近期频率,不代表规律)"
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
        # 中位秩:并列的值算一半,避免费率长期停在 0.01% 时被误判为"100% 分位"
        return float((past < x[-1]).mean() + 0.5 * (past == x[-1]).mean())
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
                   n: int, scope: str = "", rules: list[str] | None = None, score: float | None = None,
                   watch: bool = False) -> dict:
    return {"id": f"{now}-{kind}-{symbol}-{cid}", "ts": now, "t_bar": int(t_bar), "kind": kind,
            "symbol": symbol, "cond": cid, "price": price, "scope": scope, "call": call,
            "pred_up72": pred_up72, "base_up72": base_up72, "pred_med72": pred_med72, "n": int(n),
            "rules": rules or [cid], "score": score, "watch": bool(watch), "res": {}}


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
        for h in RESOLVE_HORIZONS:
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


# ------------------------------------------------------------------ 信号后验记录表(永久累积)
# predictions.json 只保留 PRED_KEEP_DAYS 天、且只记主条件;这张表每条推送的信号一行,
# 含触发的全部规则、得分、24h/72h/1 周/2 周的真实收益(原始/相对 BTC)和持有期最大回撤,永不删除,
# 实盘样本越攒越多,按规则汇总后可以和回测(research.py / tune.py)对照。
LEDGER_COLS = ["id", "ts", "symbol", "watch", "rules", "score", "price", "call", "pred_up72", "base_up72",
               "ret24", "resid24", "mae24", "ret72", "resid72", "mae72",
               "ret168", "resid168", "mae168", "ret336", "resid336", "mae336"]


def load_ledger(path: Path) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(path)
        return df if "id" in df else None
    except Exception:
        return None


def update_ledger(ledger: pd.DataFrame | None, preds: list[dict]) -> pd.DataFrame:
    """把 signal 类预测并入记录表(按 id 更新:新信号加一行,到期的补上 24h/72h/1 周/2 周结果)。"""
    rows = {r["id"]: r for r in ledger.to_dict("records")} if ledger is not None and len(ledger) else {}
    for p in preds:
        if p.get("kind") != "signal":
            continue
        r = rows.get(p["id"], {})
        r.update({"id": p["id"], "ts": p["ts"], "symbol": p["symbol"], "watch": int(bool(p.get("watch"))),
                  "rules": ";".join(p.get("rules") or [p["cond"]]), "score": p.get("score"),
                  "price": p.get("price"), "call": p.get("call"),
                  "pred_up72": p.get("pred_up72"), "base_up72": p.get("base_up72")})
        for h in RESOLVE_HORIZONS:
            res = p["res"].get(f"{h}h")
            if res:
                r[f"ret{h}"] = res["ret"]
                r[f"resid{h}"] = None if res.get("btc") is None else res["ret"] - res["btc"]
                r[f"mae{h}"] = res.get("mae")
        rows[p["id"]] = r
    df = pd.DataFrame(list(rows.values()), columns=LEDGER_COLS)
    return df.sort_values("ts").reset_index(drop=True)


def _row_stats(sub: pd.DataFrame) -> dict | None:
    x = sub["resid72"].dropna()
    if x.empty:
        return None
    mae = sub["mae72"].dropna()
    p10 = float(mae.quantile(0.10)) if len(mae) else None
    sd = x.std()
    return {"n": int(len(x)), "resid72_mean": float(x.mean()), "resid72_median": float(x.median()),
            "hit72": float((x > 0).mean()),
            "t72": float(x.mean() / (sd / np.sqrt(len(x)))) if len(x) > 1 and sd > 0 else None,
            "mae72_p10": p10, "safe_lev": float(1 / abs(p10)) if p10 and p10 < 0 else None,
            **{k: v for h in (168, 336) for k, v in _long_stats(sub, h).items()}}


def _long_stats(sub: pd.DataFrame, h: int) -> dict:
    """1 周 / 2 周的到期数、超额中位、跑赢 BTC 的比例。同一规则的信号持有期大段重叠,不给 t 值。"""
    x = sub[f"resid{h}"].dropna() if f"resid{h}" in sub else pd.Series(dtype=float)
    return {f"n{h}": int(len(x)), f"resid{h}_median": float(x.median()) if len(x) else None,
            f"hit{h}": float((x > 0).mean()) if len(x) else None}


def ledger_summary(ledger: pd.DataFrame | None) -> dict:
    """按规则汇总已到期(72h)的实盘信号;一条信号触发多条规则时,每条规则各算一次。"""
    if ledger is None or ledger.empty:
        return {"total": 0, "resolved": 0, "by_rule": {}, "all": None}
    done = ledger[ledger["resid72"].notna()]
    out = {"total": int(len(ledger)), "resolved": int(len(done)), "all": _row_stats(done) if len(done) else None,
           "by_rule": {}}
    if len(done):
        exp = done.assign(rule=done["rules"].astype(str).str.split(";")).explode("rule")
        for rid, sub in exp.groupby("rule"):
            st = _row_stats(sub)
            if st:
                out["by_rule"][rid] = st
    return out


def ledger_text(ls: dict, min_n: int = 30) -> list[str]:
    if not ls or not ls.get("total"):
        return ["实盘还没有到期的信号,72 小时后开始累积"]
    lines = [f"累计推送 {ls['total']} 条信号,已到期 {ls['resolved']} 条(每条 72h 后结算;样本 < {min_n} 的结论不可靠)"]
    if ls.get("by_rule"):
        lines += ["| 规则 | 到期数 | 72h 超额中位 | 上涨比例 | t | safe_lev | 1周 超额中位(跑赢比例,到期数) | 2周 超额中位(跑赢比例,到期数) |",
                  "|---|---|---|---|---|---|---|---|"]
        long = lambda st, h: ("—" if not st.get(f"n{h}") else
                              f"{_p(st[f'resid{h}_median'], 1)}({_pp(st[f'hit{h}'])},{st[f'n{h}']})")
        for rid, st in sorted(ls["by_rule"].items(), key=lambda kv: -kv[1]["n"]):
            name = RULES_BY_ID[rid].name if rid in RULES_BY_ID else rid
            t = "—" if st["t72"] is None else f"{st['t72']:+.1f}"
            lev = "—" if st["safe_lev"] is None else f"{st['safe_lev']:.1f}x"
            flag = "" if st["n"] >= min_n else "(样本少)"
            lines.append(f"| {name}{flag} | {st['n']} | {_p(st['resid72_median'], 1)} | {_pp(st['hit72'])} | {t} | {lev} | "
                         f"{long(st, 168)} | {long(st, 336)} |")
    return lines
