"""统一模型 + 新预测目标的离线研究。目标、特征、模型、切分和合格标准见 studies/unified_preregistration.md
(在跑结果之前提交),这里只是照着实现,不调参。

用法(先用 backfill_vision.py / spotflow 回填币安历史,DefiLlama 历史 JSON 见 early_study.py):
  python unified_study.py --spot --fund-history fund_history.json
结果:reports/unified_study.md、reports/unified_alerts.csv
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

import evalkit as K
import lag_study as L
from cryptoradar import early
from cryptoradar.config import load_config
from cryptoradar.signals import apply_weights, merged_thresholds
from event_study import cum_excess

log = logging.getLogger("unified")
HOUR = K.HOUR
H = 168                                  # 标签持有期
UP, DN = np.log(1.15), np.log(0.90)      # 三重障碍:先到 +15% 记 1,先到 -10% 或到期未触及记 0
TOPK, DEDUP_H = 3, 72
FEATS = ["resid_24h_z", "resid_6h_z", "resid_72h", "ret_1h_z", "vol_z", "range_z", "xs_high",
         "oi_z", "oi_72h_z", "oi_up_days", "funding", "funding_z", "top_ls_z", "taker_z",
         "spot_buy_z", "spot_buy_dz12", "lev_share_z", "spot_net72_z", "spot_days_pos",
         "rs_rank", "rs_rank_chg", "de_rank", "de_rank_chg", "rb_rank", "pullback_depth", "rebound_48h",
         "f_hrev_ratio", "f_rev_ratio", "f_fees_ratio", "f_tvl_chg_7d",
         "btc_ret_7d", "btc_ret_30d", "breadth", "alt_vs_btc_30d"]
LOG_FEATS = ["f_hrev_ratio", "f_rev_ratio", "f_fees_ratio"]


# ------------------------------------------------------------------ 数据
def load_frames(cfg: dict, spot: bool, fund_history: str | None) -> dict[str, pd.DataFrame]:
    """币安历史 + 现货资金流 + DefiLlama(数据日滞后 2 天)+ 全市场横截面列。"""
    frames = L.frames_from_db(cfg, None)
    frames.pop("BTC", None)
    if spot:
        import sqlite3
        from backfill_vision import to_symbol
        from cryptoradar import spotflow
        from cryptoradar.features import spot_features
        from cryptoradar.storage import Store
        conn, store = sqlite3.connect(cfg["storage"]["db_path"]), Store(cfg["storage"]["db_path"])
        for sym, f in frames.items():
            ps = to_symbol(sym)
            perp = store.load_hourly(ps).reindex(f.index)
            sq = pd.read_sql_query("SELECT ts, quote_volume, taker_buy_quote FROM spot_hourly WHERE symbol=? ORDER BY ts",
                                   conn, params=[spotflow.spot_symbol(ps)]).drop_duplicates("ts").set_index("ts")
            if sq.empty:
                continue
            sq = sq.reindex(f.index)
            sp = spot_features(sq["taker_buy_quote"], sq["quote_volume"] - sq["taker_buy_quote"], perp["quote_volume"])
            for c in sp:
                frames[sym][c] = sp[c].to_numpy()
    if fund_history:
        from cryptoradar import fundamentals as fd
        fh = json.loads(Path(fund_history).read_text(encoding="utf-8"))
        for sym in list(frames):
            if fh["coins"].get(sym):
                frames[sym] = fd.attach_history(frames[sym], fd.daily_features(fh["coins"][sym]))
    return early.add_cross_section(frames)


def market_features(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    closes = pd.DataFrame({s: f["close"] for s, f in frames.items()})
    btc = next(iter(frames.values()))["_btc_lc"].reindex(closes.index)
    lc = np.log(closes.where(closes > 0))
    above = (closes > closes.rolling(168, min_periods=120).mean()).where(closes.notna())
    alt = lc.diff().mean(axis=1).cumsum()
    return pd.DataFrame({"btc_ret_7d": btc.diff(168), "btc_ret_30d": btc.diff(720), "breadth": above.mean(axis=1),
                         "alt_vs_btc_30d": (alt - btc).diff(720)}, index=closes.index)


def label_a(c: np.ndarray, i: int) -> float:
    """c:该币累计超额(对数,逐小时);i:时点位置。168 小时内先到 +15% 记 1,先到 -10% 或到期未触及记 0,数据不全记 NaN。"""
    if i + H >= len(c) or not np.isfinite(c[i]) or not np.isfinite(c[i + H]):
        return np.nan
    rel = c[i + 1:i + H + 1] - c[i]
    up = np.flatnonzero(rel >= UP)
    dn = np.flatnonzero(rel <= DN)
    if len(up) and (not len(dn) or up[0] < dn[0]):
        return 1.0
    return 0.0


def build_dataset(frames, C: pd.DataFrame, M: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sym, f in frames.items():
        if sym not in C:
            continue
        idx = f.index.to_numpy()
        c = C[sym].reindex(f.index).to_numpy()
        pos = np.flatnonzero((idx // HOUR) % 6 == 0)
        d = f.iloc[pos].reindex(columns=[c_ for c_ in FEATS if c_ not in M.columns]).copy()
        for k in M.columns:
            d[k] = M[k].reindex(d.index).to_numpy()
        d["symbol"], d["ts"] = sym, idx[pos]
        d["y"] = [label_a(c, i) for i in pos]
        d["ex168"] = [c[i + H] - c[i] if i + H < len(c) else np.nan for i in pos]
        rows.append(d.reset_index(drop=True))
    D = pd.concat(rows, ignore_index=True)
    for k in LOG_FEATS:
        D[k] = np.log(D[k].clip(lower=1e-3))
    D = D.replace([np.inf, -np.inf], np.nan)
    D["y_rank"] = D.groupby("ts")["ex168"].rank(pct=True)
    return D


# ------------------------------------------------------------------ 模型(超参数按事先登记固定)
class Logit:
    name = "逻辑回归"

    def fit(self, X: pd.DataFrame, y):
        from sklearn.linear_model import LogisticRegression
        self.mu, self.sd = X.mean(), X.std().replace(0, 1)
        self.m = LogisticRegression(C=0.1, max_iter=2000).fit(self._prep(X), y)
        return self

    def _prep(self, X):
        Z = ((X - self.mu) / self.sd)
        miss = X.isna().astype(float).add_suffix("_na")
        return pd.concat([Z.fillna(0), miss], axis=1).to_numpy()

    def score(self, X):
        return self.m.predict_proba(self._prep(X))[:, 1]

    def coefs(self, X):
        names = list(X.columns) + [c + "_na" for c in X.columns]
        return pd.Series(self.m.coef_[0], index=names)


class GBM:
    name = "梯度提升树"

    def fit(self, X, y):
        from sklearn.ensemble import HistGradientBoostingClassifier
        self.m = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, min_samples_leaf=1000,
                                                l2_regularization=10.0, random_state=0).fit(X.to_numpy(), y)
        return self

    def score(self, X):
        return self.m.predict_proba(X.to_numpy())[:, 1]


# ------------------------------------------------------------------ 提醒与评估
def alerts_from_scores(D: pd.DataFrame, score: np.ndarray) -> pd.DataFrame:
    """每个时点分数最高的 TOPK 个币;同一币 DEDUP_H 小时内只算一次。"""
    S = D[["symbol", "ts", "y", "ex168"]].assign(score=score)
    top = S.sort_values(["ts", "score"], ascending=[True, False]).groupby("ts").head(TOPK).sort_values("ts")
    keep, last = [], {}
    for i, r in zip(top.index, top.itertuples()):
        if r.symbol not in last or r.ts - last[r.symbol] >= DEDUP_H * HOUR:
            keep.append(i)
            last[r.symbol] = r.ts
    return top.loc[keep]


def label_at(C: pd.DataFrame, sym: str, t0: int) -> tuple[float, float]:
    """任意时点的目标 A 和 168h 超额(给现有推送用)。"""
    if sym not in C:
        return np.nan, np.nan
    c = C[sym]
    if t0 not in c.index:
        return np.nan, np.nan
    i = c.index.get_loc(t0)
    arr = c.to_numpy()
    return label_a(arr, i), (arr[i + H] - arr[i] if i + H < len(arr) else np.nan)


def earliness(frames, alerts_by_sym: dict[str, np.ndarray], lo: int, hi: int, min_gain=0.20, reversal=0.10):
    """[lo, hi) 内启动且见顶的大涨:第一次提醒时已走完的比例;没有提醒记漏报。"""
    done, n = [], 0
    for sym, f in frames.items():
        x = f["xs_lc"]
        al = alerts_by_sym.get(sym, np.array([], dtype="int64"))
        for t0, t1, _ in L.zigzag_rallies(x, min_gain, reversal):
            if t0 < lo or t1 >= hi:
                continue
            n += 1
            a = al[(al >= t0) & (al <= t1)]
            if len(a) == 0:
                done.append(np.nan)
                continue
            x0, x1 = x.get(t0), x.get(t1)
            done.append((x.get(int(a[0])) - x0) / (x1 - x0))
    d = pd.Series(done, dtype=float)
    return {"n": n, "done_med": float(d.median()) if d.notna().any() else np.nan, "miss": float(d.isna().mean()) if n else np.nan}


def evaluate(A: pd.DataFrame, base_rate: float, regimes: pd.Series | None) -> dict:
    st = K.event_stats(A.ts.to_numpy(), A.ex168.to_numpy(), H)
    out = {"n": len(A), "hit": float(A.y.mean()), "lift": float(A.y.mean() / base_rate) if base_rate else np.nan,
           "ex_mean": st.get("mean"), "ex_t": st.get("t")}
    if regimes is not None:
        g = regimes.reindex(A.ts.to_numpy()).to_numpy()
        for lab in pd.unique(g[pd.notna(g)]):
            sub = A[g == lab]
            out[f"ex_{lab}"] = float(sub.ex168.mean()) if len(sub) else np.nan
            out[f"n_{lab}"] = int(len(sub))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    ap.add_argument("--rules-config", default=str(Path(__file__).with_name("cloud_config.yaml")))
    ap.add_argument("--spot", action="store_true")
    ap.add_argument("--fund-history")
    ap.add_argument("--start", default="2025-03-01")
    ap.add_argument("--holdout-start", default=K.HOLDOUT_START)
    ap.add_argument("--out", default=str(Path(__file__).with_name("reports")))
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    frames = load_frames(load_config(args.config), args.spot, args.fund_history)
    C = cum_excess(frames)
    M = market_features(frames)
    D = build_dataset(frames, C, M)
    start = int(pd.Timestamp(args.start, tz="UTC").timestamp() * 1000)
    hold = int(pd.Timestamp(args.holdout_start, tz="UTC").timestamp() * 1000)
    D = D[(D.ts >= start) & D.y.notna()].reset_index(drop=True)
    dev_m, hold_m = K.split_holdout(D.ts.to_numpy(), args.holdout_start, H)
    DEV, HOLD = D[dev_m].reset_index(drop=True), D[hold_m].reset_index(drop=True)
    log.info("样本:开发期 %d 行(正例 %.1f%%),保留集 %d 行(正例 %.1f%%)", len(DEV), DEV.y.mean() * 100, len(HOLD), HOLD.y.mean() * 100)
    reg = K.regimes(frames)["btc_trend"]
    X = lambda d: d[FEATS]

    # 现有"读起来像在涨"的推送(和云端同样的规则、门槛、冷却)
    rc = load_config(args.rules_config)
    sc = rc["signals"]
    th, rules = merged_thresholds(sc.get("thresholds")), apply_weights(sc.get("rule_weights"))
    watch = {s.upper() for s in rc["universe"].get("watchlist", [])}
    push_rows, push_by_sym = [], {}
    for sym, f in frames.items():
        need = sc.get("watchlist_min_score", 1.0) if sym in watch else sc.get("min_score_to_push", 2.5)
        hits, pushes = L.replay(f, th, rules, need, float(sc.get("cooldown_hours", 6)))
        up = L.up_flavored(f, hits)
        t = pushes.index[up.reindex(pushes.index).fillna(False).to_numpy(dtype=bool)].to_numpy(dtype="int64")
        push_by_sym[sym] = t
        for t0 in t:
            y, ex = label_at(C, sym, int(t0))
            push_rows.append({"symbol": sym, "ts": int(t0), "y": y, "ex168": ex})
    P = pd.DataFrame(push_rows).dropna(subset=["y"])

    out = Path(args.out)
    out.mkdir(exist_ok=True)
    md = ["# 统一模型 + 新预测目标:离线研究结果\n",
          "事先登记见 `studies/unified_preregistration.md`(提交于跑结果之前)。目标 A = 168 小时内相对全市场先涨 15%、"
          "且之前没有先跌 10%。提醒 = 每 6 小时分数最高的 3 个币,同一币 72 小时内只算一次。\n",
          f"样本:开发期 {len(DEV)} 个(币 × 时点),目标 A 正例 {DEV.y.mean():.1%};保留集 {len(HOLD)} 个,正例 {HOLD.y.mean():.1%}。\n"]

    # 开发期滚动检验(只看稳定性)
    from sklearn.metrics import roc_auc_score
    md += ["## 1. 开发期滚动检验(4 段,训练与检验隔 168 小时;只看稳定性,不用于选择)\n",
           "| 模型 | 段 | AUC | 提醒命中率 | 倍数(对随机) | 提醒 168h 平均超额 |", "|---|---|---|---|---|---|"]
    for M_ in (Logit, GBM):
        for k, (tr, te) in enumerate(K.purged_walk_forward(DEV.ts.to_numpy(), H, folds=4, first_train=0.4), 1):
            m = M_().fit(X(DEV[tr]), DEV.y[tr])
            s = m.score(X(DEV[te]))
            A = alerts_from_scores(DEV[te], s)
            auc = roc_auc_score(DEV.y[te], s) if DEV.y[te].nunique() > 1 else np.nan
            md.append(f"| {M_.name} | {k} | {auc:.3f} | {A.y.mean():.1%} | {A.y.mean() / DEV.y[te].mean():.2f} | "
                      f"{A.ex168.mean() * 100:+.2f}% |")

    # 保留集:整个开发期训练一次,评一次
    base = HOLD.y.mean()
    Pd = P[(P.ts >= hold) & (P.ts < D.ts.max())]
    res = {"现有\"像在涨\"推送": evaluate(Pd, base, reg)}
    early_res = {"现有\"像在涨\"推送": earliness(frames, {s: t[t >= hold] for s, t in push_by_sym.items()}, hold, int(D.ts.max()))}
    models = {}
    for M_ in (Logit, GBM):
        m = M_().fit(X(DEV), DEV.y)
        models[M_.name] = m
        s = m.score(X(HOLD))
        A = alerts_from_scores(HOLD, s)
        A.assign(model=M_.name).to_csv(out / f"unified_alerts_{'logit' if M_ is Logit else 'gbm'}.csv", index=False)
        res[M_.name] = evaluate(A, base, reg)
        res[M_.name]["auc"] = roc_auc_score(HOLD.y, s)
        res[M_.name]["ic"] = float(pd.Series(s).groupby(HOLD.ts.to_numpy()).rank(pct=True)
                                   .corr(HOLD.y_rank.reset_index(drop=True), method="spearman"))
        early_res[M_.name] = earliness(frames, {sym: g.ts.to_numpy(dtype="int64") for sym, g in A.groupby("symbol")},
                                       hold, int(D.ts.max()))
    pf = lambda v: "—" if v is None or pd.isna(v) else f"{v * 100:+.2f}%"
    tf = lambda v: "—" if v is None or pd.isna(v) else f"{v:+.1f}"
    md += ["", f"## 2. 保留集(整个开发期训练一次;随机时点的目标 A 命中率 {base:.1%})\n",
           "| 口径 | 提醒数 | 命中率 | 倍数 | 168h 平均超额 | t | BTC 30 日上涨组超额 | BTC 30 日下跌组超额 | 大涨第一次提醒时已走完(中位) | 整段漏报 | AUC | IC |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    up_lab, dn_lab = "BTC 30 日上涨", "BTC 30 日下跌"
    for k, r in res.items():
        e = early_res[k]
        md.append(f"| {k} | {r['n']} | {r['hit']:.1%} | {r['lift']:.2f} | {pf(r['ex_mean'])} | {tf(r['ex_t'])} | "
                  f"{pf(r.get('ex_' + up_lab))}({r.get('n_' + up_lab, 0)}) | {pf(r.get('ex_' + dn_lab))}({r.get('n_' + dn_lab, 0)}) | "
                  f"{e['done_med']:.0%} | {e['miss']:.0%} | {r.get('auc', float('nan')):.3f} | {r.get('ic', float('nan')):+.3f} |")
    md.append(f"\n保留集里相对 BTC 涨幅 ≥ 20% 的大涨共 {early_res['逻辑回归']['n']} 段。\n")

    # 事先登记的 4 条合格标准
    cur, ce = res["现有\"像在涨\"推送"], early_res["现有\"像在涨\"推送"]
    md += ["## 3. 合格标准(事先登记,4 条全部满足才进入云端影子运行)\n",
           "| 模型 | 1 更准(≥1.5 倍且高于现有推送) | 2 有超额(>0 且 t≥2.5) | 3 更早(已走完低 ≥10 个百分点且漏报不更多) | 4 环境稳定(两组都 >0) | 结论 |",
           "|---|---|---|---|---|---|"]
    verdicts = {}
    for k in ("逻辑回归", "梯度提升树"):
        r, e = res[k], early_res[k]
        c1 = r["lift"] >= 1.5 and r["hit"] > cur["hit"]
        c2 = (r["ex_mean"] or 0) > 0 and (r["ex_t"] or 0) >= 2.5
        c3 = (ce["done_med"] - e["done_med"]) >= 0.10 and e["miss"] <= ce["miss"]
        c4 = all((r.get(f"ex_{lab}") or -1) > 0 for lab in (up_lab, dn_lab))
        verdicts[k] = all((c1, c2, c3, c4))
        mk = lambda b: "✅" if b else "❌"
        md.append(f"| {k} | {mk(c1)} | {mk(c2)} | {mk(c3)} | {mk(c4)} | {'通过' if verdicts[k] else '不通过'} |")
    md.append("")

    # 逻辑回归系数(哪些输入在起作用)
    co = models["逻辑回归"].coefs(X(DEV))
    co = co[~co.index.str.endswith("_na")].sort_values(key=np.abs, ascending=False)
    md += ["## 4. 逻辑回归系数(标准化后;正 = 越大越可能启动)\n", "| 特征 | 系数 |", "|---|---|"]
    md += [f"| {k} | {v:+.3f} |" for k, v in co.head(15).items()]
    p = out / "unified_study.md"
    p.write_text("\n".join(md), encoding="utf-8")
    print(p.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
