"""阈值调参 + 评分模型的滚动检验(walk-forward)。

research.py 回答"每条规则有没有用";本脚本回答"调参和建模之后,在没见过的时间段还有没有用"。

做法:把全部时间切成几段,每一轮"用前面的历史选参数 / 训练模型,到紧接着的下一段检验"。
训练和检验之间留 72 小时空档(标签要看未来 72 小时,不留空档会泄漏)。
所有检验结果都来自样本外(训练时没见过的时间段),这才是对实盘有参考价值的数字。

比较六种触发方式(都只看做多方向:触发后 72 小时的 BTC 残差收益):
  基准        任意时点
  当前规则    现在的阈值和权重 + 权重合计 ≥ min_score
  调参规则    每轮在训练期网格搜索阈值,按"满足 safe_lev 约束下 t 值最高"选组合
  学习权重    规则权重由数据学出(规则命中 → 逻辑回归系数,负系数的规则权重记 0 即砍掉),
              再在验证段选触发分数线
  逻辑回归    预测"未来 72h 残差收益为正"的概率,在验证段选取前 N% 作为触发
  梯度提升    同上,用 scikit-learn 的 HistGradientBoosting(没装则跳过)

选参统一规则:训练段内部再切 70% 拟合 / 30% 验证,参数在验证段按
"事件数 ≥ min_n、平均残差收益 > 0、safe_lev ≥ min_safe_lev 的前提下 t 值最高"来选;
都不满足就这一轮不触发(不勉强给结果)。检验段只看一次,训练和检验之间隔 72 小时。

两种检验同时给出:
  滚动检验   前 40% 起步,后面分 4 段逐段检验(walk-forward)
  70/30      前 70% 训练,后 30% 一次性检验

用法:
  python tune.py                  用数据库里所有已回填的币
  python tune.py --folds 4 --min-safe-lev 3 --pcts 2,5,10
结果保存在 reports/tune_*.csv、reports/tune_report.md 和 reports/suggested_weights.yaml
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
from research import GAP, decluster

log = logging.getLogger("tune")

HOUR = 3_600_000
FEATURES = ["oi_z", "resid_24h_z", "ret_1h_z", "funding_z", "vol_z", "top_ls_z", "taker_z"]
LABELS = ["fwd_resid_72h", "fwd_ret_72h", "mae_72h"]
GRID = {
    "oi_z": [1.5, 2.0, 2.5, 3.0],
    "resid_z": [2.0, 2.5, 3.0],
    "funding_z": [2.0, 2.5, 3.0],
    "min_score": [1.5, 2.5, 3.5],
}
BASE_MIN_SCORE = 2.5
EMBARGO = 72 * HOUR


# ------------------------------------------------------------------ 数据
def load_frames(store: Store, th: dict, symbols: list[str] | None = None) -> list[pd.DataFrame]:
    btc, eth = store.load_hourly("BTCUSDT"), store.load_hourly("ETHUSDT")
    if len(btc) < 24 * 60:
        raise SystemExit("BTC 历史数据不足,请先运行 python backfill.py")
    if symbols is None:
        symbols = [r[0] for r in store.conn.execute(
            "SELECT symbol FROM hourly WHERE oi IS NOT NULL GROUP BY symbol HAVING COUNT(*) > 2000")]
        symbols = [s for s in symbols if s != "BTCUSDT"]
    frames = []
    for s in symbols:
        df = store.load_hourly(s)
        if len(df) < 24 * 60:
            continue
        f = add_labels(build_features(df, btc, store.load_funding(s), 8.0, eth))
        f["symbol"] = s
        frames.append(f)
        log.info("%s:%d 小时样本", s, len(f))
    if not frames:
        raise SystemExit("没有可用数据(先用 backfill.py 回填)")
    return frames


def fold_edges(frames: list[pd.DataFrame], folds: int, first_train: float = 0.4) -> list[int]:
    """第一段训练期占 first_train,其余等分成 folds 个检验期。返回 folds+1 个时间边界。"""
    ts = pd.Series(np.concatenate([f.index.to_numpy() for f in frames]))
    qs = [first_train + (1 - first_train) * i / folds for i in range(folds)] + [1.0]
    edges = [int(ts.quantile(q)) for q in qs]
    edges[-1] += 1
    return edges


# ------------------------------------------------------------------ 统计
def summarize(rows: pd.DataFrame) -> dict:
    n = len(rows)
    if n == 0:
        return {"n": 0, "resid72_mean": np.nan, "resid72_median": np.nan, "hit72": np.nan,
                "t72": np.nan, "mae72_p10": np.nan, "safe_lev": np.nan}
    x = rows["fwd_resid_72h"]
    sd = x.std()
    p10 = rows["mae_72h"].quantile(0.10)
    return {
        "n": n,
        "resid72_mean": x.mean(),
        "resid72_median": x.median(),
        "hit72": (x > 0).mean(),
        "t72": x.mean() / (sd / np.sqrt(n)) if n > 1 and sd > 0 else np.nan,
        "mae72_p10": p10,
        "safe_lev": 1 / abs(p10) if p10 < 0 else np.nan,
    }


def pick_events(f: pd.DataFrame, trig: np.ndarray, lo: int, hi: int) -> pd.DataFrame:
    """在 [lo, hi) 内,标签完整的触发点去重叠后的事件行。"""
    ok = f[LABELS].notna().all(axis=1).to_numpy()
    m = trig & ok & (f.index.to_numpy() >= lo) & (f.index.to_numpy() < hi)
    if not m.any():
        return f.iloc[0:0]
    ev = decluster(pd.Series(m, index=f.index))
    return f.loc[ev]


def collect(frames: list[pd.DataFrame], trigs: list[np.ndarray], lo: int, hi: int) -> pd.DataFrame:
    parts = [pick_events(f, t, lo, hi) for f, t in zip(frames, trigs)]
    parts = [p for p in parts if len(p)]
    return pd.concat(parts) if parts else frames[0].iloc[0:0]


def baseline_rows(frames: list[pd.DataFrame], lo: int, hi: int) -> pd.DataFrame:
    parts = []
    for f in frames:
        ok = f[LABELS].notna().all(axis=1)
        g = f[ok & (f.index >= lo) & (f.index < hi)].iloc[::GAP]
        if len(g):
            parts.append(g)
    return pd.concat(parts)




# ------------------------------------------------------------------ 选参:约束 + 目标
def eligible(s: dict, min_n: int, min_lev: float) -> bool:
    """能被选中的条件:事件数够、平均残差收益为正、safe_lev 不低于约束。"""
    return bool(s["n"] >= min_n and pd.notna(s["t72"]) and s["resid72_mean"] > 0
                and pd.notna(s["safe_lev"]) and s["safe_lev"] >= min_lev)


def pick_best(frames, scores, lo, hi, cands, min_n, min_lev):
    """scores: 每个 frame 一条得分序列。在 [lo,hi) 内对每个候选分数线算事件统计,
    返回 (分数线, 统计),满足约束且 t 值最高;没有满足的返回 (None, None)。"""
    best_thr, best_s = None, None
    for thr in cands:
        s = summarize(collect(frames, [sc >= thr for sc in scores], lo, hi))
        if eligible(s, min_n, min_lev) and (best_s is None or s["t72"] > best_s["t72"]):
            best_thr, best_s = thr, s
    return best_thr, best_s


# ------------------------------------------------------------------ 规则:当前 vs 网格调参 vs 学习权重
RULE_IDS = [r.id for r in RULES]
CUR_W = np.array([r.weight for r in RULES])


def hit_matrix(frames: list[pd.DataFrame], th: dict) -> list[np.ndarray]:
    return [evaluate_frame(f, th)[RULE_IDS].to_numpy(float) for f in frames]


def rule_trigger(f: pd.DataFrame, th: dict, min_score: float) -> np.ndarray:
    return (evaluate_frame(f, th)[RULE_IDS].to_numpy(float) @ CUR_W >= min_score)


def grid_scores(frames: list[pd.DataFrame], base_th: dict) -> dict[tuple, list[np.ndarray]]:
    """按 (oi_z, resid_z, funding_z) 三个阈值各算一次规则得分;min_score 只是分数线,不用重算。"""
    out = {}
    combos = list(itertools.product(GRID["oi_z"], GRID["resid_z"], GRID["funding_z"]))
    for i, (a, b, c) in enumerate(combos, 1):
        th = {**base_th, "oi_z": a, "resid_z": b, "funding_z": c}
        out[(a, b, c)] = [h @ CUR_W for h in hit_matrix(frames, th)]
        if i % 12 == 0:
            log.info("网格 %d/%d", i, len(combos))
    return out


def fit_weights(frames, hits, lo, hi, stride=6, scale=2.0, l2=20.0) -> np.ndarray:
    """规则权重由数据决定:规则命中(0/1)预测"未来 72h 残差收益为正",用岭正则的逻辑回归。
    系数 ≤ 0 的规则权重记 0(等于砍掉);其余按比例缩放到最大权重 = scale,保持和现有权重同一量级。"""
    X, y = sample_xy(frames, hits, lo, hi, stride)
    m = Logit(l2=l2).fit(X, y)
    coef = np.maximum(m.w[1:] / m.sd, 0.0)
    return coef / coef.max() * scale if coef.max() > 0 else coef


# ------------------------------------------------------------------ 评分模型
class Logit:
    """逻辑回归(只用 numpy)。"""
    def __init__(self, l2: float = 1.0, iters: int = 25):
        self.l2, self.iters = l2, iters

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Logit":
        self.mu, self.sd = X.mean(0), X.std(0)
        self.sd[self.sd == 0] = 1.0
        Z = np.c_[np.ones(len(X)), (X - self.mu) / self.sd]
        w = np.zeros(Z.shape[1])
        reg = self.l2 * np.eye(Z.shape[1])
        reg[0, 0] = 0
        for _ in range(self.iters):
            p = 1 / (1 + np.exp(-np.clip(Z @ w, -30, 30)))
            grad = Z.T @ (p - y) + reg @ w
            hess = (Z * (p * (1 - p))[:, None]).T @ Z + reg
            step = np.linalg.solve(hess, grad)
            w -= step
            if np.abs(step).max() < 1e-6:
                break
        self.w = w
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        return np.c_[np.ones(len(X)), (X - self.mu) / self.sd] @ self.w

    def coefs(self) -> dict:
        return dict(zip(FEATURES, self.w[1:]))


class GBM:
    """梯度提升(scikit-learn HistGradientBoosting),浅树 + 强正则 + 大叶子,防过拟合。"""
    def fit(self, X: np.ndarray, y: np.ndarray) -> "GBM":
        from sklearn.ensemble import HistGradientBoostingClassifier
        self.m = HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.05, max_iter=150, l2_regularization=5.0,
            min_samples_leaf=300, random_state=0).fit(X, y)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        return self.m.predict_proba(X)[:, 1]

    def importance(self, X: np.ndarray, y: np.ndarray, n: int = 6000) -> dict:
        """在验证样本上做置换重要性:打乱某个特征后准确度(AUC)掉多少。"""
        from sklearn.inspection import permutation_importance
        rng = np.random.default_rng(0)
        idx = rng.choice(len(X), min(n, len(X)), replace=False)
        r = permutation_importance(self.m, X[idx], y[idx], scoring="roc_auc", n_repeats=3, random_state=0)
        return dict(zip(FEATURES, r.importances_mean))


def have_sklearn() -> bool:
    try:
        import sklearn  # noqa: F401
        return True
    except ImportError:
        return False


def design(f: pd.DataFrame) -> np.ndarray:
    return f[FEATURES].clip(-5, 5).fillna(0.0).to_numpy()


def sample_xy(frames, mats, lo, hi, stride=6):
    """lo ≤ ts < hi 且标签完整的点,隔 stride 小时取一个(减少自相关)。mats 与 frames 行对齐。"""
    Xs, ys = [], []
    for f, M in zip(frames, mats):
        ts = f.index.to_numpy()
        m = (ts >= lo) & (ts < hi) & f[LABELS].notna().all(axis=1).to_numpy()
        idx = np.flatnonzero(m)[::stride]
        if len(idx):
            Xs.append(M[idx])
            ys.append((f["fwd_resid_72h"].to_numpy()[idx] > 0).astype(float))
    return np.vstack(Xs), np.concatenate(ys)


def model_cands(scores, frames, lo, hi, pcts):
    """候选分数线 = 验证段得分的 (100-p) 分位数。"""
    vals = np.concatenate([s[(f.index.to_numpy() >= lo) & (f.index.to_numpy() < hi)]
                           for f, s in zip(frames, scores)])
    return [(p, float(np.quantile(vals, 1 - p / 100))) for p in pcts]


# ------------------------------------------------------------------ 主流程
METHODS = ["基准", "当前规则", "调参规则", "学习权重", "逻辑回归", "梯度提升"]
MIN_SCORES = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]


def run_tune(frames: list[pd.DataFrame], base_th: dict, folds: int = 4, top_pct: float = 5.0,
             min_n: int = 30, first_train: float = 0.4, edges: list[int] | None = None,
             min_lev: float = 3.0, pcts: list[float] | None = None, inner: float = 0.7) -> dict:
    edges = edges or fold_edges(frames, folds, first_train)
    pcts = sorted(set(pcts or [2.0, 5.0, 10.0]) | {top_pct})
    use_gbm = have_sklearn()
    if not use_gbm:
        log.warning("没有安装 scikit-learn,跳过梯度提升(pip install scikit-learn)")
    min_n_val = max(8, min_n // 2)

    log.info("计算规则命中与网格得分……")
    grid = grid_scores(frames, base_th)
    base_hits = hit_matrix(frames, base_th)
    D = [design(f) for f in frames]
    default_trig = [h @ CUR_W >= BASE_MIN_SCORE for h in base_hits]
    t0 = int(min(f.index.min() for f in frames))

    rows, chosen, wrows = [], [], []
    names = [m for m in METHODS if m != "梯度提升" or use_gbm]
    pooled: dict[str, list[pd.DataFrame]] = {m: [] for m in names}
    models: dict = {}
    for k in range(folds):
        lo, hi = edges[k], edges[k + 1]
        tr_hi = lo - EMBARGO                      # 训练段末尾:标签看 72h,所以留空档
        cut = int(t0 + inner * (tr_hi - t0))      # 训练段内部:前 70% 拟合,后 30% 验证
        fit_hi = cut - EMBARGO
        assert fit_hi > t0 and tr_hi < lo, "训练段必须在验证段之前、验证段必须在检验段之前"
        info = {"fold": k + 1}

        # --- 调参规则:训练段整段上选(只选 4 个数,不拟合参数)
        best, best_s = None, None
        for combo, sc in grid.items():
            for ms in GRID["min_score"]:
                s = summarize(collect(frames, [x >= ms for x in sc], t0, tr_hi))
                if eligible(s, min_n, min_lev) and (best_s is None or s["t72"] > best_s["t72"]):
                    best, best_s = (*combo, ms), s
        info.update({"调参规则": "无满足约束" if best is None else
                     f"oi_z={best[0]} resid_z={best[1]} funding_z={best[2]} min_score={best[3]}"})
        tuned_trig = None if best is None else [x >= best[3] for x in grid[best[:3]]]

        # --- 学习权重:拟合段学权重,验证段选分数线
        w = fit_weights(frames, base_hits, t0, fit_hi)
        wsc = [h @ w for h in base_hits]
        thr_w, _ = pick_best(frames, wsc, cut, tr_hi, MIN_SCORES, min_n_val, min_lev)
        info["学习权重"] = "无满足约束" if thr_w is None else f"min_score={thr_w}"
        wrows.append({"fold": k + 1, **dict(zip(RULE_IDS, np.round(w, 2)))})
        weight_trig = None if thr_w is None else [s >= thr_w for s in wsc]

        # --- 逻辑回归 / 梯度提升:拟合段训练,验证段选前 N%
        model_trig = {}
        X, y = sample_xy(frames, D, t0, fit_hi)
        for name, mdl in (("逻辑回归", Logit()), ("梯度提升", GBM() if use_gbm else None)):
            if mdl is None:
                continue
            mdl.fit(X, y)
            msc = [mdl.score(d) for d in D]
            cands = model_cands(msc, frames, cut, tr_hi, pcts)
            thr, s = pick_best(frames, msc, cut, tr_hi, [c[1] for c in cands], min_n_val, min_lev)
            pct = next((p for p, c in cands if c == thr), None)
            info[name] = "无满足约束" if thr is None else f"前 {pct:g}%"
            model_trig[name] = None if thr is None else [s_ >= thr for s_ in msc]
            models[name] = mdl
            if name == "梯度提升" and k == folds - 1:
                Xv, yv = sample_xy(frames, D, cut, tr_hi)
                mdl.imp = mdl.importance(Xv, yv)
        chosen.append(info)

        empty = frames[0].iloc[0:0]
        sets = {
            "基准": baseline_rows(frames, lo, hi),
            "当前规则": collect(frames, default_trig, lo, hi),
            "调参规则": empty if tuned_trig is None else collect(frames, tuned_trig, lo, hi),
            "学习权重": empty if weight_trig is None else collect(frames, weight_trig, lo, hi),
            "逻辑回归": empty if model_trig["逻辑回归"] is None else collect(frames, model_trig["逻辑回归"], lo, hi),
        }
        if use_gbm:
            sets["梯度提升"] = empty if model_trig["梯度提升"] is None else collect(frames, model_trig["梯度提升"], lo, hi)
        for name, sub in sets.items():
            rows.append({"fold": k + 1, "方法": name, **summarize(sub)})
            pooled[name].append(sub)

    per_fold = pd.DataFrame(rows)
    total = pd.DataFrame([{"方法": name, **summarize(pd.concat(parts))} for name, parts in pooled.items()])
    return {"per_fold": per_fold, "total": total, "chosen": pd.DataFrame(chosen),
            "weights_by_fold": pd.DataFrame(wrows), "models": models, "methods": names}


def final_weights(frames, base_th, min_n: int, min_lev: float, inner: float = 0.7):
    """用全部带标签的历史给出"建议权重":前 70% 学权重,后 30% 选分数线。"""
    hits = hit_matrix(frames, base_th)
    t0 = int(min(f.index.min() for f in frames))
    t1 = int(max(f.index.max() for f in frames)) - EMBARGO
    cut = int(t0 + inner * (t1 - t0))
    w = fit_weights(frames, hits, t0, cut - EMBARGO)
    sc = [h @ w for h in hits]
    thr, s = pick_best(frames, sc, cut, t1, MIN_SCORES, max(8, min_n // 2), min_lev)
    return dict(zip(RULE_IDS, w)), thr, s


def verdict(total: pd.DataFrame, min_lev: float, min_total_n: int = 100) -> pd.DataFrame:
    """自动判断哪个方法值得替换现行规则(只看样本外):
    事件数够、t ≥ 2.5、平均残差收益高于基准和当前规则、safe_lev ≥ 约束。"""
    t = total.set_index("方法")
    base, cur = t.loc["基准"], t.loc["当前规则"]
    out = []
    for m, r in t.iterrows():
        if m == "基准":
            continue
        why = []
        if r["n"] < min_total_n:
            why.append(f"事件数 {int(r['n'])} < {min_total_n}")
        if not (pd.notna(r["t72"]) and r["t72"] >= 2.5):
            why.append("t < 2.5")
        if not (pd.notna(r["resid72_mean"]) and r["resid72_mean"] > base["resid72_mean"]):
            why.append("不高于基准")
        if m != "当前规则" and pd.notna(cur["resid72_mean"]) and not (r["resid72_mean"] > cur["resid72_mean"]):
            why.append("不高于当前规则")
        if not (pd.notna(r["safe_lev"]) and r["safe_lev"] >= min_lev):
            why.append(f"safe_lev < {min_lev:g}x")
        out.append({"方法": m, "结论": "值得采用" if not why else "不采用", "原因": "、".join(why) or "全部达标"})
    return pd.DataFrame(out)


def fmt(df: pd.DataFrame) -> pd.DataFrame:
    s = df.copy()
    for c in ["resid72_mean", "resid72_median", "hit72", "mae72_p10"]:
        if c in s:
            s[c] = (s[c].astype(float) * 100).map(lambda v: "" if pd.isna(v) else f"{v:+.2f}%")
    for c in ["t72"]:
        if c in s:
            s[c] = s[c].astype(float).map(lambda v: "" if pd.isna(v) else f"{v:+.2f}")
    if "safe_lev" in s:
        s["safe_lev"] = s["safe_lev"].astype(float).map(lambda v: "" if pd.isna(v) else f"{v:.1f}x")
    return s


def _tbl(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except ImportError:
        return df.to_string(index=False)


def write_report(out_dir: Path, results: dict, args, final=None) -> Path:
    """results: {"滚动检验": run_tune 结果, "70/30": run_tune 结果}。"""
    out_dir.mkdir(exist_ok=True)
    md = [
        "# 调参与评分模型:样本外检验报告\n",
        f"选参约束:事件数 ≥ {args.min_n}、平均残差收益 > 0、safe_lev ≥ {args.min_safe_lev:g}x 的前提下 t 值最高;"
        f"事件至少间隔 {GAP} 小时。下表全部来自样本外时间段。\n",
    ]
    for tag, res in results.items():
        slug = "wf" if tag == "滚动检验" else "split"
        res["per_fold"].to_csv(out_dir / f"tune_per_fold_{slug}.csv", index=False, encoding="utf-8-sig")
        res["total"].to_csv(out_dir / f"tune_total_{slug}.csv", index=False, encoding="utf-8-sig")
        res["chosen"].to_csv(out_dir / f"tune_chosen_{slug}.csv", index=False, encoding="utf-8-sig")
        md += [f"## {tag}:汇总\n", _tbl(fmt(res["total"])), "\n",
               f"### {tag}:是否值得替换现行规则(自动判断)\n", _tbl(verdict(res["total"], args.min_safe_lev)), "\n",
               f"### {tag}:每轮结果\n", fmt(res["per_fold"]).to_string(index=False), "\n",
               f"### {tag}:每轮在训练段选中的参数\n", res["chosen"].to_string(index=False), "\n",
               f"### {tag}:每轮学出的规则权重(0 = 被砍掉;现行权重见下)\n",
               res["weights_by_fold"].to_string(index=False), "\n"]
    wf = next(iter(results.values()))
    md += ["现行权重:" + "、".join(f"{r.id}={r.weight:g}" for r in RULES) + "\n"]
    lg = wf["models"]["逻辑回归"]
    md += ["## 逻辑回归系数(最后一轮,标准化特征;正数 = 该特征越高,未来 72h 越可能跑赢 BTC)\n",
           "\n".join(f"- {k}: {v:+.3f}" for k, v in lg.coefs().items()), "\n"]
    if "梯度提升" in wf["models"] and hasattr(wf["models"]["梯度提升"], "imp"):
        imp = wf["models"]["梯度提升"].imp
        md += ["## 梯度提升的特征重要性(最后一轮验证段,置换后 AUC 下降量;越大越重要,≈0 = 没用)\n",
               "\n".join(f"- {k}: {v:+.4f}" for k, v in sorted(imp.items(), key=lambda kv: -kv[1])), "\n"]
    if final:
        w, thr, s = final
        md += ["## 建议权重(用全部历史:前 70% 学权重,后 30% 选分数线)\n",
               "\n".join(f"- {k}: 现行 {RULES_BY[k].weight:g} → 建议 {v:.2f}" for k, v in w.items()), "\n",
               f"建议的触发分数线:{thr if thr is not None else '无满足约束的分数线'}\n"]
        if thr is not None:
            yaml_lines = ["# 仅在上面的汇总里「学习权重」被判定为「值得采用」时才拿去用",
                          "signals:", f"  min_score_to_push: {thr}", "  rule_weights:"]
            yaml_lines += [f"    {k}: {v:.2f}" for k, v in w.items()]
            (out_dir / "suggested_weights.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")
    md += ["## 怎么读\n",
           "- 方法 vs 基准:`resid72_mean` 和 `hit72` 要明显更高,且 `t72` ≥ 2.5 才值得关注。\n"
           "- 如果「调参规则」在训练期很好、这里却不如「当前规则」,说明是过拟合,不要采用。\n"
           "- `safe_lev` 是让 90% 的事件不被强平的杠杆上限近似值(未计手续费和维持保证金)。\n"
           "- 事件数为 0 / 「无满足约束」表示这一轮在训练段里没有任何参数同时满足事件数、正收益、safe_lev,宁可不触发。\n"
           "- 单轮样本量小时波动很大,看汇总和方向是否一致;滚动检验和 70/30 两种切法都通过才可信。\n"]
    p = out_dir / "tune_report.md"
    p.write_text("\n".join(md), encoding="utf-8")
    return p


RULES_BY = {r.id: r for r in RULES}


def main() -> None:
    ap = argparse.ArgumentParser(description="阈值/权重调参 + 评分模型样本外检验")
    ap.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--top-pct", type=float, default=5.0, help="模型触发的默认百分比(也在候选里)")
    ap.add_argument("--pcts", default="2,5,10", help="模型触发候选:验证段得分最高的百分之几")
    ap.add_argument("--min-n", type=int, default=30, help="训练段事件数少于该值的参数组合不参选")
    ap.add_argument("--min-safe-lev", type=float, default=3.0,
                    help="选参约束:safe_lev(90%% 事件不被强平的杠杆上限)至少多少倍")
    ap.add_argument("--symbols", help="逗号分隔;默认用库里所有已回填的币")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cfg = load_config(args.config)
    store = Store(cfg["storage"]["db_path"])
    th = merged_thresholds(cfg["signals"].get("thresholds"))
    syms = [s.strip().upper() for s in args.symbols.split(",")] if args.symbols else None
    frames = load_frames(store, th, syms)
    pcts = [float(x) for x in args.pcts.split(",")]
    kw = dict(top_pct=args.top_pct, min_n=args.min_n, min_lev=args.min_safe_lev, pcts=pcts)

    results = {
        "滚动检验": run_tune(frames, th, folds=args.folds, first_train=0.4, **kw),
        "70/30": run_tune(frames, th, folds=1, first_train=0.7, **kw),
    }
    final = final_weights(frames, th, args.min_n, args.min_safe_lev)
    path = write_report(Path(cfg["_base_dir"]) / "reports", results, args, final)
    pd.set_option("display.width", 250)
    for tag, res in results.items():
        print(f"\n== {tag} ==")
        print(fmt(res["total"]).to_string(index=False))
        print(verdict(res["total"], args.min_safe_lev).to_string(index=False))
    print(f"\n完整报告:{path}")


if __name__ == "__main__":
    main()
