"""阈值调参 + 评分模型的滚动检验(walk-forward)。

research.py 回答"每条规则有没有用";本脚本回答"调参和建模之后,在没见过的时间段还有没有用"。

做法:把全部时间切成几段,每一轮"用前面的历史选参数 / 训练模型,到紧接着的下一段检验"。
训练和检验之间留 72 小时空档(标签要看未来 72 小时,不留空档会泄漏)。
所有检验结果都来自样本外(训练时没见过的时间段),这才是对实盘有参考价值的数字。

比较三种触发方式(都只看做多方向:触发后 72 小时的 BTC 残差收益):
  基准      任意时点
  当前规则  现在的阈值 + 规则权重合计 ≥ min_score
  调参规则  每轮在训练期网格搜索阈值,选训练期 t 值最高的组合,拿到检验期去看
  评分模型  逻辑回归预测"未来 72h 残差收益为正"的概率,取训练期得分最高的 5% 作为触发

用法:
  python tune.py                  用数据库里所有已回填的币
  python tune.py --folds 4 --top-pct 5 --min-n 30
结果保存在 reports/tune_*.csv 和 reports/tune_report.md
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


# ------------------------------------------------------------------ 规则:当前 vs 网格调参
def rule_trigger(f: pd.DataFrame, th: dict, min_score: float) -> np.ndarray:
    hits = evaluate_frame(f, th)
    w = pd.Series({r.id: r.weight for r in RULES})
    return ((hits * w).sum(axis=1) >= min_score).to_numpy()


def grid_triggers(frames: list[pd.DataFrame], base_th: dict) -> dict[tuple, list[np.ndarray]]:
    """每个参数组合在全部历史上算一次触发(只用当时已知的特征,所以可以先算好再按时间段切)。"""
    out = {}
    keys = list(GRID)
    combos = list(itertools.product(*GRID.values()))
    for i, combo in enumerate(combos, 1):
        p = dict(zip(keys, combo))
        th = {**base_th, "oi_z": p["oi_z"], "resid_z": p["resid_z"], "funding_z": p["funding_z"]}
        out[combo] = [rule_trigger(f, th, p["min_score"]) for f in frames]
        if i % 20 == 0:
            log.info("网格 %d/%d", i, len(combos))
    return out


# ------------------------------------------------------------------ 评分模型:逻辑回归(只用 numpy)
class Logit:
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


def design(f: pd.DataFrame) -> np.ndarray:
    return f[FEATURES].clip(-5, 5).fillna(0.0).to_numpy()


def train_model(frames: list[pd.DataFrame], hi: int, stride: int = 6) -> tuple[Logit, float]:
    """用 ts < hi - 空档 的数据训练(隔 stride 小时取一个点,减少自相关),返回模型和训练期得分的分位函数基础。"""
    Xs, ys = [], []
    for f in frames:
        g = f[(f.index < hi - EMBARGO) & f[LABELS].notna().all(axis=1)].iloc[::stride]
        if len(g):
            Xs.append(design(g))
            ys.append((g["fwd_resid_72h"] > 0).to_numpy(float))
    X, y = np.vstack(Xs), np.concatenate(ys)
    return Logit().fit(X, y), float(y.mean())


# ------------------------------------------------------------------ 主流程
def run_tune(frames: list[pd.DataFrame], base_th: dict, folds: int = 4, top_pct: float = 5.0,
             min_n: int = 30, first_train: float = 0.4, edges: list[int] | None = None):
    edges = edges or fold_edges(frames, folds, first_train)
    log.info("计算 %d 个参数组合的触发……", np.prod([len(v) for v in GRID.values()]))
    grid = grid_triggers(frames, base_th)
    default_trig = [rule_trigger(f, base_th, BASE_MIN_SCORE) for f in frames]

    rows, chosen = [], []
    last_model = None
    pooled: dict[str, list[pd.DataFrame]] = {"基准": [], "当前规则": [], "调参规则": [], "评分模型": []}
    for k in range(folds):
        train_hi, lo, hi = edges[k], edges[k], edges[k + 1]
        tr_lo, tr_hi = int(min(f.index.min() for f in frames)), train_hi - EMBARGO
        assert tr_hi < lo, "训练期必须在检验期之前"

        # 调参:训练期 t 值最高(n 足够)的组合
        best, best_t = None, -np.inf
        for combo, trigs in grid.items():
            s = summarize(collect(frames, trigs, tr_lo, tr_hi))
            if s["n"] >= min_n and pd.notna(s["t72"]) and s["t72"] > best_t:
                best, best_t = combo, s["t72"]
        chosen.append({"fold": k + 1, **(dict(zip(GRID, best)) if best else {}), "train_t72": best_t})

        # 模型
        model, _ = train_model(frames, train_hi)
        last_model = model
        tr_scores = np.concatenate([
            model.score(design(f[(f.index < tr_hi) & (f.index >= tr_lo)])) for f in frames])
        thr = np.quantile(tr_scores, 1 - top_pct / 100)
        model_trig = [model.score(design(f)) >= thr for f in frames]

        sets = {
            "基准": baseline_rows(frames, lo, hi),
            "当前规则": collect(frames, default_trig, lo, hi),
            "调参规则": collect(frames, grid[best], lo, hi) if best else frames[0].iloc[0:0],
            "评分模型": collect(frames, model_trig, lo, hi),
        }
        for name, sub in sets.items():
            rows.append({"fold": k + 1, "方法": name, **summarize(sub)})
            pooled[name].append(sub)

    per_fold = pd.DataFrame(rows)
    total = pd.DataFrame([{"方法": name, **summarize(pd.concat(parts))} for name, parts in pooled.items()])
    return per_fold, total, pd.DataFrame(chosen), last_model


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


def write_report(out_dir: Path, per_fold, total, chosen, model, args) -> Path:
    out_dir.mkdir(exist_ok=True)
    per_fold.to_csv(out_dir / "tune_per_fold.csv", index=False, encoding="utf-8-sig")
    total.to_csv(out_dir / "tune_total.csv", index=False, encoding="utf-8-sig")
    chosen.to_csv(out_dir / "tune_chosen_params.csv", index=False, encoding="utf-8-sig")
    coefs = "\n".join(f"- {k}: {v:+.3f}" for k, v in model.coefs().items())
    md = [
        "# 调参与评分模型:滚动检验报告\n",
        f"检验轮数 {args.folds},模型触发取训练期得分最高的 {args.top_pct}%,事件至少间隔 {GAP} 小时。"
        "下表全部来自样本外时间段。\n",
        "## 汇总(所有检验期合并)\n", fmt(total).to_markdown(index=False) if _has_tabulate()
        else fmt(total).to_string(index=False), "\n",
        "## 每轮结果\n", fmt(per_fold).to_string(index=False), "\n",
        "## 每轮在训练期选中的阈值\n", chosen.to_string(index=False), "\n",
        "## 评分模型系数(最后一轮,标准化特征;正数=该特征越高,未来 72h 越可能跑赢 BTC)\n", coefs, "\n",
        "## 怎么读\n",
        "- 方法 vs 基准:`resid72_mean` 和 `hit72` 要明显更高,且 `t72` ≥ 2.5 才值得关注。\n"
        "- 如果「调参规则」在训练期很好、这里却不如「当前规则」,说明是过拟合,不要采用。\n"
        "- `safe_lev` 是让 90% 的事件不被强平的杠杆上限近似值,比平均收益更能说明能不能上杠杆。\n"
        "- 每轮样本量小时单轮结果波动很大,看汇总和方向是否一致。\n",
    ]
    p = out_dir / "tune_report.md"
    p.write_text("\n".join(md), encoding="utf-8")
    return p


def _has_tabulate() -> bool:
    try:
        import tabulate  # noqa: F401
        return True
    except ImportError:
        return False


def main() -> None:
    ap = argparse.ArgumentParser(description="阈值调参 + 评分模型滚动检验")
    ap.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--top-pct", type=float, default=5.0, help="模型触发:训练期得分最高的百分之几")
    ap.add_argument("--min-n", type=int, default=30, help="训练期事件数少于该值的参数组合不参选")
    ap.add_argument("--symbols", help="逗号分隔;默认用库里所有已回填的币")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cfg = load_config(args.config)
    store = Store(cfg["storage"]["db_path"])
    th = merged_thresholds(cfg["signals"].get("thresholds"))
    syms = [s.strip().upper() for s in args.symbols.split(",")] if args.symbols else None
    frames = load_frames(store, th, syms)

    per_fold, total, chosen, model = run_tune(frames, th, args.folds, args.top_pct, args.min_n)
    path = write_report(Path(cfg["_base_dir"]) / "reports", per_fold, total, chosen, model, args)
    pd.set_option("display.width", 250)
    print(fmt(total).to_string(index=False))
    print(f"\n完整报告:{path}")


if __name__ == "__main__":
    main()
