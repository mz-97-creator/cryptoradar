"""72 小时机会模型:在监控名单里找出「波动大 / 上涨概率高 / 下跌概率高」的币,并给出回撤与可承受杠杆。

对每个币、每个小时,预测未来 72 小时(都是相对 BTC 剔除 beta 后的超额收益 resid72):
  P_up     resid72 > +T 的概率(默认 T = 5%)
  P_down   resid72 < -T 的概率
  波动幅度  未来 72h 最高价/最低价之间的幅度(取对数)
  回撤 q10  持有期最大回撤的 10% 分位(最坏的 1/10 情形);1/|q10| 即"90% 情形不被强平"的杠杆上限
模型用梯度提升,输入币自身特征 + 大盘行情特征(BTC 趋势/波动、市场广度、全市场资金费率等)。

评估(全部样本外,训练与检验之间隔 72 小时):
  - 按时间截面算 Spearman IC:每个时间点把所有币按预测排序,和真实结果比;截面间隔 72 小时,互不重叠,
    t 值用截面序列算,避免"同一时间很多币共享同一波行情"造成的虚高
  - 前 k 名 / 后 k 名的真实表现,扣除手续费与资金费率后的净收益
  - 概率校准:预测 60% 的时候实际是不是约 60%;Brier 技能分(对比"只看波动"的基线和历史平均)
  - 对照:波动靠"最近的波动",方向靠"现行规则得分"。模型必须赢过它们才有意义
  - 按大盘风格(BTC 30 天上行/横盘/下行)拆分
  - 滚动检验(前 40% 起步,后面 4 段)和固定的前 70% 训练 / 后 30% 检验

用法:
  python model.py                      评估 + 当前排名(结果在 reports/model_*.csv、model_report.md)
  python model.py --threshold 0.05 --topk 8 --cost 0.002
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from cryptoradar.config import load_config
from cryptoradar.signals import merged_thresholds
from cryptoradar.storage import Store
import tune

log = logging.getLogger("model")

HOUR = 3_600_000
H = 72
EMBARGO = H * HOUR
COIN_FEATS = ["ret_24h", "ret_72h", "ret_7d", "resid_24h_z", "ret_1h_z", "oi_chg_24h", "oi_z", "funding",
              "funding_z", "vol_z", "top_ls_z", "taker_z", "range_24h", "adr_14d", "range_z", "rv_24h",
              "rv_7d", "beta", "ethbtc_ret_24h"]
VOL_FEATS = ["range_24h", "adr_14d", "range_z", "rv_24h", "rv_7d"]
REGIME_FEATS = ["btc_ret_24h", "btc_ret_7d", "btc_ret_30d", "btc_rv_7d", "breadth", "mkt_funding",
                "mkt_oi_z", "mkt_ret_24h", "dispersion"]
FEATS = COIN_FEATS + REGIME_FEATS
LABEL_COLS = ["fwd_resid_72h", "fwd_ret_72h", "mae_72h", "mfe_72h"]


# ------------------------------------------------------------------ 数据
def build_table(frames: list[pd.DataFrame], th: dict) -> pd.DataFrame:
    """把各币的特征表拼成一张长表(每行 = 一个币的一个小时),并加上大盘行情特征。只用当时已知的信息。"""
    from cryptoradar.signals import RULES
    parts = []
    w = np.array([r.weight for r in RULES])
    hits = tune.hit_matrix(frames, th)
    for f, hm in zip(frames, hits):
        lc = np.log(f["close"])
        g = pd.DataFrame(index=f.index)
        g["symbol"] = f["symbol"].iloc[0]
        for c in ["ret_24h", "resid_24h_z", "ret_1h_z", "oi_chg_24h", "oi_z", "funding", "funding_z", "vol_z",
                  "top_ls_z", "taker_z", "range_24h", "adr_14d", "range_z", "beta", "btc_ret_24h"]:
            g[c] = f[c] if c in f else np.nan
        g["ethbtc_ret_24h"] = f["ethbtc_ret_24h"] if "ethbtc_ret_24h" in f else np.nan
        g["ret_72h"] = lc.diff(72)
        g["ret_7d"] = lc.diff(168)
        r1 = lc.diff()
        g["rv_24h"] = r1.rolling(24, min_periods=20).std()
        g["rv_7d"] = r1.rolling(168, min_periods=120).std()
        b = f["_btc_lc"]
        g["btc_ret_7d"] = b.diff(168)
        g["btc_ret_30d"] = b.diff(720)
        g["btc_rv_7d"] = b.diff().rolling(168, min_periods=120).std()
        g["above_sma7d"] = (f["close"] > f["close"].rolling(168, min_periods=120).mean()).astype(float)
        g["rule_score"] = hm @ w
        for c in LABEL_COLS:
            g[c] = f[c]
        g["close"] = f["close"]
        parts.append(g)
    D = pd.concat(parts)
    D.index.name = "ts"
    D = D.reset_index()
    cs = D.groupby("ts")
    D["breadth"] = D["ts"].map(cs["above_sma7d"].mean())
    D["mkt_funding"] = D["ts"].map(cs["funding"].median())
    D["mkt_oi_z"] = D["ts"].map(cs["oi_z"].median())
    D["mkt_ret_24h"] = D["ts"].map(cs["ret_24h"].median())
    D["dispersion"] = D["ts"].map(cs["ret_24h"].std())
    D["lrange"] = np.log((D["mfe_72h"] - D["mae_72h"]).clip(lower=1e-4))
    D["h"] = (D["ts"] // HOUR).astype("int64")
    num = FEATS + ["rule_score", "fwd_resid_72h", "fwd_ret_72h", "mae_72h", "mfe_72h", "lrange", "above_sma7d"]
    D[num] = D[num].astype("float32")
    return D


# ------------------------------------------------------------------ 模型
def _hgb_kwargs():
    return dict(max_depth=4, learning_rate=0.05, max_iter=200, l2_regularization=5.0,
                min_samples_leaf=500, random_state=0)


class Models:
    """四个预测目标 + 一个"只看波动"的基线。"""

    def __init__(self, thr: float):
        self.thr = thr

    def fit(self, tr: pd.DataFrame) -> "Models":
        from sklearn.ensemble import HistGradientBoostingClassifier as C, HistGradientBoostingRegressor as R
        X = tr[FEATS]
        up, dn = (tr["fwd_resid_72h"] > self.thr), (tr["fwd_resid_72h"] < -self.thr)
        self.base_up, self.base_dn = float(up.mean()), float(dn.mean())
        self.up = C(**_hgb_kwargs()).fit(X, up)
        self.dn = C(**_hgb_kwargs()).fit(X, dn)
        self.rng = R(**_hgb_kwargs()).fit(X, tr["lrange"])
        self.mae = R(loss="quantile", quantile=0.10, **_hgb_kwargs()).fit(X, tr["mae_72h"])
        Xv = tr[VOL_FEATS]
        self.up_v = C(**_hgb_kwargs()).fit(Xv, up)
        self.dn_v = C(**_hgb_kwargs()).fit(Xv, dn)
        return self

    def predict(self, d: pd.DataFrame) -> pd.DataFrame:
        X, Xv = d[FEATS], d[VOL_FEATS]
        out = pd.DataFrame(index=d.index)
        out["p_up"] = self.up.predict_proba(X)[:, 1]
        out["p_dn"] = self.dn.predict_proba(X)[:, 1]
        out["pred_lrange"] = self.rng.predict(X)
        out["pred_mae_q10"] = np.minimum(self.mae.predict(X), -1e-3)
        out["pv_up"] = self.up_v.predict_proba(Xv)[:, 1]
        out["pv_dn"] = self.dn_v.predict_proba(Xv)[:, 1]
        return out


# ------------------------------------------------------------------ 评估
def _ic_series(T: pd.DataFrame, score: str, target: str, min_n: int = 10) -> pd.Series:
    g = T.groupby("ts")
    rs, rt = g[score].rank(pct=True), g[target].rank(pct=True)
    X = pd.DataFrame({"ts": T["ts"], "a": rs, "b": rt})
    n = X.groupby("ts")["a"].transform("size")
    X = X[n >= min_n]
    if X.empty:
        return pd.Series(dtype=float)
    return X.groupby("ts").apply(lambda q: q["a"].corr(q["b"]), include_groups=False).dropna()


def _ic_stat(s: pd.Series) -> dict:
    if len(s) < 3:
        return {"截面数": len(s), "IC均值": np.nan, "t": np.nan, "IC>0占比": np.nan}
    return {"截面数": len(s), "IC均值": s.mean(), "t": s.mean() / (s.std() / np.sqrt(len(s))),
            "IC>0占比": (s > 0).mean()}


def _topk(T: pd.DataFrame, score: str, k: int, target: str, ascending: bool = False) -> pd.DataFrame:
    """每个截面里按 score 取前 k(或后 k),返回那些行。"""
    r = T.groupby("ts")[score].rank(method="first", ascending=ascending)
    n = T.groupby("ts")[score].transform("size")
    return T[(r <= k) & (n >= 2 * k)]


def _brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def calib_table(p: np.ndarray, y: np.ndarray, bins: int = 10) -> pd.DataFrame:
    q = pd.qcut(pd.Series(p), bins, duplicates="drop")
    df = pd.DataFrame({"p": p, "y": y, "bin": q})
    return df.groupby("bin", observed=True).agg(预测均值=("p", "mean"), 实际频率=("y", "mean"), n=("y", "size")).reset_index(drop=True)


def evaluate_fold(models: Models, te: pd.DataFrame, topk: int, cost: float, thr: float) -> dict:
    P = models.predict(te)
    T = pd.concat([te[["ts", "symbol", "fwd_resid_72h", "lrange", "mae_72h", "adr_14d", "rule_score",
                       "funding", "btc_ret_30d"]], P], axis=1)
    T["up"] = (T["fwd_resid_72h"] > thr).astype(float)
    T["dn"] = (T["fwd_resid_72h"] < -thr).astype(float)
    T["dir"] = T["p_up"] - T["p_dn"]
    T["dir_v"] = T["pv_up"] - T["pv_dn"]
    h0 = int(T["ts"].min() // HOUR)
    CS = T[((T["ts"] // HOUR) - h0) % H == 0]            # 间隔 72h 的截面,互不重叠
    res = {"T": T, "CS": CS, "ic": {}, "top": {}}

    res["ic"]["波动:模型"] = _ic_series(CS, "pred_lrange", "lrange")
    res["ic"]["波动:基线(最近波动)"] = _ic_series(CS, "adr_14d", "lrange")
    res["ic"]["方向:模型(P上-P下)"] = _ic_series(CS, "dir", "fwd_resid_72h")
    res["ic"]["方向:现行规则得分"] = _ic_series(CS, "rule_score", "fwd_resid_72h")
    res["ic"]["方向:只看波动的基线"] = _ic_series(CS, "dir_v", "fwd_resid_72h")

    # 前 k 名:真实表现
    rows = []
    allm = CS.groupby("ts")["fwd_resid_72h"].transform("mean")
    for name, score, asc, tgt in [("P上最高 k 个", "p_up", False, "up"), ("P下最高 k 个", "p_dn", False, "dn"),
                                  ("P上最高 k 个(只看波动基线)", "pv_up", False, "up"),
                                  ("P下最高 k 个(只看波动基线)", "pv_dn", False, "dn")]:
        s = _topk(CS, score, topk, tgt, asc)
        rows.append({"组": name, "n": len(s), "实际频率": s[tgt].mean(), "全体频率": CS[tgt].mean(),
                     "72h超额均值": s["fwd_resid_72h"].mean(), "中位回撤": s["mae_72h"].median()})
    s = _topk(CS, "pred_lrange", topk, "lrange", False)
    rows.append({"组": "预测波动最大 k 个", "n": len(s), "实际频率": np.nan, "全体频率": np.nan,
                 "72h超额均值": s["fwd_resid_72h"].mean(), "中位回撤": s["mae_72h"].median(),
                 "实际波动中位": float(np.exp(s["lrange"].median())), "全体波动中位": float(np.exp(CS["lrange"].median()))})
    res["top"] = pd.DataFrame(rows)

    # 多空与扣成本:做多 P上-P下 最高 k,做空最低 k;资金费率按 8h 费率 × 9 次结算近似
    long_ = _topk(CS, "dir", topk, "fwd_resid_72h", False)
    short_ = _topk(CS, "dir", topk, "fwd_resid_72h", True)
    fund_l = (long_["funding"].fillna(0) * 9).mean()
    fund_s = (short_["funding"].fillna(0) * 9).mean()
    res["pnl"] = {"做多前k 毛": long_["fwd_resid_72h"].mean(),
                  "做多前k 净": long_["fwd_resid_72h"].mean() - cost - fund_l,
                  "做空后k 毛": -short_["fwd_resid_72h"].mean(),
                  "做空后k 净": -short_["fwd_resid_72h"].mean() - cost + fund_s,
                  "全体均值": CS["fwd_resid_72h"].mean(), "n多": len(long_), "n空": len(short_)}

    # 校准与 Brier(用全部检验行,每 3 小时取一个)
    S = T.iloc[::3]
    res["brier"] = {}
    res["calib"] = {}
    for key, pm, pv, y in [("上涨", "p_up", "pv_up", "up"), ("下跌", "p_dn", "pv_dn", "dn")]:
        base = models.base_up if key == "上涨" else models.base_dn
        bm, bv, bc = _brier(S[pm], S[y]), _brier(S[pv], S[y]), _brier(np.full(len(S), base), S[y])
        res["brier"][key] = {"Brier(模型)": bm, "Brier(只看波动)": bv, "Brier(历史平均)": bc,
                             "技能分 vs 历史平均": 1 - bm / bc, "技能分 vs 只看波动": 1 - bm / bv}
        res["calib"][key] = calib_table(S[pm].to_numpy(), S[y].to_numpy())

    # 回撤 q10:覆盖率(实际比预测更差的比例,理想 10%),按预测风险分 5 档
    S2 = S.copy()
    S2["breach"] = (S2["mae_72h"] < S2["pred_mae_q10"]).astype(float)
    S2["bucket"] = pd.qcut(S2["pred_mae_q10"], 5, labels=False, duplicates="drop")
    res["mae_cov"] = S2.groupby("bucket").agg(预测q10均值=("pred_mae_q10", "mean"), 实际q10=("mae_72h", lambda x: x.quantile(0.10)),
                                             越界比例=("breach", "mean"), n=("breach", "size")).reset_index(drop=True)
    res["mae_cov_all"] = float(S2["breach"].mean())

    # 按大盘风格拆分方向 IC
    lab = pd.cut(CS["btc_ret_30d"], [-np.inf, -0.05, 0.05, np.inf], labels=["BTC下行", "BTC横盘", "BTC上行"])
    ic_dir = res["ic"]["方向:模型(P上-P下)"]
    reg = lab.groupby(CS["ts"]).first().reindex(ic_dir.index)
    res["ic_regime"] = {str(k): ic_dir[reg == k] for k in ["BTC下行", "BTC横盘", "BTC上行"]}
    return res


def run_eval(D: pd.DataFrame, folds: int, first_train: float, thr: float, topk: int, cost: float) -> dict:
    lab = D[LABEL_COLS + ["lrange"]].notna().all(axis=1)
    L = D[lab]
    ts_all = D["ts"].to_numpy()
    qs = [first_train + (1 - first_train) * i / folds for i in range(folds)] + [1.0]
    edges = [int(np.quantile(ts_all, q)) for q in qs]
    edges[-1] += 1
    out = []
    for k in range(folds):
        lo, hi = edges[k], edges[k + 1]
        tr = L[(L["ts"] < lo - EMBARGO) & (L["h"] % 6 == 0)]
        te = L[(L["ts"] >= lo) & (L["ts"] < hi)]
        log.info("第 %d/%d 轮:训练 %d 行,检验 %d 行", k + 1, folds, len(tr), len(te))
        m = Models(thr).fit(tr)
        r = evaluate_fold(m, te, topk, cost, thr)
        r["fold"] = k + 1
        r["window"] = (pd.to_datetime(lo, unit="ms").date(), pd.to_datetime(hi, unit="ms").date())
        out.append(r)
    return {"folds": out}


def pool(res: dict) -> dict:
    """把各轮合并成总体指标。"""
    f = res["folds"]
    ic = {k: pd.concat([x["ic"][k] for x in f]) for k in f[0]["ic"]}
    ic_tab = pd.DataFrame({k: _ic_stat(v) for k, v in ic.items()}).T
    CS = pd.concat([x["CS"] for x in f])
    top = pd.concat([x["top"].assign(fold=x["fold"]) for x in f])
    top = top.groupby("组", sort=False).agg({c: "mean" for c in top.columns if c not in ("组", "fold")})
    brier = {key: pd.DataFrame([x["brier"][key] for x in f]).mean() for key in ("上涨", "下跌")}
    pnl = pd.DataFrame([x["pnl"] for x in f]).mean()
    reg = {key: _ic_stat(pd.concat([x["ic_regime"][key] for x in f])) for key in ["BTC下行", "BTC横盘", "BTC上行"]}
    calib = {key: pd.concat([x["calib"][key] for x in f]).groupby(level=0).apply(
        lambda q: pd.Series({"预测均值": np.average(q["预测均值"], weights=q["n"]),
                             "实际频率": np.average(q["实际频率"], weights=q["n"]), "n": q["n"].sum()}))
        for key in ("上涨", "下跌")}
    mae_cov = pd.concat([x["mae_cov"] for x in f]).groupby(level=0).mean()
    return {"ic": ic_tab, "top": top, "brier": brier, "pnl": pnl, "regime": reg, "calib": calib,
            "mae_cov": mae_cov, "mae_cov_all": float(np.mean([x["mae_cov_all"] for x in f]))}


# ------------------------------------------------------------------ 当前排名
def rank_now(D: pd.DataFrame, thr: float, top: int = 10) -> pd.DataFrame:
    """用全部有标签的历史训练,对每个币最新一行打分。"""
    lab = D[LABEL_COLS + ["lrange"]].notna().all(axis=1)
    tr = D[lab & (D["h"] % 6 == 0)]
    m = Models(thr).fit(tr)
    tmax = D["ts"].max()
    last = D[D["ts"] >= tmax - 6 * HOUR].sort_values("ts").groupby("symbol").tail(1)
    P = m.predict(last)
    R = last[["symbol", "ts", "close"]].join(P)
    R["预测波动幅度"] = np.exp(R["pred_lrange"])
    R["可承受杠杆(90%)"] = 1 / R["pred_mae_q10"].abs()
    R["P上(>+{:.0%})".format(thr)] = R["p_up"]
    R["P下(<-{:.0%})".format(thr)] = R["p_dn"]
    R["数据时间"] = pd.to_datetime(R["ts"], unit="ms")
    return R.drop(columns=["pv_up", "pv_dn", "pred_lrange", "ts"]).reset_index(drop=True), m


# ------------------------------------------------------------------ 报告
def _t(df: pd.DataFrame, pct: list[str] = ()) -> str:
    d = df.copy()
    for c in d.columns:
        if c in pct:
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else f"{v * 100:+.2f}%")
        elif d[c].dtype.kind == "f":
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else f"{v:.3f}")
    try:
        return d.to_markdown()
    except ImportError:
        return d.to_string()


def section(tag: str, res: dict, P: dict, args) -> list[str]:
    md = [f"## {tag}\n", "检验窗口:" + "、".join(f"第{x['fold']}轮 {x['window'][0]}~{x['window'][1]}" for x in res["folds"]) + "\n"]
    md += ["### 1. 排序能力(按时间截面的 Spearman IC;t 用截面序列算)\n", _t(P["ic"]), "\n",
           "解读:波动一行要明显高于「基线(最近波动)」才说明模型有增量;方向一行要高于「现行规则得分」和「只看波动的基线」。"
           "IC 在 0.02~0.05 就算有价值,t ≥ 2.5 才可信。\n"]
    md += ["### 2. 前 k 名的真实表现\n", _t(P["top"], ["72h超额均值", "中位回撤", "实际频率", "全体频率"]), "\n",
           "「P上最高 k 个」的实际频率要高于「只看波动基线」的同一行,才说明方向判断有增量(高波动的币本来就两头都容易出大涨大跌)。\n"]
    md += ["### 3. 概率校准\n"]
    for key in ("上涨", "下跌"):
        md += [f"**{key}**:Brier 技能分(越大越好,0=和基线一样):\n", _t(P["brier"][key].to_frame("均值").T), "\n",
               "校准曲线(预测均值应接近实际频率):\n", _t(P["calib"][key]), "\n"]
    md += ["### 4. 回撤 q10 是否可靠(越界比例应约 10%)\n", f"总体越界比例 {P['mae_cov_all']:.1%}\n", _t(P["mae_cov"]), "\n"]
    md += [f"### 5. 扣成本后的组合(每 72h 一期,k={args.topk},成本 {args.cost:.2%}/期 + 资金费率)\n",
           _t(P["pnl"].to_frame("均值").T, ["做多前k 毛", "做多前k 净", "做空后k 毛", "做空后k 净", "全体均值"]), "\n"]
    md += ["### 6. 按大盘风格拆分的方向 IC\n", _t(pd.DataFrame(P["regime"]).T), "\n"]
    return md


def write_report(out: Path, wf: dict, sp: dict, Pw: dict, Ps: dict, latest: pd.DataFrame, args) -> Path:
    out.mkdir(exist_ok=True)
    md = ["# 72 小时机会模型:样本外评估\n",
          f"目标:未来 72h 相对 BTC 超额收益 > +{args.threshold:.0%}(上涨)/ < -{args.threshold:.0%}(下跌),以及波动幅度与回撤 q10。"
          "全部为样本外,训练与检验之间隔 72 小时;截面每 72h 取一次,互不重叠。\n"]
    md += section("滚动检验(前 40% 起步,后面 4 段)", wf, Pw, args)
    md += section("固定 70/30(前 70% 训练,后 30% 检验)", sp, Ps, args)
    p = out / "model_report.md"
    p.write_text("\n".join(md), encoding="utf-8")
    Pw["ic"].to_csv(out / "model_ic_wf.csv", encoding="utf-8-sig")
    Ps["ic"].to_csv(out / "model_ic_split.csv", encoding="utf-8-sig")
    latest.to_csv(out / "model_latest.csv", index=False, encoding="utf-8-sig")
    return p


def main() -> None:
    ap = argparse.ArgumentParser(description="72 小时机会模型:评估 + 当前排名")
    ap.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    ap.add_argument("--threshold", type=float, default=0.05, help="上涨/下跌的超额收益门槛")
    ap.add_argument("--topk", type=int, default=8, help="每个截面取前/后 k 个币")
    ap.add_argument("--cost", type=float, default=0.002, help="每期往返手续费+滑点")
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--symbols")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cfg = load_config(args.config)
    store = Store(cfg["storage"]["db_path"])
    th = merged_thresholds(cfg["signals"].get("thresholds"))
    syms = [s.strip().upper() for s in args.symbols.split(",")] if args.symbols else None
    frames = tune.load_frames(store, th, syms)
    D = build_table(frames, th)
    log.info("样本表 %d 行 × %d 币", len(D), D["symbol"].nunique())

    wf = run_eval(D, args.folds, 0.4, args.threshold, args.topk, args.cost)
    sp = run_eval(D, 1, 0.7, args.threshold, args.topk, args.cost)
    Pw, Ps = pool(wf), pool(sp)
    latest, _ = rank_now(D, args.threshold)
    path = write_report(Path(cfg["_base_dir"]) / "reports", wf, sp, Pw, Ps, latest, args)
    pd.set_option("display.width", 250)
    print("\n== 滚动检验:IC ==\n", Pw["ic"].to_string())
    print("\n== 70/30:IC ==\n", Ps["ic"].to_string())
    print(f"\n完整报告:{path}")


if __name__ == "__main__":
    main()
