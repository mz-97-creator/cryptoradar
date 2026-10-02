"""72 小时机会模型的共用部分:特征表、模型、概率校准、打分。训练评估(model.py)和云端推理(cloud_run.py)都用这一份,
避免"回测一套、实盘一套"。

预测目标(相对 BTC 剔除 beta 后的 72 小时超额收益 resid72):
  P_up / P_dn   resid72 > +T / < -T 的概率(T 默认 5%)
  波动幅度       未来 72h 最高价/最低价之间的幅度
  回撤 q10      持有期最大回撤的 10% 分位;1/|q10| 是"90% 情形不被强平"的杠杆上限

两套特征:
  full   币自身特征 + 持仓量/资金费率/多空比/主动买卖比 + 大盘行情。训练评估用(币安历史数据)。
  price  只用价格与成交额派生的特征。云端用:云端数据来自 OKX,而模型用币安历史训练,
         实测两个交易所的价格类特征几乎一致(相关性 0.95~1.00),但持仓量、资金费率、多空比、主动买卖比差别很大。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

HOUR = 3_600_000
H = 72
EMBARGO = H * HOUR

PRICE_COIN = ["ret_24h", "ret_72h", "ret_7d", "resid_24h_z", "ret_1h_z", "vol_z", "range_24h", "adr_14d",
              "range_z", "rv_24h", "rv_7d", "ethbtc_ret_24h", "resid_rv_7d", "resid_rv_30d"]
DERIV_COIN = ["oi_chg_24h", "oi_z", "funding", "funding_z", "top_ls_z", "taker_z", "beta"]
PRICE_REGIME = ["btc_ret_24h", "btc_ret_7d", "btc_ret_30d", "btc_rv_7d", "breadth", "mkt_ret_24h", "dispersion"]
DERIV_REGIME = ["mkt_funding", "mkt_oi_z"]
FEATURE_SETS = {
    "full": PRICE_COIN + DERIV_COIN + PRICE_REGIME + DERIV_REGIME,
    "price": PRICE_COIN + PRICE_REGIME,
}
VOL_FEATS = ["range_24h", "adr_14d", "range_z", "rv_24h", "rv_7d"]
LABEL_COLS = ["fwd_resid_72h", "fwd_ret_72h", "mae_72h", "mfe_72h"]
TRAIN_COLS = LABEL_COLS + ["lrange", "lrv72"]      # 训练/评估时这些列都必须有值
_RAW_COLS = ["ret_24h", "resid_24h_z", "ret_1h_z", "oi_chg_24h", "oi_z", "funding", "funding_z", "vol_z",
             "top_ls_z", "taker_z", "range_24h", "adr_14d", "range_z", "beta", "btc_ret_24h", "ethbtc_ret_24h"]


# ------------------------------------------------------------------ 特征表
def build_table(frames: list[pd.DataFrame], th: dict | None = None, with_labels: bool = True) -> pd.DataFrame:
    """各币的特征表(build_features 的输出,带 symbol 列)拼成长表,并加上大盘行情特征。只用当时已知的信息。
    th 给定时额外算现行规则得分(research/评估对照用)。with_labels=False 用于云端推理(没有未来标签)。"""
    hits = w = None
    if th is not None:
        from .signals import RULES, evaluate_frame
        w = np.array([r.weight for r in RULES])
        ids = [r.id for r in RULES]
        hits = [evaluate_frame(f, th)[ids].to_numpy(float) for f in frames]
    parts = []
    for i, f in enumerate(frames):
        lc = np.log(f["close"])
        g = pd.DataFrame(index=f.index)
        g["symbol"] = f["symbol"].iloc[0]
        for c in _RAW_COLS:
            g[c] = f[c] if c in f else np.nan
        g["ret_72h"] = lc.diff(72)
        g["ret_7d"] = lc.diff(168)
        r1 = lc.diff()
        g["rv_24h"] = r1.rolling(24, min_periods=20).std()
        g["rv_7d"] = r1.rolling(168, min_periods=120).std()
        # 剔除 BTC 之后这个币自己的波动:跟 BTC 高度同步的币(ETH 等)相对 BTC 的超额波动很小,
        # 不加这个特征,模型会按总波动把它们的大涨大跌概率高估好几倍
        res1 = f["ret_1h"] - f["beta"] * f["btc_ret_1h"]
        if with_labels:   # 未来 72h 已实现的"剔除 BTC 后波动":72 个小时收益平方和的平方根(比单个 |收益| 噪音小得多)
            g["lrv72"] = np.log(np.sqrt((res1 ** 2).rolling(72).sum().shift(-72)).clip(lower=1e-4))
        g["resid_rv_7d"] = res1.rolling(168, min_periods=120).std()
        g["resid_rv_30d"] = res1.rolling(720, min_periods=480).std()
        b = f["_btc_lc"]
        g["btc_ret_7d"] = b.diff(168)
        g["btc_ret_30d"] = b.diff(720)
        g["btc_rv_7d"] = b.diff().rolling(168, min_periods=120).std()
        g["above_sma7d"] = (f["close"] > f["close"].rolling(168, min_periods=120).mean()).astype(float)
        if hits is not None:
            g["rule_score"] = hits[i] @ w
        if with_labels:
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
    D["h"] = (D["ts"] // HOUR).astype("int64")
    num = FEATURE_SETS["full"] + ["above_sma7d"] + (["rule_score"] if hits is not None else [])
    if with_labels:
        D["lrange"] = np.log((D["mfe_72h"] - D["mae_72h"]).clip(lower=1e-4))
        num += TRAIN_COLS
    D[num] = D[num].astype("float32")
    return D


# ------------------------------------------------------------------ 模型
def _hgb_kwargs() -> dict:
    return dict(max_depth=4, learning_rate=0.05, max_iter=200, l2_regularization=5.0,
                min_samples_leaf=500, random_state=0)


def _oof_proba(X: pd.DataFrame, y: np.ndarray, ts: pd.Series, nb: int = 5) -> np.ndarray:
    """时间块交叉拟合:把训练期按时间切成 nb 块,每块用其余块(两侧各留 72h)训练、预测本块,
    得到整个训练期的"样本外"概率,用来衡量模型平均偏自信多少。"""
    from sklearn.ensemble import HistGradientBoostingClassifier as C
    q = np.quantile(ts, np.linspace(0, 1, nb + 1))
    q[-1] += 1
    out = np.full(len(X), np.nan)
    for i in range(nb):
        blk = ((ts >= q[i]) & (ts < q[i + 1])).to_numpy()
        keep = ~((ts > q[i] - EMBARGO) & (ts < q[i + 1] + EMBARGO)).to_numpy()
        out[blk] = C(**_hgb_kwargs()).fit(X[keep], y[keep]).predict_proba(X[blk])[:, 1]
    return out


def _oof_reg(X: pd.DataFrame, y: np.ndarray, ts: pd.Series, nb: int = 5, forward: bool = True) -> np.ndarray:
    """训练期内"样本外"预测,用来估计模型的预测误差分布。
    forward=True(默认):只用过去预测未来——按时间切成 nb+1 块,第 k 块只用前面的块(中间隔 72h)训练,第一块没有过去可用、留空。
                      这和真实部署的顺序一致。
    forward=False:    每块用"其余所有块(含它之后的)"训练,历史上用过,留作对照;它用了未来信息,误差分布会比真实部署时偏乐观。"""
    from sklearn.ensemble import HistGradientBoostingRegressor as R
    out = np.full(len(X), np.nan)
    if forward:
        q = np.quantile(ts, np.linspace(0, 1, nb + 2))
        q[-1] += 1
        for i in range(1, nb + 1):
            blk = ((ts >= q[i]) & (ts < q[i + 1])).to_numpy()
            past = (ts < q[i] - EMBARGO).to_numpy()
            if past.sum() < 5000 or not blk.any():
                continue
            out[blk] = R(**_hgb_kwargs()).fit(X[past], y[past]).predict(X[blk])
        return out
    q = np.quantile(ts, np.linspace(0, 1, nb + 1))
    q[-1] += 1
    for i in range(nb):
        blk = ((ts >= q[i]) & (ts < q[i + 1])).to_numpy()
        keep = ~((ts > q[i] - EMBARGO) & (ts < q[i + 1] + EMBARGO)).to_numpy()
        out[blk] = R(**_hgb_kwargs()).fit(X[keep], y[keep]).predict(X[blk])
    return out


class Platt:
    """对 logit(p) 做一维逻辑回归:修正模型整体偏自信/偏高偏低。(用类而不是闭包,方便 joblib 保存)"""

    def __init__(self, p_raw: np.ndarray | None = None, y: np.ndarray | None = None):
        self.lr = None
        if p_raw is not None:
            from sklearn.linear_model import LogisticRegression
            self.lr = LogisticRegression(C=1e3).fit(self._z(p_raw), y)

    @staticmethod
    def _z(p: np.ndarray) -> np.ndarray:
        p = np.clip(p, 1e-4, 1 - 1e-4)
        return np.log(p / (1 - p)).reshape(-1, 1)

    def __call__(self, p: np.ndarray) -> np.ndarray:
        return p if self.lr is None else self.lr.predict_proba(self._z(p))[:, 1]


class Models:
    """四个预测目标 + 一个"只看波动"的基线。

    上涨/下跌概率有两种做法(prob 参数):
      scale(默认) 先预测未来 72h 剔除 BTC 后的波动尺度 σ,再用"收益/σ"的历史样本外分布推出 P(收益>+T)、P(收益<-T)。
                  波动小的币(ETH 等)概率自然小,不依赖分类树去外推,对不同币、不同波动水平更稳。
      tree        直接用分类树预测 + 交叉拟合 Platt 校准(旧做法,留作对照)。
    概率的局限(样本外实测):全市场"当期有多少币大跌"随行情漂移、很难预测,所以绝对概率有约 ±4pp 的不确定性,
    币与币之间的相对高低比绝对数值可靠。"""

    def __init__(self, thr: float = 0.05, calib: str = "cv", feature_set: str = "full", prob: str = "scale",
                 forward: bool = True):
        self.thr, self.calib, self.feature_set, self.prob, self.forward = thr, calib, feature_set, prob, forward
        self.feats = FEATURE_SETS[feature_set]

    def fit(self, tr: pd.DataFrame) -> "Models":
        from sklearn.ensemble import HistGradientBoostingClassifier as C, HistGradientBoostingRegressor as R
        X = tr[self.feats]
        up, dn = (tr["fwd_resid_72h"] > self.thr), (tr["fwd_resid_72h"] < -self.thr)
        self.base_up, self.base_dn = float(up.mean()), float(dn.mean())
        self.up = C(**_hgb_kwargs()).fit(X, up)      # 分类树:scale 模式下只用来算"方向倾向",不用于概率
        self.dn = C(**_hgb_kwargs()).fit(X, dn)
        if self.prob == "scale":
            y = tr["lrv72"].to_numpy()
            self.scale = R(**_hgb_kwargs()).fit(X, y)
            ret = tr["fwd_resid_72h"].to_numpy()
            z = ret / np.exp(_oof_reg(X, y, tr["ts"], forward=self.forward))   # 样本外标准化收益
            if np.isfinite(z).sum() < 2000:     # 训练数据太少,"只用过去"凑不出足够的样本外点:退回分块方式(并非部署顺序,仅作兜底)
                z = ret / np.exp(_oof_reg(X, y, tr["ts"], forward=False))
            self.z_sorted = np.sort(z[np.isfinite(z)])
            if len(self.z_sorted) == 0:
                raise ValueError("训练样本不足,无法估计标准化收益分布")
        else:
            if self.calib == "cv":
                self.cal_up = Platt(_oof_proba(X, up.to_numpy(float), tr["ts"]), up.to_numpy(float))
                self.cal_dn = Platt(_oof_proba(X, dn.to_numpy(float), tr["ts"]), dn.to_numpy(float))
            else:
                self.cal_up = self.cal_dn = Platt()
        self.rng = R(**_hgb_kwargs()).fit(X, tr["lrange"])
        self.mae = R(loss="quantile", quantile=0.10, **_hgb_kwargs()).fit(X, tr["mae_72h"])
        Xv = tr[VOL_FEATS]
        self.up_v = C(**_hgb_kwargs()).fit(Xv, up)
        self.dn_v = C(**_hgb_kwargs()).fit(Xv, dn)
        return self

    def predict(self, d: pd.DataFrame) -> pd.DataFrame:
        X, Xv = d[self.feats], d[VOL_FEATS]
        out = pd.DataFrame(index=d.index)
        t_up, t_dn = self.up.predict_proba(X)[:, 1], self.dn.predict_proba(X)[:, 1]
        out["tilt"] = t_up - t_dn            # 方向倾向:同一波动水平下偏涨还是偏跌(样本外证据弱,见 direction_evidence)
        if self.prob == "scale":
            sig = np.exp(self.scale.predict(X))
            n = len(self.z_sorted)
            out["p_up"] = 1 - np.searchsorted(self.z_sorted, self.thr / sig, side="right") / n
            out["p_dn"] = np.searchsorted(self.z_sorted, -self.thr / sig, side="left") / n
            out["p_up_raw"], out["p_dn_raw"] = out["p_up"], out["p_dn"]
            out["sigma72"] = sig
        else:
            out["p_up_raw"], out["p_dn_raw"] = t_up, t_dn
            out["p_up"], out["p_dn"] = self.cal_up(t_up), self.cal_dn(t_dn)
        out["pred_lrange"] = self.rng.predict(X)
        out["pred_mae_q10"] = np.minimum(self.mae.predict(X), -1e-3)
        out["pv_up"] = self.up_v.predict_proba(Xv)[:, 1]
        out["pv_dn"] = self.dn_v.predict_proba(Xv)[:, 1]
        return out


# ------------------------------------------------------------------ 打分(训练评估与云端共用)
def latest_rows(D: pd.DataFrame) -> pd.DataFrame:
    """每个币取最新一行(最近 6 小时内有数据的币)。"""
    tmax = D["ts"].max()
    return D[D["ts"] >= tmax - 6 * HOUR].sort_values("ts").groupby("symbol").tail(1)


def score_rows(models: Models, last: pd.DataFrame, topk: int = 8, evidence: dict | None = None) -> pd.DataFrame:
    """对每个币最新一行打分:概率、波动幅度、回撤与杠杆上限、方向档位(按截面排名)与该档位的历史实绩。"""
    P = models.predict(last)
    R = last[["symbol", "ts", "close"]].join(P)
    topk = max(1, min(topk, len(R) // 10))          # 币少时(自检/新上线)前后各取 10%
    R["预测波动幅度"] = np.exp(R["pred_lrange"])
    R["vol_range"], R["mae_q10"] = R["预测波动幅度"], R["pred_mae_q10"]
    R["可承受杠杆(90%)"] = 1 / R["pred_mae_q10"].abs()
    R["方向分(P上-P下)"] = R["tilt"]
    rk = R["tilt"].rank(method="first", ascending=False)
    tier = np.where(rk <= topk, "偏涨", np.where(rk > len(R) - topk, "偏跌", "中性"))
    ev = evidence or {}
    # 证据弱(样本外 |t|<2 或方向相反)时不下结论,只保留档位供研究
    R["方向档位(研究)"] = tier
    R["方向"] = [t if t == "中性" or not str(ev.get(t, {}).get("强度", "弱")).startswith("弱") else "无明确方向" for t in tier]
    ev = {k: v for k, v in ev.items()}
    R["该档位历史:涨>阈值比例"] = R["方向档位(研究)"].map(lambda d: ev.get(d, {}).get("涨"))
    R["该档位历史:跌<-阈值比例"] = R["方向档位(研究)"].map(lambda d: ev.get(d, {}).get("跌"))
    R["该档位历史:72h超额均值"] = R["方向档位(研究)"].map(lambda d: ev.get(d, {}).get("超额"))
    R["证据强度"] = R["方向档位(研究)"].map(lambda d: ev.get(d, {}).get("强度", ""))
    R[f"P上(>+{models.thr:.0%})"] = R["p_up"]
    R[f"P下(<-{models.thr:.0%})"] = R["p_dn"]
    R["数据时间"] = pd.to_datetime(R["ts"], unit="ms")
    # 方向分与截面排名(1 = 最偏涨)留作实盘方向核对
    R["dir_score"], R["dir_rank"], R["n_ranked"] = R["tilt"], rk.astype(int), len(R)
    R = R.rename(columns={"ts": "t_bar"})
    return R.drop(columns=["pv_up", "pv_dn", "pred_lrange", "p_up_raw", "p_dn_raw", "pred_mae_q10", "sigma72", "tilt"],
                  errors="ignore").reset_index(drop=True)


def save_bundle(path, models: Models, meta: dict) -> None:
    import joblib
    joblib.dump({"models": models, "meta": meta}, path, compress=3)


def load_bundle(path) -> dict:
    import joblib
    return joblib.load(path)


# ------------------------------------------------------------------ 云端:打分、实盘记录、事后结算
DAY_MS = 86_400_000
MIN_HISTORY_H = 700          # 滚动 30 天的 z 分数需要约 720 根;不足时特征不可靠


def assess_coin(list_ms: int | None, history_h: int, now_ms: int, min_age_days: float, young_age_days: float) -> dict:
    """币的"可评估性":新上市、历史不足的币,模型训练时见得少、特征(30 天滚动 z 分数)也不稳,暂不判断。
    status: 新币(暂不判断) / 上市较短(给结果但提示) / 正常 / 未知(拿不到上市时间,按历史长度判断)。"""
    age = None if not list_ms else (now_ms - list_ms) / DAY_MS
    if history_h < MIN_HISTORY_H:
        return {"age_days": age, "status": "新币", "usable": False,
                "reason": f"可用历史只有 {history_h} 小时,不足 {MIN_HISTORY_H} 小时,特征不稳"}
    if age is not None and age < min_age_days:
        return {"age_days": age, "status": "新币", "usable": False,
                "reason": f"合约上市仅 {age:.0f} 天(不足 {min_age_days:.0f} 天),模型对新币见得少,暂不判断"}
    if age is not None and age < young_age_days:
        return {"age_days": age, "status": "上市较短", "usable": True,
                "reason": f"合约上市 {age:.0f} 天(不足 {young_age_days:.0f} 天),结果仅供参考,回撤与杠杆上限已更保守"}
    return {"age_days": age, "status": "正常" if age is not None else "未知", "usable": True, "reason": ""}


def cloud_opportunity(bundle: dict, frames: dict[str, pd.DataFrame], uni: list[dict], topk: int = 8,
                      notable_rank: int = 15, now_ms: int | None = None, min_age_days: float = 30,
                      young_age_days: float = 60, young_dd_mult: float = 1.3,
                      rules: dict[str, list[str]] | None = None):
    """frames: {币: build_features 的输出}。返回 (写进 signals.json 的字典, 可评估币的打分表)。
    新币 / 历史不足的币不给数字(置为 None),不进榜单和排名,单独列在 abstained 里;
    先剔除它们再打分,方向档位和排名只在可评估的币之间比较。
    上市较短(min_age_days~young_age_days)的币回撤估计历史上偏乐观(样本外越界约 17% 而不是 10%),
    回撤放大 young_dd_mult 倍、杠杆上限相应缩小。
    rules: {币: 本轮触发的规则 id},一并写进打分表,留作实盘方向核对。"""
    import time
    now_ms = now_ms or int(time.time() * 1000)
    models, meta = bundle["models"], bundle["meta"]
    fl = [f.assign(symbol=sym) for sym, f in frames.items() if len(f) >= 200]
    D = build_table(fl, None, with_labels=False)
    last = latest_rows(D)
    info = {u["ccy"]: u for u in uni}
    hist = {sym: len(f) for sym, f in frames.items()}
    num = lambda v: None if v is None or pd.isna(v) else float(v)
    head = lambda sym, a, close, t_bar: {
        "symbol": sym, "rank": info.get(sym, {}).get("rank"), "watch": bool(info.get(sym, {}).get("watch")),
        "price": num(close), "age_days": None if a["age_days"] is None else round(a["age_days"]),
        "status": a["status"], "note": a["reason"], "t_bar": int(t_bar)}
    assess = {sym: assess_coin(info.get(sym, {}).get("list_ms"), hist.get(sym, 0), now_ms, min_age_days, young_age_days)
              for sym in last["symbol"]}
    ok = last["symbol"].map(lambda sym: assess[sym]["usable"]).astype(bool)
    abstained = [{**head(r["symbol"], assess[r["symbol"]], r["close"], r["ts"]),
                  "vol_range": None, "mae_q10": None, "safe_lev": None, "p_up": None, "p_dn": None,
                  "direction": "暂不判断", "direction_tier_research": None, "direction_evidence": "—"}
                 for r in last[~ok].to_dict("records")]
    if ok.any():
        R = score_rows(models, last[ok], topk, meta.get("direction_evidence"))
    else:
        R = pd.DataFrame(columns=LOG_COLS)
    R["dir_tier"] = R["方向档位(研究)"] if len(R) else []
    R["rules"] = [";".join((rules or {}).get(sym) or []) for sym in R["symbol"]]
    coins = []
    for r in R.to_dict("records"):
        a = assess[r["symbol"]]
        mult = young_dd_mult if a["status"] == "上市较短" else 1.0
        coins.append({**head(r["symbol"], a, r["close"], r["t_bar"]),
                      "vol_range": num(r["vol_range"]), "mae_q10": num(r["mae_q10"] * mult),
                      "safe_lev": num(r["可承受杠杆(90%)"] / mult), "p_up": num(r["p_up"]), "p_dn": num(r["p_dn"]),
                      "direction": r["方向"], "direction_tier_research": r["方向档位(研究)"],
                      "direction_evidence": r["证据强度"] or "—",
                      "dir_score": num(r["dir_score"]), "dir_rank": int(r["dir_rank"])})
    top = lambda key: [c["symbol"] for c in sorted(coins, key=lambda c: -(c[key] or 0))[:topk]]
    # 每个币在可评估币里的排名(1 = 最高),自选币有任何一项进前 notable_rank 名就单独标出来
    names = {"vol_range": "波动", "p_up": "上涨概率", "p_dn": "下跌概率"}
    for key in names:
        for i, c in enumerate(sorted(coins, key=lambda c: -(c[key] or 0)), 1):
            c["rank_" + key] = i
    cut = min(notable_rank, max(1, len(coins) // 5))
    highlights = []
    for c in coins:
        if c["watch"]:
            why = [f"{n}第{c['rank_' + k]}位" for k, n in names.items() if c["rank_" + k] <= cut]
            if why:
                highlights.append({"symbol": c["symbol"], "reasons": why})
    block = {
        "model": {"feature_set": meta.get("feature_set"), "trained_through": meta.get("trained_through"),
                  "horizon_h": H, "threshold": models.thr,
                  "oos_vol_ic": meta.get("oos_vol_ic"), "oos_vol_ic_baseline": meta.get("oos_vol_ic_baseline"),
                  "oos_dir_ic": meta.get("oos_dir_ic"), "oos_dir_ic_t": meta.get("oos_dir_ic_t"),
                  "calib_ece": meta.get("calib_ece"), "mae_breach_rate": meta.get("mae_breach_rate"),
                  "min_age_days": min_age_days, "young_age_days": young_age_days, "young_dd_mult": young_dd_mult},
        "notes": [
            f"vol_range = 预测未来 {H}h 最高价与最低价之间的幅度(占价格比例);样本外它对币种波动大小的排序相关性约 "
            f"{meta.get('oos_vol_ic', 0):.2f}(只看最近波动为 {meta.get('oos_vol_ic_baseline', 0):.2f})",
            f"p_up / p_dn = 相对 BTC 的 {H}h 超额收益大于 +{models.thr:.0%} / 小于 -{models.thr:.0%} 的概率。"
            "由预测的波动大小推出,所以波动越大两头概率都越高,榜单顺序与波动榜相同;绝对数值有约 ±4~6 个百分点的误差",
            "mae_q10 = 持有期最大回撤的 10% 分位(90% 的情形回撤不会比它更深),safe_lev = 1/|mae_q10|,"
            "未计手续费和维持保证金,实际应更保守",
            "direction:目前没有统计上站得住的方向判断(样本外 t 值不足),一律为「无明确方向」;研究中的档位见 direction_tier_research",
            f"status=新币 的币(合约上市不足 {min_age_days:.0f} 天,或可用历史不足)暂不判断,所有数字为空,单独列在 abstained;"
            f"status=上市较短(不足 {young_age_days:.0f} 天)给结果但仅供参考,回撤放大 {young_dd_mult:g} 倍、杠杆上限相应缩小(历史上这类币回撤估计偏乐观)",
        ],
        "coins": coins,
        "abstained": abstained,
        "top_vol": top("vol_range"),
        "watchlist": [c["symbol"] for c in coins if c["watch"]],
        "watch_highlights": highlights, "n_coins": len(coins), "n_abstained": len(abstained),
    }
    return block, R


LOG_COLS = ["t_bar", "symbol", "p_up", "p_dn", "vol_range", "mae_q10",
            "dir_score", "dir_rank", "n_ranked", "dir_tier", "rules"]
HORIZONS = (72, 168, 336)       # 实盘方向核对的持有期:72h / 1 周 / 2 周


def log_snapshot(prev: pd.DataFrame | None, R: pd.DataFrame, every_h: int = 6, keep_days: int = 120) -> pd.DataFrame:
    """每 every_h 小时记一次全部币的预测(同一时刻只保留第一次),留作事后核对。
    旧日志没有方向列时补空值,核对方向时自动跳过这些行。"""
    log = prev.reindex(columns=LOG_COLS) if prev is not None and len(prev) else pd.DataFrame(columns=LOG_COLS)
    if R is None or R.empty:
        return log.reset_index(drop=True)
    t = int(R["t_bar"].max())
    if (t // HOUR) % every_h == 0:
        log = pd.concat([log, R.reindex(columns=LOG_COLS)]).drop_duplicates(["t_bar", "symbol"], keep="first")
    return log[log["t_bar"] >= t - keep_days * 24 * HOUR].reset_index(drop=True)


def resolve_log(log: pd.DataFrame, combined: dict[str, pd.DataFrame], horizons=(H,)) -> pd.DataFrame:
    """用样本库里的真实价格核对记录,每个持有期 h 给出:resid{h} 剔除 BTC beta 后的超额(对数收益)、
    mae{h} 相对入场价的最大不利变动、range{h} 期间最高与最低价的幅度。
    只返回最短持有期已到期的行;更长的持有期没到期时为空。终点必须是已收盘的 K 线,
    期间 K 线缺失超过 10% 的不结算。"""
    need = {"close", "_high", "_low", "_btc_lc", "beta"}
    parts = []
    for sym, g in log.groupby("symbol"):
        f = combined.get(sym)
        if f is None or f.empty or not need <= set(f.columns):
            continue
        f = f[~f.index.duplicated(keep="last")]
        last_bar = int(f.index.max())             # 最后一根可能还没收盘,不当作终点
        f = f.reindex(np.arange(int(f.index.min()), last_bar + HOUR, HOUR))
        g = g.copy()
        t0 = g["t_bar"].astype("int64").to_numpy()
        at = lambda col, t: f[col].reindex(t).to_numpy(dtype=float)
        c0, b0, beta = at("close", t0), at("_btc_lc", t0), at("beta", t0)
        for h in horizons:
            t1 = t0 + h * HOUR
            c1, b1 = at("close", t1), at("_btc_lc", t1)
            mp = int(np.ceil(0.9 * h))
            lo = f["_low"][::-1].rolling(h, min_periods=mp).min()[::-1].shift(-1)     # t 之后 h 小时内的最低
            hi = f["_high"][::-1].rolling(h, min_periods=mp).max()[::-1].shift(-1)
            mae, mfe = lo.reindex(t0).to_numpy() / c0 - 1, hi.reindex(t0).to_numpy() / c0 - 1
            with np.errstate(divide="ignore", invalid="ignore"):
                resid = np.log(c1 / c0) - beta * (b1 - b0)
            done = (t1 < last_bar) & np.isfinite(resid) & np.isfinite(mae)
            g[f"resid{h}"] = np.where(done, resid, np.nan)
            g[f"mae{h}"] = np.where(done, mae, np.nan)
            g[f"range{h}"] = np.where(done, mfe - mae, np.nan)
        parts.append(g[g[f"resid{min(horizons)}"].notna()])
    if not parts:
        return pd.DataFrame(columns=list(log.columns) + [f"{k}{h}" for h in horizons for k in ("resid", "mae", "range")])
    return pd.concat(parts, ignore_index=True)


def live_summary(resolved: pd.DataFrame, thr: float, min_n: int = 200) -> dict:
    """实盘核对:预测概率 vs 实际发生率、波动排序相关性、回撤分位数越界率。样本不足时只报条数。"""
    n = len(resolved)
    out = {"resolved": int(n), "since": int(resolved["t_bar"].min()) if n else None}
    if n < min_n:
        out["status"] = f"样本积累中(已核对 {n} 条,满 {min_n} 条后给出统计)"
        return out
    up, dn = resolved["resid72"] > thr, resolved["resid72"] < -thr
    ics = []
    for _, q in resolved.groupby("t_bar"):
        if len(q) >= 10:
            ics.append(q["vol_range"].rank().corr(q["range72"].rank()))
    out.update({"status": "ok",
                "p_up": {"预测均值": float(resolved["p_up"].mean()), "实际频率": float(up.mean())},
                "p_dn": {"预测均值": float(resolved["p_dn"].mean()), "实际频率": float(dn.mean())},
                "vol_ic": float(np.nanmean(ics)) if ics else None,
                "mae_breach_rate": float((resolved["mae72"] < resolved["mae_q10"]).mean())})
    return out


# ------------------------------------------------------------------ 实盘方向核对(72h / 1 周 / 2 周)
OUTCOME_COLS = LOG_COLS + [f"{k}{h}" for h in HORIZONS for k in ("resid", "mae", "range")]


def merge_outcomes(prev: pd.DataFrame | None, resolved: pd.DataFrame) -> pd.DataFrame:
    """永久累积的实盘结果表(opp_outcomes.csv.gz):每条留档预测一行,到期一个持有期补一个持有期。
    日志只保留 120 天,这张表不删,攒下来可以直接拿去检验、改进方向模型。"""
    new = resolved.reindex(columns=OUTCOME_COLS)
    if prev is None or prev.empty:
        out = new
    else:
        old = prev.reindex(columns=OUTCOME_COLS).set_index(["t_bar", "symbol"])
        new = new.set_index(["t_bar", "symbol"])
        old = old[~old.index.duplicated(keep="last")]
        new = new[~new.index.duplicated(keep="last")]
        out = new.combine_first(old).reset_index()      # 新核对的值优先,旧表里已有、日志里已删掉的行保留
    return out.reindex(columns=OUTCOME_COLS).sort_values(["t_bar", "symbol"]).reset_index(drop=True)


def nw_t(x, lags: int) -> float:
    """均值的 t 值,标准误用 Newey-West(Bartlett 核)校正自相关:每 6 小时一个截面、持有期 1~2 周,
    相邻截面的收益大段重叠,不校正 t 会被高估好几倍。"""
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


def direction_cross_sections(outcomes: pd.DataFrame, h: int, min_coins: int = 10) -> pd.DataFrame:
    """持有期 h 下每个截面(同一 t_bar)的方向成绩:方向分与超额的排序相关性(IC)、偏涨组减偏跌组的超额差、
    偏涨组跑赢 BTC 的比例 / 偏跌组跑输的比例,及同期全体的对应比例(基准)。"""
    col = f"resid{h}"
    cols = ["t_bar", "n", "ic", "spread", "top", "bottom", "top_hit", "bottom_hit", "base_up", "base_dn"]
    if outcomes is None or outcomes.empty or col not in outcomes:
        return pd.DataFrame(columns=cols)
    d = outcomes[outcomes[col].notna() & pd.to_numeric(outcomes["dir_score"], errors="coerce").notna()]
    rows = []
    for t, q in d.groupby("t_bar"):
        if len(q) < min_coins:
            continue
        x, s = q[col].astype(float), q["dir_score"].astype(float)
        up, dn = x[q["dir_tier"] == "偏涨"], x[q["dir_tier"] == "偏跌"]
        rows.append({"t_bar": int(t), "n": len(q), "ic": s.rank().corr(x.rank()),
                     "spread": up.mean() - dn.mean() if len(up) and len(dn) else np.nan,
                     "top": up.mean() if len(up) else np.nan, "bottom": dn.mean() if len(dn) else np.nan,
                     "top_hit": (up > 0).mean() if len(up) else np.nan,
                     "bottom_hit": (dn < 0).mean() if len(dn) else np.nan,
                     "base_up": (x > 0).mean(), "base_dn": (x < 0).mean()})
    return pd.DataFrame(rows, columns=cols)


def direction_summary(outcomes: pd.DataFrame | None, horizons=HORIZONS, step_h: int = 6,
                      min_cross: int = 20) -> dict:
    """按持有期汇总实盘方向成绩。t 值按持有期做 Newey-West 校正;
    indep ≈ 覆盖时长 / 持有期,是真正互不重叠的区间数,它小于 10 时结论很不稳。"""
    out = {}
    for h in horizons:
        cs = direction_cross_sections(outcomes, h)
        n = len(cs)
        st = {"h": h, "n_cross": n}
        if n:
            span_h = (cs["t_bar"].max() - cs["t_bar"].min()) / HOUR + h
            st.update({"since": int(cs["t_bar"].min()), "n_obs": int(cs["n"].sum()), "indep": round(span_h / h, 1)})
        if n < min_cross:
            st["status"] = f"样本积累中(已到期 {n} 个截面,满 {min_cross} 个后给出统计)"
            out[f"{h}h"] = st
            continue
        lags = int(np.ceil(h / step_h))
        num = lambda v: None if v is None or not np.isfinite(v) else float(v)
        st.update({"status": "ok",
                   "ic_mean": num(cs["ic"].mean()), "ic_t": num(nw_t(cs["ic"], lags)),
                   "spread_mean": num(cs["spread"].mean()), "spread_t": num(nw_t(cs["spread"], lags)),
                   "top_mean": num(cs["top"].mean()), "bottom_mean": num(cs["bottom"].mean()),
                   "top_hit": num(cs["top_hit"].mean()), "bottom_hit": num(cs["bottom_hit"].mean()),
                   "base_up": num(cs["base_up"].mean()), "base_dn": num(cs["base_dn"].mean())})
        out[f"{h}h"] = st
    return out


def direction_daily(outcomes: pd.DataFrame | None, horizons=HORIZONS) -> pd.DataFrame:
    """按天(UTC,按预测时点)汇总的方向成绩,写进 opp_direction_daily.csv,方便看随时间的变化。"""
    parts = []
    for h in horizons:
        cs = direction_cross_sections(outcomes, h)
        if cs.empty:
            continue
        cs["date"] = pd.to_datetime(cs["t_bar"], unit="ms").dt.strftime("%Y-%m-%d")
        g = cs.groupby("date").agg(n_cross=("t_bar", "size"), n_obs=("n", "sum"), ic=("ic", "mean"),
                                   spread=("spread", "mean"), top_hit=("top_hit", "mean"),
                                   bottom_hit=("bottom_hit", "mean"), base_up=("base_up", "mean"),
                                   base_dn=("base_dn", "mean")).reset_index()
        g.insert(1, "horizon_h", h)
        parts.append(g)
    cols = ["date", "horizon_h", "n_cross", "n_obs", "ic", "spread", "top_hit", "bottom_hit", "base_up", "base_dn"]
    return pd.concat(parts, ignore_index=True)[cols] if parts else pd.DataFrame(columns=cols)
