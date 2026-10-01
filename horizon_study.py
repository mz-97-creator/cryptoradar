"""方向研究:更长的持有周期(72h / 1 周 / 2 周 / 4 周)下,现有特征 + 经典跨币因子能不能把"相对强弱"排出来。

做法(都是样本外):
  - 目标:未来 h 小时相对 BTC 的超额收益,在同一时间点所有币里的排名(0~1)
  - 特征:model.py 的全部特征 + 它们的"同一时间点内排名" + 经典因子(多周期动量、30 天波动、规模/流动性代理、距 90 天高点回撤)
  - 梯度提升回归,滚动检验(前 40% 起步,后面 4 段),训练与检验之间隔 h 小时
  - 评分:按时间截面的 Spearman IC(截面间隔 h 小时,互不重叠),前 k / 后 k 名相对全体的超额及 t 值
    (前后 k 名才是你真正会去选的币,IC 好看但两端不赚钱就没有用)

用法:
  python horizon_study.py                         72h / 168h / 336h / 672h
  python horizon_study.py --horizons 168,336 --topk 8
结果保存在 reports/horizon_study.csv
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

import model
import tune
from cryptoradar.config import load_config
from cryptoradar.features import add_labels
from cryptoradar.opportunity import FEATURE_SETS, HOUR, build_table
from cryptoradar.signals import merged_thresholds
from cryptoradar.storage import Store

log = logging.getLogger("horizon")
CS_BASE = ["ret_24h", "ret_72h", "ret_7d", "resid_24h_z", "oi_chg_24h", "oi_z", "funding", "funding_z", "vol_z",
           "top_ls_z", "taker_z", "range_24h", "adr_14d", "rv_7d"]
CLASSIC = ["ret_14d", "ret_30d", "ret_60d", "rv_30d", "log_vol24h", "dd_90d"]


def with_classic(frames: list[pd.DataFrame]) -> list[pd.DataFrame]:
    out = []
    for f in frames:
        g = f.copy()
        lc = np.log(g["close"])
        g["ret_14d"], g["ret_30d"], g["ret_60d"] = lc.diff(336), lc.diff(720), lc.diff(1440)
        g["rv_30d"] = lc.diff().rolling(720, min_periods=480).std()
        g["log_vol24h"] = np.log(g["vol_24h"].where(g["vol_24h"] > 0))
        g["dd_90d"] = g["close"] / g["close"].rolling(2160, min_periods=720).max() - 1
        out.append(g)
    return out


def table_for(frames: list[pd.DataFrame], h: int, th: dict) -> pd.DataFrame:
    """把 h 小时的标签改名成 72h 的列名,复用 build_table;再补上经典因子和同一时间点内排名。"""
    fl = []
    for f in frames:
        base = f.drop(columns=[c for c in f.columns if c.startswith(("fwd_", "mae_", "mfe_"))])
        g = add_labels(base, horizons=(h,))
        g = g.rename(columns={f"fwd_resid_{h}h": "fwd_resid_72h", f"fwd_ret_{h}h": "fwd_ret_72h",
                              f"mae_{h}h": "mae_72h", f"mfe_{h}h": "mfe_72h"})
        fl.append(g)
    D = build_table(fl, None, with_labels=True)
    extra = pd.concat([f[CLASSIC].assign(symbol=f["symbol"].iloc[0]) for f in frames])
    extra.index.name = "ts"
    D = D.merge(extra.reset_index(), on=["ts", "symbol"], how="left")
    gb = D.groupby("ts")
    for c in CS_BASE + CLASSIC:
        D["r_" + c] = gb[c].rank(pct=True).astype("float32")
    D["y_rank"] = gb["fwd_resid_72h"].rank(pct=True).astype("float32")
    return D


def study(D: pd.DataFrame, h: int, topk: int, cost: float, folds: int = 4, first: float = 0.4) -> list[dict]:
    from sklearn.ensemble import HistGradientBoostingRegressor as R
    emb = h * HOUR
    feats = FEATURE_SETS["full"] + CLASSIC + ["r_" + c for c in CS_BASE + CLASSIC]
    sets = {"原有特征": FEATURE_SETS["full"], "+经典因子与横向排名": feats,
            "仅经典因子(横向排名)": ["r_" + c for c in CLASSIC]}
    L = D[D["y_rank"].notna() & D["fwd_resid_72h"].notna()]
    ts = D["ts"].to_numpy()
    qs = [first + (1 - first) * i / folds for i in range(folds)] + [1.0]
    e = [int(np.quantile(ts, q)) for q in qs]
    e[-1] += 1
    kw = dict(max_depth=4, learning_rate=0.05, max_iter=200, l2_regularization=5.0, min_samples_leaf=500, random_state=0)
    stride = max(1, h // 12)                                  # 训练样本间隔随周期拉长,减少高度重叠的样本
    rows = []
    for name, F in sets.items():
        parts = []
        for i in range(folds):
            lo, hi = e[i], e[i + 1]
            tr = L[(L["ts"] < lo - emb) & (L["h"] % stride == 0)]
            te = L[(L["ts"] >= lo) & (L["ts"] < hi)]
            m = R(**kw).fit(tr[F], tr["y_rank"])
            T = te[["ts", "symbol", "fwd_resid_72h", "funding"]].copy()
            T["s"] = m.predict(te[F])
            parts.append(T)
        T = pd.concat(parts)
        h0 = int(T["ts"].min() // HOUR)
        CS = T[((T["ts"] // HOUR) - h0) % h == 0]               # 截面间隔 h 小时
        ic = model._ic_series(CS, "s", "fwd_resid_72h")
        st = model._ic_stat(ic)
        allm = CS.groupby("ts")["fwd_resid_72h"].transform("mean")
        res = {"周期h": h, "特征": name, "截面数": st["截面数"], "IC": st["IC均值"], "IC_t": st["t"]}
        for tag, asc in (("前k", False), ("后k", True)):
            sub = model._topk(CS, "s", topk, "fwd_resid_72h", asc)
            sp = (sub["fwd_resid_72h"] - allm.loc[sub.index]).groupby(sub["ts"]).mean()
            res[f"{tag}-全体"] = sp.mean()
            res[f"{tag}_t"] = sp.mean() / (sp.std() / np.sqrt(len(sp))) if len(sp) > 2 and sp.std() > 0 else np.nan
        top = model._topk(CS, "s", topk, "fwd_resid_72h", False)
        res["前k 净收益(扣成本)"] = top["fwd_resid_72h"].mean() - cost
        rows.append(res)
        log.info("h=%d %s: IC=%.3f(t=%.1f) 前k-全体=%+.2f%%(t=%.1f) 后k-全体=%+.2f%%(t=%.1f)", h, name, res["IC"], res["IC_t"],
                 res["前k-全体"] * 100, res["前k_t"], res["后k-全体"] * 100, res["后k_t"])
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="更长周期的方向研究")
    ap.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    ap.add_argument("--horizons", default="72,168,336,672")
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--cost", type=float, default=0.002)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cfg = load_config(args.config)
    store = Store(cfg["storage"]["db_path"])
    th = merged_thresholds(cfg["signals"].get("thresholds"))
    frames = with_classic(tune.load_frames(store, th))
    rows = []
    for h in [int(x) for x in args.horizons.split(",")]:
        log.info("==== 周期 %dh ====", h)
        rows += study(table_for(frames, h, th), h, args.topk, args.cost)
    R = pd.DataFrame(rows)
    out = Path(cfg["_base_dir"]) / "reports"
    out.mkdir(exist_ok=True)
    R.to_csv(out / "horizon_study.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    pd.set_option("display.float_format", "{:.4f}".format)
    print(R.to_string(index=False))


if __name__ == "__main__":
    main()
