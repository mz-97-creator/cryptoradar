"""模型在 OKX 数据上的"跨交易所"检验:模型用币安历史训练,云端喂的是 OKX 数据。
  1. 同一批币、同一批小时,用 OKX 特征和用币安特征各算一遍预测,看两者有多一致
  2. 用 OKX 自己的价格算出 72h 后的真实结果,检查波动排序、概率校准、回撤分位数是否仍然成立
注意:模型训练数据到 9 月 27 日,与 OKX 近 5 周窗口在时间上有重叠,所以这里检验的是"换交易所的数据口径"这一件事,
而不是样本外时间;时间上的样本外要靠线上 opportunity_live 的实盘核对(每 6 小时记录预测、满 72h 结算)。

用法: python xvenue_model.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import model
import tune
from cryptoradar import opportunity as opp
from cryptoradar.config import load_config
from cryptoradar.features import add_labels, build_features
from cryptoradar.okx_api import OKX
from cryptoradar.okx_collect import collect
from cryptoradar.signals import merged_thresholds
from cryptoradar.storage import Store

HOUR = 3_600_000


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    base = Path(__file__).parent
    cfg = load_config(str(base / "config.yaml"))
    th = merged_thresholds(cfg["signals"].get("thresholds"))
    bundle = opp.load_bundle(base / "models" / "opportunity_price.joblib")
    models, thr = bundle["models"], bundle["models"].thr
    store = Store(cfg["storage"]["db_path"])

    # ---- 币安侧:同样的币
    bfr = {f["symbol"].iloc[0].replace("USDT", ""): f for f in tune.load_frames(store, th)}
    coins = [c for c in bfr if c not in ("BTC",) and not c.startswith("1000")] + ["BTC"]

    # ---- OKX 侧:和云端完全一样的采集与特征
    okx = OKX()
    data = {}
    for c in coins:
        try:
            data[c] = collect(okx, c, f"{c}-USDT-SWAP")
        except Exception as e:
            print("跳过", c, str(e)[:60])
    btc, eth = data["BTC"][0], data.get("ETH", (pd.DataFrame(),))[0]
    ofr = {}
    for c, (df, fund, live) in data.items():
        if len(df) < 200:
            continue
        f = build_features(df, btc, fund, live.get("funding_interval_h") or 8.0, eth)
        if c != "BTC":
            f = add_labels(f, horizons=(72,))
        f["symbol"] = c
        ofr[c] = f
    common = [c for c in ofr if c in bfr and c != "BTC"]
    print("OKX 与币安都有的币:", len(common))

    def table(frs):
        D = opp.build_table(list(frs), None, with_labels=False)
        P = models.predict(D)
        D = D[["ts", "symbol", "close"]].join(P[["p_up", "p_dn", "pred_lrange", "pred_mae_q10"]])
        D["coin"] = D["symbol"].str.replace("USDT", "")
        return D

    Do = table([ofr[c] for c in ofr])
    tmin = int(Do["ts"].min()) + 24 * 30 * HOUR      # 前 30 天做特征预热(滚动 z 分数需要)
    Db = table([bfr[c].assign(symbol=c) for c in common + ["BTC"]])
    Db = Db[Db["ts"] >= tmin]
    Do = Do[Do["ts"] >= tmin]

    # 1. 两个交易所的预测一致性
    J = Do.merge(Db, on=["ts", "coin"], suffixes=("_o", "_b"))
    J = J[J["coin"] != "BTC"]
    cs = J.groupby("ts").apply(lambda g: g["pred_lrange_o"].rank().corr(g["pred_lrange_b"].rank()), include_groups=False)
    print(f"\n[1] 同币同小时的预测一致性({len(J)} 行)")
    print(f"   预测波动(截面排序相关性,各时刻平均): {cs.mean():.3f}")
    print(f"   预测波动 绝对差中位: {np.exp(J['pred_lrange_o']).sub(np.exp(J['pred_lrange_b'])).abs().median():.3f}")
    print(f"   P上 绝对差中位: {(J['p_up_o'] - J['p_up_b']).abs().median():.3f}   P下 绝对差中位: {(J['p_dn_o'] - J['p_dn_b']).abs().median():.3f}")

    # 2. OKX 自己的价格算真实结果
    rows = []
    for c in common:
        f = ofr[c]
        y = pd.DataFrame({"ts": f.index.to_numpy(), "coin": c, "resid": f["fwd_resid_72h"].to_numpy(),
                          "mae": f["mae_72h"].to_numpy(), "mfe": f["mfe_72h"].to_numpy()})
        rows.append(y)
    Y = pd.concat(rows).dropna()
    Y["lrange"] = np.log((Y["mfe"] - Y["mae"]).clip(lower=1e-4))
    V = Do[Do["coin"] != "BTC"].merge(Y, on=["ts", "coin"])
    V = V[(V["ts"] - V["ts"].min()) // HOUR % 6 == 0]
    ic = V.groupby("ts").apply(lambda g: g["pred_lrange"].rank().corr(g["lrange"].rank()) if len(g) >= 10 else np.nan,
                               include_groups=False).dropna()
    print(f"\n[2] 用 OKX 价格算的真实结果({len(V)} 个预测,{ic.shape[0]} 个时刻)")
    print(f"   波动排序相关性(OKX 特征→OKX 真实): {ic.mean():.3f}   (币安训练样本外参考值 0.66)")
    print(f"   P上 预测均值 {V['p_up'].mean():.3f} / 实际 {(V['resid'] > thr).mean():.3f};  P下 预测均值 {V['p_dn'].mean():.3f} / 实际 {(V['resid'] < -thr).mean():.3f}")
    print(f"   回撤 q10 越界比例: {(V['mae'] < V['pred_mae_q10']).mean():.3f} (理想 0.10)")
    print(f"   预测波动中位 {np.exp(V['pred_lrange']).median():.3f} / 实际波动中位 {np.exp(V['lrange']).median():.3f}")


if __name__ == "__main__":
    main()
