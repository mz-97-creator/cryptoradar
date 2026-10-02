"""检验"现货资金流 × 合约杠杆"的组合假设,以及现货特征对相对强弱排序有没有增量。

四个待检验的组合(每个同时检验"父条件"——不加现货条件的版本——才能看出现货信息有没有增量):
  H1 价格上涨 + 现货主动买入持续偏强 + 资金费率温和      → 上涨可能有较持续的买盘支持
  H2 价格上涨 + 持仓/费率急升 + 现货买盘不强              → 上涨可能更依赖杠杆,比较脆弱
  H3 持仓骤降 + 价格下跌 + 现货买盘占比从低位回升          → 去杠杆可能接近尾声
  H4 资金费率为负 + 价格不再下跌 + 现货买盘增强            → 是否存在轧空条件

统计口径和 rule_validation.py 相同,避免被"共同因子"和"同一天多个币一起触发"夸大:
  事件 = 每个币连续触发只算一次(间隔 ≥ 72h);超额 = 事件后收益 - 同一时刻全部币的平均;
  同一天的事件取均值再算 t(Newey-West);19 个检验同时看,门槛 |t| ≥ 3 且前后半段方向一致。

另外检验:在价格类特征的基础上加入现货资金流特征,对"同一时刻哪个币更强"的排序(Spearman IC、前 8/后 8 名)有没有帮助。

用法:  python combo_study.py        (先 python -m cryptoradar.spotflow 回填现货数据)
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

import model
import tune
from cryptoradar import spotflow as sf
from cryptoradar.config import load_config
from cryptoradar.features import add_labels
from cryptoradar.opportunity import FEATURE_SETS, HOUR
from cryptoradar.signals import merged_thresholds
from cryptoradar.storage import Store
from horizon_study import CLASSIC, CS_BASE, table_for, with_classic
from rule_validation import nw_t

log = logging.getLogger("combo")


def conditions(g: pd.DataFrame) -> dict[str, np.ndarray]:
    ret24 = g["ret_24h"]
    ret12 = np.log(g["close"]).diff(12)
    sb, dz = g["spot_buy_z"], g["spot_buy_dz12"]
    f = g["funding_z"]
    c = {}
    c["H1 父:上涨+费率温和"] = (ret24 > 0) & (f.abs() <= 1)
    c["H1 上涨+现货买盘持续强+费率温和"] = c["H1 父:上涨+费率温和"] & (sb >= 1) & (sb.shift(12) >= 0.5)
    c["H2 父:上涨+持仓费率急升"] = (ret24 > 0) & (g["oi_z"] >= 1.5) & (f >= 1.5)
    c["H2 上涨+持仓费率急升+现货买盘不强"] = c["H2 父:上涨+持仓费率急升"] & (sb <= 0)
    c["H3 父:持仓骤降+下跌"] = (g["oi_z"] <= -1.5) & (ret24 < 0)
    c["H3 持仓骤降+下跌+现货买盘从低位回升"] = c["H3 父:持仓骤降+下跌"] & (sb.shift(12) <= -1) & (dz >= 0.5)
    # 要求费率本身为负:funding_z 只说明低于该币自己 30 天的均值,不等于负费率
    c["H4 父:负费率+不再下跌"] = (g["funding"] < 0) & (f <= -1.5) & (ret12 >= 0)
    c["H4 负费率+不再下跌+现货买盘增强"] = c["H4 父:负费率+不再下跌"] & (dz >= 0.5) & (sb >= 0)
    return {k: v.fillna(False).to_numpy() for k, v in c.items()}


def event_table(frames, spot: dict[str, pd.DataFrame], h: int) -> tuple[pd.DataFrame, list[str]]:
    parts, names = [], None
    for f in frames:
        sym = f["symbol"].iloc[0]
        g = f.copy()
        sp = spot.get(sym)
        if sp is None or sp.empty:
            continue
        for c in sf.SPOT_COLS:
            g[c] = sp[c]
        lab = add_labels(g.drop(columns=[c for c in g.columns if c.startswith(("fwd_", "mae_", "mfe_"))]), horizons=(h,))
        cond = conditions(lab)
        names = list(cond)
        d = pd.DataFrame({"ts": lab.index.to_numpy(), "sym": sym, "r": lab[f"fwd_resid_{h}h"].to_numpy()})
        for k, v in cond.items():
            d[k] = v
        parts.append(d)
    return pd.concat(parts, ignore_index=True), names


def test_events(D: pd.DataFrame, names: list[str], market: pd.Series, h: int) -> pd.DataFrame:
    D = D[D.r.notna()].copy()
    D["ex"] = D["r"] - D["ts"].map(market)
    D["day"] = D.ts // 86_400_000
    mid = D.ts.quantile(0.5)
    gap = h
    rows = []
    for k in names:
        ev = []
        for _, g in D.groupby("sym"):
            g = g.sort_values("ts")
            # 按真实时间间隔去重:D 已去掉缺标签的行,行号之差不等于小时数
            ts, keep, last = g["ts"].to_numpy(), [], None
            for p in np.flatnonzero(g[k].to_numpy()):
                if last is None or ts[p] - last >= gap * 3_600_000:
                    keep.append(p)
                    last = ts[p]
            if keep:
                ev.append(g.iloc[keep])
        if not ev:
            rows.append({"组合": k, "周期h": h, "事件数": 0})
            continue
        E = pd.concat(ev)
        daily = E.groupby("day")["ex"].mean()
        e1 = E[E.ts < mid].groupby("day")["ex"].mean().mean()
        e2 = E[E.ts >= mid].groupby("day")["ex"].mean().mean()
        t = nw_t(daily, max(3, h // 24))
        rows.append({"组合": k, "周期h": h, "事件数": len(E), "独立日数": len(daily), "相对全市场超额%": daily.mean() * 100,
                     "t": t, "前半段%": e1 * 100, "后半段%": e2 * 100,
                     "通过": bool(abs(t) >= 3 and np.sign(e1) == np.sign(e2))})
    return pd.DataFrame(rows)


def ic_with_spot(frames, spot: dict[str, pd.DataFrame], th: dict, h: int, topk: int = 8) -> list[dict]:
    from sklearn.ensemble import HistGradientBoostingRegressor as R
    D = table_for(frames, h, th, None)
    ext = pd.concat([sp[sf.SPOT_COLS].assign(symbol=s).rename_axis("ts").reset_index() for s, sp in spot.items() if sp is not None and not sp.empty])
    D = D.merge(ext, on=["ts", "symbol"], how="left")
    gb = D.groupby("ts")
    for c in sf.SPOT_COLS:
        D["r_" + c] = gb[c].rank(pct=True).astype("float32")
    base = FEATURE_SETS["price"] + CLASSIC + ["r_" + c for c in CS_BASE + CLASSIC]
    sets = {"价格类+经典因子": base, "价格类+经典因子+现货资金流": base + sf.SPOT_COLS + ["r_" + c for c in sf.SPOT_COLS]}
    L = D[D["y_rank"].notna() & D["fwd_resid_72h"].notna()]
    ts = D["ts"].to_numpy()
    e = [int(np.quantile(ts, q)) for q in [0.4, 0.55, 0.7, 0.85, 1.0]]
    e[-1] += 1
    kw = dict(max_depth=4, learning_rate=0.05, max_iter=200, l2_regularization=5.0, min_samples_leaf=500, random_state=0)
    stride = max(1, h // 12)
    rows = []
    cov = set(ext.dropna(subset=["spot_buy_z"])["symbol"])
    for name, F in sets.items():
        parts = []
        for i in range(4):
            lo, hi = e[i], e[i + 1]
            tr = L[(L["ts"] < lo - h * HOUR) & (L["h"] % stride == 0) & L["symbol"].isin(cov)]
            te = L[(L["ts"] >= lo) & (L["ts"] < hi) & L["symbol"].isin(cov)]
            m = R(**kw).fit(tr[F], tr["y_rank"])
            T = te[["ts", "symbol", "fwd_resid_72h"]].copy()
            T["s"] = m.predict(te[F])
            parts.append(T)
        T = pd.concat(parts)
        h0 = int(T["ts"].min() // HOUR)
        CS = T[((T["ts"] // HOUR) - h0) % 24 == 0]
        ic = model._ic_series(CS, "s", "fwd_resid_72h")
        allm = CS.groupby("ts")["fwd_resid_72h"].transform("mean")
        res = {"周期h": h, "特征": name, "截面数": len(ic), "IC": ic.mean(), "IC_t(NW)": nw_t(ic, max(1, h // 24))}
        for tag, asc in (("前k", False), ("后k", True)):
            sub = model._topk(CS, "s", topk, "fwd_resid_72h", asc)
            sp_ = (sub["fwd_resid_72h"] - allm.loc[sub.index]).groupby(sub["ts"]).mean()
            res[f"{tag}-全体"] = sp_.mean()
            res[f"{tag}_t(NW)"] = nw_t(sp_, max(1, h // 24))
        rows.append(res)
    return rows


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cfg = load_config(str(Path(__file__).with_name("config.yaml")))
    th = merged_thresholds(cfg["signals"].get("thresholds"))
    store = Store(cfg["storage"]["db_path"])
    frames = with_classic(tune.load_frames(store, th))
    spot = {}
    for f in frames:
        s = f["symbol"].iloc[0]
        spot[s] = sf.features(store.conn, s, store.load_hourly(s))
    covered = [s for s, v in spot.items() if v["spot_buy_z"].notna().any()]
    log.info("有现货数据的币 %d / %d", len(covered), len(frames))
    out = Path(cfg["_base_dir"]) / "reports"
    out.mkdir(exist_ok=True)
    pd.set_option("display.width", 250)
    pd.set_option("display.float_format", "{:.2f}".format)
    res = []
    for h in (72, 168):
        D, names = event_table(frames, spot, h)
        allr = pd.concat([add_labels(f.drop(columns=[c for c in f.columns if c.startswith(("fwd_", "mae_", "mfe_"))]),
                                     horizons=(h,))[f"fwd_resid_{h}h"].rename(f["symbol"].iloc[0]) for f in frames], axis=1)
        market = allr.mean(axis=1)
        market.index = market.index.astype("int64")
        R = test_events(D, names, market, h)
        res.append(R)
        print(f"\n== 事件检验:{h}h 后相对全市场的超额 ==")
        print(R.to_string(index=False))
    pd.concat(res).to_csv(out / "combo_events.csv", index=False, encoding="utf-8-sig")
    rows = []
    for h in (72, 168):
        rows += ic_with_spot(frames, spot, th, h)
    I = pd.DataFrame(rows)
    I.to_csv(out / "combo_ic.csv", index=False, encoding="utf-8-sig")
    print("\n== 现货资金流特征对相对强弱排序的增量(只在有现货数据的币里) ==")
    print(I.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
