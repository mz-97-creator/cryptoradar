"""特征计算。实时监控和历史研究共用这一份代码,避免"回测一套、实盘一套"。

所有 z-score 都是和该代币自己过去 30 天(720 小时)比较,且只用过去数据(shift(1)),
不含当前值,因此没有前视偏差。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

HOUR = 3_600_000
WINDOW = 720      # 30 天
MIN_PERIODS = 168  # 至少 7 天历史才给出 z-score


def rolling_z(x: pd.Series, window: int = WINDOW, min_periods: int = MIN_PERIODS) -> pd.Series:
    roll = x.rolling(window, min_periods=min_periods)
    m = roll.mean().shift(1)
    s = roll.std().shift(1)
    return (x - m) / s.where(s > 0)


def _to_grid(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    idx = np.arange(int(df.index.min()), int(df.index.max()) + HOUR, HOUR)
    return df.reindex(idx)


def _safe_log(s: pd.Series) -> pd.Series:
    return np.log(s.where(s > 0))


def build_features(df: pd.DataFrame, btc: pd.DataFrame, funding: pd.Series | None = None,
                   funding_interval_h: float = 8.0, eth: pd.DataFrame | None = None,
                   live: dict | None = None) -> pd.DataFrame:
    """df/btc/eth: hourly 表(index=ts 毫秒)。funding: 已结算资金费率(index=结算时间)。

    live: 实时覆盖最后一行,{"oi_now", "funding_rate", "mark_price"}。
    """
    df = _to_grid(df.copy())
    if df.empty:
        return pd.DataFrame()
    idx = df.index

    if live:
        last = idx[-1]
        if live.get("oi_now") and pd.isna(df.at[last, "oi"]):
            df.at[last, "oi"] = live["oi_now"]

    f = pd.DataFrame(index=idx)
    f["close"] = df["close"]
    lc = _safe_log(df["close"])
    f["ret_1h"] = lc.diff()
    f["ret_24h"] = lc.diff(24)

    blc = _safe_log(btc["close"]).reindex(idx)
    f["btc_ret_1h"] = blc.diff()
    f["btc_ret_24h"] = blc.diff(24)
    f["_btc_lc"] = blc

    cov = f["ret_1h"].rolling(WINDOW, min_periods=MIN_PERIODS).cov(f["btc_ret_1h"])
    var = f["btc_ret_1h"].rolling(WINDOW, min_periods=MIN_PERIODS).var()
    f["beta"] = (cov / var.where(var > 0)).shift(1)
    f["resid_24h"] = f["ret_24h"] - f["beta"] * f["btc_ret_24h"]
    f["resid_24h_z"] = rolling_z(f["resid_24h"])
    f["ret_1h_z"] = rolling_z(f["ret_1h"])

    oi = df["oi"].ffill(limit=2)
    f["oi"] = oi
    f["oi_chg_24h"] = _safe_log(oi).diff(24)
    f["oi_z"] = rolling_z(f["oi_chg_24h"])

    qv24 = df["quote_volume"].rolling(24, min_periods=20).sum()
    f["vol_24h"] = qv24
    f["vol_z"] = rolling_z(_safe_log(qv24))

    ls = df["top_ls"].ffill(limit=2)
    f["top_ls"] = ls
    f["top_ls_z"] = rolling_z(_safe_log(ls))

    tk = _safe_log(df["taker_ratio"]).rolling(24, min_periods=20).mean()
    f["taker_24h"] = tk
    f["taker_z"] = rolling_z(tk)

    # 资金费率:统一折算成"每 8 小时"口径;每根 K 线收盘时点上已知的最近一次结算值
    scale = 8.0 / (funding_interval_h or 8.0)
    fund = pd.Series(np.nan, index=idx)
    if funding is not None and len(funding):
        fr = funding.sort_index()
        close_times = idx + HOUR
        aligned = fr.reindex(fr.index.union(close_times)).ffill().reindex(close_times)
        fund = pd.Series(aligned.to_numpy() * scale, index=idx)
    if live and live.get("funding_rate") is not None:
        fund.iloc[-1] = live["funding_rate"] * scale  # 当期预测费率
    f["funding"] = fund
    f["funding_z"] = rolling_z(fund)

    rng = (df["high"].rolling(24, min_periods=20).max() - df["low"].rolling(24, min_periods=20).min())
    f["range_24h"] = rng / df["close"]
    f["adr_14d"] = f["range_24h"].rolling(336, min_periods=72).mean()

    if eth is not None and not eth.empty:
        elc = _safe_log(eth["close"]).reindex(idx)
        f["ethbtc_ret_24h"] = (elc - blc).diff(24)

    f["_high"] = df["high"]
    f["_low"] = df["low"]
    return f


def add_labels(f: pd.DataFrame, horizons=(24, 72)) -> pd.DataFrame:
    """研究用:未来收益、剔除 BTC beta 后的残差收益、持有期内最大不利/有利波动。

    入场价 = 当前 K 线收盘价;MAE/MFE 用未来 h 根 K 线的最低/最高价。
    """
    f = f.copy()
    lc = np.log(f["close"])
    for h in horizons:
        fwd = lc.shift(-h) - lc
        btc_fwd = f["_btc_lc"].shift(-h) - f["_btc_lc"]
        f[f"fwd_ret_{h}h"] = fwd
        f[f"fwd_resid_{h}h"] = fwd - f["beta"] * btc_fwd
        fut_low = f["_low"].iloc[::-1].rolling(h, min_periods=h).min().iloc[::-1].shift(-1)
        fut_high = f["_high"].iloc[::-1].rolling(h, min_periods=h).max().iloc[::-1].shift(-1)
        f[f"mae_{h}h"] = fut_low / f["close"] - 1
        f[f"mfe_{h}h"] = fut_high / f["close"] - 1
    return f
