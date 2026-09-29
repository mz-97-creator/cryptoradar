"""把 OKX 返回的数据整理成与 features.build_features 相同的小时表。

时间对齐(与本地版约定一致):
  hourly.ts = K 线开盘时间 T
  oi / top_ls = T+1h 时刻的快照(OKX rubik 的时间戳 t 视为快照时点,记到 K 线 t-1h)
  taker_ratio = [T, T+1h) 内主动买量 / 主动卖量(rubik 时间戳 t 视为该小时的开始)
  top_ls 在云端版是"多空账户比"(OKX 不提供按币种汇总的大户持仓比)
"""
from __future__ import annotations

import pandas as pd

from .okx_api import HOUR_MS, OKX


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def collect(okx: OKX, ccy: str, inst_id: str) -> tuple[pd.DataFrame, pd.Series, dict]:
    kl = okx.candles(inst_id)
    if not kl:
        return pd.DataFrame(), pd.Series(dtype=float), {}
    df = pd.DataFrame({
        "open": [_f(r[1]) for r in kl], "high": [_f(r[2]) for r in kl],
        "low": [_f(r[3]) for r in kl], "close": [_f(r[4]) for r in kl],
        "quote_volume": [_f(r[7]) for r in kl],
    }, index=pd.Index([int(r[0]) for r in kl], name="ts"))
    df = df[~df.index.duplicated(keep="last")].sort_index()

    oi = pd.Series({int(r[0]) - HOUR_MS: _f(r[1]) for r in okx.oi_volume(ccy)}, dtype=float)
    # rubik 给的是 USD 持仓额,换算成币本位数量,避免把价格涨跌误当作持仓变化
    df["oi"] = (oi.reindex(df.index) / df["close"]).where(lambda s: s > 0)

    ls = pd.Series({int(r[0]) - HOUR_MS: _f(r[1]) for r in okx.long_short(ccy)}, dtype=float)
    df["top_ls"] = ls.reindex(df.index)

    tk = {}
    for r in okx.taker(ccy):
        sell, buy = _f(r[1]), _f(r[2])
        if sell > 0:
            tk[int(r[0])] = buy / sell
    df["taker_ratio"] = pd.Series(tk, dtype=float).reindex(df.index)

    hist = okx.funding_history(inst_id)
    funding = pd.Series({int(h["fundingTime"]): _f(h.get("realizedRate") or h.get("fundingRate"))
                         for h in hist}, dtype=float).sort_index()

    now = okx.funding_now(inst_id)
    interval_h = 8.0
    try:
        interval_h = (int(now["nextFundingTime"]) - int(now["fundingTime"])) / HOUR_MS or 8.0
    except (KeyError, TypeError, ValueError):
        pass
    live = {"funding_rate": _f(now.get("fundingRate")) if now else None,
            "funding_interval_h": interval_h}
    return df, funding, live
