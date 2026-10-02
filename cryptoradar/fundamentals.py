"""基本面(DefiLlama 免费接口):协议费用、协议收入、持币人收入(回购/分红)、TVL,按币对应到协议或公链。

为什么要它:lag_study.py 显示价格/合约类规则基本都是"涨起来才响";AAVE、PUMP、HYPE 这类币的上涨常由
回购、收入、使用量的变化驱动,这些变化可能先于价格。

数据(日频,DefiLlama 的日期 = 当天 00:00 UTC,代表当天全天;当天结束后才完整):
  fees   dailyFees            协议/公链产生的总费用
  rev    dailyRevenue         协议拿到的收入
  hrev   dailyHoldersRevenue  分给持币人的部分(回购、销毁、质押分红);HYPE、PUMP 的回购都在这里
  tvl    公链用 historicalChainTvl;协议用 /protocols 里的 change_7d(避免下载每个协议几 MB 的完整历史)

特征(daily_features):
  f_{x}_7d      最近 7 个完整日合计(x = fees / rev / hrev)
  f_{x}_ratio   最近 7 日 / 之前 28 日的周均值;> 1 表示在加速
  f_hrev_annual 最近 30 日持币人收入年化;除以市值即动态的"回购收益率"
  f_tvl_chg_7d  TVL 7 日对数变化

云端每轮最多花 budget_s 秒刷新最久没更新的币(每币每天一次),结果存进 data 分支 fundamentals.json。
注意:DefiLlama 会事后修订、补录数据,历史回测比实盘"干净",结论要打折扣。
"""
from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd
import requests

from .defillama import BASE, _get, build_mapping

log = logging.getLogger("fundamentals")
DAY_S = 86_400
KINDS = {"fees": "dailyFees", "rev": "dailyRevenue", "hrev": "dailyHoldersRevenue"}
KEEP_DAYS = 120
COLS = (["f_fees_7d", "f_fees_ratio", "f_rev_7d", "f_rev_ratio", "f_hrev_7d", "f_hrev_ratio", "f_hrev_annual",
         "f_tvl_chg_7d"])


def _series(kind: str, m: dict, data_type: str) -> dict[str, float] | None:
    """{日期秒: 值};先公链后协议。"""
    for k, name in (("chain", m.get("chain")), ("protocol", m.get("protocol"))):
        if not name:
            continue
        path = "overview/fees" if k == "chain" else "summary/fees"
        d = _get(f"{BASE}/{path}/{requests.utils.quote(name)}", {"dataType": data_type})
        if d and d.get("totalDataChart"):
            return {str(int(t)): float(v) for t, v in d["totalDataChart"] if v is not None}
    return None


def _chain_tvl(chain: str) -> dict[str, float] | None:
    d = _get(f"{BASE}/v2/historicalChainTvl/{requests.utils.quote(chain)}")
    if not d:
        return None
    return {str(int(x["date"])): float(x["tvl"]) for x in d if x.get("tvl") is not None}


def protocol_tvl_change(protos: list[dict] | None = None) -> dict[str, float]:
    """{协议 slug: TVL 7 日对数变化},来自 /protocols 的 change_7d(百分比)。"""
    protos = protos if protos is not None else (_get(f"{BASE}/protocols") or [])
    out = {}
    for p in protos:
        c = p.get("change_7d")
        if p.get("slug") and c is not None and c > -100:
            out[p["slug"]] = float(np.log1p(c / 100))
    return out


def fetch_coin(m: dict, keep_days: int | None = KEEP_DAYS, now_s: float | None = None) -> dict:
    """一个币的全部日序列;keep_days=None 时保留完整历史(研究用)。"""
    out = {k: _series(k, m, dt) for k, dt in KINDS.items()}
    out["tvl"] = _chain_tvl(m["chain"]) if m.get("chain") else None
    if keep_days:
        cut = (now_s or time.time()) - keep_days * DAY_S
        out = {k: (None if v is None else {d: x for d, x in v.items() if int(d) >= cut}) for k, v in out.items()}
    return out


def refresh(store: dict | None, coins: list[str], budget_s: float = 60, max_age_h: float = 20,
            mapping_age_h: float = 24 * 7) -> dict:
    """云端:在时间预算内刷新最久没更新的币。store 结构:
    {"mapping": {币: {chain, protocol}}, "mapping_at": 秒, "tvl7": {slug: x}, "tvl7_at": 秒,
     "coins": {币: {"at": 秒, "fees": {...}, "rev": {...}, "hrev": {...}, "tvl": {...}}}}"""
    t0 = time.time()
    st = dict(store or {})
    st.setdefault("coins", {})
    if not st.get("mapping") or t0 - st.get("mapping_at", 0) > mapping_age_h * 3600 or set(coins) - set(st["mapping"]):
        try:
            st["mapping"], st["mapping_at"] = build_mapping(coins), t0
        except Exception as e:
            log.warning("DefiLlama 对应关系获取失败:%s", e)
            st.setdefault("mapping", {})
    if t0 - st.get("tvl7_at", 0) > max_age_h * 3600:
        try:
            st["tvl7"], st["tvl7_at"] = protocol_tvl_change(), t0
        except Exception as e:
            log.warning("DefiLlama 协议 TVL 获取失败:%s", e)
    todo = [c for c in coins if (st["mapping"].get(c) or {}).get("chain") or (st["mapping"].get(c) or {}).get("protocol")]
    todo = [c for c in todo if t0 - st["coins"].get(c, {}).get("at", 0) > max_age_h * 3600]
    todo.sort(key=lambda c: st["coins"].get(c, {}).get("at", 0))
    done = 0
    for c in todo:
        if time.time() - t0 > budget_s:
            break
        try:
            st["coins"][c] = {"at": time.time(), **fetch_coin(st["mapping"][c])}
            done += 1
        except Exception as e:
            log.warning("DefiLlama %s 失败:%s", c, e)
    log.info("DefiLlama:本轮刷新 %d 个币,待刷新 %d 个,用时 %.0f 秒", done, len(todo) - done, time.time() - t0)
    return st


def _daily(d: dict | None) -> pd.Series | None:
    if not d:
        return None
    s = pd.Series({int(k): float(v) for k, v in d.items()}).sort_index()
    s.index = pd.to_datetime(s.index, unit="s").normalize()
    s = s[~s.index.duplicated(keep="last")]
    return s.asfreq("D")


def daily_features(series: dict, protocol_tvl7: float | None = None) -> pd.DataFrame:
    """按天的特征表(index = 数据日)。只用完整日:每个序列最后一天可能还没收完,统一丢掉。"""
    parts = {}
    for k in KINDS:
        s = _daily(series.get(k))
        if s is None or len(s) < 2:
            continue
        s = s.iloc[:-1].fillna(0).clip(lower=0)
        s7 = s.rolling(7, min_periods=7).sum()
        prior = s.shift(7).rolling(28, min_periods=28).sum() / 4
        parts[f"f_{k}_7d"] = s7
        parts[f"f_{k}_ratio"] = s7 / prior.where(prior > 0)
        if k == "hrev":
            parts["f_hrev_annual"] = s.rolling(30, min_periods=30).sum() * 365 / 30
    tvl = _daily(series.get("tvl"))
    if tvl is not None and len(tvl) > 8:
        lt = np.log(tvl.ffill().where(tvl > 0))
        parts["f_tvl_chg_7d"] = lt.diff(7)
    g = pd.DataFrame(parts)
    if protocol_tvl7 is not None and ("f_tvl_chg_7d" not in g or g["f_tvl_chg_7d"].dropna().empty):
        if g.empty:
            g = pd.DataFrame(index=[pd.Timestamp.utcnow().tz_localize(None).normalize()])
        g.loc[g.index[-1], "f_tvl_chg_7d"] = protocol_tvl7
    return g.reindex(columns=COLS)


def latest(store: dict | None) -> dict[str, dict]:
    """{币: 最近一天的特征};给云端用。"""
    out = {}
    if not store:
        return out
    tvl7 = store.get("tvl7") or {}
    for c, s in (store.get("coins") or {}).items():
        m = (store.get("mapping") or {}).get(c) or {}
        g = daily_features(s, None if m.get("chain") else tvl7.get(m.get("protocol")))
        if g.empty:
            continue
        row = g.ffill().iloc[-1]
        out[c] = {k: (None if pd.isna(v) else float(v)) for k, v in row.items()}
        out[c]["day"] = str(g.index[-1].date())
    return out


def attach_latest(frames: dict[str, pd.DataFrame], feats: dict[str, dict]) -> dict[str, pd.DataFrame]:
    """把最新基本面特征写到每个币小时表的最后一行(云端只对最后一行做检测)。"""
    out = {}
    for sym, f in frames.items():
        x = feats.get(sym)
        if x and len(f):
            f = f.copy()
            for k in COLS:
                f.loc[f.index[-1], k] = np.nan if x.get(k) is None else x[k]
        out[sym] = f
    return out


def attach_history(f: pd.DataFrame, daily: pd.DataFrame, lag_days: int = 2) -> pd.DataFrame:
    """研究用:把日频特征并到小时表,数据日 + lag_days 天之后才算已知(DefiLlama 当天数据要到次日以后才完整)。"""
    f = f.copy()
    if daily.empty:
        return f
    avail = (daily.index.astype("int64") // 1_000_000) + lag_days * DAY_S * 1000
    d = daily.set_axis(avail).sort_index()
    left = pd.DataFrame({"ts": f.index.to_numpy()})
    m = pd.merge_asof(left, d.reset_index(names="ts"), on="ts", direction="backward")
    for k in COLS:
        f[k] = m[k].to_numpy()
    return f
