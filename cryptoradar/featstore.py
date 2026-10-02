"""永久特征库:把云端每轮从 OKX 拿到的原始数据按小时存下来,攒够以后用 OKX 自己的数据训练和检验
(现在的机会模型用币安历史训练、在 OKX 上运行,持仓量、费率等差别很大)。

存什么:只存原始输入(K 线、成交额、持仓、多空比、主动买卖比、现货主动买卖量、资金费率),
所有特征都能用 features.build_features 从原始输入重新算出来,存储量约为存全部特征的四分之一。

怎么存(data 分支,按月分文件,只改写有新数据的月份):
  features/YYYY-MM.csv.gz      每币每小时一行,只存已收盘的 K 线
  fundamentals/YYYY-MM.csv.gz  每币每个数据日一行(DefiLlama 的 fundamentals.latest 结果)

"首次看到"原则:同一行(币 + 小时 / 币 + 数据日)以第一次存入时的值为准,之后接口修订也不覆盖,
只用新数据补上原来缺失的列(例如持仓快照晚到)。seen_at = 这一行第一次存入的时间(毫秒),
训练和检验时可以严格只用"当时已经看到"的数据,也能检查某个信息第一次出现时价格是否已经反应。
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("featstore")
HOUR = 3_600_000
RAW_COLS = ["open", "high", "low", "close", "quote_volume", "oi", "top_ls", "taker_ratio",
            "spot_buy", "spot_sell", "funding"]
FEAT_DIR, FUND_DIR = "features", "fundamentals"


def _month(ts_ms) -> pd.Series:
    return pd.to_datetime(pd.Series(ts_ms), unit="ms").dt.strftime("%Y-%m")


def hourly_rows(data: dict, frames: dict, now: int) -> pd.DataFrame:
    """data: {币: (原始小时表, 资金费率, live)},frames: {币: build_features 输出}。
    只取已收盘的小时(开盘时间 + 1 小时 ≤ now);funding 用特征表里折算成 8 小时口径的值。"""
    parts = []
    for sym, (df, _, _) in data.items():
        if df is None or df.empty:
            continue
        d = df.reindex(columns=RAW_COLS).copy()
        f = frames.get(sym)
        if f is not None and "funding" in f:
            d["funding"] = f["funding"].reindex(d.index)
        d = d[d.index.to_numpy() + HOUR <= now]
        if d.empty:
            continue
        d.insert(0, "symbol", sym)
        d.insert(0, "ts", d.index.astype("int64"))
        parts.append(d.reset_index(drop=True))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["ts", "symbol"] + RAW_COLS)


def fundamental_rows(feats: dict[str, dict]) -> pd.DataFrame:
    """fundamentals.latest 的结果 -> 每币一行(day 为数据日)。"""
    rows = [{"day": x.get("day"), "symbol": c, **{k: v for k, v in x.items() if k != "day"}}
            for c, x in (feats or {}).items() if x.get("day")]
    return pd.DataFrame(rows)


def _merge_first(old: pd.DataFrame | None, new: pd.DataFrame, keys: list[str], now: int) -> pd.DataFrame:
    """旧值优先,新数据只补缺失的值和新行;新行的 seen_at = now。"""
    new = new.copy()
    new["seen_at"] = now
    new = new.drop_duplicates(keys, keep="last").set_index(keys)
    if old is None or old.empty:
        out = new
    else:
        old = old.drop_duplicates(keys, keep="first").set_index(keys)
        out = old.combine_first(new)
        out = out[[c for c in list(old.columns) + [c for c in new.columns if c not in old.columns]]]
    return out.reset_index().sort_values(keys).reset_index(drop=True)


def _save_partitioned(prev: Path, out: Path, sub: str, new: pd.DataFrame, keys: list[str], month_of, now: int,
                      float_format: str = "%.8g") -> dict:
    """把 prev/sub 下的旧月份文件原样带到 out/sub,只合并改写有新行的月份。返回 {月份: 行数}。"""
    src, dst = prev / sub, out / sub
    dst.mkdir(parents=True, exist_ok=True)
    if src.exists() and src.resolve() != dst.resolve():
        for p in src.glob("*.csv.gz"):
            shutil.copy2(p, dst / p.name)
    written = {}
    if new is None or new.empty:
        return written
    months = month_of(new).to_numpy()
    for m in np.unique(months):
        path = dst / f"{m}.csv.gz"
        old = None
        if path.exists():
            try:
                old = pd.read_csv(path)
            except Exception as e:           # 文件损坏时不覆盖,另存一份,避免丢掉历史
                log.warning("%s 读取失败(%s),新数据另存", path, e)
                path = dst / f"{m}-{now}.csv.gz"
        merged = _merge_first(old, new[months == m], keys, now)
        merged.to_csv(path, index=False, float_format=float_format)
        written[m] = len(merged)
    return written


def save(prev: Path, out: Path, hourly: pd.DataFrame | None, fund: pd.DataFrame | None, now: int) -> dict:
    """一轮结束时调用。返回各目录写入的月份与行数,用于日志。"""
    res = {FEAT_DIR: _save_partitioned(prev, out, FEAT_DIR, hourly, ["symbol", "ts"], lambda d: _month(d["ts"]), now)}
    res[FUND_DIR] = _save_partitioned(prev, out, FUND_DIR, fund, ["symbol", "day"],
                                      lambda d: d["day"].astype(str).str[:7], now)
    return res


def load(root: Path, sub: str = FEAT_DIR, months: list[str] | None = None) -> pd.DataFrame:
    """研究用:读回特征库(可只读部分月份)。"""
    files = sorted((Path(root) / sub).glob("*.csv.gz"))
    if months:
        files = [p for p in files if p.name[:7] in months]
    return pd.concat([pd.read_csv(p) for p in files], ignore_index=True) if files else pd.DataFrame()


def to_frames(raw: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """特征库原始数据 -> {币: 与 okx_collect 相同列的小时表},可直接交给 features.build_features 重算特征。"""
    out = {}
    for sym, g in raw.groupby("symbol"):
        g = g.drop_duplicates("ts", keep="first").set_index("ts").sort_index()
        out[sym] = g[[c for c in RAW_COLS if c in g]]
    return out
