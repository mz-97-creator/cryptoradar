"""DefiLlama 公开接口(免费,无需 key)的链上/协议基本面数据:TVL 与费用,按"币"对应到公链或协议。

不用的(付费):/emissions(代币解锁/排放)。

对应关系:
  - 公链币(ETH、SOL、OP、ARB……):/v2/chains 里每条链有 tokenSymbol,按它对应;取公链的 TVL 和公链费用
  - 协议币(AAVE、UNI、PENDLE、ENA……):/protocols 里按 symbol 对应,同一协议的各版本按父协议合并,取 TVL 与费用
  - 两者都有时优先公链
特征(日频,全部"只用当时已知的信息",使用时再统一滞后 2 天,见 attach):
  tvl_chg_7d / tvl_chg_30d   TVL 的对数变化
  tvl_dd_90d                 TVL 相对 90 天高点的回撤
  fees_chg_7d / fees_chg_30d 近 7/30 天费用合计 vs 前一个同长度区间的对数变化
注意:DefiLlama 的历史数据会事后修订、补录,回测里比实盘更"干净",用它得出的任何结论都要打折扣。
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

log = logging.getLogger("defillama")
BASE = "https://api.llama.fi"
DAY = 86_400_000
FEATURES = ["tvl_chg_7d", "tvl_chg_30d", "tvl_dd_90d", "fees_chg_7d", "fees_chg_30d"]
_THOUSAND = ("1000",)


def coin_of(symbol: str) -> str:
    s = symbol.upper().removesuffix("USDT")
    return s[4:] if s.startswith(_THOUSAND) else s


def _get(url: str, params: dict | None = None, retries: int = 4):
    for i in range(retries):
        try:
            r = requests.get(url, params=params, timeout=90)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            if i == retries - 1:
                log.warning("请求失败 %s: %s", url, e)
                return None
            time.sleep(2 * (i + 1))
    return None


def build_mapping(coins: list[str]) -> dict[str, dict]:
    """{币: {"chain": 链名或 None, "protocol": 协议 slug 或 None}}。"""
    chains = _get(f"{BASE}/v2/chains") or []
    by_tok: dict[str, dict] = {}
    for c in chains:
        t = (c.get("tokenSymbol") or "").upper()
        if t and (t not in by_tok or (c.get("tvl") or 0) > (by_tok[t].get("tvl") or 0)):
            by_tok[t] = c
    protos = _get(f"{BASE}/protocols") or []
    groups: dict[str, dict[str, float]] = {}
    for p in protos:
        sym = (p.get("symbol") or "").upper()
        if not sym or sym == "-":
            continue
        key = (p.get("parentProtocol") or "").replace("parent#", "") or p.get("slug")
        groups.setdefault(sym, {})
        groups[sym][key] = groups[sym].get(key, 0.0) + float(p.get("tvl") or 0)
    out = {}
    for c in coins:
        chain = by_tok.get(c)
        g = groups.get(c)
        out[c] = {"chain": chain["name"] if chain else None,
                  "protocol": max(g, key=g.get) if g else None}
    return out


def _tvl_chain(name: str) -> pd.Series | None:
    d = _get(f"{BASE}/v2/historicalChainTvl/{requests.utils.quote(name)}")
    if not d:
        return None
    return pd.Series({int(x["date"]): float(x["tvl"]) for x in d if x.get("tvl") is not None})


def _tvl_protocol(slug: str) -> pd.Series | None:
    d = _get(f"{BASE}/protocol/{slug}")
    if not d or not d.get("tvl"):
        return None
    return pd.Series({int(x["date"]): float(x["totalLiquidityUSD"]) for x in d["tvl"]})


def _fees(kind: str, name: str) -> pd.Series | None:
    path = "overview/fees" if kind == "chain" else "summary/fees"
    d = _get(f"{BASE}/{path}/{requests.utils.quote(name)}", {"dataType": "dailyFees"})
    if not d or not d.get("totalDataChart"):
        return None
    return pd.Series({int(t): float(v) for t, v in d["totalDataChart"] if v is not None})


def fetch_all(coins: list[str], cache_dir: Path, refresh: bool = False) -> dict[str, dict]:
    """抓取并缓存每个币的 TVL / 费用日序列。返回 mapping(含是否取到数据)。"""
    cache_dir.mkdir(parents=True, exist_ok=True)
    mp = build_mapping(coins)
    for i, c in enumerate(coins, 1):
        f = cache_dir / f"{c}.json"
        m = mp[c]
        if f.exists() and not refresh:
            m["cached"] = True
            continue
        tvl = fees = None
        if m["chain"]:
            tvl, fees = _tvl_chain(m["chain"]), _fees("chain", m["chain"])
        if (tvl is None or tvl.empty) and m["protocol"]:
            tvl = _tvl_protocol(m["protocol"])
        if (fees is None or fees.empty) and m["protocol"]:
            fees = _fees("protocol", m["protocol"])
        f.write_text(json.dumps({"map": m, "tvl": None if tvl is None else {str(k): v for k, v in tvl.items()},
                                 "fees": None if fees is None else {str(k): v for k, v in fees.items()}}),
                     encoding="utf-8")
        log.info("[%d/%d] %s -> 链 %s / 协议 %s: TVL %s 天, 费用 %s 天", i, len(coins), c, m["chain"], m["protocol"],
                 0 if tvl is None else len(tvl), 0 if fees is None else len(fees))
        time.sleep(0.3)
    return mp


def _daily(s: dict | None) -> pd.Series | None:
    if not s:
        return None
    x = pd.Series({int(k): v for k, v in s.items()})
    x.index = pd.to_datetime(x.index, unit="s").normalize()
    x = x[~x.index.duplicated(keep="last")].sort_index()
    return x.asfreq("D").ffill()


def coin_features(cache_dir: Path) -> pd.DataFrame:
    """所有币的日频特征,列:coin、avail_ts(毫秒,这一天的数据最早可用的时刻,已滞后 2 天)+ FEATURES。"""
    rows = []
    for f in sorted(cache_dir.glob("*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        coin = f.stem
        tvl, fees = _daily(d.get("tvl")), _daily(d.get("fees"))
        if tvl is None and fees is None:
            continue
        g = pd.DataFrame(index=(tvl if tvl is not None else fees).index)
        if tvl is not None:
            lt = np.log(tvl.where(tvl > 0))
            g["tvl_chg_7d"], g["tvl_chg_30d"] = lt.diff(7), lt.diff(30)
            g["tvl_dd_90d"] = tvl / tvl.rolling(90, min_periods=30).max() - 1
        if fees is not None:
            fees = fees.reindex(g.index).fillna(0.0).clip(lower=0)
            f7, f30 = fees.rolling(7).sum(), fees.rolling(30).sum()
            g["fees_chg_7d"] = np.log((f7 + 1) / (f7.shift(7) + 1))
            g["fees_chg_30d"] = np.log((f30 + 1) / (f30.shift(30) + 1))
        g = g.reindex(columns=FEATURES)
        g["coin"] = coin
        # 数据日 + 2 天 之后才算"已知";和 pandas 的时间精度无关地换成毫秒
        g["avail_ts"] = (g.index - pd.Timestamp(0)) // pd.Timedelta(milliseconds=1) + 2 * DAY
        rows.append(g.reset_index(drop=True))
    return pd.concat(rows) if rows else pd.DataFrame(columns=["coin", "avail_ts"] + FEATURES)


def attach(D: pd.DataFrame, feats: pd.DataFrame) -> pd.DataFrame:
    """把日频特征并到小时表 D(要有 ts、symbol 列):每个币每小时取"已可用"的最新一天。"""
    D = D.copy()
    D["coin"] = D["symbol"].map(coin_of)
    left = D.sort_values("ts")
    right = feats.sort_values("avail_ts")
    out = pd.merge_asof(left, right, left_on="ts", right_on="avail_ts", by="coin", direction="backward")
    return out.drop(columns=["avail_ts"]).reset_index(drop=True)


def main() -> None:
    import argparse
    import sqlite3
    ap = argparse.ArgumentParser(description="抓取 DefiLlama 的 TVL / 费用并缓存到 data/defillama/")
    ap.add_argument("--db", default="data/cryptoradar.db")
    ap.add_argument("--out", default="data/defillama")
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    syms = [r[0] for r in sqlite3.connect(args.db).execute("SELECT DISTINCT symbol FROM hourly")]
    coins = sorted({coin_of(s) for s in syms})
    mp = fetch_all(coins, Path(args.out), args.refresh)
    print(f"覆盖 {sum(1 for v in mp.values() if v['chain'] or v['protocol'])}/{len(coins)} 个币")


if __name__ == "__main__":
    main()
