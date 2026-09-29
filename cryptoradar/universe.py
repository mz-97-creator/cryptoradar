"""监控范围:CoinGecko 市值前 N 名 ∩ 币安 USDT 永续,加上自选观察列表。"""
from __future__ import annotations

import logging

import requests

from .storage import Store, now_ms

log = logging.getLogger(__name__)

CG_MARKETS = "https://api.coingecko.com/api/v3/coins/markets"

# 稳定币、包装资产、质押凭证:它们的"异动"没有交易意义
DEFAULT_EXCLUDE = {
    "USDT", "USDC", "DAI", "FDUSD", "TUSD", "USDE", "USDS", "PYUSD", "USDD", "FRAX",
    "USD1", "RLUSD", "USDTB", "BUSD", "GHO", "CRVUSD", "LUSD", "SUSDE", "USDX", "USDF",
    "EURC", "EURS", "XAUT", "PAXG",
    "WBTC", "WETH", "STETH", "WSTETH", "WEETH", "CBBTC", "RETH", "CBETH", "METH",
    "EZETH", "RSETH", "WBETH", "BSC-USD", "BTCB", "LBTC", "SOLVBTC", "JITOSOL",
    "MSOL", "BNSOL", "JUPSOL", "WBNB", "WTRX", "STX_WRAPPED", "CLBTC", "TBTC",
}

# 币安对低价币用倍数前缀,例如 1000PEPEUSDT
PREFIXES = ["", "1000", "10000", "1000000", "1M"]


def fetch_coingecko_top(top_n: int, api_key: str | None = None) -> list[dict]:
    headers = {"accept": "application/json"}
    if api_key:
        headers["x-cg-demo-api-key"] = api_key
    out: list[dict] = []
    page = 1
    while len(out) < top_n:
        per_page = min(250, top_n - len(out))
        r = requests.get(
            CG_MARKETS,
            params={"vs_currency": "usd", "order": "market_cap_desc",
                    "per_page": per_page, "page": page},
            headers=headers, timeout=20,
        )
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        out.extend(batch)
        page += 1
    return out[:top_n]


def _looks_like_stable(coin: dict) -> bool:
    price = coin.get("current_price") or 0
    sym = (coin.get("symbol") or "").upper()
    return "USD" in sym and 0.97 <= price <= 1.03


def map_to_binance(coins: list[dict], perps: dict[str, str], exclude: set[str]) -> list[dict]:
    """把 CoinGecko 代币映射到币安永续 symbol。同一 ticker 只保留市值排名最高的那个。"""
    used: set[str] = set()
    rows: list[dict] = []
    for c in coins:
        sym = (c.get("symbol") or "").upper()
        if not sym or sym in exclude or _looks_like_stable(c):
            continue
        for p in PREFIXES:
            cand = f"{p}{sym}USDT"
            if cand in perps and cand not in used:
                used.add(cand)
                rows.append({
                    "symbol": cand, "base": sym, "cg_id": c.get("id"), "name": c.get("name"),
                    "rank": c.get("market_cap_rank"), "market_cap": c.get("market_cap"),
                    "watchlist": 0,
                })
                break
    return rows


def refresh_universe(store: Store, api, cfg: dict) -> list[dict]:
    ucfg = cfg["universe"]
    exclude = DEFAULT_EXCLUDE | {s.upper() for s in ucfg.get("exclude", [])}
    watch = [s.upper() for s in ucfg.get("watchlist", [])]

    perps = api.perp_symbols()
    try:
        coins = fetch_coingecko_top(int(ucfg.get("top_n", 200)), ucfg.get("coingecko_api_key") or None)
    except Exception as e:  # CoinGecko 限频或宕机时沿用旧名单
        log.warning("CoinGecko 获取失败(%s),沿用上次的监控名单", e)
        old = store.load_universe()
        if old:
            return old
        coins = []

    rows = map_to_binance(coins, perps, exclude)
    have = {r["symbol"] for r in rows}

    # 自选:按 ticker 找对应合约,并从 CoinGecko 结果里补上排名
    by_sym = {}
    for c in coins:
        by_sym.setdefault((c.get("symbol") or "").upper(), c)
    for base in watch + ["BTC", "ETH"]:  # BTC/ETH 是市场因子,必须在
        sym = next((f"{p}{base}USDT" for p in PREFIXES if f"{p}{base}USDT" in perps), None)
        if sym is None:
            log.warning("自选 %s 在币安没有 USDT 永续合约,跳过", base)
            continue
        if sym in have:
            for r in rows:
                if r["symbol"] == sym and base in watch:
                    r["watchlist"] = 1
            continue
        c = by_sym.get(base, {})
        rows.append({
            "symbol": sym, "base": base, "cg_id": c.get("id"), "name": c.get("name", base),
            "rank": c.get("market_cap_rank"), "market_cap": c.get("market_cap"),
            "watchlist": 1 if base in watch else 0,
        })
        have.add(sym)

    ts = now_ms()
    for r in rows:
        r["updated_at"] = ts
    store.save_universe(rows)
    store.set_state("universe_refreshed_at", ts)
    log.info("监控名单已更新:%d 个合约(市值前 %s 且有币安永续,含自选)",
             len(rows), ucfg.get("top_n"))
    return rows
