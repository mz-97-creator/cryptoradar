"""事件库:有明确时间(和金额)的事件,记录"官方时间"和"我们第一次看到的时间",用来检验信息是否早于价格反应。

来源(全部免费、无需 key):
  OKX 公告        /api/v5/support/announcements(上新),pTime = 发布时间
  币安公告        bapi cms 列表 catalogId=48(新币上线),releaseDate = 发布时间;美国 IP 可能被拒,失败就跳过
  交易对清单比对  OKX 现货(有 listTime)、币安现货、Upbit、Coinbase:每轮和上一轮比,新出现的交易对记为上新,
                  没有官方时间的以首次看到的时间为准(第一次运行只建立清单、不产生事件)
  HYPE 回购执行   Hyperliquid 援助基金地址的现货成交(每一笔买入的数量、价格、时间),按天汇总成回购金额
解锁时间表目前没有可靠的免费来源(DefiLlama emissions、Tokenomist 都要付费),留空。

事件表列(EVENT_COLS):
  id        去重键
  kind      spot_list 现货上新 / perp_list 合约上新 / buyback_day 回购(每天一行,amount_usd = 当天金额)
  source    okx_ann / binance_ann / okx_spot / binance_spot / upbit / coinbase / hyperliquid
  symbol    币(大写,去掉 1000 前缀)
  event_ts  官方时间(毫秒);没有就等于 seen_at
  seen_at   我们第一次看到的时间(毫秒);历史回填的事件 seen_at 为空
  amount_usd、title
"""
from __future__ import annotations

import logging
import re
import time

import numpy as np
import pandas as pd
import requests

log = logging.getLogger("events")
EVENT_COLS = ["id", "kind", "source", "symbol", "event_ts", "seen_at", "amount_usd", "title"]
UA = {"User-Agent": "CryptoRadar/1.0"}
HL_AF = "0xfefefefefefefefefefefefefefefefefefefefe"      # Hyperliquid 援助基金(手续费回购 HYPE)
DAY = 86_400_000


def _get(url, params=None, timeout=20):
    try:
        r = requests.get(url, params=params, headers=UA, timeout=timeout)
        if r.status_code != 200:
            log.info("%s -> HTTP %s", url, r.status_code)
            return None
        return r.json()
    except Exception as e:
        log.info("%s 失败:%s", url, e)
        return None


def _norm(sym: str) -> str:
    s = sym.upper().strip()
    return s[4:] if s.startswith("1000") and len(s) > 4 else s


# ------------------------------------------------------------------ 公告
_PERP = re.compile(r"\b([A-Z0-9]{2,15}?)USDT?\s+(?:and\s+[A-Z0-9]+\s+)?Perpetual", re.I)
_PERP_SYM = re.compile(r"\b([A-Z0-9]{2,15}?)USDT\b")
_PERP_FOR = re.compile(r"perpetual (?:futures|swaps?|contracts?) for ([A-Z0-9]{2,15}(?:\s*,\s*[A-Z0-9]{2,15})*)", re.I)
_PAIR = re.compile(r"\b([A-Z0-9]{2,15})/(?:USDT|USD|USDC)\b")
_PAREN = re.compile(r"\(([A-Z0-9]{2,15})\)")
_SKIP = re.compile(r"\b(delist\w*|remov\w*|stocks?|tokenized|collateral|convert|bstocks?|tradfi)\b", re.I)


def classify(title: str) -> list[tuple[str, str]]:
    """从公告标题里认出 (kind, symbol)。只认上新,其余(下架、理财、股票代币等)返回空。"""
    t = title or ""
    low = t.lower()
    if _SKIP.search(t):
        return []
    out = []
    if "perpetual" in low:
        out += [("perp_list", _norm(s)) for s in _PERP_SYM.findall(t)]
        for grp in _PERP_FOR.findall(t):
            out += [("perp_list", _norm(x)) for x in re.split(r"\s*,\s*", grp)]
    elif "hodler airdrops" in low or re.search(r"\b(will list|to list|will launch|adds? .* spot)", low):
        syms = _PAIR.findall(t) or _PAREN.findall(t)
        out += [("spot_list", _norm(s)) for s in syms]
    return [(k, s) for k, s in dict.fromkeys(out) if s not in ("USDT", "USD", "USDC", "")]


def okx_announcements(pages: int = 1) -> list[dict]:
    rows = []
    for p in range(1, pages + 1):
        d = _get("https://www.okx.com/api/v5/support/announcements",
                 {"annType": "announcements-new-listings", "page": p})
        det = ((d or {}).get("data") or [{}])[0].get("details") or []
        for a in det:
            for kind, sym in classify(a.get("title", "")):
                rows.append({"id": f"okx_ann|{a.get('url') or a['title']}|{sym}", "kind": kind, "source": "okx_ann",
                             "symbol": sym, "event_ts": int(a["pTime"]), "title": a["title"][:200]})
        if not det:
            break
    return rows


def binance_announcements(pages: int = 1, page_size: int = 20) -> list[dict]:
    rows = []
    for p in range(1, pages + 1):
        d = _get("https://www.binance.com/bapi/composite/v1/public/cms/article/list/query",
                 {"type": 1, "catalogId": 48, "pageNo": p, "pageSize": page_size})
        cats = ((d or {}).get("data") or {}).get("catalogs") or []
        arts = cats[0].get("articles") if cats else []
        for a in arts or []:
            for kind, sym in classify(a.get("title", "")):
                rows.append({"id": f"binance_ann|{a.get('code') or a['id']}|{sym}", "kind": kind,
                             "source": "binance_ann", "symbol": sym, "event_ts": int(a["releaseDate"]),
                             "title": a["title"][:200]})
        if not arts:
            break
        time.sleep(0.3)
    return rows


# ------------------------------------------------------------------ 交易对清单比对
def listing_sets() -> dict[str, dict[str, int | None]]:
    """{来源: {币: 官方上线时间或 None}}。拿不到的来源不出现在结果里(下一轮再比)。"""
    out = {}
    d = _get("https://www.okx.com/api/v5/public/instruments", {"instType": "SPOT"})
    if d and d.get("data"):
        out["okx_spot"] = {_norm(x["baseCcy"]): (int(x["listTime"]) if x.get("listTime") else None)
                           for x in d["data"] if x.get("quoteCcy") in ("USDT", "USDC", "USD")}
    d = _get("https://data-api.binance.vision/api/v3/exchangeInfo", {"permissions": "SPOT"})
    if d and d.get("symbols"):
        out["binance_spot"] = {_norm(x["baseAsset"]): None for x in d["symbols"]
                               if x.get("quoteAsset") in ("USDT", "USDC", "FDUSD") and x.get("status") == "TRADING"}
    d = _get("https://api.upbit.com/v1/market/all")
    if isinstance(d, list) and d:
        out["upbit"] = {_norm(x["market"].split("-")[1]): None for x in d if x["market"].startswith(("KRW-", "USDT-"))}
    d = _get("https://api.exchange.coinbase.com/products")
    if isinstance(d, list) and d:
        out["coinbase"] = {_norm(x["base_currency"]): None for x in d
                           if x.get("quote_currency") in ("USD", "USDC", "USDT") and not x.get("trading_disabled")}
    return out


def diff_listings(prev: dict | None, cur: dict, now: int) -> tuple[list[dict], dict]:
    """新出现的币 -> spot_list 事件。返回 (事件, 新清单)。第一次见到某个来源时只建立清单。"""
    prev = prev or {}
    rows, merged = [], dict(prev)
    for src, coins in cur.items():
        old = prev.get(src)
        if old is not None:
            for c, lt in coins.items():
                if c not in old:
                    rows.append({"id": f"{src}|{c}", "kind": "spot_list", "source": src, "symbol": c,
                                 "event_ts": lt or now, "title": f"{src} 新增 {c}"})
        merged[src] = {c: lt for c, lt in coins.items()}
    return rows, {k: sorted(v) for k, v in merged.items()}


# ------------------------------------------------------------------ HYPE 回购
def hype_buyback_fills(start_ms: int, end_ms: int | None = None, max_pages: int = 200) -> pd.DataFrame:
    """援助基金的买入成交(time, px, sz, usd)。接口每次最多 2000 条,按时间往后翻页。"""
    rows, t = [], int(start_ms)
    end_ms = end_ms or int(time.time() * 1000)
    for _ in range(max_pages):
        try:
            r = requests.post("https://api.hyperliquid.xyz/info", json={"type": "userFillsByTime", "user": HL_AF,
                                                                    "startTime": t, "endTime": end_ms}, timeout=30)
            d = r.json() if r.status_code == 200 else []
        except Exception as e:
            log.info("Hyperliquid 失败:%s", e)
            break
        if not d:
            break
        rows += [{"time": int(x["time"]), "px": float(x["px"]), "sz": float(x["sz"]), "tid": x.get("tid")}
                 for x in d if x.get("side") == "B" and x.get("coin") == "@107"]       # @107 = HYPE 现货
        last = max(int(x["time"]) for x in d)
        if len(d) < 2000 or last <= t:
            break
        t = last + 1
        time.sleep(0.2)
    df = pd.DataFrame(rows, columns=["time", "px", "sz", "tid"]).drop_duplicates("tid")
    df["usd"] = df["px"] * df["sz"]
    return df


def buyback_days(fills: pd.DataFrame, symbol: str = "HYPE", source: str = "hyperliquid") -> list[dict]:
    """按天(UTC)汇总成回购事件;event_ts = 当天结束(当天的金额要到这时才完整)。"""
    if fills.empty:
        return []
    day = fills["time"] // DAY
    g = fills.groupby(day)["usd"].sum()
    return [{"id": f"{source}|{symbol}|{int(d)}", "kind": "buyback_day", "source": source, "symbol": symbol,
             "event_ts": int((d + 1) * DAY), "amount_usd": float(v), "title": f"{symbol} 当日回购"} for d, v in g.items()]


# ------------------------------------------------------------------ 事件表
def merge(prev: pd.DataFrame | None, rows: list[dict], now: int | None) -> pd.DataFrame:
    """按 id 去重,旧行优先(保留第一次看到的时间);新行 seen_at = now(历史回填传 None)。
    回购这类当天还在累计的事件,金额用最新值更新,seen_at 不变。"""
    new = pd.DataFrame(rows, columns=[c for c in EVENT_COLS if c != "seen_at"])
    new["seen_at"] = now
    new = new.reindex(columns=EVENT_COLS)
    if prev is None or prev.empty:
        out = new
    else:
        prev = prev.reindex(columns=EVENT_COLS)
        upd = new.set_index("id")["amount_usd"]
        out = pd.concat([prev, new[~new["id"].isin(prev["id"])]], ignore_index=True)
        m = out["id"].isin(upd.index) & out["kind"].eq("buyback_day")
        out.loc[m, "amount_usd"] = out.loc[m, "id"].map(upd)
    return out.drop_duplicates("id", keep="first").sort_values("event_ts").reset_index(drop=True)


def refresh(prev: pd.DataFrame | None, state: dict | None, now: int, budget_s: float = 20) -> tuple[pd.DataFrame, dict]:
    """云端每轮:公告第 1 页、交易对清单比对、HYPE 当天和前一天的回购。返回 (事件表, 新状态)。"""
    t0 = time.time()
    st = dict(state or {})
    rows = []
    for fn in (okx_announcements, binance_announcements):
        if time.time() - t0 < budget_s:
            rows += fn(1)
    if time.time() - t0 < budget_s:
        new_rows, st["listings"] = diff_listings(st.get("listings"), listing_sets(), now)
        rows += new_rows
    if time.time() - t0 < budget_s:
        rows += buyback_days(hype_buyback_fills(now // DAY * DAY - DAY, now, max_pages=5))
    return merge(prev, rows, now), st


def recent_text(ev: pd.DataFrame | None, now: int, hours: int = 48, watch: set | None = None) -> list[str]:
    if ev is None or ev.empty:
        return ["暂无事件"]
    e = ev[(ev["kind"] != "buyback_day") & (pd.to_numeric(ev["seen_at"], errors="coerce") >= now - hours * 3_600_000)]
    lines = []
    for r in e.sort_values("event_ts", ascending=False).head(30).itertuples():
        star = "⭐" if watch and r.symbol in watch else ""
        t = pd.to_datetime(int(r.event_ts), unit="ms").strftime("%m-%d %H:%M")
        lines.append(f"- {t} UTC {r.symbol}{star} {'现货上新' if r.kind == 'spot_list' else '合约上新'}({r.source}):{r.title}")
    bb = ev[ev["kind"] == "buyback_day"].sort_values("event_ts")
    if len(bb) >= 8:
        last = bb.iloc[-2]                                     # 最后一行是还在累计的当天
        avg = bb.iloc[-30:-2]["amount_usd"].mean()
        lines.append(f"- HYPE 链上回购:{pd.to_datetime(int(last.event_ts) - 1, unit='ms'):%m-%d} ${last.amount_usd / 1e6:,.2f}M,"
                     f"之前日均 ${avg / 1e6:,.2f}M")
    return lines or [f"近 {hours} 小时没有新事件"]
