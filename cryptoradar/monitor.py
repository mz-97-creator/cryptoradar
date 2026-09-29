"""监控主循环。"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .binance_api import HOUR_MS, BinanceBlockedError, BinanceFutures
from .collector import collect_symbol, update_live
from .features import build_features
from .notify import Notifier
from .signals import describe, evaluate_last, merged_thresholds
from .storage import Store, now_ms
from .universe import refresh_universe

log = logging.getLogger(__name__)

FEATURE_KEYS = ["close", "ret_24h", "resid_24h_z", "ret_1h_z", "oi_chg_24h", "oi_z",
                "funding", "funding_z", "vol_z", "top_ls", "top_ls_z", "taker_z",
                "adr_14d", "beta", "ethbtc_ret_24h"]


class Monitor:
    def __init__(self, cfg: dict, api: BinanceFutures | None = None, dry_run: bool = False):
        self.cfg = cfg
        if dry_run:
            cfg["notify"]["channel"] = "console"
        self.tz = ZoneInfo(cfg.get("display_timezone", "Asia/Singapore"))
        self.store = Store(cfg["storage"]["db_path"])
        self.api = api or BinanceFutures()
        self.notifier = Notifier(cfg, self.store, self.tz)
        self.th = merged_thresholds(cfg["signals"].get("thresholds"))
        self.fail_streak = 0

    # ------------------------------------------------------------ helpers
    def _local(self, ms: int | None = None) -> datetime:
        ms = ms if ms is not None else now_ms()
        return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).astimezone(self.tz)

    def _universe(self) -> list[dict]:
        uni = self.store.load_universe()
        last = self.store.get_state("universe_refreshed_at", 0) or 0
        hours = float(self.cfg["universe"].get("refresh_hours", 24))
        if not uni or now_ms() - last > hours * HOUR_MS:
            uni = refresh_universe(self.store, self.api, self.cfg)
        return uni

    def _funding_intervals(self) -> dict[str, float]:
        cached = self.store.get_state("funding_intervals")
        ts = self.store.get_state("funding_intervals_at", 0) or 0
        if cached is None or now_ms() - ts > 24 * HOUR_MS:
            try:
                cached = {r["symbol"]: float(r.get("fundingIntervalHours", 8))
                          for r in self.api.funding_info()}
                self.store.set_state("funding_intervals", cached)
                self.store.set_state("funding_intervals_at", now_ms())
            except Exception as e:
                log.warning("获取资金费率周期失败:%s", e)
                cached = cached or {}
        return cached

    # -------------------------------------------------------------- cycle
    def collect(self, uni: list[dict]) -> None:
        symbols = [u["symbol"] for u in uni]
        update_live(self.store, self.api, symbols, self._funding_intervals())
        t0 = time.time()
        failed = 0
        for i, s in enumerate(symbols, 1):
            try:
                collect_symbol(self.store, self.api, s)
            except BinanceBlockedError:
                raise
            except Exception as e:
                failed += 1
                log.warning("%s 采集失败:%s", s, e)
            if i % 25 == 0:
                log.info("采集进度 %d/%d(%.0f 秒)", i, len(symbols), time.time() - t0)
        log.info("采集完成:%d 个合约,失败 %d,用时 %.0f 秒", len(symbols), failed, time.time() - t0)

    def compute(self, uni: list[dict]) -> list[tuple[dict, pd.Series, list]]:
        since = now_ms() - 45 * 24 * HOUR_MS
        btc = self.store.load_hourly("BTCUSDT", since)
        eth = self.store.load_hourly("ETHUSDT", since)
        if btc.empty:
            raise RuntimeError("没有 BTC 数据,无法计算残差收益")
        out = []
        for u in uni:
            s = u["symbol"]
            df = self.store.load_hourly(s, since)
            if len(df) < 200:
                continue
            live = self.store.get_live(s)
            fund = self.store.load_funding(s, since - 30 * 24 * HOUR_MS)
            try:
                f = build_features(df, btc, fund, live.get("funding_interval_h") or 8.0, eth, live)
            except Exception as e:
                log.warning("%s 特征计算失败:%s", s, e)
                continue
            if f.empty:
                continue
            row = f.iloc[-1]
            fired = evaluate_last(f, self.th)
            out.append((u, row, fired))
        return out

    def _price_alerts(self, results) -> list[str]:
        msgs = []
        prices = {u["symbol"]: row.get("close") for u, row, _ in results}
        for pa in self.cfg.get("price_alerts") or []:
            s = pa.get("symbol")
            p = self.store.get_live(s).get("mark_price") or prices.get(s)
            if p is None or pd.isna(p):
                continue
            for side in ("below", "above"):
                if side not in pa:
                    continue
                level = float(pa[side])
                hit = p <= level if side == "below" else p >= level
                key = f"pa:{s}:{side}:{level}"
                was = self.store.get_state(key, False)
                if hit and not was:
                    word = "跌破" if side == "below" else "突破"
                    msgs.append(f"**{s}** 现价 {p:.6g} 已{word} {level:g}:{pa.get('note', '')}")
                self.store.set_state(key, bool(hit))
        return msgs

    def _market_line(self, results) -> tuple[str, str]:
        btc = next((row for u, row, _ in results if u["symbol"] == "BTCUSDT"), None)
        if btc is None:
            return "", ""
        b24 = np.expm1(btc.get("ret_24h"))
        eb = btc.get("ethbtc_ret_24h")
        eb_s = "—" if eb is None or pd.isna(eb) else f"{np.expm1(eb) * 100:+.1f}%"
        short = f"BTC ${btc['close']:,.0f} {b24 * 100:+.1f}%"
        line = (f"{self._local():%m-%d %H:%M} · BTC ${btc['close']:,.0f} (24h {b24 * 100:+.1f}%) · "
                f"ETH/BTC 24h {eb_s} · 扫描 {len(results)} 个合约")
        return short, line

    def alert(self, results) -> None:
        sc = self.cfg["signals"]
        cooldown = float(sc.get("cooldown_hours", 6)) * HOUR_MS
        now = now_ms()
        candidates = []
        for u, row, fired in results:
            if not fired:
                continue
            s = u["symbol"]
            score = sum(r.weight for r in fired)
            fresh = [r for r in fired
                     if (last := self.store.last_rule_fire(s, r.id)) is None or now - last > cooldown]
            watch = bool(u.get("watchlist"))
            threshold = sc.get("watchlist_min_score", 1.0) if watch else sc.get("min_score_to_push", 2.5)
            want = bool(fresh) and score >= threshold
            candidates.append((u, row, fired, score, want))

        pushing = sorted([c for c in candidates if c[4]], key=lambda c: (-c[0].get("watchlist", 0), -c[3]))
        max_n = int(sc.get("max_per_message", 10))
        shown, extra = pushing[:max_n], pushing[max_n:]
        price_msgs = self._price_alerts(results)

        ok = False
        if shown or price_msgs:
            short, line = self._market_line(results)
            parts = [line, ""]
            if price_msgs:
                parts += ["**⚠️ 价位提醒**"] + [f"- {m}" for m in price_msgs] + [""]
            for u, row, fired, score, _ in shown:
                parts.append(describe(u["symbol"], row, fired, u.get("rank"), bool(u.get("watchlist"))))
                parts.append("")
            if extra:
                parts.append("另有 " + ", ".join(f"{c[0]['symbol']}({c[3]:g})" for c in extra) + " 触发,已省略")
            parts.append("> 信号只描述仓位结构异动,不代表方向;规则尚未经历史 IC 验证。")
            head = "⚠️价位 " if price_msgs else ""
            title = f"{head}CryptoRadar {len(pushing)}个异动 | {short}"
            ok = self.notifier.send(title, "\n".join(parts))

        shown_syms = {c[0]["symbol"] for c in shown}
        for u, row, fired, score, want in candidates:
            feats = {k: (None if pd.isna(row.get(k)) else float(row.get(k))) for k in FEATURE_KEYS if k in row}
            pushed = ok and u["symbol"] in shown_syms
            self.store.log_alert(now, u["symbol"], [r.id for r in fired], score,
                                 float(row["close"]), feats, pushed)

    def heartbeat(self, results) -> None:
        hour = self.cfg["notify"].get("heartbeat_hour")
        if hour is None:
            return
        local = self._local()
        today = local.strftime("%Y-%m-%d")
        if local.hour != int(hour) or self.store.get_state("heartbeat_date") == today:
            return
        _, line = self._market_line(results)
        since = now_ms() - 24 * HOUR_MS
        n_alerts = self.store.conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(pushed),0) FROM alerts WHERE ts>=?", (since,)).fetchone()
        parts = [line, f"过去 24h:触发 {n_alerts[0]} 次,推送 {n_alerts[1]} 次", "", "**自选**"]
        for u, row, fired in results:
            if u.get("watchlist"):
                parts.append(describe(u["symbol"], row, fired, u.get("rank"), True))
        movers = sorted([(u, row) for u, row, _ in results if pd.notna(row.get("resid_24h_z"))],
                        key=lambda x: -abs(x[1]["resid_24h_z"]))[:5]
        parts += ["", "**24h 独立行情最强的 5 个(剔除 BTC 影响)**"]
        for u, row in movers:
            parts.append(f"- {u['symbol']} 24h {np.expm1(row['ret_24h']) * 100:+.1f}% · 残差 z={row['resid_24h_z']:+.1f}")
        if self.notifier.send(f"CryptoRadar 日报 {today}", "\n".join(parts)):
            self.store.set_state("heartbeat_date", today)

    def run_once(self) -> None:
        uni = self._universe()
        self.collect(uni)
        results = self.compute(uni)
        self.alert(results)
        self.heartbeat(results)

    def loop(self) -> None:
        interval = int(self.cfg["schedule"].get("interval_minutes", 15)) * 60
        offset = int(self.cfg["schedule"].get("offset_seconds", 90))
        while True:
            try:
                self.run_once()
                self.fail_streak = 0
            except BinanceBlockedError as e:
                log.error(str(e))
                self.notifier.send("CryptoRadar 无法访问币安", str(e), force=True)
                time.sleep(3600)
                continue
            except KeyboardInterrupt:
                raise
            except Exception:
                self.fail_streak += 1
                log.exception("本轮运行出错(连续第 %d 次)", self.fail_streak)
                if self.fail_streak == 3:
                    self.notifier.send("CryptoRadar 连续 3 轮出错", "请查看 logs/cryptoradar.log", force=True)
            now = time.time()
            nxt = (now // interval + 1) * interval + offset
            if nxt - now > interval + offset:
                nxt -= interval
            log.info("下一轮:%s", datetime.fromtimestamp(nxt, tz=self.tz).strftime("%H:%M:%S"))
            time.sleep(max(5, nxt - time.time()))
