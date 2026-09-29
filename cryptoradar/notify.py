"""推送:PushPlus(微信)、Server酱(微信)、Telegram、控制台。

PushPlus 免费额度(实名认证后):微信渠道每天 200 次、每分钟 5 次、同内容 1 小时内最多 3 次。
超过 200 次当天停发,超过 2000 次会被封禁 7 天,所以这里有每日上限保护。
"""
from __future__ import annotations

import logging
import os
import re
import time
from datetime import datetime, timezone

import requests

from .storage import Store

log = logging.getLogger(__name__)

PUSHPLUS_URL = "https://www.pushplus.plus/send"
SERVERCHAN_URL = "https://sctapi.ftqq.com/{key}.send"
TELEGRAM_URL = "https://api.telegram.org/bot{token}/sendMessage"


def _plain(md: str) -> str:
    return re.sub(r"\*\*(.+?)\*\*", r"\1", md)


class Notifier:
    def __init__(self, cfg: dict, store: Store, tz: timezone):
        n = cfg.get("notify", {})
        self.channel = (n.get("channel") or "console").lower()
        self.pushplus_token = os.getenv("PUSHPLUS_TOKEN") or n.get("pushplus_token") or ""
        self.serverchan_key = os.getenv("SERVERCHAN_SENDKEY") or n.get("serverchan_sendkey") or ""
        self.tg_token = os.getenv("TELEGRAM_BOT_TOKEN") or n.get("telegram_bot_token") or ""
        self.tg_chat = os.getenv("TELEGRAM_CHAT_ID") or str(n.get("telegram_chat_id") or "")
        self.daily_limit = int(n.get("daily_limit", 150))
        self.store = store
        self.tz = tz
        self._last_send = 0.0

    def _today_start_ms(self) -> int:
        now = datetime.now(self.tz)
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        return int(start.astimezone(timezone.utc).timestamp() * 1000)

    def remaining_today(self) -> int:
        return self.daily_limit - self.store.pushes_since(self._today_start_ms())

    def send(self, title: str, content: str, force: bool = False) -> bool:
        title = title[:95]
        content = content[:19000]
        if not force and self.remaining_today() <= 0:
            log.warning("今日推送已达上限 %d 条,本条只写日志:%s", self.daily_limit, title)
            return False
        # PushPlus 每分钟 5 次;两次推送至少间隔 13 秒
        gap = time.monotonic() - self._last_send
        if gap < 13 and self.channel != "console":
            time.sleep(13 - gap)
        try:
            ok = getattr(self, f"_send_{self.channel}")(title, content)
        except AttributeError:
            log.error("未知推送渠道 %s,可选 pushplus / serverchan / telegram / console", self.channel)
            ok = False
        except Exception as e:  # 推送失败不能让监控崩掉
            log.error("推送失败:%s", e)
            ok = False
        self._last_send = time.monotonic()
        self.store.log_push(self.channel, ok, title)
        return ok

    # ----------------------------------------------------------- channels
    def _send_console(self, title: str, content: str) -> bool:
        print("=" * 60)
        print(title)
        print("-" * 60)
        print(content)
        print("=" * 60, flush=True)
        return True

    def _send_pushplus(self, title: str, content: str) -> bool:
        if not self.pushplus_token:
            raise RuntimeError("未配置 pushplus_token")
        r = requests.post(PUSHPLUS_URL, json={
            "token": self.pushplus_token, "title": title,
            "content": content, "template": "markdown",
        }, timeout=15)
        data = r.json()
        if data.get("code") != 200:
            raise RuntimeError(f"PushPlus 返回 {data.get('code')}: {data.get('msg')}")
        return True

    def _send_serverchan(self, title: str, content: str) -> bool:
        if not self.serverchan_key:
            raise RuntimeError("未配置 serverchan_sendkey")
        r = requests.post(SERVERCHAN_URL.format(key=self.serverchan_key),
                          data={"title": title, "desp": content}, timeout=15)
        data = r.json()
        if data.get("code") != 0:
            raise RuntimeError(f"Server酱返回 {data.get('code')}: {data.get('message')}")
        return True

    def _send_telegram(self, title: str, content: str) -> bool:
        if not (self.tg_token and self.tg_chat):
            raise RuntimeError("未配置 telegram_bot_token / telegram_chat_id")
        text = f"{title}\n\n{_plain(content)}"[:4000]
        r = requests.post(TELEGRAM_URL.format(token=self.tg_token),
                          json={"chat_id": self.tg_chat, "text": text,
                                "disable_web_page_preview": True}, timeout=15)
        data = r.json()
        if not data.get("ok"):
            raise RuntimeError(f"Telegram 返回:{data.get('description')}")
        return True
