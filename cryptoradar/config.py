"""读取 config.yaml,补齐默认值。"""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

DEFAULTS = {
    "universe": {
        "top_n": 200,
        "watchlist": ["OP"],
        "exclude": [],
        "refresh_hours": 24,
        "coingecko_api_key": "",
    },
    "schedule": {"interval_minutes": 15, "offset_seconds": 90},
    "signals": {
        "cooldown_hours": 6,
        "min_score_to_push": 2.5,
        "watchlist_min_score": 1.0,
        "max_per_message": 10,
        "thresholds": {},
    },
    "price_alerts": [],
    "funding_alerts": [],
    "buybacks": [],
    "notify": {
        "channel": "console",
        "pushplus_token": "",
        "serverchan_sendkey": "",
        "telegram_bot_token": "",
        "telegram_chat_id": "",
        "daily_limit": 150,
        "heartbeat_hour": 9,
    },
    "display_timezone": "Asia/Singapore",
    "storage": {"db_path": "data/cryptoradar.db"},
    "research": {"symbols": ["OPUSDT", "BTCUSDT", "ETHUSDT"], "start": "2022-06-01"},
}


def _merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"找不到配置文件 {path}。请先把 config.example.yaml 复制一份,改名为 config.yaml"
        )
    with open(path, encoding="utf-8") as fh:
        user = yaml.safe_load(fh) or {}
    cfg = _merge(DEFAULTS, user)
    # 数据库路径相对配置文件所在目录
    db = Path(cfg["storage"]["db_path"])
    if not db.is_absolute():
        cfg["storage"]["db_path"] = str((path.parent / db).resolve())
    cfg["_base_dir"] = str(path.parent.resolve())
    return cfg
