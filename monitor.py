"""CryptoRadar 实时监控入口。

用法:
  python monitor.py                 持续运行(按 config.yaml 的间隔循环)
  python monitor.py --once          只跑一轮
  python monitor.py --dry-run       结果打印到屏幕,不推送微信
  python monitor.py --test-push     发一条测试消息,检查微信推送是否绑定成功
"""
from __future__ import annotations

import argparse
import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from cryptoradar.config import load_config
from cryptoradar.monitor import Monitor


def setup_logging(base_dir: str) -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    log_dir = Path(base_dir) / "logs"
    log_dir.mkdir(exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    fh = RotatingFileHandler(log_dir / "cryptoradar.log", maxBytes=5_000_000, backupCount=3, encoding="utf-8")
    fh.setFormatter(fmt)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers = [fh, ch]
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def main() -> None:
    ap = argparse.ArgumentParser(description="CryptoRadar 实时异动监控")
    ap.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    ap.add_argument("--once", action="store_true", help="只运行一轮")
    ap.add_argument("--dry-run", action="store_true", help="不推送,只打印")
    ap.add_argument("--test-push", action="store_true", help="发送测试消息")
    args = ap.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg["_base_dir"])
    mon = Monitor(cfg, dry_run=args.dry_run)

    if args.test_push:
        ok = mon.notifier.send("CryptoRadar 测试消息",
                               "**推送绑定成功** ✅\n\n之后的异动提醒会发到这里。", force=True)
        print("推送成功" if ok else "推送失败,请查看 logs/cryptoradar.log 里的错误信息")
        return
    if args.once:
        mon.run_once()
        return
    mon.loop()


if __name__ == "__main__":
    main()
