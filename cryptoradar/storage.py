"""SQLite 存储。所有时间戳均为 UTC 毫秒。

hourly 表的约定(实时采集和历史回填完全一致):
  ts = 1 小时 K 线的开盘时间 T,覆盖 [T, T+1h)
  oi / top_ls = T+1h 时刻(即这根 K 线收盘时)的快照
  taker_ratio = 这 1 小时内主动买量 / 主动卖量
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS hourly (
    symbol TEXT NOT NULL,
    ts INTEGER NOT NULL,
    open REAL, high REAL, low REAL, close REAL,
    volume REAL, quote_volume REAL, taker_buy_quote REAL, trades INTEGER,
    oi REAL, oi_value REAL, top_ls REAL, taker_ratio REAL,
    PRIMARY KEY (symbol, ts)
);
CREATE TABLE IF NOT EXISTS funding (
    symbol TEXT NOT NULL,
    ts INTEGER NOT NULL,
    rate REAL,
    PRIMARY KEY (symbol, ts)
);
CREATE TABLE IF NOT EXISTS live (
    symbol TEXT PRIMARY KEY,
    ts INTEGER,
    mark_price REAL,
    funding_rate REAL,
    next_funding_time INTEGER,
    funding_interval_h REAL,
    oi_now REAL
);
CREATE TABLE IF NOT EXISTS universe (
    symbol TEXT PRIMARY KEY,
    base TEXT, cg_id TEXT, name TEXT,
    rank INTEGER, market_cap REAL,
    watchlist INTEGER DEFAULT 0,
    updated_at INTEGER
);
CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts INTEGER, symbol TEXT, rules TEXT, score REAL,
    price REAL, features TEXT, pushed INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_alerts_sym_ts ON alerts(symbol, ts);
CREATE TABLE IF NOT EXISTS push_log (
    ts INTEGER, channel TEXT, ok INTEGER, title TEXT
);
CREATE TABLE IF NOT EXISTS state (
    key TEXT PRIMARY KEY, value TEXT
);
CREATE TABLE IF NOT EXISTS backfill_done (
    symbol TEXT, day TEXT, status TEXT,
    PRIMARY KEY (symbol, day)
);
"""


def now_ms() -> int:
    return int(time.time() * 1000)


class Store:
    def __init__(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.conn = sqlite3.connect(str(path), timeout=30)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    # --------------------------------------------------------------- state
    def get_state(self, key: str, default=None):
        row = self.conn.execute("SELECT value FROM state WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def set_state(self, key: str, value) -> None:
        self.conn.execute(
            "INSERT INTO state(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, json.dumps(value)),
        )
        self.conn.commit()

    # -------------------------------------------------------------- hourly
    def upsert_klines(self, symbol: str, klines: list) -> int:
        rows = [
            (symbol, int(k[0]), float(k[1]), float(k[2]), float(k[3]), float(k[4]),
             float(k[5]), float(k[7]), float(k[10]), int(k[8]))
            for k in klines
        ]
        self.conn.executemany(
            """INSERT INTO hourly(symbol, ts, open, high, low, close, volume,
                                  quote_volume, taker_buy_quote, trades)
               VALUES(?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(symbol, ts) DO UPDATE SET
                 open=excluded.open, high=excluded.high, low=excluded.low,
                 close=excluded.close, volume=excluded.volume,
                 quote_volume=excluded.quote_volume,
                 taker_buy_quote=excluded.taker_buy_quote, trades=excluded.trades""",
            rows,
        )
        self.conn.commit()
        return len(rows)

    def upsert_hist_column(self, symbol: str, column: str, pairs: list[tuple[int, float]]) -> int:
        """pairs: [(bar_ts, value)]。只更新指定列,不覆盖价格数据。"""
        assert column in {"oi", "oi_value", "top_ls", "taker_ratio"}
        self.conn.executemany(
            f"""INSERT INTO hourly(symbol, ts, {column}) VALUES(?,?,?)
                ON CONFLICT(symbol, ts) DO UPDATE SET {column}=excluded.{column}""",
            [(symbol, int(t), v) for t, v in pairs],
        )
        self.conn.commit()
        return len(pairs)

    def last_ts(self, symbol: str, column: str = "close") -> int | None:
        row = self.conn.execute(
            f"SELECT MAX(ts) FROM hourly WHERE symbol=? AND {column} IS NOT NULL", (symbol,)
        ).fetchone()
        return row[0] if row and row[0] is not None else None

    def load_hourly(self, symbol: str, since_ts: int | None = None) -> pd.DataFrame:
        q = "SELECT * FROM hourly WHERE symbol=?"
        params: list = [symbol]
        if since_ts is not None:
            q += " AND ts>=?"
            params.append(int(since_ts))
        q += " ORDER BY ts"
        df = pd.read_sql_query(q, self.conn, params=params)
        return df.set_index("ts") if not df.empty else df

    # ------------------------------------------------------------- funding
    def upsert_funding(self, symbol: str, rows: list[tuple[int, float]]) -> None:
        self.conn.executemany(
            "INSERT OR REPLACE INTO funding(symbol, ts, rate) VALUES(?,?,?)",
            [(symbol, int(t), float(r)) for t, r in rows],
        )
        self.conn.commit()

    def last_funding_ts(self, symbol: str) -> int | None:
        row = self.conn.execute("SELECT MAX(ts) FROM funding WHERE symbol=?", (symbol,)).fetchone()
        return row[0] if row and row[0] is not None else None

    def load_funding(self, symbol: str, since_ts: int | None = None) -> pd.Series:
        q = "SELECT ts, rate FROM funding WHERE symbol=?"
        params: list = [symbol]
        if since_ts is not None:
            q += " AND ts>=?"
            params.append(int(since_ts))
        df = pd.read_sql_query(q + " ORDER BY ts", self.conn, params=params)
        return df.set_index("ts")["rate"] if not df.empty else pd.Series(dtype=float)

    # ---------------------------------------------------------------- live
    def upsert_live(self, symbol: str, **fields) -> None:
        cols = ["ts"] + list(fields)
        vals = [now_ms()] + list(fields.values())
        sets = ", ".join(f"{c}=excluded.{c}" for c in cols)
        self.conn.execute(
            f"INSERT INTO live(symbol, {', '.join(cols)}) VALUES(?, {', '.join('?' * len(cols))}) "
            f"ON CONFLICT(symbol) DO UPDATE SET {sets}",
            [symbol] + vals,
        )

    def get_live(self, symbol: str) -> dict:
        cur = self.conn.execute("SELECT * FROM live WHERE symbol=?", (symbol,))
        row = cur.fetchone()
        if not row:
            return {}
        return dict(zip([d[0] for d in cur.description], row))

    def commit(self) -> None:
        self.conn.commit()

    # ------------------------------------------------------------ universe
    def save_universe(self, rows: list[dict]) -> None:
        self.conn.execute("DELETE FROM universe")
        self.conn.executemany(
            """INSERT INTO universe(symbol, base, cg_id, name, rank, market_cap, watchlist, updated_at)
               VALUES(:symbol, :base, :cg_id, :name, :rank, :market_cap, :watchlist, :updated_at)""",
            rows,
        )
        self.conn.commit()

    def load_universe(self) -> list[dict]:
        cur = self.conn.execute("SELECT * FROM universe ORDER BY COALESCE(rank, 99999)")
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]

    # -------------------------------------------------------------- alerts
    def log_alert(self, ts: int, symbol: str, rules: list[str], score: float,
                  price: float, features: dict, pushed: bool) -> None:
        self.conn.execute(
            "INSERT INTO alerts(ts, symbol, rules, score, price, features, pushed) VALUES(?,?,?,?,?,?,?)",
            (ts, symbol, ",".join(rules), score, price,
             json.dumps(features, ensure_ascii=False, default=float), int(pushed)),
        )
        self.conn.commit()

    def last_rule_fire(self, symbol: str, rule: str) -> int | None:
        row = self.conn.execute(
            "SELECT MAX(ts) FROM alerts WHERE symbol=? AND pushed=1 AND instr(','||rules||',', ?) > 0",
            (symbol, f",{rule},"),
        ).fetchone()
        return row[0] if row and row[0] is not None else None

    # ---------------------------------------------------------------- push
    def log_push(self, channel: str, ok: bool, title: str) -> None:
        self.conn.execute("INSERT INTO push_log VALUES(?,?,?,?)", (now_ms(), channel, int(ok), title))
        self.conn.commit()

    def pushes_since(self, since_ms: int) -> int:
        row = self.conn.execute("SELECT COUNT(*) FROM push_log WHERE ts>=?", (since_ms,)).fetchone()
        return int(row[0])
