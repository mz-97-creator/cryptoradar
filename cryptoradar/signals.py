"""异动规则。每条规则都是对特征表的向量化判断:
- 实时监控只看最后一行
- research.py 用同一批规则在全部历史上做事件研究
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

DEFAULT_THRESHOLDS = {
    "oi_z": 2.0,          # OI 24h 变化的 z-score
    "resid_z": 2.5,       # 剔除 BTC beta 后 24h 收益的 z-score
    "funding_z": 2.5,     # 资金费率 z-score
    "funding_abs": 0.001,  # 资金费率绝对值(每 8 小时,0.1%)
    "vol_z": 2.5,         # 24h 成交额 z-score
    "ret1h_z": 3.5,       # 1 小时收益 z-score
    "ls_z": 2.5,          # 大户持仓多空比 z-score
}


@dataclass(frozen=True)
class Rule:
    id: str
    name: str
    weight: float
    fn: Callable[[pd.DataFrame, dict], pd.Series]
    hint: str


def _c(f: pd.DataFrame, col: str) -> pd.Series:
    return f[col] if col in f else pd.Series(np.nan, index=f.index)


RULES: list[Rule] = [
    Rule("OI_DIV", "OI 激增但价格未涨", 2.0,
         lambda f, t: (_c(f, "oi_z") >= t["oi_z"]) & (_c(f, "resid_24h_z") <= 0.5),
         "合约在建仓但现货没跟,方向未定;结合资金费率/多空比看是谁在加仓"),
    Rule("OI_TREND", "放量上涨且 OI 增加", 1.5,
         lambda f, t: (_c(f, "oi_z") >= t["oi_z"]) & (_c(f, "resid_24h_z") >= 1.0),
         "新资金顺势加仓;若资金费率同时偏高,追多拥挤"),
    Rule("OI_FLUSH", "OI 骤降伴随下跌", 1.5,
         lambda f, t: (_c(f, "oi_z") <= -t["oi_z"]) & (_c(f, "ret_24h") < 0),
         "多头去杠杆/被清算,杠杆出清后常有技术性反弹,但方向需确认"),
    Rule("SHORT_COVER", "OI 骤降伴随上涨", 1.5,
         lambda f, t: (_c(f, "oi_z") <= -t["oi_z"]) & (_c(f, "ret_24h") > 0),
         "空头回补推动的上涨,燃料耗尽后容易回落"),
    Rule("FUND_HOT", "资金费率偏高", 1.5,
         lambda f, t: (_c(f, "funding_z") >= t["funding_z"]) | (_c(f, "funding") >= t["funding_abs"]),
         "多头拥挤,持多成本上升,回调时易连环爆仓"),
    Rule("FUND_COLD", "资金费率偏负", 1.5,
         lambda f, t: (_c(f, "funding_z") <= -t["funding_z"]) | (_c(f, "funding") <= -t["funding_abs"]),
         "空头拥挤,存在轧空燃料"),
    Rule("RESID", "脱离大盘的独立行情", 1.0,
         lambda f, t: _c(f, "resid_24h_z").abs() >= t["resid_z"],
         "剔除 BTC 影响后仍显著,多为代币自身消息或资金驱动"),
    Rule("SHOCK", "1 小时急涨/急跌", 1.0,
         lambda f, t: _c(f, "ret_1h_z").abs() >= t["ret1h_z"],
         "短时冲击,先确认是否有消息面"),
    Rule("VOL", "成交额异常放大", 1.0,
         lambda f, t: _c(f, "vol_z") >= t["vol_z"],
         "关注度突然上升"),
    Rule("LS_EXT", "多空比极值", 0.5,
         lambda f, t: _c(f, "top_ls_z").abs() >= t["ls_z"],
         "仓位一边倒(噪音较大,仅作辅助)"),
]
RULES_BY_ID = {r.id: r for r in RULES}


def merged_thresholds(cfg_th: dict | None) -> dict:
    t = dict(DEFAULT_THRESHOLDS)
    t.update(cfg_th or {})
    return t


def evaluate_frame(f: pd.DataFrame, thresholds: dict) -> pd.DataFrame:
    """返回 bool 表:每列一条规则。"""
    return pd.DataFrame({r.id: r.fn(f, thresholds).fillna(False).astype(bool) for r in RULES},
                        index=f.index)


def evaluate_last(f: pd.DataFrame, thresholds: dict) -> list[Rule]:
    if f.empty:
        return []
    last = f.iloc[[-1]]
    return [r for r in RULES if bool(r.fn(last, thresholds).fillna(False).iloc[0])]


# ------------------------------------------------------------------ 格式化
def _pct(x, digits=1) -> str:
    if x is None or pd.isna(x):
        return "—"
    return f"{x * 100:+.{digits}f}%"


def _z(x) -> str:
    return "—" if x is None or pd.isna(x) else f"{x:+.1f}"


def _price(p) -> str:
    if p is None or pd.isna(p):
        return "—"
    if p >= 100:
        return f"${p:,.1f}"
    if p >= 1:
        return f"${p:,.3f}"
    return f"${p:.4g}"


def describe(symbol: str, row: pd.Series, fired: list[Rule], rank=None, watch=False,
             ls_label: str = "大户多空比") -> str:
    rk = f"#{rank}" if rank else "#?"
    star = " ★自选" if watch else ""
    ret24 = np.expm1(row.get("ret_24h")) if pd.notna(row.get("ret_24h")) else np.nan
    oi24 = np.expm1(row.get("oi_chg_24h")) if pd.notna(row.get("oi_chg_24h")) else np.nan
    lines = [
        f"**{symbol}** {rk}{star} {_price(row.get('close'))} · 24h {_pct(ret24)}"
        f"(剔除BTC后 z={_z(row.get('resid_24h_z'))})",
    ]
    for r in fired:
        lines.append(f"- {r.name}:{r.hint}")
    adr = row.get("adr_14d")
    adr_s = "—" if adr is None or pd.isna(adr) else f"{adr * 100:.1f}%"
    ls = row.get("top_ls")
    ls_s = "—" if ls is None or pd.isna(ls) else f"{ls:.2f}"
    lines.append(
        f"- OI 24h {_pct(oi24)} (z={_z(row.get('oi_z'))}) · 资金费率 {_pct(row.get('funding'), 3)}/8h "
        f"(z={_z(row.get('funding_z'))}) · 成交额 z={_z(row.get('vol_z'))} · "
        f"{ls_label} {ls_s} · 日均波动 {adr_s}"
    )
    return "\n".join(lines)
