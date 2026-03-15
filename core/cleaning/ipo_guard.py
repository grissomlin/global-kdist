# core/cleaning/ipo_guard.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import os

import pandas as pd


@dataclass
class IPOGuardStats:
    exempt_rows: int = 0
    notes: str = ""


def _env_bool(name: str, default: str = "0") -> bool:
    return str(os.getenv(name, default)).strip().lower() in ("1", "true", "yes", "y", "on")


def _env_int(name: str, default: str) -> int:
    try:
        return int(os.getenv(name, default) or default)
    except Exception:
        return int(default)


def _default_ipo_free_days(market_code: Optional[str]) -> int:
    """
    Conservative defaults for markets with IPO-day / early-day special behavior.

    TW / CN / KR:
      user specifically wants early IPO days exempted from extreme filtering.
      We use 5 trading rows by default.

    TH:
      DO NOT default to 5.
      SET official materials indicate a special first trading day band,
      not a five-day no-limit rule.
    """
    mc = (market_code or "").strip().lower()

    if mc in {"tw", "cn", "kr"}:
        return 5
    if mc == "th":
        return 1
    return 0


def apply_ipo_guard(
    df: pd.DataFrame,
    *,
    market_code: Optional[str] = None,
    ticker: str = "",
) -> Tuple[pd.DataFrame, IPOGuardStats]:
    """
    Add an 'ipo_exempt' boolean column to mark the first N trading rows
    as exempt from extreme filtering.

    This does NOT remove rows. It only tags rows.

    Env:
      IPO_GUARD_ON=1
      IPO_GUARD_FREE_DAYS=5
      IPO_GUARD_TW_FREE_DAYS=5
      IPO_GUARD_CN_FREE_DAYS=5
      IPO_GUARD_KR_FREE_DAYS=5
      IPO_GUARD_TH_FREE_DAYS=1
    """
    st = IPOGuardStats()

    if df is None or df.empty:
        out = pd.DataFrame() if df is None else df.copy()
        if "ipo_exempt" not in out.columns:
            out["ipo_exempt"] = False
        return out, st

    out = df.copy()

    if "ipo_exempt" not in out.columns:
        out["ipo_exempt"] = False

    if not _env_bool("IPO_GUARD_ON", "1"):
        return out, st

    mc = (market_code or "").strip().lower()

    # market-specific override first
    per_market_name = f"IPO_GUARD_{mc.upper()}_FREE_DAYS" if mc else ""
    if per_market_name and os.getenv(per_market_name):
        free_days = _env_int(per_market_name, "0")
    elif os.getenv("IPO_GUARD_FREE_DAYS"):
        free_days = _env_int("IPO_GUARD_FREE_DAYS", "0")
    else:
        free_days = _default_ipo_free_days(mc)

    if free_days <= 0:
        return out, st

    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.tz_localize(None)
        out = out.sort_values("date").reset_index(drop=True)
    else:
        out = out.reset_index(drop=True)

    n = min(int(free_days), len(out))
    if n > 0:
        out.loc[out.index[:n], "ipo_exempt"] = True

    st.exempt_rows = n
    parts = []
    if mc:
        parts.append(f"market={mc}")
    if ticker:
        parts.append(f"ticker={ticker}")
    parts.append(f"free_days={free_days}")
    st.notes = ", ".join(parts)

    return out, st