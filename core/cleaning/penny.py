# core/cleaning/penny.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import os

import pandas as pd


@dataclass
class PennyStats:
    dropped_rows: int = 0
    notes: str = ""


def _env_bool(name: str, default: str = "0") -> bool:
    return str(os.getenv(name, default)).strip().lower() in ("1", "true", "yes", "y", "on")


def _env_float(name: str, default: str) -> float:
    try:
        return float(os.getenv(name, default) or default)
    except Exception:
        return float(default)


def apply_penny_rules(
    df: pd.DataFrame,
    *,
    market_code: Optional[str] = None,
    ticker: str = "",
) -> Tuple[pd.DataFrame, PennyStats]:
    """
    Garbage-value cleaning only (NOT research low-price filtering).

    Purpose:
      - remove obviously bad micro-price rows caused by bad source / parsing / ghost data
      - keep this conservative
      - do NOT apply market-specific "low-price stock" research rules here

    Controlled by:
      PENNY_CLEAN_ON=1
      PENNY_GARBAGE_MIN_CLOSE=1e-8
      PENNY_DROP_NONPOSITIVE=1
      PENNY_REQUIRE_OHLC_GE_MIN=0

    Notes:
      - This function should remain a light cleaning pass.
      - Market-specific low-price filtering belongs in core/filtering/low_price.py
    """
    st = PennyStats()
    if df is None or df.empty:
        return df, st

    if not _env_bool("PENNY_CLEAN_ON", "0"):
        return df, st

    out = df.copy()

    min_close = _env_float("PENNY_GARBAGE_MIN_CLOSE", "1e-8")
    drop_nonpositive = _env_bool("PENNY_DROP_NONPOSITIVE", "1")
    require_ohlc_ge_min = _env_bool("PENNY_REQUIRE_OHLC_GE_MIN", "0")

    notes = []
    keep = pd.Series(True, index=out.index)

    # -------------------------------------------------------------------------
    # 1) Non-positive close is always suspicious for normal OHLC daily bars
    # -------------------------------------------------------------------------
    if "close" in out.columns:
        close_num = pd.to_numeric(out["close"], errors="coerce")

        if drop_nonpositive:
            m = close_num > 0
            dropped = int((~m).sum())
            if dropped:
                keep &= m
                notes.append(f"drop close<=0: {dropped}")

        # absurdly tiny positive prices: usually bad source artifacts
        m2 = close_num.fillna(0) >= float(min_close)
        dropped2 = int((~m2).sum())
        if dropped2:
            keep &= m2
            notes.append(f"drop close<{min_close}: {dropped2}")

    # -------------------------------------------------------------------------
    # 2) Optionally require OHLC to also be >= min_close
    #    keep OFF by default because some sources may have sparse fields
    # -------------------------------------------------------------------------
    if require_ohlc_ge_min:
        for col in ("open", "high", "low", "close"):
            if col in out.columns:
                x = pd.to_numeric(out[col], errors="coerce")
                # only check non-null values
                m = x.isna() | (x >= float(min_close))
                dropped = int((~m).sum())
                if dropped:
                    keep &= m
                    notes.append(f"drop {col}<{min_close}: {dropped}")

    before = len(out)
    out = out.loc[keep].copy().reset_index(drop=True)
    st.dropped_rows = int(before - len(out))

    mc = (market_code or "").strip().lower()
    tk = (ticker or "").strip()
    head = []
    if mc:
        head.append(f"market={mc}")
    if tk:
        head.append(f"ticker={tk}")

    prefix = ", ".join(head)
    body = "; ".join(notes)
    st.notes = f"{prefix} | {body}".strip(" |") if (prefix or body) else ""

    return out, st