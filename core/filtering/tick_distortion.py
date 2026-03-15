# core/filtering/tick_distortion.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import math
import os

import numpy as np
import pandas as pd


# =============================================================================
# Config / defaults
# =============================================================================

@dataclass
class TickDistortionStats:
    threshold_pct: float = 0.0
    total_rows: int = 0
    distorted_rows: int = 0
    distorted_ratio: float = 0.0
    median_one_tick_pct: float = 0.0
    max_one_tick_pct: float = 0.0
    dropped_ticker: bool = False
    notes: str = ""


def _env_bool(name: str, default: str = "0") -> bool:
    return str(os.getenv(name, default)).strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_float(name: str, default: str) -> float:
    try:
        return float(os.getenv(name, default) or default)
    except Exception:
        return float(default)


# =============================================================================
# Tick tables
# NOTE:
# These are practical research approximations, not exchange-rule legal guarantees.
# Good enough for distortion screening / article research / histogram cleaning.
# =============================================================================

def _tick_table_us(price: float) -> float:
    # Simplified for common listed stocks.
    # Sub-dollar names often still 0.0001 or 0.0001~0.001 depending on venue,
    # but for research distortion we usually want a conservative broad brush.
    if price < 1.0:
        return 0.0001
    return 0.01


def _tick_table_ca(price: float) -> float:
    # Conservative research approximation
    if price < 0.10:
        return 0.005
    if price < 1.0:
        return 0.01
    return 0.01


def _tick_table_uk(price: float) -> float:
    # LSE usually quotes in pence for many names; yfinance often returns GBP-like price.
    # Keep a simplified approximation for research.
    if price < 0.10:
        return 0.0025
    if price < 0.25:
        return 0.005
    if price < 1.0:
        return 0.01
    return 0.01


def _tick_table_au(price: float) -> float:
    # ASX simplified
    if price < 0.10:
        return 0.001
    if price < 2.0:
        return 0.005
    return 0.01


def _tick_table_hk(price: float) -> float:
    # HKEX board-lot tick size table (simplified standard table)
    if price < 0.25:
        return 0.001
    if price < 0.50:
        return 0.005
    if price < 10.0:
        return 0.01
    if price < 20.0:
        return 0.02
    if price < 100.0:
        return 0.05
    if price < 200.0:
        return 0.10
    if price < 500.0:
        return 0.20
    if price < 1000.0:
        return 0.50
    if price < 2000.0:
        return 1.0
    if price < 5000.0:
        return 2.0
    return 5.0


def _tick_table_fr(price: float) -> float:
    # Euronext Paris simplified
    if price < 0.50:
        return 0.0001
    if price < 1.0:
        return 0.0005
    return 0.01


def _tick_table_de(price: float) -> float:
    # Deutsche Börse simplified
    if price < 0.50:
        return 0.0001
    if price < 1.0:
        return 0.0005
    return 0.01


def _tick_table_jp(price: float) -> float:
    # TSE tick table simplified enough for distortion research
    if price <= 3000:
        return 1.0
    if price <= 5000:
        return 5.0
    if price <= 30000:
        return 10.0
    if price <= 50000:
        return 50.0
    if price <= 300000:
        return 100.0
    if price <= 500000:
        return 500.0
    if price <= 3000000:
        return 1000.0
    if price <= 5000000:
        return 5000.0
    return 10000.0


def _tick_table_kr(price: float) -> float:
    # KRX simplified
    if price < 1000:
        return 1.0
    if price < 5000:
        return 5.0
    if price < 10000:
        return 10.0
    if price < 50000:
        return 50.0
    if price < 100000:
        return 100.0
    if price < 500000:
        return 500.0
    return 1000.0


def _tick_table_tw(price: float) -> float:
    # TWSE/TPEx simplified
    if price < 10:
        return 0.01
    if price < 50:
        return 0.05
    if price < 100:
        return 0.10
    if price < 500:
        return 0.50
    if price < 1000:
        return 1.0
    return 5.0


def _tick_table_cn(price: float) -> float:
    # SSE / SZSE A-share common minimum tick
    return 0.01


def _tick_table_th(price: float) -> float:
    # SET simplified
    if price < 2.0:
        return 0.01
    if price < 5.0:
        return 0.02
    if price < 10.0:
        return 0.05
    if price < 25.0:
        return 0.10
    if price < 100.0:
        return 0.25
    if price < 200.0:
        return 0.50
    if price < 400.0:
        return 1.0
    return 2.0


def _tick_table_india(price: float) -> float:
    # NSE simplified
    if price < 15.0:
        return 0.01
    return 0.05


TICK_TABLE_FUNCS = {
    "us": _tick_table_us,
    "ca": _tick_table_ca,
    "uk": _tick_table_uk,
    "au": _tick_table_au,
    "hk": _tick_table_hk,
    "fr": _tick_table_fr,
    "de": _tick_table_de,
    "jp": _tick_table_jp,
    "kr": _tick_table_kr,
    "tw": _tick_table_tw,
    "cn": _tick_table_cn,
    "th": _tick_table_th,
    "india": _tick_table_india,
}


# =============================================================================
# Core helpers
# =============================================================================

def get_tick_size(
    price: float,
    *,
    market_code: Optional[str] = None,
    custom_tick_size: Optional[float] = None,
) -> Optional[float]:
    """
    Return estimated minimum tick size for given price / market.
    Priority:
      1) custom_tick_size
      2) env TICK_SIZE_<MARKET>
      3) built-in simplified tick table
    """
    try:
        px = float(price)
    except Exception:
        return None

    if not math.isfinite(px) or px <= 0:
        return None

    if custom_tick_size is not None:
        try:
            v = float(custom_tick_size)
            return v if v > 0 else None
        except Exception:
            return None

    mc = (market_code or "").strip().lower()
    if mc:
        env_key = f"TICK_SIZE_{mc.upper()}"
        env_val = os.getenv(env_key)
        if env_val not in (None, ""):
            try:
                v = float(env_val)
                return v if v > 0 else None
            except Exception:
                pass

    fn = TICK_TABLE_FUNCS.get(mc)
    if fn is None:
        return None
    try:
        v = float(fn(px))
        return v if v > 0 else None
    except Exception:
        return None


def calc_one_tick_pct(
    price: float,
    *,
    market_code: Optional[str] = None,
    custom_tick_size: Optional[float] = None,
) -> Optional[float]:
    """
    one_tick_pct = tick_size / price * 100
    """
    try:
        px = float(price)
    except Exception:
        return None

    if not math.isfinite(px) or px <= 0:
        return None

    tick = get_tick_size(px, market_code=market_code, custom_tick_size=custom_tick_size)
    if tick is None or tick <= 0:
        return None

    return float(tick / px * 100.0)


def add_tick_distortion_columns(
    df: pd.DataFrame,
    *,
    market_code: Optional[str] = None,
    price_col: str = "close",
    custom_tick_size: Optional[float] = None,
) -> pd.DataFrame:
    """
    Add:
      - tick_size
      - one_tick_pct
    """
    out = df.copy()
    if out.empty or price_col not in out.columns:
        if "tick_size" not in out.columns:
            out["tick_size"] = np.nan
        if "one_tick_pct" not in out.columns:
            out["one_tick_pct"] = np.nan
        return out

    px = pd.to_numeric(out[price_col], errors="coerce")

    tick_vals: List[Optional[float]] = []
    one_tick_vals: List[Optional[float]] = []

    for v in px.tolist():
        tick = get_tick_size(v, market_code=market_code, custom_tick_size=custom_tick_size)
        one = calc_one_tick_pct(v, market_code=market_code, custom_tick_size=custom_tick_size)
        tick_vals.append(tick)
        one_tick_vals.append(one)

    out["tick_size"] = tick_vals
    out["one_tick_pct"] = one_tick_vals
    return out


def calc_tick_distortion_stats(
    df: pd.DataFrame,
    *,
    market_code: Optional[str] = None,
    ticker: str = "",
    price_col: str = "close",
    threshold_pct: float = 10.0,
    custom_tick_size: Optional[float] = None,
) -> TickDistortionStats:
    st = TickDistortionStats()
    st.threshold_pct = float(threshold_pct)

    if df is None or df.empty or price_col not in df.columns:
        st.notes = "empty df or missing price_col"
        return st

    tmp = add_tick_distortion_columns(
        df,
        market_code=market_code,
        price_col=price_col,
        custom_tick_size=custom_tick_size,
    )

    one = pd.to_numeric(tmp["one_tick_pct"], errors="coerce").dropna()
    total = int(one.shape[0])

    if total <= 0:
        st.notes = "no valid one_tick_pct"
        return st

    distorted = one > float(threshold_pct)
    distorted_n = int(distorted.sum())
    distorted_ratio = float(distorted_n / total) if total > 0 else 0.0

    st.total_rows = total
    st.distorted_rows = distorted_n
    st.distorted_ratio = distorted_ratio
    st.median_one_tick_pct = float(one.median())
    st.max_one_tick_pct = float(one.max())

    mc = (market_code or "").strip().lower()
    tk = (ticker or "").strip()
    st.notes = (
        f"market={mc} | ticker={tk} | threshold_pct={threshold_pct:.2f} | "
        f"distorted={distorted_n}/{total} ({distorted_ratio:.2%}) | "
        f"median_one_tick_pct={st.median_one_tick_pct:.4f} | "
        f"max_one_tick_pct={st.max_one_tick_pct:.4f}"
    ).strip(" |")

    return st


def should_drop_tick_distortion_ticker(
    df: pd.DataFrame,
    *,
    market_code: Optional[str] = None,
    ticker: str = "",
    price_col: str = "close",
    threshold_pct: float = 10.0,
    min_ratio: Optional[float] = None,
    mode: str = "ratio",
    custom_tick_size: Optional[float] = None,
    enabled: Optional[bool] = None,
) -> Tuple[bool, TickDistortionStats]:
    """
    Decide whether a ticker should be excluded due to excessive tick distortion.

    Modes:
      - ratio  : drop if ratio(one_tick_pct > threshold_pct) >= min_ratio
      - median : drop if median(one_tick_pct) > threshold_pct
      - both   : drop if either condition is met

    Env:
      TICK_DISTORTION_FILTER_ON=0/1
      TICK_DISTORTION_MIN_RATIO=0.5
      TICK_DISTORTION_THRESHOLD_PCT=10
    """
    st = TickDistortionStats()

    if df is None or df.empty or price_col not in df.columns:
        st.notes = "empty df or missing price_col"
        return False, st

    if enabled is None:
        enabled = _env_bool("TICK_DISTORTION_FILTER_ON", "0")
    if not enabled:
        st.notes = "TICK_DISTORTION_FILTER_ON=0"
        return False, st

    thr = float(threshold_pct)
    thr = float(os.getenv("TICK_DISTORTION_THRESHOLD_PCT", thr) or thr)

    ratio_cut = float(min_ratio) if min_ratio is not None else _env_float("TICK_DISTORTION_MIN_RATIO", "0.5")

    tmp = add_tick_distortion_columns(
        df,
        market_code=market_code,
        price_col=price_col,
        custom_tick_size=custom_tick_size,
    )

    one = pd.to_numeric(tmp["one_tick_pct"], errors="coerce").dropna()
    total = int(one.shape[0])

    st.threshold_pct = thr

    if total <= 0:
        st.notes = "no valid one_tick_pct"
        return False, st

    distorted = one > thr
    distorted_n = int(distorted.sum())
    distorted_ratio = float(distorted_n / total) if total > 0 else 0.0
    median_one = float(one.median())
    max_one = float(one.max())

    mode_norm = (mode or "ratio").strip().lower()
    by_ratio = distorted_ratio >= ratio_cut
    by_median = median_one > thr

    if mode_norm == "median":
        drop = by_median
    elif mode_norm == "both":
        drop = by_ratio or by_median
    else:
        drop = by_ratio

    st.total_rows = total
    st.distorted_rows = distorted_n
    st.distorted_ratio = distorted_ratio
    st.median_one_tick_pct = median_one
    st.max_one_tick_pct = max_one
    st.dropped_ticker = bool(drop)

    mc = (market_code or "").strip().lower()
    tk = (ticker or "").strip()
    st.notes = (
        f"market={mc} | ticker={tk} | mode={mode_norm} | "
        f"threshold_pct={thr:.2f} | distorted={distorted_n}/{total} ({distorted_ratio:.2%}) | "
        f"median_one_tick_pct={median_one:.4f} | max_one_tick_pct={max_one:.4f} | "
        f"ratio_cut={ratio_cut:.2%} | drop={drop}"
    ).strip(" |")

    return bool(drop), st


def filter_tick_distortion_ticker_rows(
    df: pd.DataFrame,
    *,
    market_code: Optional[str] = None,
    ticker: str = "",
    price_col: str = "close",
    threshold_pct: float = 10.0,
    min_ratio: Optional[float] = None,
    mode: str = "ratio",
    custom_tick_size: Optional[float] = None,
    enabled: Optional[bool] = None,
) -> Tuple[pd.DataFrame, TickDistortionStats]:
    """
    If ticker qualifies as distorted ticker, return empty df; else return unchanged df.
    """
    st = TickDistortionStats()
    if df is None or df.empty:
        return df, st

    drop, st = should_drop_tick_distortion_ticker(
        df,
        market_code=market_code,
        ticker=ticker,
        price_col=price_col,
        threshold_pct=threshold_pct,
        min_ratio=min_ratio,
        mode=mode,
        custom_tick_size=custom_tick_size,
        enabled=enabled,
    )
    if not drop:
        return df, st

    return df.iloc[0:0].copy(), st


__all__ = [
    "TickDistortionStats",
    "get_tick_size",
    "calc_one_tick_pct",
    "add_tick_distortion_columns",
    "calc_tick_distortion_stats",
    "should_drop_tick_distortion_ticker",
    "filter_tick_distortion_ticker_rows",
]