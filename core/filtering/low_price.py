# core/filtering/low_price.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import os

import pandas as pd


# =============================================================================
# Market-specific research thresholds
# These are for research / ranking / distribution filtering,
# NOT for base raw-data cleaning.
# =============================================================================
LOW_PRICE_THRESHOLDS: Dict[str, float] = {
    "us": 1.0,      # USD
    "ca": 1.0,      # CAD
    "uk": 1.0,      # GBP
    "au": 1.0,      # AUD
    "hk": 1.0,      # HKD
    "fr": 1.0,      # EUR
    "de": 1.0,      # EUR
    "eu": 1.0,      # EUR
    "jp": 100.0,    # JPY
    "kr": 1000.0,   # KRW
    "tw": 30.0,     # TWD
    "cn": 5.0,      # CNY
    "th": 1.0,      # THB
    "india": 10.0,  # INR
}


@dataclass
class LowPriceStats:
    threshold: float = 0.0
    low_price_rows: int = 0
    total_rows: int = 0
    low_price_ratio: float = 0.0
    dropped_rows: int = 0
    dropped_ticker: bool = False
    notes: str = ""


def _env_bool(name: str, default: str = "0") -> bool:
    return str(os.getenv(name, default)).strip().lower() in ("1", "true", "yes", "y", "on")


def _env_float(name: str, default: str) -> float:
    try:
        return float(os.getenv(name, default) or default)
    except Exception:
        return float(default)


def get_low_price_threshold(
    market_code: Optional[str] = None,
    *,
    custom_threshold: Optional[float] = None,
) -> Optional[float]:
    """
    Priority:
      1) custom_threshold
      2) env LOW_PRICE_THRESHOLD_<MARKET>
      3) built-in LOW_PRICE_THRESHOLDS
    """
    if custom_threshold is not None:
        try:
            return float(custom_threshold)
        except Exception:
            return None

    mc = (market_code or "").strip().lower()
    if not mc:
        return None

    env_key = f"LOW_PRICE_THRESHOLD_{mc.upper()}"
    env_val = os.getenv(env_key)
    if env_val not in (None, ""):
        try:
            return float(env_val)
        except Exception:
            pass

    return LOW_PRICE_THRESHOLDS.get(mc)


def calc_low_price_stats(
    df: pd.DataFrame,
    *,
    market_code: Optional[str] = None,
    ticker: str = "",
    custom_threshold: Optional[float] = None,
) -> LowPriceStats:
    st = LowPriceStats()

    if df is None or df.empty or "close" not in df.columns:
        st.notes = "empty df or missing close"
        return st

    thr = get_low_price_threshold(market_code, custom_threshold=custom_threshold)
    if thr is None:
        st.notes = "no threshold for market"
        return st

    close_num = pd.to_numeric(df["close"], errors="coerce")
    valid = close_num.notna()
    total = int(valid.sum())

    if total <= 0:
        st.threshold = float(thr)
        st.notes = "no valid close rows"
        return st

    low_mask = valid & (close_num < float(thr))
    low_n = int(low_mask.sum())
    ratio = float(low_n / total) if total > 0 else 0.0

    st.threshold = float(thr)
    st.low_price_rows = low_n
    st.total_rows = total
    st.low_price_ratio = ratio

    mc = (market_code or "").strip().lower()
    tk = (ticker or "").strip()
    parts = []
    if mc:
        parts.append(f"market={mc}")
    if tk:
        parts.append(f"ticker={tk}")
    parts.append(f"threshold={thr}")
    parts.append(f"low={low_n}/{total} ({ratio:.2%})")
    st.notes = " | ".join(parts)

    return st


def drop_low_price_rows(
    df: pd.DataFrame,
    *,
    market_code: Optional[str] = None,
    ticker: str = "",
    custom_threshold: Optional[float] = None,
    enabled: Optional[bool] = None,
) -> Tuple[pd.DataFrame, LowPriceStats]:
    """
    Drop rows where close < market threshold.

    Good for:
      - daily bigmove ranking
      - return distribution analysis
      - research output

    Less suitable for:
      - raw base OHLC archive
    """
    st = LowPriceStats()
    if df is None or df.empty:
        return df, st

    if enabled is None:
        enabled = _env_bool("LOW_PRICE_FILTER_ON", "0")
    if not enabled:
        st.notes = "LOW_PRICE_FILTER_ON=0"
        return df, st

    if "close" not in df.columns:
        st.notes = "missing close"
        return df, st

    thr = get_low_price_threshold(market_code, custom_threshold=custom_threshold)
    if thr is None:
        st.notes = "no threshold for market"
        return df, st

    out = df.copy()
    close_num = pd.to_numeric(out["close"], errors="coerce")
    valid = close_num.notna()

    low_mask = valid & (close_num < float(thr))
    total = int(valid.sum())
    low_n = int(low_mask.sum())
    ratio = float(low_n / total) if total > 0 else 0.0

    keep = (~valid) | (~low_mask)
    before = len(out)
    out = out.loc[keep].copy().reset_index(drop=True)

    st.threshold = float(thr)
    st.low_price_rows = low_n
    st.total_rows = total
    st.low_price_ratio = ratio
    st.dropped_rows = int(before - len(out))

    mc = (market_code or "").strip().lower()
    tk = (ticker or "").strip()
    st.notes = (
        f"market={mc} | ticker={tk} | drop rows close<{thr} | "
        f"low={low_n}/{total} ({ratio:.2%}) | dropped={st.dropped_rows}"
    ).strip(" |")

    return out, st


def should_drop_low_price_ticker(
    df: pd.DataFrame,
    *,
    market_code: Optional[str] = None,
    ticker: str = "",
    custom_threshold: Optional[float] = None,
    min_ratio: Optional[float] = None,
    mode: str = "ratio",
    enabled: Optional[bool] = None,
) -> Tuple[bool, LowPriceStats]:
    """
    Decide whether a ticker should be excluded from research universe.

    Modes:
      - ratio  : drop if (close < threshold) ratio >= min_ratio
      - median : drop if median(close) < threshold
      - both   : drop if either condition is met

    Env:
      LOW_PRICE_TICKER_FILTER_ON=0/1
      LOW_PRICE_TICKER_MIN_RATIO=0.5
    """
    st = LowPriceStats()
    if df is None or df.empty:
        st.notes = "empty df"
        return False, st

    if enabled is None:
        enabled = _env_bool("LOW_PRICE_TICKER_FILTER_ON", "0")
    if not enabled:
        st.notes = "LOW_PRICE_TICKER_FILTER_ON=0"
        return False, st

    if "close" not in df.columns:
        st.notes = "missing close"
        return False, st

    thr = get_low_price_threshold(market_code, custom_threshold=custom_threshold)
    if thr is None:
        st.notes = "no threshold for market"
        return False, st

    ratio_cut = float(min_ratio) if min_ratio is not None else _env_float("LOW_PRICE_TICKER_MIN_RATIO", "0.5")

    close_num = pd.to_numeric(df["close"], errors="coerce")
    valid = close_num.dropna()

    if valid.empty:
        st.threshold = float(thr)
        st.notes = "no valid close"
        return False, st

    low_mask = valid < float(thr)
    low_n = int(low_mask.sum())
    total = int(valid.shape[0])
    ratio = float(low_n / total) if total > 0 else 0.0
    median_close = float(valid.median())

    mode_norm = (mode or "ratio").strip().lower()
    by_ratio = ratio >= ratio_cut
    by_median = median_close < float(thr)

    if mode_norm == "median":
        drop = by_median
    elif mode_norm == "both":
        drop = by_ratio or by_median
    else:
        drop = by_ratio

    st.threshold = float(thr)
    st.low_price_rows = low_n
    st.total_rows = total
    st.low_price_ratio = ratio
    st.dropped_ticker = bool(drop)

    mc = (market_code or "").strip().lower()
    tk = (ticker or "").strip()
    st.notes = (
        f"market={mc} | ticker={tk} | mode={mode_norm} | "
        f"threshold={thr} | low={low_n}/{total} ({ratio:.2%}) | "
        f"median_close={median_close:.6f} | ratio_cut={ratio_cut:.2%} | "
        f"drop={drop}"
    ).strip(" |")

    return bool(drop), st


def filter_low_price_ticker_rows(
    df: pd.DataFrame,
    *,
    market_code: Optional[str] = None,
    ticker: str = "",
    custom_threshold: Optional[float] = None,
    min_ratio: Optional[float] = None,
    mode: str = "ratio",
    enabled: Optional[bool] = None,
) -> Tuple[pd.DataFrame, LowPriceStats]:
    """
    If ticker qualifies as low-price ticker by should_drop_low_price_ticker(),
    return empty df; otherwise return original df unchanged.

    Useful for:
      - per-ticker research pipeline
      - universe building before ranking
    """
    st = LowPriceStats()
    if df is None or df.empty:
        return df, st

    drop, st = should_drop_low_price_ticker(
        df,
        market_code=market_code,
        ticker=ticker,
        custom_threshold=custom_threshold,
        min_ratio=min_ratio,
        mode=mode,
        enabled=enabled,
    )
    if not drop:
        return df, st

    out = df.iloc[0:0].copy()
    return out, st