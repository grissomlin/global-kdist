# core/cleaning/ohlc.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Literal

import pandas as pd


NonPositivePolicy = Literal["all", "close"]


@dataclass
class OHLCCleanStats:
    rows_in: int = 0
    rows_out: int = 0

    dropped_bad_date: int = 0
    dropped_duplicates: int = 0
    dropped_nonpositive: int = 0
    dropped_bad_hilo: int = 0
    dropped_no_close: int = 0

    fixed_high_low_swap: int = 0
    fixed_cover: int = 0
    clipped_negative_volume: int = 0


def standardize_history(df: pd.DataFrame) -> pd.DataFrame:
    """Convert common yfinance outputs into: date, open, high, low, close, volume"""
    if df is None or df.empty:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])

    out = df.copy()

    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [str(c[0]) for c in out.columns]

    out = out.reset_index()

    date_col = None
    for cand in ["Date", "Datetime", "date", "datetime", "index", "Index"]:
        if cand in out.columns:
            date_col = cand
            break
    if date_col is None:
        for c in out.columns:
            if str(c).lower() in ("date", "datetime", "time"):
                date_col = c
                break
    if date_col is None:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])

    out["date"] = pd.to_datetime(out[date_col], errors="coerce").dt.tz_localize(None)

    rename_map = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "close",
        "Volume": "volume",
    }
    out = out.rename(columns=rename_map)

    # Ensure existence (case-insensitive)
    for c in ["open", "high", "low", "close", "volume"]:
        if c not in out.columns:
            for cc in out.columns:
                if str(cc).lower() == c:
                    out[c] = out[cc]
                    break

    for c in ("open", "high", "low", "close"):
        out[c] = pd.to_numeric(out.get(c), errors="coerce")
    out["volume"] = pd.to_numeric(out.get("volume"), errors="coerce")

    out = out[["date", "open", "high", "low", "close", "volume"]]
    out = out.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return out


def clean_ohlc(
    df: pd.DataFrame,
    *,
    nonpositive_policy: NonPositivePolicy = "all",
    fix_high_low: bool = True,
    allow_zero_volume: bool = True,
    drop_if_no_close: bool = True,
) -> Tuple[pd.DataFrame, OHLCCleanStats]:
    """
    Market-agnostic OHLC sanity cleaning:
      - normalize date, drop bad dates
      - sort by date and drop duplicate dates (keep last)
      - volume negative -> 0
      - optionally drop rows with no close
      - drop nonpositive prices (policy: all OHLC >0, or only close>0)
      - optional soft repair: high covers max(open,close), low covers min(open,close)
      - hard sanity: high>=low and covers O/C
    """
    st = OHLCCleanStats()
    if df is None or df.empty:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"]), st

    out = df.copy()
    st.rows_in = len(out)

    for c in ["date", "open", "high", "low", "close", "volume"]:
        if c not in out.columns:
            out[c] = pd.NA

    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.tz_localize(None)
    bad_date = int(out["date"].isna().sum())
    if bad_date:
        st.dropped_bad_date = bad_date
        out = out.dropna(subset=["date"])

    for c in ["open", "high", "low", "close", "volume"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    out = out.sort_values("date")
    before_dups = len(out)
    out = out.drop_duplicates(subset=["date"], keep="last")
    st.dropped_duplicates = int(before_dups - len(out))
    out = out.reset_index(drop=True)

    neg_vol = int((out["volume"].fillna(0) < 0).sum())
    if neg_vol:
        st.clipped_negative_volume = neg_vol
        out.loc[out["volume"] < 0, "volume"] = 0.0

    if drop_if_no_close:
        no_close = int(out["close"].isna().sum())
        if no_close:
            st.dropped_no_close = no_close
            out = out.dropna(subset=["close"])

    # drop nonpositive
    if nonpositive_policy == "all":
        mask = (out["open"] > 0) & (out["high"] > 0) & (out["low"] > 0) & (out["close"] > 0)
    else:  # "close"
        mask = (out["close"] > 0)
    dropped = int((~mask).sum())
    if dropped:
        st.dropped_nonpositive = dropped
        out = out.loc[mask].copy()

    if not allow_zero_volume:
        out = out.loc[out["volume"].fillna(0) > 0].copy()

    if out.empty:
        st.rows_out = 0
        return out.reset_index(drop=True), st

    if fix_high_low:
        mx = out[["open", "close", "high"]].max(axis=1)
        mn = out[["open", "close", "low"]].min(axis=1)

        need_high = out["high"] < mx
        if need_high.any():
            st.fixed_cover += int(need_high.sum())
            out.loc[need_high, "high"] = mx.loc[need_high]

        need_low = out["low"] > mn
        if need_low.any():
            st.fixed_cover += int(need_low.sum())
            out.loc[need_low, "low"] = mn.loc[need_low]

        need_swap = out["high"] < out["low"]
        if need_swap.any():
            st.fixed_high_low_swap += int(need_swap.sum())
            tmp = out.loc[need_swap, "high"].copy()
            out.loc[need_swap, "high"] = out.loc[need_swap, "low"]
            out.loc[need_swap, "low"] = tmp

    mx2 = out[["open", "close"]].max(axis=1)
    mn2 = out[["open", "close"]].min(axis=1)
    ok = (out["high"] >= out["low"]) & (out["high"] >= mx2) & (out["low"] <= mn2)

    bad = int((~ok).sum())
    if bad:
        st.dropped_bad_hilo = bad
        out = out.loc[ok].copy()

    out = out.sort_values("date").reset_index(drop=True)
    st.rows_out = len(out)
    return out, st