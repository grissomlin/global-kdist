# core/cleaning/extremes.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import os

import numpy as np
import pandas as pd


@dataclass
class ExtremeStats:
    dropped_rows: int = 0
    reason: str = ""


def _env_float(name: str, default: float) -> float:
    v = (os.getenv(name) or "").strip()
    if not v:
        return float(default)
    try:
        return float(v)
    except Exception:
        return float(default)


def _env_bool(name: str, default: bool = False) -> bool:
    v = (os.getenv(name) or "").strip().lower()
    if not v:
        return bool(default)
    return v in {"1", "true", "yes", "y", "on"}


def apply_extreme_filters(
    df: pd.DataFrame,
    *,
    market_code: Optional[str] = None,
    ticker: str = "",
) -> Tuple[pd.DataFrame, ExtremeStats]:
    """
    Market-agnostic extreme move filter.
    Default: ENABLED unless EXTREME_FILTER_ON=0.

    Goals:
      1) remove obvious one-day scale errors
      2) remove clear reverse-split-like artifacts
      3) remove intraday high spikes that pollute period max(high)
      4) avoid over-deleting legitimate IPO early-day moves by respecting ipo_exempt

    Expected optional upstream column:
      - ipo_exempt : bool
        first N trading rows that should be exempt from this filter

    Base filter:
      - drop row if |close / prev_close - 1| > EXTREME_RET_THRESHOLD
      - drop row if |open  / prev_close - 1| > EXTREME_RET_THRESHOLD

    Additional filters:
      - scale spike:
          prev normal -> current exploded -> next collapses back
      - reverse split like:
          prev very different -> current jumps a lot -> next remains near new level
      - high spike:
          intraday high explodes vs prev close,
          but close does NOT confirm,
          and next close also returns / stays near normal range

    Env:
      EXTREME_FILTER_ON=1
      EXTREME_RET_THRESHOLD=20.0

      EXTREME_SCALE_SPIKE_ON=1
      EXTREME_SCALE_SPIKE_UP=20
      EXTREME_SCALE_SPIKE_BACK=0.2
      EXTREME_SCALE_SPIKE_REQUIRE_NEXT=1

      EXTREME_REVERSE_SPLIT_ON=1
      EXTREME_REVERSE_SPLIT_UP=5
      EXTREME_REVERSE_SPLIT_BACK_MIN=0.5
      EXTREME_REVERSE_SPLIT_BACK_MAX=1.5
      EXTREME_REVERSE_SPLIT_REQUIRE_NEXT=1

      EXTREME_HIGH_SPIKE_ON=1
      EXTREME_HIGH_SPIKE_UP=5
      EXTREME_HIGH_SPIKE_CLOSE_MAX=2
      EXTREME_HIGH_SPIKE_NEXT_MAX=2
      EXTREME_HIGH_SPIKE_REQUIRE_NEXT=1
    """
    st = ExtremeStats()

    if df is None or df.empty or len(df) < 2:
        return df, st

    required = {"date", "open", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")

    # 預設直接打開；只有明確設 EXTREME_FILTER_ON=0 才關閉
    on = _env_bool("EXTREME_FILTER_ON", True)
    if not on:
        return df, st

    # 基礎 extreme threshold
    thr = _env_float("EXTREME_RET_THRESHOLD", 20.0)

    # scale spike detection
    scale_on = _env_bool("EXTREME_SCALE_SPIKE_ON", True)
    scale_up = _env_float("EXTREME_SCALE_SPIKE_UP", 20.0)
    scale_back = _env_float("EXTREME_SCALE_SPIKE_BACK", 0.2)
    scale_require_next = _env_bool("EXTREME_SCALE_SPIKE_REQUIRE_NEXT", True)

    # reverse-split-like detection
    rs_on = _env_bool("EXTREME_REVERSE_SPLIT_ON", True)
    rs_up = _env_float("EXTREME_REVERSE_SPLIT_UP", 5.0)
    rs_back_min = _env_float("EXTREME_REVERSE_SPLIT_BACK_MIN", 0.5)
    rs_back_max = _env_float("EXTREME_REVERSE_SPLIT_BACK_MAX", 1.5)
    rs_require_next = _env_bool("EXTREME_REVERSE_SPLIT_REQUIRE_NEXT", True)

    # intraday high-spike detection
    high_spike_on = _env_bool("EXTREME_HIGH_SPIKE_ON", True)
    high_spike_up = _env_float("EXTREME_HIGH_SPIKE_UP", 5.0)
    high_spike_close_max = _env_float("EXTREME_HIGH_SPIKE_CLOSE_MAX", 2.0)
    high_spike_next_max = _env_float("EXTREME_HIGH_SPIKE_NEXT_MAX", 2.0)
    high_spike_require_next = _env_bool("EXTREME_HIGH_SPIKE_REQUIRE_NEXT", True)

    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.tz_localize(None)
    out["open"] = pd.to_numeric(out["open"], errors="coerce")
    out["close"] = pd.to_numeric(out["close"], errors="coerce")

    has_high = "high" in out.columns
    if has_high:
        out["high"] = pd.to_numeric(out["high"], errors="coerce")

    out = out.sort_values("date").reset_index(drop=True)

    prev_c = out["close"].shift(1).replace(0, np.nan)
    next_c = out["close"].shift(-1).replace(0, np.nan)

    ret_close = (out["close"] / prev_c) - 1.0
    ret_gap = (out["open"] / prev_c) - 1.0

    # base extreme move filter
    bad_ret_close = ret_close.abs() > thr
    bad_ret_gap = ret_gap.abs() > thr

    # scale spike detection: close explodes vs prev, then next day collapses back
    scale_spike = pd.Series(False, index=out.index)
    if scale_on:
        up_vs_prev = out["close"] / prev_c
        back_vs_curr = next_c / out["close"]

        if scale_require_next:
            scale_spike = (
                prev_c.notna()
                & next_c.notna()
                & (up_vs_prev >= scale_up)
                & (back_vs_curr <= scale_back)
            )
        else:
            scale_spike = (
                prev_c.notna()
                & (up_vs_prev >= scale_up)
            )

        scale_spike = scale_spike.fillna(False)

    # reverse split like: current jumps a lot vs prev, next day remains near new level
    reverse_split_like = pd.Series(False, index=out.index)
    if rs_on:
        up_vs_prev = out["close"] / prev_c
        back_vs_curr = next_c / out["close"]

        if rs_require_next:
            reverse_split_like = (
                prev_c.notna()
                & next_c.notna()
                & (up_vs_prev >= rs_up)
                & (back_vs_curr >= rs_back_min)
                & (back_vs_curr <= rs_back_max)
            )
        else:
            reverse_split_like = (
                prev_c.notna()
                & (up_vs_prev >= rs_up)
            )

        reverse_split_like = reverse_split_like.fillna(False)

    # high spike:
    # intraday high is absurdly large vs prev close,
    # but close does not confirm the move,
    # and optionally next close also stays near normal range.
    high_spike = pd.Series(False, index=out.index)
    if high_spike_on and has_high:
        high_vs_prev = out["high"] / prev_c
        close_vs_prev = out["close"] / prev_c
        next_vs_prev = next_c / prev_c

        if high_spike_require_next:
            high_spike = (
                prev_c.notna()
                & next_c.notna()
                & out["high"].notna()
                & (high_vs_prev >= high_spike_up)
                & (close_vs_prev <= high_spike_close_max)
                & (next_vs_prev <= high_spike_next_max)
            )
        else:
            high_spike = (
                prev_c.notna()
                & out["high"].notna()
                & (high_vs_prev >= high_spike_up)
                & (close_vs_prev <= high_spike_close_max)
            )

        high_spike = high_spike.fillna(False)

    # IPO guard: exempt early rows if upstream already marked them
    ipo_exempt = pd.Series(False, index=out.index)
    if "ipo_exempt" in out.columns:
        ipo_exempt = out["ipo_exempt"].fillna(False).astype(bool)

    # suspicious if one of the detectors fires
    bad = (
        bad_ret_close
        | bad_ret_gap
        | scale_spike
        | reverse_split_like
        | high_spike
    )
    bad = bad.fillna(False)

    # do not drop exempt IPO rows
    bad = bad & (~ipo_exempt)

    ok = ~bad
    ok = ok.fillna(True)

    dropped = int((~ok).sum())
    if dropped > 0:
        reasons = []

        if bool((bad_ret_close.fillna(False) | bad_ret_gap.fillna(False)).any()):
            reasons.append(f"abs(ret_close|ret_gap)>{thr}")

        if bool(scale_spike.any()):
            reasons.append(
                f"scale_spike(up>={scale_up}x"
                + (f", next_back<={scale_back}" if scale_require_next else "")
                + ")"
            )

        if bool(reverse_split_like.any()):
            reasons.append(
                f"reverse_split_like(up>={rs_up}x"
                + (
                    f", next_ratio in [{rs_back_min}, {rs_back_max}]"
                    if rs_require_next else ""
                )
                + ")"
            )

        if bool(high_spike.any()):
            reasons.append(
                f"high_spike(high/prev>={high_spike_up}x, close/prev<={high_spike_close_max}x"
                + (
                    f", next_close/prev<={high_spike_next_max}x"
                    if high_spike_require_next else ""
                )
                + ")"
            )

        if bool(ipo_exempt.any()):
            reasons.append("ipo_exempt respected")

        st.dropped_rows = dropped
        st.reason = "; ".join(reasons)

        out = out.loc[ok].copy()

    out = out.reset_index(drop=True)
    return out, st