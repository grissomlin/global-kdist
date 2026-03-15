# core/cleaning/resume_ghost.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import os

import numpy as np
import pandas as pd


@dataclass
class GhostStats:
    removed_rows: int = 0
    notes: str = ""


def _env_bool(name: str, default: str = "0") -> bool:
    return str(os.getenv(name, default)).strip().lower() in ("1", "true", "yes", "y", "on")


def apply_resume_ghost(
    df: pd.DataFrame,
    *,
    market_code: Optional[str] = None,
    ticker: str = "",
) -> Tuple[pd.DataFrame, GhostStats]:
    """
    Optional: remove "ghost" rows (0 volume + OHLC all equal) often seen in some sources.
    Default off. Enable with GHOST_FILTER_ON=1.

    If you want the full TW-style resume flag logic later, we extend here.
    """
    st = GhostStats()
    if df is None or df.empty:
        return df, st

    if not _env_bool("GHOST_FILTER_ON", "0"):
        return df, st

    out = df.copy().sort_values("date").reset_index(drop=True)

    eps = float(os.getenv("GHOST_EPS", "1e-8") or "1e-8")
    vol0 = out["volume"].fillna(0) == 0
    o = out["open"].to_numpy(dtype="float64", copy=True)
    h = out["high"].to_numpy(dtype="float64", copy=True)
    l = out["low"].to_numpy(dtype="float64", copy=True)
    c = out["close"].to_numpy(dtype="float64", copy=True)

    finite = np.isfinite(o) & np.isfinite(h) & np.isfinite(l) & np.isfinite(c)
    mx = np.maximum.reduce([o, h, l, c])
    mn = np.minimum.reduce([o, h, l, c])
    equal4 = finite & ((mx - mn) <= eps)

    ghost = vol0.to_numpy() & equal4
    removed = int(ghost.sum())
    if removed:
        st.removed_rows = removed
        out = out.loc[~ghost].copy().reset_index(drop=True)

    return out, st