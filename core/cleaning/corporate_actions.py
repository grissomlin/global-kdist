# core/cleaning/corporate_actions.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import os

import pandas as pd


@dataclass
class CorporateActionStats:
    notes: str = ""


def apply_corporate_action_fixes(
    df: pd.DataFrame,
    *,
    market_code: Optional[str] = None,
    ticker: str = "",
) -> Tuple[pd.DataFrame, CorporateActionStats]:
    """
    Placeholder for corporate-action related fixes (splits/dividends anomalies).

    For now: no-op.
    Later: you can detect split days and avoid treating them as "bad OHLC".
    """
    st = CorporateActionStats()
    return df, st