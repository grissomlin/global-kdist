# markets/us.py
# -*- coding: utf-8 -*-
"""
US market module for dayK downloader.

Required by runner/downloader:
- get_universe(cfg) -> list[UniverseRow]
- to_ticker(row) -> str
"""

from __future__ import annotations

from typing import Any, Dict, List, Union

import pandas as pd

from core.us_list import build_us_universe

UniverseRow = Union[str, Dict[str, Any], pd.Series, tuple, list]


def get_universe(cfg: Dict[str, Any]) -> List[UniverseRow]:
    """
    Return universe rows for US.

    Normalized output:
      [{"symbol": "AAPL", "name": "Apple Inc.", ...}, ...]
    """
    df = build_us_universe(cfg)

    if df is None or len(df) == 0:
        return []

    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"[US] build_us_universe must return DataFrame, got {type(df)}")

    # Ensure symbol column exists
    if "symbol" not in df.columns:
        raise RuntimeError(f"[US] universe dataframe missing required column: 'symbol'. got={list(df.columns)}")

    # Basic clean
    df = df.copy()
    df["symbol"] = df["symbol"].astype(str).str.strip()
    df = df[df["symbol"].str.len() > 0].copy()

    # Convert to list[dict] so the shared runner won't iterate dataframe columns by mistake
    return df.to_dict(orient="records")


def to_ticker(row: UniverseRow) -> str:
    """
    Convert universe row -> yfinance ticker string.

    Supported inputs:
    - dict: {"symbol": "..."}
    - pandas Series
    - tuple/list: first element is symbol
    - str: already a ticker
    """
    if isinstance(row, dict):
        return str(row.get("symbol") or row.get("ticker") or row.get("id") or "").strip()

    if isinstance(row, pd.Series):
        return str(row.get("symbol") or row.get("ticker") or row.get("id") or "").strip()

    if isinstance(row, (tuple, list)) and row:
        return str(row[0]).strip()

    return str(row).strip()