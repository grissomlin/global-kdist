# markets/th.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any, Dict, List, Tuple, Union

from core.th_list import fetch_th_list

UniverseRow = Union[str, Tuple[str, str], Dict[str, Any]]


def get_universe(cfg: Dict[str, Any]) -> List[UniverseRow]:
    """
    Return universe rows for TH.

    Prefer returning (ticker, name) tuples:
      [("CPALL.BK", "CP ALL ..."), ...]
    """
    rows = fetch_th_list()

    limit = int(cfg.get("limit", 0) or 0)
    if limit > 0:
        rows = rows[:limit]
    return rows


def to_ticker(row: UniverseRow) -> str:
    """
    Convert universe row -> yfinance ticker string.

    - tuple/list: (ticker, name) -> ticker
    - dict: {"symbol": "..."} -> symbol
    - str: "CPALL.BK" -> as-is
    """
    if isinstance(row, (tuple, list)) and row:
        return str(row[0]).strip()
    if isinstance(row, dict):
        return str(row.get("symbol") or row.get("id") or row.get("ticker") or "").strip()
    return str(row).strip()