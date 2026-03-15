# markets/uk.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any, Dict, List, Tuple, Union

from core.uk_list import get_uk_universe


UniverseRow = Union[str, Tuple[str, str], Dict[str, Any]]


def get_universe(cfg: Dict[str, Any]) -> List[UniverseRow]:
    """
    Return universe rows for UK.

    Prefer returning (symbol, name) tuples:
      [("ALF.L", "Alpha plc"), ...]
    """
    limit = int(cfg.get("limit", 0) or 0)
    rows = get_uk_universe(limit=limit)
    return rows


def to_ticker(row: UniverseRow) -> str:
    """
    Convert universe row -> yfinance ticker string.

    - tuple/list: (symbol, name) -> symbol
    - dict: {"symbol": "..."} -> symbol
    - str: "ALF.L" -> as-is
    """
    if isinstance(row, (tuple, list)) and row:
        return str(row[0]).strip()
    if isinstance(row, dict):
        return str(row.get("symbol") or row.get("id") or row.get("ticker") or "").strip()
    return str(row).strip()