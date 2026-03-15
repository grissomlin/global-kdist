# markets/germany.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any, Dict, List, Tuple, Union

from core.germany_list import fetch_germany_list

UniverseRow = Union[str, Tuple[str, str], Dict[str, Any]]


def get_universe(cfg: Dict[str, Any]) -> List[UniverseRow]:
    """
    Return universe rows for GERMANY.

    Preferred output:
      [("SIE.DE", "Siemens AG"), ...]
    """
    master_xls_path = cfg.get("master_xls_path") or None

    rows = fetch_germany_list(
        master_xls_path=master_xls_path,
        cfg=cfg,
    )

    limit = int(cfg.get("limit", 0) or 0)
    if limit > 0:
        rows = rows[:limit]
    return rows


def to_ticker(row: UniverseRow) -> str:
    """
    Convert universe row -> yfinance ticker string.

    - tuple/list: (symbol, name) -> symbol
    - dict: {"symbol": "..."} -> symbol
    - str: "SIE.DE" -> as-is
    """
    if isinstance(row, (tuple, list)) and row:
        return str(row[0]).strip()
    if isinstance(row, dict):
        return str(row.get("symbol") or row.get("id") or row.get("ticker") or "").strip()
    return str(row).strip()