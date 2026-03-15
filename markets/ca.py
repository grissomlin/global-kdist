# markets/ca.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any, Dict, List, Tuple, Union

from core.ca_list import get_ca_universe

UniverseRow = Union[str, Tuple[str, str], Dict[str, Any]]


def get_universe(cfg: Dict[str, Any]) -> List[UniverseRow]:
    """
    Return universe rows for CA (Canada).

    Preferred output:
      [("BNS.TO", "Bank of Nova Scotia"), ("DML.V", "Denison Mines"), ...]
    """
    limit = int(cfg.get("limit", 0) or 0)
    rows = get_ca_universe(limit=limit, cfg=cfg)
    return rows


def to_ticker(row: UniverseRow) -> str:
    """
    Convert universe row -> yfinance ticker string.

    - tuple/list: (symbol, name) -> symbol
    - dict: {"symbol": "..."} -> symbol
    - str: "BNS.TO" -> as-is
    """
    if isinstance(row, (tuple, list)) and row:
        return str(row[0]).strip()

    if isinstance(row, dict):
        return str(
            row.get("symbol")
            or row.get("yf_ticker")
            or row.get("id")
            or row.get("ticker")
            or ""
        ).strip()

    return str(row).strip()