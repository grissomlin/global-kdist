# markets/jp.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any, Dict, List, Tuple, Union

from core.jpx_list import fetch_jpx_list


UniverseRow = Union[str, Tuple[str, str], Dict[str, Any]]


def get_universe(cfg: Dict[str, Any]) -> List[UniverseRow]:
    """
    Return universe rows for JP.

    Prefer returning (symbol, name) tuples:
      [("7203.T", "TOYOTA MOTOR CORP"), ...]
    """
    include_tokyo_pro = bool(cfg.get("include_tokyo_pro", False))
    list_url = cfg.get("list_url") or None

    rows = fetch_jpx_list(
        list_url=list_url,
        include_tokyo_pro=include_tokyo_pro,
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
    - str: "7203.T" -> as-is
    """
    if isinstance(row, (tuple, list)) and row:
        return str(row[0]).strip()
    if isinstance(row, dict):
        return str(row.get("symbol") or row.get("id") or row.get("ticker") or "").strip()
    return str(row).strip()