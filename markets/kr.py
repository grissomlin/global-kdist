# markets/kr.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any, Dict, List, Tuple, Union

from core.krx_list import fetch_krx_list

UniverseRow = Union[str, Tuple[str, str], Dict[str, Any]]


def get_universe(cfg: Dict[str, Any]) -> List[UniverseRow]:
    """
    Return universe rows for KR.

    Prefer returning (symbol, name) tuples:
      [("005930.KS", "삼성전자"), ...]
    """
    list_url = cfg.get("list_url") or None

    rows = fetch_krx_list(
        list_url=list_url,
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
    - str: "005930.KS" -> as-is
    """
    if isinstance(row, (tuple, list)) and row:
        return str(row[0]).strip()
    if isinstance(row, dict):
        return str(row.get("symbol") or row.get("id") or row.get("ticker") or "").strip()
    return str(row).strip()