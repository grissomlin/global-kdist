# markets/fr.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any, Dict, List, Tuple, Union

from core.fr_list import fetch_fr_list

UniverseRow = Union[str, Tuple[str, str], Dict[str, Any]]


def get_universe(cfg: Dict[str, Any]) -> List[UniverseRow]:
    """
    Return universe rows for FR.

    Preferred output:
      [("AIR.PA", "AIRBUS SE"), ...]
    """
    rows = fetch_fr_list(cfg=cfg)

    limit = int(cfg.get("limit", 0) or 0)
    if limit > 0:
        rows = rows[:limit]
    return rows


def to_ticker(row: UniverseRow) -> str:
    """
    Convert universe row -> yfinance ticker string.
    """
    if isinstance(row, (tuple, list)) and row:
        return str(row[0]).strip()
    if isinstance(row, dict):
        return str(row.get("symbol") or row.get("yf_symbol") or row.get("ticker") or row.get("id") or "").strip()
    return str(row).strip()