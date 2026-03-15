# markets/cn.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any, Dict, List, Tuple, Union

from core.cn_list import fetch_cn_a_share_list


UniverseRow = Union[str, Tuple[str, str], Dict[str, Any]]


def get_universe(cfg: Dict[str, Any]) -> List[UniverseRow]:
    """
    Return universe rows for CN (A-Share).

    Prefer returning (ticker, name) tuples:
      [("600519.SS","贵州茅台"), ("000001.SZ","平安银行"), ...]
    """
    include_bj = bool(cfg.get("include_bj", False))

    rows = fetch_cn_a_share_list(
        include_bj=include_bj,
    )

    limit = int(cfg.get("limit", 0) or 0)
    if limit > 0:
        rows = rows[:limit]
    return rows


def to_ticker(row: UniverseRow) -> str:
    """
    Convert universe row -> yfinance ticker string.

    - tuple/list: (ticker, name) -> ticker
    - dict: {"symbol": "..."} -> symbol
    - str: "600519.SS" -> as-is
    """
    if isinstance(row, (tuple, list)) and row:
        return str(row[0]).strip()
    if isinstance(row, dict):
        return str(row.get("symbol") or row.get("id") or row.get("ticker") or "").strip()
    return str(row).strip()