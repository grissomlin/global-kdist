# markets/tw.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any, Dict, List, Tuple, Union

from core.tw_list import get_tw_universe

UniverseRow = Union[str, Tuple[str, str], Dict[str, Any]]


def get_universe(cfg: Dict[str, Any]) -> List[UniverseRow]:
    """
    Return universe rows for TW.
    Prefer: [(symbol, name), ...]
    symbol format: 2330.TW / 6488.TWO / ...
    """
    limit = int(cfg.get("limit", 0) or 0)
    return get_tw_universe(limit=limit, cfg=cfg)


def to_ticker(row: UniverseRow) -> str:
    if isinstance(row, (tuple, list)) and row:
        return str(row[0]).strip()
    if isinstance(row, dict):
        return str(row.get("symbol") or row.get("ticker") or "").strip()
    return str(row).strip()