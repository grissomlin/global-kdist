# core/tw_list.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import os
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests


DEFAULT_JSON_PATH = "data/tw_stock_list.json"

TW_URL_CONFIGS = [
    # TWSE listed common stocks
    {
        "name": "listed",
        "url": "https://isin.twse.com.tw/isin/class_main.jsp?market=1&issuetype=1&Page=1&chklike=Y",
        "suffix": ".TW",
    },
    # Taiwan Depositary Receipt
    {
        "name": "dr",
        "url": "https://isin.twse.com.tw/isin/class_main.jsp?owncode=&stockname=&isincode=&market=1&issuetype=J&industry_code=&Page=1&chklike=Y",
        "suffix": ".TW",
    },
    # OTC
    {
        "name": "otc",
        "url": "https://isin.twse.com.tw/isin/class_main.jsp?market=2&issuetype=4&Page=1&chklike=Y",
        "suffix": ".TWO",
    },
    # Emerging / ROTC
    {
        "name": "rotc",
        "url": "https://isin.twse.com.tw/isin/class_main.jsp?owncode=&stockname=&isincode=&market=E&issuetype=R&industry_code=&Page=1&chklike=Y",
        "suffix": ".TWO",
    },
    # TW innovation board
    {
        "name": "tw_innovation",
        "url": "https://isin.twse.com.tw/isin/class_main.jsp?owncode=&stockname=&isincode=&market=C&issuetype=C&industry_code=&Page=1&chklike=Y",
        "suffix": ".TW",
    },
    # OTC innovation board
    {
        "name": "otc_innovation",
        "url": "https://isin.twse.com.tw/isin/class_main.jsp?owncode=&stockname=&isincode=&market=A&issuetype=C&industry_code=&Page=1&chklike=Y",
        "suffix": ".TWO",
    },
]

DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept-Language": "zh-TW,zh;q=0.9,en-US;q=0.8,en;q=0.7",
    "Referer": "https://isin.twse.com.tw/",
}


def _cfg_str(cfg: Optional[Dict[str, Any]], key: str, default: str = "") -> str:
    if cfg is None:
        return default
    v = cfg.get(key, default)
    if v is None:
        return default
    s = str(v).strip()
    return s if s else default


def _cfg_bool(cfg: Optional[Dict[str, Any]], key: str, default: bool) -> bool:
    if cfg is None:
        return default
    v = cfg.get(key, default)
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _cfg_int(cfg: Optional[Dict[str, Any]], key: str, default: int) -> int:
    if cfg is None:
        return default
    try:
        return int(cfg.get(key, default))
    except Exception:
        return default


def _safe_str(x: Any) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return str(x).strip()


def _load_local_json(path_str: str, limit: int = 0) -> List[Tuple[str, str]]:
    p = Path(path_str)
    if not p.exists():
        raise FileNotFoundError(
            f"TW list not found: {p}\n"
            f"Put data/tw_stock_list.json or set TW_LIST_JSON_PATH."
        )

    raw = json.loads(p.read_text(encoding="utf-8"))
    rows: List[Tuple[str, str]] = []

    for it in raw:
        sym = _safe_str(it.get("symbol"))
        if not sym:
            continue
        name = _safe_str(it.get("name")) or "Unknown"
        rows.append((sym, name))

    # dedup keep order
    seen = set()
    out: List[Tuple[str, str]] = []
    for sym, name in rows:
        if sym in seen:
            continue
        seen.add(sym)
        out.append((sym, name))

    if limit > 0:
        out = out[:limit]
    return out


def _fetch_one_table(url: str, timeout: int) -> pd.DataFrame:
    r = requests.get(url, headers=DEFAULT_HEADERS, timeout=timeout)
    r.raise_for_status()
    dfs = pd.read_html(StringIO(r.text), header=0)
    if not dfs:
        return pd.DataFrame()
    return dfs[0].copy()


def _extract_symbol_name_rows(df: pd.DataFrame, suffix: str) -> List[Tuple[str, str]]:
    if df is None or df.empty:
        return []

    col_code = None
    col_name = None
    for c in df.columns:
        cs = str(c).strip()
        if cs == "有價證券代號":
            col_code = c
        elif cs == "有價證券名稱":
            col_name = c

    if col_code is None or col_name is None:
        return []

    out: List[Tuple[str, str]] = []
    for _, r in df.iterrows():
        code = _safe_str(r.get(col_code))
        name = _safe_str(r.get(col_name))
        if not code or not suffix:
            continue
        sym = f"{code}{suffix}"
        out.append((sym, name or code))
    return out


def _fetch_tw_online(cfg: Optional[Dict[str, Any]] = None, limit: int = 0) -> List[Tuple[str, str]]:
    timeout = _cfg_int(cfg, "timeout_sec", 20)

    include_dr = _cfg_bool(cfg, "include_dr", True)
    include_rotc = _cfg_bool(cfg, "include_rotc", True)
    include_tw_innovation = _cfg_bool(cfg, "include_tw_innovation", True)
    include_otc_innovation = _cfg_bool(cfg, "include_otc_innovation", True)

    wanted_names = {"listed", "otc"}
    if include_dr:
        wanted_names.add("dr")
    if include_rotc:
        wanted_names.add("rotc")
    if include_tw_innovation:
        wanted_names.add("tw_innovation")
    if include_otc_innovation:
        wanted_names.add("otc_innovation")

    rows: List[Tuple[str, str]] = []

    for item in TW_URL_CONFIGS:
        name = item["name"]
        if name not in wanted_names:
            continue

        df = _fetch_one_table(item["url"], timeout=timeout)
        rows.extend(_extract_symbol_name_rows(df, item["suffix"]))

    # dedup keep order
    seen = set()
    out: List[Tuple[str, str]] = []
    for sym, name in rows:
        if sym in seen:
            continue
        seen.add(sym)
        out.append((sym, name))

    if limit > 0:
        out = out[:limit]

    return out


def get_tw_universe(limit: int = 0, cfg: Optional[Dict[str, Any]] = None) -> List[Tuple[str, str]]:
    """
    Return [(symbol, name), ...] for Taiwan.

    Priority:
    1. local JSON if provided / exists
    2. fetch online from TWSE ISIN pages

    Supported cfg:
      - list_json_path
      - timeout_sec
      - include_dr
      - include_rotc
      - include_tw_innovation
      - include_otc_innovation
      - prefer_online   (default True)
    """
    prefer_online = _cfg_bool(cfg, "prefer_online", True)

    json_path = (
        _cfg_str(cfg, "list_json_path", "")
        or os.getenv("TW_LIST_JSON_PATH", "").strip()
        or DEFAULT_JSON_PATH
    )

    if not prefer_online:
        return _load_local_json(json_path, limit=limit)

    # prefer online; fallback to local JSON
    try:
        rows = _fetch_tw_online(cfg=cfg, limit=limit)
        if rows:
            return rows
    except Exception:
        pass

    return _load_local_json(json_path, limit=limit)