# core/th_list.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
from typing import Any, List, Tuple, Optional

import pandas as pd


DEFAULT_YF_SUFFIX = ".BK"


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "y", "on")


def _yf_suffix() -> str:
    # 跟你 th_config 的語意一致：可用 env 覆蓋
    return str(os.getenv("TH_YF_SUFFIX", DEFAULT_YF_SUFFIX)).strip() or DEFAULT_YF_SUFFIX


def _disable_thaifin() -> bool:
    return _env_bool("TH_DISABLE_THAIFIN", False)


def _list_xlsx_path() -> str:
    # 你原本 th_config 有自動路徑；這裡給最常見 env 控制
    # 使用者沒設就回傳空字串，代表不使用 xlsx
    return str(os.getenv("TH_LIST_XLSX_PATH", "")).strip()


def _is_blankish(x: Any) -> bool:
    s = ("" if x is None else str(x)).strip()
    if (not s) or s in ("-", "—", "--", "－", "–", "nan", "None"):
        return True
    if s.lower() in ("unknown", "unclassified"):
        return True
    return False


def _norm_text(x: Any, default: str) -> str:
    s = ("" if x is None else str(x)).strip()
    return default if _is_blankish(s) else s


def _to_yf_symbol(local_symbol: str) -> str:
    s = (local_symbol or "").strip().upper()
    if not s:
        return ""
    suf = _yf_suffix()
    if s.endswith(suf.upper()) or s.endswith(suf.lower()):
        return s
    return f"{s}{suf}"


def _try_load_list_from_thaifin() -> Optional[pd.DataFrame]:
    """
    Expected DF columns from thaifin:
      ['symbol', 'name', 'industry', 'sector', 'market']
    """
    if _disable_thaifin():
        return None

    try:
        from thaifin import Stocks  # type: ignore
    except Exception:
        return None

    try:
        df = Stocks.list_with_names()
        if df is None or len(df) == 0:
            return None
        df = df.copy()
        df.columns = [str(c).strip().lower() for c in df.columns]
        return df
    except Exception:
        return None


def _load_list_from_xlsx(xlsx_path: str) -> Optional[pd.DataFrame]:
    if not xlsx_path or not os.path.exists(xlsx_path):
        return None
    try:
        df = pd.read_excel(xlsx_path, engine="openpyxl")
        df = df.copy()
        df.columns = [str(c).strip().lower() for c in df.columns]
        return df
    except Exception:
        return None


def fetch_th_list(
    *,
    include_fields: bool = False,
) -> List[Tuple[str, str]]:
    """
    Return:
      [( "CPALL.BK", "CP ALL PUBLIC COMPANY LIMITED" ), ...]

    Source priority:
      1) thaifin (unless TH_DISABLE_THAIFIN=1)
      2) xlsx from TH_LIST_XLSX_PATH

    Notes:
      - This is a *pure* universe provider (no DB writes).
      - We keep only (ticker, name) to match JP style.
    """
    df = _try_load_list_from_thaifin()
    if df is None or df.empty:
        df = _load_list_from_xlsx(_list_xlsx_path())

    if df is None or df.empty:
        return []

    for c in ["symbol", "name"]:
        if c not in df.columns:
            df[c] = None

    out: List[Tuple[str, str]] = []
    for _, row in df.iterrows():
        local_symbol = _norm_text(row.get("symbol"), "")
        if not local_symbol:
            continue

        ticker = _to_yf_symbol(local_symbol)
        if not ticker:
            continue

        name = _norm_text(row.get("name"), "Unknown")
        out.append((ticker, name))

    return out