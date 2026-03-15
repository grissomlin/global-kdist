# core/germany_list.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests


DEFAULT_DB_XLS_URL = (
    "https://www.cashmarket.deutsche-boerse.com/resource/blob/313988/"
    "f4f80e006bf17d42e84dfb4b6b9ef341/data/Gelistete-Unternehmen.xls"
)
DEFAULT_YF_SUFFIX = ".DE"
DEFAULT_SEGMENTS = ["Prime Standard", "General Standard", "Entry Standard"]


def _is_blankish(x: Any) -> bool:
    s = ("" if x is None else str(x)).strip()
    return (not s) or s in {"-", "—", "--", "－", "–", "nan", "None", "NaN"}


def _norm_text(x: Any, default: str = "") -> str:
    s = ("" if x is None else str(x)).strip()
    if _is_blankish(s):
        return default
    return s


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


def _cfg_list_str(
    cfg: Optional[Dict[str, Any]],
    key: str,
    default: Optional[List[str]] = None,
) -> List[str]:
    if cfg is None:
        return list(default or [])
    v = cfg.get(key, default or [])
    if v is None:
        return list(default or [])
    if isinstance(v, (list, tuple)):
        out = []
        for x in v:
            s = str(x).strip()
            if s:
                out.append(s)
        return out
    s = str(v).strip()
    if not s:
        return list(default or [])
    return [p.strip() for p in s.split(",") if str(p).strip()]


def _yf_suffix(cfg: Optional[Dict[str, Any]] = None) -> str:
    v = _cfg_str(cfg, "yf_suffix", "") or (os.getenv("GERMANY_YF_SUFFIX") or "").strip()
    return v or DEFAULT_YF_SUFFIX


def _to_yf_symbol(local_symbol: str, cfg: Optional[Dict[str, Any]] = None) -> str:
    s = (local_symbol or "").strip().upper()
    if not s:
        return ""
    suf = _yf_suffix(cfg)
    if s.endswith(suf.upper()) or s.endswith(suf.lower()):
        return s
    return f"{s}{suf}"


def _download_xls(url: str, timeout: int = 30) -> pd.ExcelFile:
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.deutsche-boerse.com/",
    }

    with requests.Session() as s:
        s.headers.update(headers)
        r = s.get(url, timeout=timeout)
        r.raise_for_status()
        return pd.ExcelFile(BytesIO(r.content))


def _load_local_xls(path: str) -> pd.ExcelFile:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"GERMANY master XLS not found: {p}")
    return pd.ExcelFile(str(p))


def _open_germany_master(cfg: Optional[Dict[str, Any]] = None) -> pd.ExcelFile:
    """
    Priority:
    1. cfg['master_xls_path']
    2. env GERMANY_MASTER_XLS_PATH
    3. Deutsche Börse official XLS URL
    """
    local_path = _cfg_str(cfg, "master_xls_path", "") or (os.getenv("GERMANY_MASTER_XLS_PATH") or "").strip()
    if local_path:
        return _load_local_xls(local_path)

    url = _cfg_str(cfg, "list_url", DEFAULT_DB_XLS_URL)
    timeout = _cfg_int(cfg, "timeout_sec", 30)
    return _download_xls(url=url, timeout=timeout)


def _find_header_row(df_raw: pd.DataFrame) -> Optional[int]:
    """
    Find row containing header. Prefer rows containing ISIN / Trading Symbol / Company.
    """
    for idx, row in df_raw.iterrows():
        vals = [str(v).strip() for v in row.values]
        joined = " | ".join(vals).upper()
        if "ISIN" in joined and ("TRADING SYMBOL" in joined or "COMPANY" in joined):
            return int(idx)

    for idx, row in df_raw.iterrows():
        first = str(row.values[0]).strip().upper() if len(row.values) > 0 else ""
        if "ISIN" in first:
            return int(idx)

    return None


def _load_sheet(xls: pd.ExcelFile, sheet_name: str) -> Optional[pd.DataFrame]:
    df_raw = pd.read_excel(xls, sheet_name=sheet_name, header=None)
    header_row = _find_header_row(df_raw)
    if header_row is None:
        return None

    df = pd.read_excel(xls, sheet_name=sheet_name, skiprows=header_row)
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    df["segment"] = sheet_name
    return df


def _pick_col(df: pd.DataFrame, *candidates: str) -> Optional[str]:
    cols = [str(c).strip() for c in df.columns]
    cols_upper = {c.upper(): c for c in cols}

    for cand in candidates:
        key = cand.strip().upper()
        if key in cols_upper:
            return cols_upper[key]

    for cand in candidates:
        c_up = cand.strip().upper()
        for c in cols:
            cu = c.upper()
            if c_up in cu or cu in c_up:
                return c

    return None


def _clean_category_value(x: Any) -> str:
    s = _norm_text(x, "")
    if not s:
        return ""
    s = s.replace("+", "&")
    s = " ".join(s.split())
    lower_map = {
        "basic resources": "Basic Resources",
        "financial services": "Financial Services",
        "pharma & healthcare": "Pharma & Healthcare",
        "transportation & logistics": "Transportation & Logistics",
        "transportation & logistics ": "Transportation & Logistics",
        "consumer": "Consumer",
        "industrial": "Industrial",
        "technology": "Technology",
        "retail": "Retail",
        "software": "Software",
        "media": "Media",
        "banks": "Banks",
        "insurance": "Insurance",
        "utilities": "Utilities",
        "chemicals": "Chemicals",
        "construction": "Construction",
        "food & beverages": "Food & Beverages",
        "telecommunication": "Telecommunication",
        "automobile": "Automobile",
    }
    return lower_map.get(s.lower(), s)


def _is_common_stock(company_name: str) -> bool:
    """
    Heuristic:
    exclude preferred / non-common names
    """
    s = _norm_text(company_name, "").upper()
    if not s:
        return False

    bad_patterns = [
        r"\bVZ\b",
        r"\bVORZUG\b",
        r"\bVORZUGSAKTIEN\b",
        r"\bPREF\b",
        r"\bPREFERENCE\b",
        r"\bPREFERRED\b",
    ]

    import re
    for pat in bad_patterns:
        if re.search(pat, s, flags=re.IGNORECASE):
            return False
    return True


def _normalize_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize to columns:
      - ISIN
      - TRADING_SYMBOL
      - COMPANY
      - SECTOR
      - SUBSECTOR
      - SEGMENT
    """
    df = df.copy()

    col_isin = _pick_col(df, "ISIN")
    col_symbol = _pick_col(df, "Trading Symbol", "Symbol", "Ticker")
    col_company = _pick_col(df, "Company", "Issuer")
    col_sector = _pick_col(df, "Sector")
    col_subsector = _pick_col(df, "Subsector", "Sub-Sector", "Sub Sector")
    col_segment = _pick_col(df, "segment")

    if not col_symbol:
        raise RuntimeError(f"GERMANY list missing trading-symbol column. got={list(df.columns)}")
    if not col_company:
        raise RuntimeError(f"GERMANY list missing company column. got={list(df.columns)}")

    out = pd.DataFrame()
    out["ISIN"] = df[col_isin].astype(str).str.strip() if col_isin else ""
    out["TRADING_SYMBOL"] = df[col_symbol].astype(str).str.strip().str.upper()
    out["COMPANY"] = df[col_company].astype(str).str.strip()
    out["SECTOR"] = df[col_sector].astype(str).str.strip() if col_sector else ""
    out["SUBSECTOR"] = df[col_subsector].astype(str).str.strip() if col_subsector else ""
    out["SEGMENT"] = df[col_segment].astype(str).str.strip() if col_segment else ""

    return out


def _load_all_segments(
    xls: pd.ExcelFile,
    segments: List[str],
) -> pd.DataFrame:
    dfs: List[pd.DataFrame] = []

    available = set(xls.sheet_names)

    for sheet in segments:
        if sheet not in available:
            continue
        df = _load_sheet(xls, sheet)
        if df is not None and len(df) > 0:
            dfs.append(df)

    if not dfs:
        raise RuntimeError(
            f"No usable sheets found. requested={segments}, available={xls.sheet_names}"
        )

    df_all = pd.concat(dfs, ignore_index=True)
    return _normalize_schema(df_all)


def fetch_germany_list(
    *,
    master_xls_path: Optional[str] = None,
    cfg: Optional[Dict[str, Any]] = None,
) -> List[Tuple[str, str]]:
    """
    Return:
      [("SIE.DE", "Siemens AG"), ...]

    Source priority:
      1. local XLS
      2. Deutsche Börse official XLS
    """
    local_cfg = dict(cfg or {})
    if master_xls_path:
        local_cfg["master_xls_path"] = master_xls_path

    xls = _open_germany_master(local_cfg)

    segments = _cfg_list_str(local_cfg, "segments", DEFAULT_SEGMENTS)
    df = _load_all_segments(xls, segments)

    df["TRADING_SYMBOL"] = df["TRADING_SYMBOL"].astype(str).str.strip().str.upper()
    df["COMPANY"] = df["COMPANY"].astype(str).str.strip()
    df["SECTOR"] = df["SECTOR"].map(_clean_category_value)
    df["SUBSECTOR"] = df["SUBSECTOR"].map(_clean_category_value)
    df["SEGMENT"] = df["SEGMENT"].astype(str).str.strip()

    df = df[df["TRADING_SYMBOL"].notna()].copy()
    df = df[df["TRADING_SYMBOL"].str.len() > 0].copy()
    df = df[df["COMPANY"].notna()].copy()
    df = df[df["COMPANY"].str.len() > 0].copy()

    # 排除 symbol 含空白
    df = df[~df["TRADING_SYMBOL"].str.contains(r"\s", regex=True, na=False)].copy()

    # 預設只留普通股
    only_common = _cfg_bool(local_cfg, "only_common", True)
    if only_common:
        df = df[df["COMPANY"].map(_is_common_stock)].copy()

    # 去重：優先保留 Prime -> General -> Entry
    segment_rank = {
        "Prime Standard": 0,
        "General Standard": 1,
        "Entry Standard": 2,
    }
    df["_segment_rank"] = df["SEGMENT"].map(lambda x: segment_rank.get(str(x), 999))
    df = df.sort_values(["TRADING_SYMBOL", "_segment_rank", "COMPANY"]).copy()
    df = df.drop_duplicates(subset=["TRADING_SYMBOL"], keep="first").reset_index(drop=True)

    out: List[Tuple[str, str]] = []
    for _, r in df.iterrows():
        local_symbol = _norm_text(r.get("TRADING_SYMBOL"), "")
        if not local_symbol:
            continue

        yf_symbol = _to_yf_symbol(local_symbol, local_cfg)
        if not yf_symbol:
            continue

        name = _norm_text(r.get("COMPANY"), local_symbol)
        out.append((yf_symbol, name))

    return out


def fetch_germany_rows(
    *,
    master_xls_path: Optional[str] = None,
    cfg: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Rich rows version if future scripts need sector / segment.
    """
    local_cfg = dict(cfg or {})
    if master_xls_path:
        local_cfg["master_xls_path"] = master_xls_path

    xls = _open_germany_master(local_cfg)
    segments = _cfg_list_str(local_cfg, "segments", DEFAULT_SEGMENTS)
    df = _load_all_segments(xls, segments)

    df["TRADING_SYMBOL"] = df["TRADING_SYMBOL"].astype(str).str.strip().str.upper()
    df["COMPANY"] = df["COMPANY"].astype(str).str.strip()
    df["SECTOR"] = df["SECTOR"].map(_clean_category_value)
    df["SUBSECTOR"] = df["SUBSECTOR"].map(_clean_category_value)
    df["SEGMENT"] = df["SEGMENT"].astype(str).str.strip()

    df = df[df["TRADING_SYMBOL"].notna()].copy()
    df = df[df["TRADING_SYMBOL"].str.len() > 0].copy()
    df = df[df["COMPANY"].notna()].copy()
    df = df[df["COMPANY"].str.len() > 0].copy()
    df = df[~df["TRADING_SYMBOL"].str.contains(r"\s", regex=True, na=False)].copy()

    only_common = _cfg_bool(local_cfg, "only_common", True)
    if only_common:
        df = df[df["COMPANY"].map(_is_common_stock)].copy()

    segment_rank = {
        "Prime Standard": 0,
        "General Standard": 1,
        "Entry Standard": 2,
    }
    df["_segment_rank"] = df["SEGMENT"].map(lambda x: segment_rank.get(str(x), 999))
    df = df.sort_values(["TRADING_SYMBOL", "_segment_rank", "COMPANY"]).copy()
    df = df.drop_duplicates(subset=["TRADING_SYMBOL"], keep="first").reset_index(drop=True)

    rows: List[Dict[str, Any]] = []
    for _, r in df.iterrows():
        local_symbol = _norm_text(r.get("TRADING_SYMBOL"), "")
        if not local_symbol:
            continue

        yf_symbol = _to_yf_symbol(local_symbol, local_cfg)
        if not yf_symbol:
            continue

        rows.append(
            {
                "symbol": yf_symbol,
                "local_symbol": local_symbol,
                "name": _norm_text(r.get("COMPANY"), local_symbol),
                "isin": _norm_text(r.get("ISIN"), ""),
                "sector": _norm_text(r.get("SECTOR"), ""),
                "subsector": _norm_text(r.get("SUBSECTOR"), ""),
                "segment": _norm_text(r.get("SEGMENT"), ""),
            }
        )

    return rows