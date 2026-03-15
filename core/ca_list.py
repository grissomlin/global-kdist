# core/ca_list.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import re
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests


TMX_CA_ISSUERS_URL = "https://www.tsx.com/resource/en/571"


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


def _download_tmx_excel(url: str, timeout: int = 90) -> pd.ExcelFile:
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return pd.ExcelFile(BytesIO(r.content))


def _find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    norm = {str(c).replace("\n", " ").strip().lower(): c for c in df.columns}
    for cand in candidates:
        k = cand.replace("\n", " ").strip().lower()
        if k in norm:
            return norm[k]
    return None


def _read_exchange_sheet(
    xls: pd.ExcelFile,
    sheet_name_or_idx: Any,
    exchange_name: str,
    skiprows: int,
) -> pd.DataFrame:
    df = pd.read_excel(xls, sheet_name=sheet_name_or_idx, skiprows=skiprows)
    df = df.dropna(how="all").reset_index(drop=True)

    col_symbol = _find_col(df, ["Root Ticker", "Root\nTicker", "Ticker", "Symbol"])
    col_name = _find_col(df, ["Name", "Company Name", "Issuer Name"])
    col_sector = _find_col(df, ["Sector", "Industry", "Industry Group"])

    if not col_symbol or not col_name:
        raise RuntimeError(
            f"[CA] Cannot detect required columns in sheet={sheet_name_or_idx}. "
            f"columns={list(df.columns)}"
        )

    out = pd.DataFrame()
    out["symbol"] = df[col_symbol].map(_safe_str).str.upper()
    out["name"] = df[col_name].map(_safe_str)
    out["sector"] = df[col_sector].map(_safe_str) if col_sector else ""
    out["exchange"] = exchange_name
    return out


def _read_tmx_issuers(xls: pd.ExcelFile, cfg: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    skiprows = _cfg_int(cfg, "skiprows", 9)
    tsx_sheet = cfg.get("tsx_sheet", 0) if cfg else 0
    tsxv_sheet = cfg.get("tsxv_sheet", 1) if cfg else 1

    df_tsx = _read_exchange_sheet(xls, tsx_sheet, "TSX", skiprows=skiprows)
    df_tsxv = _read_exchange_sheet(xls, tsxv_sheet, "TSXV", skiprows=skiprows)

    full = pd.concat([df_tsx, df_tsxv], ignore_index=True)

    full["symbol"] = full["symbol"].map(_safe_str).str.upper()
    full["name"] = full["name"].map(_safe_str)
    full["sector"] = full["sector"].map(_safe_str)
    full["exchange"] = full["exchange"].map(_safe_str).str.upper()

    full = full[full["symbol"] != ""].copy()
    full = full[full["name"] != ""].copy()

    return full.reset_index(drop=True)


def _is_probably_common_equity(
    symbol: Any,
    name: Any,
    exchange: Any,
    cfg: Optional[Dict[str, Any]] = None,
) -> bool:
    s = _safe_str(symbol).upper()
    n = _safe_str(name).upper()
    e = _safe_str(exchange).upper()

    if not s or not n:
        return False

    if re.search(r"\s", s):
        return False

    bad_symbol_patterns = [
        r"\.WT$",
        r"\.WS$",
        r"\.RT$",
        r"\.RT\.[A-Z]$",
        r"\.PR\.",
    ]

    exclude_units = _cfg_bool(cfg, "exclude_units", True)
    if exclude_units:
        bad_symbol_patterns.extend([
            r"\.U$",
            r"\.UN$",
        ])

    for p in bad_symbol_patterns:
        if re.search(p, s):
            return False

    bad_name_patterns = [
        r"\bETF\b",
        r"\bETFS\b",
        r"\bFUND\b",
        r"\bFUNDS\b",
        r"\bINDEX\b",
        r"\bTRUST\b",
        r"\bSPLIT\b",
        r"\bPREFERRED\b",
        r"\bPREF\b",
        r"\bDEBENTURE\b",
        r"\bDEBENTURES\b",
        r"\bNOTE\b",
        r"\bNOTES\b",
        r"\bWARRANT\b",
        r"\bWARRANTS\b",
        r"\bRIGHT\b",
        r"\bRIGHTS\b",
        r"\bSUBORDINATE VOTING\b",
        r"\bEXCHANGEABLE\b",
        r"\bRECEIPT\b",
        r"\bRECEIPTS\b",
        r"\bINCOME FUND\b",
        r"\bCAPITAL POOL\b",
        r"\bCPC\b",
    ]

    if exclude_units:
        bad_name_patterns.extend([
            r"\bUNIT\b",
            r"\bUNITS\b",
        ])

    for p in bad_name_patterns:
        if re.search(p, n):
            return False

    return True


def _to_yf_ticker(symbol: Any, exchange: Any) -> str:
    s = _safe_str(symbol).upper()
    e = _safe_str(exchange).upper()

    if not s:
        return ""

    if s.endswith(".TO") or s.endswith(".V"):
        return s

    if e == "TSXV":
        return f"{s}.V"
    return f"{s}.TO"


def get_ca_universe_df(cfg: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    url = _cfg_str(cfg, "list_url", TMX_CA_ISSUERS_URL)
    include_tsxv = _cfg_bool(cfg, "include_tsxv", True)
    limit = _cfg_int(cfg, "limit", 0)

    xls = _download_tmx_excel(url=url)
    df = _read_tmx_issuers(xls, cfg=cfg)

    df = df[df["exchange"].isin(["TSX", "TSXV"])].copy()

    if not include_tsxv:
        df = df[df["exchange"].eq("TSX")].copy()

    keep_mask = [
        _is_probably_common_equity(sym, name, exch, cfg=cfg)
        for sym, name, exch in zip(df["symbol"], df["name"], df["exchange"])
    ]
    df = df[keep_mask].copy()

    df["yf_ticker"] = [
        _to_yf_ticker(sym, exch)
        for sym, exch in zip(df["symbol"], df["exchange"])
    ]

    df = df[df["yf_ticker"].map(_safe_str) != ""].copy()
    df = df[~df["yf_ticker"].astype(str).str.contains(r"\s", regex=True, na=False)].copy()

    df = df.drop_duplicates(subset=["yf_ticker"]).reset_index(drop=True)

    if limit > 0:
        df = df.head(limit).copy()

    return df.reset_index(drop=True)


def get_ca_universe(limit: int = 0, cfg: Optional[Dict[str, Any]] = None) -> List[Tuple[str, str]]:
    local_cfg = dict(cfg or {})
    if limit > 0:
        local_cfg["limit"] = limit

    df = get_ca_universe_df(local_cfg)
    return list(df[["yf_ticker", "name"]].itertuples(index=False, name=None))