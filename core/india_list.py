# core/india_list.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests


DEFAULT_NSE_CSV_URL = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
DEFAULT_YF_SUFFIX = ".NS"


def _is_blankish(x: Any) -> bool:
    s = ("" if x is None else str(x)).strip()
    return (not s) or s in ("-", "—", "--", "－", "–", "nan", "None")


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


def _yf_suffix(cfg: Optional[Dict[str, Any]] = None) -> str:
    v = _cfg_str(cfg, "yf_suffix", "") or (os.getenv("INDIA_YF_SUFFIX") or "").strip()
    return v or DEFAULT_YF_SUFFIX


def _to_yf_symbol(local_symbol: str, cfg: Optional[Dict[str, Any]] = None) -> str:
    s = (local_symbol or "").strip().upper()
    if not s:
        return ""
    suf = _yf_suffix(cfg)
    if s.endswith(suf.upper()) or s.endswith(suf.lower()):
        return s
    return f"{s}{suf}"


def _download_nse_csv(url: str, timeout: int = 30) -> pd.DataFrame:
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.nseindia.com/",
    }

    with requests.Session() as s:
        s.headers.update(headers)
        r = s.get(url, timeout=timeout)
        r.raise_for_status()
        df = pd.read_csv(io.BytesIO(r.content))

    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _load_local_csv(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"INDIA master CSV not found: {p}")
    df = pd.read_csv(p)
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _open_india_master(cfg: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """
    Priority:
    1. cfg['master_csv_path']
    2. env INDIA_MASTER_CSV_PATH
    3. NSE official CSV URL
    """
    local_path = _cfg_str(cfg, "master_csv_path", "") or (os.getenv("INDIA_MASTER_CSV_PATH") or "").strip()
    if local_path:
        return _load_local_csv(local_path)

    url = _cfg_str(cfg, "list_url", DEFAULT_NSE_CSV_URL)
    return _download_nse_csv(url=url, timeout=_cfg_int(cfg, "timeout_sec", 30))


def _normalize_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize to columns:
      - SYMBOL
      - NAME OF COMPANY
      - SERIES
    """
    df = df.copy()
    cols = {str(c).strip().upper(): c for c in df.columns}

    def pick(*names: str) -> Optional[str]:
        for n in names:
            if n.upper() in cols:
                return cols[n.upper()]
        return None

    col_symbol = pick("SYMBOL", "TICKER")
    col_name = pick("NAME OF COMPANY", "NAME", "COMPANY", "COMPANY NAME")
    col_series = pick("SERIES")

    if not col_symbol:
        raise RuntimeError(f"INDIA list missing SYMBOL column. got={list(df.columns)}")
    if not col_name:
        raise RuntimeError(f"INDIA list missing company-name column. got={list(df.columns)}")

    out = pd.DataFrame()
    out["SYMBOL"] = df[col_symbol].astype(str).str.strip().str.upper()
    out["NAME OF COMPANY"] = df[col_name].astype(str).str.strip()
    out["SERIES"] = df[col_series].astype(str).str.strip().str.upper() if col_series else "EQ"

    return out


def fetch_india_list(
    *,
    master_csv_path: Optional[str] = None,
    cfg: Optional[Dict[str, Any]] = None,
) -> List[Tuple[str, str]]:
    """
    Return:
      [("RELIANCE.NS", "RELIANCE INDUSTRIES LIMITED"), ...]

    Source priority:
      1. local CSV
      2. NSE official EQUITY_L.csv
    """
    local_cfg = dict(cfg or {})
    if master_csv_path:
        local_cfg["master_csv_path"] = master_csv_path

    df = _open_india_master(local_cfg)
    df = _normalize_schema(df)

    # basic clean
    df["SYMBOL"] = df["SYMBOL"].astype(str).str.strip().str.upper()
    df["NAME OF COMPANY"] = df["NAME OF COMPANY"].astype(str).str.strip()
    df["SERIES"] = df["SERIES"].astype(str).str.strip().str.upper()

    df = df[df["SYMBOL"].notna()].copy()
    df = df[df["SYMBOL"].str.len() > 0].copy()

    # 預設只留 EQ
    only_eq = _cfg_bool(local_cfg, "only_eq", True)
    if only_eq and "SERIES" in df.columns:
        df = df[df["SERIES"].eq("EQ")].copy()

    # 排除一些奇怪 symbol
    df = df[~df["SYMBOL"].str.contains(r"\s", regex=True, na=False)].copy()

    # 去重
    df = df.drop_duplicates(subset=["SYMBOL"]).reset_index(drop=True)

    out: List[Tuple[str, str]] = []
    for _, r in df.iterrows():
        local_symbol = _norm_text(r.get("SYMBOL"), "")
        if not local_symbol:
            continue

        yf_symbol = _to_yf_symbol(local_symbol, local_cfg)
        if not yf_symbol:
            continue

        name = _norm_text(r.get("NAME OF COMPANY"), local_symbol)
        out.append((yf_symbol, name))

    return out