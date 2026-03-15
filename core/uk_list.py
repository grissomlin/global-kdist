# core/uk_list.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
from io import BytesIO
from typing import List, Tuple

import pandas as pd
import requests

DEFAULT_UK_LIST_URL = "https://docs.londonstockexchange.com/sites/default/files/reports/Instrument%20list_75.xlsx"
DEFAULT_SHEET_NAME = "1.1 Shares"


def _env_bool(name: str, default: str = "0") -> bool:
    return str(os.getenv(name, default)).strip().lower() in ("1", "true", "yes", "y", "on")


def _suffix() -> str:
    return (os.getenv("UK_TICKER_SUFFIX", ".L") or ".L").strip() or ".L"


def _normalize_for_yf(tidm: str) -> str:
    """
    Convert LSE TIDM -> yfinance ticker:
    - RE.  -> RE.L
    - BP.A -> BP-A.L
    - already endswith .L -> keep
    """
    s = (tidm or "").strip().upper()
    if not s:
        return s

    suf = _suffix().upper()

    if s.endswith(suf):
        return s

    if _env_bool("UK_MAP_TRAILING_DOT", "1") and s.endswith("."):
        base = s[:-1].strip()
        return f"{base}{suf}" if base else s

    if _env_bool("UK_MAP_DOT_CLASS_TO_DASH", "1") and "." in s:
        base = s.replace(".", "-").strip("-")
        return f"{base}{suf}" if base else s

    return f"{s}{suf}"


def _download_xlsx(url: str) -> pd.ExcelFile:
    r = requests.get(url, timeout=90)
    r.raise_for_status()
    return pd.ExcelFile(BytesIO(r.content))


def _pick_sheet(xls: pd.ExcelFile) -> str:
    want = (os.getenv("UK_SHEET_NAME") or DEFAULT_SHEET_NAME).strip() or DEFAULT_SHEET_NAME
    if want in xls.sheet_names:
        return want

    # fallbacks
    for cand in ["1.1 Shares", "Shares", "1.0 All Equity"]:
        if cand in xls.sheet_names:
            return cand

    # any sheet with "share"
    for s in xls.sheet_names:
        if "share" in s.lower():
            return s

    return xls.sheet_names[0]


def _detect_header_row(xls: pd.ExcelFile, sheet: str) -> int:
    key = (os.getenv("UK_HEADER_KEY") or "TIDM").strip() or "TIDM"
    scan = int(os.getenv("UK_HEADER_SCAN_ROWS", "30") or "30")
    tmp = pd.read_excel(xls, sheet_name=sheet, header=None, nrows=scan)
    for i in range(len(tmp)):
        row = tmp.iloc[i].astype(str).str.strip().tolist()
        if key in row:
            return i
    return 8  # typical


def get_uk_universe(limit: int = 0) -> List[Tuple[str, str]]:
    """
    Return [(yf_ticker, issuer_name), ...]
    """
    url = (os.getenv("UK_LIST_URL") or DEFAULT_UK_LIST_URL).strip() or DEFAULT_UK_LIST_URL
    xls = _download_xlsx(url)
    sheet = _pick_sheet(xls)
    hdr = _detect_header_row(xls, sheet)

    df = pd.read_excel(xls, sheet_name=sheet, header=hdr).dropna(how="all").reset_index(drop=True)

    # Required columns
    if "TIDM" not in df.columns or "Issuer Name" not in df.columns:
        raise RuntimeError(f"UK list missing columns. got={list(df.columns)}")

    # Filter SHRS if available (普通股)
    if _env_bool("UK_REQUIRE_MIFIR_SHRS", "1") and "MiFIR Identifier Code" in df.columns:
        df = df[df["MiFIR Identifier Code"].astype(str).str.strip().str.upper() == "SHRS"].copy()

    rows: List[Tuple[str, str]] = []
    for _, r in df.iterrows():
        tidm = str(r.get("TIDM") or "").strip().upper()
        if not tidm:
            continue
        name = str(r.get("Issuer Name") or tidm).strip() or tidm

        yf_ticker = _normalize_for_yf(tidm)
        if not yf_ticker:
            continue

        rows.append((yf_ticker, name))
        if limit > 0 and len(rows) >= limit:
            break

    return rows