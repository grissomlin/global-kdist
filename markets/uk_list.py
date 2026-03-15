# markets/uk/uk_list.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import re
import sqlite3
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import List, Tuple, Optional

import pandas as pd
import requests

from markets.us.us_db import init_db
from markets.us.us_config import TICKER_RE, EXCLUDE_NAME_RE, log

DEFAULT_UK_LIST_URL = "https://docs.londonstockexchange.com/sites/default/files/reports/Instrument%20list_75.xlsx"
DEFAULT_SHEET_NAME = "1.1 Shares"
DEFAULT_HEADER_KEY = "TIDM"

# allow: ABC, ABCD, 1234, BP.A, RE., 3IN, 80M, etc (TIDM style)
_TIDM_RE = re.compile(r"^[0-9A-Z]{1,6}(\.[0-9A-Z])?\.?$")

# words that are clearly NOT tickers (when header shifts)
_BAD_WORD_RE = re.compile(
    r"^(BANK(ING|ERS)?|MINERALS|SOFTWARE|WORLD|FUTURE|HEALTHCARE|CAPITAL|INCOME|HOLDINGS|ENGINEERING|CREDIT|VALUE|EUROPEAN|EMERGING)$"
)

# final yfinance symbol gate (after normalize): e.g. VOD.L, BP-A.L, 3IN.L, 80M.L
_YF_UK_RE = re.compile(r"^[0-9A-Z][0-9A-Z\-]{0,10}\.L$")


def _safe_str(x: object, default: str = "") -> str:
    try:
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return default
    except Exception:
        pass
    s = str(x).strip()
    return s if s else default


def _env_bool(name: str, default: str = "0") -> bool:
    return str(os.getenv(name, default)).strip().lower() in ("1", "true", "yes", "y", "on")


def _suffix() -> str:
    return (os.getenv("UK_TICKER_SUFFIX", ".L") or ".L").strip() or ".L"


def _clean_cell_symbol(s: str) -> str:
    """
    Clean raw Excel cell to a candidate TIDM-like string.
    """
    s = (s or "").strip()
    if not s:
        return ""
    # Excel sometimes stores leading apostrophe
    s = s.lstrip("'").strip()
    # remove obvious wrapping quotes
    s = s.strip('"').strip()
    return s.upper()


def _normalize_for_yf(tidm: str) -> str:
    """
    Convert LSE TIDM -> yfinance ticker:
    - "RE."  -> "RE.L"
    - "BP.A" -> "BP-A.L"
    - "VOD"  -> "VOD.L"
    """
    tidm = _clean_cell_symbol(tidm)
    if not tidm:
        return tidm

    if _env_bool("UK_KEEP_RAW_TICKER", "0"):
        return tidm

    suf = _suffix().upper()

    # already suffix
    if tidm.endswith(suf):
        return tidm

    # trailing dot: RE. => RE.L
    if _env_bool("UK_MAP_TRAILING_DOT", "1") and tidm.endswith("."):
        base = tidm[:-1].strip()
        return f"{base}{suf}" if base else tidm

    # class dot: BP.A => BP-A.L
    if _env_bool("UK_MAP_DOT_CLASS_TO_DASH", "1") and "." in tidm:
        base = tidm.replace(".", "-").strip("-")
        return f"{base}{suf}" if base else tidm

    return f"{tidm}{suf}"


def _looks_like_tidm(sym: str) -> bool:
    sym = _clean_cell_symbol(sym)
    if not sym:
        return False
    if any(ch.isspace() for ch in sym):
        return False
    if len(sym) > 8:
        return False
    if _BAD_WORD_RE.match(sym):
        return False
    # quick reject weird symbols
    if any(ch in sym for ch in ("$", "&", "(", ")", "/", "\\", ":", ";", ",", "'", '"')):
        return False
    return bool(_TIDM_RE.match(sym))


def _open_excel() -> pd.ExcelFile:
    local_path = (os.getenv("UK_LIST_XLSX_PATH") or "").strip()
    if local_path:
        p = Path(local_path)
        if p.exists():
            log(f"📄 Reading UK instrument list from local file: {p}")
            return pd.ExcelFile(str(p))
        log(f"⚠️ UK_LIST_XLSX_PATH set but not found: {p} (will download from URL)")

    url = (os.getenv("UK_LIST_URL") or DEFAULT_UK_LIST_URL).strip() or DEFAULT_UK_LIST_URL
    log(f"📡 Downloading UK instrument list ... {url}")
    r = requests.get(url, timeout=90)
    r.raise_for_status()
    return pd.ExcelFile(BytesIO(r.content))


def _pick_sheet(xls: pd.ExcelFile) -> str:
    sheet = (os.getenv("UK_SHEET_NAME") or DEFAULT_SHEET_NAME).strip() or DEFAULT_SHEET_NAME
    if sheet in xls.sheet_names:
        return sheet

    log(f"⚠️ sheet '{sheet}' not found. Available sheets: {xls.sheet_names}")
    for cand in ["1.1 Shares", "Shares", "1.0 All Equity"]:
        if cand in xls.sheet_names:
            log(f"➡️ fallback sheet = {cand}")
            return cand
    for s in xls.sheet_names:
        if "share" in s.lower():
            log(f"➡️ fallback sheet = {s}")
            return s
    return xls.sheet_names[0]


def _score_header_row(tmp: pd.DataFrame, header_row: int) -> float:
    """
    Read using this header_row and score how 'ticker-like' the TIDM column is.
    Higher is better.
    """
    try:
        # Build header names from row
        cols = tmp.iloc[header_row].astype(str).str.strip().tolist()
        if "TIDM" not in cols or "Issuer Name" not in cols:
            return -1.0

        # slice data rows below header_row
        data = tmp.iloc[header_row + 1 : header_row + 1 + 200].copy()
        data.columns = cols

        if "TIDM" not in data.columns:
            return -1.0

        s = data["TIDM"].astype(str).tolist()
        if not s:
            return -1.0

        good = 0
        total = 0
        for x in s:
            xx = _clean_cell_symbol(str(x))
            if not xx or xx.lower() == "nan":
                continue
            total += 1
            if _looks_like_tidm(xx):
                good += 1
        if total == 0:
            return -1.0

        return good / total
    except Exception:
        return -1.0


def _detect_header_row_best(xls: pd.ExcelFile, sheet_name: str) -> int:
    scan_rows = int(os.getenv("UK_HEADER_SCAN_ROWS", "60") or "60")
    tmp = pd.read_excel(xls, sheet_name=sheet_name, header=None, nrows=scan_rows)

    # candidates: rows containing BOTH TIDM + Issuer Name
    candidates: List[int] = []
    for i in range(len(tmp)):
        row = tmp.iloc[i].astype(str).str.strip().tolist()
        has_tidm = any(v == "TIDM" for v in row)
        has_issuer = any(v == "Issuer Name" for v in row)
        if has_tidm and has_issuer:
            candidates.append(int(i))

    if not candidates:
        # fallback: first row containing TIDM
        for i in range(len(tmp)):
            row = tmp.iloc[i].astype(str).str.strip().tolist()
            if any(v == "TIDM" for v in row):
                return int(i)
        return 8

    # pick best by scoring ticker-likeness
    best_row = candidates[0]
    best_score = -1.0
    for r in candidates:
        sc = _score_header_row(tmp, r)
        if sc > best_score:
            best_row = r
            best_score = sc

    log(f"🧩 UK header candidates={candidates} | picked={best_row} | score={best_score:.2%}")
    return int(best_row)


def _read_sheet(xls: pd.ExcelFile, sheet_name: str) -> pd.DataFrame:
    header_row = _detect_header_row_best(xls, sheet_name)
    df = pd.read_excel(xls, sheet_name=sheet_name, header=header_row)
    df = df.dropna(how="all").reset_index(drop=True)

    # soft-normalize column names (strip)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _final_symbol_gate(symbol_yf: str) -> bool:
    s = (symbol_yf or "").strip().upper()
    if not s:
        return False
    if _BAD_WORD_RE.match(s.replace(".L", "")):
        return False
    # must be yfinance uk format
    return bool(_YF_UK_RE.match(s))


def get_uk_stock_list(db_path: Path, refresh_list: bool = True) -> List[Tuple[str, str]]:
    init_db(db_path)

    if not refresh_list and db_path.exists():
        conn = sqlite3.connect(str(db_path))
        try:
            df = pd.read_sql_query("SELECT symbol, name FROM stock_info WHERE market='UK'", conn)
            if not df.empty:
                items = [(str(r["symbol"]), str(r["name"])) for _, r in df.iterrows()]
                log(f"✅ 使用 DB stock_info 既有 UK 清單: {len(items)} 檔")
                return items
            log("⚠️ refresh_list=False but UK stock_info is empty; will fetch list.")
        finally:
            conn.close()

    try:
        xls = _open_excel()
    except Exception as e:
        log(f"❌ UK list open failed: {e}")
        return []

    sheet = _pick_sheet(xls)
    df = _read_sheet(xls, sheet)
    if df is None or df.empty:
        log("❌ UK list sheet empty")
        return []

    # required columns
    if "TIDM" not in df.columns or "Issuer Name" not in df.columns:
        log(f"❌ Missing required columns. columns={list(df.columns)}")
        return []

    # sanity-check
    sample = df["TIDM"].head(200).astype(str).tolist()
    bad = sum(1 for s in sample if not _looks_like_tidm(str(s)))
    if len(sample) >= 50 and bad / len(sample) > 0.25:
        log(f"⚠️ UK list sanity FAIL-ish: non-ticker TIDM in sample ({bad}/{len(sample)}). Header row may be wrong.")

    # Keep SHRS if available
    if _env_bool("UK_REQUIRE_MIFIR_SHRS", "1") and "MiFIR Identifier Code" in df.columns:
        df = df[df["MiFIR Identifier Code"].astype(str).str.strip().str.upper() == "SHRS"].copy()

    has_super = "ICB Super-Sector Name" in df.columns
    has_ind = "ICB Industry" in df.columns

    limit_n = int(os.getenv("UK_LIMIT_SYMBOLS", "0") or "0")
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    conn = sqlite3.connect(str(db_path))
    stock_list: List[Tuple[str, str]] = []
    rejected = 0

    try:
        for _, row in df.iterrows():
            tidm_raw = _safe_str(row.get("TIDM", ""), "")
            tidm_raw = _clean_cell_symbol(tidm_raw)
            if not _looks_like_tidm(tidm_raw):
                continue

            name = _safe_str(row.get("Issuer Name", ""), tidm_raw) or tidm_raw
            if EXCLUDE_NAME_RE.search(name or ""):
                continue

            industry = _safe_str(row.get("ICB Industry", ""), "Unknown") if has_ind else "Unknown"
            super_sector = _safe_str(row.get("ICB Super-Sector Name", ""), "Unknown") if has_super else "Unknown"
            sector = super_sector if super_sector and super_sector != "Unknown" else (industry or "Unknown")

            lse_market = _safe_str(row.get("LSE Market", ""), "Unknown") if "LSE Market" in df.columns else "Unknown"

            symbol = _normalize_for_yf(tidm_raw)

            # final hard gate: only allow real-looking UK yfinance tickers
            if not _final_symbol_gate(symbol):
                rejected += 1
                continue

            conn.execute(
                """
                INSERT OR REPLACE INTO stock_info
                (symbol, name, sector, market, market_detail, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (symbol, name, sector, "UK", lse_market, now),
            )
            stock_list.append((symbol, name))

            if limit_n > 0 and len(stock_list) >= limit_n:
                break

        conn.commit()
    finally:
        conn.close()

    log(f"✅ UK list imported: {len(stock_list)} (rejected={rejected}, sheet={sheet}, limit={limit_n or 'ALL'})")
    return stock_list