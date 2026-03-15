# core/hkex_list.py
from __future__ import annotations

import io
import re
from typing import List, Tuple

import pandas as pd
import requests

HKEX_URL = (
    "https://www.hkex.com.hk/-/media/HKEX-Market/Services/Trading/Securities/"
    "Securities-Lists/Securities-Using-Standard-Transfer-Form-(including-GEM)-By-Stock-Code-Order/secstkorder.xls"
)

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def normalize_code5_any(s: str) -> str:
    digits = re.sub(r"\D", "", str(s or ""))
    return digits[-5:].zfill(5) if digits else ""

def _clean_cell(x) -> str:
    return str(x or "").replace("\xa0", " ").strip()

def locate_header_row(df_raw: pd.DataFrame, max_scan: int = 40) -> int | None:
    """
    HKEX xls often has disclaimer rows above the real header.
    We find the row that contains both "Stock Code" and "English Stock Short Name"
    (case-insensitive, allow spacing variations).
    """
    code_pat = re.compile(r"stock\s*code", re.I)
    name_pat = re.compile(r"english\s*stock\s*short\s*name", re.I)

    for i in range(min(max_scan, len(df_raw))):
        row = [_clean_cell(x) for x in df_raw.iloc[i].tolist()]
        if any(code_pat.search(v) for v in row) and any(name_pat.search(v) for v in row):
            return i
    return None

def parse_table(df_raw: pd.DataFrame) -> pd.DataFrame:
    hdr = locate_header_row(df_raw)
    if hdr is None:
        # dump a tiny hint for debugging
        head_preview = df_raw.head(12).astype(str).to_string(index=False)
        raise RuntimeError(
            "HKEX xls: cannot locate header row.\n"
            "Expected columns include 'Stock Code' and 'English Stock Short Name'.\n"
            f"Preview:\n{head_preview}"
        )

    cols = [_clean_cell(x) for x in df_raw.iloc[hdr].tolist()]
    df = df_raw.iloc[hdr + 1 :].copy()
    df.columns = cols
    df = df.dropna(how="all")
    return df

def find_cols(df: pd.DataFrame) -> tuple[str, str]:
    """
    Find the actual column names robustly using regex.
    """
    col_code = next((c for c in df.columns if re.search(r"stock\s*code", str(c), re.I)), None)
    col_name = next((c for c in df.columns if re.search(r"english\s*stock\s*short\s*name", str(c), re.I)), None)
    if not col_code or not col_name:
        raise RuntimeError(f"HKEX xls: cannot find required columns. columns={list(df.columns)}")
    return col_code, col_name

def clean_equities(df: pd.DataFrame) -> List[Tuple[str, str]]:
    col_code, col_name = find_cols(df)

    base = df[[col_code, col_name]].copy()
    base["raw_code"] = base[col_code].astype(str)
    base["name"] = base[col_name].astype(str).map(_clean_cell)
    base["code5"] = base["raw_code"].map(normalize_code5_any)

    # 1) invalid code
    base = base[base["code5"].str.fullmatch(r"\d{5}", na=False)].copy()

    # 2) filter non-equities by keywords (same idea as your Colab)
    bad_kw = r"CBBC|WARRANT|RIGHTS|ETF|ETN|REIT|BOND|NOTE|PREF|PREFERENCE|TRUST|FUND|DERIV|牛熊|權證|輪證|房託|債"
    base = base[~base["name"].str.contains(bad_kw, case=False, regex=True, na=False)].copy()

    # 3) de-dup
    base = base.drop_duplicates(subset=["code5"], keep="first")

    rows = list(zip(base["code5"].tolist(), base["name"].tolist()))
    return rows

# ------------------------------------------------------------
# Public API
# ------------------------------------------------------------
def fetch_hkex_list(url: str = HKEX_URL) -> List[Tuple[str, str]]:
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    df_raw = pd.read_excel(io.BytesIO(r.content), header=None)
    df_tbl = parse_table(df_raw)
    return clean_equities(df_tbl)