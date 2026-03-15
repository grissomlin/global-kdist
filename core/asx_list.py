# core/asx_list.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple, Optional

import pandas as pd


ASX_LIST_URL_DEFAULT = "https://www.asx.com.au/asx/research/ASXListedCompanies.csv"


def _read_asx_csv(src: str) -> pd.DataFrame:
    # ASX 檔案常有前兩行文字，做一個保守探測
    for skip in (0, 2, 1, 3, 4, 5):
        try:
            df = pd.read_csv(src, skiprows=skip)
            if df.shape[1] >= 3:
                return df
        except Exception:
            continue
    return pd.read_csv(src, engine="python")


def _normalize_asx_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["symbol", "name"])

    cols = {str(c).strip().lower(): c for c in df.columns}

    def pick(*names: str) -> Optional[str]:
        for n in names:
            if n in cols:
                return cols[n]
        return None

    c_company = pick("company name", "companyname", "company")
    c_code = pick("asx code", "asxcode", "code")

    if not (c_company and c_code):
        # fallback to first two columns
        df2 = df.iloc[:, :2].copy()
        df2.columns = ["Company", "Code"]
        c_company, c_code = "Company", "Code"
        df = df2

    out = pd.DataFrame()
    out["name"] = df[c_company].astype(str).str.strip()
    out["symbol"] = df[c_code].astype(str).str.strip() + ".AX"

    out = out[out["symbol"].astype(str).str.len() > 3].copy()
    out["name"] = out["name"].replace("", "Unknown")
    out = out.drop_duplicates(subset=["symbol"]).reset_index(drop=True)
    return out[["symbol", "name"]]


def get_au_universe(limit: int = 0) -> List[Tuple[str, str]]:
    """
    Return [(symbol, name), ...] for AU (ASX).
    Source priority:
      1) AU_LIST_CSV_PATH (local cache)
      2) ASX official URL (download live)
    """
    local = (os.getenv("AU_LIST_CSV_PATH") or "").strip()
    src = local if local else (os.getenv("AU_LIST_URL") or ASX_LIST_URL_DEFAULT)

    df = _read_asx_csv(src)
    df = _normalize_asx_df(df)

    rows = [(str(r.symbol), str(r.name)) for r in df.itertuples(index=False)]
    if limit and limit > 0:
        rows = rows[:limit]
    return rows