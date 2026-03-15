# core/au_list.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
from typing import List, Tuple, Optional

import pandas as pd


ASX_LIST_URL_DEFAULT = "https://www.asx.com.au/asx/research/ASXListedCompanies.csv"

# 你原本的排除策略（保守排除非普通股/基金/權證/信託等）
EXCLUDE_SECTOR_KEYWORDS = [
    "ETF", "ETP", "Fund", "Structured", "Warrant", "Option", "Note", "Bond", "Debenture",
    "Trust", "REIT", "Mortgage",
    "Closed-End", "Closed End", "Not Applic", "Not Applicable", "Class Pend",
]


def _bool_env(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    s = str(v).strip().lower()
    if s in ("1", "true", "yes", "y", "on"):
        return True
    if s in ("0", "false", "no", "n", "off"):
        return False
    return default


def _read_asx_list_csv(url: str, local_path: Optional[str] = None) -> pd.DataFrame:
    """
    ASXListedCompanies.csv 常見格式：
      line0: "ASX listed companies as at ..."
      line1: empty
      line2: header
    所以用自動探測避免 pandas 讀成單欄。
    """
    src = local_path.strip() if (local_path or "").strip() else url

    # 先嘗試正常讀
    try:
        df0 = pd.read_csv(src)
        if df0.shape[1] >= 3:
            return df0
    except Exception:
        pass

    # 常見情況：skip 2 行
    for skip in (2, 1, 3, 0, 4, 5):
        try:
            df = pd.read_csv(src, skiprows=skip)
            if df.shape[1] >= 3:
                return df
        except Exception:
            continue

    # 最後兜底
    return pd.read_csv(src, engine="python")


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    盡量兼容不同欄位名：
      - Company name / Company Name / Company
      - ASX code / ASX Code / Code
      - GICS industry group / GICS Industry Group / Sector
    """
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    def pick(*cands: str) -> Optional[str]:
        for c in cands:
            if c in df.columns:
                return c
        lower_map = {c.lower(): c for c in df.columns}
        for c in cands:
            if c.lower() in lower_map:
                return lower_map[c.lower()]
        return None

    c_company = pick("Company name", "Company Name", "Company")
    c_code = pick("ASX code", "ASX Code", "Code")
    c_sector = pick("GICS industry group", "GICS Industry Group", "Sector")

    # 欄位名太怪就用前三欄兜底
    if not (c_company and c_code and c_sector):
        if df.shape[1] >= 3:
            out = df.iloc[:, :3].copy()
            out.columns = ["Company", "Code", "Sector"]
            return out
        raise ValueError(f"ASX list csv columns not recognized: {df.columns.tolist()}")

    out = df[[c_company, c_code, c_sector]].copy()
    out.columns = ["Company", "Code", "Sector"]
    return out


def get_au_universe(limit: int = 0) -> List[Tuple[str, str]]:
    """
    Return [(yfinance_ticker, name), ...] for AU.

    Env:
    - AU_LIST_URL: override official URL
    - AU_LIST_CSV_PATH: read local csv instead of downloading
    - AU_INCLUDE_REITS: 1/0 (default 0)
    """
    url = (os.getenv("AU_LIST_URL") or "").strip() or ASX_LIST_URL_DEFAULT
    local_path = (os.getenv("AU_LIST_CSV_PATH") or "").strip() or None
    include_reits = _bool_env("AU_INCLUDE_REITS", False)

    df_raw = _read_asx_list_csv(url=url, local_path=local_path)
    df = _normalize_columns(df_raw)

    # clean
    df["Company"] = df["Company"].astype(str).str.strip()
    df["Code"] = df["Code"].astype(str).str.strip()
    df["Sector"] = df["Sector"].astype(str).str.strip()

    # drop empty code
    df = df[df["Code"].astype(str).str.len() > 0].copy()

    # REIT flag
    df["is_reit"] = df["Sector"].str.contains(r"\bREIT\b", case=False, na=False)

    # filter out non-common-stock instruments
    mask_excl = pd.Series(False, index=df.index)
    for kw in EXCLUDE_SECTOR_KEYWORDS:
        mask_excl = mask_excl | df["Sector"].str.contains(str(kw), case=False, na=False)

    # 如果允許 REIT，就把 REIT 排除條件拿掉（只解除 REIT 類）
    if include_reits:
        mask_excl = mask_excl & (~df["is_reit"])

    df2 = df[~mask_excl].copy()

    out: List[Tuple[str, str]] = []
    for _, r in df2.iterrows():
        code = str(r["Code"]).strip().upper()
        name = str(r["Company"]).strip() or code
        if not code:
            continue
        ticker = code if code.endswith(".AX") else f"{code}.AX"
        out.append((ticker, name))
        if limit > 0 and len(out) >= limit:
            break

    return out