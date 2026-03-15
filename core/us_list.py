# markets/us_list.py
# -*- coding: utf-8 -*-
"""
US universe builder (no DB) for dayK downloader repo.

Source:
- Nasdaq Trader symbol directory:
  - nasdaqlisted.txt
  - otherlisted.txt

Filters:
- exclude ETF by default
- exclude test issues
- allow include_adr / include_etf / include_test via cfg or env-like cfg keys
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd
import requests


NASDAQ_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
OTHER_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"


def _cfg_bool(cfg: Dict[str, Any], key: str, default: bool) -> bool:
    v = cfg.get(key, default)
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("1", "true", "yes", "y", "on"):
        return True
    if s in ("0", "false", "no", "n", "off"):
        return False
    return default


def _download_text(url: str, timeout: int = 30) -> str:
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    # NasdaqTrader is plain text, mostly ASCII
    return r.text


def _read_pipe_text(txt: str) -> pd.DataFrame:
    """
    NasdaqTrader files are pipe-delimited with a trailing footer line like:
      "File Creation Time: ..."
    and sometimes an "EOF" line.

    We'll:
    - split lines
    - drop footer-ish lines
    - parse with pandas read_csv from string buffer
    """
    lines = []
    for ln in txt.splitlines():
        ln = ln.strip("\r\n")
        if not ln:
            continue
        if ln.startswith("File Creation Time:"):
            continue
        if ln == "EOF":
            continue
        lines.append(ln)

    from io import StringIO

    buf = StringIO("\n".join(lines))
    df = pd.read_csv(buf, sep="|")
    # sometimes the last column is unnamed due to trailing pipe; drop empties
    df = df.loc[:, [c for c in df.columns if str(c).strip() != ""]]
    return df


def _normalize_nasdaq_listed(df: pd.DataFrame) -> pd.DataFrame:
    """
    nasdaqlisted.txt columns typically include:
    Symbol, Security Name, Market Category, Test Issue, Financial Status, Round Lot Size, ETF, NextShares
    """
    df = df.copy()
    cols = {c.lower().strip(): c for c in df.columns}

    sym = cols.get("symbol")
    name = cols.get("security name") or cols.get("securityname")
    test = cols.get("test issue") or cols.get("testissue")
    etf = cols.get("etf")

    out = pd.DataFrame()
    out["symbol"] = df[sym].astype(str).str.strip() if sym else ""
    out["name"] = df[name].astype(str).str.strip() if name else ""
    out["exchange"] = "NASDAQ"
    out["is_test"] = df[test].astype(str).str.strip().str.upper().eq("Y") if test else False
    out["is_etf"] = df[etf].astype(str).str.strip().str.upper().eq("Y") if etf else False
    out["asset_type"] = "EQUITY"
    return out


def _normalize_other_listed(df: pd.DataFrame) -> pd.DataFrame:
    """
    otherlisted.txt columns typically include:
    ACT Symbol, Security Name, Exchange, CQS Symbol, ETF, Round Lot Size, Test Issue, NASDAQ Symbol
    """
    df = df.copy()
    cols = {c.lower().strip(): c for c in df.columns}

    sym = cols.get("act symbol") or cols.get("actsymbol")
    name = cols.get("security name") or cols.get("securityname")
    exch = cols.get("exchange")
    test = cols.get("test issue") or cols.get("testissue")
    etf = cols.get("etf")

    out = pd.DataFrame()
    out["symbol"] = df[sym].astype(str).str.strip() if sym else ""
    out["name"] = df[name].astype(str).str.strip() if name else ""
    out["exchange"] = df[exch].astype(str).str.strip().replace({"A": "AMEX", "N": "NYSE", "P": "ARCA", "Z": "BATS"}) if exch else ""
    out["is_test"] = df[test].astype(str).str.strip().str.upper().eq("Y") if test else False
    out["is_etf"] = df[etf].astype(str).str.strip().str.upper().eq("Y") if etf else False
    out["asset_type"] = "EQUITY"
    return out


def build_us_universe(cfg: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    cfg = cfg or {}

    include_etf = _cfg_bool(cfg, "include_etf", False)
    include_test = _cfg_bool(cfg, "include_test", False)

    # Download + parse
    t1 = _download_text(NASDAQ_LISTED_URL)
    t2 = _download_text(OTHER_LISTED_URL)

    df1 = _read_pipe_text(t1)
    df2 = _read_pipe_text(t2)

    u1 = _normalize_nasdaq_listed(df1)
    u2 = _normalize_other_listed(df2)

    uni = pd.concat([u1, u2], ignore_index=True)

    # Clean
    uni["symbol"] = uni["symbol"].astype(str).str.strip()
    uni = uni[uni["symbol"].str.len() > 0].copy()

    # Filters
    if not include_test:
        uni = uni[~uni["is_test"]].copy()
    if not include_etf:
        uni = uni[~uni["is_etf"]].copy()

    # Dedup
    uni = uni.drop_duplicates(subset=["symbol"]).reset_index(drop=True)

    # Optional: keep only common-ish tickers (avoid weird chars)
    # yfinance usually handles '.', '-' ok; but caret/space are bad.
    uni = uni[~uni["symbol"].str.contains(r"\s", regex=True, na=False)].copy()

    return uni.reset_index(drop=True)