# core/fr_list.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import re
import time
import unicodedata
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
from bs4 import BeautifulSoup


# =============================================================================
# Defaults
# =============================================================================
DEFAULT_SA_PAGES = 2
DEFAULT_MS_PAGES = 11
DEFAULT_HTTP_TIMEOUT = 20
DEFAULT_SLEEP_SEC = 2.0
DEFAULT_FUZZY_CUTOFF = 0.90

DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/121.0.0.0 Safari/537.36"
)

SA_BASE = "https://stockanalysis.com/list/euronext-paris/?page={p}"
MS_BASE = "https://uk.marketscreener.com/stock-exchange/shares/europe/france-51/?p={p}"
MS_HOME = "https://uk.marketscreener.com/"


# =============================================================================
# Config helpers
# =============================================================================
def _cfg_int(cfg: Optional[Dict[str, Any]], key: str, default: int) -> int:
    if cfg is None:
        return default
    try:
        return int(cfg.get(key, default))
    except Exception:
        return default


def _cfg_float(cfg: Optional[Dict[str, Any]], key: str, default: float) -> float:
    if cfg is None:
        return default
    try:
        return float(cfg.get(key, default))
    except Exception:
        return default


def _cfg_str(cfg: Optional[Dict[str, Any]], key: str, default: str = "") -> str:
    if cfg is None:
        return default
    v = cfg.get(key, default)
    if v is None:
        return default
    s = str(v).strip()
    return s if s else default


def _headers(cfg: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
    ua = _cfg_str(cfg, "user_agent", DEFAULT_USER_AGENT)
    return {
        "User-Agent": ua,
        "Accept": "text/html,application/xhtml+xml,*/*;q=0.9",
        "Accept-Language": "en-US,en;q=0.9",
    }


# =============================================================================
# Normalization
# =============================================================================
LEGAL_SUFFIXES = [
    "S.A.", "SA", "SE", "SOCIETE ANONYME", "SOCIÉTÉ ANONYME",
    "SCA", "S.C.A.", "NV", "N.V.", "PLC", "LTD", "LIMITED",
    "GROUPE", "GROUP",
]


def strip_accents(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in s if not unicodedata.combining(ch))


def norm_name(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return ""
    s = strip_accents(s)
    s = s.upper()
    s = re.sub(r"\([^)]*\)", " ", s)
    s = s.replace("’", "'")

    for suf in LEGAL_SUFFIXES:
        suf2 = strip_accents(suf).upper()
        s = re.sub(rf"\b{re.escape(suf2)}\b", " ", s)

    s = re.sub(r"[^A-Z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def sim(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


# =============================================================================
# stockanalysis: symbol + company
# =============================================================================
def scrape_stockanalysis(cfg: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    sa_pages = _cfg_int(cfg, "sa_pages", DEFAULT_SA_PAGES)
    timeout = _cfg_int(cfg, "http_timeout_sec", DEFAULT_HTTP_TIMEOUT)
    sleep_sec = _cfg_float(cfg, "sleep_sec", DEFAULT_SLEEP_SEC)
    headers = _headers(cfg)

    all_records: List[Dict[str, str]] = []

    for p in range(1, sa_pages + 1):
        url = SA_BASE.format(p=p)
        r = requests.get(url, headers=headers, timeout=timeout)
        r.encoding = "utf-8"
        r.raise_for_status()

        soup = BeautifulSoup(r.text, "html.parser")
        table = soup.find("table")
        if not table:
            raise RuntimeError(f"[FR] stockanalysis no table found: page={p}")

        rows = table.find_all("tr")
        if not rows:
            continue

        header = [th.get_text(strip=True) for th in rows[0].find_all(["th", "td"])]
        for row in rows[1:]:
            cells = row.find_all(["td", "th"])
            vals = [c.get_text(strip=True) for c in cells]
            if not any(vals):
                continue
            rec = dict(zip(header, vals))
            all_records.append(rec)

        if p < sa_pages:
            time.sleep(sleep_sec)

    df = pd.DataFrame(all_records)

    rename_map = {
        "No.": "no",
        "Symbol": "symbol",
        "Company Name": "company_name",
        "Market Cap": "market_cap",
        "Stock Price": "stock_price",
        "% Change": "pct_change",
        "Revenue": "revenue",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    if "symbol" not in df.columns or "company_name" not in df.columns:
        raise RuntimeError(f"[FR] stockanalysis schema changed. cols={list(df.columns)}")

    df["symbol"] = df["symbol"].astype(str).str.strip().str.upper()
    df["yf_symbol"] = df["symbol"].apply(lambda s: f"{s}.PA" if s and s != "NAN" else "")
    df["ticker"] = df["symbol"]
    df["company_name"] = df["company_name"].astype(str).str.strip()
    df["name_key"] = df["company_name"].apply(norm_name)

    df = df[df["symbol"].str.len() > 0].copy()
    df = df.drop_duplicates(subset=["symbol"]).reset_index(drop=True)
    return df


# =============================================================================
# marketscreener: sector
# =============================================================================
def _ms_full_url(href: str) -> str:
    if not href:
        return ""
    if href.startswith("http://") or href.startswith("https://"):
        return href
    if href.startswith("/"):
        return MS_HOME.rstrip("/") + href
    return MS_HOME.rstrip("/") + "/" + href


def _find_first_table(soup: BeautifulSoup):
    for cls in ["std_tlist", "table", "tlist", "screener-table"]:
        t = soup.find("table", {"class": cls})
        if t:
            return t
    tables = soup.find_all("table")
    return tables[0] if tables else None


def scrape_marketscreener_sector(cfg: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    ms_pages = _cfg_int(cfg, "ms_pages", DEFAULT_MS_PAGES)
    timeout = _cfg_int(cfg, "http_timeout_sec", DEFAULT_HTTP_TIMEOUT)
    sleep_sec = _cfg_float(cfg, "sleep_sec", DEFAULT_SLEEP_SEC)
    headers = {**_headers(cfg), "Referer": MS_HOME}

    rows_out: List[Dict[str, str]] = []

    for p in range(1, ms_pages + 1):
        url = MS_BASE.format(p=p)
        r = requests.get(url, headers=headers, timeout=timeout)
        r.encoding = "utf-8"
        r.raise_for_status()

        soup = BeautifulSoup(r.text, "html.parser")
        table = _find_first_table(soup)
        if not table:
            break

        trs = table.find_all("tr")
        if len(trs) < 2:
            break

        header = [th.get_text(strip=True) for th in trs[0].find_all(["th", "td"])]
        header_lower = [h.lower() for h in header]

        sector_idx = None
        for i, h in enumerate(header_lower):
            if "sector" in h:
                sector_idx = i
                break

        for tr in trs[1:]:
            tds = tr.find_all(["td", "th"])
            vals = [td.get_text(strip=True) for td in tds]
            if not any(vals):
                continue

            a = tr.find("a", href=True)
            ms_link = _ms_full_url(a["href"]) if a else ""

            name = ""
            for v in vals:
                if v and "Add to a list" not in v:
                    name = v
                    break

            sector = ""
            if sector_idx is not None and sector_idx < len(vals):
                sector = vals[sector_idx]

            name = (name or "").strip()
            sector = (sector or "").strip()
            if not name:
                continue

            rows_out.append(
                {
                    "ms_name": name,
                    "ms_sector": sector,
                    "ms_link": ms_link,
                    "ms_name_key": norm_name(name),
                }
            )

        if p < ms_pages:
            time.sleep(sleep_sec)

    df = pd.DataFrame(rows_out)
    if df.empty:
        return pd.DataFrame(columns=["ms_name", "ms_sector", "ms_link", "ms_name_key"])

    df = df.drop_duplicates(subset=["ms_name_key"], keep="first").reset_index(drop=True)
    return df


# =============================================================================
# Merge
# =============================================================================
def merge_sa_ms(df_sa: pd.DataFrame, df_ms: pd.DataFrame, cfg: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    fuzzy_cutoff = _cfg_float(cfg, "fuzzy_cutoff", DEFAULT_FUZZY_CUTOFF)
    df = df_sa.copy()

    if df_ms is None or df_ms.empty:
        df["sector"] = ""
        return df

    ms_map = dict(zip(df_ms["ms_name_key"], df_ms["ms_sector"]))
    df["sector"] = df["name_key"].map(lambda k: ms_map.get(k, "")).fillna("")

    unmatched = df["sector"].astype(str).str.strip().eq("")
    unmatched_idx = df.index[unmatched].tolist()
    if not unmatched_idx:
        return df

    ms_keys = df_ms["ms_name_key"].astype(str).tolist()
    ms_sector_map = dict(zip(df_ms["ms_name_key"], df_ms["ms_sector"]))

    for i in unmatched_idx:
        key = str(df.at[i, "name_key"] or "")
        if not key:
            continue

        first = key.split(" ")[0]
        candidates = [k for k in ms_keys if k.startswith(first)] or ms_keys

        best_k = ""
        best_s = 0.0
        for k in candidates:
            sc = sim(key, k)
            if sc > best_s:
                best_s = sc
                best_k = k

        if best_s >= fuzzy_cutoff and best_k:
            df.at[i, "sector"] = ms_sector_map.get(best_k, "")

    return df


# =============================================================================
# Public API
# =============================================================================
def build_fr_master_df(cfg: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    df_sa = scrape_stockanalysis(cfg)
    df_ms = scrape_marketscreener_sector(cfg)
    df = merge_sa_ms(df_sa, df_ms, cfg)

    keep_cols = [
        "ticker",
        "yf_symbol",
        "symbol",
        "company_name",
        "sector",
        "market_cap",
        "stock_price",
        "pct_change",
        "revenue",
        "name_key",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    return df[keep_cols].copy()


def fetch_fr_list(cfg: Optional[Dict[str, Any]] = None) -> List[Tuple[str, str]]:
    """
    Return:
      [("AIR.PA", "AIRBUS SE"), ...]
    """
    df = build_fr_master_df(cfg)

    if df.empty:
        return []

    yf_col = "yf_symbol" if "yf_symbol" in df.columns else "symbol"
    name_col = "company_name" if "company_name" in df.columns else "symbol"

    df[yf_col] = df[yf_col].astype(str).str.strip().str.upper()
    df[name_col] = df[name_col].astype(str).str.strip()

    df = df[df[yf_col].str.len() > 0].copy()
    df = df.drop_duplicates(subset=[yf_col]).reset_index(drop=True)

    out: List[Tuple[str, str]] = []
    for _, r in df.iterrows():
        yf_symbol = str(r.get(yf_col) or "").strip().upper()
        company_name = str(r.get(name_col) or yf_symbol).strip() or yf_symbol
        if not yf_symbol:
            continue
        out.append((yf_symbol, company_name))

    return out