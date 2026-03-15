# core/krx_list.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
from io import StringIO
from typing import List, Tuple, Optional

import pandas as pd
import requests


DEFAULT_KRX_LIST_URL = "https://kind.krx.co.kr/corpgeneral/corpList.do?method=download&searchType=13"


def _normalize_code(v) -> str:
    s = str(v).strip()
    s = s.split(".")[0].strip()
    s = s.upper()
    s = "".join(ch for ch in s if ch.isalnum())
    if len(s) > 6:
        s = s[-6:]
    return s.rjust(6, "0")


def _to_yahoo_symbol(code6: str, market: str) -> Optional[str]:
    m = (market or "").strip().upper()
    if m in ("KOSPI", "유가증권"):
        return f"{code6}.KS"
    if m in ("KOSDAQ", "코스닥"):
        return f"{code6}.KQ"
    if m in ("KONEX", "코넥스"):
        return None  # KONEX 排除（跟你原本一致）
    # fallback
    return f"{code6}.KS"


def _fetch_krx_corplist_html(url: str, *, timeout_sec: int = 45) -> str:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
        ),
        "Referer": "https://kind.krx.co.kr/",
        "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Connection": "close",
    }

    resp = requests.get(url, headers=headers, timeout=timeout_sec)
    resp.raise_for_status()
    content = resp.content

    # KIND 常見編碼
    for enc in ("euc-kr", "cp949", "utf-8"):
        try:
            return content.decode(enc)
        except Exception:
            continue
    return content.decode("euc-kr", errors="replace")


def fetch_krx_list(
    *,
    list_url: Optional[str] = None,
    timeout_sec: int = 45,
) -> List[Tuple[str, str]]:
    """
    Return:
      [("005930.KS", "삼성전자"), ("035420.KS", "NAVER"), ("091990.KQ", "셀트리온헬스케어"), ...]
    Notes:
      - KOSPI -> .KS
      - KOSDAQ -> .KQ
      - KONEX excluded
    """
    url = (list_url or os.getenv("KR_LIST_URL") or DEFAULT_KRX_LIST_URL).strip()

    html = _fetch_krx_corplist_html(url, timeout_sec=timeout_sec)
    tables = pd.read_html(StringIO(html))
    if not tables:
        return []

    df = tables[0].copy()

    name_col = "회사명" if "회사명" in df.columns else ("Company Name" if "Company Name" in df.columns else None)
    code_col = "종목코드" if "종목코드" in df.columns else ("Stock Code" if "Stock Code" in df.columns else None)
    mkt_col = "시장구분" if "시장구분" in df.columns else ("Market" if "Market" in df.columns else None)

    if not (name_col and code_col and mkt_col):
        # 欄位若變動：回空（讓上層決定要不要 failfast）
        return []

    out: List[Tuple[str, str]] = []
    for _, r in df.iterrows():
        code6 = _normalize_code(r.get(code_col, ""))
        if len(code6) != 6:
            continue

        name = str(r.get(name_col, "")).strip() or "Unknown"
        market_raw = str(r.get(mkt_col, "")).strip()

        sym = _to_yahoo_symbol(code6, market_raw)
        if not sym:
            continue

        out.append((sym, name))

    return out