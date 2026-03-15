# core/jpx_list.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import io
import os
import subprocess
import sys
from typing import List, Tuple, Optional

import pandas as pd
import requests


DEFAULT_JPX_LIST_URL = (
    "https://www.jpx.co.jp/english/markets/statistics-equities/misc/"
    "tvdivq0000001vg2-att/data_e.xls"
)


def _ensure_xls_reader() -> None:
    """
    JPX 提供 .xls，pandas 讀取通常需要 xlrd。
    在 CI/乾淨環境可能沒裝 → 嘗試自動安裝（失敗也不炸）。
    """
    try:
        import xlrd  # noqa: F401
        return
    except Exception:
        pass

    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "xlrd"], check=False)
    except Exception:
        pass


def _is_tokyo_pro_market(product: str) -> bool:
    p = (product or "").strip().lower()
    return ("tokyo pro market" in p) or ("pro market" in p)


def fetch_jpx_list(
    *,
    list_url: Optional[str] = None,
    include_tokyo_pro: bool = False,
    timeout_sec: int = 45,
) -> List[Tuple[str, str]]:
    """
    Return:
      [( "7203.T", "TOYOTA MOTOR CORP" ), ...]
    """
    _ensure_xls_reader()

    url = (list_url or os.getenv("JP_LIST_URL") or DEFAULT_JPX_LIST_URL).strip()
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Referer": "https://www.jpx.co.jp/english/markets/statistics-equities/misc/01.html",
    }

    r = requests.get(url, headers=headers, timeout=timeout_sec)
    r.raise_for_status()

    df = pd.read_excel(io.BytesIO(r.content))

    C_CODE = "Local Code"
    C_NAME = "Name (English)"
    C_PROD = "Section/Products"

    out: List[Tuple[str, str]] = []

    for _, row in df.iterrows():
        raw_code = row.get(C_CODE)
        if pd.isna(raw_code):
            continue

        code = str(raw_code).split(".")[0].strip()
        if not (len(code) == 4 and code.isdigit()):
            continue

        product = str(row.get(C_PROD, "")).strip()

        # ETFs 排除（跟你 JP 版一致）
        if product.lower().startswith("etfs"):
            continue

        # Tokyo Pro Market 預設排除
        if (not include_tokyo_pro) and _is_tokyo_pro_market(product):
            continue

        symbol = f"{code}.T"
        name = str(row.get(C_NAME, "")).strip() or "Unknown"
        out.append((symbol, name))

    return out