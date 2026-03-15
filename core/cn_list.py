# core/cn_list.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
from typing import List, Tuple, Optional

import pandas as pd


# A-share valid prefixes (跟你現有 cn_stock_list 的邏輯一致)
_VALID_PREFIXES = (
    "000", "001", "002", "003",
    "300", "301",
    "600", "601", "603", "605",
    "688",
)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "y", "on")


def _normalize_code_name_df(df: pd.DataFrame) -> Tuple[str, str]:
    code_col = "code" if "code" in df.columns else ("代码" if "代码" in df.columns else None)
    name_col = "name" if "name" in df.columns else ("名称" if "名称" in df.columns else None)
    if not code_col or not name_col:
        raise RuntimeError(f"unexpected columns: {list(df.columns)}")
    return code_col, name_col


def _to_yahoo_ticker(code6: str) -> str:
    """
    6xxxxxx -> .SS (Shanghai)
    others  -> .SZ (Shenzhen)
    """
    c = str(code6).zfill(6)
    return f"{c}.SS" if c.startswith("6") else f"{c}.SZ"


def fetch_cn_a_share_list(
    *,
    timeout_sec: int = 30,
    include_bj: Optional[bool] = None,
) -> List[Tuple[str, str]]:
    """
    Return:
      [( "600519.SS", "贵州茅台" ), ( "000001.SZ", "平安银行" ), ...]
    Data source:
      - Prefer ak.stock_info_a_code_name()
      - Fallback ak.stock_zh_a_spot_em()

    NOTE:
      - Default excludes Beijing Stock Exchange (BJ) because Yahoo suffix differs and
        your existing CN pipeline focuses on .SS/.SZ.
      - You can override with env CN_INCLUDE_BJ=1 (still best-effort).
    """
    if include_bj is None:
        include_bj = _env_bool("CN_INCLUDE_BJ", False)

    # import here so repo users without akshare can see a clear error
    try:
        import akshare as ak  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "akshare is required for CN universe. Install with: pip install akshare"
        ) from e

    # 1) prefer code_name table
    try:
        df = ak.stock_info_a_code_name()
        code_col, name_col = _normalize_code_name_df(df)

        out: List[Tuple[str, str]] = []
        for _, row in df.iterrows():
            code = str(row.get(code_col, "")).zfill(6)
            if not code.startswith(_VALID_PREFIXES):
                continue

            # BJ usually starts with 8/4? (varies). Default exclude.
            if (not include_bj) and (code.startswith("8") or code.startswith("4")):
                continue

            ticker = _to_yahoo_ticker(code)
            name = str(row.get(name_col, "")).strip() or "Unknown"
            out.append((ticker, name))

        return out
    except Exception:
        pass

    # 2) fallback: spot_em
    df2 = ak.stock_zh_a_spot_em()
    code_col = "代码" if "代码" in df2.columns else ("code" if "code" in df2.columns else None)
    name_col = "名称" if "名称" in df2.columns else ("name" if "name" in df2.columns else None)
    if not code_col or not name_col:
        raise RuntimeError(f"unexpected columns from stock_zh_a_spot_em: {list(df2.columns)}")

    out2: List[Tuple[str, str]] = []
    for _, row in df2.iterrows():
        code = str(row.get(code_col, "")).zfill(6)
        if not code.startswith(_VALID_PREFIXES):
            continue
        if (not include_bj) and (code.startswith("8") or code.startswith("4")):
            continue

        ticker = _to_yahoo_ticker(code)
        name = str(row.get(name_col, "")).strip() or "Unknown"
        out2.append((ticker, name))

    return out2