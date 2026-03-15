# core/download_dayk.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Union, List

import pandas as pd
import yfinance as yf
from tqdm import tqdm

from core.cleaning import (
    standardize_history,
    clean_ohlc,
    should_apply_uk_scale,
    normalize_uk_scale,
)

# =============================================================================
# Helpers
# =============================================================================
def _safe_filename(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return "UNKNOWN"
    s = re.sub(r"[^\w\.\-]+", "_", s)
    return s[:200]


def _as_symbol_name(row: Any) -> Tuple[str, str]:
    if isinstance(row, dict):
        sym = str(row.get("symbol") or row.get("id") or row.get("ticker") or "").strip()
        name = str(row.get("name") or row.get("issuer") or row.get("company") or sym).strip()
        return sym, (name or sym)
    if isinstance(row, (tuple, list)):
        sym = str(row[0]).strip() if len(row) >= 1 else ""
        name = str(row[1]).strip() if len(row) >= 2 else sym
        return sym, (name or sym)
    if isinstance(row, str):
        sym = row.strip()
        return sym, sym
    sym = str(row).strip()
    return sym, sym


def _env_bool(name: str, default: str = "0") -> bool:
    return str(os.getenv(name, default)).strip().lower() in ("1", "true", "yes", "y", "on")


def _env_int(name: str, default: str) -> int:
    try:
        return int(str(os.getenv(name, default)).strip())
    except Exception:
        return int(default)


def _rotate_errors_if_needed(out_dir: Path) -> None:
    """
    Rotate _errors.txt to _errors_YYYYMMDD_HHMMSS.txt each run (optional but recommended).
    Controlled by env: DAYK_ROTATE_ERRORS (default=1)
    """
    if not _env_bool("DAYK_ROTATE_ERRORS", "1"):
        return
    p = out_dir / "_errors.txt"
    if not p.exists() or p.stat().st_size <= 0:
        return
    ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    dst = out_dir / f"_errors_{ts}.txt"
    try:
        p.replace(dst)
    except Exception:
        pass


def _write_error(out_dir: Path, sym: str, ticker: str, msg: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "_errors.txt").open("a", encoding="utf-8") as f:
        f.write(f"{sym}\t{ticker}\t{msg}\n")


# =============================================================================
# No-data / delisted detection + blacklist
# =============================================================================
_NO_DATA_PATTERNS = [
    "no data found",
    "quote not found",
    "possibly delisted",
    "http error 404",
    "404",
    "not found",
    "no timezone found",
]


def _is_no_data_error(msg: str) -> bool:
    s = (msg or "").strip().lower()
    if not s:
        return False
    return any(p in s for p in _NO_DATA_PATTERNS)


def _blacklist_path(out_dir: Path, market_code: Optional[str]) -> Path:
    mc = (market_code or "market").strip().lower()
    return out_dir / f"_{mc}_no_data.tsv"


def _load_blacklist(out_dir: Path, market_code: Optional[str]) -> Dict[str, str]:
    """
    Return {TICKER: reason}
    """
    p = _blacklist_path(out_dir, market_code)
    if not p.exists():
        return {}
    out: Dict[str, str] = {}
    try:
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t", 1)
            t = parts[0].strip().upper()
            reason = parts[1].strip() if len(parts) > 1 else ""
            if t:
                out[t] = reason
    except Exception:
        return {}
    return out


def _append_blacklist(out_dir: Path, market_code: Optional[str], ticker: str, reason: str) -> None:
    p = _blacklist_path(out_dir, market_code)
    try:
        with p.open("a", encoding="utf-8") as f:
            f.write(f"{ticker.upper()}\t{reason}\n")
    except Exception:
        pass


def _confirm_no_data_quick(ticker: str, timeout: int = 30) -> bool:
    """
    Cheap confirmation: try 7d daily data once.
    If still empty/no close -> treat as no-data.
    """
    try:
        df = yf.download(
            ticker,
            period="7d",
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False,
            timeout=timeout,
        )
        df_std = standardize_history(df)
        df_std, _ = clean_ohlc(df_std, allow_zero_volume=True)
        if df_std.empty:
            return True
        if "close" not in df_std.columns:
            return True
        if pd.to_numeric(df_std["close"], errors="coerce").notna().sum() == 0:
            return True
        return False
    except Exception as e:
        if _is_no_data_error(f"{type(e).__name__}: {e}"):
            return True
        return False


# =============================================================================
# Batch download helpers
# =============================================================================
def _chunk(lst: List[str], n: int) -> List[List[str]]:
    if n <= 0:
        return [lst]
    return [lst[i: i + n] for i in range(0, len(lst), n)]


def _yf_download_batch(
    tickers: List[str],
    *,
    start: Optional[str],
    end: Optional[str],
    period: Optional[str],
    timeout: int = 60,
    threads: bool = True,
) -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame()

    kwargs: Dict[str, Any] = dict(
        tickers=" ".join(tickers),
        interval="1d",
        group_by="ticker",
        auto_adjust=True,
        progress=False,
        threads=threads,
        timeout=timeout,
    )

    # start/end 優先於 period
    if start or end:
        if start:
            kwargs["start"] = start
        if end:
            kwargs["end"] = end
    else:
        kwargs["period"] = period or "5y"

    df = yf.download(**kwargs)
    return df if df is not None else pd.DataFrame()


def _extract_one_from_batch(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    if not isinstance(df.columns, pd.MultiIndex):
        out = df.copy()
        out.index.name = "Date"
        return out

    cols0 = [c[0] for c in df.columns]
    cols1 = [c[1] for c in df.columns]

    # (Field, Ticker)
    if ticker in set(cols1):
        sub = df.xs(ticker, axis=1, level=1, drop_level=True)
        sub.index.name = "Date"
        return sub

    # (Ticker, Field)
    if ticker in set(cols0):
        sub = df.xs(ticker, axis=1, level=0, drop_level=True)
        sub.index.name = "Date"
        return sub

    return pd.DataFrame()


# =============================================================================
# Public API
# =============================================================================
def download_all(
    universe: Iterable[Any],
    to_ticker: Callable[[Any], str],
    out_dir: Union[str, Path],
    *,
    start: Optional[str] = None,
    end: Optional[str] = None,
    period: str = "5y",
    market_code: Optional[str] = None,
    batch_size: int = 200,
    batch_threads: bool = True,
    batch_timeout: int = 60,
    batch_sleep_sec: float = 0.05,
    fallback_single: bool = True,
    single_sleep_sec: float = 0.02,
    max_retries: int = 2,
    sleep_sec: float = 0.0,
) -> Dict[str, int]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # normalize blank strings
    start = str(start).strip() if start is not None else None
    end = str(end).strip() if end is not None else None
    start = start or None
    end = end or None

    _rotate_errors_if_needed(out_dir)

    use_blacklist = _env_bool("DAYK_USE_NO_DATA_BLACKLIST", "1")
    blacklist_on_confirmed = _env_bool("DAYK_BLACKLIST_ON_CONFIRMED_NO_DATA", "1")
    blacklist_on_empty = _env_bool("DAYK_BLACKLIST_ON_EMPTY_OR_NO_CLOSE", "0")
    no_data_no_retry = _env_bool("DAYK_NO_DATA_NO_RETRY", "1")

    fallback_max_retries = _env_int("DAYK_FALLBACK_MAX_RETRIES", "1")

    quick_confirm_enabled = _env_bool("DAYK_QUICK_CONFIRM_NO_DATA", "1")
    quick_confirm_timeout = _env_int("DAYK_QUICK_CONFIRM_TIMEOUT", "30")

    uk_debug = _env_bool("UK_SCALE_DEBUG", "0")
    uk_down_total = 0
    uk_up_total = 0

    blacklist = _load_blacklist(out_dir, market_code) if use_blacklist else {}

    rows = list(universe)
    total = len(rows)

    ticker_list: List[str] = []
    sym_by_ticker: Dict[str, str] = {}

    for row in rows:
        sym, _name = _as_symbol_name(row)
        ticker = str(to_ticker(row)).strip()
        if not sym or not ticker:
            continue
        safe_sym = _safe_filename(sym)
        ticker_list.append(ticker)
        sym_by_ticker[ticker] = safe_sym

    need: List[str] = []
    skipped = 0
    skipped_blacklist = 0

    for ticker in ticker_list:
        sym = sym_by_ticker.get(ticker, "")
        if not sym:
            continue

        if use_blacklist and ticker.upper() in blacklist:
            skipped_blacklist += 1
            continue

        out_path = out_dir / f"{sym}.csv"
        if out_path.exists() and out_path.stat().st_size > 50:
            skipped += 1
            continue

        need.append(ticker)

    if not need:
        return {
            "total": total,
            "ok": 0,
            "fail": 0,
            "skipped": skipped,
            "skipped_blacklist": skipped_blacklist,
        }

    ok = 0
    fail = 0

    # ---------------------------------------
    # batch
    # ---------------------------------------
    batches = _chunk(need, batch_size)
    pbar = tqdm(batches, desc="Download dayK (batch)", unit="batch")

    missing: List[str] = []

    for batch in pbar:
        df_batch = pd.DataFrame()
        last_err: Optional[str] = None

        for attempt in range(max_retries + 1):
            try:
                df_batch = _yf_download_batch(
                    batch,
                    start=start,
                    end=end,
                    period=None if (start or end) else period,
                    timeout=batch_timeout,
                    threads=batch_threads,
                )
                last_err = None
                break
            except Exception as e:
                last_err = f"{type(e).__name__}: {e}"
                if no_data_no_retry and _is_no_data_error(last_err):
                    break
                if attempt < max_retries:
                    time.sleep(1.0)
                    continue

        if df_batch is None or df_batch.empty:
            msg = last_err or "batch_empty"
            for t in batch:
                sym = sym_by_ticker.get(t, t)
                _write_error(out_dir, sym, t, msg)
                missing.append(t)
            time.sleep(batch_sleep_sec)
            continue

        for t in batch:
            sym = sym_by_ticker.get(t, "")
            if not sym:
                continue

            if use_blacklist and t.upper() in blacklist:
                continue

            try:
                sub = _extract_one_from_batch(df_batch, t)
                df_std = standardize_history(sub)
                df_std, _st = clean_ohlc(df_std, allow_zero_volume=True)

                close_ok = (
                    (not df_std.empty)
                    and ("close" in df_std.columns)
                    and (pd.to_numeric(df_std["close"], errors="coerce").notna().sum() > 0)
                )

                if not close_ok:
                    if quick_confirm_enabled:
                        confirmed = _confirm_no_data_quick(t, timeout=quick_confirm_timeout)
                        if confirmed:
                            msg = "no_data_confirmed(batch_extract_empty)"
                            _write_error(out_dir, sym, t, msg)
                            fail += 1
                            if blacklist_on_confirmed and use_blacklist:
                                _append_blacklist(out_dir, market_code, t, msg)
                                blacklist[t.upper()] = msg
                            continue

                    missing.append(t)
                    continue

                if should_apply_uk_scale(market_code=market_code, ticker=t):
                    df_std2, n_down, n_up = normalize_uk_scale(df_std)
                    df_std = df_std2
                    uk_down_total += int(n_down)
                    uk_up_total += int(n_up)

                out_path = out_dir / f"{sym}.csv"
                df_std.to_csv(out_path, index=False, encoding="utf-8-sig")
                ok += 1

            except Exception as e:
                err = f"batch_extract_failed: {type(e).__name__}: {e}"
                _write_error(out_dir, sym, t, err)
                missing.append(t)

        time.sleep(batch_sleep_sec)

    # ---------------------------------------
    # fallback single
    # ---------------------------------------
    if fallback_single and missing:
        uniq_missing: List[str] = []
        seen = set()

        for t in missing:
            if t in seen:
                continue
            seen.add(t)

            sym = sym_by_ticker.get(t, "")
            if not sym:
                continue

            if use_blacklist and t.upper() in blacklist:
                continue

            out_path = out_dir / f"{sym}.csv"
            if out_path.exists() and out_path.stat().st_size > 50:
                continue

            uniq_missing.append(t)

        if uniq_missing:
            pbar2 = tqdm(uniq_missing, desc="Download dayK (fallback)", unit="ticker")

            for t in pbar2:
                sym = sym_by_ticker.get(t, t)

                if use_blacklist and t.upper() in blacklist:
                    continue

                last_err: Optional[str] = None
                df_std = pd.DataFrame()

                for attempt in range(fallback_max_retries + 1):
                    try:
                        if start or end:
                            df = yf.download(
                                t,
                                start=start,
                                end=end,
                                interval="1d",
                                auto_adjust=True,
                                progress=False,
                                threads=False,
                                timeout=45,
                            )
                        else:
                            df = yf.Ticker(t).history(period=period, auto_adjust=True)

                        df_std = standardize_history(df)
                        df_std, _st = clean_ohlc(df_std, allow_zero_volume=True)

                        close_ok = (
                            (not df_std.empty)
                            and ("close" in df_std.columns)
                            and (pd.to_numeric(df_std["close"], errors="coerce").notna().sum() > 0)
                        )

                        if not close_ok:
                            last_err = "empty_or_no_close"
                            if blacklist_on_empty and use_blacklist:
                                _append_blacklist(out_dir, market_code, t, last_err)
                                blacklist[t.upper()] = last_err
                            break

                        if should_apply_uk_scale(market_code=market_code, ticker=t):
                            df_std2, n_down, n_up = normalize_uk_scale(df_std)
                            df_std = df_std2
                            uk_down_total += int(n_down)
                            uk_up_total += int(n_up)

                        out_path = out_dir / f"{sym}.csv"
                        df_std.to_csv(out_path, index=False, encoding="utf-8-sig")
                        ok += 1
                        last_err = None
                        break

                    except Exception as e:
                        last_err = f"{type(e).__name__}: {e}"

                        if no_data_no_retry and _is_no_data_error(last_err):
                            if blacklist_on_confirmed and use_blacklist:
                                _append_blacklist(out_dir, market_code, t, last_err)
                                blacklist[t.upper()] = last_err
                            break

                        if attempt < fallback_max_retries:
                            time.sleep(1.0)
                            continue
                        break

                if last_err is not None:
                    fail += 1
                    _write_error(out_dir, sym, t, last_err)

                time.sleep(single_sleep_sec)

    if sleep_sec and sleep_sec > 0:
        time.sleep(float(sleep_sec))

    if uk_debug:
        print(
            f"[UK_SCALE_DEBUG] scaled_down(/100)={uk_down_total} scaled_up(*100)={uk_up_total} "
            f"(market_code={market_code})",
            flush=True,
        )

    return {
        "total": total,
        "ok": ok,
        "fail": fail,
        "skipped": skipped,
        "skipped_blacklist": skipped_blacklist,
    }