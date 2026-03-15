# scripts/scan_daily_extreme_spikes.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data"
MARKETS_YAML = ROOT / "configs" / "markets.yaml"

KEEP_COLS = ["date", "open", "high", "low", "close", "volume"]


# =============================================================================
# Config / IO
# =============================================================================
def _load_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing config file: {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _load_markets_cfg() -> Dict[str, dict]:
    cfg = _load_yaml(MARKETS_YAML)
    markets = cfg.get("markets", {}) or {}
    if not isinstance(markets, dict):
        raise TypeError("configs/markets.yaml: 'markets' must be a dict")
    out: Dict[str, dict] = {}
    for code, m in markets.items():
        if isinstance(m, dict):
            out[str(code).strip().lower()] = m
    return out


def _load_enabled_markets() -> List[str]:
    markets = _load_markets_cfg()
    out: List[str] = []
    for code, m in markets.items():
        if bool(m.get("enabled", False)):
            out.append(code)
    return out


def _safe_read_day(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    need = {"date", "close"}
    if not need.issubset(df.columns):
        raise ValueError(f"missing required cols in {csv_path.name}: {sorted(need)}")
    keep = [c for c in KEEP_COLS if c in df.columns]
    return df[keep].copy()


# =============================================================================
# Bin helpers
# =============================================================================
def _build_bin_edges(start_pct: int, end_pct: int, step_pct: int) -> np.ndarray:
    if start_pct <= 0:
        raise ValueError("start_pct must be > 0")
    if end_pct < start_pct:
        raise ValueError("end_pct must be >= start_pct")
    if step_pct <= 0:
        raise ValueError("step_pct must be > 0")

    arr = list(range(start_pct, end_pct + step_pct, step_pct))
    arr.append(np.inf)
    return np.array(arr, dtype=float)


def _make_bin_label(lo: float, hi: float) -> str:
    lo_i = int(lo)
    if np.isinf(hi):
        return f"{lo_i}%+"
    hi_i = int(hi)
    return f"{lo_i}~{hi_i}%"


def _make_bin_labels(edges: np.ndarray) -> List[str]:
    labels: List[str] = []
    for i in range(len(edges) - 1):
        labels.append(_make_bin_label(edges[i], edges[i + 1]))
    return labels


def _bucketize_event_pct(v: float, edges: np.ndarray) -> str | None:
    if v < edges[0]:
        return None
    for i in range(len(edges) - 1):
        lo = edges[i]
        hi = edges[i + 1]
        if (v >= lo) and (v < hi):
            return _make_bin_label(lo, hi)
    if v >= edges[-2]:
        return _make_bin_label(edges[-2], edges[-1])
    return None


def _apply_ordered_bin(df: pd.DataFrame, col: str, bin_labels: List[str]) -> pd.DataFrame:
    if df.empty or col not in df.columns:
        return df
    out = df.copy()
    out[col] = pd.Categorical(out[col], categories=bin_labels, ordered=True)
    return out


# =============================================================================
# Market sample size / normalization
# =============================================================================
def _count_csv_files(day_dir: Path) -> int:
    if not day_dir.exists():
        return 0
    return len(list(day_dir.glob("*.csv")))


def _build_market_sample_table(markets: List[str]) -> pd.DataFrame:
    rows = []
    for m in markets:
        day_dir = DATA_ROOT / "cache_dayk" / m
        rows.append(
            {
                "market": m,
                "day_dir": str(day_dir),
                "csv_file_count": _count_csv_files(day_dir),
            }
        )
    return pd.DataFrame(rows)


# =============================================================================
# Suspicious / cleaning helpers
# =============================================================================
def _empty_events_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "date",
            "year",
            "ticker",
            "market",
            "market_ticker",
            "ret_close_pct",
            "bin",
            "prev_close",
            "close",
            "open",
            "volume",
            "is_scale_spike",
            "is_reverse_split_like",
            "is_tiny_prev_close",
            "clean_keep",
            "suspicious_reason",
        ]
    )


def _flag_suspicious_rows(
    df: pd.DataFrame,
    *,
    tiny_prev_close: float,
    scale_spike_up: float,
    scale_spike_back: float,
    reverse_split_up: float,
    reverse_split_back_min: float,
    reverse_split_back_max: float,
) -> pd.DataFrame:
    out = df.copy()

    prev_c = out["close"].shift(1).replace(0, np.nan)
    next_c = out["close"].shift(-1).replace(0, np.nan)

    out["prev_close"] = prev_c
    out["next_close"] = next_c

    out["ret_close_pct"] = ((out["close"] / prev_c) - 1.0) * 100.0

    out["is_tiny_prev_close"] = prev_c.notna() & (prev_c < tiny_prev_close)

    up_vs_prev = out["close"] / prev_c
    back_vs_curr = next_c / out["close"]

    out["is_scale_spike"] = (
        prev_c.notna()
        & next_c.notna()
        & (up_vs_prev >= scale_spike_up)
        & (back_vs_curr <= scale_spike_back)
    )

    out["is_reverse_split_like"] = (
        prev_c.notna()
        & next_c.notna()
        & (up_vs_prev >= reverse_split_up)
        & (back_vs_curr >= reverse_split_back_min)
        & (back_vs_curr <= reverse_split_back_max)
    )

    def _reason(row: pd.Series) -> str:
        rs = []
        if bool(row.get("is_tiny_prev_close", False)):
            rs.append("tiny_prev_close")
        if bool(row.get("is_scale_spike", False)):
            rs.append("scale_spike")
        if bool(row.get("is_reverse_split_like", False)):
            rs.append("reverse_split_like")
        return ";".join(rs)

    out["suspicious_reason"] = out.apply(_reason, axis=1)
    out["clean_keep"] = ~(
        out["is_tiny_prev_close"]
        | out["is_scale_spike"]
        | out["is_reverse_split_like"]
    )

    return out


# =============================================================================
# File scan
# =============================================================================
def _scan_one_file(
    csv_path: Path,
    *,
    min_pct: float,
    edges: np.ndarray,
    market_code: str,
    clean_mode: str,
    tiny_prev_close: float,
    scale_spike_up: float,
    scale_spike_back: float,
    reverse_split_up: float,
    reverse_split_back_min: float,
    reverse_split_back_max: float,
) -> Tuple[pd.DataFrame, Optional[dict]]:
    df = _safe_read_day(csv_path)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for c in ["open", "high", "low", "close", "volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["date", "close"]).copy()
    df = df.sort_values("date").reset_index(drop=True)

    if len(df) < 2:
        return _empty_events_df(), None

    if "open" not in df.columns:
        df["open"] = np.nan
    if "volume" not in df.columns:
        df["volume"] = np.nan

    flagged = _flag_suspicious_rows(
        df,
        tiny_prev_close=tiny_prev_close,
        scale_spike_up=scale_spike_up,
        scale_spike_back=scale_spike_back,
        reverse_split_up=reverse_split_up,
        reverse_split_back_min=reverse_split_back_min,
        reverse_split_back_max=reverse_split_back_max,
    )

    flagged["ticker"] = csv_path.stem

    d = flagged[
        [
            "date",
            "ticker",
            "ret_close_pct",
            "prev_close",
            "close",
            "open",
            "volume",
            "is_scale_spike",
            "is_reverse_split_like",
            "is_tiny_prev_close",
            "clean_keep",
            "suspicious_reason",
        ]
    ].dropna(subset=["ret_close_pct"]).copy()

    d = d[d["ret_close_pct"] >= min_pct].copy()
    if d.empty:
        return _empty_events_df(), None

    if clean_mode == "clean":
        d = d[d["clean_keep"]].copy()
    elif clean_mode == "flag":
        pass
    elif clean_mode == "raw":
        pass
    else:
        raise ValueError(f"Unknown clean_mode: {clean_mode}")

    if d.empty:
        return _empty_events_df(), None

    d["market"] = market_code
    d["market_ticker"] = d["market"] + "::" + d["ticker"]
    d["year"] = pd.to_datetime(d["date"]).dt.year.astype("Int64")
    d["bin"] = d["ret_close_pct"].apply(lambda x: _bucketize_event_pct(float(x), edges))
    d = d.dropna(subset=["bin"]).copy()

    if d.empty:
        return _empty_events_df(), None

    peak = d.sort_values(["ret_close_pct", "date"], ascending=[False, True]).iloc[0]

    peak_row = {
        "market": market_code,
        "ticker": str(peak["ticker"]),
        "market_ticker": f"{market_code}::{peak['ticker']}",
        "peak_date": peak["date"],
        "peak_year": int(pd.to_datetime(peak["date"]).year),
        "peak_ret_close_pct": float(peak["ret_close_pct"]),
        "peak_bin": peak["bin"],
        "peak_is_scale_spike": bool(peak["is_scale_spike"]),
        "peak_is_reverse_split_like": bool(peak["is_reverse_split_like"]),
        "peak_is_tiny_prev_close": bool(peak["is_tiny_prev_close"]),
        "peak_clean_keep": bool(peak["clean_keep"]),
        "peak_suspicious_reason": str(peak["suspicious_reason"]),
    }

    d = d[
        [
            "date",
            "year",
            "ticker",
            "market",
            "market_ticker",
            "ret_close_pct",
            "bin",
            "prev_close",
            "close",
            "open",
            "volume",
            "is_scale_spike",
            "is_reverse_split_like",
            "is_tiny_prev_close",
            "clean_keep",
            "suspicious_reason",
        ]
    ].copy()
    return d, peak_row


def scan_market(
    market_code: str,
    *,
    min_pct: float,
    end_pct: int,
    step_pct: int,
    workers: int,
    clean_mode: str,
    tiny_prev_close: float,
    scale_spike_up: float,
    scale_spike_back: float,
    reverse_split_up: float,
    reverse_split_back_min: float,
    reverse_split_back_max: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    day_dir = DATA_ROOT / "cache_dayk" / market_code
    if not day_dir.exists():
        raise FileNotFoundError(f"day dir not found: {day_dir}")

    csvs = sorted(day_dir.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"no csv files in: {day_dir}")

    start_pct = int(np.floor(min_pct / step_pct) * step_pct)
    start_pct = max(step_pct, start_pct)
    edges = _build_bin_edges(start_pct=start_pct, end_pct=end_pct, step_pct=step_pct)

    event_rows: List[pd.DataFrame] = []
    ticker_peak_rows: List[dict] = []

    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futures = {
            ex.submit(
                _scan_one_file,
                p,
                min_pct=min_pct,
                edges=edges,
                market_code=market_code,
                clean_mode=clean_mode,
                tiny_prev_close=tiny_prev_close,
                scale_spike_up=scale_spike_up,
                scale_spike_back=scale_spike_back,
                reverse_split_up=reverse_split_up,
                reverse_split_back_min=reverse_split_back_min,
                reverse_split_back_max=reverse_split_back_max,
            ): p
            for p in csvs
        }

        for fut in tqdm(
            as_completed(futures),
            total=len(futures),
            desc=f"Scan {market_code}",
            unit="file",
            leave=False,
        ):
            p = futures[fut]
            try:
                d, peak = fut.result()
                if not d.empty:
                    event_rows.append(d)
                if peak is not None:
                    ticker_peak_rows.append(peak)
            except Exception as e:
                print(f"❌ [{market_code}] {p.name}: {type(e).__name__}: {e}")

    events_df = pd.concat(event_rows, ignore_index=True) if event_rows else _empty_events_df()
    ticker_peak_df = pd.DataFrame(ticker_peak_rows)

    return events_df, ticker_peak_df


# =============================================================================
# Summaries
# =============================================================================
def summarize_events(
    events_df: pd.DataFrame,
    ticker_peak_df: pd.DataFrame,
    *,
    bin_labels: List[str],
    market_sample_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if events_df.empty:
        empty = pd.DataFrame()
        return empty, empty, empty, empty, empty, empty, empty

    work = _apply_ordered_bin(events_df, "bin", bin_labels)

    event_summary = (
        work.groupby(["market", "bin"], observed=False)
        .size()
        .reset_index(name="event_count")
        .sort_values(["market", "bin"])
        .reset_index(drop=True)
    )

    ticker_bin_summary = (
        work.groupby(["market", "bin"], observed=False)["ticker"]
        .nunique()
        .reset_index(name="unique_ticker_count")
        .sort_values(["market", "bin"])
        .reset_index(drop=True)
    )

    if ticker_peak_df.empty:
        peak_summary = pd.DataFrame()
    else:
        peak_work = _apply_ordered_bin(ticker_peak_df, "peak_bin", bin_labels)
        peak_summary = (
            peak_work.groupby(["market", "peak_bin"], observed=False)["ticker"]
            .nunique()
            .reset_index(name="ticker_peak_count")
            .sort_values(["market", "peak_bin"])
            .reset_index(drop=True)
        )

    market_overview = (
        work.groupby("market")
        .agg(
            event_count=("ticker", "size"),
            unique_ticker_count=("ticker", "nunique"),
            unique_market_ticker_count=("market_ticker", "nunique"),
            max_ret_close_pct=("ret_close_pct", "max"),
            mean_ret_close_pct=("ret_close_pct", "mean"),
            median_ret_close_pct=("ret_close_pct", "median"),
            suspicious_event_count=("clean_keep", lambda s: int((~s).sum())),
            kept_event_count=("clean_keep", lambda s: int(s.sum())),
            first_event_date=("date", "min"),
            last_event_date=("date", "max"),
        )
        .reset_index()
    )

    market_overview = market_overview.merge(market_sample_df, on="market", how="left")
    market_overview["events_per_1000_files"] = np.where(
        market_overview["csv_file_count"] > 0,
        market_overview["event_count"] / market_overview["csv_file_count"] * 1000.0,
        np.nan,
    )
    market_overview["unique_tickers_per_1000_files"] = np.where(
        market_overview["csv_file_count"] > 0,
        market_overview["unique_ticker_count"] / market_overview["csv_file_count"] * 1000.0,
        np.nan,
    )
    market_overview = market_overview.sort_values(
        ["event_count", "unique_ticker_count"], ascending=False
    ).reset_index(drop=True)

    yearly_event_summary = (
        work.groupby(["market", "year", "bin"], observed=False)
        .size()
        .reset_index(name="event_count")
        .sort_values(["market", "year", "bin"])
        .reset_index(drop=True)
    )

    yearly_ticker_summary = (
        work.groupby(["market", "year", "bin"], observed=False)["ticker"]
        .nunique()
        .reset_index(name="unique_ticker_count")
        .sort_values(["market", "year", "bin"])
        .reset_index(drop=True)
    )

    suspicious_summary = (
        work.groupby(["market", "suspicious_reason"], dropna=False)
        .size()
        .reset_index(name="event_count")
        .sort_values(["market", "event_count"], ascending=[True, False])
        .reset_index(drop=True)
    )

    return (
        event_summary,
        ticker_bin_summary,
        peak_summary,
        market_overview,
        yearly_event_summary,
        yearly_ticker_summary,
        suspicious_summary,
    )


def summarize_global(
    events_df: pd.DataFrame,
    ticker_peak_df: pd.DataFrame,
    *,
    bin_labels: List[str],
    market_sample_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if events_df.empty:
        empty = pd.DataFrame()
        return empty, empty, empty, empty, empty, empty, empty

    work = _apply_ordered_bin(events_df, "bin", bin_labels)

    global_event_summary = (
        work.groupby(["bin"], observed=False)
        .size()
        .reset_index(name="event_count")
        .sort_values(["bin"])
        .reset_index(drop=True)
    )

    global_ticker_bin = (
        work.groupby(["bin"], observed=False)["market_ticker"]
        .nunique()
        .reset_index(name="unique_market_ticker_count")
        .sort_values(["bin"])
        .reset_index(drop=True)
    )

    if ticker_peak_df.empty:
        global_peak_summary = pd.DataFrame()
    else:
        peak_work = _apply_ordered_bin(ticker_peak_df, "peak_bin", bin_labels)
        global_peak_summary = (
            peak_work.groupby(["peak_bin"], observed=False)["market_ticker"]
            .nunique()
            .reset_index(name="ticker_peak_count")
            .sort_values(["peak_bin"])
            .reset_index(drop=True)
        )

    global_overview = pd.DataFrame(
        [
            {
                "event_count": int(len(work)),
                "unique_market_ticker_count": int(work["market_ticker"].nunique()),
                "market_count": int(work["market"].nunique()),
                "max_ret_close_pct": float(work["ret_close_pct"].max()),
                "mean_ret_close_pct": float(work["ret_close_pct"].mean()),
                "median_ret_close_pct": float(work["ret_close_pct"].median()),
                "suspicious_event_count": int((~work["clean_keep"]).sum()),
                "kept_event_count": int(work["clean_keep"].sum()),
                "first_event_date": work["date"].min(),
                "last_event_date": work["date"].max(),
                "total_csv_file_count": int(market_sample_df["csv_file_count"].sum()),
                "events_per_1000_files": (
                    float(len(work)) / float(market_sample_df["csv_file_count"].sum()) * 1000.0
                    if float(market_sample_df["csv_file_count"].sum()) > 0
                    else np.nan
                ),
            }
        ]
    )

    yearly_global_event_summary = (
        work.groupby(["year", "bin"], observed=False)
        .size()
        .reset_index(name="event_count")
        .sort_values(["year", "bin"])
        .reset_index(drop=True)
    )

    yearly_global_ticker_summary = (
        work.groupby(["year", "bin"], observed=False)["market_ticker"]
        .nunique()
        .reset_index(name="unique_market_ticker_count")
        .sort_values(["year", "bin"])
        .reset_index(drop=True)
    )

    suspicious_global_summary = (
        work.groupby(["suspicious_reason"], dropna=False)
        .size()
        .reset_index(name="event_count")
        .sort_values(["event_count"], ascending=False)
        .reset_index(drop=True)
    )

    return (
        global_event_summary,
        global_ticker_bin,
        global_peak_summary,
        global_overview,
        yearly_global_event_summary,
        yearly_global_ticker_summary,
        suspicious_global_summary,
    )


# =============================================================================
# Raw vs Clean summary
# =============================================================================
def build_raw_vs_clean_summary(
    events_df: pd.DataFrame,
    market_sample_df: pd.DataFrame,
) -> pd.DataFrame:
    if events_df.empty:
        return pd.DataFrame()

    rows = []
    for market, g in events_df.groupby("market"):
        csv_file_count = np.nan
        hit = market_sample_df.loc[market_sample_df["market"] == market, "csv_file_count"]
        if len(hit) > 0:
            csv_file_count = float(hit.iloc[0])

        raw_event_count = int(len(g))
        clean_event_count = int(g["clean_keep"].sum())
        dropped_event_count = int(raw_event_count - clean_event_count)

        raw_unique_ticker_count = int(g["ticker"].nunique())
        clean_unique_ticker_count = int(g.loc[g["clean_keep"], "ticker"].nunique())

        raw_max_ret_close_pct = float(g["ret_close_pct"].max()) if raw_event_count > 0 else np.nan
        clean_max_ret_close_pct = (
            float(g.loc[g["clean_keep"], "ret_close_pct"].max())
            if clean_event_count > 0 else np.nan
        )

        rows.append(
            {
                "market": market,
                "csv_file_count": csv_file_count,
                "raw_event_count": raw_event_count,
                "clean_event_count": clean_event_count,
                "dropped_event_count": dropped_event_count,
                "drop_ratio_pct": (dropped_event_count / raw_event_count * 100.0) if raw_event_count > 0 else np.nan,
                "raw_unique_ticker_count": raw_unique_ticker_count,
                "clean_unique_ticker_count": clean_unique_ticker_count,
                "dropped_unique_ticker_count": raw_unique_ticker_count - clean_unique_ticker_count,
                "raw_events_per_1000_files": (
                    raw_event_count / csv_file_count * 1000.0
                    if pd.notna(csv_file_count) and csv_file_count > 0 else np.nan
                ),
                "clean_events_per_1000_files": (
                    clean_event_count / csv_file_count * 1000.0
                    if pd.notna(csv_file_count) and csv_file_count > 0 else np.nan
                ),
                "diff_events_per_1000_files": (
                    (raw_event_count - clean_event_count) / csv_file_count * 1000.0
                    if pd.notna(csv_file_count) and csv_file_count > 0 else np.nan
                ),
                "raw_max_ret_close_pct": raw_max_ret_close_pct,
                "clean_max_ret_close_pct": clean_max_ret_close_pct,
            }
        )

    out = pd.DataFrame(rows).sort_values(
        ["drop_ratio_pct", "dropped_event_count"],
        ascending=[False, False]
    ).reset_index(drop=True)
    return out


def build_global_raw_vs_clean_summary(
    events_df: pd.DataFrame,
    market_sample_df: pd.DataFrame,
) -> pd.DataFrame:
    if events_df.empty:
        return pd.DataFrame()

    raw_event_count = int(len(events_df))
    clean_event_count = int(events_df["clean_keep"].sum())
    dropped_event_count = int(raw_event_count - clean_event_count)

    raw_unique_market_ticker_count = int(events_df["market_ticker"].nunique())
    clean_unique_market_ticker_count = int(events_df.loc[events_df["clean_keep"], "market_ticker"].nunique())

    total_csv_file_count = float(market_sample_df["csv_file_count"].sum()) if not market_sample_df.empty else np.nan

    raw_max_ret_close_pct = float(events_df["ret_close_pct"].max()) if raw_event_count > 0 else np.nan
    clean_max_ret_close_pct = (
        float(events_df.loc[events_df["clean_keep"], "ret_close_pct"].max())
        if clean_event_count > 0 else np.nan
    )

    return pd.DataFrame(
        [
            {
                "scope": "all_markets",
                "total_csv_file_count": total_csv_file_count,
                "raw_event_count": raw_event_count,
                "clean_event_count": clean_event_count,
                "dropped_event_count": dropped_event_count,
                "drop_ratio_pct": (dropped_event_count / raw_event_count * 100.0) if raw_event_count > 0 else np.nan,
                "raw_unique_market_ticker_count": raw_unique_market_ticker_count,
                "clean_unique_market_ticker_count": clean_unique_market_ticker_count,
                "dropped_unique_market_ticker_count": raw_unique_market_ticker_count - clean_unique_market_ticker_count,
                "raw_events_per_1000_files": (
                    raw_event_count / total_csv_file_count * 1000.0
                    if pd.notna(total_csv_file_count) and total_csv_file_count > 0 else np.nan
                ),
                "clean_events_per_1000_files": (
                    clean_event_count / total_csv_file_count * 1000.0
                    if pd.notna(total_csv_file_count) and total_csv_file_count > 0 else np.nan
                ),
                "diff_events_per_1000_files": (
                    (raw_event_count - clean_event_count) / total_csv_file_count * 1000.0
                    if pd.notna(total_csv_file_count) and total_csv_file_count > 0 else np.nan
                ),
                "raw_max_ret_close_pct": raw_max_ret_close_pct,
                "clean_max_ret_close_pct": clean_max_ret_close_pct,
            }
        ]
    )


# =============================================================================
# Top spikes
# =============================================================================
def make_top_spikes(
    events_df: pd.DataFrame,
    *,
    top_n_market: int,
    top_n_global: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if events_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    market_top = (
        events_df.sort_values(["market", "ret_close_pct", "date"], ascending=[True, False, True])
        .groupby("market", group_keys=False)
        .head(top_n_market)
        .reset_index(drop=True)
    )

    global_top = (
        events_df.sort_values(["ret_close_pct", "date"], ascending=[False, True])
        .head(top_n_global)
        .reset_index(drop=True)
    )

    return market_top, global_top


# =============================================================================
# CLI
# =============================================================================
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--markets", default="", help="Comma-separated market codes. Empty = all enabled markets")
    ap.add_argument("--min-pct", type=float, default=100.0, help="Minimum single-day ret_close_pct to include")
    ap.add_argument("--end-pct", type=int, default=2000, help="Upper explicit bin end. Last bin becomes end_pct%+")
    ap.add_argument("--step-pct", type=int, default=100, help="Bin width in percent")
    ap.add_argument("--workers", type=int, default=8, help="Thread workers per market")
    ap.add_argument("--top-n-market", type=int, default=100, help="Rows for per-market top spikes output")
    ap.add_argument("--top-n-global", type=int, default=300, help="Rows for global top spikes output")
    ap.add_argument("--out-dir", default="data/research/daily_extreme_scan", help="Output directory")

    ap.add_argument(
        "--clean-mode",
        choices=["raw", "flag", "clean"],
        default="flag",
        help="raw=keep all, flag=keep all but mark suspicious, clean=drop suspicious rows",
    )
    ap.add_argument(
        "--tiny-prev-close",
        type=float,
        default=0.05,
        help="Flag event if prev_close is below this threshold",
    )
    ap.add_argument(
        "--scale-spike-up",
        type=float,
        default=20.0,
        help="Scale spike if close/prev_close >= this and next day collapses back",
    )
    ap.add_argument(
        "--scale-spike-back",
        type=float,
        default=0.2,
        help="Scale spike if next_close/close <= this",
    )
    ap.add_argument(
        "--reverse-split-up",
        type=float,
        default=5.0,
        help="Reverse-split-like if close/prev_close >= this and next day stays near new level",
    )
    ap.add_argument(
        "--reverse-split-back-min",
        type=float,
        default=0.5,
        help="Reverse-split-like if next_close/close >= this",
    )
    ap.add_argument(
        "--reverse-split-back-max",
        type=float,
        default=1.5,
        help="Reverse-split-like if next_close/close <= this",
    )

    return ap.parse_args()


# =============================================================================
# Main
# =============================================================================
def main():
    args = parse_args()

    if args.markets.strip():
        markets = [x.strip().lower() for x in args.markets.split(",") if x.strip()]
    else:
        markets = _load_enabled_markets()

    if not markets:
        raise RuntimeError("No markets to scan.")

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    start_pct = int(np.floor(args.min_pct / args.step_pct) * args.step_pct)
    start_pct = max(args.step_pct, start_pct)
    edges = _build_bin_edges(start_pct=start_pct, end_pct=args.end_pct, step_pct=args.step_pct)
    bin_labels = _make_bin_labels(edges)

    market_sample_df = _build_market_sample_table(markets)

    print("Markets:", markets)
    print("min_pct:", args.min_pct)
    print("step_pct:", args.step_pct)
    print("end_pct:", args.end_pct)
    print("workers:", args.workers)
    print("top_n_market:", args.top_n_market)
    print("top_n_global:", args.top_n_global)
    print("clean_mode:", args.clean_mode)
    print("tiny_prev_close:", args.tiny_prev_close)
    print("scale_spike_up:", args.scale_spike_up)
    print("scale_spike_back:", args.scale_spike_back)
    print("reverse_split_up:", args.reverse_split_up)
    print("reverse_split_back_min:", args.reverse_split_back_min)
    print("reverse_split_back_max:", args.reverse_split_back_max)
    print("out_dir:", out_dir)

    market_sample_df.to_csv(out_dir / "market_sample_table.csv", index=False, encoding="utf-8-sig")

    all_events: List[pd.DataFrame] = []
    all_peaks: List[pd.DataFrame] = []

    for market_code in tqdm(markets, desc="Markets", unit="market"):
        try:
            events_df, ticker_peak_df = scan_market(
                market_code,
                min_pct=args.min_pct,
                end_pct=args.end_pct,
                step_pct=args.step_pct,
                workers=args.workers,
                clean_mode=args.clean_mode,
                tiny_prev_close=args.tiny_prev_close,
                scale_spike_up=args.scale_spike_up,
                scale_spike_back=args.scale_spike_back,
                reverse_split_up=args.reverse_split_up,
                reverse_split_back_min=args.reverse_split_back_min,
                reverse_split_back_max=args.reverse_split_back_max,
            )

            if not events_df.empty:
                all_events.append(events_df)
            if not ticker_peak_df.empty:
                all_peaks.append(ticker_peak_df)

            (
                event_summary,
                ticker_bin_summary,
                peak_summary,
                market_overview,
                yearly_event_summary,
                yearly_ticker_summary,
                suspicious_summary,
            ) = summarize_events(
                events_df,
                ticker_peak_df,
                bin_labels=bin_labels,
                market_sample_df=market_sample_df,
            )

            market_top_spikes, _ = make_top_spikes(
                events_df,
                top_n_market=args.top_n_market,
                top_n_global=args.top_n_global,
            )

            raw_vs_clean_summary = build_raw_vs_clean_summary(
                events_df=events_df,
                market_sample_df=market_sample_df,
            )

            if not events_df.empty:
                events_df.to_csv(out_dir / f"{market_code}_events.csv", index=False, encoding="utf-8-sig")
            if not ticker_peak_df.empty:
                ticker_peak_df.to_csv(out_dir / f"{market_code}_ticker_peaks.csv", index=False, encoding="utf-8-sig")
            if not event_summary.empty:
                event_summary.to_csv(out_dir / f"{market_code}_event_summary.csv", index=False, encoding="utf-8-sig")
            if not ticker_bin_summary.empty:
                ticker_bin_summary.to_csv(out_dir / f"{market_code}_ticker_bin_summary.csv", index=False, encoding="utf-8-sig")
            if not peak_summary.empty:
                peak_summary.to_csv(out_dir / f"{market_code}_ticker_peak_summary.csv", index=False, encoding="utf-8-sig")
            if not market_overview.empty:
                market_overview.to_csv(out_dir / f"{market_code}_market_overview.csv", index=False, encoding="utf-8-sig")
            if not yearly_event_summary.empty:
                yearly_event_summary.to_csv(out_dir / f"{market_code}_yearly_event_summary.csv", index=False, encoding="utf-8-sig")
            if not yearly_ticker_summary.empty:
                yearly_ticker_summary.to_csv(out_dir / f"{market_code}_yearly_ticker_summary.csv", index=False, encoding="utf-8-sig")
            if not suspicious_summary.empty:
                suspicious_summary.to_csv(out_dir / f"{market_code}_suspicious_summary.csv", index=False, encoding="utf-8-sig")
            if not raw_vs_clean_summary.empty:
                raw_vs_clean_summary.to_csv(out_dir / f"{market_code}_raw_vs_clean_summary.csv", index=False, encoding="utf-8-sig")
            if not market_top_spikes.empty:
                market_top_spikes.to_csv(out_dir / f"{market_code}_top_spikes.csv", index=False, encoding="utf-8-sig")

            print(
                f"[{market_code}] "
                f"events={0 if events_df.empty else len(events_df)} "
                f"tickers={0 if ticker_peak_df.empty else len(ticker_peak_df)}"
            )

        except Exception as e:
            print(f"❌ [{market_code}] {type(e).__name__}: {e}")

    all_events_df = pd.concat(all_events, ignore_index=True) if all_events else _empty_events_df()
    all_peaks_df = pd.concat(all_peaks, ignore_index=True) if all_peaks else pd.DataFrame()

    if not all_events_df.empty:
        all_events_df.to_csv(out_dir / "all_markets_events.csv", index=False, encoding="utf-8-sig")
    if not all_peaks_df.empty:
        all_peaks_df.to_csv(out_dir / "all_markets_ticker_peaks.csv", index=False, encoding="utf-8-sig")

    (
        global_event_summary,
        global_ticker_bin,
        global_peak_summary,
        global_overview,
        yearly_global_event_summary,
        yearly_global_ticker_summary,
        suspicious_global_summary,
    ) = summarize_global(
        all_events_df,
        all_peaks_df,
        bin_labels=bin_labels,
        market_sample_df=market_sample_df,
    )

    _, global_top_spikes = make_top_spikes(
        all_events_df,
        top_n_market=args.top_n_market,
        top_n_global=args.top_n_global,
    )

    all_markets_raw_vs_clean_summary = build_global_raw_vs_clean_summary(
        events_df=all_events_df,
        market_sample_df=market_sample_df,
    )

    if not global_event_summary.empty:
        global_event_summary.to_csv(out_dir / "all_markets_event_summary.csv", index=False, encoding="utf-8-sig")
    if not global_ticker_bin.empty:
        global_ticker_bin.to_csv(out_dir / "all_markets_ticker_bin_summary.csv", index=False, encoding="utf-8-sig")
    if not global_peak_summary.empty:
        global_peak_summary.to_csv(out_dir / "all_markets_ticker_peak_summary.csv", index=False, encoding="utf-8-sig")
    if not global_overview.empty:
        global_overview.to_csv(out_dir / "all_markets_overview.csv", index=False, encoding="utf-8-sig")
    if not yearly_global_event_summary.empty:
        yearly_global_event_summary.to_csv(out_dir / "all_markets_yearly_event_summary.csv", index=False, encoding="utf-8-sig")
    if not yearly_global_ticker_summary.empty:
        yearly_global_ticker_summary.to_csv(out_dir / "all_markets_yearly_ticker_summary.csv", index=False, encoding="utf-8-sig")
    if not suspicious_global_summary.empty:
        suspicious_global_summary.to_csv(out_dir / "all_markets_suspicious_summary.csv", index=False, encoding="utf-8-sig")
    if not all_markets_raw_vs_clean_summary.empty:
        all_markets_raw_vs_clean_summary.to_csv(out_dir / "all_markets_raw_vs_clean_summary.csv", index=False, encoding="utf-8-sig")
    if not global_top_spikes.empty:
        global_top_spikes.to_csv(out_dir / "all_markets_top_spikes.csv", index=False, encoding="utf-8-sig")

    print("✅ Done")
    print("Saved to:", out_dir)


if __name__ == "__main__":
    main()