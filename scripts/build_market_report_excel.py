# scripts/build_market_report_excel.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data"

DERIVED_DIR = DATA_ROOT / "derived"
OUT_XLSX_DIR = DATA_ROOT / "reports" / "market_reports"
OUT_CSV_DIR = DATA_ROOT / "reports" / "market_reports_csv"

MARKETS_YAML = ROOT / "configs" / "markets.yaml"


# ============================================================
# timeframe config
# ============================================================

TIMEFRAME_CONFIG = {
    "W": {
        "dir": DERIVED_DIR / "weekK",
        "ret_cols": {
            "close": "ret_close_W",
            "high": "ret_high_W",
            "low": "ret_low_W",
        },
    },
    "M": {
        "dir": DERIVED_DIR / "monthK",
        "ret_cols": {
            "close": "ret_close_M",
            "high": "ret_high_M",
            "low": "ret_low_M",
        },
    },
    "Y": {
        "dir": DERIVED_DIR / "yearK",
        "ret_cols": {
            "close": "ret_close_Y",
            "high": "ret_high_Y",
            "low": "ret_low_Y",
        },
    },
}


# ============================================================
# load enabled markets
# ============================================================

def _load_enabled_markets() -> List[str]:

    if not MARKETS_YAML.exists():
        raise FileNotFoundError(MARKETS_YAML)

    cfg = yaml.safe_load(MARKETS_YAML.read_text(encoding="utf-8")) or {}

    markets = cfg.get("markets", {})

    out = []

    for k, v in markets.items():
        if isinstance(v, dict) and v.get("enabled", False):
            out.append(str(k).lower())

    return out


# ============================================================
# safe csv
# ============================================================

def _safe_read_csv(path: Path, usecols: List[str]) -> pd.DataFrame | None:
    try:
        return pd.read_csv(path, usecols=usecols, encoding="utf-8-sig")
    except Exception:
        return None


# ============================================================
# collect returns
# ============================================================

def _collect_returns(
    market: str,
    timeframe: str,
    ret_col: str,
    start_year: int,
    end_year: int,
) -> Tuple[np.ndarray, int]:

    base_dir = TIMEFRAME_CONFIG[timeframe]["dir"] / market

    if not base_dir.exists():
        return np.array([], dtype=float), 0

    files = sorted(base_dir.glob("*.csv"))

    all_vals = []
    included_symbols = 0

    for p in files:

        df = _safe_read_csv(p, ["date", ret_col])

        if df is None or df.empty:
            continue

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df[ret_col] = pd.to_numeric(df[ret_col], errors="coerce")

        df = df.dropna(subset=["date", ret_col])

        if df.empty:
            continue

        df = df[
            (df["date"].dt.year >= start_year)
            & (df["date"].dt.year <= end_year)
        ]

        if df.empty:
            continue

        vals = (
            df[ret_col].astype(float) * 100.0
        ).replace([np.inf, -np.inf], np.nan).dropna()

        if vals.empty:
            continue

        included_symbols += 1
        all_vals.extend(vals.tolist())

    arr = np.asarray(all_vals, dtype=float)
    arr = arr[np.isfinite(arr)]

    return arr, included_symbols


# ============================================================
# helpers
# ============================================================

def _ratio(arr: np.ndarray, cond) -> float:

    if arr.size == 0:
        return 0.0

    return float(np.mean(cond(arr)))


# ============================================================
# summary
# ============================================================

def _summary_row(
    market,
    timeframe,
    ret_name,
    arr,
    n_symbols,
):

    if arr.size == 0:

        return {
            "market": market,
            "timeframe": timeframe,
            "ret_col": ret_name,
            "n_symbols": n_symbols,
            "n_period_returns": 0,
        }

    q01, q05, q10, q25, q50, q75, q90, q95, q99 = np.quantile(
        arr,
        [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99],
    )

    return {

        "market": market,
        "timeframe": timeframe,
        "ret_col": ret_name,

        "n_symbols": n_symbols,
        "n_period_returns": int(arr.size),

        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),

        "min": float(np.min(arr)),
        "p01": float(q01),
        "p05": float(q05),
        "p10": float(q10),

        "p25": float(q25),
        "p50": float(q50),
        "p75": float(q75),

        "p90": float(q90),
        "p95": float(q95),
        "p99": float(q99),

        "max": float(np.max(arr)),

        "positive_ratio": _ratio(arr, lambda x: x > 0),
        "negative_ratio": _ratio(arr, lambda x: x < 0),

        "gt_10_ratio": _ratio(arr, lambda x: x > 10),
        "gt_20_ratio": _ratio(arr, lambda x: x > 20),
        "gt_50_ratio": _ratio(arr, lambda x: x > 50),
        "gt_100_ratio": _ratio(arr, lambda x: x > 100),

        "lt_m10_ratio": _ratio(arr, lambda x: x < -10),
        "lt_m20_ratio": _ratio(arr, lambda x: x < -20),
        "lt_m50_ratio": _ratio(arr, lambda x: x < -50),

        "iqr": float(q75 - q25),
        "p95_p05_spread": float(q95 - q05),

    }


# ============================================================
# build one market report
# ============================================================

def build_one_market(
    market,
    start_year,
    end_year,
):

    summary_rows = []

    for timeframe, cfg in TIMEFRAME_CONFIG.items():

        for ret_name, ret_col in cfg["ret_cols"].items():

            arr, n_symbols = _collect_returns(
                market,
                timeframe,
                ret_col,
                start_year,
                end_year,
            )

            summary_rows.append(
                _summary_row(
                    market,
                    timeframe,
                    ret_name,
                    arr,
                    n_symbols,
                )
            )

    summary_df = pd.DataFrame(summary_rows)

    metadata_df = pd.DataFrame(
        [
            {
                "market": market,
                "start_year": start_year,
                "end_year": end_year,
                "generated_at": datetime.now().isoformat(timespec="seconds"),
            }
        ]
    )

    xlsx_path = OUT_XLSX_DIR / f"{market}_market_report_{start_year}_{end_year}.xlsx"

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:

        summary_df.to_excel(writer, sheet_name="summary", index=False)
        metadata_df.to_excel(writer, sheet_name="metadata", index=False)

    csv_dir = OUT_CSV_DIR / market
    csv_dir.mkdir(parents=True, exist_ok=True)

    summary_df.to_csv(
        csv_dir / f"{market}_summary_{start_year}_{end_year}.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print(f"✅ {market} report saved")


# ============================================================
# main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--markets",
        default="",
        help="tw,us,jp or empty=all enabled markets",
    )

    parser.add_argument(
        "--start-year",
        type=int,
        default=2020,
    )

    parser.add_argument(
        "--end-year",
        type=int,
        default=2025,
    )

    args = parser.parse_args()

    OUT_XLSX_DIR.mkdir(parents=True, exist_ok=True)

    if args.markets:

        markets = [
            x.strip().lower()
            for x in args.markets.split(",")
            if x.strip()
        ]

    else:

        markets = _load_enabled_markets()

    print("Markets:", markets)

    for m in markets:

        build_one_market(
            market=m,
            start_year=args.start_year,
            end_year=args.end_year,
        )

    print("\n🎉 All market reports finished.")


if __name__ == "__main__":
    main()