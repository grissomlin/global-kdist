# scripts/compute_return_distributions.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from core.cleaning.scale_uk import should_apply_uk_scale, apply_scale_fix


THIS = Path(__file__).resolve()
REPO_ROOT = THIS.parents[1]

CACHE_DAYK = REPO_ROOT / "data" / "cache_dayk"
BLACKLIST_DIR = REPO_ROOT / "data" / "blacklists"
OUT_DIR = REPO_ROOT / "data" / "reports" / "return_dists"


def _read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.tz_localize(None)
    for c in ("open", "high", "low", "close"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _load_blacklist(market: str) -> set[str]:
    market = market.lower()
    if market == "uk":
        p = BLACKLIST_DIR / "uk_scale_residual.txt"
        if p.exists():
            lines = [x.strip() for x in p.read_text(encoding="utf-8").splitlines()]
            return {x for x in lines if x}
    return set()


def _resample_close_ret(df: pd.DataFrame, freq: str) -> pd.Series:
    """
    df must have columns date, close.
    freq examples:
      - W-FRI  (weekly, Fri close)
      - ME     (month-end)
      - YE     (year-end)
    returns: series of returns (close/prev_close - 1)
    """
    x = df.dropna(subset=["date", "close"]).copy()
    if x.empty:
        return pd.Series(dtype=float)

    x = x.sort_values("date").set_index("date")
    close = x["close"].astype(float)

    rclose = close.resample(freq).last().dropna()
    if len(rclose) < 2:
        return pd.Series(dtype=float)

    ret = (rclose / rclose.shift(1)) - 1.0
    ret = ret.replace([np.inf, -np.inf], np.nan).dropna()
    return ret


def _clip(ret: pd.Series, lo: float, hi: float) -> pd.Series:
    if ret is None or ret.empty:
        return ret
    return ret.clip(lower=lo, upper=hi)


def _hist(ret: np.ndarray, bins: int, lo: float, hi: float) -> Tuple[np.ndarray, np.ndarray]:
    edges = np.linspace(lo, hi, bins + 1)
    counts, edges = np.histogram(ret, bins=edges)
    return counts, edges


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--markets", type=str, required=True, help="e.g. uk or 'us,uk,tw'")
    ap.add_argument("--timeframe", type=str, default="W", choices=["W", "M", "Y"])
    ap.add_argument("--clip-lo", type=float, default=-0.80, help="e.g. -0.80 = -80%")
    ap.add_argument("--clip-hi", type=float, default=3.00, help="e.g. 3.00 = +300%")
    ap.add_argument("--bins", type=int, default=80)
    ap.add_argument("--max-files", type=int, default=0, help="debug: limit number of files")
    ap.add_argument("--uk-scale-fix", action="store_true", help="apply uk scale_fix before resample (recommended)")
    args = ap.parse_args()

    markets = [m.strip().lower() for m in args.markets.split(",") if m.strip()]
    tf = args.timeframe.upper().strip()

    # pandas newer freq aliases:
    # - month end: ME (instead of M)
    # - year end : YE (instead of A)
    if tf == "W":
        freq = "W-FRI"
        tf_name = "week"
    elif tf == "M":
        freq = "ME"
        tf_name = "month"
    else:
        freq = "YE"
        tf_name = "year"

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for m in markets:
        mdir = CACHE_DAYK / m
        if not mdir.exists():
            print(f"[SKIP] market={m} dayK dir not found: {mdir}")
            continue

        blacklist = _load_blacklist(m)
        files = sorted([p for p in mdir.glob("*.csv") if p.is_file() and not p.name.startswith("_")])

        if args.max_files and args.max_files > 0:
            files = files[: int(args.max_files)]

        included = 0
        excluded = 0
        rets_all: List[float] = []

        for p in files:
            if p.name in blacklist:
                excluded += 1
                continue

            df = _read_csv(p)
            if df is None or df.empty or "close" not in df.columns:
                continue

            ticker = p.stem

            if args.uk_scale_fix and should_apply_uk_scale(market_code=m, ticker=ticker):
                try:
                    df = apply_scale_fix(
                        df,
                        mode="near_factor",
                        factor=100.0,
                        require_reversion=False,
                        max_passes=5,
                        return_stats=False,
                    )
                except Exception:
                    pass

            ret = _resample_close_ret(df, freq=freq)
            ret = _clip(ret, lo=float(args.clip_lo), hi=float(args.clip_hi))

            if ret is None or ret.empty:
                continue

            included += 1
            rets_all.extend(ret.astype(float).tolist())

        arr = np.asarray(rets_all, dtype=float)
        arr = arr[np.isfinite(arr)]

        out_base = OUT_DIR / f"{m}_{tf_name}_return_dist"

        if arr.size == 0:
            summary = {
                "market": m,
                "timeframe": tf_name,
                "symbols_total": len(files),
                "symbols_included": included,
                "symbols_excluded_blacklist": excluded,
                "n_returns": 0,
            }
            pd.DataFrame([summary]).to_csv(str(out_base) + "_summary.csv", index=False, encoding="utf-8-sig")
            print(f"[OK] market={m} no returns. wrote {out_base}_summary.csv")
            continue

        qs = [0.01, 0.05, 0.50, 0.95, 0.99]
        qv = np.quantile(arr, qs)

        summary = {
            "market": m,
            "timeframe": tf_name,
            "symbols_total": len(files),
            "symbols_included": included,
            "symbols_excluded_blacklist": excluded,
            "n_returns": int(arr.size),
            "clip_lo": float(args.clip_lo),
            "clip_hi": float(args.clip_hi),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "p01": float(qv[0]),
            "p05": float(qv[1]),
            "p50": float(qv[2]),
            "p95": float(qv[3]),
            "p99": float(qv[4]),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }
        pd.DataFrame([summary]).to_csv(str(out_base) + "_summary.csv", index=False, encoding="utf-8-sig")

        counts, edges = _hist(arr, bins=int(args.bins), lo=float(args.clip_lo), hi=float(args.clip_hi))
        hist_df = pd.DataFrame(
            {"bin_left": edges[:-1], "bin_right": edges[1:], "count": counts}
        )
        hist_df.to_csv(str(out_base) + "_hist.csv", index=False, encoding="utf-8-sig")

        print(
            f"[OK] market={m} tf={tf_name} symbols_total={len(files)} included={included} "
            f"excluded_blacklist={excluded} n_returns={arr.size} "
            f"-> {out_base}_summary.csv / {out_base}_hist.csv"
        )


if __name__ == "__main__":
    main()