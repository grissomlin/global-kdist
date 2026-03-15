# scripts/check_kdist_health.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from core.cleaning.scale_uk import (
    detect_scale_candidates,
    apply_scale_fix,
    should_apply_uk_scale,
)

THIS = Path(__file__).resolve()
REPO_ROOT = THIS.parents[1]
CACHE_DAYK = REPO_ROOT / "data" / "cache_dayk"


def _env_int(name: str, default: str) -> int:
    try:
        return int(str(os.getenv(name, default)).strip())
    except Exception:
        return int(default)


def _count_error_lines(market_dir: Path) -> int:
    n = 0
    for p in market_dir.glob("_errors*.txt"):
        try:
            n += sum(1 for _ in p.open("r", encoding="utf-8", errors="ignore"))
        except Exception:
            pass
    return n


def _read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.tz_localize(None)
    for c in ("open", "high", "low", "close"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _ohlc_issue_count(df: pd.DataFrame) -> int:
    need = {"open", "high", "low", "close"}
    if df is None or df.empty or not need.issubset(set(df.columns)):
        return 0

    o = df["open"]
    h = df["high"]
    l = df["low"]
    c = df["close"]

    mask = o.notna() & h.notna() & l.notna() & c.notna()
    if not mask.any():
        return 0

    o = o[mask]
    h = h[mask]
    l = l[mask]
    c = c[mask]

    hi_bad = h < pd.concat([o, c, l], axis=1).max(axis=1)
    lo_bad = l > pd.concat([o, c, h], axis=1).min(axis=1)
    neg_bad = (o < 0) | (h < 0) | (l < 0) | (c < 0)

    return 1 if (hi_bad | lo_bad | neg_bad).any() else 0


def _scale_counts_uk(df: pd.DataFrame, ticker: str, market_code: str) -> Tuple[int, int, int, int]:
    """
    Return:
      before, after (main metrics; require_reversion=False)
      before_strict, after_strict (reference; require_reversion=True)
    """
    if df is None or df.empty:
        return 0, 0, 0, 0

    if not should_apply_uk_scale(market_code=market_code, ticker=ticker):
        return 0, 0, 0, 0

    # MAIN (stable / monotonic): no reversion
    cand_before = detect_scale_candidates(df, mode="near_factor", factor=100.0, require_reversion=False)
    b = int(len(cand_before))

    fixed = apply_scale_fix(df, mode="near_factor", factor=100.0, require_reversion=False, max_passes=5, return_stats=False)
    cand_after = detect_scale_candidates(fixed, mode="near_factor", factor=100.0, require_reversion=False)
    a = int(len(cand_after))

    # STRICT (reference only)
    cand_before_s = detect_scale_candidates(df, mode="near_factor", factor=100.0, require_reversion=True)
    bs = int(len(cand_before_s))
    cand_after_s = detect_scale_candidates(fixed, mode="near_factor", factor=100.0, require_reversion=True)
    as_ = int(len(cand_after_s))

    return b, a, bs, as_


def _market_report(
    market: str,
    sample: int = 0,
) -> Tuple[str, Dict[str, int], List[Tuple[str, int, int]], List[str], Tuple[Optional[str], Optional[str]]]:
    market = (market or "").strip().lower()
    mdir = CACHE_DAYK / market
    stats: Dict[str, int] = {}

    if not mdir.exists():
        stats.update(
            files=0, ok=0, bad=0, _errors_txt=0, ohlc_issues=0,
            scale_files=0, scale_residual_files=0,
            scale_candidates=0, scale_residual=0,
            strict_scale_candidates=0, strict_scale_residual=0,
        )
        return market, stats, [], [], (None, None)

    files = sorted([p for p in mdir.glob("*.csv") if p.is_file()])
    files = [p for p in files if not p.name.startswith("_")]

    ok = 0
    bad = 0
    ohlc_issues = 0

    scale_candidates = 0
    scale_residual = 0
    strict_candidates = 0
    strict_residual = 0

    scale_files = 0
    scale_residual_files = 0

    scale_details: List[Tuple[str, int, int]] = []
    residual_files: List[str] = []

    min_date: Optional[pd.Timestamp] = None
    max_date: Optional[pd.Timestamp] = None

    for p in files:
        try:
            df = _read_csv(p)
            if df is None or df.empty:
                bad += 1
                continue
            ok += 1

            if "date" in df.columns:
                d = pd.to_datetime(df["date"], errors="coerce")
                dmin = d.min()
                dmax = d.max()
                if pd.notna(dmin):
                    min_date = dmin if min_date is None else min(min_date, dmin)
                if pd.notna(dmax):
                    max_date = dmax if max_date is None else max(max_date, dmax)

            ohlc_issues += _ohlc_issue_count(df)

            ticker = p.stem
            b, a, bs, as_ = _scale_counts_uk(df, ticker=ticker, market_code=market)

            if b > 0:
                scale_files += 1
                scale_candidates += b
                scale_residual += a
                scale_details.append((p.name, b, a))
                if a > 0:
                    scale_residual_files += 1
                    residual_files.append(p.name)

            strict_candidates += bs
            strict_residual += as_

        except Exception:
            bad += 1
            continue

    errors = _count_error_lines(mdir)

    stats["files"] = len(files)
    stats["ok"] = ok
    stats["bad"] = bad
    stats["_errors.txt"] = errors
    stats["ohlc_issues"] = ohlc_issues

    stats["scale_files"] = scale_files
    stats["scale_residual_files"] = scale_residual_files
    stats["scale_candidates"] = scale_candidates
    stats["scale_residual"] = scale_residual

    stats["strict_scale_candidates"] = strict_candidates
    stats["strict_scale_residual"] = strict_residual

    sample_rows: List[Tuple[str, int, int]] = []
    if sample and sample > 0 and scale_details:
        scale_details_sorted = sorted(scale_details, key=lambda x: (-x[1], x[0]))
        sample_rows = scale_details_sorted[:sample]

    rng = (
        min_date.strftime("%Y-%m-%d") if min_date is not None else None,
        max_date.strftime("%Y-%m-%d") if max_date is not None else None,
    )

    return market, stats, sample_rows, residual_files, rng


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--markets", type=str, required=True, help="e.g. uk or 'us,uk,tw'")
    ap.add_argument("--sample", type=int, default=0, help="print pre/post samples for top-N files")
    args = ap.parse_args()

    markets = [m.strip().lower() for m in args.markets.split(",") if m.strip()]
    sample = int(args.sample or 0)

    print("\n================= KDist Health Report =================")

    for m in markets:
        market, st, sample_rows, residual_files, rng = _market_report(m, sample=sample)

        files = st.get("files", 0)
        ok = st.get("ok", 0)
        bad = st.get("bad", 0)
        errors = st.get("_errors.txt", 0)
        ohlc_issues = st.get("ohlc_issues", 0)

        sc = st.get("scale_candidates", 0)
        sr = st.get("scale_residual", 0)
        sf = st.get("scale_files", 0)
        srf = st.get("scale_residual_files", 0)

        ssc = st.get("strict_scale_candidates", 0)
        ssr = st.get("strict_scale_residual", 0)

        if rng[0] and rng[1]:
            try:
                days = (pd.to_datetime(rng[1]) - pd.to_datetime(rng[0])).days + 1
            except Exception:
                days = 0
            range_s = f"{rng[0]}..{rng[1]} (days={days})"
        else:
            range_s = "N/A"

        fix_rate = 0.0
        if sc > 0:
            fix_rate = (float(sc - sr) / float(sc)) * 100.0

        print(
            f"[{market}] files={files}, ok={ok}, bad={bad}, _errors.txt={errors}, range={range_s}, "
            f"ohlc_issues={ohlc_issues}, "
            f"scale_files={sf}, scale_residual_files={srf}, "
            f"scale_candidates={sc}, scale_residual={sr}, scale_fix_rate={fix_rate:.2f}%, "
            f"strict_candidates={ssc}, strict_residual={ssr}"
        )

    print("=======================================================")

    if sample > 0:
        for m in markets:
            market, st, sample_rows, residual_files, _rng = _market_report(m, sample=sample)
            if not sample_rows:
                continue

            print(f"\n--- [{market}] scale pre/post samples ---")
            for fn, b, a in sample_rows:
                print(f"  {fn}: before={b}, after={a}")

            if residual_files:
                print(f"--- [{market}] scale residual files ---")
                cap = _env_int("KDIST_HEALTH_RESIDUAL_CAP", "200")
                for fn in residual_files[:cap]:
                    print(f"  {fn}")
                if len(residual_files) > cap:
                    print(f"  ... ({len(residual_files) - cap} more)")


if __name__ == "__main__":
    main()