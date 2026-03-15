# scripts/export_uk_scale_blacklist.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import pandas as pd

from core.cleaning.scale_uk import detect_scale_candidates, apply_scale_fix

THIS = Path(__file__).resolve()
REPO_ROOT = THIS.parents[1]

CACHE_DAYK = REPO_ROOT / "data" / "cache_dayk"
OUT_DIR = REPO_ROOT / "data" / "blacklists"
OUT_TXT = OUT_DIR / "uk_scale_residual.txt"
OUT_CSV = OUT_DIR / "uk_scale_residual.csv"


def _read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.tz_localize(None)
    for c in ("open", "high", "low", "close"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", type=str, default="uk")
    ap.add_argument("--factor", type=float, default=100.0)
    ap.add_argument("--max-passes", type=int, default=5)
    ap.add_argument("--require-reversion", action="store_true", help="use stricter detector (reference only)")
    args = ap.parse_args()

    market = (args.market or "").strip().lower()
    mdir = CACHE_DAYK / market
    if not mdir.exists():
        raise SystemExit(f"Market dayK dir not found: {mdir}")

    files = sorted([p for p in mdir.glob("*.csv") if p.is_file() and not p.name.startswith("_")])

    rows: List[Tuple[str, int, int]] = []
    residual_files: List[str] = []

    for p in files:
        try:
            df = _read_csv(p)
            if df is None or df.empty:
                continue

            # before
            cand_before = detect_scale_candidates(
                df,
                mode="near_factor",
                factor=float(args.factor),
                require_reversion=bool(args.require_reversion),
            )
            b = int(len(cand_before))

            if b == 0:
                continue

            # apply fix (no reversion dependency)
            fixed = apply_scale_fix(
                df,
                mode="near_factor",
                factor=float(args.factor),
                require_reversion=False,
                max_passes=int(args.max_passes),
                return_stats=False,
            )

            cand_after = detect_scale_candidates(
                fixed,
                mode="near_factor",
                factor=float(args.factor),
                require_reversion=bool(args.require_reversion),
            )
            a = int(len(cand_after))

            rows.append((p.name, b, a))
            if a > 0:
                residual_files.append(p.name)

        except Exception:
            continue

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # write txt (one filename per line)
    residual_files_sorted = sorted(set(residual_files))
    OUT_TXT.write_text("\n".join(residual_files_sorted) + ("\n" if residual_files_sorted else ""), encoding="utf-8")

    # write csv (details)
    df_out = pd.DataFrame(rows, columns=["file", "before", "after"]).sort_values(["after", "before", "file"], ascending=[False, False, True])
    df_out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    print(f"[OK] scanned files={len(files)}")
    print(f"[OK] candidates_files={len(df_out)}")
    print(f"[OK] residual_files={len(residual_files_sorted)}")
    print(f"[OUT] {OUT_TXT}")
    print(f"[OUT] {OUT_CSV}")


if __name__ == "__main__":
    main()