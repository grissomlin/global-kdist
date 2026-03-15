# scripts/debug_uk_scale_suspects.py
# -*- coding: utf-8 -*-
"""
Debug / list / optionally fix scale suspects for UK dayK CSVs.
Usage:
  python -m scripts.debug_uk_scale_suspects          # list suspects
  python -m scripts.debug_uk_scale_suspects --fix   # attempt safe fixes (writes CSV backups)
"""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import shutil
import sys
from core.cleaning.scale_uk import detect_scale_candidates, apply_scale_fix

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "data" / "cache_dayk" / "uk"

def _load_csv(p: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(p, encoding='utf-8-sig')
    except Exception:
        return pd.read_csv(p, encoding='utf-8', errors='ignore')

def main(fix: bool):
    if not DATA_DIR.exists():
        print("UK dayK dir not found:", DATA_DIR)
        return

    files = sorted(DATA_DIR.glob("*.csv"))
    total = 0
    suspects_all = []
    for p in files:
        total += 1
        df = _load_csv(p)
        # require date column normalized
        if 'date' not in df.columns and '日期' not in df.columns:
            # try to detect date-like column, else skip
            continue

        # normalize column names to lower-case 'date' etc for detection util
        df2 = df.copy()
        if '日期' in df2.columns and 'date' not in df2.columns:
            df2.rename(columns={'日期': 'date', '開盤':'open','收盤':'close','最高':'high','最低':'low','成交量':'volume'}, inplace=True)
        elif 'Date' in df2.columns and 'date' not in df2.columns:
            df2.rename(columns={'Date':'date'}, inplace=True)

        try:
            cands = detect_scale_candidates(
                df2,
                factor_candidates=(100.0, 0.01),
                tol=0.005,
                require_prev_close_ge=0.0001,
                require_volume_le=1000
            )
        except Exception as e:
            # if parsing failed, skip
            continue

        if not cands.empty:
            # attach filename symbol
            for _, r in cands.iterrows():
                suspects_all.append({
                    "file": p.name,
                    "symbol": p.stem,
                    "date": r['date'],
                    "prev_close": r['prev_close'],
                    "close": r['close'],
                    "factor_est": r['factor_est'],
                    "nearest_int": r['nearest_int'],
                })

            if fix:
                # backup
                bak = p.with_suffix(p.suffix + ".bak")
                if not bak.exists():
                    shutil.copy2(p, bak)
                # apply fix for the indices
                idx_list = list(cands['idx'].astype(int).tolist())
                df_fixed = apply_scale_fix(df2, idx_list, factor=100.0)
                # map column names back if needed and write CSV
                # attempt to preserve original column order where possible
                df_fixed = df_fixed.rename(columns={'date':'date','open':'open','high':'high','low':'low','close':'close','volume':'volume'})
                df_fixed.to_csv(p, index=False, encoding='utf-8-sig')
                print(f"[FIXED] {p.name}: applied {len(idx_list)} fixes (bak created)")

    # summary
    print(f"suspects: {len(suspects_all)}")
    if suspects_all:
        df_s = pd.DataFrame(suspects_all)
        # print top 80
        pd.set_option('display.max_rows', 200)
        print(df_s.sort_values(['symbol','date']).head(200).to_string(index=False))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fix", action="store_true", help="Attempt safe fixes and overwrite CSVs (creates .bak copies)")
    args = ap.parse_args()
    main(fix=args.fix)