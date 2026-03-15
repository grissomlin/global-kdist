# scripts/build_market_distribution_summaries.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data"
MARKETS_YAML = ROOT / "configs" / "markets.yaml"


# ============================================================
# Config helpers
# ============================================================
def _load_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing config file: {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _load_enabled_markets() -> List[str]:
    cfg = _load_yaml(MARKETS_YAML)
    markets = cfg.get("markets", {}) or {}
    if not isinstance(markets, dict):
        raise TypeError("configs/markets.yaml: 'markets' must be a dict")

    out: List[str] = []
    for code, m in markets.items():
        if not isinstance(m, dict):
            continue
        if bool(m.get("enabled", False)):
            out.append(str(code).strip().lower())
    return out


# ============================================================
# Labels / period helpers
# ============================================================
def _period_label(freq_name: str, period: int) -> str:
    if freq_name == "W":
        y = int(period // 100)
        w = int(period % 100)
        return f"{y}W{w:02d}"
    if freq_name == "M":
        y = int(period // 100)
        m = int(period % 100)
        return f"{y}-{m:02d}"
    return str(int(period))


def _make_period_code(freq_name: str, dt: pd.Series) -> pd.Series:
    if freq_name == "W":
        iso = dt.dt.isocalendar()
        return (iso.year.astype(str) + iso.week.astype(str).str.zfill(2)).astype(int)

    if freq_name == "M":
        return (dt.dt.year.astype(str) + dt.dt.month.astype(str).str.zfill(2)).astype(int)

    return dt.dt.year.astype(int)


def _ret_col_for_freq(freq_name: str, metric: str) -> str:
    # metric: high / close / low
    metric = str(metric).strip().lower()
    if freq_name == "W":
        return {"high": "ret_high_W", "close": "ret_close_W", "low": "ret_low_W"}[metric]
    if freq_name == "M":
        return {"high": "ret_high_M", "close": "ret_close_M", "low": "ret_low_M"}[metric]
    return {"high": "ret_high_Y", "close": "ret_close_Y", "low": "ret_low_Y"}[metric]


def _freq_dir_name(freq_name: str) -> str:
    return {"W": "weekK", "M": "monthK", "Y": "yearK"}[freq_name]


# ============================================================
# Bin helpers
# ============================================================
def make_bins_10pct() -> np.ndarray:
    # [-100, -90, ..., 100, 110]
    return np.append(np.arange(-100, 100 + 1e-9, 10.0), 110.0)


def make_bins_100pct() -> Tuple[List[Tuple[float, float, str]], str]:
    """
    Custom bins:
      negative: [-100, 0)
      positive: [0,100), [100,200), ..., [900,1000), [1000,+inf)
    """
    bins: List[Tuple[float, float, str]] = []
    bins.append((-100.0, 0.0, "-100~0%"))
    for lo in range(0, 1000, 100):
        hi = lo + 100
        bins.append((float(lo), float(hi), f"{lo}~{hi}%"))
    overflow_label = "1000%+"
    return bins, overflow_label


def build_bin_rows_10pct(
    arr_pct: np.ndarray,
    *,
    period: int,
    period_label: str,
    universe: int,
) -> List[Dict]:
    rows: List[Dict] = []

    bins = make_bins_10pct()
    clipped = np.clip(arr_pct, -100.0, bins[-1])
    cnt, edges = np.histogram(clipped, bins=bins)

    for lo, hi, c in zip(edges[:-1], edges[1:], cnt):
        lo_i = int(round(lo))
        hi_i = int(round(hi))
        if lo_i == 100:
            lab = "100%+"
        else:
            lab = f"{lo_i}~{hi_i}%"

        pct = (float(c) / universe * 100.0) if universe else 0.0
        rows.append(
            {
                "Period": int(period),
                "PeriodLabel": period_label,
                "BinMode": "10pct",
                "bin_left": float(lo),
                "bin_right": float(hi),
                "bin_label": lab,
                "count": int(c),
                "pct": float(pct),
                "universe": int(universe),
            }
        )
    return rows


def build_bin_rows_100pct(
    arr_pct: np.ndarray,
    *,
    period: int,
    period_label: str,
    universe: int,
) -> List[Dict]:
    rows: List[Dict] = []
    bins, overflow_label = make_bins_100pct()

    arr = np.asarray(arr_pct, dtype=float)

    for lo, hi, lab in bins:
        if lab == "-100~0%":
            # include all negatives clipped at -100
            c = int(((arr < 0.0) & (arr >= -100.0)).sum() + (arr < -100.0).sum())
        else:
            c = int(((arr >= lo) & (arr < hi)).sum())

        pct = (float(c) / universe * 100.0) if universe else 0.0
        rows.append(
            {
                "Period": int(period),
                "PeriodLabel": period_label,
                "BinMode": "100pct",
                "bin_left": float(lo),
                "bin_right": float(hi),
                "bin_label": lab,
                "count": int(c),
                "pct": float(pct),
                "universe": int(universe),
            }
        )

    c_over = int((arr >= 1000.0).sum())
    pct_over = (float(c_over) / universe * 100.0) if universe else 0.0
    rows.append(
        {
            "Period": int(period),
            "PeriodLabel": period_label,
            "BinMode": "100pct",
            "bin_left": 1000.0,
            "bin_right": np.inf,
            "bin_label": overflow_label,
            "count": int(c_over),
            "pct": float(pct_over),
            "universe": int(universe),
        }
    )

    return rows


# ============================================================
# Read data
# ============================================================
def read_market_period_returns(
    market_code: str,
    *,
    freq_name: str,
    metric: str,
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    in_dir = DATA_ROOT / "derived" / _freq_dir_name(freq_name) / market_code
    if not in_dir.exists():
        raise FileNotFoundError(f"Input dir not found: {in_dir}")

    files = sorted(in_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV found: {in_dir}")

    ret_col = _ret_col_for_freq(freq_name, metric)
    rows = []

    for fp in tqdm(files, desc=f"Read {market_code} {_freq_dir_name(freq_name)}", unit="file"):
        try:
            df = pd.read_csv(fp, usecols=["date", ret_col])
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date", ret_col]).copy()
            if df.empty:
                continue

            df["ret_pct"] = pd.to_numeric(df[ret_col], errors="coerce") * 100.0
            df = df.dropna(subset=["ret_pct"]).copy()
            if df.empty:
                continue

            df = df[(df["date"].dt.year >= start_year) & (df["date"].dt.year <= end_year)].copy()
            if df.empty:
                continue

            df["StockID"] = fp.stem
            df["Period"] = _make_period_code(freq_name, df["date"])
            rows.append(df[["date", "StockID", "Period", "ret_pct"]].copy())
        except Exception:
            continue

    if not rows:
        raise RuntimeError(
            f"No readable rows for market={market_code}, freq={freq_name}, metric={metric}"
        )

    out = pd.concat(rows, ignore_index=True)
    return out


# ============================================================
# Summary builder
# ============================================================
def build_summary_and_bins(
    market_code: str,
    *,
    freq_name: str,
    metric: str,
    start_year: int,
    end_year: int,
    top_n: int = 5,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = read_market_period_returns(
        market_code,
        freq_name=freq_name,
        metric=metric,
        start_year=start_year,
        end_year=end_year,
    )

    summary_rows: List[Dict] = []
    bin_rows: List[Dict] = []

    for period, sub in df.groupby("Period", sort=True):
        arr = sub["ret_pct"].to_numpy(dtype=float)
        universe = int(len(arr))
        if universe <= 0:
            continue

        period_label = _period_label(freq_name, int(period))

        mean_ret = float(np.mean(arr))
        median_ret = float(np.median(arr))
        std_ret = float(np.std(arr, ddof=0))

        pct_up_gt_100 = float(np.mean(arr > 100.0) * 100.0)
        pct_down_lt_m50 = float(np.mean(arr < -50.0) * 100.0)
        pct_ge_50 = float(np.mean(arr >= 50.0) * 100.0)
        pct_ge_100 = float(np.mean(arr >= 100.0) * 100.0)
        pct_ge_200 = float(np.mean(arr >= 200.0) * 100.0)
        pct_ge_500 = float(np.mean(arr >= 500.0) * 100.0)
        pct_ge_1000 = float(np.mean(arr >= 1000.0) * 100.0)
        pct_le_m20 = float(np.mean(arr <= -20.0) * 100.0)
        pct_le_m50 = float(np.mean(arr <= -50.0) * 100.0)

        items = list(zip(sub["StockID"].astype(str), sub["ret_pct"].astype(float)))
        items_sorted = sorted(items, key=lambda x: x[1], reverse=True)

        top1_stock, top1_ret = items_sorted[0]
        topn = items_sorted[:top_n]
        topn_stocks = "|".join([x[0] for x in topn])
        topn_rets = "|".join([f"{x[1]:.1f}" for x in topn])

        top5_cut = float(np.sort(arr)[-5]) if universe >= 5 else float(np.min(arr))

        summary_rows.append(
            {
                "Market": market_code.upper(),
                "Freq": freq_name,
                "Metric": metric,
                "Period": int(period),
                "PeriodLabel": period_label,
                "Universe": universe,
                "MeanRet_%": round(mean_ret, 4),
                "MedianRet_%": round(median_ret, 4),
                "StdRet_%": round(std_ret, 4),
                "Pct_Up_gt_100": round(pct_up_gt_100, 3),
                "Pct_Down_lt_-50": round(pct_down_lt_m50, 3),
                "Pct_ge_50": round(pct_ge_50, 3),
                "Pct_ge_100": round(pct_ge_100, 3),
                "Pct_ge_200": round(pct_ge_200, 3),
                "Pct_ge_500": round(pct_ge_500, 3),
                "Pct_ge_1000": round(pct_ge_1000, 3),
                "Pct_le_-20": round(pct_le_m20, 3),
                "Pct_le_-50": round(pct_le_m50, 3),
                "Top1Stock": top1_stock,
                "Top1Ret_%": round(top1_ret, 2),
                f"Top{top_n}Stocks": topn_stocks,
                f"Top{top_n}Rets_%": topn_rets,
                "Top5CutRet_%": round(top5_cut, 2),
            }
        )

        bin_rows.extend(
            build_bin_rows_10pct(
                arr,
                period=int(period),
                period_label=period_label,
                universe=universe,
            )
        )

        bin_rows.extend(
            build_bin_rows_100pct(
                arr,
                period=int(period),
                period_label=period_label,
                universe=universe,
            )
        )

    df_summary = pd.DataFrame(summary_rows).sort_values("Period").reset_index(drop=True)
    df_bins = pd.DataFrame(bin_rows)

    if not df_bins.empty:
        df_bins.insert(0, "Market", market_code.upper())
        df_bins.insert(1, "Freq", freq_name)
        df_bins.insert(2, "Metric", metric)
        df_bins = df_bins.sort_values(["Period", "BinMode", "bin_left"], ascending=[True, True, True]).reset_index(drop=True)

    return df_summary, df_bins


# ============================================================
# Export
# ============================================================
def export_market(
    market_code: str,
    *,
    start_year: int,
    end_year: int,
    metric: str,
    top_n: int,
):
    out_dir = DATA_ROOT / "research" / "market_summary" / market_code
    out_dir.mkdir(parents=True, exist_ok=True)

    for freq_name in ("W", "M", "Y"):
        print(f"\n📊 Building summary: market={market_code} freq={freq_name} metric={metric}")
        df_summary, df_bins = build_summary_and_bins(
            market_code,
            freq_name=freq_name,
            metric=metric,
            start_year=start_year,
            end_year=end_year,
            top_n=top_n,
        )

        out_summary = out_dir / f"{market_code}_{freq_name}_{metric}_summary_{start_year}_{end_year}.csv"
        out_bins = out_dir / f"{market_code}_{freq_name}_{metric}_bins_{start_year}_{end_year}.csv"

        df_summary.to_csv(out_summary, index=False, encoding="utf-8-sig")
        df_bins.to_csv(out_bins, index=False, encoding="utf-8-sig")

        print(f"✅ Saved summary: {out_summary} rows={len(df_summary)}")
        print(f"✅ Saved bins   : {out_bins} rows={len(df_bins)}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--markets",
        default="",
        help="Comma-separated market codes, e.g. hk,cn,th. Empty = all enabled markets",
    )
    ap.add_argument("--start-year", type=int, default=2020)
    ap.add_argument("--end-year", type=int, default=2025)
    ap.add_argument(
        "--metric",
        default="close",
        choices=["high", "close", "low"],
        help="Use return metric high/close/low",
    )
    ap.add_argument("--top-n", type=int, default=5)
    return ap.parse_args()


def main():
    args = parse_args()

    if args.markets.strip():
        markets = [x.strip().lower() for x in args.markets.split(",") if x.strip()]
    else:
        markets = _load_enabled_markets()

    if not markets:
        raise RuntimeError("No markets to process.")

    print("Markets:", markets)
    print("Years:", args.start_year, "-", args.end_year)
    print("Metric:", args.metric)

    for market_code in markets:
        try:
            export_market(
                market_code,
                start_year=args.start_year,
                end_year=args.end_year,
                metric=args.metric,
                top_n=args.top_n,
            )
        except Exception as e:
            print(f"❌ [{market_code}] {e}")


if __name__ == "__main__":
    main()