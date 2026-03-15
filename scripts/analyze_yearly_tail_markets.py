# scripts/analyze_yearly_tail_markets.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

DATA_ROOT = ROOT / "data"
MARKETS_YAML = ROOT / "configs" / "markets.yaml"
OUT_DIR = DATA_ROOT / "research" / "yearly_tail_charts"
OUT_DIR.mkdir(parents=True, exist_ok=True)


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
# Read yearly returns
# ============================================================
def read_market_yearly_close_returns(
    market_code: str,
    *,
    start_year: int = 2020,
    end_year: int = 2025,
    ret_col: str = "ret_close_Y",
) -> pd.DataFrame:
    year_dir = DATA_ROOT / "derived" / "yearK" / market_code
    if not year_dir.exists():
        raise FileNotFoundError(f"yearK dir not found: {year_dir}")

    csvs = sorted(year_dir.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No yearK csv found: {year_dir}")

    rows = []

    for p in csvs:
        try:
            df = pd.read_csv(p, usecols=["date", ret_col])
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date", ret_col]).copy()
            if df.empty:
                continue

            df = df[(df["date"].dt.year >= start_year) & (df["date"].dt.year <= end_year)].copy()
            if df.empty:
                continue

            s = pd.to_numeric(df[ret_col], errors="coerce") * 100.0
            tmp = pd.DataFrame({
                "date": df["date"].values,
                "ret_pct": s.values,
            }).dropna(subset=["ret_pct"])

            if tmp.empty:
                continue

            tmp["Market"] = market_code.upper()
            tmp["Period"] = tmp["date"].dt.year.astype(str)
            rows.append(tmp[["Market", "date", "Period", "ret_pct"]])
        except Exception:
            continue

    if not rows:
        raise RuntimeError(f"No readable yearly rows for market={market_code}")

    return pd.concat(rows, ignore_index=True)


# ============================================================
# Bins
# ============================================================
def build_bins_10pct(arr: np.ndarray) -> pd.DataFrame:
    bins = np.append(np.arange(-100, 100 + 1e-9, 10.0), 110.0)
    clipped = np.clip(arr, -100.0, bins[-1])
    cnt, edges = np.histogram(clipped, bins=bins)

    rows = []
    total = len(arr)
    for lo, hi, c in zip(edges[:-1], edges[1:], cnt):
        lo_i = int(round(lo))
        hi_i = int(round(hi))
        label = "100%+" if lo_i == 100 else f"{lo_i}~{hi_i}%"
        rows.append({
            "BinMode": "10pct",
            "bin_left": float(lo),
            "bin_right": float(hi),
            "bin_label": label,
            "count": int(c),
            "pct": float(c / total * 100.0) if total else 0.0,
        })
    return pd.DataFrame(rows)


def build_bins_100pct(arr: np.ndarray) -> pd.DataFrame:
    rows = []
    total = len(arr)

    neg = int((arr < 0).sum())
    rows.append({
        "BinMode": "100pct",
        "bin_left": -100.0,
        "bin_right": 0.0,
        "bin_label": "-100~0%",
        "count": neg,
        "pct": float(neg / total * 100.0) if total else 0.0,
    })

    for lo in range(0, 1000, 100):
        hi = lo + 100
        c = int(((arr >= lo) & (arr < hi)).sum())
        rows.append({
            "BinMode": "100pct",
            "bin_left": float(lo),
            "bin_right": float(hi),
            "bin_label": f"{lo}~{hi}%",
            "count": c,
            "pct": float(c / total * 100.0) if total else 0.0,
        })

    c_over = int((arr >= 1000.0).sum())
    rows.append({
        "BinMode": "100pct",
        "bin_left": 1000.0,
        "bin_right": np.inf,
        "bin_label": "1000%+",
        "count": c_over,
        "pct": float(c_over / total * 100.0) if total else 0.0,
    })

    return pd.DataFrame(rows)


# ============================================================
# Summary
# ============================================================
def build_market_summary(df_all: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for market, g in df_all.groupby("Market", sort=True):
        arr = g["ret_pct"].to_numpy(dtype=float)
        if len(arr) == 0:
            continue

        rows.append({
            "Market": market,
            "Samples": int(len(arr)),
            "MeanRet_%": float(np.mean(arr)),
            "MedianRet_%": float(np.median(arr)),
            "StdRet_%": float(np.std(arr, ddof=0)),
            "P90_%": float(np.percentile(arr, 90)),
            "P95_%": float(np.percentile(arr, 95)),
            "P99_%": float(np.percentile(arr, 99)),
            "Max_%": float(np.max(arr)),
            "Pct_ge_50": float((arr >= 50).mean() * 100.0),
            "Pct_ge_100": float((arr >= 100).mean() * 100.0),
            "Pct_ge_200": float((arr >= 200).mean() * 100.0),
            "Pct_ge_500": float((arr >= 500).mean() * 100.0),
            "Pct_ge_1000": float((arr >= 1000).mean() * 100.0),
        })

    out = pd.DataFrame(rows).sort_values("P99_%", ascending=False).reset_index(drop=True)
    return out


def build_article_highlights(df_summary: pd.DataFrame) -> pd.DataFrame:
    def _winner(col: str) -> Tuple[str, float]:
        d = df_summary.sort_values(col, ascending=False).iloc[0]
        return str(d["Market"]), float(d[col])

    metrics = [
        "P99_%",
        "Max_%",
        "Pct_ge_100",
        "Pct_ge_200",
        "Pct_ge_500",
        "Pct_ge_1000",
        "StdRet_%",
    ]

    rows = []
    for m in metrics:
        market, value = _winner(m)
        rows.append({
            "Metric": m,
            "WinnerMarket": market,
            "WinnerValue": value,
        })

    return pd.DataFrame(rows)


# ============================================================
# Plots
# ============================================================
def plot_yearly_tail_percentiles(df: pd.DataFrame, out_path: Path) -> None:
    d = df.sort_values("P99_%", ascending=False).reset_index(drop=True)
    x = np.arange(len(d))
    width = 0.22

    plt.figure(figsize=(14, 7))
    plt.bar(x - width, d["P90_%"], width=width, label="P90")
    plt.bar(x, d["P95_%"], width=width, label="P95")
    plt.bar(x + width, d["P99_%"], width=width, label="P99")

    plt.xticks(x, d["Market"])
    plt.ylabel("Yearly close return (%)")
    plt.title("Top Tail of Yearly Return Distribution Across Markets (2020–2025)")
    plt.legend()
    plt.grid(axis="y", alpha=0.25)

    for i, v in enumerate(d["P99_%"]):
        plt.text(i + width, v, f"{v:.1f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()


def plot_yearly_extreme_incidence(df: pd.DataFrame, out_path: Path) -> None:
    d = df.sort_values("Pct_ge_100", ascending=False).reset_index(drop=True)
    x = np.arange(len(d))
    width = 0.26

    plt.figure(figsize=(14, 7))
    plt.bar(x - width, d["Pct_ge_100"], width=width, label="≥100%")
    plt.bar(x, d["Pct_ge_200"], width=width, label="≥200%")
    plt.bar(x + width, d["Pct_ge_500"], width=width, label="≥500%")

    plt.xticks(x, d["Market"])
    plt.ylabel("Share of yearly observations (%)")
    plt.title("Extreme Yearly Return Incidence Across Markets (2020–2025)")
    plt.legend()
    plt.grid(axis="y", alpha=0.25)

    for i, v in enumerate(d["Pct_ge_100"]):
        plt.text(i - width, v, f"{v:.2f}%", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()


def plot_yearly_tail_scatter(df: pd.DataFrame, out_path: Path) -> None:
    d = df.sort_values("P99_%", ascending=False).reset_index(drop=True)

    plt.figure(figsize=(12, 7))
    plt.scatter(d["P99_%"], d["Pct_ge_100"], s=90)

    for _, r in d.iterrows():
        plt.text(r["P99_%"], r["Pct_ge_100"], f" {r['Market']}", va="center", fontsize=10)

    plt.xlabel("P99 yearly return (%)")
    plt.ylabel("Share of yearly returns ≥ 100% (%)")
    plt.title("Market Tail Thickness: Yearly P99 vs Extreme Incidence (2020–2025)")
    plt.grid(alpha=0.25)

    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()


# ============================================================
# Main
# ============================================================
def main():
    markets = _load_enabled_markets()

    dfs = []
    for m in markets:
        try:
            dfs.append(read_market_yearly_close_returns(m, start_year=2020, end_year=2025))
        except Exception as e:
            print(f"⚠️ Skip {m}: {e}")

    if not dfs:
        raise RuntimeError("No yearly market data loaded.")

    df_all = pd.concat(dfs, ignore_index=True)

    # summary
    df_summary = build_market_summary(df_all)
    df_summary.to_csv(OUT_DIR / "yearly_tail_summary_2020_2025.csv", index=False, encoding="utf-8-sig")

    # article highlights
    df_hi = build_article_highlights(df_summary)
    df_hi.to_csv(OUT_DIR / "yearly_tail_article_highlights_2020_2025.csv", index=False, encoding="utf-8-sig")

    # market-level bins
    bins10_rows = []
    bins100_rows = []

    for market, g in df_all.groupby("Market", sort=True):
        arr = g["ret_pct"].to_numpy(dtype=float)
        b10 = build_bins_10pct(arr)
        b10.insert(0, "Market", market)
        bins10_rows.append(b10)

        b100 = build_bins_100pct(arr)
        b100.insert(0, "Market", market)
        bins100_rows.append(b100)

    df_bins10 = pd.concat(bins10_rows, ignore_index=True)
    df_bins100 = pd.concat(bins100_rows, ignore_index=True)

    df_bins10.to_csv(OUT_DIR / "yearly_tail_bins_10pct_2020_2025.csv", index=False, encoding="utf-8-sig")
    df_bins100.to_csv(OUT_DIR / "yearly_tail_bins_100pct_2020_2025.csv", index=False, encoding="utf-8-sig")

    # plots
    plot_yearly_tail_percentiles(
        df_summary,
        OUT_DIR / "yearly_tail_percentiles_2020_2025.png",
    )
    plot_yearly_extreme_incidence(
        df_summary,
        OUT_DIR / "yearly_tail_incidence_2020_2025.png",
    )
    plot_yearly_tail_scatter(
        df_summary,
        OUT_DIR / "yearly_tail_scatter_2020_2025.png",
    )

    print("✅ Saved:")
    print(" ", OUT_DIR / "yearly_tail_summary_2020_2025.csv")
    print(" ", OUT_DIR / "yearly_tail_article_highlights_2020_2025.csv")
    print(" ", OUT_DIR / "yearly_tail_bins_10pct_2020_2025.csv")
    print(" ", OUT_DIR / "yearly_tail_bins_100pct_2020_2025.csv")
    print(" ", OUT_DIR / "yearly_tail_percentiles_2020_2025.png")
    print(" ", OUT_DIR / "yearly_tail_incidence_2020_2025.png")
    print(" ", OUT_DIR / "yearly_tail_scatter_2020_2025.png")
    print("\n--- preview ---")
    print(df_summary.head(12).to_string(index=False))


if __name__ == "__main__":
    main()