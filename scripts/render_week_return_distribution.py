# scripts/render_week_return_distribution.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from datetime import date as _date
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from tqdm import tqdm
import yaml


# ============================================================
# Global dark style
# ============================================================
plt.style.use("dark_background")

BG_COLOR = "#0F1117"
AX_BG_COLOR = "#121212"
GRID_COLOR = "#333844"
TEXT_COLOR = "#e0e0e0"
LIGHT_TEXT = "#b0b0b0"
ACCENT_COLOR = "#00aaff"
NEG_COLOR = "#c0392b"

# ============================================================
# Histogram bins
# ============================================================
BIN = 10.0
XMIN, XMAX = -100, 100
BINS = np.append(np.arange(XMIN, XMAX + 1e-6, BIN), XMAX + BIN)

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data"
MARKETS_YAML = ROOT / "configs" / "markets.yaml"


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing config file: {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _cfg_str(d: dict, key: str, default: str = "") -> str:
    v = d.get(key, default)
    if v is None:
        return default
    s = str(v).strip()
    return s if s else default


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


def _iso_week_to_range(period: int, trading_week_only: bool = True):
    s = str(int(period))
    y = int(s[:4])
    w = int(s[4:])
    start = _date.fromisocalendar(y, w, 1)
    end = _date.fromisocalendar(y, w, 5 if trading_week_only else 7)
    return start, end


def _fmt_ymd(d: _date) -> str:
    return f"{d.year:04d}-{d.month:02d}-{d.day:02d}"


def _wk_label(period: int) -> str:
    y = int(period // 100)
    w = int(period % 100)
    return f"{y}W{w:02d}"


def _formula_text(ret_label: str) -> str:
    formula_map = {
        "ret_high_W": "Ret = (High_t - Close_{t-1}) / Close_{t-1}",
        "ret_close_W": "Ret = (Close_t - Close_{t-1}) / Close_{t-1}",
        "ret_low_W": "Ret = (Low_t - Close_{t-1}) / Close_{t-1}",
    }
    return formula_map.get(ret_label, "Ret = (X_t - Close_{t-1}) / Close_{t-1}")


def build_bin_table_only(arr_pct: np.ndarray, bins: np.ndarray, *, xmin: float, xmax: float) -> str:
    arr_pct = np.asarray(arr_pct, dtype=float)
    total = int(len(arr_pct))
    if total <= 0:
        return "No data"

    clipped = np.clip(arr_pct, xmin, bins[-1])
    cnt, edges = np.histogram(clipped, bins=bins)

    rows = []
    for lo, hi, c in zip(edges[:-1], edges[1:], cnt):
        lo_i, hi_i = int(lo), int(hi)
        if lo_i == int(xmax) and hi_i == int(xmax + (hi - lo)):
            lab = f"{int(xmax)}%+"
        else:
            lab = f"{lo_i}~{hi_i}%"
        pct = (c / total * 100.0) if total else 0.0
        rows.append((lab, int(c), pct, lo_i, hi_i))

    def _key(r):
        _, _, _, lo_i, _ = r
        is_overflow = 1 if lo_i == int(xmax) else 0
        return (is_overflow, lo_i)

    rows_sorted = sorted(rows, key=_key, reverse=True)

    lines = []
    lines.append(f"{'Bin':<10} | {'Cnt':>5} | {'Pct':>6}")
    lines.append("-" * 28)
    for lab, c, pct, *_ in rows_sorted:
        lines.append(f"{lab:<10} | {c:>5} | {pct:>5.1f}%")

    return "\n".join(lines)


def _market_display(code: str) -> str:
    return code.upper()


def _read_market_weekk(
    market_code: str,
    *,
    start_year: int,
    end_year: int,
    ret_col: str = "ret_high_W",
) -> pd.DataFrame:
    """
    Read all weekly csv from:
      data/derived/weekK/{market_code}/*.csv

    Returns columns:
      date, StockID, ret_pct, Period
    """
    week_dir = DATA_ROOT / "derived" / "weekK" / market_code
    if not week_dir.exists():
        raise FileNotFoundError(f"weekK dir not found: {week_dir}")

    csvs = sorted(week_dir.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No weekK csv found: {week_dir}")

    rows = []
    for p in tqdm(csvs, desc=f"Read {market_code} weekK", unit="file"):
        try:
            d = pd.read_csv(p, usecols=["date", ret_col])
            d["StockID"] = p.stem
            d["date"] = pd.to_datetime(d["date"], errors="coerce")
            rows.append(d)
        except Exception:
            continue

    if not rows:
        raise RuntimeError(f"No readable rows for market={market_code}. Check files/columns.")

    df = pd.concat(rows, ignore_index=True)
    df = df.dropna(subset=["date", ret_col]).copy()
    df["ret_pct"] = pd.to_numeric(df[ret_col], errors="coerce") * 100.0
    df = df.dropna(subset=["ret_pct"]).copy()

    df = df[(df["date"].dt.year >= start_year) & (df["date"].dt.year <= end_year)].copy()

    iso = df["date"].dt.isocalendar()
    df["Period"] = (iso.year.astype(str) + iso.week.astype(str).str.zfill(2)).astype(int)

    return df[["date", "StockID", "ret_pct", "Period"]].copy()


def _plot_one_week(
    market_code: str,
    period: int,
    dsub: pd.DataFrame,
    *,
    out_dir: Path,
    top_n: int = 10,
    metric_name: str = "High Return",
    ret_label: str = "ret_high_W",
):
    arr = dsub["ret_pct"].to_numpy(dtype=float)
    if len(arr) == 0:
        return

    arrc = np.clip(arr, XMIN, BINS[-1])
    cnt, edges = np.histogram(arrc, bins=BINS)
    lefts = edges[:-1]
    widths = np.diff(edges)

    colors = [NEG_COLOR if lo < 0 else ACCENT_COLOR for lo in lefts]

    top = dsub.sort_values("ret_pct", ascending=False).head(top_n)
    top_lines = [
        f"{i+1:>2}. {sid:>12} : {v:+.1f}%"
        for i, (sid, v) in enumerate(zip(top["StockID"], top["ret_pct"]))
    ]

    bin_table_text = build_bin_table_only(arr, BINS, xmin=XMIN, xmax=XMAX)
    formula_text = _formula_text(ret_label)

    fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
    fig.patch.set_facecolor(BG_COLOR)

    ax = fig.add_axes([0.06, 0.18, 0.68, 0.72])
    ax2 = fig.add_axes([0.76, 0.18, 0.20, 0.72])
    ax.set_facecolor(AX_BG_COLOR)
    ax2.axis("off")

    ax.bar(
        lefts,
        cnt,
        width=widths * 0.95,
        align="edge",
        color=colors,
        edgecolor=colors,
        alpha=0.92,
    )

    mmax = int(cnt.max()) if len(cnt) else 1
    pad = max(1, int(0.10 * mmax))
    ax.set_ylim(0, mmax + pad + int(0.05 * mmax) + 1)

    tot = len(arr)
    for h, lo, hi in zip(cnt, edges[:-1], edges[1:]):
        if h <= 0:
            continue
        cx = lo + (hi - lo) / 2.0
        ax.text(
            cx, h / 2, f"{int(h)}",
            ha="center", va="center",
            fontsize=14, color="white", fontweight="bold"
        )
        ax.text(
            cx, h + pad * 0.15, f"{h / tot * 100:4.1f}%",
            ha="center", va="bottom",
            fontsize=12, color=LIGHT_TEXT
        )

    for t in np.arange(XMIN, XMAX + 1e-9, BIN):
        ax.axvline(t, color=GRID_COLOR, linestyle="--", linewidth=1, alpha=0.4)

    ax.set_xlim(XMIN, XMAX + BIN)
    xticks = np.arange(XMIN, XMAX + 1e-9, BIN)
    ax.set_xticks(xticks)

    xtick_labels = []
    for x in xticks:
        if int(x) == int(XMAX):
            xtick_labels.append(f"{int(XMAX)}%+")
        else:
            xtick_labels.append(f"{int(x)}%")
    ax.set_xticklabels(xtick_labels, fontsize=11, color=TEXT_COLOR)

    wk_s, wk_e = _iso_week_to_range(period, trading_week_only=True)
    mkt = _market_display(market_code)

    ax.set_title(
        f"{mkt} Weekly {metric_name} Distribution — {_wk_label(period)}\n"
        f"({_fmt_ymd(wk_s)} to {_fmt_ymd(wk_e)})",
        fontsize=18, color=TEXT_COLOR, pad=12
    )
    ax.set_xlabel("Return bins (%) — step = 10%", fontsize=14, color=TEXT_COLOR, labelpad=10)
    ax.set_ylabel("Number of stocks", fontsize=14, color=TEXT_COLOR)

    legend_items = [
        Patch(facecolor=NEG_COLOR, edgecolor=NEG_COLOR, label="Negative return bins"),
        Patch(facecolor=ACCENT_COLOR, edgecolor=ACCENT_COLOR, label="Non-negative return bins"),
        Patch(facecolor="white", edgecolor="0.6", label="Bar: white=count, gray=%"),
    ]
    ax.legend(
        handles=legend_items,
        loc="upper right",
        framealpha=0.8,
        facecolor=AX_BG_COLOR,
        edgecolor=GRID_COLOR,
        labelcolor=TEXT_COLOR,
    )

    right_text = (
        f"{metric_name} formula\n"
        f"{formula_text}\n"
        f"Source column: {ret_label}\n"
        f"Universe: {len(arr)} stocks\n"
        "\n"
        f"Top {top_n} movers\n"
        + "\n".join(top_lines)
        + "\n\n"
        "Bin stats (count / %)\n"
        + bin_table_text
    )

    ax2.text(
        0.0, 1.0, right_text,
        fontsize=11.2,
        va="top",
        family="monospace",
        color=TEXT_COLOR,
    )

    out = out_dir / f"{market_code}_week_{period}_dark.png"
    fig.savefig(out, bbox_inches="tight", facecolor=fig.get_facecolor(), dpi=100)
    plt.close(fig)


def render_market(
    market_code: str,
    *,
    start_year: int,
    end_year: int,
    top_n: int,
    ret_col: str,
):
    metric_name_map = {
        "ret_high_W": "High Return",
        "ret_close_W": "Close Return",
        "ret_low_W": "Low Return",
    }
    metric_name = metric_name_map.get(ret_col, ret_col)

    df = _read_market_weekk(
        market_code,
        start_year=start_year,
        end_year=end_year,
        ret_col=ret_col,
    )

    if df.empty:
        print(f"[WARN] market={market_code}: no rows in {start_year}-{end_year}")
        return

    weeks = sorted(df["Period"].unique().tolist())
    print(
        f"[{market_code}] Weeks={len(weeks)} "
        f"Rows={len(df)} "
        f"Stocks={df['StockID'].nunique()}"
    )

    out_dir = DATA_ROOT / "derived_images" / "weekK" / market_code / f"{start_year}_{end_year}_{ret_col}"
    out_dir.mkdir(parents=True, exist_ok=True)

    for wk in tqdm(weeks, desc=f"Render {market_code}", unit="week"):
        dsub = df[df["Period"] == wk][["StockID", "ret_pct"]].copy()
        _plot_one_week(
            market_code,
            int(wk),
            dsub,
            out_dir=out_dir,
            top_n=top_n,
            metric_name=metric_name,
            ret_label=ret_col,
        )

    print(f"✅ Done [{market_code}] -> {out_dir}")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--markets",
        default="",
        help="Comma-separated market codes, e.g. hk,tw,jp. Empty = all enabled markets in configs/markets.yaml",
    )
    ap.add_argument("--start-year", type=int, default=2020)
    ap.add_argument("--end-year", type=int, default=2025)
    ap.add_argument("--top-n", type=int, default=10)
    ap.add_argument(
        "--ret-col",
        default="ret_high_W",
        choices=["ret_high_W", "ret_close_W", "ret_low_W"],
        help="Which weekly return column to render",
    )
    return ap.parse_args()


def main():
    args = parse_args()

    if args.markets.strip():
        markets = [x.strip().lower() for x in args.markets.split(",") if x.strip()]
    else:
        markets = _load_enabled_markets()

    if not markets:
        raise RuntimeError("No markets to render.")

    print("Markets:", markets)
    print("Return column:", args.ret_col)
    print("Year range:", args.start_year, "-", args.end_year)

    for market_code in markets:
        try:
            render_market(
                market_code,
                start_year=args.start_year,
                end_year=args.end_year,
                top_n=args.top_n,
                ret_col=args.ret_col,
            )
        except Exception as e:
            print(f"❌ [{market_code}] {e}")


if __name__ == "__main__":
    main()