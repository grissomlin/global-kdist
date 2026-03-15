# scripts/research_penny_charts.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_INPUT = "research_outputs/penny_stats/market_year_summary.csv"
DEFAULT_OUT_DIR = "research_outputs/penny_stats/charts"


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def read_summary(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        raise RuntimeError(f"Summary CSV is empty: {path}")

    # normalize basic columns
    if "market" not in df.columns or "year" not in df.columns:
        raise RuntimeError(f"Missing required columns in {path}")

    df = df.copy()
    df["market"] = df["market"].astype(str).str.upper()
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df = df[df["year"].notna()].copy()
    df["year"] = df["year"].astype(int)
    return df.sort_values(["market", "year"]).reset_index(drop=True)


def save_line_chart(
    df: pd.DataFrame,
    value_col: str,
    title: str,
    ylabel: str,
    out_path: Path,
    pct: bool = True,
    markets: List[str] | None = None,
) -> None:
    data = df.copy()
    if markets:
        wanted = [m.upper() for m in markets]
        data = data[data["market"].isin(wanted)].copy()

    plt.figure(figsize=(12, 7))
    for market in sorted(data["market"].unique().tolist()):
        g = data[data["market"] == market].sort_values("year")
        if g.empty or value_col not in g.columns:
            continue
        plt.plot(g["year"], g[value_col], marker="o", label=market)

    plt.title(title)
    plt.xlabel("Year")
    plt.ylabel(ylabel)
    plt.xticks(sorted(data["year"].unique().tolist()))
    plt.legend()
    plt.grid(True, alpha=0.3)
    if pct:
        ymin, ymax = plt.ylim()
        plt.ylim(bottom=max(0, ymin))
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()


def save_bar_rank(
    df: pd.DataFrame,
    year: int,
    value_col: str,
    title: str,
    ylabel: str,
    out_path: Path,
    top_n: int = 50,
) -> None:
    data = df[df["year"] == year].copy()
    if data.empty:
        return
    if value_col not in data.columns:
        return

    data = data.sort_values(value_col, ascending=False).head(top_n)

    plt.figure(figsize=(12, 7))
    plt.bar(data["market"], data[value_col])
    plt.title(title)
    plt.xlabel("Market")
    plt.ylabel(ylabel)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()


def save_heatmap(
    df: pd.DataFrame,
    value_col: str,
    title: str,
    out_path: Path,
) -> None:
    if value_col not in df.columns:
        return

    pivot = df.pivot(index="market", columns="year", values=value_col)
    if pivot.empty:
        return

    plt.figure(figsize=(10, max(5, 0.45 * len(pivot.index))))
    plt.imshow(pivot.values, aspect="auto")
    plt.title(title)
    plt.xticks(range(len(pivot.columns)), [str(c) for c in pivot.columns])
    plt.yticks(range(len(pivot.index)), list(pivot.index))
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()


def save_scatter(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    year: int,
    title: str,
    xlabel: str,
    ylabel: str,
    out_path: Path,
) -> None:
    data = df[df["year"] == year].copy()
    if data.empty or x_col not in data.columns or y_col not in data.columns:
        return

    plt.figure(figsize=(10, 7))
    plt.scatter(data[x_col], data[y_col])

    for _, r in data.iterrows():
        plt.annotate(str(r["market"]), (r[x_col], r[y_col]), fontsize=8)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()


def write_chart_notes(df: pd.DataFrame, out_path: Path) -> None:
    lines: List[str] = []
    lines.append("# Penny Charts Notes")
    lines.append("")

    latest_year = int(df["year"].max())
    latest = df[df["year"] == latest_year].copy()

    if not latest.empty:
        def top3(col: str) -> str:
            if col not in latest.columns:
                return ""
            sub = latest[["market", col]].dropna().sort_values(col, ascending=False).head(3)
            return ", ".join([f"{r.market} ({r[col]:.2%})" for _, r in sub.iterrows()])

        if "stocks_ever_penny_ratio" in latest.columns:
            lines.append(f"- {latest_year} 低價股占比前三：{top3('stocks_ever_penny_ratio')}")
        if "penny_day_ratio_all" in latest.columns:
            lines.append(f"- {latest_year} 低價股 stock-day 占比前三：{top3('penny_day_ratio_all')}")
        if "stocks_one_tick_ge_10_ratio" in latest.columns:
            lines.append(f"- {latest_year} 一個 tick >=10% 比例前三：{top3('stocks_one_tick_ge_10_ratio')}")
        if "stocks_one_tick_ge_20_ratio" in latest.columns:
            lines.append(f"- {latest_year} 一個 tick >=20% 比例前三：{top3('stocks_one_tick_ge_20_ratio')}")
        if "stocks_abs_gap_ge_10_ratio" in latest.columns:
            lines.append(f"- {latest_year} abs gap >=10% 比例前三：{top3('stocks_abs_gap_ge_10_ratio')}")
        if "stocks_abs_ret_ge_10_ratio" in latest.columns:
            lines.append(f"- {latest_year} abs return >=10% 比例前三：{top3('stocks_abs_ret_ge_10_ratio')}")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=DEFAULT_INPUT, help="Path to market_year_summary.csv")
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="Output charts directory")
    ap.add_argument("--markets", default="", help="Optional comma-separated market filter, e.g. us,tw,jp,de")
    args = ap.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    df = read_summary(input_path)

    markets = [x.strip().upper() for x in str(args.markets).split(",") if x.strip()]
    if markets:
        df = df[df["market"].isin(markets)].copy()

    # Line charts
    save_line_chart(
        df,
        "stocks_ever_penny_ratio",
        "Stocks Ever Penny Ratio by Market and Year",
        "Ratio",
        out_dir / "line_stocks_ever_penny_ratio.png",
        pct=True,
        markets=markets or None,
    )

    save_line_chart(
        df,
        "penny_day_ratio_all",
        "Penny Stock-Day Ratio by Market and Year",
        "Ratio",
        out_dir / "line_penny_day_ratio_all.png",
        pct=True,
        markets=markets or None,
    )

    save_line_chart(
        df,
        "stocks_one_tick_ge_10_ratio",
        "Stocks with One Tick >= 10% by Market and Year",
        "Ratio",
        out_dir / "line_stocks_one_tick_ge_10_ratio.png",
        pct=True,
        markets=markets or None,
    )

    save_line_chart(
        df,
        "stocks_one_tick_ge_20_ratio",
        "Stocks with One Tick >= 20% by Market and Year",
        "Ratio",
        out_dir / "line_stocks_one_tick_ge_20_ratio.png",
        pct=True,
        markets=markets or None,
    )

    save_line_chart(
        df,
        "stocks_one_tick_ge_30_ratio",
        "Stocks with One Tick >= 30% by Market and Year",
        "Ratio",
        out_dir / "line_stocks_one_tick_ge_30_ratio.png",
        pct=True,
        markets=markets or None,
    )

    save_line_chart(
        df,
        "stocks_abs_gap_ge_10_ratio",
        "Stocks with Abs Gap >= 10% by Market and Year",
        "Ratio",
        out_dir / "line_stocks_abs_gap_ge_10_ratio.png",
        pct=True,
        markets=markets or None,
    )

    save_line_chart(
        df,
        "stocks_abs_ret_ge_10_ratio",
        "Stocks with Abs Return >= 10% by Market and Year",
        "Ratio",
        out_dir / "line_stocks_abs_ret_ge_10_ratio.png",
        pct=True,
        markets=markets or None,
    )

    save_line_chart(
        df,
        "stocks_range_ge_10_ratio",
        "Stocks with Intraday Range >= 10% by Market and Year",
        "Ratio",
        out_dir / "line_stocks_range_ge_10_ratio.png",
        pct=True,
        markets=markets or None,
    )

    # Latest year rankings
    latest_year = int(df["year"].max())

    save_bar_rank(
        df,
        latest_year,
        "stocks_ever_penny_ratio",
        f"{latest_year} Stocks Ever Penny Ratio Ranking",
        "Ratio",
        out_dir / f"rank_{latest_year}_stocks_ever_penny_ratio.png",
    )

    save_bar_rank(
        df,
        latest_year,
        "penny_day_ratio_all",
        f"{latest_year} Penny Stock-Day Ratio Ranking",
        "Ratio",
        out_dir / f"rank_{latest_year}_penny_day_ratio_all.png",
    )

    save_bar_rank(
        df,
        latest_year,
        "stocks_one_tick_ge_10_ratio",
        f"{latest_year} One Tick >= 10% Ranking",
        "Ratio",
        out_dir / f"rank_{latest_year}_stocks_one_tick_ge_10_ratio.png",
    )

    save_bar_rank(
        df,
        latest_year,
        "stocks_one_tick_ge_20_ratio",
        f"{latest_year} One Tick >= 20% Ranking",
        "Ratio",
        out_dir / f"rank_{latest_year}_stocks_one_tick_ge_20_ratio.png",
    )

    save_bar_rank(
        df,
        latest_year,
        "stocks_abs_gap_ge_10_ratio",
        f"{latest_year} Abs Gap >= 10% Ranking",
        "Ratio",
        out_dir / f"rank_{latest_year}_stocks_abs_gap_ge_10_ratio.png",
    )

    # Heatmaps
    save_heatmap(
        df,
        "stocks_ever_penny_ratio",
        "Heatmap: Stocks Ever Penny Ratio",
        out_dir / "heatmap_stocks_ever_penny_ratio.png",
    )

    save_heatmap(
        df,
        "stocks_one_tick_ge_10_ratio",
        "Heatmap: Stocks with One Tick >= 10%",
        out_dir / "heatmap_stocks_one_tick_ge_10_ratio.png",
    )

    save_heatmap(
        df,
        "stocks_abs_gap_ge_10_ratio",
        "Heatmap: Stocks with Abs Gap >= 10%",
        out_dir / "heatmap_stocks_abs_gap_ge_10_ratio.png",
    )

    # Scatter for article argument
    save_scatter(
        df,
        "stocks_ever_penny_ratio",
        "stocks_one_tick_ge_10_ratio",
        latest_year,
        f"{latest_year} Penny Ratio vs One Tick >= 10%",
        "Stocks Ever Penny Ratio",
        "Stocks One Tick >= 10% Ratio",
        out_dir / f"scatter_{latest_year}_penny_vs_one_tick10.png",
    )

    save_scatter(
        df,
        "stocks_ever_penny_ratio",
        "stocks_abs_gap_ge_10_ratio",
        latest_year,
        f"{latest_year} Penny Ratio vs Abs Gap >= 10%",
        "Stocks Ever Penny Ratio",
        "Stocks Abs Gap >= 10% Ratio",
        out_dir / f"scatter_{latest_year}_penny_vs_abs_gap10.png",
    )

    write_chart_notes(df, out_dir / "chart_notes.md")

    print("\n================ CHARTS DONE ================")
    print(f"input  : {input_path}")
    print(f"out_dir: {out_dir}")
    print("=============================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())