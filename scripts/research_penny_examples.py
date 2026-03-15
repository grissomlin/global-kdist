# scripts/research_penny_examples.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd


DEFAULT_STOCK_YEAR = "research_outputs/penny_stats/stock_year_detail.csv"
DEFAULT_STOCK_DETAIL = "research_outputs/penny_stats/stock_detail.csv"
DEFAULT_OUT_DIR = "research_outputs/penny_stats/examples"


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise RuntimeError(f"CSV is empty: {path}")
    return df


def read_inputs(stock_year_path: Path, stock_detail_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    sy = _read_csv(stock_year_path)
    sd = _read_csv(stock_detail_path)

    for df in (sy, sd):
        if "market" in df.columns:
            df["market"] = df["market"].astype(str).str.upper()
        if "symbol" in df.columns:
            df["symbol"] = df["symbol"].astype(str).str.upper()

    if "year" in sy.columns:
        sy["year"] = pd.to_numeric(sy["year"], errors="coerce").astype("Int64")
        sy = sy[sy["year"].notna()].copy()
        sy["year"] = sy["year"].astype(int)

    return sy, sd


def top_n_per_market_year(
    df: pd.DataFrame,
    value_col: str,
    n: int = 10,
    ascending: bool = False,
) -> pd.DataFrame:
    if value_col not in df.columns:
        return pd.DataFrame()

    out: List[pd.DataFrame] = []
    for (market, year), g in df.groupby(["market", "year"], sort=True):
        gg = g.copy()
        gg = gg[pd.to_numeric(gg[value_col], errors="coerce").notna()].copy()
        if gg.empty:
            continue
        gg = gg.sort_values(value_col, ascending=ascending).head(n).copy()
        gg.insert(0, "rank", range(1, len(gg) + 1))
        out.append(gg)

    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def top_n_per_market(
    df: pd.DataFrame,
    value_col: str,
    n: int = 10,
    ascending: bool = False,
) -> pd.DataFrame:
    if value_col not in df.columns:
        return pd.DataFrame()

    out: List[pd.DataFrame] = []
    for market, g in df.groupby("market", sort=True):
        gg = g.copy()
        gg = gg[pd.to_numeric(gg[value_col], errors="coerce").notna()].copy()
        if gg.empty:
            continue
        gg = gg.sort_values(value_col, ascending=ascending).head(n).copy()
        gg.insert(0, "rank", range(1, len(gg) + 1))
        out.append(gg)

    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def build_threshold_example_table(
    sy: pd.DataFrame,
    year: int,
    prefix: str,
    thresholds: List[int],
    n: int = 10,
) -> pd.DataFrame:
    """
    prefix examples:
      abs_gap_ge
      abs_ret_ge
      range_ge
      one_tick_ge
    """
    rows: List[pd.DataFrame] = []

    base = sy[sy["year"] == year].copy()
    if base.empty:
        return pd.DataFrame()

    for th in thresholds:
        col = f"{prefix}_{th}_days"
        if col not in base.columns:
            continue

        for market, g in base.groupby("market", sort=True):
            gg = g.copy()
            gg[col] = pd.to_numeric(gg[col], errors="coerce").fillna(0)
            gg = gg[gg[col] > 0].copy()
            if gg.empty:
                continue

            gg = gg.sort_values([col, "penny_day_ratio", "median_close"], ascending=[False, False, True]).head(n).copy()
            gg.insert(0, "rank", range(1, len(gg) + 1))
            gg.insert(1, "threshold_pct", th)
            gg.insert(2, "metric", prefix)
            rows.append(gg)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def compact_columns(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    keep = [c for c in cols if c in df.columns]
    return df[keep].copy() if keep else pd.DataFrame()


def write_markdown_summary(
    out_path: Path,
    latest_year: int,
    penny_top: pd.DataFrame,
    tick_top: pd.DataFrame,
    abs_gap_top: pd.DataFrame,
    abs_ret_top: pd.DataFrame,
    range_top: pd.DataFrame,
) -> None:
    lines: List[str] = []
    lines.append("# Penny / Tick Distortion Example Notes")
    lines.append("")
    lines.append(f"Latest year used: {latest_year}")
    lines.append("")

    def section(title: str, df: pd.DataFrame, value_cols: List[str]) -> None:
        lines.append(f"## {title}")
        lines.append("")
        if df.empty:
            lines.append("- No data")
            lines.append("")
            return

        for market in sorted(df["market"].dropna().unique().tolist()):
            sub = df[df["market"] == market].copy()
            if sub.empty:
                continue
            lines.append(f"### {market}")
            lines.append("")
            for _, r in sub.head(5).iterrows():
                bits = [f"{r.get('symbol', '')}"]
                if "year" in r:
                    bits.append(f"year={r.get('year')}")
                for c in value_cols:
                    if c in sub.columns and pd.notna(r.get(c)):
                        v = r.get(c)
                        if "ratio" in c or "pct" in c:
                            try:
                                bits.append(f"{c}={float(v):.2%}")
                            except Exception:
                                bits.append(f"{c}={v}")
                        else:
                            bits.append(f"{c}={v}")
                lines.append(f"- " + " | ".join(bits))
            lines.append("")

    section(
        "Highest Penny Exposure Stocks",
        penny_top[penny_top["year"] == latest_year].copy() if not penny_top.empty else penny_top,
        ["penny_day_ratio", "median_close", "min_close", "max_close"],
    )
    section(
        "Highest One-Tick Percentage Stocks",
        tick_top,
        ["one_tick_pct_max", "one_tick_pct_median", "penny_day_ratio_total", "median_close_all"],
    )
    section(
        "Most Frequent Abs Gap Examples",
        abs_gap_top,
        ["threshold_pct", "abs_gap_ge_10_days", "abs_gap_ge_20_days", "abs_gap_ge_30_days", "penny_day_ratio", "median_close"],
    )
    section(
        "Most Frequent Abs Return Examples",
        abs_ret_top,
        ["threshold_pct", "abs_ret_ge_10_days", "abs_ret_ge_20_days", "abs_ret_ge_30_days", "penny_day_ratio", "median_close"],
    )
    section(
        "Most Frequent Intraday Range Examples",
        range_top,
        ["threshold_pct", "range_ge_10_days", "range_ge_20_days", "range_ge_30_days", "penny_day_ratio", "median_close"],
    )

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stock-year", default=DEFAULT_STOCK_YEAR, help="Path to stock_year_detail.csv")
    ap.add_argument("--stock-detail", default=DEFAULT_STOCK_DETAIL, help="Path to stock_detail.csv")
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="Output directory")
    ap.add_argument("--year", type=int, default=0, help="Target year for event examples (0 = latest)")
    ap.add_argument("--top", type=int, default=10, help="Top N per market")
    ap.add_argument("--markets", default="", help="Optional market filter, e.g. us,tw,jp,de")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    sy, sd = read_inputs(Path(args.stock_year), Path(args.stock_detail))

    wanted_markets = [x.strip().upper() for x in str(args.markets).split(",") if x.strip()]
    if wanted_markets:
        sy = sy[sy["market"].isin(wanted_markets)].copy()
        sd = sd[sd["market"].isin(wanted_markets)].copy()

    if sy.empty:
        raise RuntimeError("Filtered stock_year_detail is empty.")
    if sd.empty:
        raise RuntimeError("Filtered stock_detail is empty.")

    latest_year = int(sy["year"].max())
    target_year = int(args.year) if int(args.year or 0) > 0 else latest_year
    top_n = max(1, int(args.top))

    # 1) 各市場各年度 低價股比例最高股票
    penny_top = top_n_per_market_year(
        sy,
        value_col="penny_day_ratio",
        n=top_n,
        ascending=False,
    )

    # 2) 各市場 overall one-tick 百分比最高股票
    tick_top = top_n_per_market(
        sd,
        value_col="one_tick_pct_max",
        n=top_n,
        ascending=False,
    )

    # 3) 指定年度，各市場最常出現 abs gap / abs ret / range 的股票
    abs_gap_top = build_threshold_example_table(
        sy,
        year=target_year,
        prefix="abs_gap_ge",
        thresholds=[10, 20, 30],
        n=top_n,
    )
    abs_ret_top = build_threshold_example_table(
        sy,
        year=target_year,
        prefix="abs_ret_ge",
        thresholds=[10, 20, 30],
        n=top_n,
    )
    range_top = build_threshold_example_table(
        sy,
        year=target_year,
        prefix="range_ge",
        thresholds=[10, 20, 30],
        n=top_n,
    )
    one_tick_year_top = build_threshold_example_table(
        sy,
        year=target_year,
        prefix="one_tick_ge",
        thresholds=[10, 20, 30],
        n=top_n,
    )

    # 輸出：完整榜單
    penny_top.to_csv(out_dir / "top_penny_day_ratio_by_market_year.csv", index=False, encoding="utf-8-sig")
    tick_top.to_csv(out_dir / "top_one_tick_pct_by_market.csv", index=False, encoding="utf-8-sig")
    abs_gap_top.to_csv(out_dir / f"top_abs_gap_examples_{target_year}.csv", index=False, encoding="utf-8-sig")
    abs_ret_top.to_csv(out_dir / f"top_abs_ret_examples_{target_year}.csv", index=False, encoding="utf-8-sig")
    range_top.to_csv(out_dir / f"top_range_examples_{target_year}.csv", index=False, encoding="utf-8-sig")
    one_tick_year_top.to_csv(out_dir / f"top_one_tick_examples_{target_year}.csv", index=False, encoding="utf-8-sig")

    # 輸出：精簡版，文章比較好看
    penny_compact = compact_columns(
        penny_top[penny_top["year"] == target_year].copy(),
        [
            "rank", "market", "year", "symbol",
            "n_days", "penny_days", "penny_day_ratio",
            "min_close", "median_close", "mean_close", "max_close",
            "one_tick_ge_10_days", "one_tick_ge_20_days", "one_tick_ge_30_days",
            "abs_gap_ge_10_days", "abs_ret_ge_10_days", "range_ge_10_days",
        ],
    )
    tick_compact = compact_columns(
        tick_top.copy(),
        [
            "rank", "market", "symbol",
            "n_days_total", "penny_days_total", "penny_day_ratio_total",
            "min_close_all", "median_close_all", "mean_close_all", "max_close_all",
            "one_tick_pct_max", "one_tick_pct_median",
        ],
    )
    gap_compact = compact_columns(
        abs_gap_top.copy(),
        [
            "rank", "threshold_pct", "market", "year", "symbol",
            "n_days", "penny_day_ratio", "median_close",
            "abs_gap_ge_10_days", "abs_gap_ge_20_days", "abs_gap_ge_30_days",
            "abs_ret_ge_10_days", "range_ge_10_days",
        ],
    )
    ret_compact = compact_columns(
        abs_ret_top.copy(),
        [
            "rank", "threshold_pct", "market", "year", "symbol",
            "n_days", "penny_day_ratio", "median_close",
            "abs_ret_ge_10_days", "abs_ret_ge_20_days", "abs_ret_ge_30_days",
            "abs_gap_ge_10_days", "range_ge_10_days",
        ],
    )
    range_compact = compact_columns(
        range_top.copy(),
        [
            "rank", "threshold_pct", "market", "year", "symbol",
            "n_days", "penny_day_ratio", "median_close",
            "range_ge_10_days", "range_ge_20_days", "range_ge_30_days",
            "abs_gap_ge_10_days", "abs_ret_ge_10_days",
        ],
    )

    penny_compact.to_csv(out_dir / f"compact_penny_examples_{target_year}.csv", index=False, encoding="utf-8-sig")
    tick_compact.to_csv(out_dir / "compact_one_tick_examples.csv", index=False, encoding="utf-8-sig")
    gap_compact.to_csv(out_dir / f"compact_abs_gap_examples_{target_year}.csv", index=False, encoding="utf-8-sig")
    ret_compact.to_csv(out_dir / f"compact_abs_ret_examples_{target_year}.csv", index=False, encoding="utf-8-sig")
    range_compact.to_csv(out_dir / f"compact_range_examples_{target_year}.csv", index=False, encoding="utf-8-sig")

    write_markdown_summary(
        out_dir / "example_notes.md",
        latest_year=target_year,
        penny_top=penny_top,
        tick_top=tick_top,
        abs_gap_top=abs_gap_top,
        abs_ret_top=abs_ret_top,
        range_top=range_top,
    )

    print("\n================ EXAMPLES DONE ================")
    print(f"stock_year : {args.stock_year}")
    print(f"stock_detail: {args.stock_detail}")
    print(f"target_year: {target_year}")
    print(f"out_dir    : {out_dir}")
    print("outputs:")
    print(f"  - top_penny_day_ratio_by_market_year.csv")
    print(f"  - top_one_tick_pct_by_market.csv")
    print(f"  - top_abs_gap_examples_{target_year}.csv")
    print(f"  - top_abs_ret_examples_{target_year}.csv")
    print(f"  - top_range_examples_{target_year}.csv")
    print(f"  - top_one_tick_examples_{target_year}.csv")
    print(f"  - compact_*.csv")
    print(f"  - example_notes.md")
    print("===============================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())