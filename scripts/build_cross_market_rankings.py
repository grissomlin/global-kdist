# scripts/build_cross_market_rankings.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd
import yaml


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
# IO
# ============================================================
def _summary_path(market_code: str, freq: str, metric: str, start_year: int, end_year: int) -> Path:
    return (
        DATA_ROOT
        / "research"
        / "market_summary"
        / market_code
        / f"{market_code}_{freq}_{metric}_summary_{start_year}_{end_year}.csv"
    )


def load_all_summaries(
    markets: List[str],
    *,
    freq: str,
    metric: str,
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    dfs = []

    for market_code in markets:
        p = _summary_path(market_code, freq, metric, start_year, end_year)
        if not p.exists():
            print(f"⚠️ Missing summary file: {p}")
            continue

        df = pd.read_csv(p)
        if df.empty:
            continue

        # 保底：若舊檔沒有 Market 欄，就補
        if "Market" not in df.columns:
            df["Market"] = market_code.upper()

        dfs.append(df)

    if not dfs:
        raise RuntimeError(
            f"No summary files found for freq={freq}, metric={metric}, years={start_year}-{end_year}"
        )

    out = pd.concat(dfs, ignore_index=True)
    return out


# ============================================================
# Ranking builders
# ============================================================
def build_period_top1_rankings(df: pd.DataFrame, metric_cols: List[str]) -> pd.DataFrame:
    """
    For each Period, pick the market with highest value of each metric.
    Output:
      Period, PeriodLabel, MetricName, WinnerMarket, WinnerValue
    """
    rows = []

    for metric_name in metric_cols:
        if metric_name not in df.columns:
            continue

        sub = df[["Period", "PeriodLabel", "Market", metric_name]].copy()
        sub[metric_name] = pd.to_numeric(sub[metric_name], errors="coerce")
        sub = sub.dropna(subset=[metric_name])
        if sub.empty:
            continue

        idx = sub.groupby("Period")[metric_name].idxmax()
        top = sub.loc[idx].copy()

        for _, r in top.iterrows():
            rows.append(
                {
                    "Period": int(r["Period"]),
                    "PeriodLabel": r["PeriodLabel"],
                    "MetricName": metric_name,
                    "WinnerMarket": r["Market"],
                    "WinnerValue": float(r[metric_name]),
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    return out.sort_values(["MetricName", "Period"]).reset_index(drop=True)


def build_period_top3_rankings(df: pd.DataFrame, metric_cols: List[str], top_n: int = 3) -> pd.DataFrame:
    """
    For each Period and each metric, keep top N markets.
    """
    rows = []

    for metric_name in metric_cols:
        if metric_name not in df.columns:
            continue

        sub = df[["Period", "PeriodLabel", "Market", metric_name]].copy()
        sub[metric_name] = pd.to_numeric(sub[metric_name], errors="coerce")
        sub = sub.dropna(subset=[metric_name])
        if sub.empty:
            continue

        for period, g in sub.groupby("Period", sort=True):
            gg = g.sort_values(metric_name, ascending=False).head(top_n).copy()
            gg.insert(0, "Rank", range(1, len(gg) + 1))
            gg["MetricName"] = metric_name
            gg = gg.rename(columns={metric_name: "Value"})
            rows.append(gg[["Rank", "Period", "PeriodLabel", "MetricName", "Market", "Value"]])

    if not rows:
        return pd.DataFrame()

    out = pd.concat(rows, ignore_index=True)
    return out.sort_values(["MetricName", "Period", "Rank"]).reset_index(drop=True)


def build_market_aggregate_rankings(df: pd.DataFrame, metric_cols: List[str]) -> pd.DataFrame:
    """
    Aggregate across all periods by market:
      - Avg / Median / Max / Min
      - NumPeriods
    """
    rows = []

    for metric_name in metric_cols:
        if metric_name not in df.columns:
            continue

        sub = df[["Market", metric_name]].copy()
        sub[metric_name] = pd.to_numeric(sub[metric_name], errors="coerce")
        sub = sub.dropna(subset=[metric_name])
        if sub.empty:
            continue

        grp = sub.groupby("Market")[metric_name]
        agg = grp.agg(["count", "mean", "median", "max", "min"]).reset_index()
        agg["MetricName"] = metric_name
        agg = agg.rename(
            columns={
                "count": "NumPeriods",
                "mean": "AvgValue",
                "median": "MedianValue",
                "max": "MaxValue",
                "min": "MinValue",
            }
        )
        rows.append(agg[["MetricName", "Market", "NumPeriods", "AvgValue", "MedianValue", "MaxValue", "MinValue"]])

    if not rows:
        return pd.DataFrame()

    out = pd.concat(rows, ignore_index=True)
    return out.sort_values(["MetricName", "AvgValue"], ascending=[True, False]).reset_index(drop=True)


def build_market_leaderboard(df_agg: pd.DataFrame) -> pd.DataFrame:
    """
    For each metric, assign ranking by AvgValue descending.
    """
    if df_agg.empty:
        return df_agg

    rows = []
    for metric_name, g in df_agg.groupby("MetricName", sort=True):
        gg = g.sort_values("AvgValue", ascending=False).copy()
        gg.insert(0, "Rank", range(1, len(gg) + 1))
        rows.append(gg)

    out = pd.concat(rows, ignore_index=True)
    return out.sort_values(["MetricName", "Rank"]).reset_index(drop=True)


def build_article_highlights(df_agg: pd.DataFrame) -> pd.DataFrame:
    """
    One-line highlight per metric: who wins by AvgValue.
    """
    if df_agg.empty:
        return pd.DataFrame()

    rows = []
    for metric_name, g in df_agg.groupby("MetricName", sort=True):
        gg = g.sort_values("AvgValue", ascending=False).copy()
        r = gg.iloc[0]
        rows.append(
            {
                "MetricName": metric_name,
                "WinnerMarket": r["Market"],
                "AvgValue": float(r["AvgValue"]),
                "MedianValue": float(r["MedianValue"]),
                "MaxValue": float(r["MaxValue"]),
                "NumPeriods": int(r["NumPeriods"]),
            }
        )

    out = pd.DataFrame(rows)
    return out.sort_values("MetricName").reset_index(drop=True)


# ============================================================
# Export
# ============================================================
def export_rankings(
    markets: List[str],
    *,
    freq: str,
    metric: str,
    start_year: int,
    end_year: int,
):
    df = load_all_summaries(
        markets,
        freq=freq,
        metric=metric,
        start_year=start_year,
        end_year=end_year,
    )

    # 你最可能用來寫文章的核心欄位
    metric_cols = [
        "MeanRet_%",
        "MedianRet_%",
        "StdRet_%",
        "Pct_Up_gt_100",
        "Pct_Down_lt_-50",
        "Pct_ge_50",
        "Pct_ge_100",
        "Pct_ge_200",
        "Pct_ge_500",
        "Pct_ge_1000",
        "Pct_le_-20",
        "Pct_le_-50",
        "Top1Ret_%",
        "Top5CutRet_%",
    ]

    out_dir = DATA_ROOT / "research" / "cross_market_rankings" / f"{freq}_{metric}_{start_year}_{end_year}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) 每期每指標第一名
    df_period_top1 = build_period_top1_rankings(df, metric_cols)
    p1 = out_dir / f"{freq}_{metric}_period_top1_{start_year}_{end_year}.csv"
    df_period_top1.to_csv(p1, index=False, encoding="utf-8-sig")

    # 2) 每期每指標前三名
    df_period_top3 = build_period_top3_rankings(df, metric_cols, top_n=3)
    p3 = out_dir / f"{freq}_{metric}_period_top3_{start_year}_{end_year}.csv"
    df_period_top3.to_csv(p3, index=False, encoding="utf-8-sig")

    # 3) 市場整體聚合
    df_agg = build_market_aggregate_rankings(df, metric_cols)
    pa = out_dir / f"{freq}_{metric}_market_aggregate_{start_year}_{end_year}.csv"
    df_agg.to_csv(pa, index=False, encoding="utf-8-sig")

    # 4) 排名版 leaderboard
    df_leader = build_market_leaderboard(df_agg)
    pl = out_dir / f"{freq}_{metric}_market_leaderboard_{start_year}_{end_year}.csv"
    df_leader.to_csv(pl, index=False, encoding="utf-8-sig")

    # 5) 文章 highlights
    df_hi = build_article_highlights(df_agg)
    ph = out_dir / f"{freq}_{metric}_article_highlights_{start_year}_{end_year}.csv"
    df_hi.to_csv(ph, index=False, encoding="utf-8-sig")

    print(f"\n✅ Exported rankings -> {out_dir}")
    print(f"  period top1      : {p1}")
    print(f"  period top3      : {p3}")
    print(f"  market aggregate : {pa}")
    print(f"  market leaderboard: {pl}")
    print(f"  article highlights: {ph}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--markets",
        default="",
        help="Comma-separated market codes, e.g. hk,cn,th. Empty = all enabled markets",
    )
    ap.add_argument(
        "--freq",
        default="W",
        choices=["W", "M", "Y"],
        help="W=week, M=month, Y=year",
    )
    ap.add_argument(
        "--metric",
        default="close",
        choices=["high", "close", "low"],
        help="Use summary metric high/close/low",
    )
    ap.add_argument("--start-year", type=int, default=2020)
    ap.add_argument("--end-year", type=int, default=2025)
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
    print("Freq:", args.freq)
    print("Metric:", args.metric)
    print("Years:", args.start_year, "-", args.end_year)

    export_rankings(
        markets,
        freq=args.freq,
        metric=args.metric,
        start_year=args.start_year,
        end_year=args.end_year,
    )


if __name__ == "__main__":
    main()