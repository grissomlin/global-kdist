# scripts/research_penny_stats.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# Config
# =============================================================================

DATA_ROOT = Path("data")
OUT_DIR = Path("research_outputs/penny_stats")
DEFAULT_START = "2020-01-01"
DEFAULT_END = "2025-12-31"

# 低價股門檻（可自行改）
# 注意：跨國「便士股」定義不一致，這裡先用實務研究用的 local threshold
PENNY_THRESHOLDS: Dict[str, float] = {
    "us": 1.0,    # USD
    "ca": 1.0,    # CAD
    "uk": 1.0,    # GBP（你也可以改成 100 pence 等價概念，但這裡統一看 Yahoo/CSV 價格單位）
    "hk": 1.0,    # HKD
    "au": 1.0,    # AUD
    "jp": 100.0,  # JPY
    "kr": 1000.0, # KRW
    "tw": 30.0,   # TWD
    "de": 1.0,    # EUR
    "fr": 1.0,    # EUR
    "eu": 1.0,    # EUR
    "india": 10.0,# INR
    "cn": 5.0,    # CNY
}

# 你可以只跑這些市場
DEFAULT_MARKETS = ["us", "ca", "uk", "hk", "au", "jp", "kr", "tw", "de"]

# 日內 / 跳空 / 報酬的門檻
PCT_THRESHOLDS = [0.10, 0.20, 0.30, 0.50, 1.00]


# =============================================================================
# Tick size rules
# 這裡先做一版「夠研究用」的簡化表
# 之後你如果要更精準，可逐國微調
# =============================================================================

def tick_us(price: float) -> float:
    if price <= 0:
        return np.nan
    return 0.01

def tick_ca(price: float) -> float:
    if price <= 0:
        return np.nan
    return 0.01

def tick_au(price: float) -> float:
    if price <= 0:
        return np.nan
    # ASX rough ladder
    if price < 0.10: return 0.001
    if price < 2.00: return 0.005
    if price < 10.00: return 0.01
    return 0.05

def tick_uk(price: float) -> float:
    if price <= 0:
        return np.nan
    # LSE rough ladder
    if price < 1.00: return 0.0025
    if price < 5.00: return 0.005
    if price < 10.00: return 0.01
    return 0.05

def tick_hk(price: float) -> float:
    if price <= 0:
        return np.nan
    if price < 0.25: return 0.001
    if price < 0.50: return 0.005
    if price < 10.00: return 0.01
    if price < 20.00: return 0.02
    if price < 100.00: return 0.05
    if price < 200.00: return 0.10
    if price < 500.00: return 0.20
    if price < 1000.00: return 0.50
    if price < 2000.00: return 1.00
    if price < 5000.00: return 2.00
    return 5.00

def tick_jp(price: float) -> float:
    if price <= 0:
        return np.nan
    # Tokyo rough ladder
    if price < 1000: return 1
    if price < 3000: return 5
    if price < 5000: return 10
    if price < 30000: return 10
    if price < 50000: return 50
    if price < 300000: return 100
    if price < 500000: return 500
    if price < 3000000: return 1000
    if price < 5000000: return 5000
    return 10000

def tick_kr(price: float) -> float:
    if price <= 0:
        return np.nan
    if price < 1000: return 1
    if price < 5000: return 5
    if price < 10000: return 10
    if price < 50000: return 50
    if price < 100000: return 100
    if price < 500000: return 500
    return 1000

def tick_tw(price: float) -> float:
    if price <= 0:
        return np.nan
    if price < 10: return 0.01
    if price < 50: return 0.05
    if price < 100: return 0.10
    if price < 500: return 0.50
    if price < 1000: return 1.00
    return 5.00

def tick_de(price: float) -> float:
    if price <= 0:
        return np.nan
    # research simplification
    if price < 10: return 0.01
    if price < 50: return 0.05
    if price < 100: return 0.10
    return 0.50

def tick_fr(price: float) -> float:
    return tick_de(price)

def tick_eu(price: float) -> float:
    return tick_de(price)

def tick_india(price: float) -> float:
    if price <= 0:
        return np.nan
    if price < 15: return 0.05
    if price < 100: return 0.10
    if price < 1000: return 0.50
    return 1.00

def tick_cn(price: float) -> float:
    if price <= 0:
        return np.nan
    return 0.01

TICK_FUNC_MAP: Dict[str, Callable[[float], float]] = {
    "us": tick_us,
    "ca": tick_ca,
    "uk": tick_uk,
    "hk": tick_hk,
    "au": tick_au,
    "jp": tick_jp,
    "kr": tick_kr,
    "tw": tick_tw,
    "de": tick_de,
    "fr": tick_fr,
    "eu": tick_eu,
    "india": tick_india,
    "cn": tick_cn,
}


# =============================================================================
# Helpers
# =============================================================================

@dataclass
class MarketCfg:
    code: str
    dayk_dir: Path
    penny_threshold: float
    tick_func: Callable[[float], float]


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def list_csvs(dayk_dir: Path) -> List[Path]:
    if not dayk_dir.exists():
        return []
    return sorted([p for p in dayk_dir.glob("*.csv") if p.is_file()])


def safe_read_csv(path: Path) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_csv(path)
        if df is None or df.empty:
            return None
        return df
    except Exception:
        return None


def normalize_ohlcv(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    df = df.copy()
    cols = {str(c).strip().lower(): c for c in df.columns}

    def pick(*names: str) -> Optional[str]:
        for n in names:
            if n.lower() in cols:
                return cols[n.lower()]
        return None

    c_date = pick("date")
    c_open = pick("open")
    c_high = pick("high")
    c_low = pick("low")
    c_close = pick("close")
    c_volume = pick("volume")

    if not c_date or not c_close:
        return None

    out = pd.DataFrame()
    out["date"] = pd.to_datetime(df[c_date], errors="coerce")
    out["open"] = pd.to_numeric(df[c_open], errors="coerce") if c_open else np.nan
    out["high"] = pd.to_numeric(df[c_high], errors="coerce") if c_high else np.nan
    out["low"] = pd.to_numeric(df[c_low], errors="coerce") if c_low else np.nan
    out["close"] = pd.to_numeric(df[c_close], errors="coerce")
    out["volume"] = pd.to_numeric(df[c_volume], errors="coerce") if c_volume else np.nan

    out = out.dropna(subset=["date", "close"]).copy()
    if out.empty:
        return None

    out = out.sort_values("date").reset_index(drop=True)
    return out


def add_features(df: pd.DataFrame, tick_func: Callable[[float], float], penny_threshold: float) -> pd.DataFrame:
    df = df.copy()

    df["year"] = df["date"].dt.year.astype(int)
    df["prev_close"] = df["close"].shift(1)

    df["ret_cc"] = (df["close"] / df["prev_close"]) - 1.0
    df["gap_oc_prev_close"] = (df["open"] / df["prev_close"]) - 1.0
    df["intraday_range_pct"] = (df["high"] / df["low"]) - 1.0
    df["oc_move_pct"] = (df["close"] / df["open"]) - 1.0

    df["tick_size"] = df["close"].map(lambda x: tick_func(float(x)) if pd.notna(x) else np.nan)
    df["one_tick_pct"] = df["tick_size"] / df["close"]

    df["is_penny"] = df["close"] < float(penny_threshold)

    for th in PCT_THRESHOLDS:
        tag = int(round(th * 100))
        df[f"one_tick_ge_{tag}"] = df["one_tick_pct"] >= th
        df[f"gap_up_ge_{tag}"] = df["gap_oc_prev_close"] >= th
        df[f"gap_down_le_{tag}"] = df["gap_oc_prev_close"] <= -th
        df[f"abs_gap_ge_{tag}"] = df["gap_oc_prev_close"].abs() >= th
        df[f"ret_up_ge_{tag}"] = df["ret_cc"] >= th
        df[f"ret_down_le_{tag}"] = df["ret_cc"] <= -th
        df[f"abs_ret_ge_{tag}"] = df["ret_cc"].abs() >= th
        df[f"range_ge_{tag}"] = df["intraday_range_pct"] >= th

    return df


def summarize_stock_year(df: pd.DataFrame, symbol: str, market: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for year, g in df.groupby("year", sort=True):
        if g.empty:
            continue

        row: Dict[str, Any] = {
            "market": market,
            "symbol": symbol,
            "year": int(year),
            "n_days": int(len(g)),
            "first_date": g["date"].min().date().isoformat(),
            "last_date": g["date"].max().date().isoformat(),
            "min_close": float(g["close"].min()),
            "median_close": float(g["close"].median()),
            "mean_close": float(g["close"].mean()),
            "max_close": float(g["close"].max()),
            "penny_days": int(g["is_penny"].sum()),
            "penny_day_ratio": float(g["is_penny"].mean()),
        }

        for th in PCT_THRESHOLDS:
            tag = int(round(th * 100))
            row[f"one_tick_ge_{tag}_days"] = int(g[f"one_tick_ge_{tag}"].fillna(False).sum())
            row[f"gap_up_ge_{tag}_days"] = int(g[f"gap_up_ge_{tag}"].fillna(False).sum())
            row[f"gap_down_le_{tag}_days"] = int(g[f"gap_down_le_{tag}"].fillna(False).sum())
            row[f"abs_gap_ge_{tag}_days"] = int(g[f"abs_gap_ge_{tag}"].fillna(False).sum())
            row[f"abs_ret_ge_{tag}_days"] = int(g[f"abs_ret_ge_{tag}"].fillna(False).sum())
            row[f"range_ge_{tag}_days"] = int(g[f"range_ge_{tag}"].fillna(False).sum())

        rows.append(row)

    return pd.DataFrame(rows)


def summarize_market_year(stock_year_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for (market, year), g in stock_year_df.groupby(["market", "year"], sort=True):
        total_stocks = int(g["symbol"].nunique())
        total_days = int(g["n_days"].sum())
        penny_stock_count = int((g["penny_days"] > 0).sum())
        heavy_penny_stock_count = int((g["penny_day_ratio"] >= 0.5).sum())

        row: Dict[str, Any] = {
            "market": market,
            "year": int(year),
            "total_stocks": total_stocks,
            "total_stock_days": total_days,
            "stocks_ever_penny": penny_stock_count,
            "stocks_ever_penny_ratio": penny_stock_count / total_stocks if total_stocks else np.nan,
            "stocks_penny_majority_days": heavy_penny_stock_count,
            "stocks_penny_majority_ratio": heavy_penny_stock_count / total_stocks if total_stocks else np.nan,
            "penny_days_total": int(g["penny_days"].sum()),
            "penny_day_ratio_all": float(g["penny_days"].sum() / total_days) if total_days else np.nan,
            "median_close_median_of_stocks": float(g["median_close"].median()),
            "mean_close_mean_of_stocks": float(g["mean_close"].mean()),
        }

        for th in PCT_THRESHOLDS:
            tag = int(round(th * 100))

            stocks_tick = int((g[f"one_tick_ge_{tag}_days"] > 0).sum())
            days_tick = int(g[f"one_tick_ge_{tag}_days"].sum())

            row[f"stocks_one_tick_ge_{tag}"] = stocks_tick
            row[f"stocks_one_tick_ge_{tag}_ratio"] = stocks_tick / total_stocks if total_stocks else np.nan
            row[f"days_one_tick_ge_{tag}"] = days_tick
            row[f"days_one_tick_ge_{tag}_ratio"] = days_tick / total_days if total_days else np.nan

            for prefix in ["gap_up_ge", "gap_down_le", "abs_gap_ge", "abs_ret_ge", "range_ge"]:
                col = f"{prefix}_{tag}_days"
                stocks_any = int((g[col] > 0).sum())
                days_sum = int(g[col].sum())

                row[f"stocks_{prefix}_{tag}"] = stocks_any
                row[f"stocks_{prefix}_{tag}_ratio"] = stocks_any / total_stocks if total_stocks else np.nan
                row[f"days_{prefix}_{tag}"] = days_sum
                row[f"days_{prefix}_{tag}_ratio"] = days_sum / total_days if total_days else np.nan

        rows.append(row)

    return pd.DataFrame(rows)


def build_markdown_report(summary_df: pd.DataFrame, out_path: Path) -> None:
    lines: List[str] = []
    lines.append("# Penny / Low-Price Stock Research Summary")
    lines.append("")
    lines.append("Period: 2020-01-01 to 2025-12-31")
    lines.append("")
    lines.append("This report focuses on two concepts:")
    lines.append("")
    lines.append("1. Low-price / penny-stock prevalence")
    lines.append("2. Tick-size distortion: when one minimum tick already exceeds 10%, 20%, 30%")
    lines.append("")

    for market in sorted(summary_df["market"].dropna().unique().tolist()):
        sub = summary_df[summary_df["market"] == market].sort_values("year")
        if sub.empty:
            continue

        lines.append(f"## {market.upper()}")
        lines.append("")

        for _, r in sub.iterrows():
            y = int(r["year"])
            total_stocks = int(r["total_stocks"])
            stocks_ever_penny = int(r["stocks_ever_penny"])
            penny_ratio = float(r["stocks_ever_penny_ratio"]) if pd.notna(r["stocks_ever_penny_ratio"]) else np.nan
            penny_day_ratio = float(r["penny_day_ratio_all"]) if pd.notna(r["penny_day_ratio_all"]) else np.nan

            lines.append(f"### {y}")
            lines.append("")
            lines.append(f"- Total stocks: **{total_stocks}**")
            lines.append(f"- Stocks ever penny: **{stocks_ever_penny}** ({penny_ratio:.2%})")
            lines.append(f"- Penny stock-day ratio: **{penny_day_ratio:.2%}**")
            lines.append(f"- Stocks with one tick >= 10%: **{int(r['stocks_one_tick_ge_10'])}** ({float(r['stocks_one_tick_ge_10_ratio']):.2%})")
            lines.append(f"- Stocks with one tick >= 20%: **{int(r['stocks_one_tick_ge_20'])}** ({float(r['stocks_one_tick_ge_20_ratio']):.2%})")
            lines.append(f"- Stocks with one tick >= 30%: **{int(r['stocks_one_tick_ge_30'])}** ({float(r['stocks_one_tick_ge_30_ratio']):.2%})")
            lines.append(f"- Stocks with abs gap >= 10%: **{int(r['stocks_abs_gap_ge_10'])}** ({float(r['stocks_abs_gap_ge_10_ratio']):.2%})")
            lines.append(f"- Stocks with abs return >= 10%: **{int(r['stocks_abs_ret_ge_10'])}** ({float(r['stocks_abs_ret_ge_10_ratio']):.2%})")
            lines.append(f"- Stocks with intraday range >= 10%: **{int(r['stocks_range_ge_10'])}** ({float(r['stocks_range_ge_10_ratio']):.2%})")
            lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


# =============================================================================
# Pipeline
# =============================================================================

def infer_market_cfg(code: str, data_root: Path) -> MarketCfg:
    c = code.strip().lower()
    dayk_dir = data_root / c / "dayK"
    penny_threshold = PENNY_THRESHOLDS.get(c, 1.0)
    tick_func = TICK_FUNC_MAP.get(c, tick_us)
    return MarketCfg(
        code=c,
        dayk_dir=dayk_dir,
        penny_threshold=penny_threshold,
        tick_func=tick_func,
    )


def run_market(mcfg: MarketCfg, start: str, end: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    stock_year_rows: List[pd.DataFrame] = []
    detail_rows: List[Dict[str, Any]] = []

    csvs = list_csvs(mcfg.dayk_dir)
    if not csvs:
        return pd.DataFrame(), pd.DataFrame()

    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)

    for path in csvs:
        symbol = path.stem
        raw = safe_read_csv(path)
        if raw is None:
            continue

        df = normalize_ohlcv(raw)
        if df is None or df.empty:
            continue

        df = df[(df["date"] >= start_ts) & (df["date"] <= end_ts)].copy()
        if df.empty:
            continue

        df = add_features(df, tick_func=mcfg.tick_func, penny_threshold=mcfg.penny_threshold)
        sy = summarize_stock_year(df, symbol=symbol, market=mcfg.code)
        if sy.empty:
            continue

        stock_year_rows.append(sy)

        detail_rows.append(
            {
                "market": mcfg.code,
                "symbol": symbol,
                "n_days_total": int(len(df)),
                "start_date": df["date"].min().date().isoformat(),
                "end_date": df["date"].max().date().isoformat(),
                "min_close_all": float(df["close"].min()),
                "median_close_all": float(df["close"].median()),
                "mean_close_all": float(df["close"].mean()),
                "max_close_all": float(df["close"].max()),
                "penny_days_total": int(df["is_penny"].sum()),
                "penny_day_ratio_total": float(df["is_penny"].mean()),
                "one_tick_pct_max": float(df["one_tick_pct"].max()) if df["one_tick_pct"].notna().any() else np.nan,
                "one_tick_pct_median": float(df["one_tick_pct"].median()) if df["one_tick_pct"].notna().any() else np.nan,
            }
        )

    stock_year_df = pd.concat(stock_year_rows, ignore_index=True) if stock_year_rows else pd.DataFrame()
    detail_df = pd.DataFrame(detail_rows)
    return stock_year_df, detail_df


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--markets", default=",".join(DEFAULT_MARKETS), help="Comma-separated market codes")
    ap.add_argument("--data-root", default=str(DATA_ROOT), help="Root folder, e.g. data")
    ap.add_argument("--start", default=DEFAULT_START, help="YYYY-MM-DD")
    ap.add_argument("--end", default=DEFAULT_END, help="YYYY-MM-DD")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    data_root = Path(args.data_root)
    markets = [x.strip().lower() for x in str(args.markets).split(",") if x.strip()]

    ensure_dir(OUT_DIR)

    all_stock_year: List[pd.DataFrame] = []
    all_detail: List[pd.DataFrame] = []

    for code in markets:
        mcfg = infer_market_cfg(code, data_root=data_root)
        print(f"▶ market={code} dir={mcfg.dayk_dir}")
        sy_df, detail_df = run_market(mcfg, start=args.start, end=args.end)

        if sy_df.empty:
            print(f"  ↳ no usable data")
            continue

        print(f"  ↳ stock_year_rows={len(sy_df)} detail_rows={len(detail_df)}")
        all_stock_year.append(sy_df)
        all_detail.append(detail_df)

    if not all_stock_year:
        print("No results.")
        return 1

    stock_year_df = pd.concat(all_stock_year, ignore_index=True)
    detail_df = pd.concat(all_detail, ignore_index=True) if all_detail else pd.DataFrame()
    summary_df = summarize_market_year(stock_year_df)

    stock_year_path = OUT_DIR / "stock_year_detail.csv"
    detail_path = OUT_DIR / "stock_detail.csv"
    summary_path = OUT_DIR / "market_year_summary.csv"
    md_path = OUT_DIR / "report.md"

    stock_year_df.to_csv(stock_year_path, index=False, encoding="utf-8-sig")
    detail_df.to_csv(detail_path, index=False, encoding="utf-8-sig")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    build_markdown_report(summary_df, md_path)

    print("\n================ DONE ================")
    print(f"stock_year_detail : {stock_year_path}")
    print(f"stock_detail      : {detail_path}")
    print(f"market_year_sum   : {summary_path}")
    print(f"markdown_report   : {md_path}")
    print("======================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())