# scripts/live_weekly.py
from __future__ import annotations

import argparse
import sys
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from core.registry import load_market
from core.download_dayk import download_all
from core.resample_k import resample_all
from core.io import ensure_dir

BASE = ROOT / "data"


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing config file: {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _cfg_bool(d: dict, key: str, default: bool) -> bool:
    v = d.get(key, default)
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _cfg_float(d: dict, key: str, default: float) -> float:
    try:
        return float(d.get(key, default))
    except Exception:
        return float(default)


def _cfg_str(d: dict, key: str, default: str) -> str:
    v = d.get(key, default)
    if v is None:
        return default
    s = str(v).strip()
    return s if s else default


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--markets",
        default="",
        help="Comma-separated market codes, e.g. tw,cn,th. Empty = all enabled markets in configs/markets.yaml",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing week/month/year csv outputs",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    markets_cfg = _load_yaml(ROOT / "configs" / "markets.yaml")
    profile = _load_yaml(ROOT / "configs" / "profiles" / "live.yaml")

    markets = markets_cfg.get("markets", {}) or {}
    if not isinstance(markets, dict):
        raise TypeError("configs/markets.yaml: 'markets' must be a dict")

    download_cfg = profile.get("download", {}) or {}
    resample_cfg = profile.get("resample", {}) or {}

    # -------- global fallback: low price --------
    global_low_price_filter_on = _cfg_bool(resample_cfg, "low_price_filter_on", False)
    global_low_price_mode = _cfg_str(resample_cfg, "low_price_mode", "both")
    global_low_price_min_ratio = _cfg_float(resample_cfg, "low_price_min_ratio", 0.5)

    # -------- global fallback: tick distortion --------
    global_tick_distortion_filter_on = _cfg_bool(resample_cfg, "tick_distortion_filter_on", True)
    global_tick_distortion_mode = _cfg_str(resample_cfg, "tick_distortion_mode", "both")
    global_tick_distortion_threshold_pct = _cfg_float(resample_cfg, "tick_distortion_threshold_pct", 10.0)
    global_tick_distortion_min_ratio = _cfg_float(resample_cfg, "tick_distortion_min_ratio", 0.5)

    # -------- download window --------
    # start/end 優先於 period
    # 建議研究 2020~2025 時，至少抓到 2019-01-01
    period = _cfg_str(download_cfg, "period", "5y")
    download_start = _cfg_str(download_cfg, "start", "2019-01-01")
    download_end = _cfg_str(download_cfg, "end", "")
    download_end = download_end if download_end else None

    if args.markets.strip():
        wanted = {x.strip().lower() for x in args.markets.split(",") if x.strip()}
    else:
        wanted = {
            str(code).strip().lower()
            for code, m in markets.items()
            if isinstance(m, dict) and m.get("enabled", False)
        }

    print("Download window:")
    print(f"  start={download_start}")
    print(f"  end={download_end or '(none)'}")
    print(f"  period(fallback only)={period}")

    for code, m in markets.items():
        if not isinstance(m, dict):
            continue

        code = str(code).strip().lower()

        if code not in wanted:
            continue

        if not m.get("enabled", False):
            print(f"⚠️ Skip disabled market: {code}")
            continue

        week_rule = _cfg_str(m, "week_rule", "W-FRI")
        month_rule = _cfg_str(m, "month_rule", "ME")
        year_rule = _cfg_str(m, "year_rule", "YE")

        # -------- market-specific: low price --------
        low_price_filter_on = _cfg_bool(
            m,
            "low_price_filter_on",
            global_low_price_filter_on,
        )
        low_price_mode = _cfg_str(
            m,
            "low_price_mode",
            global_low_price_mode,
        )
        low_price_min_ratio = _cfg_float(
            m,
            "low_price_min_ratio",
            global_low_price_min_ratio,
        )

        # -------- market-specific: tick distortion --------
        tick_distortion_filter_on = _cfg_bool(
            m,
            "tick_distortion_filter_on",
            global_tick_distortion_filter_on,
        )
        tick_distortion_mode = _cfg_str(
            m,
            "tick_distortion_mode",
            global_tick_distortion_mode,
        )
        tick_distortion_threshold_pct = _cfg_float(
            m,
            "tick_distortion_threshold_pct",
            global_tick_distortion_threshold_pct,
        )
        tick_distortion_min_ratio = _cfg_float(
            m,
            "tick_distortion_min_ratio",
            global_tick_distortion_min_ratio,
        )

        market = load_market(code)

        print(f"\n=== Running market: {code} ===")

        universe = market.get_universe(m)

        day_dir = ensure_dir(BASE / "cache_dayk" / code)
        week_dir = ensure_dir(BASE / "derived" / "weekK" / code)
        month_dir = ensure_dir(BASE / "derived" / "monthK" / code)
        year_dir = ensure_dir(BASE / "derived" / "yearK" / code)

        dl_stats = download_all(
            universe,
            market.to_ticker,
            day_dir,
            start=download_start,
            end=download_end,
            period=period,
            market_code=code,
        )

        st = resample_all(
            day_dir,
            out_week_dir=week_dir,
            out_month_dir=month_dir,
            out_year_dir=year_dir,
            market_code=code,
            week_rule=week_rule,
            month_rule=month_rule,
            year_rule=year_rule,
            overwrite=bool(args.overwrite),
            low_price_filter_on=low_price_filter_on,
            low_price_mode=low_price_mode,
            low_price_min_ratio=low_price_min_ratio,
            tick_distortion_filter_on=tick_distortion_filter_on,
            tick_distortion_mode=tick_distortion_mode,
            tick_distortion_threshold_pct=tick_distortion_threshold_pct,
            tick_distortion_min_ratio=tick_distortion_min_ratio,
        )

        print(f"✅ Finished {code}")
        print(f"  dayK  : {day_dir}")
        print(f"  weekK : {week_dir}")
        print(f"  monthK: {month_dir}")
        print(f"  yearK : {year_dir}")
        print(
            f"  download_start={download_start} "
            f"download_end={download_end or '(none)'} "
            f"period_fallback={period}"
        )
        print(f"  download stats: {dl_stats}")
        print(
            f"  tick_distortion_filter_on={tick_distortion_filter_on} "
            f"mode={tick_distortion_mode} "
            f"threshold_pct={tick_distortion_threshold_pct} "
            f"min_ratio={tick_distortion_min_ratio}"
        )
        print(
            f"  low_price_filter_on={low_price_filter_on} "
            f"mode={low_price_mode} "
            f"min_ratio={low_price_min_ratio}"
        )
        print(f"  resample stats: {st}")


if __name__ == "__main__":
    main()