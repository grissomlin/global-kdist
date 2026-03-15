from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

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
        help="Comma-separated market codes, e.g. cn,th,tw. Empty = all enabled markets in configs/markets.yaml",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing week/month/year csv outputs",
    )
    ap.add_argument(
        "--debug-cleaning",
        action="store_true",
        help="Print file-level cleaning/filter debug logs",
    )
    ap.add_argument(
        "--debug-max-logs",
        type=int,
        default=20,
        help="Max file-level debug logs to print per market",
    )
    ap.add_argument(
        "--no-save-research",
        action="store_true",
        help="Do not save research CSV outputs",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Thread workers per market for resampling, default=8",
    )
    return ap.parse_args()


def _print_market_summary(code: str, st: dict) -> None:
    print("\n--- Summary ---")
    print(
        f"files={st.get('files', 0)} "
        f"ok={st.get('ok', 0)} "
        f"fail={st.get('fail', 0)} "
        f"skipped={st.get('skipped', 0)}"
    )
    print(
        f"filtered_tick_distortion={st.get('filtered_tick_distortion', 0)} "
        f"filtered_low_price={st.get('filtered_low_price', 0)} "
        f"empty_after_clean={st.get('empty_after_clean', 0)} "
        f"scale_fixed_files={st.get('scale_fixed_files', 0)}"
    )

    print("\n--- Coverage ---")
    print(
        f"eligible_after_clean={st.get('eligible_after_clean', 0)} "
        f"eligible_after_tick_filter={st.get('eligible_after_tick_filter', 0)} "
        f"eligible_after_all_filters={st.get('eligible_after_all_filters', 0)}"
    )
    print(
        f"coverage_before_filter_pct={st.get('coverage_before_filter_pct', 0.0):.2f}% "
        f"coverage_after_tick_filter_pct={st.get('coverage_after_tick_filter_pct', 0.0):.2f}% "
        f"coverage_after_all_filters_pct={st.get('coverage_after_all_filters_pct', 0.0):.2f}%"
    )

    print("\n--- Row cleaning totals ---")
    print(
        f"rows_input_total={st.get('rows_input_total', 0)} "
        f"rows_after_read_total={st.get('rows_after_read_total', 0)} "
        f"rows_after_ohlc_total={st.get('rows_after_ohlc_total', 0)} "
        f"rows_after_scale_total={st.get('rows_after_scale_total', 0)}"
    )
    print(
        f"rows_after_corporate_actions_total={st.get('rows_after_corporate_actions_total', 0)} "
        f"rows_after_ghost_total={st.get('rows_after_ghost_total', 0)} "
        f"rows_after_ipo_guard_total={st.get('rows_after_ipo_guard_total', 0)} "
        f"rows_after_extreme_total={st.get('rows_after_extreme_total', 0)} "
        f"rows_after_penny_total={st.get('rows_after_penny_total', 0)}"
    )

    print("\n--- Rows removed by cleaner ---")
    print(
        f"ohlc_rows_removed_total={st.get('ohlc_rows_removed_total', 0)} "
        f"corporate_action_rows_removed_total={st.get('corporate_action_rows_removed_total', 0)} "
        f"ghost_rows_removed_total={st.get('ghost_rows_removed_total', 0)} "
        f"extreme_rows_removed_total={st.get('extreme_rows_removed_total', 0)} "
        f"penny_rows_removed_total={st.get('penny_rows_removed_total', 0)}"
    )

    print("\n--- IPO guard ---")
    print(
        f"ipo_exempt_rows_total={st.get('ipo_exempt_rows_total', 0)}"
    )

    print("\n--- Empty after stage ---")
    print(
        f"empty_after_ohlc={st.get('empty_after_ohlc', 0)} "
        f"empty_after_scale={st.get('empty_after_scale', 0)} "
        f"empty_after_corporate_actions={st.get('empty_after_corporate_actions', 0)} "
        f"empty_after_ghost={st.get('empty_after_ghost', 0)} "
        f"empty_after_ipo_guard={st.get('empty_after_ipo_guard', 0)} "
        f"empty_after_extreme={st.get('empty_after_extreme', 0)} "
        f"empty_after_penny={st.get('empty_after_penny', 0)}"
    )


def _print_debug_logs(st: dict, max_logs: int) -> None:
    logs = st.get("debug_logs", []) or []
    if not logs:
        print("\n--- Debug logs ---")
        print("(none)")
        return

    print(f"\n--- Debug logs (showing up to {max_logs}) ---")
    for i, x in enumerate(logs[:max_logs], 1):
        print(f"\n[{i}] {x.get('file', '')} | {x.get('stop_reason', '')}")
        print(json.dumps(x, ensure_ascii=False, indent=2, default=str))


def _build_summary_row(code: str, st: dict) -> dict:
    return {
        "market": code,
        "files": st.get("files", 0),
        "ok": st.get("ok", 0),
        "fail": st.get("fail", 0),
        "skipped": st.get("skipped", 0),
        "filtered_tick_distortion": st.get("filtered_tick_distortion", 0),
        "filtered_low_price": st.get("filtered_low_price", 0),
        "empty_after_clean": st.get("empty_after_clean", 0),
        "scale_fixed_files": st.get("scale_fixed_files", 0),

        "eligible_after_clean": st.get("eligible_after_clean", 0),
        "eligible_after_tick_filter": st.get("eligible_after_tick_filter", 0),
        "eligible_after_all_filters": st.get("eligible_after_all_filters", 0),
        "coverage_before_filter_pct": st.get("coverage_before_filter_pct", 0.0),
        "coverage_after_tick_filter_pct": st.get("coverage_after_tick_filter_pct", 0.0),
        "coverage_after_all_filters_pct": st.get("coverage_after_all_filters_pct", 0.0),

        "rows_input_total": st.get("rows_input_total", 0),
        "rows_after_read_total": st.get("rows_after_read_total", 0),
        "rows_after_ohlc_total": st.get("rows_after_ohlc_total", 0),
        "rows_after_scale_total": st.get("rows_after_scale_total", 0),
        "rows_after_corporate_actions_total": st.get("rows_after_corporate_actions_total", 0),
        "rows_after_ghost_total": st.get("rows_after_ghost_total", 0),
        "rows_after_ipo_guard_total": st.get("rows_after_ipo_guard_total", 0),
        "rows_after_extreme_total": st.get("rows_after_extreme_total", 0),
        "rows_after_penny_total": st.get("rows_after_penny_total", 0),

        "ohlc_rows_removed_total": st.get("ohlc_rows_removed_total", 0),
        "corporate_action_rows_removed_total": st.get("corporate_action_rows_removed_total", 0),
        "ghost_rows_removed_total": st.get("ghost_rows_removed_total", 0),
        "extreme_rows_removed_total": st.get("extreme_rows_removed_total", 0),
        "penny_rows_removed_total": st.get("penny_rows_removed_total", 0),

        "ipo_exempt_rows_total": st.get("ipo_exempt_rows_total", 0),

        "empty_after_ohlc": st.get("empty_after_ohlc", 0),
        "empty_after_scale": st.get("empty_after_scale", 0),
        "empty_after_corporate_actions": st.get("empty_after_corporate_actions", 0),
        "empty_after_ghost": st.get("empty_after_ghost", 0),
        "empty_after_ipo_guard": st.get("empty_after_ipo_guard", 0),
        "empty_after_extreme": st.get("empty_after_extreme", 0),
        "empty_after_penny": st.get("empty_after_penny", 0),
    }


def _build_tick_bin_rows(code: str, st: dict) -> list[dict]:
    out: list[dict] = []
    dist = st.get("tick_distortion_distribution", {}) or {}
    for group, bins in dist.items():
        for bin_label, count in (bins or {}).items():
            out.append(
                {
                    "market": code,
                    "metric": "median_one_tick_pct",
                    "group": group,
                    "bin": bin_label,
                    "ticker_count": count,
                }
            )
    return out


def _build_low_price_bin_rows(code: str, st: dict) -> list[dict]:
    out: list[dict] = []
    dist = st.get("low_price_distribution", {}) or {}
    for group, bins in dist.items():
        for bin_label, count in (bins or {}).items():
            out.append(
                {
                    "market": code,
                    "metric": "low_price_ratio",
                    "group": group,
                    "bin": bin_label,
                    "ticker_count": count,
                }
            )
    return out


def main() -> None:
    args = parse_args()

    markets_cfg = _load_yaml(ROOT / "configs" / "markets.yaml")
    profile = _load_yaml(ROOT / "configs" / "profiles" / "live.yaml")

    markets = markets_cfg.get("markets", {}) or {}
    if not isinstance(markets, dict):
        raise TypeError("configs/markets.yaml: 'markets' must be a dict")

    resample_cfg = profile.get("resample", {}) or {}

    global_low_price_filter_on = _cfg_bool(resample_cfg, "low_price_filter_on", False)
    global_low_price_mode = _cfg_str(resample_cfg, "low_price_mode", "both")
    global_low_price_min_ratio = _cfg_float(resample_cfg, "low_price_min_ratio", 0.5)

    global_tick_distortion_filter_on = _cfg_bool(resample_cfg, "tick_distortion_filter_on", True)
    global_tick_distortion_mode = _cfg_str(resample_cfg, "tick_distortion_mode", "both")
    global_tick_distortion_threshold_pct = _cfg_float(resample_cfg, "tick_distortion_threshold_pct", 10.0)
    global_tick_distortion_min_ratio = _cfg_float(resample_cfg, "tick_distortion_min_ratio", 0.5)

    if args.markets.strip():
        wanted = [x.strip().lower() for x in args.markets.split(",") if x.strip()]
    else:
        wanted = [
            str(code).strip().lower()
            for code, m in markets.items()
            if isinstance(m, dict) and m.get("enabled", False)
        ]

    print("Markets to resample:", wanted)
    print("workers:", args.workers)

    summary_rows: list[dict] = []
    tick_bin_rows: list[dict] = []
    low_price_bin_rows: list[dict] = []

    for code in wanted:
        if code not in markets:
            print(f"⚠️ Skip unknown market: {code}")
            continue

        m = markets[code]
        if not isinstance(m, dict):
            continue

        week_rule = _cfg_str(m, "week_rule", "W-FRI")
        month_rule = _cfg_str(m, "month_rule", "ME")
        year_rule = _cfg_str(m, "year_rule", "YE")

        low_price_filter_on = _cfg_bool(m, "low_price_filter_on", global_low_price_filter_on)
        low_price_mode = _cfg_str(m, "low_price_mode", global_low_price_mode)
        low_price_min_ratio = _cfg_float(m, "low_price_min_ratio", global_low_price_min_ratio)

        tick_distortion_filter_on = _cfg_bool(
            m, "tick_distortion_filter_on", global_tick_distortion_filter_on
        )
        tick_distortion_mode = _cfg_str(
            m, "tick_distortion_mode", global_tick_distortion_mode
        )
        tick_distortion_threshold_pct = _cfg_float(
            m, "tick_distortion_threshold_pct", global_tick_distortion_threshold_pct
        )
        tick_distortion_min_ratio = _cfg_float(
            m, "tick_distortion_min_ratio", global_tick_distortion_min_ratio
        )

        print(f"\n=== Resample only: {code} ===")

        day_dir = ensure_dir(BASE / "cache_dayk" / code)
        week_dir = ensure_dir(BASE / "derived" / "weekK" / code)
        month_dir = ensure_dir(BASE / "derived" / "monthK" / code)
        year_dir = ensure_dir(BASE / "derived" / "yearK" / code)

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
            debug=bool(args.debug_cleaning),
            debug_max_logs=int(args.debug_max_logs),
            workers=int(args.workers),
        )

        print(f"✅ Finished {code}")
        print(f"  dayK  : {day_dir}")
        print(f"  weekK : {week_dir}")
        print(f"  monthK: {month_dir}")
        print(f"  yearK : {year_dir}")
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

        _print_market_summary(code, st)

        if args.debug_cleaning:
            _print_debug_logs(st, args.debug_max_logs)

        summary_rows.append(_build_summary_row(code, st))
        tick_bin_rows.extend(_build_tick_bin_rows(code, st))
        low_price_bin_rows.extend(_build_low_price_bin_rows(code, st))

    if not args.no_save_research:
        research_dir = ensure_dir(BASE / "research")

        summary_df = pd.DataFrame(summary_rows)
        tick_df = pd.DataFrame(tick_bin_rows)
        low_df = pd.DataFrame(low_price_bin_rows)

        summary_path = research_dir / "resample_filter_summary.csv"
        tick_path = research_dir / "tick_distortion_bins.csv"
        low_path = research_dir / "low_price_bins.csv"

        summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
        tick_df.to_csv(tick_path, index=False, encoding="utf-8-sig")
        low_df.to_csv(low_path, index=False, encoding="utf-8-sig")

        print("\n=== Research outputs saved ===")
        print(summary_path)
        print(tick_path)
        print(low_path)


if __name__ == "__main__":
    main()