from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

from core.cleaning.penny import apply_penny_rules
from core.cleaning.ohlc import clean_ohlc
from core.cleaning.extremes import apply_extreme_filters
from core.cleaning.resume_ghost import apply_resume_ghost
from core.cleaning.corporate_actions import apply_corporate_action_fixes
from core.cleaning.ipo_guard import apply_ipo_guard
from core.filtering.low_price import should_drop_low_price_ticker
from core.filtering.tick_distortion import should_drop_tick_distortion_ticker

try:
    from core.cleaning.scale_uk import should_apply_uk_scale, apply_scale_fix
except Exception:
    def should_apply_uk_scale(*args, **kwargs) -> bool:
        return False

    def apply_scale_fix(df: pd.DataFrame, *args, **kwargs):
        if kwargs.get("return_stats", False):
            class _DummyScaleStats:
                n_scaled_down = 0
                n_scaled_up = 0
                notes = "scale_uk unavailable"
            return df, _DummyScaleStats()
        return df


KEEP_COLS = ["date", "open", "high", "low", "close", "volume"]


def _normalize_rule(rule: str) -> str:
    r = (rule or "").strip().upper()
    if not r:
        return rule
    if r == "M":
        return "ME"
    if r in {"Y", "A"}:
        return "YE"
    return rule


def _safe_read_day(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if not set(KEEP_COLS).issubset(df.columns):
        raise ValueError(f"missing cols in {csv_path.name}")
    return df[KEEP_COLS].copy()


def _resample_ohlcv(df_day: pd.DataFrame, rule: str) -> pd.DataFrame:
    rule = _normalize_rule(rule)
    g = df_day.set_index("date").sort_index()

    out = pd.DataFrame(
        {
            "open": g["open"].resample(rule).first(),
            "high": g["high"].resample(rule).max(),
            "low": g["low"].resample(rule).min(),
            "close": g["close"].resample(rule).last(),
            "volume": g["volume"].resample(rule).sum(min_count=1),
        }
    ).dropna(subset=["open", "high", "low", "close"])

    out = out.reset_index()
    return out[KEEP_COLS]


def _add_returns(df: pd.DataFrame, tag: str) -> pd.DataFrame:
    df = df.sort_values("date").reset_index(drop=True).copy()
    prev = df["close"].shift(1).replace(0, np.nan)
    df[f"prev_close_{tag}"] = prev
    df[f"ret_high_{tag}"] = (df["high"] / prev) - 1
    df[f"ret_close_{tag}"] = (df["close"] / prev) - 1
    df[f"ret_low_{tag}"] = (df["low"] / prev) - 1
    return df


def _clip_range(df: pd.DataFrame, start: Optional[str], end: Optional[str]) -> pd.DataFrame:
    if df.empty:
        return df
    if start:
        s = pd.to_datetime(start)
        df = df[df["date"] >= s]
    if end:
        e = pd.to_datetime(end)
        df = df[df["date"] <= e]
    return df


def _stats_to_dict(obj: Any) -> Dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    if is_dataclass(obj):
        return asdict(obj)
    if hasattr(obj, "__dict__"):
        return {k: v for k, v in vars(obj).items() if not k.startswith("_")}
    return {}


def _int_from_stats(d: Dict[str, Any], key: str, default: int = 0) -> int:
    try:
        return int(d.get(key, default) or 0)
    except Exception:
        return int(default)


def _float_from_stats(d: Dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(d.get(key, default) or 0.0)
    except Exception:
        return float(default)


def _init_counter(keys: list[str]) -> Dict[str, int]:
    return {k: 0 for k in keys}


TICK_BINS = [
    "<=0.1%",
    "0.1%~0.5%",
    "0.5%~1%",
    "1%~2%",
    "2%~5%",
    "5%~10%",
    ">10%",
]

LOW_PRICE_RATIO_BINS = [
    "0%",
    "(0,10%]",
    "(10%,25%]",
    "(25%,50%]",
    "(50%,75%]",
    "(75%,100%]",
]


def _bucket_tick_pct(x: float) -> str:
    if x <= 0.1:
        return "<=0.1%"
    if x <= 0.5:
        return "0.1%~0.5%"
    if x <= 1.0:
        return "0.5%~1%"
    if x <= 2.0:
        return "1%~2%"
    if x <= 5.0:
        return "2%~5%"
    if x <= 10.0:
        return "5%~10%"
    return ">10%"


def _bucket_low_price_ratio(x: float) -> str:
    if x <= 0:
        return "0%"
    if x <= 0.10:
        return "(0,10%]"
    if x <= 0.25:
        return "(10%,25%]"
    if x <= 0.50:
        return "(25%,50%]"
    if x <= 0.75:
        return "(50%,75%]"
    return "(75%,100%]"


def _pct(n: int, d: int) -> float:
    return float(n / d * 100.0) if d > 0 else 0.0


def _make_empty_stats() -> Dict[str, Any]:
    return {
        "files": 0,
        "ok": 0,
        "fail": 0,
        "skipped": 0,
        "filtered_tick_distortion": 0,
        "filtered_low_price": 0,
        "empty_after_clean": 0,
        "scale_fixed_files": 0,

        "rows_input_total": 0,
        "rows_after_read_total": 0,
        "rows_after_ohlc_total": 0,
        "rows_after_scale_total": 0,
        "rows_after_corporate_actions_total": 0,
        "rows_after_ghost_total": 0,
        "rows_after_ipo_guard_total": 0,
        "rows_after_extreme_total": 0,
        "rows_after_penny_total": 0,

        "ohlc_rows_removed_total": 0,
        "ghost_rows_removed_total": 0,
        "extreme_rows_removed_total": 0,
        "penny_rows_removed_total": 0,
        "corporate_action_rows_removed_total": 0,

        "ipo_exempt_rows_total": 0,

        "empty_after_ohlc": 0,
        "empty_after_scale": 0,
        "empty_after_corporate_actions": 0,
        "empty_after_ghost": 0,
        "empty_after_ipo_guard": 0,
        "empty_after_extreme": 0,
        "empty_after_penny": 0,

        "eligible_after_clean": 0,
        "eligible_after_tick_filter": 0,
        "eligible_after_all_filters": 0,
        "coverage_before_filter_pct": 0.0,
        "coverage_after_tick_filter_pct": 0.0,
        "coverage_after_all_filters_pct": 0.0,

        "tick_distortion_distribution": {
            "all": _init_counter(TICK_BINS),
            "kept": _init_counter(TICK_BINS),
            "filtered_tick": _init_counter(TICK_BINS),
        },
        "low_price_distribution": {
            "all": _init_counter(LOW_PRICE_RATIO_BINS),
            "kept": _init_counter(LOW_PRICE_RATIO_BINS),
            "filtered_low_price": _init_counter(LOW_PRICE_RATIO_BINS),
        },

        "debug_logs": [],
    }


def _merge_stats(total: Dict[str, Any], part: Dict[str, Any]) -> None:
    for k, v in part.items():
        if k in {"tick_distortion_distribution", "low_price_distribution"}:
            for subk, subd in v.items():
                for bucket, count in subd.items():
                    total[k][subk][bucket] += count
        elif k == "debug_logs":
            total["debug_logs"].extend(v)
        elif isinstance(v, (int, float)) and k in total:
            total[k] += v


def _process_one_file(
    csv: Path,
    *,
    out_week_dir: Path,
    out_month_dir: Path,
    out_year_dir: Path,
    market_code: Optional[str],
    week_rule: str,
    month_rule: str,
    year_rule: str,
    overwrite: bool,
    start: Optional[str],
    end: Optional[str],
    low_price_filter_on: bool,
    low_price_mode: str,
    low_price_min_ratio: float,
    tick_distortion_filter_on: bool,
    tick_distortion_mode: str,
    tick_distortion_threshold_pct: float,
    tick_distortion_min_ratio: float,
    debug: bool,
) -> Dict[str, Any]:
    part = _make_empty_stats()
    part["files"] = 1

    wk_out = out_week_dir / csv.name
    mo_out = out_month_dir / csv.name
    yr_out = out_year_dir / csv.name

    if (not overwrite) and wk_out.exists() and mo_out.exists() and yr_out.exists():
        part["skipped"] = 1
        return part

    ticker = csv.stem
    file_log: Dict[str, Any] = {"ticker": ticker, "file": csv.name}

    try:
        # 1) raw read
        df_day = _safe_read_day(csv)
        rows0 = len(df_day)
        part["rows_input_total"] += rows0
        part["rows_after_read_total"] += rows0
        file_log["rows_input"] = rows0

        # 2) base OHLC cleaning
        df_day, ohlc_stats = clean_ohlc(
            df_day,
            nonpositive_policy="all",
            fix_high_low=True,
            allow_zero_volume=True,
            drop_if_no_close=True,
        )
        ohlc_d = _stats_to_dict(ohlc_stats)
        rows1 = len(df_day)
        part["rows_after_ohlc_total"] += rows1
        part["ohlc_rows_removed_total"] += max(0, rows0 - rows1)
        file_log["rows_after_ohlc"] = rows1
        file_log["ohlc_stats"] = ohlc_d

        if df_day.empty:
            part["empty_after_clean"] += 1
            part["empty_after_ohlc"] += 1
            if debug:
                file_log["stop_reason"] = "empty_after_ohlc"
                part["debug_logs"].append(file_log)
            return part

        # 3) UK scale fix
        scale_d: Dict[str, Any] = {}
        if should_apply_uk_scale(market_code=market_code, ticker=ticker):
            df_day, scale_stats = apply_scale_fix(df_day, return_stats=True)
            scale_d = _stats_to_dict(scale_stats)
            n_scaled_down = _int_from_stats(scale_d, "n_scaled_down", 0)
            n_scaled_up = _int_from_stats(scale_d, "n_scaled_up", 0)
            if (n_scaled_down > 0) or (n_scaled_up > 0):
                part["scale_fixed_files"] += 1

        rows2 = len(df_day)
        part["rows_after_scale_total"] += rows2
        file_log["rows_after_scale"] = rows2
        file_log["scale_stats"] = scale_d

        if df_day.empty:
            part["empty_after_clean"] += 1
            part["empty_after_scale"] += 1
            if debug:
                file_log["stop_reason"] = "empty_after_scale"
                part["debug_logs"].append(file_log)
            return part

        # 4) corporate actions
        rows_before_ca = len(df_day)
        df_day, ca_stats = apply_corporate_action_fixes(
            df_day,
            market_code=market_code,
            ticker=ticker,
        )
        ca_d = _stats_to_dict(ca_stats)
        rows3 = len(df_day)
        part["rows_after_corporate_actions_total"] += rows3
        part["corporate_action_rows_removed_total"] += max(0, rows_before_ca - rows3)
        file_log["rows_after_corporate_actions"] = rows3
        file_log["corporate_action_stats"] = ca_d

        if df_day.empty:
            part["empty_after_clean"] += 1
            part["empty_after_corporate_actions"] += 1
            if debug:
                file_log["stop_reason"] = "empty_after_corporate_actions"
                part["debug_logs"].append(file_log)
            return part

        # 5) ghost rows
        rows_before_ghost = len(df_day)
        df_day, ghost_stats = apply_resume_ghost(
            df_day,
            market_code=market_code,
            ticker=ticker,
        )
        ghost_d = _stats_to_dict(ghost_stats)
        rows4 = len(df_day)
        part["rows_after_ghost_total"] += rows4
        part["ghost_rows_removed_total"] += max(0, rows_before_ghost - rows4)
        file_log["rows_after_ghost"] = rows4
        file_log["ghost_stats"] = ghost_d

        if df_day.empty:
            part["empty_after_clean"] += 1
            part["empty_after_ghost"] += 1
            if debug:
                file_log["stop_reason"] = "empty_after_ghost"
                part["debug_logs"].append(file_log)
            return part

        # 6) IPO guard
        df_day, ipo_stats = apply_ipo_guard(
            df_day,
            market_code=market_code,
            ticker=ticker,
        )
        ipo_d = _stats_to_dict(ipo_stats)
        rows4b = len(df_day)
        part["rows_after_ipo_guard_total"] += rows4b
        part["ipo_exempt_rows_total"] += _int_from_stats(ipo_d, "exempt_rows", 0)
        file_log["rows_after_ipo_guard"] = rows4b
        file_log["ipo_guard_stats"] = ipo_d

        if df_day.empty:
            part["empty_after_clean"] += 1
            part["empty_after_ipo_guard"] += 1
            if debug:
                file_log["stop_reason"] = "empty_after_ipo_guard"
                part["debug_logs"].append(file_log)
            return part

        # 7) extreme rows
        rows_before_extreme = len(df_day)
        df_day, extreme_stats = apply_extreme_filters(
            df_day,
            market_code=market_code,
            ticker=ticker,
        )
        extreme_d = _stats_to_dict(extreme_stats)
        rows5 = len(df_day)
        part["rows_after_extreme_total"] += rows5
        part["extreme_rows_removed_total"] += max(0, rows_before_extreme - rows5)
        file_log["rows_after_extreme"] = rows5
        file_log["extreme_stats"] = extreme_d

        if df_day.empty:
            part["empty_after_clean"] += 1
            part["empty_after_extreme"] += 1
            if debug:
                file_log["stop_reason"] = "empty_after_extreme"
                part["debug_logs"].append(file_log)
            return part

        # 8) penny cleaning
        rows_before_penny = len(df_day)
        df_day, penny_stats = apply_penny_rules(
            df_day,
            market_code=market_code,
            ticker=ticker,
        )
        penny_d = _stats_to_dict(penny_stats)
        rows6 = len(df_day)
        part["rows_after_penny_total"] += rows6
        part["penny_rows_removed_total"] += max(0, rows_before_penny - rows6)
        file_log["rows_after_penny"] = rows6
        file_log["penny_stats"] = penny_d

        if df_day.empty:
            part["empty_after_clean"] += 1
            part["empty_after_penny"] += 1
            if debug:
                file_log["stop_reason"] = "empty_after_penny"
                part["debug_logs"].append(file_log)
            return part

        part["eligible_after_clean"] += 1

        # 9) tick distortion
        drop_tick, tick_stats = should_drop_tick_distortion_ticker(
            df_day,
            market_code=market_code,
            ticker=ticker,
            price_col="close",
            threshold_pct=tick_distortion_threshold_pct,
            min_ratio=tick_distortion_min_ratio,
            mode=tick_distortion_mode,
            enabled=tick_distortion_filter_on,
        )
        tick_d = _stats_to_dict(tick_stats)
        file_log["tick_distortion_stats"] = tick_d

        tick_median = _float_from_stats(tick_d, "median_one_tick_pct", 0.0)
        tick_bucket = _bucket_tick_pct(tick_median)
        part["tick_distortion_distribution"]["all"][tick_bucket] += 1

        # 10) low price
        drop_lp, lp_stats = should_drop_low_price_ticker(
            df_day,
            market_code=market_code,
            ticker=ticker,
            mode=low_price_mode,
            min_ratio=low_price_min_ratio,
            enabled=low_price_filter_on,
        )
        lp_d = _stats_to_dict(lp_stats)
        file_log["low_price_stats"] = lp_d

        low_ratio = _float_from_stats(lp_d, "low_price_ratio", 0.0)
        low_bucket = _bucket_low_price_ratio(low_ratio)
        part["low_price_distribution"]["all"][low_bucket] += 1

        if not drop_tick:
            part["eligible_after_tick_filter"] += 1
        else:
            part["filtered_tick_distortion"] += 1
            part["tick_distortion_distribution"]["filtered_tick"][tick_bucket] += 1
            if debug:
                file_log["stop_reason"] = "filtered_tick_distortion"
                part["debug_logs"].append(file_log)
            return part

        if drop_lp:
            part["filtered_low_price"] += 1
            part["low_price_distribution"]["filtered_low_price"][low_bucket] += 1
            if debug:
                file_log["stop_reason"] = "filtered_low_price"
                part["debug_logs"].append(file_log)
            return part

        part["eligible_after_all_filters"] += 1
        part["tick_distortion_distribution"]["kept"][tick_bucket] += 1
        part["low_price_distribution"]["kept"][low_bucket] += 1

        # 11) resample + save
        wk = _add_returns(_resample_ohlcv(df_day, week_rule), "W")
        mo = _add_returns(_resample_ohlcv(df_day, month_rule), "M")
        yr = _add_returns(_resample_ohlcv(df_day, year_rule), "Y")

        wk = _clip_range(wk, start, end)
        mo = _clip_range(mo, start, end)
        yr = _clip_range(yr, start, end)

        wk.to_csv(wk_out, index=False, encoding="utf-8-sig")
        mo.to_csv(mo_out, index=False, encoding="utf-8-sig")
        yr.to_csv(yr_out, index=False, encoding="utf-8-sig")

        part["ok"] += 1

        if debug:
            file_log["stop_reason"] = "ok"
            file_log["week_rows"] = len(wk)
            file_log["month_rows"] = len(mo)
            file_log["year_rows"] = len(yr)
            part["debug_logs"].append(file_log)

        return part

    except Exception as e:
        part["fail"] += 1
        if debug:
            part["debug_logs"].append(
                {
                    "file": csv.name,
                    "ticker": csv.stem,
                    "stop_reason": "exception",
                    "error_type": type(e).__name__,
                    "error": str(e),
                }
            )
        print(f"❌ resample fail: {csv.name} | {type(e).__name__}: {e}")
        return part


def resample_all(
    day_dir: Union[str, Path],
    *,
    out_week_dir: Union[str, Path],
    out_month_dir: Union[str, Path],
    out_year_dir: Union[str, Path],
    market_code: Optional[str] = None,
    week_rule: str = "W-FRI",
    month_rule: str = "ME",
    year_rule: str = "YE",
    overwrite: bool = False,
    start: Optional[str] = None,
    end: Optional[str] = None,
    low_price_filter_on: bool = False,
    low_price_mode: str = "both",
    low_price_min_ratio: float = 0.5,
    tick_distortion_filter_on: bool = True,
    tick_distortion_mode: str = "both",
    tick_distortion_threshold_pct: float = 10.0,
    tick_distortion_min_ratio: float = 0.5,
    debug: bool = False,
    debug_max_logs: int = 50,
    workers: int = 8,
) -> Dict[str, Any]:
    day_dir = Path(day_dir)
    out_week_dir = Path(out_week_dir)
    out_month_dir = Path(out_month_dir)
    out_year_dir = Path(out_year_dir)

    out_week_dir.mkdir(parents=True, exist_ok=True)
    out_month_dir.mkdir(parents=True, exist_ok=True)
    out_year_dir.mkdir(parents=True, exist_ok=True)

    stats = _make_empty_stats()

    week_rule = _normalize_rule(week_rule)
    month_rule = _normalize_rule(month_rule)
    year_rule = _normalize_rule(year_rule)

    csvs = sorted(day_dir.glob("*.csv"))
    if not csvs:
        return stats

    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futures = {
            ex.submit(
                _process_one_file,
                csv,
                out_week_dir=out_week_dir,
                out_month_dir=out_month_dir,
                out_year_dir=out_year_dir,
                market_code=market_code,
                week_rule=week_rule,
                month_rule=month_rule,
                year_rule=year_rule,
                overwrite=overwrite,
                start=start,
                end=end,
                low_price_filter_on=low_price_filter_on,
                low_price_mode=low_price_mode,
                low_price_min_ratio=low_price_min_ratio,
                tick_distortion_filter_on=tick_distortion_filter_on,
                tick_distortion_mode=tick_distortion_mode,
                tick_distortion_threshold_pct=tick_distortion_threshold_pct,
                tick_distortion_min_ratio=tick_distortion_min_ratio,
                debug=debug,
            ): csv
            for csv in csvs
        }

        for fut in tqdm(as_completed(futures), total=len(futures), desc=f"Resample {market_code or day_dir.name}", unit="file"):
            part = fut.result()
            _merge_stats(stats, part)

    if len(stats["debug_logs"]) > debug_max_logs:
        stats["debug_logs"] = stats["debug_logs"][:debug_max_logs]

    files_effective = int(stats["files"] - stats["skipped"])
    eligible_after_clean = int(stats["eligible_after_clean"])
    eligible_after_tick_filter = int(stats["eligible_after_tick_filter"])
    eligible_after_all_filters = int(stats["eligible_after_all_filters"])

    stats["coverage_before_filter_pct"] = _pct(eligible_after_clean, files_effective)
    stats["coverage_after_tick_filter_pct"] = _pct(eligible_after_tick_filter, eligible_after_clean)
    stats["coverage_after_all_filters_pct"] = _pct(eligible_after_all_filters, eligible_after_clean)

    return stats