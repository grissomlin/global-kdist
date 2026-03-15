# core/cleaning/scale_uk.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union, Literal

import pandas as pd


def should_apply_uk_scale(*, market_code: Optional[str], ticker: str) -> bool:
    mc = (market_code or "").strip().lower()
    t = (ticker or "").strip().upper()
    if mc == "uk":
        return True
    if t.endswith(".L"):
        return True
    return False


def normalize_uk_scale(
    df: pd.DataFrame,
    *,
    upper_ratio: float = 20.0,
    lower_ratio: float = 0.05,
    factor: float = 100.0,
    max_passes: int = 3,
) -> Tuple[pd.DataFrame, int, int]:
    """
    Backward compatible wrapper:
      returns (df_fixed, n_scaled_down, n_scaled_up)
    """
    fixed, st = apply_scale_fix(
        df,
        upper_ratio=upper_ratio,
        lower_ratio=lower_ratio,
        factor=factor,
        max_passes=max_passes,
        return_stats=True,
    )
    return fixed, st.n_scaled_down, st.n_scaled_up


def _ensure_std_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.tz_localize(None)
    for c in ("open", "high", "low", "close"):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def _near(x: pd.Series, center: float, tol: float) -> pd.Series:
    lo = center * (1.0 - tol)
    hi = center * (1.0 + tol)
    return x.between(lo, hi)


def detect_scale_candidates(
    df: pd.DataFrame,
    *,
    mode: Literal["near_factor", "ratio"] = "near_factor",
    factor: float = 100.0,
    tol: float = 0.08,
    min_prev_close: float = 0.10,
    # IMPORTANT: keep as an option, but default False for stability in metrics/fixing
    require_reversion: bool = False,
    reversion_tol: float = 0.35,
    upper_ratio: float = 20.0,
    lower_ratio: float = 0.05,
) -> pd.DataFrame:
    cols = ["idx", "date", "prev_close", "close", "ratio", "direction"]
    if df is None or df.empty or len(df) < 2:
        return pd.DataFrame(columns=cols)

    x = _ensure_std_cols(df)
    if "date" in x.columns:
        x = x.sort_values("date").reset_index(drop=True)
    else:
        x = x.reset_index(drop=True)

    if "close" not in x.columns:
        return pd.DataFrame(columns=cols)

    c = pd.to_numeric(x["close"], errors="coerce")
    prev = c.shift(1)
    nxt = c.shift(-1)

    ratio = (c / prev).replace([pd.NA, float("inf"), float("-inf")], pd.NA)
    ratio_next = (nxt / c).replace([pd.NA, float("inf"), float("-inf")], pd.NA)

    if mode == "ratio":
        mask_down = ratio >= float(upper_ratio)
        mask_up = ratio <= float(lower_ratio)
        mask = (mask_down | mask_up).fillna(False)
    else:
        valid = prev.notna() & c.notna() & (prev.abs() >= float(min_prev_close))
        ratio = ratio.where(valid)

        f = float(factor)
        inv = 1.0 / f
        t = float(tol)

        mask_down = _near(ratio, f, t)
        mask_up = _near(ratio, inv, t)
        mask = (mask_down | mask_up).fillna(False)

        if require_reversion:
            rt = float(reversion_tol)
            normal_next = ratio_next.between(1.0 - rt, 1.0 + rt)
            mask = mask & (normal_next.fillna(True))

    if not mask.any():
        return pd.DataFrame(columns=cols)

    out = pd.DataFrame(
        {
            "idx": mask[mask].index.astype(int),
            "date": x.loc[mask, "date"].values if "date" in x.columns else pd.NaT,
            "prev_close": prev.loc[mask].values,
            "close": c.loc[mask].values,
            "ratio": ratio.loc[mask].values,
            "direction": ["down" if bool(d) else "up" for d in mask_down.loc[mask].values],
        }
    )
    return out.reset_index(drop=True)


@dataclass
class UKScaleFixStats:
    passes: int = 0
    candidates_before: int = 0
    candidates_after: int = 0
    n_scaled_down: int = 0
    n_scaled_up: int = 0


def _apply_one_pass(
    df: pd.DataFrame,
    cand: pd.DataFrame,
    *,
    factor: float = 100.0,
) -> Tuple[pd.DataFrame, int, int]:
    if df is None or df.empty or cand is None or cand.empty:
        return df, 0, 0

    out = df.copy()
    n_down = 0
    n_up = 0

    for _, r in cand.iterrows():
        i = int(r["idx"])
        if i < 0 or i >= len(out):
            continue

        direction = str(r.get("direction") or "").lower().strip()
        if direction == "down":
            mul = 1.0 / float(factor)
            n_down += 1
        else:
            mul = float(factor)
            n_up += 1

        for col in ("open", "high", "low", "close"):
            if col in out.columns:
                v = out.at[i, col]
                if v is not None and pd.notna(v):
                    out.at[i, col] = float(v) * mul

    return out, n_down, n_up


def apply_scale_fix(
    df: pd.DataFrame,
    candidates: Optional[pd.DataFrame] = None,
    *,
    upper_ratio: float = 20.0,
    lower_ratio: float = 0.05,
    factor: float = 100.0,
    tol: float = 0.08,
    min_prev_close: float = 0.10,
    mode: Literal["near_factor", "ratio"] = "near_factor",
    # IMPORTANT: fixing should NOT depend on reversion; keep it False here
    require_reversion: bool = False,
    reversion_tol: float = 0.35,
    max_passes: int = 3,
    return_stats: bool = False,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, UKScaleFixStats]]:
    st = UKScaleFixStats()

    if df is None or df.empty or len(df) < 2:
        return (df, st) if return_stats else df

    out = _ensure_std_cols(df)
    if "date" in out.columns:
        out = out.sort_values("date").reset_index(drop=True)
    else:
        out = out.reset_index(drop=True)

    if candidates is None:
        cand0 = detect_scale_candidates(
            out,
            mode=mode,
            factor=factor,
            tol=tol,
            min_prev_close=min_prev_close,
            require_reversion=require_reversion,
            reversion_tol=reversion_tol,
            upper_ratio=upper_ratio,
            lower_ratio=lower_ratio,
        )
    else:
        cand0 = candidates.copy()

    st.candidates_before = int(len(cand0)) if cand0 is not None else 0

    for p in range(int(max_passes)):
        st.passes = p + 1

        cand = (
            cand0
            if p == 0
            else detect_scale_candidates(
                out,
                mode=mode,
                factor=factor,
                tol=tol,
                min_prev_close=min_prev_close,
                require_reversion=require_reversion,
                reversion_tol=reversion_tol,
                upper_ratio=upper_ratio,
                lower_ratio=lower_ratio,
            )
        )
        if cand is None or cand.empty:
            break

        out, n_down, n_up = _apply_one_pass(out, cand, factor=factor)
        st.n_scaled_down += int(n_down)
        st.n_scaled_up += int(n_up)

    resid = detect_scale_candidates(
        out,
        mode=mode,
        factor=factor,
        tol=tol,
        min_prev_close=min_prev_close,
        require_reversion=require_reversion,
        reversion_tol=reversion_tol,
        upper_ratio=upper_ratio,
        lower_ratio=lower_ratio,
    )
    st.candidates_after = int(len(resid)) if resid is not None else 0

    return (out, st) if return_stats else out


UKScaleStats = UKScaleFixStats

__all__ = [
    "should_apply_uk_scale",
    "normalize_uk_scale",
    "detect_scale_candidates",
    "apply_scale_fix",
    "UKScaleFixStats",
    "UKScaleStats",
]