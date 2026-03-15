# run.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import importlib
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import pandas as pd
import yaml


UniverseRow = Union[str, Tuple[Any, ...], Dict[str, Any]]


# =============================================================================
# Config loading
# =============================================================================
def load_yaml(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p.resolve()}")
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Generic universe helpers
# =============================================================================
def default_to_ticker(row: UniverseRow) -> str:
    # tuple/list: (symbol, name, ...) -> symbol
    if isinstance(row, (tuple, list)) and row:
        return str(row[0]).strip()
    # dict: {"symbol": "..."} / {"ticker": "..."} / {"id": "..."}
    if isinstance(row, dict):
        for k in ("symbol", "ticker", "id"):
            v = row.get(k)
            if v:
                return str(v).strip()
        return ""
    return str(row).strip()


# =============================================================================
# Downloader (yfinance) – batch + fallback
# =============================================================================
@dataclass
class DownloaderCfg:
    interval: str = "1d"
    auto_adjust: bool = True
    batch_size: int = 200
    threads: bool = True
    timeout_sec: int = 60
    retry: int = 2
    sleep_sec: float = 0.05


def _import_yfinance():
    try:
        import yfinance as yf  # type: ignore
        return yf
    except Exception as e:
        raise RuntimeError(f"yfinance import error: {e}")


def _fetch_batch_yf(
    yf,
    tickers: List[str],
    start: Optional[str],
    end_excl: Optional[str],
    *,
    cfg: DownloaderCfg,
) -> Tuple[pd.DataFrame, List[str], Optional[str]]:
    """
    Return (df_long, failed_tickers, err_msg)
    df_long columns: symbol,date,open,high,low,close,volume
    """
    empty = pd.DataFrame(columns=["symbol", "date", "open", "high", "low", "close", "volume"])
    if not tickers:
        return empty, [], None

    tickers_str = " ".join(tickers)
    try:
        df = yf.download(
            tickers=tickers_str,
            start=start,
            end=end_excl,  # yfinance end exclusive
            interval=cfg.interval,
            group_by="ticker",
            auto_adjust=cfg.auto_adjust,
            threads=cfg.threads,
            progress=False,
            timeout=cfg.timeout_sec,
        )
    except Exception as e:
        return empty, tickers, f"yf.download exception: {e}"

    if df is None or getattr(df, "empty", True):
        return empty, tickers, "yf.download empty"

    rows: List[Dict[str, Any]] = []
    failed: List[str] = []

    # single ticker sometimes becomes non-MultiIndex
    if not isinstance(df.columns, pd.MultiIndex):
        tmp = df.copy().reset_index()
        tmp.columns = [str(c).lower() for c in tmp.columns]
        if "date" not in tmp.columns and "index" in tmp.columns:
            tmp["date"] = tmp["index"]
        tmp["date"] = pd.to_datetime(tmp["date"], errors="coerce").dt.tz_localize(None).dt.strftime("%Y-%m-%d")

        sym = tickers[0]
        if "close" not in tmp.columns or pd.to_numeric(tmp["close"], errors="coerce").notna().sum() == 0:
            failed.append(sym)
        else:
            for _, r in tmp.iterrows():
                rows.append(
                    {
                        "symbol": sym,
                        "date": r.get("date"),
                        "open": r.get("open"),
                        "high": r.get("high"),
                        "low": r.get("low"),
                        "close": r.get("close"),
                        "volume": r.get("volume"),
                    }
                )
    else:
        # MultiIndex can be ('Open','AAPL') or ('AAPL','Open')
        level0 = set([c[0] for c in df.columns])
        level1 = set([c[1] for c in df.columns])
        use_level = 1 if any(s in level1 for s in tickers[: min(3, len(tickers))]) else 0

        for sym in tickers:
            try:
                sub = df.xs(sym, axis=1, level=use_level, drop_level=False)
                if sub is None or sub.empty:
                    failed.append(sym)
                    continue

                if use_level == 1:
                    sub.columns = [c[0] for c in sub.columns]
                else:
                    sub.columns = [c[1] for c in sub.columns]

                tmp = sub.copy().reset_index()
                tmp.columns = [str(c).lower() for c in tmp.columns]
                if "date" not in tmp.columns and "index" in tmp.columns:
                    tmp["date"] = tmp["index"]
                tmp["date"] = pd.to_datetime(tmp["date"], errors="coerce").dt.tz_localize(None).dt.strftime("%Y-%m-%d")

                if "close" not in tmp.columns or pd.to_numeric(tmp["close"], errors="coerce").notna().sum() == 0:
                    failed.append(sym)
                    continue

                for _, r in tmp.iterrows():
                    rows.append(
                        {
                            "symbol": sym,
                            "date": r.get("date"),
                            "open": r.get("open"),
                            "high": r.get("high"),
                            "low": r.get("low"),
                            "close": r.get("close"),
                            "volume": r.get("volume"),
                        }
                    )
            except Exception:
                failed.append(sym)

    out = pd.DataFrame(rows)
    if out.empty:
        return empty, sorted(list(set(failed + tickers))), "batch produced no rows"

    out = out.dropna(subset=["symbol", "date"]).sort_values(["symbol", "date"]).reset_index(drop=True)
    return out, sorted(list(set(failed))), None


def _fetch_one_yf(
    yf,
    ticker: str,
    start: Optional[str],
    end_excl: Optional[str],
    *,
    cfg: DownloaderCfg,
) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    max_retries = max(0, int(cfg.retry))
    last_err: Optional[str] = None

    for attempt in range(max_retries + 1):
        try:
            df = yf.download(
                ticker,
                start=start,
                end=end_excl,
                interval=cfg.interval,
                auto_adjust=cfg.auto_adjust,
                threads=False,
                progress=False,
                timeout=cfg.timeout_sec,
                group_by="column",
            )

            if df is None or df.empty:
                last_err = "empty"
                if attempt < max_retries:
                    time.sleep(1.5)
                    continue
                return None, last_err

            df = df.copy()
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] for c in df.columns]

            for c in ["Open", "High", "Low", "Close", "Volume"]:
                if c not in df.columns:
                    df[c] = pd.NA

            df = df.reset_index()
            # index column can be "Date" or something else
            if "Date" in df.columns:
                df = df.rename(columns={"Date": "date"})
            else:
                df = df.rename(columns={df.columns[0]: "date"})

            df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.tz_localize(None).dt.strftime("%Y-%m-%d")

            out = df[["date", "Open", "High", "Low", "Close", "Volume"]].rename(
                columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"}
            )
            for c in ["open", "high", "low", "close", "volume"]:
                out[c] = pd.to_numeric(out[c], errors="coerce")

            out = out[out["close"].notna()].copy()
            if out.empty:
                return None, "no_close"

            out.insert(0, "symbol", ticker)
            return out[["symbol", "date", "open", "high", "low", "close", "volume"]], None
        except Exception as e:
            last_err = f"exception: {e}"
            if attempt < max_retries:
                time.sleep(2.0)
                continue
            return None, last_err

    return None, last_err or "unknown"


def _write_dayk_csvs(out_dir: Path, df_long: pd.DataFrame) -> int:
    """
    Write per-ticker CSV: {out_dir}/{symbol}.csv
    """
    if df_long is None or df_long.empty:
        return 0

    _ensure_dir(out_dir)

    df = df_long.copy()
    # sanitize symbol for filename
    df["__fn__"] = df["symbol"].astype(str).str.replace("/", "_", regex=False).str.replace("\\", "_", regex=False)

    n_files = 0
    for sym, g in df.groupby("__fn__", sort=False):
        p = out_dir / f"{sym}.csv"
        gg = g.drop(columns=["__fn__"]).copy()
        # sort by date
        if "date" in gg.columns:
            gg = gg.sort_values("date")
        gg.to_csv(p, index=False, encoding="utf-8")
        n_files += 1

    return n_files


# =============================================================================
# Market runner
# =============================================================================
def run_one_market(
    code: str,
    mcfg: Dict[str, Any],
    global_downloader_cfg: DownloaderCfg,
    *,
    start: Optional[str],
    end: Optional[str],
    limit_override: int = 0,
) -> Dict[str, Any]:
    """
    Returns summary dict for one market.
    """
    t0 = time.time()

    module_name = (mcfg.get("module") or "").strip()
    out_dir = Path((mcfg.get("out_dir") or "").strip())
    if not module_name:
        raise ValueError(f"[{code}] missing markets.<code>.module in configs/markets.yaml")
    if not out_dir:
        raise ValueError(f"[{code}] missing markets.<code>.out_dir in configs/markets.yaml")

    # merge cfg: market-level keys should be passed to get_universe
    cfg_for_market = dict(mcfg)

    # allow quick limit override (for testing)
    if limit_override and limit_override > 0:
        cfg_for_market["limit"] = int(limit_override)

    mod = importlib.import_module(module_name)

    if not hasattr(mod, "get_universe"):
        raise AttributeError(f"[{code}] module {module_name} missing get_universe(cfg)")

    rows: List[UniverseRow] = list(mod.get_universe(cfg_for_market))  # type: ignore

    # optional module.to_ticker
    to_ticker_fn = getattr(mod, "to_ticker", None)
    if callable(to_ticker_fn):
        tickers = [str(to_ticker_fn(r)).strip() for r in rows]
    else:
        tickers = [default_to_ticker(r) for r in rows]

    tickers = [t for t in tickers if t]
    if not tickers:
        return {"code": code, "module": module_name, "out_dir": str(out_dir), "total": 0, "ok": 0, "failed": 0, "secs": round(time.time() - t0, 2)}

    # apply limit from cfg
    lim = int(cfg_for_market.get("limit", 0) or 0)
    if lim > 0:
        tickers = tickers[:lim]

    # yfinance work
    yf = _import_yfinance()
    cfg = global_downloader_cfg

    batches = [tickers[i : i + cfg.batch_size] for i in range(0, len(tickers), cfg.batch_size)]
    ok: set[str] = set()
    failed: set[str] = set()

    for batch in batches:
        df_long, failed_batch, err = _fetch_batch_yf(yf, batch, start, end, cfg=cfg)
        if err:
            # batch-level failure: mark all failed, still continue
            failed.update(batch)
            time.sleep(cfg.sleep_sec)
            continue

        if df_long is not None and not df_long.empty:
            _write_dayk_csvs(out_dir, df_long)
            ok.update(set(df_long["symbol"].astype(str).unique().tolist()))

        failed_set = set(failed_batch)
        # try fallback
        if failed_set:
            for sym in sorted(failed_set):
                df_one, _e = _fetch_one_yf(yf, sym, start, end, cfg=cfg)
                if df_one is not None and not df_one.empty:
                    _write_dayk_csvs(out_dir, df_one)
                    ok.add(sym)
                else:
                    failed.add(sym)

        time.sleep(cfg.sleep_sec)

    # final failed = requested - ok
    failed = set(tickers) - ok

    return {
        "code": code,
        "module": module_name,
        "out_dir": str(out_dir),
        "total": len(tickers),
        "ok": len(ok),
        "failed": len(failed),
        "failed_sample": sorted(list(failed))[:20],
        "secs": round(time.time() - t0, 2),
    }


# =============================================================================
# CLI
# =============================================================================
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/markets.yaml", help="Path to configs/markets.yaml")
    ap.add_argument("--all", action="store_true", help="Run all markets in config")
    ap.add_argument("--markets", default="", help="Comma-separated market codes, e.g. us,jp,kr")
    ap.add_argument("--start", default="", help="Start date YYYY-MM-DD (optional)")
    ap.add_argument("--end", default="", help="End date EXCLUSIVE YYYY-MM-DD (optional, yfinance end exclusive)")
    ap.add_argument("--limit", type=int, default=0, help="Override per-market limit (0 = no override)")
    ap.add_argument("--workers", type=int, default=3, help="How many markets to run in parallel")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_yaml(args.config)

    markets_cfg: Dict[str, Any] = (cfg.get("markets") or {})  # type: ignore
    if not markets_cfg:
        print("No markets found in config.", file=sys.stderr)
        return 2

    dcfg_raw = cfg.get("downloader") or {}
    dcfg = DownloaderCfg(
        interval=str(dcfg_raw.get("interval", "1d")),
        auto_adjust=bool(dcfg_raw.get("auto_adjust", True)),
        batch_size=int(dcfg_raw.get("batch_size", 200)),
        threads=bool(dcfg_raw.get("threads", True)),
        timeout_sec=int(dcfg_raw.get("timeout_sec", 60)),
        retry=int(dcfg_raw.get("retry", 2)),
        sleep_sec=float(dcfg_raw.get("sleep_sec", 0.05)),
    )

    start = (args.start or "").strip() or None
    end = (args.end or "").strip() or None

    if args.all:
        codes = list(markets_cfg.keys())
    else:
        codes = [c.strip() for c in (args.markets or "").split(",") if c.strip()]
        if not codes:
            print("Please specify --all or --markets us,jp,...", file=sys.stderr)
            return 2

    # validate codes
    bad = [c for c in codes if c not in markets_cfg]
    if bad:
        print(f"Unknown market codes in config: {bad}", file=sys.stderr)
        return 2

    print(f"▶ Run markets: {codes}")
    print(f"▶ Config: {Path(args.config).resolve()}")
    print(f"▶ Downloader: {dcfg}")
    if start or end:
        print(f"▶ Date range: start={start} end_excl={end}")
    if args.limit:
        print(f"▶ Limit override: {args.limit}")

    # parallel markets
    from concurrent.futures import ThreadPoolExecutor, as_completed

    results: List[Dict[str, Any]] = []
    failures: List[Tuple[str, str]] = []

    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as ex:
        futs = {}
        for code in codes:
            mcfg = dict(markets_cfg[code] or {})
            fut = ex.submit(
                run_one_market,
                code,
                mcfg,
                dcfg,
                start=start,
                end=end,
                limit_override=int(args.limit or 0),
            )
            futs[fut] = code

        for fut in as_completed(futs):
            code = futs[fut]
            try:
                r = fut.result()
                results.append(r)
                print(f"✅ [{code}] ok={r['ok']}/{r['total']} failed={r['failed']} secs={r['secs']} out={r['out_dir']}")
                if r.get("failed"):
                    print(f"   ↳ failed_sample: {r.get('failed_sample')}")
            except Exception as e:
                failures.append((code, str(e)))
                print(f"❌ [{code}] exception: {e}", file=sys.stderr)

    # summary
    results.sort(key=lambda x: x["code"])
    total_req = sum(int(r.get("total", 0)) for r in results)
    total_ok = sum(int(r.get("ok", 0)) for r in results)
    total_failed = sum(int(r.get("failed", 0)) for r in results)

    print("\n================ SUMMARY ================")
    for r in results:
        print(f"{r['code']:>5}  ok={r['ok']:>6}/{r['total']:<6}  failed={r['failed']:<6}  secs={r['secs']:<6}  out={r['out_dir']}")
    if failures:
        print("\n-------------- EXCEPTIONS --------------", file=sys.stderr)
        for code, err in failures:
            print(f"{code}: {err}", file=sys.stderr)

    print("----------------------------------------")
    print(f"TOTAL ok={total_ok}/{total_req} failed={total_failed} markets={len(results)} exceptions={len(failures)}")
    print("========================================\n")

    # if any market hard-exception -> non-zero
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())