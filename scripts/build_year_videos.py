# scripts/build_year_videos.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path
from typing import List

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
# ffmpeg helpers
# ============================================================
def _require_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found in PATH. Please install ffmpeg first.")


def _run_ffmpeg(cmd: List[str], label: str) -> int:
    print(f"\n🎬 Building {label} ...")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"❌ Failed: {label}")
        print(proc.stderr[:2000])
    return proc.returncode


def _write_ffmpeg_concat_list(imgs: List[Path], list_path: Path, duration_sec: float) -> None:
    with list_path.open("w", encoding="utf-8") as f:
        for p in imgs:
            f.write(f"file '{p.resolve().as_posix()}'\n")
            f.write(f"duration {duration_sec}\n")
        f.write(f"file '{imgs[-1].resolve().as_posix()}'\n")


# ============================================================
# Helpers
# ============================================================
def _normalize_bin_mode(bin_mode: str) -> str:
    s = str(bin_mode or "").strip().lower()
    if s in {"10", "10pct", "10%"}:
        return "10pct"
    if s in {"100", "100pct", "100%"}:
        return "100pct"
    if s in {"both", "all"}:
        return "both"
    raise ValueError(f"Unsupported --bin-mode: {bin_mode}")


def _image_root_by_bin_mode(bin_mode: str) -> Path:
    if bin_mode == "10pct":
        return DATA_ROOT / "derived_images" / "yearK"
    if bin_mode == "100pct":
        return DATA_ROOT / "derived_images" / "yearK_100pct"
    raise ValueError(f"Unsupported bin mode: {bin_mode}")


def _video_root_by_bin_mode(bin_mode: str) -> Path:
    if bin_mode == "10pct":
        return DATA_ROOT / "videos" / "yearK"
    if bin_mode == "100pct":
        return DATA_ROOT / "videos" / "yearK_100pct"
    raise ValueError(f"Unsupported bin mode: {bin_mode}")


def _metric_short(ret_col: str) -> str:
    # ret_close_Y -> close
    return ret_col.replace("ret_", "").replace("_Y", "")


# ============================================================
# Build one video set
# ============================================================
def build_one_video_set(
    *,
    market_code: str,
    image_dir: Path,
    out_dir: Path,
    start_year: int,
    end_year: int,
    ret_col: str,
    duration_sec: float,
    build_mp4: bool,
    build_avi: bool,
    bin_mode: str,
) -> None:
    if not image_dir.exists():
        print(f"⚠️ Skip [{market_code}] [{bin_mode}] missing image dir: {image_dir}")
        return

    imgs = sorted(image_dir.glob("*.png"))
    if not imgs:
        print(f"⚠️ Skip [{market_code}] [{bin_mode}] no png found: {image_dir}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    metric_short = _metric_short(ret_col)
    base_name = f"{market_code}_year_{metric_short}_{start_year}_{end_year}_{bin_mode}"

    list_path = out_dir / f"_{base_name}_ffmpeg_list.txt"
    _write_ffmpeg_concat_list(imgs, list_path, duration_sec)

    print(f"\n🖼️ [{market_code}] {ret_col} [{bin_mode}]")
    print(f"  image dir : {image_dir}")
    print(f"  image cnt : {len(imgs)}")
    print(f"  first img : {imgs[0].name}")
    print(f"  last img  : {imgs[-1].name}")

    vf_even = "scale=trunc(iw/2)*2:trunc(ih/2)*2"

    if build_mp4:
        out_mp4 = out_dir / f"{base_name}.mp4"
        cmd_mp4 = [
            "ffmpeg", "-y",
            "-hide_banner", "-loglevel", "error",
            "-f", "concat", "-safe", "0",
            "-i", str(list_path),
            "-vf", vf_even,
            "-fps_mode", "vfr",
            "-pix_fmt", "yuv420p",
            "-c:v", "libx264",
            "-crf", "18",
            "-preset", "medium",
            "-movflags", "+faststart",
            str(out_mp4),
        ]
        rc = _run_ffmpeg(cmd_mp4, f"MP4 {market_code} {ret_col} {bin_mode}")
        if rc == 0 and out_mp4.exists():
            print(f"✅ MP4 saved: {out_mp4} ({out_mp4.stat().st_size / 1024 / 1024:.2f} MB)")

    if build_avi:
        out_avi = out_dir / f"{base_name}.avi"
        cmd_avi = [
            "ffmpeg", "-y",
            "-hide_banner", "-loglevel", "error",
            "-f", "concat", "-safe", "0",
            "-i", str(list_path),
            "-vf", vf_even,
            "-fps_mode", "vfr",
            "-c:v", "mjpeg",
            "-q:v", "3",
            "-pix_fmt", "yuvj420p",
            str(out_avi),
        ]
        rc = _run_ffmpeg(cmd_avi, f"AVI {market_code} {ret_col} {bin_mode}")
        if rc == 0 and out_avi.exists():
            print(f"✅ AVI saved: {out_avi} ({out_avi.stat().st_size / 1024 / 1024:.2f} MB)")


# ============================================================
# Main
# ============================================================
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--markets",
        default="",
        help="Comma-separated market codes, e.g. tw,jp,th. Empty = all enabled markets",
    )
    ap.add_argument("--start-year", type=int, default=2020)
    ap.add_argument("--end-year", type=int, default=2025)
    ap.add_argument(
        "--ret-cols",
        default="ret_close_Y,ret_high_Y,ret_low_Y",
        help="Comma-separated return columns, e.g. ret_close_Y,ret_high_Y,ret_low_Y",
    )
    ap.add_argument(
        "--bin-mode",
        default="both",
        help="10pct / 100pct / both",
    )
    ap.add_argument(
        "--duration-sec",
        type=float,
        default=0.25,
        help="Per-image duration in seconds. 0.25 => about 4 images/sec",
    )
    ap.add_argument("--mp4", action="store_true", help="Build mp4 only")
    ap.add_argument("--avi", action="store_true", help="Build avi only")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    _require_ffmpeg()

    if args.markets.strip():
        markets = [x.strip().lower() for x in args.markets.split(",") if x.strip()]
    else:
        markets = _load_enabled_markets()

    ret_cols = [x.strip() for x in args.ret_cols.split(",") if x.strip()]
    if not ret_cols:
        raise RuntimeError("No ret-cols specified.")

    bin_mode = _normalize_bin_mode(args.bin_mode)
    if bin_mode == "both":
        bin_modes = ["10pct", "100pct"]
    else:
        bin_modes = [bin_mode]

    build_mp4 = True
    build_avi = True
    if args.mp4 and not args.avi:
        build_mp4 = True
        build_avi = False
    elif args.avi and not args.mp4:
        build_mp4 = False
        build_avi = True

    print("Markets:", markets)
    print("Ret cols:", ret_cols)
    print("Years:", args.start_year, "-", args.end_year)
    print("Bin modes:", bin_modes)
    print("Duration per image:", args.duration_sec, "sec")
    print("Build MP4:", build_mp4)
    print("Build AVI:", build_avi)

    for market_code in markets:
        for current_bin_mode in bin_modes:
            image_root = _image_root_by_bin_mode(current_bin_mode)
            video_root = _video_root_by_bin_mode(current_bin_mode)

            for ret_col in ret_cols:
                image_dir = (
                    image_root
                    / market_code
                    / f"{args.start_year}_{args.end_year}_{ret_col}"
                )

                out_dir = (
                    video_root
                    / market_code
                    / f"{args.start_year}_{args.end_year}"
                )

                try:
                    build_one_video_set(
                        market_code=market_code,
                        image_dir=image_dir,
                        out_dir=out_dir,
                        start_year=args.start_year,
                        end_year=args.end_year,
                        ret_col=ret_col,
                        duration_sec=args.duration_sec,
                        build_mp4=build_mp4,
                        build_avi=build_avi,
                        bin_mode=current_bin_mode,
                    )
                except Exception as e:
                    print(f"❌ [{market_code}] [{ret_col}] [{current_bin_mode}] {e}")

    print("\n🎉 All year video jobs finished.")


if __name__ == "__main__":
    main()