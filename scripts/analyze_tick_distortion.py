# scripts/analyze_tick_distortion.py
from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

DATA = ROOT / "data" / "derived" / "weekK"

MARKETS = [
    "us",
    "ca",
    "uk",
    "hk",
    "au",
    "jp",
    "kr",
    "tw",
    "cn",
    "th",
    "fr",
]

def load_market(market):

    mdir = DATA / market

    ratios = []

    for f in mdir.glob("*.csv"):

        try:

            df = pd.read_csv(f)

            if "prev_close_W" not in df.columns:
                continue

            prev = df["prev_close_W"].dropna()

            if len(prev) == 0:
                continue

            # 假設 tick = 0.01
            tick = 0.01

            pct = tick / prev

            ratios.extend(pct.values)

        except Exception:
            continue

    return pd.Series(ratios)


def main():

    all_data = {}

    for m in MARKETS:

        s = load_market(m)

        if len(s) == 0:
            continue

        all_data[m.upper()] = s

        print(m, "samples:", len(s))

    df = pd.DataFrame(dict([(k, pd.Series(v)) for k,v in all_data.items()]))

    plt.figure(figsize=(12,6))

    df.boxplot(showfliers=False)

    plt.axhline(0.1, linestyle="--")

    plt.ylabel("One Tick / Price")
    plt.title("Tick Distortion Distribution Across Markets")

    out = ROOT / "tick_distortion_markets.png"

    plt.savefig(out, dpi=200)

    print("Saved:", out)


if __name__ == "__main__":
    main()