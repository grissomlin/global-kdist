# Global K-Distribution (global-kdist)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-success)
![Markets](https://img.shields.io/badge/Markets-12-orange)

A data-driven research project analyzing the **extreme right-tail distribution of stock returns across global equity markets**.

This project collects daily stock data, resamples it into **weekly, monthly, and yearly bars**, and studies the statistical distribution of large price movements such as **10%, 50%, 100%, 500%, and 1000%+ moves** across multiple stock markets.

The goal is to understand how often **extreme winners, outliers, and multi-bagger stocks** appear in different countries, and to build a reusable research pipeline for cross-market comparison.

---

## Research Motivation

Financial markets do not follow a neat normal distribution. A small number of stocks often generate an outsized share of long-term returns, while most names produce much smaller outcomes.

Instead of focusing only on average returns, this project focuses on **extreme right-tail events**, such as:

- Weekly +10% movers
- Monthly +50% / +100% movers
- Yearly 5x / 10x / 20x multi-baggers

Key questions include:

- Which markets produce the most extreme winners?
- How often do monthly returns exceed 100%?
- How common are 10x stocks globally?
- Are right-tail events concentrated in certain markets?
- How different are weekly, monthly, and yearly tail distributions?

---

## Markets Covered

The system currently supports 12 stock markets:

| Market | Exchange |
|------|------|
| US | NYSE / NASDAQ |
| CA | TSX / TSXV |
| UK | LSE |
| AU | ASX |
| HK | HKEX |
| JP | Tokyo |
| KR | KOSPI / KOSDAQ |
| TW | Taiwan |
| CN | Shanghai / Shenzhen |
| TH | Thailand |
| FR | Euronext Paris |
| IN | NSE / BSE |

More markets can be added through the same modular pipeline design.

---

## System Architecture

The project is structured as a reusable **data research pipeline**:

```text
Raw Market Universe
        ↓
Daily OHLCV Download
        ↓
Data Cleaning / Standardization
        ↓
Weekly / Monthly / Yearly Resampling
        ↓
Return Calculation
        ↓
Extreme Move Detection
        ↓
Statistical Aggregation
        ↓
Chart Rendering
        ↓
Research Output / Articles / Shorts
```

This design allows the same workflow to be reused across countries with market-specific modules.

---

## Core Features

- Multi-market stock universe support
- Automated daily OHLCV download
- Daily to weekly / monthly / yearly resampling
- Extreme return distribution analysis
- Right-tail and outlier frequency measurement
- Cross-market comparison
- Chart generation for research content
- CSV / PNG research output export
- Market-specific modular architecture

---

## Research Focus

This repository is built around one central idea:

> **Stock returns are heavily skewed, and the right tail matters more than most investors realize.**

A few extreme winners can dominate total market wealth creation.  
This project is designed to quantify that phenomenon across different countries and timeframes.

Typical use cases include:

- studying the frequency of 10x stocks
- comparing monthly 100%+ movers across countries
- analyzing yearly extreme return distributions
- building data-driven market commentary
- generating charts for research articles and short-form content

---

## Example Research Outputs

The system can generate outputs such as:

- monthly tail summary tables
- monthly tail incidence charts
- percentile charts
- yearly return distribution charts
- extreme move bin statistics
- article highlight datasets
- market-by-market comparison visuals

Typical generated files include:

```text
data/research/monthly_tail_charts/monthly_tail_summary_2020_2025.csv
data/research/monthly_tail_charts/monthly_tail_article_highlights_2020_2025.csv
data/research/monthly_tail_charts/monthly_tail_percentiles_2020_2025.png
data/research/monthly_tail_charts/monthly_tail_incidence_2020_2025.png
data/research/yearly_tail_charts/yearly_tail_summary_2020_2025.csv
data/research/yearly_tail_charts/year_return_distribution.png
```

These outputs help visualize how fat the right tail of stock returns can be across different markets.

---

## Repository Structure

```text
global-kdist/
├── configs/
│   └── Market configuration files
├── markets/
│   └── Market-specific modules
├── scripts/
│   └── Core download / resample / analysis / rendering scripts
├── data/
│   └── Local datasets, caches, derived outputs, research results
├── README.md
└── requirements.txt
```

A typical practical interpretation is:

- `configs/` → market setup and runtime config
- `markets/` → country-specific loaders / rules / universes
- `scripts/` → pipeline entry points
- `data/` → generated files and local research outputs

---

## Installation

Clone the repository:

```bash
git clone https://github.com/grissomlin/global-kdist.git
cd global-kdist
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Basic Workflow

### 1. Download market data

```bash
python -m scripts.run --all --start 2020-01-01
```

This downloads daily OHLCV data for all enabled markets.

### 2. Resample daily data into higher timeframes

```bash
python -m scripts.resample_only --overwrite
```

This generates:

- weekly bars
- monthly bars
- yearly bars

### 3. Run monthly extreme-move analysis

```bash
python -m scripts.analyze_monthly_tail_markets
```

### 4. Run yearly extreme-move analysis

```bash
python -m scripts.analyze_yearly_tail_markets
```

### 5. Render charts

```bash
python -m scripts.render_year_return_distribution
```

Depending on the script, outputs are typically saved under:

```text
data/research/
```

---

## Example Research Questions

This dataset and pipeline can be used to study questions such as:

- Which market produces the highest rate of 100% monthly movers?
- Which countries have the fattest yearly right tail?
- How often do 10x stocks appear in each market?
- Are emerging markets more extreme than developed markets?
- Does the right tail change over time?
- Which timeframe best captures asymmetric upside?

---

## Data Philosophy

This project is focused on **distribution analysis**, not price prediction.

The objective is not to forecast which single stock will become a 10x winner.  
Instead, the objective is to understand:

- how often extreme winners occur
- how they are distributed
- which markets generate them more frequently
- how their frequency changes by timeframe

This makes the project useful for:

- asymmetric payoff research
- momentum and speculative market studies
- venture-style public equity thinking
- cross-market comparative analysis
- content creation backed by data

---

## Roadmap

Planned future improvements include:

- more market coverage
- sector-level tail analysis
- market-cap normalization
- liquidity-based filtering
- intraday extreme move analysis
- dashboard / web visualization
- automated article generation support
- richer chart templates for research publishing

---

## README Visual Section

You can place research charts directly inside this README after uploading image files to the repository.

For example:

```markdown
## Sample Chart

![Monthly Tail Percentiles](docs/images/monthly_tail_percentiles.png)

## Sample Chart

![Yearly Return Distribution](docs/images/year_return_distribution.png)
```

Recommended structure for README images:

```text
docs/
└── images/
    ├── monthly_tail_percentiles.png
    ├── monthly_tail_incidence.png
    └── year_return_distribution.png
```

Once those files are committed, GitHub will render them directly in the README.

---

---

## Included Research Reports

This repository already includes exported market-level statistics and report files that can be used directly for downstream analysis or AI-assisted interpretation.

Available report paths:

- `data/reports/market_reports/`
- `data/reports/market_reports_csv/`

GitHub links:

- https://github.com/grissomlin/global-kdist/tree/main/data/reports/market_reports
- https://github.com/grissomlin/global-kdist/tree/main/data/reports/market_reports_csv

These files are intended to make the repository more immediately usable, especially for:

- feeding structured market statistics into AI tools
- generating article drafts
- building market summaries
- comparing countries without re-running the full pipeline first

In practice, this means users can often work directly from the exported report files instead of regenerating everything from scratch.

---

## Data Availability Note

Due to GitHub file size and repository size limits, this repository does **not** include the full set of raw and resampled OHLCV datasets for every market and timeframe.

In particular, the complete daily / weekly / monthly / yearly K-bar datasets for all countries are too large to be fully uploaded to GitHub.

Therefore, the repository follows this approach:

- keep the **code pipeline** in GitHub
- keep **derived report outputs** in GitHub where practical
- regenerate larger raw datasets locally when needed

If a user wants to reproduce the full dataset, they can run the code and gradually download market data through the project scripts.

Typical workflow:

```text
Code in repository
        +
Exported report files in data/reports/
        +
Local data download / resampling when full reproduction is needed
```

This design keeps the repository lightweight enough for GitHub while preserving reproducibility through code.


## Suggested `.gitignore`

For this project, generated data and cache folders usually should not be committed.

Example:

```gitignore
data/cache_dayk/
data/cache_weekk/
data/cache_monthk/
data/cache_yeark/
data/derived/
data/reports/
__pycache__/
*.pyc
*.pyo
.DS_Store
Thumbs.db
```

---

## Suggested Professional Repo Layout

A cleaner long-term layout could look like this:

```text
global-kdist/
├── configs/
├── markets/
├── scripts/
├── docs/
│   └── images/
├── data/
│   └── .gitignore
├── requirements.txt
├── README.md
└── LICENSE
```

This keeps the repository presentation cleaner while allowing local data generation.

---

## License

MIT License

---

## Author

Created by **Grissom Lin**

GitHub: https://github.com/grissomlin

This repository is part of an ongoing research effort into:

- global stock return distributions
- extreme market events
- multi-bagger frequency analysis
- data-driven market research pipelines
