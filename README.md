# Saint Lucia Election Analysis

Historical analysis of Saint Lucia elections since independence, with the goal of writing a short article with analysis and predictions for the upcoming December 1, 2025 election.

## Project Structure

```
saint_lucia_election/
├── data/                              # Election data JSON files
│   ├── summary_results/               # Summary results by party
│   ├── vote_distribution/             # Vote distribution by constituency
│   ├── constituency_maps/             # Constituency mapping data
│   ├── election_night_results/        # Historical election night results
│   ├── swing_thresholds/              # Baseline thresholds for flip detection
│   ├── live_election_results/         # Live scraped results (gitignored)
│   └── granular_snapshots/            # Ballot box level snapshots (gitignored)
│
├── election_night_monitoring/         # Real-time election monitoring system
│   ├── orchestrator.py                # Main entry point - coordinates all monitoring
│   ├── turnout_model.py               # Turnout-based flip prediction model
│   ├── monitor_live_swings.py         # Console-based live monitoring
│   ├── generate_swing_charts.py       # Swing analysis chart generation
│   ├── scrape_live_results.py         # Live results scraper
│   ├── scrape_results_slu.py          # Primary source scraper
│   ├── scrape_granular_results.py     # Granular/fallback scraper
│   ├── alerts.py                      # Desktop/sound notifications
│   ├── data_reconciler.py             # Multi-source data validation
│   ├── test_backtest.py               # Backtesting framework
│   └── README.md                      # Detailed monitoring documentation
│
├── scrapers/                          # Historical data scrapers
│   └── scrape_election_summary_results.py
│
├── analysis/                          # Generated analysis output (gitignored)
├── main.py                            # Main historical analysis script
├── swing_utils.py                     # Shared swing calculation utilities
├── generate_swing_thresholds.py       # Generate baseline thresholds
├── scrape_constituency_map.py         # Constituency mapping scraper
└── pyproject.toml                     # Project configuration
```

## Setup

This project uses `uv` for Python package management. The virtual environment is managed via `uv pip`.

### Activate the virtual environment

```bash
source .venv/bin/activate
```

### Install dependencies

Dependencies are already installed. If you need to reinstall:

```bash
uv pip install -e .
```

## Data

The `data/` folder contains election results in JSON format:
- **Summary results**: Party-level totals, vote percentages, seats won
- **Vote distribution**: Constituency-level vote breakdowns
- **Constituency maps**: Mapping of constituencies to polling divisions, communities, and polling station locations
- **Election night results**: Granular polling division-level vote breakdowns from election night results pages

Elections covered:
- 1979, 1982, 1987 (two elections: April 30 & June 6), 1992, 1997, 2006, 2011, 2016, 2021

**Note:** There were two separate elections in 1987 (April 30 and June 6), which are treated as distinct elections in the analysis.

**Party Name Normalization:** "Saint Lucia Labour Party" and "St. Lucia Labour Party" are normalized and combined into a single entity ("Saint Lucia Labour Party") for all reporting purposes, as they refer to the same political party.

## Analysis Goals

1. Historical trends in vote share and seat distribution
2. Constituency-level patterns and swing analysis
3. Predictions for the December 1, 2025 election

## Usage

**Important:** Always activate the virtual environment before running scripts:

```bash
source .venv/bin/activate
```

Then run the main analysis script:

```bash
python main.py
```

Alternatively, use the helper script (automatically activates the virtual environment):

```bash
./run.sh
```

Or use Jupyter notebooks for interactive analysis:

```bash
jupyter notebook
```

## Constituency Mapping

The `scrape_constituency_map.py` script extracts constituency mapping data from the Saint Lucia Electoral Department website. This data includes:

- Constituency names
- Polling divisions within each constituency
- Communities served by each polling division
- Polling station addresses

### Running the Scraper

```bash
python scrape_constituency_map.py
```

The script will:
1. Scrape constituency information from the official electoral website
2. Parse polling division, community, and polling station data
3. Save the results to `data/constituency_maps/saint_lucia_constituencies.json`

**Note:** The script includes a 1-second delay between requests to be courteous to the server. If a constituency page is unavailable (404 error), it will be skipped with a warning message.

## Election Night Results

The `scrape_historical_election_night_results.py` script extracts election night results with granular polling division-level vote breakdowns from the Saint Lucia Electoral Department website. This data provides more detailed information than summary results, showing votes by polling division.

### Running the Scraper

```bash
python scrape_historical_election_night_results.py
```

The script will:
1. Scrape election night results pages for 2011, 2016, and 2021
2. Extract table data containing polling division-level vote breakdowns
3. Save the results to `data/election_night_results/election_results_{year}.json`

**Note:** These results contain polling division-level data, which is more granular than constituency-level summaries.

## Election Night Monitoring

The `election_night_monitoring/` folder contains a complete real-time monitoring system for election night with turnout-based flip prediction.

### Quick Start (Election Night)

```bash
source .venv/bin/activate
python election_night_monitoring/orchestrator.py --interval 300
```

The orchestrator automatically:
1. Scrapes from multiple sources with fallback (PRIMARY → SECONDARY → GRANULAR)
2. Generates swing analysis charts
3. Runs the turnout model with flip predictions
4. Sends desktop/sound alerts on state changes
5. Saves granular snapshots for archive

### Turnout Model

The system uses a turnout-based projection model that:
- Projects final results under two scenarios (Same as 2021 / Trend-adjusted)
- Classifies seats as SURE_FLIP, WATCH, or SAFE
- Uses dynamic thresholds based on % of votes reported
- Achieved 100% accuracy in backtesting at 50%+ reporting

See `election_night_monitoring/README.md` for detailed documentation.

