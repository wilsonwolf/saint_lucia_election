# Saint Lucia Election Analysis

Historical analysis of Saint Lucia elections since independence, with the goal of writing a short article with analysis and predictions for the upcoming December 1, 2025 election.

## Project Structure

```
saint_lucia_election/
├── data/                          # Election data JSON files
│   ├── summary_results/           # Summary results by party
│   │   └── *_summary_results.json
│   ├── vote_distribution/         # Vote distribution by constituency
│   │   └── *_vote_distribution.json
│   ├── constituency_maps/         # Constituency mapping data
│   │   └── saint_lucia_constituencies.json
│   └── election_night_results/    # Election night results with polling division breakdowns
│       └── election_results_*.json
├── analysis/                      # Analysis scripts and notebooks
├── .venv/                        # Virtual environment (managed by uv)
├── pyproject.toml                # Project configuration
├── main.py                       # Main analysis script
└── scrape_constituency_map.py    # Script to scrape constituency maps
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

