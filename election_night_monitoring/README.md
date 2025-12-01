# Election Night Monitoring System

Real-time election monitoring tools for Saint Lucia elections with turnout-based prediction modeling.

## Overview

This system monitors live election results, calculates swings against baseline thresholds, and predicts constituency flips using a turnout-based projection model.

## Components

### Core Scripts

| Script | Description |
|--------|-------------|
| `orchestrator.py` | **Main entry point** - coordinates all monitoring with multi-source redundancy |
| `scrape_live_results.py` | Scrapes live results from the Electoral Department website |
| `scrape_results_slu.py` | Scrapes from results.sluelectoral.com/summary.php (PRIMARY source) |
| `scrape_granular_results.py` | Scrapes individual district pages (FALLBACK source) |
| `generate_swing_charts.py` | Generates swing analysis charts for all constituencies |
| `monitor_live_swings.py` | Console-based live monitoring with turnout projections |
| `turnout_model.py` | Core turnout and flip prediction logic |
| `data_reconciler.py` | Validates and reconciles data from multiple sources |
| `alerts.py` | Desktop/sound notifications on state changes |
| `test_backtest.py` | Backtesting framework using historical data |

### Data Files

- `data/swing_thresholds/swing_thresholds_2021.json` - 2021 baseline thresholds
- `data/swing_thresholds/swing_thresholds_2016.json` - 2016 baseline (for backtesting)
- `data/live_election_results/` - Live scraped results
- `data/granular_snapshots/` - Timestamped snapshots with ballot box counts

## Turnout Model Design Decisions

The turnout model makes several key design decisions documented below:

### 1. Turnout Baseline: Vote Counts vs 2021

**Decision**: Use current votes as percentage of 2021 votes (not electors)

**Rationale**: Per-division elector counts are not available in the election night data. Only constituency-level elector totals exist. Using vote counts provides a consistent, available metric.

**Implementation**: `vote_completion.pct_of_2021` = (current_votes / 2021_votes) * 100

### 2. Projection Threshold: Always Project

**Decision**: Always show projections regardless of reporting percentage, with confidence warnings

**Rationale**: Partial information is better than no information. Early projections with appropriate warnings allow users to track trends as results come in.

**Confidence Levels**:
- `HIGH`: >= 75% of baseline votes reported
- `MEDIUM`: >= 50% reported
- `LOW`: >= 25% reported
- `VERY_LOW`: < 25% reported

### 3. Swing Weighting: Weight by Votes

**Decision**: Calculate weighted average swing where larger divisions have more influence

**Rationale**: A 10% swing in a division with 1,000 votes is more significant than the same swing in a division with 100 votes. Weighting ensures the overall swing reflects the actual vote distribution.

**Formula**: `weighted_swing = sum(swing_pct * division_votes) / sum(division_votes)`

### 4. Early Bias Handling: Show Warning Only

**Decision**: Display warnings about early reporting bias but still show projections

**Rationale**: Early-reporting areas may be systematically different (urban vs rural, party strongholds first). Rather than attempting complex bias correction, we transparently warn users.

**Warning text**: Confidence level indicator + specific warning message

### 5. Flip Thresholds: Dynamic SURE_FLIP, Fixed WATCH

**Decision**:
- `WATCH` threshold: Fixed at 0.5%
- `SURE_FLIP` threshold: Dynamic based on % of votes reported

**Dynamic SURE_FLIP Threshold**:
The threshold decreases as more votes are reported (higher confidence):

| % Reported | SURE_FLIP Threshold |
|------------|---------------------|
| 0-50%      | 1.25% (fixed)       |
| 60%        | 0.91%               |
| 75%        | 0.56%               |
| 100%       | 0.25%               |

**Formula**: For % > 50: `threshold = 1.25 * e^(-0.0322 * (pct - 50))`

**% Reported Calculation**: Uses `current_votes / 2021_baseline_votes * 100` from the "Same as 2021" scenario as the stable baseline denominator.

**Classification Rules**:
- `SURE_FLIP`: Challenger leads by >= threshold in BOTH turnout scenarios
- `WATCH`: Either scenario has margin <= 0.5%, OR scenarios disagree on winner
- `SAFE`: Incumbent leads comfortably in at least one scenario

**Rationale**: Early in the night, projections are less reliable, so we require a larger margin to call a flip. As more votes come in, we can be confident with smaller margins. The 0.25% minimum at 100% reporting allows calling very close races that show consistent challenger leads.

## Two Turnout Scenarios

The model projects final results under two different turnout assumptions:

### Scenario 1: Same as 2021

**Assumption**: Unreported polling divisions will have the same vote counts as in 2021

**Calculation**:
- Reported divisions: Use actual votes
- Unreported divisions: Use exact 2021 vote counts
- Apply weighted swing from reported divisions

**When this is useful**: Baseline scenario, assumes nothing has changed in unreported areas

### Scenario 2: Trend-Adjusted

**Assumption**: Unreported divisions will follow the turnout trend observed in reported divisions

**Calculation**:
1. Calculate turnout change % in reported divisions: `(reported_current - reported_2021) / reported_2021 * 100`
2. Apply this change to unreported divisions: `unreported_projected = unreported_2021 * (1 + change_pct/100)`
3. Apply weighted swing from reported divisions

**When this is useful**: Captures differential turnout patterns (e.g., if turnout is down overall, expect it to be down in unreported areas too)

## Usage

### Quick Start (Election Night)

```bash
# Run the orchestrator - handles everything automatically
python election_night_monitoring/orchestrator.py --interval 300

# With all options
python election_night_monitoring/orchestrator.py \
    --interval 300 \
    --thresholds data/swing_thresholds/swing_thresholds_2021.json
```

The orchestrator:
1. Tries PRIMARY source (results.sluelectoral.com/summary.php)
2. Falls back to SECONDARY source (sluelectoral.com/election-night-results-2026)
3. Falls back to GRANULAR source (individual district.php pages)
4. Saves granular snapshots for archive
5. Generates swing charts automatically
6. Runs swing analysis on new data
7. Sends desktop/sound alerts on state changes

Options:
- `--interval N`: Seconds between scrapes (default: 300 = 5 minutes)
- `--no-alerts`: Disable desktop/sound notifications
- `--no-granular`: Skip granular snapshot saving
- `--test`: Run once and exit (for testing)

### Manual Components (if needed)

```bash
# Run scraper only
python election_night_monitoring/scrape_live_results.py --interval 60

# Run monitor only (on existing data)
python election_night_monitoring/monitor_live_swings.py \
    --live-results data/live_election_results/results_LATEST.json \
    --thresholds data/swing_thresholds/swing_thresholds_2021.json

# Generate charts only
python election_night_monitoring/generate_swing_charts.py \
    --results data/live_election_results/results_LATEST.json \
    --thresholds data/swing_thresholds/swing_thresholds_2021.json \
    --output-dir analysis/live_2025/
```

### Backtesting

```bash
# Run backtest using 2021 data against 2016 baseline
python election_night_monitoring/test_backtest.py

# With verbose output
python election_night_monitoring/test_backtest.py --verbose

# Specify output directory
python election_night_monitoring/test_backtest.py --output-dir analysis/backtest
```

### Generating New Thresholds

```bash
# Generate thresholds from a baseline year
python generate_swing_thresholds.py --baseline-year 2021
```

## Output Formats

### Console Output

```
================================================================================
SAINT LUCIA ELECTION SWING MONITOR (with Turnout Model)
Baseline: 2021 | Updated: 2025-12-01 18:30:00
================================================================================

PROJECTED SEATS (by Turnout Scenario):
  Same as 2021 Turnout:  SLP 12  |  UWP 3  |  Other 2
  Trend-Adjusted:        SLP 11  |  UWP 4  |  Other 2

FLIP STATUS SUMMARY (Turnout Model)
----------------------------------------
  SURE FLIP:    2 constituencies
                [BABONNEAU, SOUFRIERE]
  WATCH:        3 constituencies
                [CASTRIES SOUTH EAST, DENNERY NORTH, CHOISEUL]
  SAFE:         10 constituencies
  NOT REPORTED: 2 constituencies
```

### Chart Output

Each constituency chart displays:
- Horizontal bars showing swing per polling division
- Enhanced header with:
  - Vote progress (current vs 2021)
  - Current vote share by party
  - Projections under both scenarios
  - Flip status and confidence level

### JSON Analysis Output

The `swing_analysis.json` file contains complete analysis data including:
- Per-constituency turnout analysis
- Division-level swings
- Scenario projections
- Flip classifications

## Data Flow

```
[Electoral Dept Website]
         |
         v
[scrape_live_results.py]
         |
         v
[data/live_election_results/results_TIMESTAMP.json]
         |
    +----+----+
    |         |
    v         v
[monitor_live_swings.py]  [generate_swing_charts.py]
    |                              |
    v                              v
[Console output]           [PNG charts + JSON]
```

## Thresholds File Structure

```json
{
  "baseline_year": "2021",
  "constituencies": {
    "CONSTITUENCY_NAME": {
      "status": "contested",
      "winner": {"party": "SLP", "votes": 3500, "pct": 55.5},
      "margin": {"votes": 800, "pct": 12.0},
      "breakeven_swing_pct": 6.0,
      "all_districts": {
        "A1(a)": {
          "baseline_total_votes": 1200,
          "winner_party_pct": 58.2,
          "pct_of_constituency": 15.5
        }
      },
      "meaningful_districts": ["A1(a)", "A2(b)"]
    }
  }
}
```

## Testing

The backtesting framework validates the model by:

1. Using 2016 as baseline, 2021 as "live" results
2. Generating 4 progressive snapshots (25%, 50%, 75%, 100% of divisions)
3. Ensuring snapshots are consistent (votes only increase, divisions don't disappear)
4. Comparing predictions at each stage to actual 2021 outcomes

### Backtest Metrics

- **Accuracy by %**: Correct winner predictions at each reporting percentage
- **Flip Detection**: What percentage of actual flips were flagged as SURE_FLIP or WATCH
- **Final Accuracy**: Overall accuracy at 100% reporting

## Limitations

1. **No per-division electors**: True turnout % cannot be calculated; we use vote completion instead
2. **Early bias**: Early-reporting divisions may not be representative
3. **Third parties/Independents**: Model is optimized for SLP vs UWP two-party contests
4. **Historical patterns**: Model assumes swing patterns will be similar to past elections

## Backtest Results (2016 → 2021)

Using 2016 as baseline and 2021 as test data:

| % Reported | Accuracy | Flip Detection |
|------------|----------|----------------|
| 25%        | 88.2%    | 88.9%          |
| 50%        | 100.0%   | 100.0%         |
| 75%        | 100.0%   | 100.0%         |
| 100%       | 100.0%   | 100.0%         |

All 9 actual flips are detected by 50% reporting.

## Version History

- **v1.1** (Dec 1, 2025): Bug fixes and margin display
  - **Bug fix: Swing sign inversion** - Fixed incorrect swing application in `project_final_margin()`. When swing was negative (incumbent losing vote share), the code incorrectly added to incumbent votes instead of subtracting. This caused wrong predictions for flips with large margins (16-40%).
  - **Bug fix: Challenger selection** - Fixed challenger party selection when incumbent loses. Previously picked first non-incumbent party in dict iteration order; now correctly selects the challenger with the highest current vote share.
  - **Enhancement: Margin display** - All outputs (console, charts, JSON) now show projected margins for both scenarios to enable manual verification.
  - **Enhancement: Dynamic SURE_FLIP threshold** - Threshold decays from 1.25% to 0.25% as reporting increases above 50%.

- **v1.0** (Dec 2025): Initial release with turnout model integration
  - Two-scenario projections
  - SURE_FLIP / WATCH / SAFE classification
  - Confidence warnings
  - Backtesting framework
