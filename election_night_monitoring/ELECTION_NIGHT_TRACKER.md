# Election Night Flip Tracker

## How the Turnout Model Works

### Two Projection Scenarios

The model projects final results under two turnout assumptions:

| Scenario | Assumption | Use Case |
|----------|------------|----------|
| **Same as 2021** | Unreported divisions will match their 2021 vote counts | Baseline - assumes nothing changed |
| **Trend-Adjusted** | Unreported divisions follow the turnout trend seen in reported areas | Captures differential turnout (e.g., if turnout is down 10% in reported areas, apply that to unreported) |

### Flip Classification

| Status | Meaning | Threshold |
|--------|---------|-----------|
| **SURE_FLIP** | Challenger leads in BOTH scenarios by >= threshold | Dynamic: 1.25% at <50% → 0.25% at 100% |
| **WATCH** | Either scenario within 0.5%, OR scenarios disagree | Fixed: 0.5% |
| **SAFE** | Incumbent leads comfortably | - |

### Dynamic SURE_FLIP Threshold

As more votes are reported, less margin is needed to call a flip:

| % Reported | Threshold |
|------------|-----------|
| 0-50% | 1.25% |
| 60% | 0.91% |
| 75% | 0.56% |
| 100% | 0.25% |

### Backtest Results (2016 → 2021)

- **100% accuracy** at 50%+ reporting
- **100% flip detection** at 50%+ reporting
- Once a flip is predicted at 50%+, it never reverts

---

## Live Tracking Table

Update this table as results come in. All 17 constituencies start as the 2021 incumbent.

### Flip Watch List

| Constituency | 2021 Winner | % Reported | Status | S1 Projection | S2 Projection | Notes |
|--------------|-------------|------------|--------|---------------|---------------|-------|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

### All Constituencies

| Constituency | 2021 Winner | % Reported | Status | Projected Winner | Margin |
|--------------|-------------|------------|--------|------------------|--------|
| Anse La Raye/Canaries | SLP | | | | |
| Babonneau | SLP | | | | |
| Castries Central | IND | | | | |
| Castries East | SLP | | | | |
| Castries North | IND | | | | |
| Castries South | SLP | | | | |
| Castries South East | SLP | | | | |
| Choiseul | UWP | | | | |
| Dennery North | SLP | | | | |
| Dennery South | SLP | | | | |
| Gros Islet | SLP | | | | |
| Laborie | SLP | | | | |
| Micoud North | SLP | | | | |
| Micoud South | UWP | | | | |
| Soufriere | SLP | | | | |
| Vieux Fort North | SLP | | | | |
| Vieux Fort South | SLP | | | | |

### Seat Count Projection

| Party | 2021 Result | Same as 2021 | Trend-Adjusted |
|-------|-------------|--------------|----------------|
| SLP | 13 | | |
| UWP | 2 | | |
| IND | 2 | | |

---

## Quick Reference

**Confidence Levels:**
- HIGH: 75%+ reported
- MEDIUM: 50-74% reported
- LOW: 25-49% reported
- VERY_LOW: <25% reported

**Key Insight:** At 50%+ reporting, predictions are stable. Early predictions (25%) may change as data comes in.

**Running the Monitor:**
```bash
python election_night_monitoring/orchestrator.py --interval 300
```
