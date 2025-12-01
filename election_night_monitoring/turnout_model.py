#!/usr/bin/env python3
"""
Turnout Model for Election Night Monitoring

This module provides turnout-based predictions and flip classification
for the Saint Lucia election monitoring system.

Key Design Decisions:
- Turnout baseline: Compare current votes to 2021 votes (not electors, as per-division
  elector counts are not available)
- Always project: Show projections even with low reporting %, with confidence warnings
- Swing weighting: Larger divisions have more influence on weighted averages
- Flip thresholds: Fixed 1.25% for "sure flip", 0.5% for "watch"
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from swing_utils import normalize_constituency_name, normalize_polling_division_name


# Flip classification thresholds
WATCH_THRESHOLD = 0.5  # Fixed: Either scenario within 0.5%

# SURE_FLIP threshold parameters (dynamic based on % reported)
# Below 50%: Fixed at 1.25%
# 50% to 100%: Exponential decay from 1.25% to 0.25%
SURE_FLIP_THRESHOLD_MAX = 1.25  # Threshold at <= 50% reported
SURE_FLIP_THRESHOLD_MIN = 0.25  # Threshold at 100% reported
SURE_FLIP_DECAY_START = 50  # Start decaying at this % reported
SURE_FLIP_DECAY_RATE = 0.0322  # Calculated to hit 0.25% at 100%

# Confidence thresholds (% of votes reported)
CONFIDENCE_HIGH = 75
CONFIDENCE_MEDIUM = 50
CONFIDENCE_LOW = 25


def get_sure_flip_threshold(pct_reported: float) -> float:
    """
    Calculate dynamic SURE_FLIP threshold based on % of votes reported.

    The threshold decreases as more votes are reported, reflecting increased
    confidence in the projection:
    - Below 50%: Fixed at 1.25% (high bar, low confidence)
    - 50% to 100%: Exponential decay from 1.25% to 0.25%

    Uses "Same as 2021" scenario for the denominator (stable baseline).

    Args:
        pct_reported: Percentage of baseline (2021) votes that have been reported

    Returns:
        SURE_FLIP threshold in percentage points
    """
    import math

    if pct_reported <= SURE_FLIP_DECAY_START:
        return SURE_FLIP_THRESHOLD_MAX

    # Exponential decay: threshold = max * e^(-k * (pct - start))
    decay_amount = pct_reported - SURE_FLIP_DECAY_START
    threshold = SURE_FLIP_THRESHOLD_MAX * math.exp(-SURE_FLIP_DECAY_RATE * decay_amount)

    # Don't go below minimum
    return max(threshold, SURE_FLIP_THRESHOLD_MIN)


def calculate_vote_completion(
    live_division_totals: Dict[str, int],
    baseline_data: Dict,
) -> Dict:
    """
    Calculate vote completion (current votes as % of 2021 baseline).

    Args:
        live_division_totals: Dict of {division_code: current_total_votes}
        baseline_data: Constituency data from swing_thresholds with all_districts

    Returns:
        {
            "current_votes": int,
            "baseline_2021_votes": int,
            "pct_of_2021": float,
            "per_division": {
                "A1(a)": {"current": 823, "baseline": 1833, "pct": 44.9, "reported": True},
                ...
            },
            "reported_divisions": int,
            "total_divisions": int,
            "pct_divisions_reported": float
        }
    """
    all_districts = baseline_data.get("all_districts", {})

    per_division = {}
    current_votes = 0
    baseline_votes = 0
    reported_count = 0

    for div_code, div_baseline in all_districts.items():
        baseline_total = div_baseline.get("baseline_total_votes", 0)
        baseline_votes += baseline_total

        # Normalize division name for matching
        normalized_div = normalize_polling_division_name(div_code)

        # Try to find matching live data
        live_total = 0
        reported = False
        for live_div, live_votes in live_division_totals.items():
            if normalize_polling_division_name(live_div) == normalized_div:
                live_total = live_votes
                reported = live_total > 0
                break

        if reported:
            reported_count += 1
            current_votes += live_total

        pct = (live_total / baseline_total * 100) if baseline_total > 0 else 0.0

        per_division[div_code] = {
            "current": live_total,
            "baseline": baseline_total,
            "pct": round(pct, 2),
            "reported": reported,
        }

    total_divisions = len(all_districts)
    pct_of_2021 = (current_votes / baseline_votes * 100) if baseline_votes > 0 else 0.0
    pct_divisions_reported = (reported_count / total_divisions * 100) if total_divisions > 0 else 0.0

    return {
        "current_votes": current_votes,
        "baseline_2021_votes": baseline_votes,
        "pct_of_2021": round(pct_of_2021, 2),
        "per_division": per_division,
        "reported_divisions": reported_count,
        "total_divisions": total_divisions,
        "pct_divisions_reported": round(pct_divisions_reported, 2),
    }


def calculate_vote_share(
    candidate_votes: Dict[str, Dict[str, int]],
) -> Dict[str, Dict]:
    """
    Calculate current vote share per candidate/party.

    Args:
        candidate_votes: {
            "SLP": {"candidate": "CASIMIR", "votes": 3150},
            "UWP": {"candidate": "JOHNSON", "votes": 2270},
        }

    Returns:
        {
            "SLP": {"candidate": "CASIMIR", "votes": 3150, "pct": 58.1},
            "UWP": {"candidate": "JOHNSON", "votes": 2270, "pct": 41.9},
            "total_votes": 5420
        }
    """
    total_votes = sum(p.get("votes", 0) for p in candidate_votes.values())

    result = {}
    for party, data in candidate_votes.items():
        votes = data.get("votes", 0)
        pct = (votes / total_votes * 100) if total_votes > 0 else 0.0
        result[party] = {
            "candidate": data.get("candidate", ""),
            "votes": votes,
            "pct": round(pct, 2),
        }

    result["total_votes"] = total_votes
    return result


def calculate_weighted_swing(
    division_swings: List[Dict],
) -> float:
    """
    Calculate weighted average swing from reported divisions.

    Swing is weighted by division vote count (larger divisions count more).

    Args:
        division_swings: [
            {"division": "A1(a)", "swing_pct": 2.3, "votes": 823, "reported": True},
            ...
        ]

    Returns:
        Weighted average swing percentage (positive = toward challenger)
    """
    reported = [d for d in division_swings if d.get("reported", False)]

    if not reported:
        return 0.0

    total_weight = sum(d.get("votes", 0) for d in reported)
    if total_weight == 0:
        return 0.0

    weighted_sum = sum(
        d.get("swing_pct", 0) * d.get("votes", 0)
        for d in reported
    )

    return round(weighted_sum / total_weight, 2)


def generate_turnout_scenarios(
    vote_completion: Dict,
    current_swing_pct: float,
    baseline_data: Dict,
) -> Dict:
    """
    Generate two turnout scenarios for projection.

    Scenario 1 (Same as 2021): Unreported divisions get their 2021 vote counts
    Scenario 2 (Trend-adjusted): Apply reported divisions' turnout change to unreported

    Args:
        vote_completion: Output from calculate_vote_completion()
        current_swing_pct: Current weighted swing from reported divisions
        baseline_data: Constituency data from swing_thresholds

    Returns:
        {
            "same_as_2021": {
                "projected_total_votes": 12171,
                "unreported_votes": 6751,
                "description": "Unreported divisions match 2021 turnout"
            },
            "trend_adjusted": {
                "turnout_change_pct": -5.2,
                "projected_total_votes": 11538,
                "unreported_votes": 6118,
                "description": "Unreported divisions follow -5.2% turnout trend"
            }
        }
    """
    per_division = vote_completion.get("per_division", {})
    current_votes = vote_completion.get("current_votes", 0)

    # Calculate reported and unreported baseline votes
    reported_baseline = 0
    unreported_baseline = 0

    for div_code, div_data in per_division.items():
        baseline = div_data.get("baseline", 0)
        if div_data.get("reported", False):
            reported_baseline += baseline
        else:
            unreported_baseline += baseline

    # Scenario 1: Same as 2021
    # Unreported divisions get their exact 2021 vote counts
    same_as_2021_unreported = unreported_baseline
    same_as_2021_total = current_votes + same_as_2021_unreported

    # Scenario 2: Trend-adjusted
    # Calculate turnout change % in reported divisions
    if reported_baseline > 0:
        reported_current = current_votes
        turnout_change_pct = ((reported_current - reported_baseline) / reported_baseline) * 100
    else:
        turnout_change_pct = 0.0

    # Apply trend to unreported divisions
    trend_adjusted_unreported = unreported_baseline * (1 + turnout_change_pct / 100)
    trend_adjusted_total = current_votes + trend_adjusted_unreported

    return {
        "same_as_2021": {
            "projected_total_votes": int(round(same_as_2021_total)),
            "unreported_votes": int(round(same_as_2021_unreported)),
            "description": "Unreported divisions match 2021 turnout",
        },
        "trend_adjusted": {
            "turnout_change_pct": round(turnout_change_pct, 2),
            "projected_total_votes": int(round(trend_adjusted_total)),
            "unreported_votes": int(round(trend_adjusted_unreported)),
            "description": f"Unreported divisions follow {turnout_change_pct:+.1f}% turnout trend",
        },
    }


def project_final_margin(
    vote_completion: Dict,
    vote_share: Dict,
    scenarios: Dict,
    weighted_swing: float,
    baseline_data: Dict,
) -> Dict:
    """
    Project final margin under each scenario.

    For reported divisions: Use actual votes and swing
    For unreported divisions: Use projected votes and apply weighted swing from reported

    Args:
        vote_completion: Output from calculate_vote_completion()
        vote_share: Output from calculate_vote_share()
        scenarios: Output from generate_turnout_scenarios()
        weighted_swing: Weighted average swing from reported divisions
        baseline_data: Constituency data from swing_thresholds

    Returns:
        {
            "same_as_2021": {
                "projected_total_votes": 12171,
                "projected_margin_votes": 234,
                "projected_margin_pct": 1.92,
                "projected_winner": "SLP",
                "incumbent_final_pct": 50.96,
                "challenger_final_pct": 49.04
            },
            "trend_adjusted": { ... }
        }
    """
    baseline_winner = baseline_data.get("winner", {}).get("party", "")
    baseline_margin_pct = baseline_data.get("margin", {}).get("pct", 0)

    # Get current vote shares
    incumbent_pct = 0
    challenger_pct = 0
    for party, data in vote_share.items():
        if party == "total_votes":
            continue
        if party == baseline_winner:
            incumbent_pct = data.get("pct", 0)
        else:
            # Sum all non-incumbent parties
            challenger_pct += data.get("pct", 0)

    current_votes = vote_completion.get("current_votes", 0)

    results = {}

    for scenario_name, scenario_data in scenarios.items():
        projected_total = scenario_data.get("projected_total_votes", 0)
        unreported_votes = scenario_data.get("unreported_votes", 0)

        if projected_total == 0:
            results[scenario_name] = {
                "projected_total_votes": 0,
                "projected_margin_votes": 0,
                "projected_margin_pct": baseline_margin_pct,
                "projected_winner": baseline_winner,
                "incumbent_final_pct": 50 + baseline_margin_pct / 2,
                "challenger_final_pct": 50 - baseline_margin_pct / 2,
            }
            continue

        # Calculate projected margin
        # Reported divisions: Use actual vote share
        # Unreported divisions: Apply weighted swing to 2021 baseline

        # Current incumbent votes
        incumbent_current_votes = current_votes * incumbent_pct / 100 if incumbent_pct > 0 else 0
        challenger_current_votes = current_votes * challenger_pct / 100 if challenger_pct > 0 else 0

        # For unreported divisions, start with 2021 baseline split
        baseline_total_votes = baseline_data.get("total_votes", 0)
        baseline_incumbent_votes = baseline_data.get("winner", {}).get("votes", 0)
        baseline_challenger_votes = baseline_total_votes - baseline_incumbent_votes

        # Unreported baseline ratio
        if baseline_total_votes > 0:
            unreported_ratio = unreported_votes / baseline_total_votes if scenario_name == "same_as_2021" else (unreported_votes / baseline_total_votes)
            # Actually we should get the unreported divisions specifically
            # For now, use proportion
            unreported_incumbent_baseline = baseline_incumbent_votes * unreported_ratio
            unreported_challenger_baseline = baseline_challenger_votes * unreported_ratio
        else:
            unreported_incumbent_baseline = unreported_votes / 2
            unreported_challenger_baseline = unreported_votes / 2

        # Apply swing to unreported divisions
        # weighted_swing is (live_winner_pct - baseline_winner_pct)
        # Negative swing = incumbent losing vote share = challenger gaining
        # So: incumbent_votes should ADD swing_votes (negative when losing)
        #     challenger_votes should SUBTRACT swing_votes (negative becomes addition)
        swing_votes = unreported_votes * (weighted_swing / 100)

        projected_incumbent_votes = incumbent_current_votes + unreported_incumbent_baseline + swing_votes
        projected_challenger_votes = challenger_current_votes + unreported_challenger_baseline - swing_votes

        projected_margin_votes = projected_incumbent_votes - projected_challenger_votes
        projected_margin_pct = (projected_margin_votes / projected_total * 100) if projected_total > 0 else 0

        incumbent_final_pct = (projected_incumbent_votes / projected_total * 100) if projected_total > 0 else 0
        challenger_final_pct = (projected_challenger_votes / projected_total * 100) if projected_total > 0 else 0

        # Determine winner
        if projected_margin_votes > 0:
            projected_winner = baseline_winner
        else:
            # Find challenger party with highest current vote share
            # (not just first in dict order!)
            best_challenger = None
            best_challenger_votes = -1
            for party, data in vote_share.items():
                if party == "total_votes" or party == baseline_winner:
                    continue
                if isinstance(data, dict):
                    party_votes = data.get("votes", 0)
                    if party_votes > best_challenger_votes:
                        best_challenger_votes = party_votes
                        best_challenger = party
            projected_winner = best_challenger if best_challenger else "CHALLENGER"

        results[scenario_name] = {
            "projected_total_votes": projected_total,
            "projected_margin_votes": int(round(abs(projected_margin_votes))),
            "projected_margin_pct": round(abs(projected_margin_pct), 2),
            "projected_winner": projected_winner,
            "incumbent_final_pct": round(incumbent_final_pct, 2),
            "challenger_final_pct": round(challenger_final_pct, 2),
            "incumbent_leads": projected_margin_votes > 0,
        }

    return results


def classify_flip_status(
    projections: Dict,
    baseline_winner: str,
    pct_reported: float = 0.0,
) -> Tuple[str, float]:
    """
    Classify the seat as SURE_FLIP, WATCH, or SAFE.

    SURE_FLIP threshold is dynamic based on % of votes reported:
    - Below 50%: Fixed at 1.25%
    - 50% to 100%: Exponential decay from 1.25% to 0.25%

    WATCH threshold is fixed at 0.5%.

    Args:
        projections: Output from project_final_margin()
        baseline_winner: The 2021 winner party
        pct_reported: Percentage of baseline votes reported (current_votes / 2021_votes * 100)

    Returns:
        Tuple of (status, threshold_used)
        - status: "SURE_FLIP", "WATCH", or "SAFE"
        - threshold_used: The SURE_FLIP threshold that was applied
    """
    s1 = projections.get("same_as_2021", {})
    s2 = projections.get("trend_adjusted", {})

    s1_incumbent_leads = s1.get("incumbent_leads", True)
    s2_incumbent_leads = s2.get("incumbent_leads", True)

    s1_margin = s1.get("projected_margin_pct", 0)
    s2_margin = s2.get("projected_margin_pct", 0)

    # Get dynamic SURE_FLIP threshold based on % reported
    sure_flip_threshold = get_sure_flip_threshold(pct_reported)

    # Check for SURE_FLIP: Challenger wins in BOTH scenarios by >= threshold
    if not s1_incumbent_leads and not s2_incumbent_leads:
        if s1_margin >= sure_flip_threshold and s2_margin >= sure_flip_threshold:
            return ("SURE_FLIP", sure_flip_threshold)

    # Check for WATCH: Either scenario has margin <= 0.5%
    if s1_margin <= WATCH_THRESHOLD or s2_margin <= WATCH_THRESHOLD:
        return ("WATCH", sure_flip_threshold)

    # Also mark as WATCH if scenarios disagree on winner
    if s1_incumbent_leads != s2_incumbent_leads:
        return ("WATCH", sure_flip_threshold)

    return ("SAFE", sure_flip_threshold)


def get_confidence_level(pct_reported: float) -> Tuple[str, Optional[str]]:
    """
    Determine confidence level based on % of votes/divisions reported.

    Args:
        pct_reported: Percentage of baseline votes that have been reported

    Returns:
        (confidence_level, warning_message)
        - confidence_level: "HIGH", "MEDIUM", "LOW", or "VERY_LOW"
        - warning_message: None for HIGH, otherwise a warning string
    """
    if pct_reported >= CONFIDENCE_HIGH:
        return ("HIGH", None)
    elif pct_reported >= CONFIDENCE_MEDIUM:
        return ("MEDIUM", f"{pct_reported:.0f}% of divisions reported")
    elif pct_reported >= CONFIDENCE_LOW:
        return ("LOW", f"Only {pct_reported:.0f}% reported - projections preliminary")
    else:
        return ("VERY_LOW", f"Caution: Only {pct_reported:.0f}% reported")


def analyze_constituency_turnout(
    live_results: Dict,
    baseline_data: Dict,
    division_swings: Optional[List[Dict]] = None,
) -> Dict:
    """
    Complete turnout analysis for a constituency.

    This is the main entry point that combines all analysis functions.

    Args:
        live_results: {
            "division_totals": {"A1(a)": 823, "A1(b)": 0, ...},
            "candidate_votes": {
                "SLP": {"candidate": "CASIMIR", "votes": 3150},
                "UWP": {"candidate": "JOHNSON", "votes": 2270},
            }
        }
        baseline_data: Constituency data from swing_thresholds_2021.json
        division_swings: Optional list of division swing data for weighted calculation

    Returns:
        Complete analysis dict with all projections and classifications
    """
    division_totals = live_results.get("division_totals", {})
    candidate_votes = live_results.get("candidate_votes", {})

    # 1. Calculate vote completion
    vote_completion = calculate_vote_completion(division_totals, baseline_data)

    # 2. Calculate current vote share
    vote_share = calculate_vote_share(candidate_votes)

    # 3. Calculate weighted swing (if division swings provided)
    if division_swings:
        weighted_swing = calculate_weighted_swing(division_swings)
    else:
        # Default: no swing detected (use 0)
        weighted_swing = 0.0

    # 4. Generate turnout scenarios
    scenarios = generate_turnout_scenarios(
        vote_completion, weighted_swing, baseline_data
    )

    # 5. Project final margins
    projections = project_final_margin(
        vote_completion, vote_share, scenarios, weighted_swing, baseline_data
    )

    # 6. Classify flip status (using pct_of_2021 as the % reported metric)
    baseline_winner = baseline_data.get("winner", {}).get("party", "")
    pct_reported = vote_completion.get("pct_of_2021", 0)
    flip_status, sure_flip_threshold = classify_flip_status(
        projections, baseline_winner, pct_reported
    )

    # 7. Get confidence level
    confidence, confidence_warning = get_confidence_level(pct_reported)

    return {
        "vote_completion": vote_completion,
        "vote_share": vote_share,
        "weighted_swing": weighted_swing,
        "scenarios": scenarios,
        "projections": projections,
        "flip_status": flip_status,
        "sure_flip_threshold": round(sure_flip_threshold, 2),
        "confidence": confidence,
        "confidence_warning": confidence_warning,
        "baseline_winner": baseline_winner,
        "breakeven_swing_pct": baseline_data.get("breakeven_swing_pct", 0),
    }


def format_turnout_summary(analysis: Dict) -> str:
    """
    Format turnout analysis as a human-readable summary string.

    Args:
        analysis: Output from analyze_constituency_turnout()

    Returns:
        Multi-line summary string
    """
    vc = analysis.get("vote_completion", {})
    vs = analysis.get("vote_share", {})
    proj = analysis.get("projections", {})

    lines = []

    # Vote progress
    lines.append(f"Vote Progress: {vc.get('current_votes', 0):,} votes ({vc.get('pct_of_2021', 0):.1f}% of 2021)")

    # Current share
    parties = [p for p in vs if p != "total_votes"]
    share_parts = [f"{p} {vs[p].get('pct', 0):.1f}% ({vs[p].get('votes', 0):,})" for p in parties]
    lines.append(f"Current Share: {' | '.join(share_parts)}")

    # Projections
    s1 = proj.get("same_as_2021", {})
    s2 = proj.get("trend_adjusted", {})

    s1_str = f"{s1.get('projected_winner', '?')} +{s1.get('projected_margin_pct', 0):.1f}%"
    s2_str = f"{s2.get('projected_winner', '?')} +{s2.get('projected_margin_pct', 0):.1f}%"
    lines.append(f"Projection: {s1_str} (Same) | {s2_str} (Trend)")

    # Status and confidence with margins
    status = analysis.get("flip_status", "?")
    confidence = analysis.get("confidence", "?")
    warning = analysis.get("confidence_warning", "")
    threshold = analysis.get("sure_flip_threshold", 1.25)

    # Show projected margins for both scenarios
    s1_margin = s1.get('projected_margin_pct', 0)
    s2_margin = s2.get('projected_margin_pct', 0)
    s1_winner = s1.get('projected_winner', '?')
    s2_winner = s2.get('projected_winner', '?')
    margin_str = f"Margins: {s1_winner} +{s1_margin:.1f}% / {s2_winner} +{s2_margin:.1f}%"

    conf_str = f"{confidence}" + (f" ({warning})" if warning else "")
    lines.append(f"Status: {status} | {margin_str} | Threshold: {threshold:.2f}%")

    return "\n".join(lines)


if __name__ == "__main__":
    # Quick test with sample data
    import json

    # Load thresholds
    thresholds_file = Path(__file__).parent.parent / "data" / "swing_thresholds" / "swing_thresholds_2021.json"
    with open(thresholds_file) as f:
        thresholds = json.load(f)

    # Get Gros Islet baseline
    gros_islet = thresholds["constituencies"]["GROS - ISLET"]

    # Simulate partial results (50% reported)
    sample_live = {
        "division_totals": {
            "A1(a)": 920,  # ~50% of 2021
            "A1(b)": 880,
            "A3(a)": 850,
            "A4(a)": 0,  # Not reported
            "A4(b)": 0,  # Not reported
            "A1(c)": 0,
            "A1(d)": 0,
            "A2(a)": 0,
            "A2(b)": 0,
            "A3(b)": 0,
        },
        "candidate_votes": {
            "SLP": {"candidate": "CASIMIR", "votes": 1500},
            "UWP": {"candidate": "JOHNSON", "votes": 1150},
        },
    }

    # Sample division swings
    sample_swings = [
        {"division": "A1(a)", "swing_pct": 2.5, "votes": 920, "reported": True},
        {"division": "A1(b)", "swing_pct": 1.8, "votes": 880, "reported": True},
        {"division": "A3(a)", "swing_pct": 3.2, "votes": 850, "reported": True},
    ]

    # Run analysis
    analysis = analyze_constituency_turnout(sample_live, gros_islet, sample_swings)

    print("=" * 60)
    print("GROS ISLET - Turnout Model Test")
    print("=" * 60)
    print(format_turnout_summary(analysis))
    print()
    print("Full analysis:")
    print(json.dumps(analysis, indent=2))
