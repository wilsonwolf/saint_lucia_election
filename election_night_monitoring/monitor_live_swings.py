#!/usr/bin/env python3
"""
Monitor Live Election Swings

Monitors live election results against baseline thresholds to detect
potential constituency flips in real-time.

Usage:
    python monitor_live_swings.py --thresholds data/swing_thresholds_2021.json \
        --live-results data/election_night_results/st_lucia_2025_full_results.json

    # Test with historical data
    python monitor_live_swings.py --thresholds data/swing_thresholds_2016.json \
        --live-results data/election_night_results/st_lucia_2021_full_results.json
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from swing_utils import (
    parse_vote_value,
    normalize_constituency_name,
    normalize_polling_division_name,
    load_election_results,
    get_all_constituencies,
    extract_constituency_data,
    constituencies_match,
)
from election_night_monitoring.turnout_model import (
    analyze_constituency_turnout,
    format_turnout_summary,
    calculate_weighted_swing,
)


def load_thresholds(thresholds_path: str) -> Dict:
    """Load and validate thresholds JSON."""
    with open(thresholds_path, 'r', encoding='utf-8') as f:
        thresholds = json.load(f)
    return thresholds


def match_district_names(baseline_districts: List[str], live_districts: List[str]) -> Dict[str, str]:
    """
    Match polling division names between baseline and live data.
    Returns {baseline_name: live_name} mapping.
    """
    mapping = {}
    live_normalized = {normalize_polling_division_name(d): d for d in live_districts}

    for baseline_div in baseline_districts:
        baseline_norm = normalize_polling_division_name(baseline_div)
        if baseline_norm in live_normalized:
            mapping[baseline_div] = live_normalized[baseline_norm]
        else:
            # Try direct match
            if baseline_div in live_districts:
                mapping[baseline_div] = baseline_div

    return mapping


def calculate_district_swing(
    baseline_data: Dict,
    live_division_votes: Dict[str, int],
    live_total: int,
    winner_party: str
) -> Dict:
    """
    Calculate swing for a single district.

    Returns:
        {
            'reported': bool,
            'baseline_total': int,
            'live_total': int,
            'baseline_winner_pct': float,
            'live_winner_pct': float,
            'swing_pct': float,  # Negative = away from winner
            'swing_direction': str,  # 'towards' or 'away'
            'turnout_change_pct': float
        }
    """
    baseline_total = baseline_data['baseline_total_votes']
    baseline_winner_pct = baseline_data['winner_party_pct']

    if live_total == 0:
        return {
            'reported': False,
            'baseline_total': baseline_total,
            'live_total': 0,
            'baseline_winner_pct': baseline_winner_pct,
            'live_winner_pct': 0,
            'swing_pct': 0,
            'swing_direction': 'unknown',
            'turnout_change_pct': 0
        }

    # Calculate live winner percentage
    live_winner_votes = live_division_votes.get(winner_party, 0)
    live_winner_pct = (live_winner_votes / live_total * 100) if live_total > 0 else 0

    # Swing = change in winner's vote share
    swing_pct = live_winner_pct - baseline_winner_pct
    swing_direction = 'towards' if swing_pct >= 0 else 'away'

    # Turnout change
    turnout_change = ((live_total - baseline_total) / baseline_total * 100) if baseline_total > 0 else 0

    return {
        'reported': True,
        'baseline_total': baseline_total,
        'live_total': live_total,
        'baseline_winner_pct': round(baseline_winner_pct, 2),
        'live_winner_pct': round(live_winner_pct, 2),
        'swing_pct': round(swing_pct, 2),
        'swing_direction': swing_direction,
        'turnout_change_pct': round(turnout_change, 2)
    }


def analyze_constituency(
    const_name: str,
    threshold_data: Dict,
    live_results: List[Dict]
) -> Optional[Dict]:
    """
    Analyze a single constituency against baseline.

    Returns full analysis with swing data, turnout model, and projections.
    """
    if threshold_data['status'] != 'contested':
        return {
            'status': 'uncontested',
            'baseline_winner': threshold_data['winner']['party'],
            'message': 'Uncontested in baseline election',
            'turnout_analysis': None
        }

    # Extract live data for this constituency
    live_data = extract_constituency_data(live_results, threshold_data.get('original_name', const_name))

    if live_data is None:
        return {
            'status': 'not_reported',
            'baseline_winner': threshold_data['winner']['party'],
            'baseline_margin_pct': threshold_data['margin']['pct'],
            'breakeven_swing_pct': threshold_data['breakeven_swing_pct'],
            'message': 'No live data available',
            'turnout_analysis': None
        }

    winner_party = threshold_data['winner']['party']
    baseline_districts = threshold_data['all_districts']
    meaningful_list = threshold_data['meaningful_districts']

    # Match district names
    live_divisions = live_data['polling_divisions']
    district_mapping = match_district_names(list(baseline_districts.keys()), live_divisions)

    # Calculate swing for each district
    district_swings = {}
    division_swing_list = []  # For weighted swing calculation
    division_totals = {}  # For turnout model
    candidate_votes = {}  # For turnout model

    for baseline_div, baseline_info in baseline_districts.items():
        live_div = district_mapping.get(baseline_div)

        if live_div:
            # Get live votes by party for this division
            live_party_votes = {}
            live_total = 0
            for candidate in live_data['candidates']:
                party = candidate['party']
                votes = candidate['division_votes'].get(live_div, 0)
                live_party_votes[party] = live_party_votes.get(party, 0) + votes
                live_total += votes

                # Aggregate for turnout model
                if party not in candidate_votes:
                    candidate_votes[party] = {
                        'candidate': candidate['name'],
                        'votes': 0
                    }
                candidate_votes[party]['votes'] += votes

            swing_data = calculate_district_swing(
                baseline_info, live_party_votes, live_total, winner_party
            )
            division_totals[baseline_div] = live_total

            # Add to swing list for weighted calculation
            if swing_data['reported']:
                division_swing_list.append({
                    'division': baseline_div,
                    'swing_pct': swing_data['swing_pct'],
                    'votes': live_total,
                    'reported': True
                })
        else:
            swing_data = {
                'reported': False,
                'baseline_total': baseline_info['baseline_total_votes'],
                'live_total': 0,
                'baseline_winner_pct': baseline_info['winner_party_pct'],
                'live_winner_pct': 0,
                'swing_pct': 0,
                'swing_direction': 'unknown',
                'turnout_change_pct': 0
            }
            division_totals[baseline_div] = 0

        swing_data['is_meaningful'] = baseline_div in meaningful_list
        district_swings[baseline_div] = swing_data

    # Calculate averages
    all_reported = [d for d in district_swings.values() if d['reported']]
    meaningful_reported = [d for d in district_swings.values() if d['reported'] and d['is_meaningful']]

    avg_swing_all = 0
    avg_swing_meaningful = 0

    if all_reported:
        avg_swing_all = sum(d['swing_pct'] for d in all_reported) / len(all_reported)

    if meaningful_reported:
        avg_swing_meaningful = sum(d['swing_pct'] for d in meaningful_reported) / len(meaningful_reported)

    # Use turnout model's weighted swing calculation
    weighted_avg_swing = calculate_weighted_swing(division_swing_list)

    # Run turnout analysis
    turnout_input = {
        'division_totals': division_totals,
        'candidate_votes': candidate_votes
    }
    turnout_analysis = analyze_constituency_turnout(
        turnout_input, threshold_data, division_swing_list
    )

    # Determine status based on turnout model's flip_status
    breakeven = threshold_data['breakeven_swing_pct']
    flip_status = turnout_analysis.get('flip_status', 'SAFE')

    if len(all_reported) == 0:
        status = 'not_reported'
    elif flip_status == 'SURE_FLIP':
        status = 'flip_projected'
    elif flip_status == 'WATCH':
        status = 'at_risk'
    else:
        # Fallback to original logic
        if weighted_avg_swing <= -breakeven:
            status = 'flip_projected'
        elif weighted_avg_swing <= -(breakeven - 2):
            status = 'at_risk'
        else:
            status = 'holding'

    # Calculate live totals
    live_total_votes = sum(c['total_votes'] for c in live_data['candidates'])
    live_party_totals = {}
    for c in live_data['candidates']:
        party = c['party']
        live_party_totals[party] = live_party_totals.get(party, 0) + c['total_votes']

    # Determine current leader
    if live_party_totals:
        current_leader = max(live_party_totals.items(), key=lambda x: x[1])
        current_leader_party = current_leader[0]
        current_leader_votes = current_leader[1]
    else:
        current_leader_party = None
        current_leader_votes = 0

    return {
        'status': status,
        'baseline_winner': winner_party,
        'baseline_margin_pct': threshold_data['margin']['pct'],
        'breakeven_swing_pct': breakeven,
        'reporting': {
            'total_districts': len(baseline_districts),
            'reported_districts': len(all_reported),
            'meaningful_total': len(meaningful_list),
            'meaningful_reported': len(meaningful_reported),
            'pct_reported': round(len(all_reported) / len(baseline_districts) * 100, 1) if baseline_districts else 0
        },
        'swing_analysis': {
            'avg_swing_all_districts': round(avg_swing_all, 2),
            'avg_swing_meaningful_only': round(avg_swing_meaningful, 2),
            'weighted_avg_swing': round(weighted_avg_swing, 2),
            'swing_direction': 'towards' if avg_swing_all >= 0 else 'away'
        },
        'live_totals': {
            'total_votes': live_total_votes,
            'party_votes': live_party_totals,
            'current_leader': current_leader_party,
            'current_leader_votes': current_leader_votes
        },
        'district_swings': district_swings,
        'turnout_analysis': turnout_analysis
    }


def calculate_national_swing(
    thresholds: Dict,
    live_results: List[Dict]
) -> Dict:
    """
    Calculate national-level swing across all constituencies.
    """
    baseline_national = thresholds.get('national_baseline', {})
    baseline_total = baseline_national.get('total_votes', 0)
    baseline_party_votes = baseline_national.get('party_votes', {})
    baseline_party_pct = baseline_national.get('party_pct', {})

    # Get all constituencies from live results
    live_constituencies = get_all_constituencies(live_results)

    # Aggregate live totals
    live_party_votes = {}
    live_total_votes = 0

    for const_name in live_constituencies:
        const_data = extract_constituency_data(live_results, const_name)
        if const_data:
            for candidate in const_data['candidates']:
                party = candidate['party']
                votes = candidate['total_votes']
                live_party_votes[party] = live_party_votes.get(party, 0) + votes
                live_total_votes += votes

    # Calculate live percentages
    live_party_pct = {}
    if live_total_votes > 0:
        live_party_pct = {
            party: round(votes / live_total_votes * 100, 2)
            for party, votes in live_party_votes.items()
        }

    # Calculate swings for main parties
    party_swings = {}
    for party in set(list(baseline_party_pct.keys()) + list(live_party_pct.keys())):
        baseline_pct = baseline_party_pct.get(party, 0)
        live_pct = live_party_pct.get(party, 0)
        party_swings[party] = round(live_pct - baseline_pct, 2)

    # Estimate total expected votes
    estimated_total = baseline_total  # Use baseline as estimate

    return {
        'baseline_total': baseline_total,
        'live_total': live_total_votes,
        'estimated_total': estimated_total,
        'reporting_pct': round(live_total_votes / estimated_total * 100, 1) if estimated_total > 0 else 0,
        'baseline_party_votes': baseline_party_votes,
        'baseline_party_pct': baseline_party_pct,
        'live_party_votes': live_party_votes,
        'live_party_pct': live_party_pct,
        'party_swings': party_swings
    }


def format_console_output(
    national: Dict,
    constituency_analyses: Dict[str, Dict],
    thresholds: Dict
) -> str:
    """Format full console output with turnout model information."""
    lines = []
    baseline_year = thresholds.get('baseline_year', 'N/A')

    # Header
    lines.append("=" * 80)
    lines.append("SAINT LUCIA ELECTION SWING MONITOR (with Turnout Model)")
    lines.append(f"Baseline: {baseline_year} | Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 80)
    lines.append("")

    # National Swing Section
    lines.append("NATIONAL SWING (all reported votes)")
    lines.append("-" * 40)

    reporting_pct = national.get('reporting_pct', 0)
    live_total = national.get('live_total', 0)
    estimated_total = national.get('estimated_total', 0)
    baseline_total = national.get('baseline_total', 0)
    lines.append(f"Total Votes Reported: {live_total:,} / ~{estimated_total:,} ({reporting_pct}%)")

    # Turnout comparison vs baseline
    if baseline_total > 0 and reporting_pct > 0:
        # Project current turnout to 100% and compare to baseline
        projected_total = live_total / (reporting_pct / 100) if reporting_pct > 0 else 0
        turnout_change = ((projected_total - baseline_total) / baseline_total) * 100
        direction = "UP" if turnout_change > 0 else "DOWN"
        turnout_str = f"TURNOUT: {direction} {turnout_change:+.1f}% vs {baseline_year} (projected {projected_total:,.0f} total)"
        lines.append(turnout_str)
    lines.append("")

    # Baseline vs Live comparison
    baseline_pct = national.get('baseline_party_pct', {})
    live_pct = national.get('live_party_pct', {})
    party_swings = national.get('party_swings', {})

    # Format party comparison - prioritize SLP and UWP
    main_parties = ['SLP', 'UWP']
    other_parties = [p for p in baseline_pct.keys() if p not in main_parties]

    baseline_str = "Baseline:  "
    live_str = "Live:      "
    for party in main_parties + other_parties:
        if party in baseline_pct:
            baseline_str += f"{party} {baseline_pct.get(party, 0):.1f}%  |  "
            live_str += f"{party} {live_pct.get(party, 0):.1f}%  |  "

    lines.append(baseline_str.rstrip(" |  "))
    lines.append(live_str.rstrip(" |  "))
    lines.append("")

    # Main swing (SLP vs UWP typically)
    slp_swing = party_swings.get('SLP', 0)
    uwp_swing = party_swings.get('UWP', 0)
    if slp_swing != 0 or uwp_swing != 0:
        if slp_swing > 0:
            lines.append(f"National Swing: +{slp_swing}% (towards SLP)")
        else:
            lines.append(f"National Swing: {slp_swing}% (towards UWP)")

    lines.append("")

    # Calculate projected seats under both scenarios
    def count_projected_seats_by_scenario(scenario_key):
        slp_total, uwp_total, other_total = 0, 0, 0
        for analysis in constituency_analyses.values():
            turnout = analysis.get('turnout_analysis', {})
            projections = turnout.get('projections', {}) if turnout else {}
            scenario = projections.get(scenario_key, {})
            incumbent_leads = scenario.get('incumbent_leads', True)
            projected_winner = scenario.get('projected_winner', analysis.get('baseline_winner', '?'))

            if incumbent_leads:
                winner = analysis.get('baseline_winner', '?')
            else:
                winner = projected_winner

            if winner == 'SLP':
                slp_total += 1
            elif winner == 'UWP':
                uwp_total += 1
            else:
                other_total += 1
        return slp_total, uwp_total, other_total

    slp_s1, uwp_s1, other_s1 = count_projected_seats_by_scenario('same_as_2021')
    slp_s2, uwp_s2, other_s2 = count_projected_seats_by_scenario('trend_adjusted')

    lines.append("PROJECTED SEATS (by Turnout Scenario):")
    lines.append(f"  Same as 2021 Turnout:  SLP {slp_s1}  |  UWP {uwp_s1}" + (f"  |  Other {other_s1}" if other_s1 > 0 else ""))
    lines.append(f"  Trend-Adjusted:        SLP {slp_s2}  |  UWP {uwp_s2}" + (f"  |  Other {other_s2}" if other_s2 > 0 else ""))
    lines.append("=" * 80)
    lines.append("")

    # Flip Status Summary (from turnout model)
    sure_flips = []
    watch_list = []
    safe_list = []
    not_reported = []

    for name, analysis in constituency_analyses.items():
        turnout = analysis.get('turnout_analysis', {})
        flip_status = turnout.get('flip_status', 'SAFE') if turnout else 'SAFE'
        status = analysis.get('status', 'not_reported')

        if status == 'not_reported':
            not_reported.append(name)
        elif flip_status == 'SURE_FLIP':
            sure_flips.append(name)
        elif flip_status == 'WATCH':
            watch_list.append(name)
        else:
            safe_list.append(name)

    lines.append("FLIP STATUS SUMMARY (Turnout Model)")
    lines.append("-" * 40)
    lines.append(f"  SURE FLIP:    {len(sure_flips)} constituencies")
    if sure_flips:
        lines.append(f"                [{', '.join(sure_flips[:5])}{'...' if len(sure_flips) > 5 else ''}]")
    lines.append(f"  WATCH:        {len(watch_list)} constituencies")
    if watch_list:
        lines.append(f"                [{', '.join(watch_list[:5])}{'...' if len(watch_list) > 5 else ''}]")
    lines.append(f"  SAFE:         {len(safe_list)} constituencies")
    lines.append(f"  NOT REPORTED: {len(not_reported)} constituencies")
    lines.append("")

    # Priority alerts
    alerts = []
    for name, analysis in constituency_analyses.items():
        turnout = analysis.get('turnout_analysis', {})
        flip_status = turnout.get('flip_status', 'SAFE') if turnout else 'SAFE'
        projections = turnout.get('projections', {}) if turnout else {}
        confidence = turnout.get('confidence', 'UNKNOWN') if turnout else 'UNKNOWN'

        if flip_status == 'SURE_FLIP':
            winner = analysis.get('baseline_winner', '?')
            s1 = projections.get('same_as_2021', {})
            s2 = projections.get('trend_adjusted', {})
            s1_margin = s1.get('projected_margin_pct', 0)
            s2_margin = s2.get('projected_margin_pct', 0)
            alerts.append((name, 'CRITICAL', f"SURE FLIP - {winner} losing (margins: {s1_margin:.1f}% / {s2_margin:.1f}%) [{confidence}]"))
        elif flip_status == 'WATCH':
            winner = analysis.get('baseline_winner', '?')
            s1 = projections.get('same_as_2021', {})
            s2 = projections.get('trend_adjusted', {})
            s1_margin = s1.get('projected_margin_pct', 0)
            s2_margin = s2.get('projected_margin_pct', 0)
            s1_winner = s1.get('projected_winner', '?')
            s2_winner = s2.get('projected_winner', '?')
            alerts.append((name, 'WARNING', f"WATCH - {s1_winner} +{s1_margin:.1f}% / {s2_winner} +{s2_margin:.1f}% [{confidence}]"))

    if alerts:
        lines.append("=" * 80)
        lines.append("PRIORITY ALERTS")
        lines.append("=" * 80)
        for name, severity, message in sorted(alerts, key=lambda x: 0 if x[1] == 'CRITICAL' else 1):
            lines.append(f"[{severity}] {name}: {message}")
        lines.append("")

    # Constituency details
    lines.append("=" * 80)
    lines.append("CONSTITUENCY DETAILS")
    lines.append("=" * 80)

    # Sort by flip status priority then name
    def get_sort_priority(item):
        name, analysis = item
        turnout = analysis.get('turnout_analysis', {})
        flip_status = turnout.get('flip_status', 'SAFE') if turnout else 'SAFE'
        status = analysis.get('status', 'not_reported')

        if status == 'not_reported':
            return (3, name)
        flip_order = {'SURE_FLIP': 0, 'WATCH': 1, 'SAFE': 2}
        return (flip_order.get(flip_status, 3), name)

    sorted_constituencies = sorted(constituency_analyses.items(), key=get_sort_priority)

    for name, analysis in sorted_constituencies:
        lines.append("")
        status = analysis.get('status', 'unknown')
        winner = analysis.get('baseline_winner', '?')
        margin = analysis.get('baseline_margin_pct', 0)
        breakeven = analysis.get('breakeven_swing_pct', 0)

        # Header line
        lines.append(f"{name} - {winner} DEFENDING (margin: {margin:.2f}%, breakeven: {breakeven:.2f}%)")
        lines.append("-" * 70)

        if status == 'not_reported':
            lines.append("No live data available")
            continue

        # Reporting status
        reporting = analysis.get('reporting', {})
        reported = reporting.get('reported_districts', 0)
        total = reporting.get('total_districts', 0)
        pct = reporting.get('pct_reported', 0)
        meaningful_rep = reporting.get('meaningful_reported', 0)
        meaningful_tot = reporting.get('meaningful_total', 0)
        lines.append(f"Reporting: {reported}/{total} districts ({pct}%)")

        # Turnout model info
        turnout = analysis.get('turnout_analysis', {})
        if turnout:
            vote_completion = turnout.get('vote_completion', {})
            vote_share = turnout.get('vote_share', {})
            projections = turnout.get('projections', {})
            flip_status = turnout.get('flip_status', 'SAFE')
            confidence = turnout.get('confidence', 'UNKNOWN')
            confidence_warning = turnout.get('confidence_warning', '')

            # Vote progress
            current_votes = vote_completion.get('current_votes', 0)
            pct_of_2021 = vote_completion.get('pct_of_2021', 0)
            lines.append(f"Vote Progress: {current_votes:,} votes ({pct_of_2021:.1f}% of 2021)")

            # Current share
            share_parts = []
            for party, data in vote_share.items():
                if party != 'total_votes' and isinstance(data, dict):
                    share_parts.append(f"{party} {data.get('pct', 0):.1f}%")
            if share_parts:
                lines.append(f"Current Share: {' | '.join(share_parts)}")

            # Projections
            s1 = projections.get('same_as_2021', {})
            s2 = projections.get('trend_adjusted', {})
            if s1 and s2:
                s1_winner = s1.get('projected_winner', '?')
                s1_margin = s1.get('projected_margin_pct', 0)
                s2_winner = s2.get('projected_winner', '?')
                s2_margin = s2.get('projected_margin_pct', 0)
                lines.append(f"Projection: {s1_winner} +{s1_margin:.1f}% (Same) | {s2_winner} +{s2_margin:.1f}% (Trend)")

            # Status and confidence (with dynamic threshold)
            sure_flip_threshold = turnout.get('sure_flip_threshold', 1.25)
            conf_str = f"{confidence}" + (f" - {confidence_warning}" if confidence_warning else "")
            lines.append(f"Flip Status: {flip_status} (threshold: {sure_flip_threshold:.2f}%) | Confidence: {conf_str}")
        else:
            # Swings (fallback to old display)
            swing_data = analysis.get('swing_analysis', {})
            avg_all = swing_data.get('avg_swing_all_districts', 0)
            weighted = swing_data.get('weighted_avg_swing', 0)
            direction = 'UWP' if avg_all < 0 else 'SLP' if avg_all > 0 else 'neutral'
            lines.append(f"Avg Swing: {avg_all:+.2f}% | Weighted: {weighted:+.2f}% (towards {direction})")

        # Live totals
        live = analysis.get('live_totals', {})
        current_leader = live.get('current_leader', 'N/A')
        party_votes = live.get('party_votes', {})
        if party_votes:
            vote_str = "  ".join(f"{p}: {v:,}" for p, v in sorted(party_votes.items(), key=lambda x: -x[1]))
            lines.append(f"Current Votes: {vote_str} | Leading: {current_leader}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Monitor live election swings against baseline thresholds"
    )
    parser.add_argument(
        "--thresholds",
        default="data/swing_thresholds/swing_thresholds_2021.json",
        help="Path to thresholds JSON file (default: data/swing_thresholds/swing_thresholds_2021.json)"
    )
    parser.add_argument(
        "--live-results",
        required=True,
        help="Path to live results JSON file"
    )
    parser.add_argument(
        "--constituencies",
        nargs="*",
        default=None,
        help="Specific constituencies to monitor (default: all)"
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Output detailed analysis to JSON file"
    )

    args = parser.parse_args()

    # Load data
    thresholds = load_thresholds(args.thresholds)

    # Load live results (handles both wrapped and unwrapped formats)
    with open(args.live_results, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Handle wrapped format with metadata
    if isinstance(data, dict) and 'results' in data:
        live_results = data['results']
    else:
        live_results = data

    # Determine which constituencies to analyze
    if args.constituencies:
        constituencies_to_analyze = args.constituencies
    else:
        constituencies_to_analyze = list(thresholds['constituencies'].keys())

    # Analyze each constituency
    constituency_analyses = {}
    for const_name in constituencies_to_analyze:
        normalized = normalize_constituency_name(const_name)
        threshold_data = thresholds['constituencies'].get(normalized)
        if threshold_data is None:
            # Try to find by original name
            for key, data in thresholds['constituencies'].items():
                if constituencies_match(const_name, data.get('original_name', '')):
                    threshold_data = data
                    normalized = key
                    break

        if threshold_data:
            analysis = analyze_constituency(const_name, threshold_data, live_results)
            if analysis:
                constituency_analyses[normalized] = analysis

    # Calculate national swing
    national = calculate_national_swing(thresholds, live_results)

    # Format and print console output
    console_output = format_console_output(national, constituency_analyses, thresholds)
    print(console_output)

    # Output JSON if requested
    if args.output_json:
        output_data = {
            'timestamp': datetime.now().isoformat(),
            'baseline_year': thresholds.get('baseline_year'),
            'live_results_file': args.live_results,
            'national': national,
            'constituencies': constituency_analyses
        }
        with open(args.output_json, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"\nDetailed analysis saved to: {args.output_json}")


if __name__ == "__main__":
    main()
