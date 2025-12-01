#!/usr/bin/env python3
"""
Generate Swing Thresholds

Generates a JSON file with breakeven swing thresholds for each constituency
based on baseline election data. Used for live election night monitoring.

Usage:
    python generate_swing_thresholds.py --baseline-year 2021
    python generate_swing_thresholds.py --baseline-year 2021 --threshold 10
    python generate_swing_thresholds.py --baseline-year 2021 --constituencies "Micoud North" "Soufriere"
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from swing_utils import (
    parse_vote_value,
    normalize_constituency_name,
    load_election_results,
    get_all_constituencies,
    extract_constituency_data,
)


def identify_winner_and_other(candidates: List[Dict]) -> Dict:
    """
    Identify the winner and calculate margin against combined "Other".

    Returns:
        {
            'winner': {'party': str, 'candidate': str, 'votes': int, 'pct': float},
            'runner_up': {'party': str, 'candidate': str, 'votes': int, 'pct': float},
            'other_combined': {'votes': int, 'pct': float},
            'margin': {'votes': int, 'pct': float},
            'total_votes': int
        }
    """
    if not candidates:
        return None

    # Sort by total votes descending
    sorted_candidates = sorted(candidates, key=lambda c: c['total_votes'], reverse=True)

    winner = sorted_candidates[0]
    total_votes = sum(c['total_votes'] for c in candidates)

    # Calculate Other (all non-winner)
    other_votes = sum(c['total_votes'] for c in sorted_candidates[1:])
    other_pct = (other_votes / total_votes * 100) if total_votes > 0 else 0

    # Get runner up
    runner_up = sorted_candidates[1] if len(sorted_candidates) > 1 else None

    # Calculate winner percentage (recalculate for consistency)
    winner_pct = (winner['total_votes'] / total_votes * 100) if total_votes > 0 else 0

    margin_votes = winner['total_votes'] - other_votes
    margin_pct = winner_pct - other_pct

    result = {
        'winner': {
            'party': winner['party'],
            'candidate': winner['name'],
            'votes': winner['total_votes'],
            'pct': round(winner_pct, 2)
        },
        'other_combined': {
            'votes': other_votes,
            'pct': round(other_pct, 2)
        },
        'margin': {
            'votes': margin_votes,
            'pct': round(margin_pct, 2)
        },
        'total_votes': total_votes
    }

    if runner_up:
        runner_up_pct = (runner_up['total_votes'] / total_votes * 100) if total_votes > 0 else 0
        result['runner_up'] = {
            'party': runner_up['party'],
            'candidate': runner_up['name'],
            'votes': runner_up['total_votes'],
            'pct': round(runner_up_pct, 2)
        }

    return result


def calculate_district_data(
    constituency_data: Dict,
    winner_party: str,
    threshold_pct: float
) -> Dict:
    """
    Calculate data for all polling districts.

    Returns dict mapping district code to:
        {
            'baseline_total_votes': int,
            'pct_of_constituency': float,
            'is_meaningful': bool,
            'winner_party_votes': int,
            'winner_party_pct': float,
            'other_votes': int,
            'other_pct': float,
            'breakeven_absolute_swing': int
        }
    """
    candidates = constituency_data['candidates']
    polling_divisions = constituency_data['polling_divisions']

    # Calculate total votes in constituency
    total_constituency_votes = sum(c['total_votes'] for c in candidates)

    district_data = {}
    for div in polling_divisions:
        # Sum votes across all candidates for this division
        div_total = sum(c['division_votes'].get(div, 0) for c in candidates)

        # Winner party votes in this division
        winner_votes = 0
        other_votes = 0
        for c in candidates:
            votes = c['division_votes'].get(div, 0)
            if c['party'] == winner_party:
                winner_votes += votes
            else:
                other_votes += votes

        # Calculate percentages
        pct_of_constituency = (div_total / total_constituency_votes * 100) if total_constituency_votes > 0 else 0
        winner_pct = (winner_votes / div_total * 100) if div_total > 0 else 0
        other_pct = (other_votes / div_total * 100) if div_total > 0 else 0

        is_meaningful = pct_of_constituency >= threshold_pct

        # Breakeven absolute swing: votes that need to shift for this district's contribution
        # If uniform swing S is applied, this district contributes S% * div_total votes change
        # For now, we'll calculate what the swing means in absolute terms at the district level
        # breakeven_swing_pct is calculated at constituency level, but we show per-district impact
        # This will be filled in later once we have the constituency breakeven

        district_data[div] = {
            'baseline_total_votes': div_total,
            'pct_of_constituency': round(pct_of_constituency, 2),
            'is_meaningful': is_meaningful,
            'winner_party_votes': winner_votes,
            'winner_party_pct': round(winner_pct, 2),
            'other_votes': other_votes,
            'other_pct': round(other_pct, 2),
            'breakeven_absolute_swing': 0  # Will be filled in
        }

    return district_data


def generate_constituency_threshold(
    constituency_data: Dict,
    threshold_pct: float
) -> Optional[Dict]:
    """
    Generate threshold data for a single constituency.
    """
    if not constituency_data:
        return None

    candidates = constituency_data['candidates']
    if not candidates:
        return None

    # Check for uncontested races
    if len(candidates) == 1:
        return {
            'status': 'uncontested',
            'winner': {
                'party': candidates[0]['party'],
                'candidate': candidates[0]['name'],
                'votes': candidates[0]['total_votes'],
                'pct': 100.0
            },
            'total_votes': candidates[0]['total_votes'],
            'breakeven_swing_pct': None,
            'meaningful_districts': [],
            'all_districts': {}
        }

    # Identify winner and margins
    margin_data = identify_winner_and_other(candidates)
    if not margin_data:
        return None

    winner_party = margin_data['winner']['party']
    margin_pct = margin_data['margin']['pct']

    # Calculate breakeven swing: S = margin / 2
    # A swing of S% means winner loses S percentage points, Other gains S points
    # Net margin change = 2S, so to flip: 2S >= margin_pct => S >= margin_pct / 2
    breakeven_swing_pct = margin_pct / 2

    # Calculate district data
    district_data = calculate_district_data(
        constituency_data, winner_party, threshold_pct
    )

    # Update breakeven absolute swing for each district
    # At breakeven, each district contributes: breakeven_swing_pct% * district_total_votes / 100
    for div, data in district_data.items():
        # Absolute votes that need to swing at breakeven
        # This represents votes moving from winner to other
        abs_swing = int(breakeven_swing_pct * data['baseline_total_votes'] / 100)
        data['breakeven_absolute_swing'] = abs_swing

    # Get list of meaningful districts
    meaningful_districts = [
        div for div, data in district_data.items()
        if data['is_meaningful']
    ]

    return {
        'status': 'contested',
        'original_name': constituency_data['original_name'],
        'winner': margin_data['winner'],
        'runner_up': margin_data.get('runner_up'),
        'other_combined': margin_data['other_combined'],
        'margin': margin_data['margin'],
        'total_votes': margin_data['total_votes'],
        'summary': constituency_data['summary'],
        'breakeven_swing_pct': round(breakeven_swing_pct, 2),
        'meaningful_districts': meaningful_districts,
        'all_districts': district_data
    }


def generate_all_thresholds(
    baseline_year: str,
    constituencies: Optional[List[str]] = None,
    threshold_pct: float = 10.0,
    data_dir: str = "data",
    closest_races_count: int = 5
) -> Dict:
    """
    Generate thresholds for all (or specified) constituencies.

    Args:
        baseline_year: Year to use as baseline (e.g., "2021")
        constituencies: List of constituency names, or None for all
        threshold_pct: Minimum % of constituency votes for "meaningful" district
        data_dir: Path to data directory
        closest_races_count: Number of closest races to include in summary

    Returns:
        Complete thresholds structure
    """
    results = load_election_results(baseline_year, data_dir)

    if constituencies is None or constituencies == ["all"]:
        constituencies = get_all_constituencies(results)

    thresholds = {
        'baseline_year': baseline_year,
        'threshold_pct': threshold_pct,
        'generated_at': datetime.now().isoformat(),
        'constituencies': {},
        'national_baseline': {
            'total_votes': 0,
            'party_votes': {},
            'party_pct': {}
        }
    }

    # Track national totals
    national_party_votes = {}
    national_total_votes = 0

    for const_name in constituencies:
        const_data = extract_constituency_data(results, const_name)
        if const_data is None:
            print(f"Warning: Could not find data for constituency '{const_name}'")
            continue

        threshold_data = generate_constituency_threshold(const_data, threshold_pct)
        if threshold_data:
            normalized_name = normalize_constituency_name(const_name)
            thresholds['constituencies'][normalized_name] = threshold_data

            # Aggregate national totals
            if threshold_data['status'] == 'contested':
                for candidate in const_data['candidates']:
                    party = candidate['party']
                    votes = candidate['total_votes']
                    national_party_votes[party] = national_party_votes.get(party, 0) + votes
                    national_total_votes += votes

    # Calculate national percentages
    thresholds['national_baseline']['total_votes'] = national_total_votes
    thresholds['national_baseline']['party_votes'] = national_party_votes
    if national_total_votes > 0:
        thresholds['national_baseline']['party_pct'] = {
            party: round(votes / national_total_votes * 100, 2)
            for party, votes in national_party_votes.items()
        }

    # Add summary stats
    contested = [c for c in thresholds['constituencies'].values() if c['status'] == 'contested']
    if contested:
        thresholds['summary'] = {
            'total_constituencies': len(thresholds['constituencies']),
            'contested': len(contested),
            'avg_breakeven_swing': round(
                sum(c['breakeven_swing_pct'] for c in contested) / len(contested), 2
            ),
            'closest_races': sorted(
                [(name, c['breakeven_swing_pct']) for name, c in thresholds['constituencies'].items()
                 if c['status'] == 'contested'],
                key=lambda x: x[1]
            )[:closest_races_count]
        }

    return thresholds


def main():
    parser = argparse.ArgumentParser(
        description="Generate swing thresholds for election monitoring"
    )
    parser.add_argument(
        "--baseline-year",
        required=True,
        help="Baseline election year (e.g., 2021)"
    )
    parser.add_argument(
        "--constituencies",
        nargs="*",
        default=None,
        help="Specific constituencies to analyze (default: all)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=10.0,
        help="Minimum %% of constituency votes for meaningful district (default: 10)"
    )
    parser.add_argument(
        "--closest-races",
        type=int,
        default=5,
        help="Number of closest races to show in summary (default: 5)"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON file path (default: data/swing_thresholds/swing_thresholds_{year}.json)"
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Data directory path (default: data)"
    )

    args = parser.parse_args()

    # Generate thresholds
    print(f"Generating swing thresholds from {args.baseline_year} election...")
    print(f"Meaningful district threshold: {args.threshold}%")

    thresholds = generate_all_thresholds(
        baseline_year=args.baseline_year,
        constituencies=args.constituencies,
        threshold_pct=args.threshold,
        data_dir=args.data_dir,
        closest_races_count=args.closest_races
    )

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path(args.data_dir) / "swing_thresholds" / f"swing_thresholds_{args.baseline_year}.json"

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write output
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(thresholds, f, indent=2, ensure_ascii=False)

    print(f"\nThresholds saved to: {output_path}")
    print(f"\nSummary:")
    print(f"  Constituencies analyzed: {len(thresholds['constituencies'])}")

    if 'summary' in thresholds:
        print(f"  Contested races: {thresholds['summary']['contested']}")
        print(f"  Average breakeven swing: {thresholds['summary']['avg_breakeven_swing']}%")
        print(f"\n  Closest races (lowest breakeven swing):")
        for name, swing in thresholds['summary']['closest_races']:
            print(f"    {name}: {swing}%")


if __name__ == "__main__":
    main()
