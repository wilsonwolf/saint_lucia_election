#!/usr/bin/env python3
"""
Backtesting for Turnout Model

Tests the turnout model using 2021 election data against 2016 baseline.
Generates progressive snapshots (25%, 50%, 75%, 100%) and evaluates predictions.

The snapshots are:
- Consistent: votes can only increase, divisions can't disappear
- Progressive: earlier snapshots are subsets of later ones
- Realistic: simulates how results come in on election night

Usage:
    python test_backtest.py
    python test_backtest.py --output-dir analysis/backtest
    python test_backtest.py --verbose
"""

import argparse
import json
import random
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
)
from election_night_monitoring.turnout_model import (
    analyze_constituency_turnout,
    calculate_weighted_swing,
)
from election_night_monitoring.generate_swing_charts import (
    analyze_constituency_swing,
    load_thresholds,
)


# Snapshot percentages for backtesting
SNAPSHOT_PERCENTAGES = [25, 50, 75, 100]


def get_actual_2021_winners() -> Dict[str, str]:
    """
    Get actual 2021 election winners for comparison.

    Returns:
        Dict mapping normalized constituency name to winning party
    """
    # These are the actual 2021 results
    return {
        "GROS ISLET": "SLP",
        "BABONNEAU": "SLP",
        "CASTRIES NORTH": "IND",  # Richard Frederick
        "CASTRIES CENTRAL": "IND",  # Sarah Flood-Beaubrun
        "CASTRIES EAST": "SLP",
        "CASTRIES SOUTH": "SLP",
        "CASTRIES SOUTH EAST": "SLP",
        "ANSE LA RAYE/CANARIES": "SLP",
        "SOUFRIERE": "SLP",
        "CHOISEUL": "UWP",
        "LABORIE": "SLP",
        "VIEUX FORT NORTH": "SLP",
        "VIEUX FORT SOUTH": "SLP",
        "MICOUD SOUTH": "UWP",
        "MICOUD NORTH": "SLP",
        "DENNERY SOUTH": "SLP",
        "DENNERY NORTH": "SLP",
    }


def generate_progressive_snapshots(
    constituency_data: Dict,
    percentages: List[int],
    seed: Optional[int] = None
) -> List[Dict]:
    """
    Generate progressive snapshots for a constituency.

    Each snapshot reveals more polling divisions in a way that:
    - Is consistent (votes never decrease)
    - Is progressive (divisions don't disappear)
    - Simulates realistic election night reporting

    Args:
        constituency_data: Full election data for constituency
        percentages: List of reporting percentages [25, 50, 75, 100]
        seed: Random seed for reproducibility

    Returns:
        List of snapshot dicts, one per percentage
    """
    if seed is not None:
        random.seed(seed)

    polling_divisions = constituency_data['polling_divisions']
    candidates = constituency_data['candidates']

    # Shuffle divisions to simulate random reporting order
    shuffled_divisions = list(polling_divisions)
    random.shuffle(shuffled_divisions)

    n_divisions = len(shuffled_divisions)
    snapshots = []

    for pct in percentages:
        # Calculate how many divisions to include
        n_included = max(1, int(n_divisions * pct / 100))
        included_divisions = set(shuffled_divisions[:n_included])

        # Build snapshot data
        snapshot = {
            'percentage': pct,
            'n_divisions_reported': n_included,
            'n_divisions_total': n_divisions,
            'divisions_included': list(included_divisions),
            'results': []
        }

        # Create filtered results for each candidate
        for candidate in candidates:
            filtered_votes = {}
            total_votes = 0

            for div in polling_divisions:
                if div in included_divisions:
                    votes = candidate['division_votes'].get(div, 0)
                    filtered_votes[div] = votes
                    total_votes += votes
                else:
                    filtered_votes[div] = 0

            snapshot['results'].append({
                'district': constituency_data['original_name'],
                'District': constituency_data['original_name'],
                'Candidate': candidate['name'],
                'Party': candidate['party'],
                'Total': str(total_votes),
                '% Total': '0.00',  # Will be calculated by analysis
                **{div: str(votes) for div, votes in filtered_votes.items()}
            })

        snapshots.append(snapshot)

    return snapshots


def run_prediction_on_snapshot(
    snapshot: Dict,
    thresholds: Dict,
    constituency_name: str
) -> Dict:
    """
    Run turnout model prediction on a single snapshot.

    Args:
        snapshot: Generated snapshot data
        thresholds: 2016 baseline thresholds
        constituency_name: Name of constituency

    Returns:
        Prediction results including flip status and projected winner
    """
    # Get threshold data for this constituency
    normalized_name = normalize_constituency_name(constituency_name)
    threshold_data = None

    # Try to find matching threshold
    for key, data in thresholds.get('constituencies', {}).items():
        if normalize_constituency_name(key) == normalized_name:
            threshold_data = data
            break

    if threshold_data is None or threshold_data.get('status') != 'contested':
        return {
            'error': 'No threshold data or uncontested',
            'flip_status': 'UNKNOWN',
            'projected_winner': 'UNKNOWN'
        }

    # Run the swing analysis
    analysis = analyze_constituency_swing(
        constituency_name,
        threshold_data,
        snapshot['results']
    )

    if analysis is None:
        return {
            'error': 'Analysis failed',
            'flip_status': 'UNKNOWN',
            'projected_winner': 'UNKNOWN'
        }

    # Extract predictions
    turnout = analysis.get('turnout_analysis', {})
    projections = turnout.get('projections', {}) if turnout else {}

    # Use same_as_2021 scenario for primary prediction
    s1 = projections.get('same_as_2021', {})
    s2 = projections.get('trend_adjusted', {})

    baseline_winner = threshold_data.get('winner', {}).get('party', '?')

    # Determine projected winner (use conservative estimate - agree in both scenarios)
    s1_winner = s1.get('projected_winner', baseline_winner)
    s2_winner = s2.get('projected_winner', baseline_winner)
    s1_leads = s1.get('incumbent_leads', True)
    s2_leads = s2.get('incumbent_leads', True)

    # Projected winner: if both scenarios agree, use that; otherwise use baseline
    if s1_leads and s2_leads:
        projected_winner = baseline_winner
    elif not s1_leads and not s2_leads and s1_winner == s2_winner:
        projected_winner = s1_winner
    else:
        projected_winner = baseline_winner  # Conservative default

    return {
        'pct_reported': snapshot['percentage'],
        'flip_status': turnout.get('flip_status', 'SAFE') if turnout else 'SAFE',
        'confidence': turnout.get('confidence', 'UNKNOWN') if turnout else 'UNKNOWN',
        'projected_winner': projected_winner,
        'baseline_winner': baseline_winner,
        's1_projection': {
            'winner': s1_winner,
            'margin_pct': s1.get('projected_margin_pct', 0),
            'incumbent_leads': s1_leads
        },
        's2_projection': {
            'winner': s2_winner,
            'margin_pct': s2.get('projected_margin_pct', 0),
            'incumbent_leads': s2_leads
        },
        'weighted_swing': analysis.get('weighted_swing', 0)
    }


def run_constituency_backtest(
    constituency_name: str,
    results_2021: List[Dict],
    thresholds_2016: Dict,
    seed: int = 42,
    verbose: bool = False
) -> Dict:
    """
    Run full backtest for a single constituency.

    Args:
        constituency_name: Name of constituency
        results_2021: 2021 election results
        thresholds_2016: 2016 baseline thresholds
        seed: Random seed for snapshot generation
        verbose: Print detailed output

    Returns:
        Backtest results with predictions at each snapshot
    """
    # Get 2021 data for this constituency
    const_data = extract_constituency_data(results_2021, constituency_name)
    if const_data is None:
        return {
            'constituency': constituency_name,
            'error': 'No 2021 data found',
            'predictions': []
        }

    # Get actual winner
    actual_winners = get_actual_2021_winners()
    normalized_name = normalize_constituency_name(constituency_name)
    actual_winner = actual_winners.get(normalized_name, 'UNKNOWN')

    # Get 2016 baseline winner
    threshold_data = None
    for key, data in thresholds_2016.get('constituencies', {}).items():
        if normalize_constituency_name(key) == normalized_name:
            threshold_data = data
            break

    baseline_winner = threshold_data.get('winner', {}).get('party', '?') if threshold_data else '?'

    # Determine if this was actually a flip
    actual_flip = baseline_winner != actual_winner

    # Generate progressive snapshots
    snapshots = generate_progressive_snapshots(
        const_data,
        SNAPSHOT_PERCENTAGES,
        seed=seed + hash(constituency_name) % 1000  # Unique seed per constituency
    )

    # Run predictions at each snapshot
    predictions = []
    for snapshot in snapshots:
        prediction = run_prediction_on_snapshot(
            snapshot,
            thresholds_2016,
            constituency_name
        )
        prediction['n_divisions'] = snapshot['n_divisions_reported']
        predictions.append(prediction)

    # Evaluate predictions
    correct_at = []
    for pred in predictions:
        if pred.get('projected_winner') == actual_winner:
            correct_at.append(pred['pct_reported'])

    return {
        'constituency': constituency_name,
        'normalized_name': normalized_name,
        'baseline_winner_2016': baseline_winner,
        'actual_winner_2021': actual_winner,
        'was_flip': actual_flip,
        'predictions': predictions,
        'correct_at_percentages': correct_at,
        'final_correct': actual_winner == predictions[-1].get('projected_winner') if predictions else False
    }


def run_full_backtest(
    output_dir: Optional[str] = None,
    verbose: bool = False,
    seed: int = 42
) -> Dict:
    """
    Run full backtest across all constituencies.

    Args:
        output_dir: Directory to save results
        verbose: Print detailed output
        seed: Random seed

    Returns:
        Complete backtest results
    """
    print("Loading data...")

    # Load 2016 thresholds and 2021 results
    thresholds_2016 = load_thresholds('data/swing_thresholds/swing_thresholds_2016.json')
    results_2021 = load_election_results('2021')

    print(f"Baseline year: 2016")
    print(f"Test year: 2021")
    print(f"Snapshot percentages: {SNAPSHOT_PERCENTAGES}")
    print()

    # Get all constituencies
    constituencies = get_all_constituencies(results_2021)
    print(f"Testing {len(constituencies)} constituencies...")
    print()

    # Run backtest for each constituency
    all_results = []
    for const_name in constituencies:
        result = run_constituency_backtest(
            const_name,
            results_2021,
            thresholds_2016,
            seed=seed,
            verbose=verbose
        )
        all_results.append(result)

        if verbose:
            print(f"\n{const_name}:")
            print(f"  Baseline (2016): {result['baseline_winner_2016']}")
            print(f"  Actual (2021):   {result['actual_winner_2021']}")
            print(f"  Was flip: {result['was_flip']}")
            for pred in result['predictions']:
                print(f"    {pred['pct_reported']}%: {pred['flip_status']} -> {pred['projected_winner']} [{pred['confidence']}]")

    # Calculate summary statistics
    n_constituencies = len(all_results)
    n_flips = sum(1 for r in all_results if r['was_flip'])
    n_correct_final = sum(1 for r in all_results if r['final_correct'])

    # Accuracy at each percentage
    accuracy_by_pct = {}
    for pct in SNAPSHOT_PERCENTAGES:
        correct = sum(1 for r in all_results
                     if pct in r.get('correct_at_percentages', []))
        accuracy_by_pct[pct] = round(correct / n_constituencies * 100, 1)

    # Flip detection accuracy
    flip_results = [r for r in all_results if r['was_flip']]
    flip_detection_by_pct = {}
    for pct in SNAPSHOT_PERCENTAGES:
        detected = sum(1 for r in flip_results
                      if any(p['pct_reported'] == pct and p['flip_status'] in ['SURE_FLIP', 'WATCH']
                            for p in r['predictions']))
        flip_detection_by_pct[pct] = round(detected / len(flip_results) * 100, 1) if flip_results else 0

    summary = {
        'total_constituencies': n_constituencies,
        'actual_flips_2016_to_2021': n_flips,
        'final_accuracy': round(n_correct_final / n_constituencies * 100, 1),
        'accuracy_by_reporting_pct': accuracy_by_pct,
        'flip_detection_by_reporting_pct': flip_detection_by_pct,
        'flip_constituencies': [r['constituency'] for r in all_results if r['was_flip']],
    }

    full_results = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'baseline_year': '2016',
            'test_year': '2021',
            'snapshot_percentages': SNAPSHOT_PERCENTAGES,
            'seed': seed
        },
        'summary': summary,
        'constituency_results': all_results
    }

    # Print summary
    print("\n" + "=" * 60)
    print("BACKTEST SUMMARY")
    print("=" * 60)
    print(f"Total constituencies: {n_constituencies}")
    print(f"Actual flips (2016->2021): {n_flips}")
    print(f"  {', '.join(summary['flip_constituencies'])}")
    print()
    print("Accuracy by reporting %:")
    for pct, acc in accuracy_by_pct.items():
        print(f"  {pct}%: {acc}% correct")
    print()
    print(f"Final accuracy: {summary['final_accuracy']}%")
    print()
    print("Flip detection by reporting %:")
    for pct, det in flip_detection_by_pct.items():
        print(f"  {pct}%: {det}% of flips flagged as SURE_FLIP or WATCH")

    # Save results if output directory specified
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        results_file = output_path / f'backtest_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(full_results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {results_file}")

    return full_results


def main():
    parser = argparse.ArgumentParser(
        description="Backtest turnout model using 2021 data against 2016 baseline"
    )
    parser.add_argument(
        "--output-dir",
        default="analysis/backtest",
        help="Output directory for results (default: analysis/backtest)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed output for each constituency"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )

    args = parser.parse_args()

    run_full_backtest(
        output_dir=args.output_dir,
        verbose=args.verbose,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
