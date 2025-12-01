#!/usr/bin/env python3
"""
Create two swing analysis charts:
1. Breakeven swing threshold by constituency (ascending order - most vulnerable first)
2. Z-score of actual 2016→2021 swing (identifies idiosyncratic constituencies)
"""

import json
import statistics
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def parse_vote_value(val) -> int:
    """Parse vote value, handling strings, numbers, and null values."""
    if val is None:
        return 0
    if isinstance(val, (int, float)):
        if val != val:  # NaN check
            return 0
        return int(val)
    if isinstance(val, str):
        val = val.replace(",", "").replace("*", "").replace("–", "0").replace("\u2013", "0").strip()
        if not val or val in ["NaN", ""]:
            return 0
        try:
            return int(float(val))
        except ValueError:
            return 0
    return 0


def load_swing_thresholds():
    """Load breakeven swing thresholds from 2021 data."""
    filepath = Path("data/swing_thresholds/swing_thresholds_2021.json")
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_summary_results(year: str) -> list:
    """Load summary results for a given year."""
    if year == "2016":
        filepath = Path("data/summary_results/saint_lucia_2016_summary_results.json")
    elif year == "2021":
        filepath = Path("data/summary_results/saint_lucia_2021_summary_results.json")
    else:
        raise ValueError(f"Unknown year: {year}")

    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def calculate_vote_shares(results: list, treat_ind_as_slp: bool = False) -> dict:
    """Calculate SLP vote share for each constituency.

    Args:
        results: Election results data
        treat_ind_as_slp: If True, treat IND/NGP in Castries North/Central as SLP-affiliated
    """
    shares = {}

    # Constituencies where independents should be treated as SLP-affiliated
    slp_affiliated_constituencies = ["CASTRIES NORTH", "CASTRIES CENTRAL"]

    for row in results:
        constituency = row.get("Constituency", "")
        if not constituency or constituency.upper() == "TOTAL":
            continue

        slp_votes = parse_vote_value(row.get("SLP", 0))
        uwp_votes = parse_vote_value(row.get("UWP", 0))

        # Include other parties
        other_votes = 0
        for party in ["LPM", "IND", "NGP", "PLP", "PDM", "NDP"]:
            party_votes = parse_vote_value(row.get(party, 0))

            # For Castries North/Central, treat IND/NGP as SLP-affiliated
            if treat_ind_as_slp and constituency.upper() in slp_affiliated_constituencies:
                if party in ["IND", "NGP"]:
                    slp_votes += party_votes
                else:
                    other_votes += party_votes
            else:
                other_votes += party_votes

        total_votes = slp_votes + uwp_votes + other_votes

        if total_votes > 0:
            slp_share = (slp_votes / total_votes) * 100
            shares[constituency] = {
                'slp_share': slp_share,
                'slp_votes': slp_votes,
                'total_votes': total_votes
            }

    return shares


def normalize_name(name: str) -> str:
    """Normalize constituency name for matching."""
    if not name:
        return ""
    name = name.upper().strip()
    name = name.replace("–", "-").replace("\u2013", "-")
    name = name.replace("V-FORT", "VIEUX FORT").replace("VIEUX-FORT", "VIEUX FORT")
    name = name.replace("/", " ").replace("-", " ")
    name = " ".join(name.split())
    return name


def create_breakeven_threshold_chart(thresholds_data: dict):
    """Create bar chart of breakeven thresholds in ascending order."""

    # Extract threshold data
    threshold_list = []
    for const_name, data in thresholds_data['constituencies'].items():
        if data.get('status') == 'contested':
            threshold_list.append({
                'constituency': data.get('original_name', const_name),
                'breakeven': data['breakeven_swing_pct'],
                'winner': data['winner']['party'],
                'margin_pct': data['margin']['pct']
            })

    # Sort by breakeven threshold (ascending - most vulnerable first)
    threshold_list.sort(key=lambda x: x['breakeven'])

    constituencies = [d['constituency'] for d in threshold_list]
    breakevens = [d['breakeven'] for d in threshold_list]
    winners = [d['winner'] for d in threshold_list]

    # Color by defending party
    colors = ['#FF6B6B' if w == 'SLP' else '#4ECDC4' for w in winners]

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))

    y_pos = range(len(constituencies))
    bars = ax.barh(y_pos, breakevens, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

    # Add value labels
    for i, (bar, be, winner) in enumerate(zip(bars, breakevens, winners)):
        ax.text(be + 0.3, i, f'{be:.1f}%', va='center', ha='left',
                fontweight='bold', fontsize=10)
        # Add defending party indicator
        ax.text(-0.5, i, f'({winner})', va='center', ha='right',
                fontsize=9, alpha=0.7)

    # Configure axes
    ax.set_yticks(y_pos)
    ax.set_yticklabels(constituencies, fontsize=11)
    ax.set_xlabel('Breakeven Swing Threshold (%)', fontsize=13, fontweight='bold')
    ax.set_title('Constituency Vulnerability: Breakeven Swing Thresholds\n(Based on 2021 Results - Lower = More Vulnerable)',
                 fontsize=14, fontweight='bold', pad=20)

    # Grid
    ax.grid(axis='x', alpha=0.3, linestyle='--')

    # Set x-axis limits
    ax.set_xlim(-2, max(breakevens) + 5)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#FF6B6B', alpha=0.7, edgecolor='black', label='SLP Defending'),
        Patch(facecolor='#4ECDC4', alpha=0.7, edgecolor='black', label='UWP Defending')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=11)

    plt.tight_layout()

    # Save chart
    output_file = Path("analysis/breakeven_thresholds_2021.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Chart 1 saved to: {output_file}")
    plt.close()

    # Print summary
    print(f"\nBreakeven Threshold Summary:")
    print("-" * 50)
    print(f"Most vulnerable seats (lowest threshold):")
    for d in threshold_list[:5]:
        print(f"  {d['constituency']} ({d['winner']}): {d['breakeven']:.1f}%")
    print(f"\nSafest seats (highest threshold):")
    for d in threshold_list[-5:]:
        print(f"  {d['constituency']} ({d['winner']}): {d['breakeven']:.1f}%")


def create_zscore_swing_chart(data_2016: dict, data_2021: dict):
    """Create bar chart of z-score swing deviation by constituency."""

    # Calculate swing for each constituency
    swing_data = []

    for const_2021, shares_2021 in data_2021.items():
        norm_2021 = normalize_name(const_2021)

        matched_2016 = None
        for const_2016, shares_2016 in data_2016.items():
            if normalize_name(const_2016) == norm_2021:
                matched_2016 = shares_2016
                break

        if matched_2016 is None:
            continue

        swing = shares_2021['slp_share'] - matched_2016['slp_share']

        swing_data.append({
            'constituency': const_2021,
            'swing': swing,
            'slp_2016': matched_2016['slp_share'],
            'slp_2021': shares_2021['slp_share']
        })

    # Calculate national average and standard deviation
    swings = [d['swing'] for d in swing_data]
    avg_swing = statistics.mean(swings)
    std_swing = statistics.stdev(swings)

    print(f"\nNational Swing Statistics:")
    print("-" * 50)
    print(f"Average swing: {avg_swing:+.2f}%")
    print(f"Standard deviation: {std_swing:.2f}%")

    # Calculate z-scores
    for d in swing_data:
        d['zscore'] = (d['swing'] - avg_swing) / std_swing

    # Sort by z-score descending (highest outperformance first)
    swing_data.sort(key=lambda x: x['zscore'], reverse=True)

    constituencies = [d['constituency'] for d in swing_data]
    zscores = [d['zscore'] for d in swing_data]
    actual_swings = [d['swing'] for d in swing_data]

    # Color by z-score direction
    colors = ['#FF6B6B' if z >= 0 else '#4ECDC4' for z in zscores]

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))

    y_pos = range(len(constituencies))
    bars = ax.barh(y_pos, zscores, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

    # Add value labels
    for i, (bar, z, swing) in enumerate(zip(bars, zscores, actual_swings)):
        if z >= 0:
            label_x = z + 0.1
            ha = 'left'
        else:
            label_x = z - 0.1
            ha = 'right'

        ax.text(label_x, i, f'{z:+.2f}σ ({swing:+.1f}%)', va='center', ha=ha,
                fontweight='bold', fontsize=9)

    # Configure axes
    ax.set_yticks(y_pos)
    ax.set_yticklabels(constituencies, fontsize=11)
    ax.set_xlabel('Z-Score (Standard Deviations from Mean Swing)', fontsize=13, fontweight='bold')
    ax.set_title(f'Swing Deviation from National Average (2016 → 2021)\n'
                 f'National Avg: {avg_swing:+.1f}%, Std Dev: {std_swing:.1f}%\n'
                 f'(IND/NGP in Castries North & Central treated as SLP-affiliated)',
                 fontsize=13, fontweight='bold', pad=20)

    # Add zero line
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1.5, alpha=0.5)

    # Grid
    ax.grid(axis='x', alpha=0.3, linestyle='--')

    # Set x-axis limits
    x_max = max(abs(min(zscores)), abs(max(zscores))) + 0.5
    ax.set_xlim(-x_max, x_max)

    # Invert y-axis so highest z-score is at top
    ax.invert_yaxis()

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#FF6B6B', alpha=0.7, edgecolor='black', label='Above Average SLP Swing'),
        Patch(facecolor='#4ECDC4', alpha=0.7, edgecolor='black', label='Below Average SLP Swing')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)

    plt.tight_layout()

    # Save chart
    output_file = Path("analysis/swing_zscore_2016_2021.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nChart 2 saved to: {output_file}")
    plt.close()

    # Print analysis
    print(f"\nIdiosyncratic Swing Analysis:")
    print("-" * 50)
    print(f"Constituencies with unusually HIGH SLP swing (z > 1):")
    for d in swing_data:
        if d['zscore'] > 1:
            print(f"  {d['constituency']}: z={d['zscore']:+.2f} (swing: {d['swing']:+.1f}%)")

    print(f"\nConstituencies with unusually LOW SLP swing (z < -1):")
    for d in swing_data:
        if d['zscore'] < -1:
            print(f"  {d['constituency']}: z={d['zscore']:+.2f} (swing: {d['swing']:+.1f}%)")


def main():
    print("=" * 60)
    print("SWING ANALYSIS CHARTS")
    print("=" * 60)

    # Chart 1: Breakeven thresholds
    print("\n[Chart 1] Loading breakeven threshold data...")
    thresholds = load_swing_thresholds()
    create_breakeven_threshold_chart(thresholds)

    # Chart 2: Z-score swing deviation
    # Treat IND/NGP in Castries North/Central as SLP-affiliated
    print("\n[Chart 2] Loading election data for z-score analysis...")
    print("(Treating IND/NGP in Castries North & Central as SLP-affiliated)")
    results_2016 = load_summary_results("2016")
    results_2021 = load_summary_results("2021")

    data_2016 = calculate_vote_shares(results_2016, treat_ind_as_slp=True)
    data_2021 = calculate_vote_shares(results_2021, treat_ind_as_slp=True)

    create_zscore_swing_chart(data_2016, data_2021)

    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
