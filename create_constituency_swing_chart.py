#!/usr/bin/env python3
"""
Create constituency-level swing bar chart for 2016 → 2021.
Positive swing = SLP gained, Negative swing = UWP gained.
"""

import json
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


def load_summary_results(year: str) -> dict:
    """Load summary results for a given year."""
    if year == "2016":
        filepath = Path("data/summary_results/saint_lucia_2016_summary_results.json")
    elif year == "2021":
        filepath = Path("data/summary_results/saint_lucia_2021_summary_results.json")
    else:
        raise ValueError(f"Unknown year: {year}")

    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def calculate_vote_shares(results: list) -> dict:
    """Calculate SLP vote share for each constituency."""
    shares = {}

    for row in results:
        constituency = row.get("Constituency", "")
        if not constituency or constituency.upper() == "TOTAL":
            continue

        slp_votes = parse_vote_value(row.get("SLP", 0))
        uwp_votes = parse_vote_value(row.get("UWP", 0))

        # Include other parties
        other_votes = 0
        for party in ["LPM", "IND", "NGP", "PLP", "PDM", "NDP"]:
            other_votes += parse_vote_value(row.get(party, 0))

        total_votes = slp_votes + uwp_votes + other_votes

        if total_votes > 0:
            slp_share = (slp_votes / total_votes) * 100
            shares[constituency] = {
                'slp_share': slp_share,
                'slp_votes': slp_votes,
                'uwp_votes': uwp_votes,
                'total_votes': total_votes
            }

    return shares


def normalize_constituency_name(name: str) -> str:
    """Normalize constituency name for matching."""
    if not name:
        return ""
    name = name.upper().strip()
    name = name.replace("–", "-").replace("\u2013", "-")
    name = name.replace("V-FORT", "VIEUX FORT").replace("VIEUX-FORT", "VIEUX FORT")
    name = " ".join(name.split())
    return name


def create_swing_chart(data_2016: dict, data_2021: dict):
    """Create horizontal bar chart showing constituency-level swing."""

    # Calculate swing for each constituency
    swing_data = []

    for const_2021, shares_2021 in data_2021.items():
        # Find matching constituency in 2016
        norm_2021 = normalize_constituency_name(const_2021)

        matched_2016 = None
        for const_2016, shares_2016 in data_2016.items():
            if normalize_constituency_name(const_2016) == norm_2021:
                matched_2016 = shares_2016
                break

        if matched_2016 is None:
            print(f"Warning: No 2016 match for {const_2021}")
            continue

        # Calculate swing (positive = SLP gained)
        swing = shares_2021['slp_share'] - matched_2016['slp_share']

        swing_data.append({
            'constituency': const_2021,
            'swing': swing,
            'slp_2016': matched_2016['slp_share'],
            'slp_2021': shares_2021['slp_share']
        })

    # Sort by swing (highest SLP swing at top)
    swing_data.sort(key=lambda x: x['swing'], reverse=True)

    # Prepare chart data
    constituencies = [d['constituency'] for d in swing_data]
    swings = [d['swing'] for d in swing_data]

    # Assign colors based on swing direction
    colors = ['#FF6B6B' if s >= 0 else '#4ECDC4' for s in swings]

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))

    y_pos = range(len(constituencies))
    bars = ax.barh(y_pos, swings, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

    # Add value labels on bars
    for i, (bar, swing) in enumerate(zip(bars, swings)):
        # Position label on the appropriate side of the bar
        if swing >= 0:
            label_x = swing + 0.5
            ha = 'left'
        else:
            label_x = swing - 0.5
            ha = 'right'

        ax.text(label_x, i, f'{swing:+.1f}%', va='center', ha=ha,
                fontweight='bold', fontsize=10)

    # Configure axes
    ax.set_yticks(y_pos)
    ax.set_yticklabels(constituencies, fontsize=11)
    ax.set_xlabel('Vote Share Change (%)', fontsize=13, fontweight='bold')
    ax.set_title('Constituency Swing: 2016 → 2021\n(Positive = SLP Gain, Negative = Runner-Up Gain)',
                 fontsize=15, fontweight='bold', pad=20)

    # Add zero line
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1.5, alpha=0.5)

    # Grid
    ax.grid(axis='x', alpha=0.3, linestyle='--')

    # Set x-axis limits with padding
    x_min = min(swings) - 3
    x_max = max(swings) + 3
    ax.set_xlim(x_min, x_max)

    # Invert y-axis so highest swing is at top
    ax.invert_yaxis()

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#FF6B6B', alpha=0.7, edgecolor='black', label='SLP Gain'),
        Patch(facecolor='#4ECDC4', alpha=0.7, edgecolor='black', label='Runner-Up Gain')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=11)

    plt.tight_layout()

    # Save chart
    output_dir = Path("analysis")
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "constituency_swing_2016_2021.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Chart saved to: {output_file}")
    plt.close()

    # Print summary
    print(f"\nSwing Summary (2016 → 2021):")
    print("-" * 50)
    slp_gains = [d for d in swing_data if d['swing'] > 0]
    uwp_gains = [d for d in swing_data if d['swing'] < 0]
    print(f"Constituencies with SLP swing: {len(slp_gains)}")
    print(f"Constituencies with UWP swing: {len(uwp_gains)}")
    avg_swing = sum(d['swing'] for d in swing_data) / len(swing_data)
    print(f"Average swing: {avg_swing:+.2f}% (towards {'SLP' if avg_swing > 0 else 'UWP'})")

    print(f"\nTop 5 SLP swings:")
    for d in swing_data[:5]:
        print(f"  {d['constituency']}: {d['swing']:+.1f}%")

    print(f"\nTop UWP swings (if any):")
    for d in reversed(swing_data[-3:]):
        if d['swing'] < 0:
            print(f"  {d['constituency']}: {d['swing']:+.1f}%")


def main():
    print("Loading election data...")

    results_2016 = load_summary_results("2016")
    results_2021 = load_summary_results("2021")

    print(f"2016: {len(results_2016)} records")
    print(f"2021: {len(results_2021)} records")

    data_2016 = calculate_vote_shares(results_2016)
    data_2021 = calculate_vote_shares(results_2021)

    print(f"\n2016 constituencies: {len(data_2016)}")
    print(f"2021 constituencies: {len(data_2021)}")

    create_swing_chart(data_2016, data_2021)


if __name__ == "__main__":
    main()
