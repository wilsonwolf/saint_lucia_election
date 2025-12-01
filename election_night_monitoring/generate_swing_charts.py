#!/usr/bin/env python3
"""
Generate Swing Analysis Charts

Creates polling division-level swing charts for election night monitoring:
1. Individual constituency charts (17 total)
2. Summary dashboard with all constituencies

Usage:
    python generate_swing_charts.py \
        --results data/live_election_results/results_TIMESTAMP.json \
        --thresholds data/swing_thresholds/swing_thresholds_2021.json \
        --output-dir analysis/live_2025/TIMESTAMP/
"""

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from swing_utils import (
    parse_vote_value,
    normalize_constituency_name,
    normalize_polling_division_name,
    extract_constituency_data,
    constituencies_match,
)
from election_night_monitoring.turnout_model import (
    analyze_constituency_turnout,
    format_turnout_summary,
    calculate_weighted_swing,
)

# Color scheme
COLOR_SLP = '#FF6B6B'  # Red for SLP
COLOR_UWP = '#4ECDC4'  # Teal for UWP
COLOR_POSITIVE = '#2ECC71'  # Green for positive swing (defending)
COLOR_NEGATIVE = '#E74C3C'  # Red for negative swing (opposition)
COLOR_NEUTRAL = '#95A5A6'  # Gray for neutral/unknown

# Constituency mapping for community names
CONSTITUENCY_MAP_PATH = Path(__file__).parent.parent / "data" / "constituency_maps" / "saint_lucia_constituencies.json"
_constituency_map_cache = None


def load_constituency_map() -> List[Dict]:
    """Load constituency mapping data (cached)."""
    global _constituency_map_cache
    if _constituency_map_cache is None:
        if CONSTITUENCY_MAP_PATH.exists():
            with open(CONSTITUENCY_MAP_PATH, 'r', encoding='utf-8') as f:
                _constituency_map_cache = json.load(f)
        else:
            _constituency_map_cache = []
    return _constituency_map_cache


def get_division_community_name(constituency_name: str, division_code: str, max_chars: int = 25) -> str:
    """
    Get community name(s) for a polling division.

    Args:
        constituency_name: Name of the constituency
        division_code: Polling division code (e.g., "A1(a)")
        max_chars: Maximum characters for the label

    Returns:
        Community name string, truncated if needed
    """
    const_map = load_constituency_map()
    if not const_map:
        return division_code

    # Normalize for matching - strip suffix like " - (A)" from map names
    const_norm = normalize_constituency_name(constituency_name)
    div_norm = normalize_polling_division_name(division_code)

    # Find matching constituency
    for const in const_map:
        map_const_name = const.get('constituency', '')
        # Strip the suffix like " – (A)" or " (B)" for matching and normalize spaces
        map_const_clean = re.sub(r'\s*[\(\[][A-Z][\)\]]\s*$', '', map_const_name)  # Remove (A), [B], etc. at end
        map_const_clean = re.sub(r'\s*[–-]\s*[\(\[][A-Z][\)\]]\s*$', '', map_const_clean)  # Remove " – (A)" at end
        map_const_clean = map_const_clean.replace(' / ', '/').strip()  # Normalize slashes
        map_const_clean = map_const_clean.replace('-', ' ')  # Normalize hyphens to spaces
        map_const_norm = normalize_constituency_name(map_const_clean)
        const_norm_clean = const_norm.replace(' ', '').replace('-', '')  # Remove all spaces/hyphens
        map_norm_clean = map_const_norm.replace(' ', '').replace('-', '')
        if map_norm_clean == const_norm_clean or const_norm_clean in map_norm_clean:
            # Find matching division - try exact match first, then base match (e.g., H1 for H1(a))
            for div in const.get('polling_divisions', []):
                div_name = div.get('division', '')
                map_div_norm = normalize_polling_division_name(div_name)
                # Try exact match or base match (H1 matches H1A, H1B, etc.)
                if map_div_norm == div_norm or (len(map_div_norm) < len(div_norm) and div_norm.startswith(map_div_norm)):
                    communities = div.get('communities', [])
                    if communities:
                        # Filter out header rows
                        communities = [c for c in communities if c.upper() != 'COMMUNITIES']
                        if communities:
                            # Join first 2-3 communities
                            if len(communities) == 1:
                                label = communities[0]
                            elif len(communities) == 2:
                                label = f"{communities[0]}, {communities[1]}"
                            else:
                                label = f"{communities[0]}, {communities[1]}..."

                            # Truncate if needed
                            if len(label) > max_chars:
                                label = label[:max_chars-3] + "..."
                            return label
            break

    # Fallback to division code
    return division_code


def load_live_results(results_path: str) -> List[Dict]:
    """Load live results from JSON file."""
    with open(results_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Handle both wrapped and unwrapped formats
    if isinstance(data, dict) and 'results' in data:
        return data['results']
    return data


def load_thresholds(thresholds_path: str) -> Dict:
    """Load swing thresholds JSON."""
    with open(thresholds_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def calculate_division_swing(
    baseline_data: Dict,
    live_party_votes: Dict[str, int],
    live_total: int,
    winner_party: str
) -> Dict:
    """
    Calculate swing for a single polling division.

    Returns dictionary with swing analysis data.
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
            'swing_direction': 'unknown'
        }

    # Calculate live winner percentage
    live_winner_votes = live_party_votes.get(winner_party, 0)
    live_winner_pct = (live_winner_votes / live_total * 100) if live_total > 0 else 0

    # Swing = change in winner's vote share
    swing_pct = live_winner_pct - baseline_winner_pct
    swing_direction = 'towards' if swing_pct >= 0 else 'away'

    return {
        'reported': True,
        'baseline_total': baseline_total,
        'live_total': live_total,
        'baseline_winner_pct': round(baseline_winner_pct, 2),
        'live_winner_pct': round(live_winner_pct, 2),
        'swing_pct': round(swing_pct, 2),
        'swing_direction': swing_direction
    }


def analyze_constituency_swing(
    const_name: str,
    threshold_data: Dict,
    live_results: List[Dict]
) -> Optional[Dict]:
    """
    Analyze swing for a single constituency.

    Returns detailed analysis including per-division swings and turnout projections.
    """
    if threshold_data.get('status') != 'contested':
        return {
            'status': 'uncontested',
            'baseline_winner': threshold_data.get('winner', {}).get('party', '?'),
            'divisions': {},
            'turnout_analysis': None
        }

    # Extract live data
    live_data = extract_constituency_data(
        live_results,
        threshold_data.get('original_name', const_name)
    )

    if live_data is None:
        return {
            'status': 'not_reported',
            'baseline_winner': threshold_data.get('winner', {}).get('party', '?'),
            'breakeven_swing_pct': threshold_data.get('breakeven_swing_pct', 0),
            'baseline_margin_pct': threshold_data.get('margin', {}).get('pct', 0),
            'divisions': {},
            'avg_swing': 0,
            'weighted_swing': 0,
            'pct_reported': 0,
            'turnout_analysis': None
        }

    winner_party = threshold_data['winner']['party']
    baseline_districts = threshold_data.get('all_districts', {})
    meaningful_list = threshold_data.get('meaningful_districts', [])

    # Match division names
    live_divisions = live_data['polling_divisions']
    live_normalized = {normalize_polling_division_name(d): d for d in live_divisions}

    # Calculate swing for each division
    division_swings = {}
    division_swing_list = []  # For weighted swing calculation
    division_totals = {}  # For turnout model
    candidate_votes = {}  # For turnout model

    for baseline_div, baseline_info in baseline_districts.items():
        baseline_norm = normalize_polling_division_name(baseline_div)
        live_div = live_normalized.get(baseline_norm)

        if live_div:
            # Get live votes by party
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

            swing_data = calculate_division_swing(
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
                'swing_pct': 0,
                'swing_direction': 'unknown'
            }
            division_totals[baseline_div] = 0

        swing_data['is_meaningful'] = baseline_div in meaningful_list
        swing_data['pct_of_constituency'] = baseline_info.get('pct_of_constituency', 0)
        division_swings[baseline_div] = swing_data

    # Calculate averages (simple and weighted)
    reported = [d for d in division_swings.values() if d['reported']]
    avg_swing = sum(d['swing_pct'] for d in reported) / len(reported) if reported else 0
    weighted_swing = calculate_weighted_swing(division_swing_list)
    pct_reported = len(reported) / len(baseline_districts) * 100 if baseline_districts else 0

    # Run turnout analysis
    turnout_input = {
        'division_totals': division_totals,
        'candidate_votes': candidate_votes
    }
    turnout_analysis = analyze_constituency_turnout(
        turnout_input, threshold_data, division_swing_list
    )

    # Determine status based on flip_status from turnout model
    breakeven = threshold_data.get('breakeven_swing_pct', 0)
    flip_status = turnout_analysis.get('flip_status', 'SAFE')

    if len(reported) == 0:
        status = 'not_reported'
    elif flip_status == 'SURE_FLIP':
        status = 'flip_projected'
    elif flip_status == 'WATCH':
        status = 'at_risk'
    else:
        # Also check old logic for backwards compatibility
        if weighted_swing <= -breakeven:
            status = 'flip_projected'
        elif weighted_swing <= -(breakeven - 2):
            status = 'at_risk'
        else:
            status = 'holding'

    return {
        'status': status,
        'baseline_winner': winner_party,
        'breakeven_swing_pct': breakeven,
        'baseline_margin_pct': threshold_data.get('margin', {}).get('pct', 0),
        'divisions': division_swings,
        'avg_swing': round(avg_swing, 2),
        'weighted_swing': round(weighted_swing, 2),
        'pct_reported': round(pct_reported, 1),
        'turnout_analysis': turnout_analysis
    }


def create_constituency_chart(
    const_name: str,
    analysis: Dict,
    output_path: Path
) -> None:
    """
    Create a single constituency swing chart with enhanced turnout information.

    Shows horizontal bars for each polling division with swing %,
    plus vote progress, current share, and projected outcomes.
    """
    divisions = analysis.get('divisions', {})
    if not divisions:
        return

    # Sort divisions by swing (most negative first)
    sorted_divs = sorted(
        divisions.items(),
        key=lambda x: x[1].get('swing_pct', 0)
    )

    # Prepare data - use community names instead of division codes
    div_codes = [d[0] for d in sorted_divs]
    div_names = [get_division_community_name(const_name, code, max_chars=30) for code in div_codes]
    swings = [d[1].get('swing_pct', 0) for d in sorted_divs]
    reported = [d[1].get('reported', False) for d in sorted_divs]
    meaningful = [d[1].get('is_meaningful', False) for d in sorted_divs]

    # Get turnout analysis
    turnout = analysis.get('turnout_analysis', {})
    vote_completion = turnout.get('vote_completion', {}) if turnout else {}
    vote_share = turnout.get('vote_share', {}) if turnout else {}
    projections = turnout.get('projections', {}) if turnout else {}
    flip_status = turnout.get('flip_status', 'SAFE') if turnout else 'SAFE'
    confidence = turnout.get('confidence', 'UNKNOWN') if turnout else 'UNKNOWN'
    confidence_warning = turnout.get('confidence_warning', '') if turnout else ''

    # Create figure with extra space for header
    n_divs = len(div_names)
    fig_height = max(8, 2.5 + n_divs * 0.5)  # Extra space for enhanced header
    fig, ax = plt.subplots(figsize=(12, fig_height))

    # Create bars
    y_pos = range(len(div_names))
    colors = []
    for swing, is_reported in zip(swings, reported):
        if not is_reported:
            colors.append(COLOR_NEUTRAL)
        elif swing >= 0:
            colors.append(COLOR_POSITIVE)
        else:
            colors.append(COLOR_NEGATIVE)

    bars = ax.barh(y_pos, swings, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)

    # Add breakeven line
    breakeven = analysis.get('breakeven_swing_pct', 0)
    if breakeven > 0:
        ax.axvline(x=-breakeven, color='red', linestyle='--', linewidth=2, label=f'Breakeven ({-breakeven:.1f}%)')
        ax.axvline(x=breakeven, color='green', linestyle='--', linewidth=2, alpha=0.5)

    # Add zero line
    ax.axvline(x=0, color='black', linewidth=1, alpha=0.5)

    # Labels
    for i, (div, swing, is_reported, is_meaningful) in enumerate(zip(div_names, swings, reported, meaningful)):
        label_color = 'black' if is_reported else 'gray'
        meaningful_marker = '*' if is_meaningful else ''

        if is_reported:
            label = f'{swing:+.1f}%'
        else:
            label = 'N/R'

        # Position label
        if swing >= 0:
            ax.text(swing + 0.3, i, f'{meaningful_marker}{label}', va='center', ha='left',
                    fontsize=9, fontweight='bold', color=label_color)
        else:
            ax.text(swing - 0.3, i, f'{label}{meaningful_marker}', va='center', ha='right',
                    fontsize=9, fontweight='bold', color=label_color)

    # Customize axes
    ax.set_yticks(y_pos)
    ax.set_yticklabels(div_names, fontsize=10)
    ax.set_xlabel('Swing (%)', fontsize=12, fontweight='bold')
    ax.invert_yaxis()

    # Set x-axis limits
    max_swing = max(abs(min(swings)), abs(max(swings)), breakeven) + 2
    ax.set_xlim(-max_swing, max_swing)

    # Title with status
    status = analysis.get('status', 'unknown').upper()
    winner = analysis.get('baseline_winner', '?')
    weighted_swing = analysis.get('weighted_swing', 0)
    pct_reported = analysis.get('pct_reported', 0)

    status_colors = {
        'HOLDING': 'green',
        'AT_RISK': 'orange',
        'FLIP_PROJECTED': 'red',
        'NOT_REPORTED': 'gray'
    }
    status_color = status_colors.get(status, 'black')

    title = f'{const_name} - {winner} DEFENDING'
    ax.set_title(title, fontsize=14, fontweight='bold', pad=80)  # More padding for multi-line header

    # Enhanced header with turnout info
    header_y = 1.12  # Starting y position for header text

    # Line 1: Vote Progress
    current_votes = vote_completion.get('current_votes', 0)
    pct_of_2021 = vote_completion.get('pct_of_2021', 0)
    vote_progress = f'Vote Progress: {current_votes:,} votes ({pct_of_2021:.1f}% of 2021)'
    ax.text(0.5, header_y, vote_progress, transform=ax.transAxes, fontsize=10,
            ha='center', color='black')

    # Line 2: Current Share by party
    share_parts = []
    for party, data in vote_share.items():
        if party != 'total_votes' and isinstance(data, dict):
            share_parts.append(f"{party} {data.get('pct', 0):.1f}% ({data.get('votes', 0):,})")
    if share_parts:
        current_share = f'Current Share: {" | ".join(share_parts)}'
    else:
        current_share = 'Current Share: No votes reported'
    ax.text(0.5, header_y - 0.04, current_share, transform=ax.transAxes, fontsize=10,
            ha='center', color='darkblue')

    # Line 3: Projections from both scenarios
    s1 = projections.get('same_as_2021', {})
    s2 = projections.get('trend_adjusted', {})
    if s1 and s2:
        s1_winner = s1.get('projected_winner', '?')
        s1_margin = s1.get('projected_margin_pct', 0)
        s1_leads = s1.get('incumbent_leads', True)
        s2_winner = s2.get('projected_winner', '?')
        s2_margin = s2.get('projected_margin_pct', 0)
        s2_leads = s2.get('incumbent_leads', True)

        proj_str = f'Projection: {s1_winner} +{s1_margin:.1f}% (Same) | {s2_winner} +{s2_margin:.1f}% (Trend)'
    else:
        proj_str = f'Projection: N/A | Breakeven: {-breakeven:.1f}% | Weighted Swing: {weighted_swing:+.1f}%'
    ax.text(0.5, header_y - 0.08, proj_str, transform=ax.transAxes, fontsize=10,
            ha='center', color='darkgreen' if status == 'HOLDING' else status_color)

    # Line 4: Status and Confidence (with dynamic threshold)
    sure_flip_threshold = turnout.get('sure_flip_threshold', 1.25) if turnout else 1.25
    conf_str = f'{confidence}' + (f' - {confidence_warning}' if confidence_warning else '')
    status_line = f'Status: {flip_status} (threshold: {sure_flip_threshold:.2f}%) | Confidence: {conf_str} | {pct_reported:.0f}% Reported'
    ax.text(0.5, header_y - 0.12, status_line, transform=ax.transAxes, fontsize=10,
            ha='center', color=status_color, fontweight='bold')

    # Grid
    ax.grid(axis='x', alpha=0.3, linestyle='--')

    # Legend
    legend_elements = [
        mpatches.Patch(color=COLOR_POSITIVE, alpha=0.8, label='Swing towards defending party'),
        mpatches.Patch(color=COLOR_NEGATIVE, alpha=0.8, label='Swing away from defending party'),
        mpatches.Patch(color=COLOR_NEUTRAL, alpha=0.8, label='Not reported'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_dashboard(
    all_analyses: Dict[str, Dict],
    thresholds: Dict,
    output_path: Path,
    timestamp: datetime
) -> None:
    """
    Create summary dashboard with all constituencies.

    Enhanced with scenario-based seat projections and flip status summary.
    """
    # Sort by flip status priority (SURE_FLIP first, then WATCH, then SAFE)
    def get_flip_priority(analysis):
        turnout = analysis.get('turnout_analysis', {})
        flip_status = turnout.get('flip_status', 'SAFE') if turnout else 'SAFE'
        priority_map = {'SURE_FLIP': 0, 'WATCH': 1, 'SAFE': 2}
        return priority_map.get(flip_status, 3)

    status_order = {'flip_projected': 0, 'at_risk': 1, 'holding': 2, 'not_reported': 3, 'uncontested': 4}
    sorted_items = sorted(
        all_analyses.items(),
        key=lambda x: (get_flip_priority(x[1]), status_order.get(x[1].get('status', 'unknown'), 99), x[0])
    )

    n_constituencies = len(sorted_items)
    n_cols = 4
    n_rows = (n_constituencies + n_cols - 1) // n_cols

    fig = plt.figure(figsize=(16, 3 * n_rows + 4))  # Extra height for enhanced header
    gs = GridSpec(n_rows + 1, n_cols, figure=fig, height_ratios=[0.6] + [1] * n_rows)

    # Enhanced Header
    header_ax = fig.add_subplot(gs[0, :])
    header_ax.axis('off')

    # Calculate projected seats under both scenarios
    def count_projected_seats(scenario_key):
        slp_total = 0
        uwp_total = 0
        ind_total = 0

        for analysis in all_analyses.values():
            turnout = analysis.get('turnout_analysis', {})
            projections = turnout.get('projections', {}) if turnout else {}
            scenario = projections.get(scenario_key, {})
            projected_winner = scenario.get('projected_winner', analysis.get('baseline_winner', '?'))
            incumbent_leads = scenario.get('incumbent_leads', True)

            # If incumbent leads, use baseline winner
            if incumbent_leads:
                winner = analysis.get('baseline_winner', '?')
            else:
                winner = projected_winner

            if winner == 'SLP':
                slp_total += 1
            elif winner == 'UWP':
                uwp_total += 1
            elif winner == 'IND':
                ind_total += 1

        return slp_total, uwp_total, ind_total

    slp_s1, uwp_s1, ind_s1 = count_projected_seats('same_as_2021')
    slp_s2, uwp_s2, ind_s2 = count_projected_seats('trend_adjusted')

    # Count flip statuses
    sure_flips = []
    watch_list = []
    safe_list = []

    for const_name, analysis in all_analyses.items():
        turnout = analysis.get('turnout_analysis', {})
        flip_status = turnout.get('flip_status', 'SAFE') if turnout else 'SAFE'
        if flip_status == 'SURE_FLIP':
            sure_flips.append(const_name)
        elif flip_status == 'WATCH':
            watch_list.append(const_name)
        else:
            safe_list.append(const_name)

    # Build header text
    header_lines = [
        f'ELECTION NIGHT DASHBOARD - {timestamp.strftime("%Y-%m-%d %H:%M:%S")}',
        '',
        f'PROJECTED SEATS:',
        f'  Same as 2021 Turnout:  SLP {slp_s1}  |  UWP {uwp_s1}' + (f'  |  IND {ind_s1}' if ind_s1 > 0 else ''),
        f'  Trend-Adjusted:        SLP {slp_s2}  |  UWP {uwp_s2}' + (f'  |  IND {ind_s2}' if ind_s2 > 0 else ''),
        '',
        f'STATUS: Sure Flips: {len(sure_flips)} | Watch: {len(watch_list)} | Safe: {len(safe_list)}',
    ]

    header_text = '\n'.join(header_lines)
    header_ax.text(0.5, 0.5, header_text, transform=header_ax.transAxes,
                   fontsize=12, fontfamily='monospace', ha='center', va='center')

    # Status symbols and colors
    flip_status_config = {
        'SURE_FLIP': ('FLIP', '#D32F2F', '#FFCDD2'),
        'WATCH': ('WATCH', '#F57C00', '#FFE0B2'),
        'SAFE': ('SAFE', '#388E3C', '#C8E6C9'),
    }

    status_symbols = {
        'holding': ('HOLDING', 'green', 'o'),
        'at_risk': ('AT RISK', 'orange', '^'),
        'flip_projected': ('FLIP', 'red', 'X'),
        'not_reported': ('N/R', 'gray', 's'),
        'uncontested': ('UNCON', 'blue', 'D')
    }

    # Draw each constituency cell
    for idx, (const_name, analysis) in enumerate(sorted_items):
        row = idx // n_cols + 1
        col = idx % n_cols
        ax = fig.add_subplot(gs[row, col])

        status = analysis.get('status', 'unknown')
        winner = analysis.get('baseline_winner', '?')
        weighted_swing = analysis.get('weighted_swing', 0)
        pct_reported = analysis.get('pct_reported', 0)
        breakeven = analysis.get('breakeven_swing_pct', 0)

        # Get turnout analysis
        turnout = analysis.get('turnout_analysis', {})
        flip_status = turnout.get('flip_status', 'SAFE') if turnout else 'SAFE'
        confidence = turnout.get('confidence', 'UNKNOWN') if turnout else 'UNKNOWN'
        projections = turnout.get('projections', {}) if turnout else {}

        # Background color based on flip status
        _, _, bg_color = flip_status_config.get(flip_status, ('SAFE', 'green', '#E8F5E9'))
        if status == 'not_reported':
            bg_color = '#F5F5F5'
        elif status == 'uncontested':
            bg_color = '#E3F2FD'
        ax.set_facecolor(bg_color)

        # Status label and color
        flip_label, flip_color, _ = flip_status_config.get(flip_status, ('SAFE', 'green', '#E8F5E9'))
        if status in ['not_reported', 'uncontested']:
            status_label, status_color, _ = status_symbols.get(status, ('?', 'gray', 'o'))
        else:
            status_label = flip_label
            status_color = flip_color

        # Draw content
        ax.text(0.5, 0.88, const_name.replace(' ', '\n'), transform=ax.transAxes,
                fontsize=9, fontweight='bold', ha='center', va='top')

        ax.text(0.5, 0.60, f'{winner} DEF', transform=ax.transAxes,
                fontsize=9, ha='center', va='center',
                color=COLOR_SLP if winner == 'SLP' else (COLOR_UWP if winner == 'UWP' else 'purple'))

        # Projection info
        if status not in ['not_reported', 'uncontested'] and projections:
            s1 = projections.get('same_as_2021', {})
            s2 = projections.get('trend_adjusted', {})
            s1_margin = s1.get('projected_margin_pct', 0)
            s2_margin = s2.get('projected_margin_pct', 0)
            s1_winner = s1.get('projected_winner', '?')
            s2_winner = s2.get('projected_winner', '?')

            # Show projected margins
            proj_text = f'{s1_winner[:3]} +{s1_margin:.1f}% / {s2_winner[:3]} +{s2_margin:.1f}%'
            ax.text(0.5, 0.42, proj_text, transform=ax.transAxes,
                    fontsize=8, ha='center', va='center', color='black')

            swing_color = 'green' if weighted_swing >= 0 else 'red'
            ax.text(0.5, 0.26, f'Swing: {weighted_swing:+.1f}%', transform=ax.transAxes,
                    fontsize=10, fontweight='bold', ha='center', va='center', color=swing_color)

        # Confidence and status badge
        if status not in ['not_reported', 'uncontested']:
            ax.text(0.5, 0.12, f'{confidence}', transform=ax.transAxes,
                    fontsize=7, ha='center', va='center', color='gray')

        ax.text(0.5, 0.02, f'{status_label} ({pct_reported:.0f}%)', transform=ax.transAxes,
                fontsize=9, fontweight='bold', ha='center', va='bottom', color=status_color)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])

        # Border based on flip status
        for spine in ax.spines.values():
            spine.set_edgecolor(status_color)
            spine.set_linewidth(2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def generate_all_charts(
    results_path: str,
    thresholds_path: str,
    output_dir: str
) -> Dict[str, Dict]:
    """
    Generate all charts and return analysis data.

    Args:
        results_path: Path to live results JSON
        thresholds_path: Path to swing thresholds JSON
        output_dir: Directory to save charts

    Returns:
        Dictionary of constituency analyses
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load data
    live_results = load_live_results(results_path)
    thresholds = load_thresholds(thresholds_path)

    timestamp = datetime.now()

    # Analyze all constituencies
    all_analyses = {}
    for const_name, threshold_data in thresholds.get('constituencies', {}).items():
        analysis = analyze_constituency_swing(const_name, threshold_data, live_results)
        if analysis:
            all_analyses[const_name] = analysis

    # Generate individual charts
    print(f"Generating {len(all_analyses)} constituency charts...")
    for const_name, analysis in all_analyses.items():
        # Create safe filename
        safe_name = const_name.replace('/', '_').replace(' ', '_')
        chart_path = output_path / f'{safe_name}_swing.png'
        create_constituency_chart(const_name, analysis, chart_path)

    # Generate dashboard
    print("Generating summary dashboard...")
    dashboard_path = output_path / 'swing_dashboard.png'
    create_dashboard(all_analyses, thresholds, dashboard_path, timestamp)

    # Save analysis JSON
    analysis_json_path = output_path / 'swing_analysis.json'
    with open(analysis_json_path, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': timestamp.isoformat(),
            'results_file': results_path,
            'thresholds_file': thresholds_path,
            'constituencies': all_analyses
        }, f, indent=2, ensure_ascii=False)

    print(f"Charts saved to {output_path}")
    return all_analyses


def main():
    parser = argparse.ArgumentParser(
        description="Generate swing analysis charts"
    )
    parser.add_argument(
        "--results",
        required=True,
        help="Path to live results JSON file"
    )
    parser.add_argument(
        "--thresholds",
        default="data/swing_thresholds/swing_thresholds_2021.json",
        help="Path to swing thresholds JSON"
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for charts"
    )

    args = parser.parse_args()

    generate_all_charts(args.results, args.thresholds, args.output_dir)


if __name__ == "__main__":
    main()
