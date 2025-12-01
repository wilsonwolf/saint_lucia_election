#!/usr/bin/env python3
"""
Analyze polling division swings for Micoud North across multiple elections (2011, 2016, 2021).
Includes analysis of independent candidate strength in 2011 and which party was most affected.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend


def parse_vote_value(val) -> int:
    """Parse vote value, handling strings, numbers, and null values."""
    if val is None:
        return 0
    if isinstance(val, str):
        val = val.replace(",", "").replace("*", "").replace("–", "0").replace("\u2013", "0").strip()
        if not val or val in ["NaN", ""]:
            return 0
        try:
            return int(float(val))
        except ValueError:
            return 0
    return int(float(val)) if isinstance(val, float) else int(val)


def normalize_constituency_name(name: str) -> str:
    """Normalize constituency name for matching."""
    if not name:
        return ""
    name = name.replace("–", "-").strip()
    name = " ".join(name.split())
    return name.upper()


def normalize_polling_division_name(div_name: str) -> str:
    """Normalize polling division name for matching."""
    if not div_name:
        return ""
    div_name = div_name.upper().strip()
    div_name = div_name.replace(" ", "").replace("(", "").replace(")", "")
    return div_name


def extract_polling_division_data(results: List[Dict], constituency_name: str) -> Dict:
    """Extract polling division-level data for a constituency."""
    normalized_const = normalize_constituency_name(constituency_name)
    
    constituency_records = []
    summary_data = {}
    current_constituency = None
    found_constituency = False
    
    for record in results:
        district = record.get('district', '')
        normalized_district = normalize_constituency_name(district)
        
        # Check if this is our constituency - use exact matching to avoid substring issues
        matches = False
        if normalized_const == normalized_district:
            matches = True
        elif normalized_const.replace(" ", "").replace("/", "") == normalized_district.replace(" ", "").replace("/", ""):
            matches = True
        
        if matches:
            found_constituency = True
            if "Party" in record and record.get("Candidate"):
                constituency_records.append(record)
                current_constituency = district
            elif "summary_label" in record:
                # Summary record for our constituency
                label = record.get("summary_label", "")
                value = record.get("summary_value", "")
                # Only store non-empty values (or update if current value is empty)
                if value and str(value).strip() and str(value).strip() not in ["", "–", "\u2013"]:
                    summary_data[label] = value
                elif label not in summary_data:
                    # Store empty value only if we don't have one yet
                    summary_data[label] = value
        elif found_constituency:
            # After finding our constituency, check if this is a summary record
            if "summary_label" in record:
                # Collect summary records that appear right after the constituency
                label = record.get("summary_label", "")
                value = record.get("summary_value", "")
                # Only store non-empty values (or update if current value is empty)
                if value and str(value).strip() and str(value).strip() not in ["", "–", "\u2013"]:
                    summary_data[label] = value
                elif label not in summary_data:
                    # Store empty value only if we don't have one yet
                    summary_data[label] = value
            elif "Party" in record and record.get("Candidate"):
                # Check if this is a different constituency
                next_district = normalize_constituency_name(district)
                if normalized_const != next_district and normalized_const.replace(" ", "").replace("/", "") != next_district.replace(" ", "").replace("/", ""):
                    # We've hit the next constituency, stop collecting
                    break
    
    if not constituency_records:
        return None
    
    # Get polling division columns
    exclude_cols = {"district", "District", "Candidate", "Party", "Total", "% Total", 
                   "%", "summary_label", "summary_value", "col_7", "col_8", "col_9", 
                   "col_10", "col_11", "col_12"}
    
    polling_divisions = set()
    for record in constituency_records:
        for key in record.keys():
            if key not in exclude_cols and key.strip() and not key.startswith("col_"):
                polling_divisions.add(key)
    
    # Extract data by polling division
    division_data = {}
    
    for div in polling_divisions:
        division_data[div] = {
            'slp_votes': 0,
            'uwp_votes': 0,
            'ind_votes': 0,
            'other_votes': 0,
            'total_votes': 0
        }
    
    for record in constituency_records:
        party = record.get("Party", "").strip()
        for div in polling_divisions:
            votes = parse_vote_value(record.get(div, 0))
            if votes > 0:
                if party == "SLP":
                    division_data[div]['slp_votes'] += votes
                elif party == "UWP":
                    division_data[div]['uwp_votes'] += votes
                elif party == "IND":
                    division_data[div]['ind_votes'] += votes
                else:
                    division_data[div]['other_votes'] += votes
                
                division_data[div]['total_votes'] += votes
    
    # Determine winner
    winner_party = None
    max_total = 0
    
    for record in constituency_records:
        total = parse_vote_value(record.get("Total", 0))
        party = record.get("Party", "").strip()
        if total > max_total:
            max_total = total
            winner_party = party
    
    # Calculate percentages
    for div in division_data:
        total = division_data[div]['total_votes']
        if total > 0:
            division_data[div]['slp_pct'] = (division_data[div]['slp_votes'] / total) * 100
            division_data[div]['uwp_pct'] = (division_data[div]['uwp_votes'] / total) * 100
            division_data[div]['ind_pct'] = (division_data[div]['ind_votes'] / total) * 100
        else:
            division_data[div]['slp_pct'] = 0
            division_data[div]['uwp_pct'] = 0
            division_data[div]['ind_pct'] = 0
    
    # Parse summary data
    electors = parse_vote_value(summary_data.get("No. of Electors", summary_data.get("No. Of Electors", 0)))
    votes_cast = parse_vote_value(summary_data.get("Votes Cast", 0))
    turnout_pct = 0
    if electors > 0:
        turnout_pct = (votes_cast / electors) * 100
    
    return {
        'divisions': division_data,
        'winner_party': winner_party,
        'electors': electors,
        'votes_cast': votes_cast,
        'turnout_pct': turnout_pct
    }


def calculate_division_swings(div_data_2016: Dict, div_data_2021: Dict) -> Dict:
    """Calculate swing for each polling division."""
    swings = {}
    
    divs_2016 = div_data_2016.get('divisions', {})
    divs_2021 = div_data_2021.get('divisions', {})
    
    all_divs = set(divs_2016.keys()) | set(divs_2021.keys())
    
    for div in all_divs:
        data_2016 = divs_2016.get(div, {})
        data_2021 = divs_2021.get(div, {})
        
        slp_pct_2016 = data_2016.get('slp_pct', 0)
        slp_pct_2021 = data_2021.get('slp_pct', 0)
        uwp_pct_2016 = data_2016.get('uwp_pct', 0)
        uwp_pct_2021 = data_2021.get('uwp_pct', 0)
        
        # Calculate swing
        slp_swing = slp_pct_2021 - slp_pct_2016
        
        # Determine swing direction
        if slp_swing > 0:
            swing_towards = "SLP"
            swing_magnitude = slp_swing
        else:
            swing_towards = "UWP"
            swing_magnitude = abs(slp_swing)
        
        # Turnout change (constituency level)
        turnout_change = div_data_2021.get('turnout_pct', 0) - div_data_2016.get('turnout_pct', 0)
        
        swings[div] = {
            'swing_towards': swing_towards,
            'swing_magnitude': swing_magnitude,
            'slp_swing': slp_swing,
            'year2_slp_pct': slp_pct_2021,
            'year2_uwp_pct': uwp_pct_2021,
            'year2_ind_pct': data_2021.get('ind_pct', 0),
            'year1_slp_pct': slp_pct_2016,
            'year1_uwp_pct': uwp_pct_2016,
            'turnout_change': turnout_change,
            'year2_total': data_2021.get('total_votes', 0),
            'year1_total': data_2016.get('total_votes', 0)
        }
    
    return swings


def load_constituency_mapping() -> Dict:
    """Load constituency mapping data."""
    mapping_file = Path("data/constituency_maps/saint_lucia_constituencies.json")
    if not mapping_file.exists():
        return {}
    
    with open(mapping_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    mapping = {}
    for item in data:
        constituency_name = item.get("constituency", "")
        normalized = normalize_constituency_name(constituency_name)
        mapping[normalized] = {
            "original_name": constituency_name,
            "polling_divisions": item.get("polling_divisions", [])
        }
    
    return mapping


def get_communities_for_division(constituency_name: str, division_code: str, mapping: Dict) -> List[str]:
    """Get communities for a polling division."""
    normalized_const = normalize_constituency_name(constituency_name)
    
    for map_name, map_data in mapping.items():
        if normalized_const in map_name or map_name in normalized_const:
            polling_divs = map_data.get("polling_divisions", [])
            for div_info in polling_divs:
                div_name = div_info.get("division", "")
                normalized_div = normalize_polling_division_name(div_name)
                normalized_code = normalize_polling_division_name(division_code)
                
                if normalized_div == normalized_code or normalized_code in normalized_div:
                    communities = div_info.get("communities", [])
                    return [c for c in communities 
                           if c.upper() not in ["COMMUNITIES", "POLLING DIVISION"]]
    
    return []


def create_swing_chart(swing_data: Dict, winner_swings: List, constituency_name: str, 
                       year1: str, year2: str, winner_party: str, mapping: Dict):
    """Create a chart highlighting divisions with greatest swing towards winner."""
    if not winner_swings:
        print("\nNo divisions found with swing towards winner.")
        return
    
    divisions = []
    swings = []
    colors = []
    communities_list = []
    vote_splits = []
    
    for div, data in winner_swings:
        divisions.append(div)
        swings.append(data['swing_magnitude'])
        colors.append("#FF6B6B" if winner_party == "SLP" else "#4ECDC4")
        
        communities = get_communities_for_division(constituency_name, div, mapping)
        if communities:
            comm_str = ", ".join(communities[:3])
            if len(communities) > 3:
                comm_str += f" (+{len(communities)-3})"
            communities_list.append(comm_str)
        else:
            communities_list.append("")
        
        vote_splits.append(f"SLP: {data['year2_slp_pct']:.1f}% | UWP: {data['year2_uwp_pct']:.1f}%")
    
    fig, ax = plt.subplots(figsize=(14, max(8, len(divisions) * 0.9)))
    
    y_pos = range(len(divisions))
    bars = ax.barh(y_pos, swings, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    # Add labels
    for i, (bar, swing, split) in enumerate(zip(bars, swings, vote_splits)):
        ax.text(swing + max(swings) * 0.02, i, f'+{swing:.1f}%',
                va='center', fontweight='bold', fontsize=11)
        ax.text(max(swings) * 0.50, i, split,
                va='center', fontsize=9, style='italic')
    
    # Y-axis labels with communities
    y_labels = []
    for div, comm in zip(divisions, communities_list):
        if comm:
            y_labels.append(f"{div} - {comm}")
        else:
            y_labels.append(div)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_labels, fontsize=10)
    ax.set_xlabel('Swing Towards Winner (%)', fontsize=13, fontweight='bold')
    ax.set_title(f'{constituency_name}: Polling Divisions with Greatest Swing Towards {winner_party}\n'
                 f'({year1} → {year2})', fontsize=15, fontweight='bold', pad=20)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_xlim(0, max(swings) * 1.3)
    
    # Legend
    party_color = "#FF6B6B" if winner_party == "SLP" else "#4ECDC4"
    ax.barh([], [], color=party_color, label=f'Swing towards {winner_party}', alpha=0.7)
    ax.legend(loc='lower right', fontsize=11)
    
    plt.tight_layout()
    
    output_dir = Path("analysis")
    output_dir.mkdir(exist_ok=True)
    safe_const_name = constituency_name.replace(" ", "_").replace("/", "_")
    output_file = output_dir / f"{safe_const_name}_swing_analysis_{year1}_{year2}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nChart saved to: {output_file}")
    plt.close()


def analyze_independent_impact(div_data_2011: Dict, div_data_2016: Dict, mapping: Dict) -> Dict:
    """
    Analyze which party was most affected by the independent candidate in 2011.
    Compares vote share changes from 2011 to 2016 when IND disappeared.
    """
    constituency_name = "Micoud North"
    impact_analysis = {}
    
    divs_2011 = div_data_2011.get('divisions', {})
    divs_2016 = div_data_2016.get('divisions', {})
    
    all_divs = set(divs_2011.keys()) | set(divs_2016.keys())
    
    for div in all_divs:
        data_2011 = divs_2011.get(div, {})
        data_2016 = divs_2016.get(div, {})
        
        ind_pct_2011 = data_2011.get('ind_pct', 0)
        slp_pct_2011 = data_2011.get('slp_pct', 0)
        uwp_pct_2011 = data_2011.get('uwp_pct', 0)
        
        slp_pct_2016 = data_2016.get('slp_pct', 0)
        uwp_pct_2016 = data_2016.get('uwp_pct', 0)
        
        # Calculate vote share changes
        slp_change = slp_pct_2016 - slp_pct_2011
        uwp_change = uwp_pct_2016 - uwp_pct_2011
        
        # Determine which party gained more from IND disappearance
        if uwp_change > slp_change:
            primary_beneficiary = "UWP"
            benefit_margin = uwp_change - slp_change
        else:
            primary_beneficiary = "SLP"
            benefit_margin = slp_change - uwp_change
        
        impact_analysis[div] = {
            'ind_pct_2011': ind_pct_2011,
            'slp_pct_2011': slp_pct_2011,
            'uwp_pct_2011': uwp_pct_2011,
            'slp_pct_2016': slp_pct_2016,
            'uwp_pct_2016': uwp_pct_2016,
            'slp_change': slp_change,
            'uwp_change': uwp_change,
            'primary_beneficiary': primary_beneficiary,
            'benefit_margin': benefit_margin,
            'votes_2011': data_2011.get('total_votes', 0),
            'votes_2016': data_2016.get('total_votes', 0)
        }
    
    return impact_analysis


def create_independent_analysis_chart(impact_analysis: Dict, mapping: Dict):
    """Create chart showing independent candidate strength and party impact."""
    constituency_name = "Micoud North"
    
    # Sort by IND percentage in 2011 (strongest first)
    sorted_data = sorted(
        impact_analysis.items(),
        key=lambda x: x[1]['ind_pct_2011'],
        reverse=True
    )
    
    divisions = []
    ind_percentages = []
    slp_changes = []
    uwp_changes = []
    communities_list = []
    
    for div, data in sorted_data:
        divisions.append(div)
        ind_percentages.append(data['ind_pct_2011'])
        slp_changes.append(data['slp_change'])
        uwp_changes.append(data['uwp_change'])
        
        communities = get_communities_for_division(constituency_name, div, mapping)
        if communities:
            comm_str = ", ".join(communities[:3])
            if len(communities) > 3:
                comm_str += f" (+{len(communities)-3})"
            communities_list.append(comm_str)
        else:
            communities_list.append("")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, max(8, len(divisions) * 0.9)))
    
    y_pos = range(len(divisions))
    
    # Chart 1: Independent candidate strength in 2011
    bars1 = ax1.barh(y_pos, ind_percentages, color='#9B59B6', alpha=0.7, edgecolor='black', linewidth=1.5)
    
    for i, (bar, pct) in enumerate(zip(bars1, ind_percentages)):
        ax1.text(pct + max(ind_percentages) * 0.02, i, f'{pct:.1f}%',
                va='center', fontweight='bold', fontsize=11)
    
    y_labels = []
    for div, comm in zip(divisions, communities_list):
        if comm:
            y_labels.append(f"{div} - {comm}")
        else:
            y_labels.append(div)
    
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(y_labels, fontsize=10)
    ax1.set_xlabel('Independent Candidate Vote Share (%)', fontsize=12, fontweight='bold')
    ax1.set_title(f'{constituency_name}: Independent Candidate Strength (2011)\n'
                 f'Ranked by Vote Share', fontsize=14, fontweight='bold', pad=15)
    ax1.grid(axis='x', alpha=0.3, linestyle='--')
    ax1.set_xlim(0, max(ind_percentages) * 1.3)
    
    # Chart 2: Party gains when IND disappeared (2011 → 2016)
    x = range(len(divisions))
    width = 0.35
    
    bars2a = ax2.barh([i - width/2 for i in y_pos], slp_changes, width,
                      label='SLP Gain', color='#FF6B6B', alpha=0.7, edgecolor='black', linewidth=1)
    bars2b = ax2.barh([i + width/2 for i in y_pos], uwp_changes, width,
                      label='UWP Gain', color='#4ECDC4', alpha=0.7, edgecolor='black', linewidth=1)
    
    # Add value labels
    for i, (slp_chg, uwp_chg) in enumerate(zip(slp_changes, uwp_changes)):
        if slp_chg != 0:
            ax2.text(slp_chg + (2 if slp_chg > 0 else -2), i - width/2, f'{slp_chg:+.1f}pp',
                    va='center', fontsize=9, fontweight='bold')
        if uwp_chg != 0:
            ax2.text(uwp_chg + (2 if uwp_chg > 0 else -2), i + width/2, f'{uwp_chg:+.1f}pp',
                    va='center', fontsize=9, fontweight='bold')
    
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(y_labels, fontsize=10)
    ax2.set_xlabel('Vote Share Change (Percentage Points)', fontsize=12, fontweight='bold')
    ax2.set_title(f'{constituency_name}: Party Gains After IND Disappeared\n'
                 f'(2011 → 2016)', fontsize=14, fontweight='bold', pad=15)
    ax2.axvline(x=0, color='black', linestyle='-', linewidth=1, alpha=0.3)
    ax2.grid(axis='x', alpha=0.3, linestyle='--')
    ax2.legend(loc='lower right', fontsize=11)
    
    # Set x-axis limits
    all_changes = slp_changes + uwp_changes
    x_min = min(all_changes) - 2
    x_max = max(all_changes) + 5
    ax2.set_xlim(x_min, x_max)
    
    plt.tight_layout()
    
    output_dir = Path("analysis")
    output_dir.mkdir(exist_ok=True)
    safe_const_name = constituency_name.replace(" ", "_").replace("/", "_")
    output_file = output_dir / f"{safe_const_name}_independent_impact_2011.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nChart saved to: {output_file}")
    plt.close()


def main():
    """Main analysis function - extended to include 2011."""
    constituency_name = "Micoud North"
    
    print(f"\n{'='*80}")
    print(f"EXTENDED POLLING DIVISION SWING ANALYSIS: {constituency_name}")
    print(f"Analyzing 2011, 2016, and 2021 elections")
    print(f"{'='*80}\n")
    
    # Load data for all three elections
    results_2011_file = Path("data/election_night_results/st_lucia_2011_full_results.json")
    results_2016_file = Path("data/election_night_results/st_lucia_2016_full_results.json")
    results_2021_file = Path("data/election_night_results/st_lucia_2021_full_results.json")
    
    with open(results_2011_file, 'r', encoding='utf-8') as f:
        results_2011 = json.load(f)
    
    with open(results_2016_file, 'r', encoding='utf-8') as f:
        results_2016 = json.load(f)
    
    with open(results_2021_file, 'r', encoding='utf-8') as f:
        results_2021 = json.load(f)
    
    # Extract polling division data
    print("Extracting polling division data...")
    div_data_2011 = extract_polling_division_data(results_2011, constituency_name)
    div_data_2016 = extract_polling_division_data(results_2016, constituency_name)
    div_data_2021 = extract_polling_division_data(results_2021, constituency_name)
    
    if not div_data_2011 or not div_data_2016 or not div_data_2021:
        print(f"Error: Could not extract data for {constituency_name}")
        return
    
    print(f"\nElection Results Summary:")
    print(f"  2011 Winner: {div_data_2011.get('winner_party')} (Turnout: {div_data_2011.get('turnout_pct'):.2f}%)")
    print(f"  2016 Winner: {div_data_2016.get('winner_party')} (Turnout: {div_data_2016.get('turnout_pct'):.2f}%)")
    print(f"  2021 Winner: {div_data_2021.get('winner_party')} (Turnout: {div_data_2021.get('turnout_pct'):.2f}%)")
    
    # Load mapping
    mapping = load_constituency_mapping()
    
    # ========== ANALYSIS 1: 2011 Independent Candidate Impact ==========
    print(f"\n\n{'='*80}")
    print(f"ANALYSIS 1: 2011 Independent Candidate Impact")
    print(f"{'='*80}\n")
    
    impact_analysis = analyze_independent_impact(div_data_2011, div_data_2016, mapping)
    
    print("Independent Candidate Strength by Polling Division (2011):\n")
    print("-" * 100)
    
    # Sort by IND percentage
    sorted_impact = sorted(
        impact_analysis.items(),
        key=lambda x: x[1]['ind_pct_2011'],
        reverse=True
    )
    
    total_ind_votes = 0
    total_slp_gain = 0
    total_uwp_gain = 0
    
    for div, data in sorted_impact:
        # Get actual IND votes from 2011 data
        ind_votes_2011 = div_data_2011['divisions'][div].get('ind_votes', 0) if div in div_data_2011.get('divisions', {}) else 0
        total_ind_votes += ind_votes_2011
        
        print(f"\nDivision: {div}")
        ind_votes_2011 = div_data_2011['divisions'][div].get('ind_votes', 0) if div in div_data_2011.get('divisions', {}) else 0
        print(f"  Independent Candidate: {data['ind_pct_2011']:.1f}% ({ind_votes_2011} votes)")
        print(f"  2011 Results: SLP {data['slp_pct_2011']:.1f}% | UWP {data['uwp_pct_2011']:.1f}% | IND {data['ind_pct_2011']:.1f}%")
        print(f"  2016 Results: SLP {data['slp_pct_2016']:.1f}% | UWP {data['uwp_pct_2016']:.1f}%")
        print(f"  Vote Share Change (2011 → 2016):")
        print(f"    SLP: {data['slp_change']:+.1f} percentage points")
        print(f"    UWP: {data['uwp_change']:+.1f} percentage points")
        print(f"  Primary Beneficiary: {data['primary_beneficiary']} (gained {data['benefit_margin']:.1f}pp more)")
        
        total_slp_gain += data['slp_change']
        total_uwp_gain += data['uwp_change']
        
        communities = get_communities_for_division(constituency_name, div, mapping)
        if communities:
            print(f"  Communities: {', '.join(communities[:8])}")
            if len(communities) > 8:
                print(f"              ({len(communities) - 8} more...)")
    
    print(f"\n{'='*80}")
    print("SUMMARY:")
    print(f"{'='*80}")
    print(f"Total Independent votes in 2011: {total_ind_votes:,}")
    print(f"\nAverage vote share change when IND disappeared (2011 → 2016):")
    num_divs = len(sorted_impact)
    avg_slp_gain = total_slp_gain / num_divs
    avg_uwp_gain = total_uwp_gain / num_divs
    print(f"  SLP: {avg_slp_gain:+.2f} percentage points")
    print(f"  UWP: {avg_uwp_gain:+.2f} percentage points")
    
    if avg_uwp_gain > avg_slp_gain:
        print(f"\nCONCLUSION: UWP was the primary beneficiary when the independent candidate")
        print(f"disappeared, gaining on average {avg_uwp_gain - avg_slp_gain:.2f} percentage points")
        print(f"more than SLP across all polling divisions.")
    else:
        print(f"\nCONCLUSION: SLP was the primary beneficiary when the independent candidate")
        print(f"disappeared, gaining on average {avg_slp_gain - avg_uwp_gain:.2f} percentage points")
        print(f"more than UWP across all polling divisions.")
    
    # Create chart for independent analysis
    print("\n\nCreating visualization for independent candidate impact...")
    create_independent_analysis_chart(impact_analysis, mapping)
    
    # ========== ANALYSIS 2: 2016-2021 Swing Analysis ==========
    print(f"\n\n{'='*80}")
    print(f"ANALYSIS 2: 2016-2021 Swing Analysis")
    print(f"{'='*80}\n")
    
    winner_party = div_data_2021.get('winner_party')
    print(f"Winner in 2021: {winner_party}\n")
    
    # Calculate swings
    print("Calculating swings...")
    swing_data = calculate_division_swings(div_data_2016, div_data_2021)
    
    # Filter divisions with swing towards winner
    winner_swings = [
        (div, data) for div, data in swing_data.items()
        if data['swing_towards'] == winner_party
    ]
    winner_swings.sort(key=lambda x: x[1]['swing_magnitude'], reverse=True)
    
    # Print results
    print(f"\nPolling Divisions with Greatest Swing Towards {winner_party}:\n")
    print("-" * 80)
    
    for div, data in winner_swings:
        print(f"\nDivision: {div}")
        print(f"  Swing: +{data['swing_magnitude']:.1f}% towards {winner_party}")
        print(f"  2021 Result: {data['year2_slp_pct']:.1f}% SLP vs {data['year2_uwp_pct']:.1f}% UWP")
        if data['year2_ind_pct'] > 0.1:
            print(f"  2021 Independent: {data['year2_ind_pct']:.1f}%")
        print(f"  Turnout Change: {data['turnout_change']:+.1f}%")
        print(f"  Votes 2016: {data['year1_total']} → 2021: {data['year2_total']}")
        
        communities = get_communities_for_division(constituency_name, div, mapping)
        if communities:
            print(f"  Communities: {', '.join(communities[:8])}")
            if len(communities) > 8:
                print(f"              ({len(communities) - 8} more...)")
    
    # Create chart
    print("\n\nCreating visualization for 2016-2021 swing...")
    create_swing_chart(swing_data, winner_swings, constituency_name, "2016", "2021", winner_party, mapping)
    
    print(f"\n{'='*80}")
    print("Analysis complete!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

