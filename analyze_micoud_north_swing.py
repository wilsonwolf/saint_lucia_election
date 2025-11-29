#!/usr/bin/env python3
"""
Analyze polling division swings for Micoud North between 2016 and 2021 elections.
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
        
        # Check if this is our constituency
        if normalized_const in normalized_district or normalized_district in normalized_const:
            found_constituency = True
            if "Party" in record and record.get("Candidate"):
                constituency_records.append(record)
                current_constituency = district
            elif "summary_label" in record and current_constituency:
                label = record.get("summary_label", "")
                value = record.get("summary_value", "")
                if district == current_constituency:
                    summary_data[label] = value
    
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


def main():
    """Main analysis function."""
    constituency_name = "Micoud North"
    year1 = "2016"
    year2 = "2021"
    
    print(f"\n{'='*80}")
    print(f"POLLING DIVISION SWING ANALYSIS: {constituency_name}")
    print(f"Comparing {year1} vs {year2}")
    print(f"{'='*80}\n")
    
    # Load data
    results_2016_file = Path(f"data/election_night_results/st_lucia_{year1}_full_results.json")
    results_2021_file = Path(f"data/election_night_results/st_lucia_{year2}_full_results.json")
    
    with open(results_2016_file, 'r', encoding='utf-8') as f:
        results_2016 = json.load(f)
    
    with open(results_2021_file, 'r', encoding='utf-8') as f:
        results_2021 = json.load(f)
    
    # Extract polling division data
    print("Extracting polling division data...")
    div_data_2016 = extract_polling_division_data(results_2016, constituency_name)
    div_data_2021 = extract_polling_division_data(results_2021, constituency_name)
    
    if not div_data_2016 or not div_data_2021:
        print(f"Error: Could not extract data for {constituency_name}")
        return
    
    winner_party = div_data_2021.get('winner_party')
    print(f"Winner in {year2}: {winner_party}\n")
    
    # Calculate swings
    print("Calculating swings...")
    swing_data = calculate_division_swings(div_data_2016, div_data_2021)
    
    # Filter divisions with swing towards winner
    winner_swings = [
        (div, data) for div, data in swing_data.items()
        if data['swing_towards'] == winner_party
    ]
    winner_swings.sort(key=lambda x: x[1]['swing_magnitude'], reverse=True)
    
    # Load mapping
    mapping = load_constituency_mapping()
    
    # Print results
    print(f"\nPolling Divisions with Greatest Swing Towards {winner_party}:\n")
    print("-" * 80)
    
    for div, data in winner_swings:
        print(f"\nDivision: {div}")
        print(f"  Swing: +{data['swing_magnitude']:.1f}% towards {winner_party}")
        print(f"  {year2} Result: {data['year2_slp_pct']:.1f}% SLP vs {data['year2_uwp_pct']:.1f}% UWP")
        if data['year2_ind_pct'] > 0.1:
            print(f"  {year2} Independent: {data['year2_ind_pct']:.1f}%")
        print(f"  Turnout Change: {data['turnout_change']:+.1f}%")
        print(f"  Votes {year1}: {data['year1_total']} → {year2}: {data['year2_total']}")
        
        communities = get_communities_for_division(constituency_name, div, mapping)
        if communities:
            print(f"  Communities: {', '.join(communities[:8])}")
            if len(communities) > 8:
                print(f"              ({len(communities) - 8} more...)")
    
    # Create chart
    print("\n\nCreating visualization...")
    create_swing_chart(swing_data, winner_swings, constituency_name, year1, year2, winner_party, mapping)
    
    print(f"\n{'='*80}")
    print("Analysis complete!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

