#!/usr/bin/env python3
"""
Competitive Areas Analysis

Analyzes polling division swings and competitive areas for constituencies
between elections. Identifies areas with significant swings and maps them
to communities.
"""

import json
import re
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
    # Add space after "AnseLa" to match "Anse La"
    name = name.replace("AnseLa", "Anse La")
    # Add space before capital letters (e.g., "DenneryNorth" -> "Dennery North")
    name = re.sub(r'([a-z])([A-Z])', r'\1 \2', name)
    name = " ".join(name.split())  # Normalize whitespace
    return name.upper()


def normalize_polling_division_name(div_name: str) -> str:
    """Normalize polling division name for matching."""
    if not div_name:
        return ""
    div_name = div_name.upper().strip()
    div_name = div_name.replace(" ", "").replace("(", "").replace(")", "")
    return div_name


def load_election_night_results(year: str) -> List[Dict]:
    """Load election night results for a specific year."""
    results_file = Path(f"data/election_night_results/st_lucia_{year}_full_results.json")
    if not results_file.exists():
        return []
    
    with open(results_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data


def extract_polling_division_data(results: List[Dict], constituency_name: str) -> Dict:
    """
    Extract polling division-level data for a constituency.
    
    Returns dict with:
    - divisions: Dict mapping division codes to vote data
    - winner_party: Winning party in this election
    - electors: Number of registered electors
    - votes_cast: Total votes cast
    - turnout_pct: Turnout percentage
    """
    normalized_const = normalize_constituency_name(constituency_name)
    
    constituency_records = []
    summary_data = {}
    current_constituency = None
    found_constituency = False
    
    for record in results:
        district = record.get('district', '')
        normalized_district = normalize_constituency_name(district)
        
        # Check if this is our constituency - use exact matching to avoid substring issues
        # (e.g., "Castries South" should NOT match "Castries South East")
        matches = False
        
        # Exact match
        if normalized_const == normalized_district:
            matches = True
        # Check if they're the same when spaces are removed (handles variations like "AnseLa Raye" vs "Anse La Raye")
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
    
    # Get polling division columns (exclude metadata)
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
    
    # Calculate percentages for each division
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


def calculate_division_swings(div_data_year1: Dict, div_data_year2: Dict, winner_party: str) -> Dict:
    """
    Calculate swing for each polling division between two elections.
    
    Returns dict mapping division codes to swing data including:
    - swing_value: Positive if towards winner, negative if away from winner
    - swing_towards: Party that gained (SLP or UWP)
    - swing_magnitude: Absolute percentage point change
    - year2_slp_pct, year2_uwp_pct: Final percentages
    - turnout_change: Change in constituency-level turnout
    """
    swings = {}
    
    divs_year1 = div_data_year1.get('divisions', {})
    divs_year2 = div_data_year2.get('divisions', {})
    
    all_divs = set(divs_year1.keys()) | set(divs_year2.keys())
    
    for div in all_divs:
        data_year1 = divs_year1.get(div, {})
        data_year2 = divs_year2.get(div, {})
        
        slp_pct_year1 = data_year1.get('slp_pct', 0)
        slp_pct_year2 = data_year2.get('slp_pct', 0)
        uwp_pct_year1 = data_year1.get('uwp_pct', 0)
        uwp_pct_year2 = data_year2.get('uwp_pct', 0)
        
        # Calculate swing (change in SLP percentage)
        slp_swing = slp_pct_year2 - slp_pct_year1
        
        # Determine swing direction
        if slp_swing > 0:
            swing_towards = "SLP"
        else:
            swing_towards = "UWP"
        
        # Calculate swing value relative to winner
        # Positive = towards winner, Negative = away from winner
        if winner_party == "SLP":
            swing_value = slp_swing  # Positive if SLP gained, negative if UWP gained
        else:
            swing_value = -slp_swing  # Positive if UWP gained (SLP lost), negative if SLP gained
        
        # Turnout change (constituency level, since we don't have division-level electors)
        turnout_change = div_data_year2.get('turnout_pct', 0) - div_data_year1.get('turnout_pct', 0)
        
        swings[div] = {
            'swing_value': swing_value,  # Positive = towards winner, negative = away
            'swing_towards': swing_towards,
            'swing_magnitude': abs(swing_value),
            'slp_swing': slp_swing,
            'year2_slp_pct': slp_pct_year2,
            'year2_uwp_pct': uwp_pct_year2,
            'year2_ind_pct': data_year2.get('ind_pct', 0),
            'year1_slp_pct': slp_pct_year1,
            'year1_uwp_pct': uwp_pct_year1,
            'turnout_change': turnout_change,
            'year2_total': data_year2.get('total_votes', 0),
            'year1_total': data_year1.get('total_votes', 0)
        }
    
    return swings


def load_constituency_mapping() -> Dict:
    """Load constituency mapping data from JSON file."""
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
    """Get communities for a polling division from the mapping data."""
    normalized_const = normalize_constituency_name(constituency_name)
    normalized_code = normalize_polling_division_name(division_code)
    
    # Get the letter prefix from the division code (e.g., "Q1" -> "Q", "B1(a)" -> "B")
    division_letter = division_code.strip()[0] if division_code else None
    
    # Try to find matching constituency
    # Remove common suffixes like "(H)", "- (B)" from normalized name for matching
    const_base = normalized_const
    # Remove patterns like "(H)", "- (B)", etc.
    import re
    const_base = re.sub(r'\s*[-\-]\s*\([A-Z]\)\s*$', '', const_base)
    const_base = re.sub(r'\s*\([A-Z]\)\s*$', '', const_base)
    const_base = const_base.strip()
    
    matched_constituency = None
    
    # Try exact match first (accounting for suffixes)
    for map_name, map_data in mapping.items():
        map_base = map_name
        map_base = re.sub(r'\s*[-\-]\s*\([A-Z]\)\s*$', '', map_base)
        map_base = re.sub(r'\s*\([A-Z]\)\s*$', '', map_base)
        map_base = map_base.strip()
        
        if const_base == map_base:
            matched_constituency = (map_name, map_data)
            break
    
    # Try partial match if exact match failed
    if not matched_constituency:
        const_words = set(normalized_const.split())
        best_score = 0
        
        for map_name, map_data in mapping.items():
            map_words = set(map_name.split())
            # Calculate overlap score
            overlap = len(const_words & map_words)
            if overlap > best_score and overlap >= 2:  # Need at least 2 words to match
                best_score = overlap
                matched_constituency = (map_name, map_data)
    
    # If still no match and we have a division letter, try matching by letter prefix
    if not matched_constituency and division_letter:
        for map_name, map_data in mapping.items():
            # Check if this constituency uses divisions with the same letter prefix
            polling_divs = map_data.get("polling_divisions", [])
            for div_info in polling_divs:
                div_name = div_info.get("division", "")
                if div_name and len(div_name.strip()) > 0:
                    first_char = div_name.strip()[0]
                    if first_char.isalpha() and first_char.isupper() and first_char == division_letter:
                        matched_constituency = (map_name, map_data)
                        break
                if matched_constituency:
                    break
    
    # If we found a match, look for the division
    if matched_constituency:
        map_name, map_data = matched_constituency
        polling_divs = map_data.get("polling_divisions", [])
        for div_info in polling_divs:
            div_name = div_info.get("division", "")
            normalized_div = normalize_polling_division_name(div_name)
            
            # Check if division codes match
            if normalized_div == normalized_code or normalized_code in normalized_div:
                communities = div_info.get("communities", [])
                # Filter out header rows
                return [c for c in communities 
                       if c.upper() not in ["COMMUNITIES", "POLLING DIVISION", "POLLING STATIONS", ""]]
    
    return []


def wrap_text(text: str, max_width: int) -> str:
    """Wrap text to fit within max_width characters."""
    if len(text) <= max_width:
        return text
    
    words = text.split(", ")
    lines = []
    current_line = ""
    
    for word in words:
        if not current_line:
            current_line = word
        elif len(current_line) + len(", ") + len(word) <= max_width:
            current_line += ", " + word
        else:
            lines.append(current_line)
            current_line = word
    
    if current_line:
        lines.append(current_line)
    
    return "\n".join(lines)


def create_swing_chart(swing_data: Dict, all_swings: List, constituency_name: str, 
                       year1: str, year2: str, winner_party: str, mapping: Dict):
    """
    Create a chart showing all polling divisions with swings towards and away from the winner.
    
    The chart shows:
    - Division codes with communities (wrapped to fit)
    - Swing values: positive (towards winner) and negative (away from winner)
    - Final vote percentages for each division
    """
    if not all_swings:
        print("\nNo polling division data found.")
        return
    
    divisions = []
    swings = []
    colors = []
    communities_list = []
    vote_splits = []
    
    # Determine color scheme
    winner_color = "#FF6B6B" if winner_party == "SLP" else "#4ECDC4"
    loser_color = "#4ECDC4" if winner_party == "SLP" else "#FF6B6B"
    
    for div, data in all_swings:
        divisions.append(div)
        swing_value = data['swing_value']
        swings.append(swing_value)
        
        # Color: green for positive (towards winner), red for negative (away from winner)
        if swing_value > 0:
            colors.append(winner_color)
        else:
            colors.append(loser_color)
        
        communities = get_communities_for_division(constituency_name, div, mapping)
        if communities:
            # Include more communities, but wrap them
            comm_str = ", ".join(communities)
            communities_list.append(comm_str)
        else:
            communities_list.append("")
        
        vote_splits.append(f"SLP: {data['year2_slp_pct']:.1f}% | UWP: {data['year2_uwp_pct']:.1f}%")
    
    # Calculate figure size - match original sizing approach exactly
    fig, ax = plt.subplots(figsize=(14, max(8, len(divisions) * 0.9)))
    
    y_pos = range(len(divisions))
    bars = ax.barh(y_pos, swings, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    # Add a vertical line at 0
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    
    # Calculate x-axis range
    max_abs_swing = max(abs(s) for s in swings) if swings else 1
    if len(swings) > 1:
        x_range = max(swings) - min(swings)
        x_min = min(swings) - x_range * 0.15
        x_max = max(swings) + x_range * 0.35  # Extra padding on right for swing label
    else:
        x_min = -max_abs_swing * 0.2
        x_max = max_abs_swing * 1.4
    
    # Position swing labels at the absolute right edge (outside the chart area)
    # Position vote percentages to avoid overlap
    for i, (bar, swing, split) in enumerate(zip(bars, swings, vote_splits)):
        bar_width = abs(swing)
        
        # Swing magnitude label - ALWAYS at the absolute right edge of chart
        swing_label_x = x_max * 0.98  # Very close to right edge
        if swing > 0:
            swing_label_text = f'+{swing:.1f}%'
        else:
            swing_label_text = f'{swing:.1f}%'
        
        # Vote percentages - position to avoid overlap with swing label on right
        # Position them well before the right edge where swing labels will be
        if swing > 0:
            # Positive swing: position percentage at fixed position (like original)
            # Use a position that's well before the right edge
            pct_x = max_abs_swing * 0.5  # Fixed at 50% of max swing, before right edge
            if pct_x > swing:
                pct_x = swing * 0.5  # If fixed position is beyond bar, use bar center
            pct_ha = 'center'
            pct_color = 'white' if bar_width > max_abs_swing * 0.3 else 'black'
            pct_weight = 'bold' if pct_color == 'white' else 'normal'
        else:
            # Negative swing: position on left side (negative values)
            pct_x = swing * 0.5  # Middle of negative bar
            pct_ha = 'center'
            pct_color = 'white'
            pct_weight = 'bold'
        
        # Add swing magnitude label (always on the far right, outside chart area)
        ax.text(swing_label_x, i, swing_label_text,
                va='center', fontweight='bold', fontsize=11,
                ha='left', clip_on=False)  # clip_on=False allows text outside axes
        
        # Add vote percentage label (positioned to avoid overlap)
        ax.text(pct_x, i, split,
                va='center', fontsize=9, style='italic',
                ha=pct_ha, color=pct_color,
                weight=pct_weight)
    
    # Y-axis labels with communities - simpler format, wrap only if needed
    y_labels = []
    max_label_width = 60  # Characters before wrapping
    
    for div, comm in zip(divisions, communities_list):
        if comm:
            # Try single line first, wrap only if too long
            single_line = f"{div} - {comm}"
            if len(single_line) > max_label_width:
                # Wrap only if necessary
                wrapped_comm = wrap_text(comm, max_label_width - len(div) - 3)
                label = f"{div}\n{wrapped_comm}" if wrapped_comm else div
            else:
                label = single_line
        else:
            label = div
        y_labels.append(label)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_labels, fontsize=10, ha='right')
    
    # Adjust left margin to accommodate wrapped labels
    ax.set_xlabel('Swing (Positive = Towards Winner, Negative = Away from Winner) (%)', 
                  fontsize=12, fontweight='bold')
    ax.set_title(f'{constituency_name}: All Polling Divisions by Swing\n'
                 f'({year1} → {year2}, Winner: {winner_party})', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Set x-axis limits with padding
    ax.set_xlim(x_min, x_max)
    
    # Legend
    ax.barh([], [], color=winner_color, label=f'Towards {winner_party}', alpha=0.7)
    ax.barh([], [], color=loser_color, label=f'Away from {winner_party}', alpha=0.7)
    ax.legend(loc='lower right', fontsize=10)
    
    # Adjust layout
    plt.tight_layout()
    
    output_dir = Path("analysis")
    output_dir.mkdir(exist_ok=True)
    safe_const_name = constituency_name.replace(" ", "_").replace("/", "_")
    output_file = output_dir / f"{safe_const_name}_swing_analysis_{year1}_{year2}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nChart saved to: {output_file}")
    plt.close()


def analyze_polling_division_swing(constituency_name: str, year1: str, year2: str):
    """
    Analyze polling division swings for a specific constituency between two elections.
    
    Args:
        constituency_name: Name of the constituency (e.g., "Micoud North")
        year1: Earlier year (e.g., "2016")
        year2: Later year (e.g., "2021")
    
    Returns:
        Tuple of (swing_data dict, winner_swings list)
    """
    print(f"\n{'='*80}")
    print(f"POLLING DIVISION SWING ANALYSIS: {constituency_name}")
    print(f"Comparing {year1} vs {year2}")
    print(f"{'='*80}\n")
    
    # Load election night results for both years
    results_year1 = load_election_night_results(year1)
    results_year2 = load_election_night_results(year2)
    
    if not results_year1 or not results_year2:
        print(f"Error: Could not load election night results for {year1} or {year2}")
        return None, None
    
    # Extract polling division data for the constituency
    print("Extracting polling division data...")
    div_data_year1 = extract_polling_division_data(results_year1, constituency_name)
    div_data_year2 = extract_polling_division_data(results_year2, constituency_name)
    
    if not div_data_year1 or not div_data_year2:
        print(f"Error: Could not extract polling division data for {constituency_name}")
        return None, None
    
    winner_party = div_data_year2.get('winner_party')
    print(f"Winner in {year2}: {winner_party}\n")
    
    # Calculate swings
    print("Calculating swings...")
    swing_data = calculate_division_swings(div_data_year1, div_data_year2, winner_party)
    
    # Get all divisions, sorted by swing value (positive = towards winner, negative = away)
    all_swings = list(swing_data.items())
    all_swings.sort(key=lambda x: x[1]['swing_value'], reverse=True)
    
    # Load constituency mapping for communities
    mapping = load_constituency_mapping()
    
    # Calculate constituency-level turnout change (same for all divisions)
    constituency_turnout_change = div_data_year2.get('turnout_pct', 0) - div_data_year1.get('turnout_pct', 0)
    constituency_turnout_year1 = div_data_year1.get('turnout_pct', 0)
    constituency_turnout_year2 = div_data_year2.get('turnout_pct', 0)
    
    # Print analysis
    print(f"\nConstituency-Level Turnout:")
    print(f"  {year1}: {constituency_turnout_year1:.2f}%")
    print(f"  {year2}: {constituency_turnout_year2:.2f}%")
    print(f"  Change: {constituency_turnout_change:+.2f} percentage points\n")
    
    print(f"All Polling Divisions (Ranked by Swing):\n")
    print("-" * 80)
    
    for div, data in all_swings:
        swing_value = data['swing_value']
        if swing_value > 0:
            swing_desc = f"+{swing_value:.1f}% towards {winner_party}"
        else:
            swing_desc = f"{swing_value:.1f}% away from {winner_party} (towards {data['swing_towards']})"
        
        print(f"\nDivision: {div}")
        print(f"  Swing: {swing_desc}")
        print(f"  {year2} Result: {data['year2_slp_pct']:.1f}% SLP vs {data['year2_uwp_pct']:.1f}% UWP")
        if data['year2_ind_pct'] > 0.1:
            print(f"  {year2} Independent: {data['year2_ind_pct']:.1f}%")
        print(f"  Votes {year1}: {data['year1_total']} → {year2}: {data['year2_total']}")
        
        # Map to communities
        communities = get_communities_for_division(constituency_name, div, mapping)
        if communities:
            print(f"  Communities: {', '.join(communities[:8])}")
            if len(communities) > 8:
                print(f"              ({len(communities) - 8} more...)")
    
    # Create visualization
    print("\n\nCreating visualization...")
    create_swing_chart(swing_data, all_swings, constituency_name, year1, year2, winner_party, mapping)
    
    print(f"\n{'='*80}")
    print("Analysis complete!")
    print(f"{'='*80}\n")
    
    return swing_data, all_swings


def main():
    """Main function - run analysis for multiple constituencies."""
    constituencies = [
        "Soufriere",
        "Micoud North",
        "Micoud South",
        "Babonneau",
        "Anse La Raye/Canaries",
        "Castries South East",
        "Dennery North",
        "Dennery South"
    ]
    
    year1 = "2016"
    year2 = "2021"
    
    print(f"\n{'='*80}")
    print(f"COMPETITIVE AREAS ANALYSIS")
    print(f"Analyzing {len(constituencies)} constituencies: {year1} vs {year2}")
    print(f"{'='*80}\n")
    
    for i, constituency in enumerate(constituencies, 1):
        print(f"\n\n{'='*80}")
        print(f"CONSTITUENCY {i}/{len(constituencies)}: {constituency}")
        print(f"{'='*80}")
        try:
            analyze_polling_division_swing(constituency, year1, year2)
        except Exception as e:
            print(f"\nError analyzing {constituency}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n\n{'='*80}")
    print("ALL ANALYSES COMPLETE!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

