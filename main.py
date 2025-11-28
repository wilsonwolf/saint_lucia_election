"""
Main analysis script for Saint Lucia election data.
Loads and analyzes historical election results.
"""

import json
import os
import sys
from pathlib import Path
import pandas as pd

# Check for required dependencies
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError as e:
    print("=" * 60)
    print("ERROR: Required dependencies not found!")
    print("=" * 60)
    print(f"\nMissing module: {e.name if hasattr(e, 'name') else 'matplotlib/seaborn'}")
    print("\nPlease activate the virtual environment and install dependencies:")
    print("  source .venv/bin/activate")
    print("  uv pip install matplotlib seaborn")
    print("\nOr use the helper script:")
    print("  ./run.sh")
    print("=" * 60)
    sys.exit(1)

from typing import Dict, List, Tuple


def load_election_data(data_dir: str = "data") -> Dict[str, Dict]:
    """
    Load all election data files from the data directory subfolders.
    
    Returns:
        Dictionary with keys like '1979_summary', '1979_distribution', etc.
    """
    data_path = Path(data_dir)
    election_data = {}
    
    # Define subfolders and their corresponding types
    subfolders = {
        "summary_results": "summary",
        "vote_distribution": "distribution"
    }
    
    # Load files from each subfolder
    for subfolder, key_type in subfolders.items():
        subfolder_path = data_path / subfolder
        
        if not subfolder_path.exists():
            continue
            
        for file_path in subfolder_path.glob("*.json"):
            filename = file_path.stem
            
            # Skip duplicate files
            if " (1)" in filename:
                continue
                
            with open(file_path, 'r') as f:
                data = json.load(f)
                
            # Extract year
            parts = filename.replace("saint_lucia_", "").split("_")
            if parts[0].startswith("apr"):
                # Handle 1987 special cases - two separate elections
                # Extract just the date part (apr30 or apr6) without "summary" or "vote"
                date_part = parts[0]  # "apr30" or "apr6"
                year = "1987"
                # Use full date identifier to distinguish the two 1987 elections
                # Normalize to just the date (apr30 or apr6)
                year_identifier = f"1987_{date_part}"
                key = f"{year_identifier}_{key_type}"
            else:
                year = parts[0]
                year_identifier = year
                key = f"{year}_{key_type}"
                
            election_data[key] = {
                "year": year,
                "year_identifier": year_identifier,  # Unique identifier for each election
                "type": key_type,
                "data": data,
                "filename": filename
            }
    
    return election_data


def normalize_party_name(party_name: str) -> str:
    """
    Normalize party names to ensure consistency.
    Combines "Saint Lucia Labour Party" and "St. Lucia Labour Party" into a single entity.
    """
    if not party_name:
        return party_name
    
    # Normalize SLP variations
    party_upper = party_name.upper()
    if "ST. LUCIA LABOUR PARTY" in party_upper or "SAINT LUCIA LABOUR PARTY" in party_upper:
        return "Saint Lucia Labour Party"
    
    return party_name


def calculate_seats_from_constituencies(data: List[Dict]) -> Dict[str, int]:
    """
    Calculate seats won by each party from constituency-level data.
    Returns dictionary mapping party codes to seats won.
    """
    seats = {}
    exclude_fields = {"Constituency", "Registered Votes", "Votes Cast", "Turnout", "Rejected", "Total"}
    
    # Get all party codes from the data
    party_codes = set()
    for row in data:
        if row.get("Constituency", "").upper() == "TOTAL":
            continue
        for key in row.keys():
            if key not in exclude_fields:
                party_codes.add(key)
    
    # Initialize seat counts
    for code in party_codes:
        seats[code] = 0
    
    # Determine winner for each constituency
    for row in data:
        constituency = row.get("Constituency", "")
        if constituency.upper() == "TOTAL":
            continue
        
        # Find the party with the most votes in this constituency
        max_votes = -1
        winner = None
        
        for code in party_codes:
            val = row.get(code, 0)
            if isinstance(val, str):
                val = val.replace(",", "").replace("*", "").replace("–", "0").replace("\u2013", "0")
                try:
                    val = float(val) if "." in val else int(val)
                except (ValueError, TypeError):
                    val = 0
            elif val is None or (isinstance(val, float) and pd.isna(val)):
                val = 0
            
            if val > max_votes:
                max_votes = val
                winner = code
        
        if winner:
            seats[winner] = seats.get(winner, 0) + 1
    
    return seats


def compute_summary_from_distribution(data: List[Dict], year_identifier: str) -> List[Dict]:
    """
    Compute summary statistics from constituency-level distribution data.
    Returns list of summary records.
    """
    summaries = []
    
    # Find the TOTAL row
    total_row = None
    for row in data:
        if row.get("Constituency", "").upper() == "TOTAL":
            total_row = row
            break
    
    if not total_row:
        return summaries
    
    # Calculate seats from constituency data
    seats_won = calculate_seats_from_constituencies(data)
    
    # Get all party codes (exclude non-party fields)
    exclude_fields = {"Constituency", "Registered Votes", "Votes Cast", "Turnout", "Rejected", "Total"}
    party_codes = set()
    for key in total_row.keys():
        if key not in exclude_fields and total_row[key] not in [None, "–", "\u2013", "", "NaN"]:
            # Try to convert to number to verify it's a vote count
            try:
                val = total_row[key]
                if isinstance(val, str):
                    val = val.replace(",", "").replace("*", "")
                float(val)
                party_codes.add(key)
            except (ValueError, TypeError):
                pass
    
    # Calculate totals and percentages
    total_votes = 0
    party_totals = {}
    
    for code in party_codes:
        val = total_row.get(code, 0)
        if isinstance(val, str):
            val = val.replace(",", "").replace("*", "").replace("–", "0").replace("\u2013", "0")
            try:
                val = float(val) if "." in val else int(val)
            except ValueError:
                val = 0
        elif val is None or (isinstance(val, float) and pd.isna(val)):
            val = 0
        party_totals[code] = val
        total_votes += val
    
    # Create summary records
    party_names = {
        "SLP": "Saint Lucia Labour Party",
        "UWP": "United Workers Party",
        "PLP": "Progressive Labour Party",
        "NA": "National Alliance",
        "NDM": "National Democratic Movement",
        "LPM": "Lucian People's Movement",
        "LG": "Lucian Greens",
        "IND": "Independent",
        "NGP": "National Green Party"
    }
    
    for code, votes in party_totals.items():
        if votes > 0:
            pct = (votes / total_votes * 100) if total_votes > 0 else 0
            party_name = normalize_party_name(party_names.get(code, code))
            summaries.append({
                "Year": year_identifier,
                "Party": party_name,
                "Code": code,
                "Total_Votes": int(votes),
                "Percent_Votes": round(pct, 2),
                "Seats": seats_won.get(code, 0),
                "Candidates": 0  # Not available from distribution data
            })
    
    return summaries


def create_summary_dataframe(election_data: Dict) -> pd.DataFrame:
    """
    Create a pandas DataFrame with summary results across all elections.
    Prioritizes summary format data, falls back to computing from constituency data.
    """
    summaries = []
    processed_elections = set()  # Track which elections we've already processed
    
    # First pass: collect all summary-format data
    # Normalize election identifiers to avoid processing same election twice
    summary_format_data = {}
    constituency_format_data = {}
    
    for key, info in election_data.items():
        year_identifier = info.get("year_identifier", info["year"])
        data = info["data"]
        
        if isinstance(data, list) and len(data) > 0:
            first_item = data[0]
            
            # Check if this is summary format
            is_summary_format = False
            if "Party" in first_item:
                is_summary_format = True
            elif "Code" in first_item and "Constituency" in first_item:
                constituency_name = first_item.get("Constituency", "").upper()
                # If constituency name is a party name, it's summary format
                if constituency_name in ["ST. LUCIA LABOUR PARTY", "UNITED WORKERS PARTY", 
                                         "SAINT LUCIA LABOUR PARTY", "LUCIAN PEOPLES MOVEMENT",
                                         "LUCIAN PEOPLE'S MOVEMENT", "NATIONAL DEVELOPMENT MOVEMENT",
                                         "LUCIAN GREENS", "INDEPENDENT", "NATIONAL GREEN PARTY"]:
                    is_summary_format = True
            
            if is_summary_format:
                # Store summary format data - prefer summary_results folder
                if year_identifier not in summary_format_data:
                    summary_format_data[year_identifier] = (key, info)
                elif info["type"] == "summary" and summary_format_data[year_identifier][1]["type"] == "distribution":
                    summary_format_data[year_identifier] = (key, info)
            else:
                # Store constituency format data for later processing if no summary available
                if "Constituency" in first_item:
                    constituency_name = first_item.get("Constituency", "").upper()
                    if constituency_name not in ["TOTAL VALID VOTES", "ST. LUCIA LABOUR PARTY", 
                                                 "UNITED WORKERS PARTY", "SAINT LUCIA LABOUR PARTY"]:
                        if year_identifier not in constituency_format_data:
                            constituency_format_data[year_identifier] = (key, info)
                        elif info["type"] == "summary":
                            # Prefer summary_results folder for constituency data too
                            constituency_format_data[year_identifier] = (key, info)
    
    # Process summary format data first
    for year_identifier, (key, info) in summary_format_data.items():
        data = info["data"]
        for party in data:
            party_name = party.get("Party", party.get("Constituency", ""))
            if "TOTAL" in party_name.upper() or not party_name:
                continue
            
            # Handle percentage as string or number
            pct = party.get("% Votes", 0)
            if isinstance(pct, str):
                pct = pct.replace("%", "").replace(",", ".").strip()
                try:
                    pct = float(pct)
                except ValueError:
                    pct = 0
            
            # Handle seats as number
            seats = party.get("Seats", 0)
            if isinstance(seats, (int, float)) and pd.isna(seats):
                seats = 0
            
            # Normalize party name to combine SLP variations
            normalized_party_name = normalize_party_name(party_name)
            
            summaries.append({
                "Year": year_identifier,
                "Party": normalized_party_name,
                "Code": party.get("Code", ""),
                "Total_Votes": int(party.get("Total Votes", 0)),
                "Percent_Votes": round(pct, 2),
                "Seats": int(seats) if seats else 0,
                "Candidates": int(party.get("Candidates", 0))
            })
        processed_elections.add(year_identifier)
    
    # Second pass: process constituency data for elections without summary format
    for year_identifier, (key, info) in constituency_format_data.items():
        # Skip if we already processed this election (has summary format)
        if year_identifier in processed_elections:
            continue
        
        data = info["data"]
        computed = compute_summary_from_distribution(data, year_identifier)
        summaries.extend(computed)
        processed_elections.add(year_identifier)
    
    return pd.DataFrame(summaries)


def print_data_overview(election_data: Dict):
    """Print an overview of loaded election data."""
    print("=" * 60)
    print("Saint Lucia Election Data Overview")
    print("=" * 60)
    print(f"\nTotal files loaded: {len(election_data)}")
    
    years = sorted(set(info["year"] for info in election_data.values()))
    print(f"\nElections covered: {', '.join(years)}")
    
    print("\nData files by type:")
    summary_count = sum(1 for info in election_data.values() if info["type"] == "summary")
    dist_count = sum(1 for info in election_data.values() if info["type"] == "distribution")
    print(f"  Summary results: {summary_count}")
    print(f"  Vote distributions: {dist_count}")
    
    print("\n" + "=" * 60)


def main():
    """Main analysis function."""
    print("Loading Saint Lucia election data...\n")
    
    # Load all election data
    election_data = load_election_data()
    
    # Print overview
    print_data_overview(election_data)
    
    # Create summary DataFrame
    df_summary = create_summary_dataframe(election_data)
    
    if not df_summary.empty:
        print("\nSummary Statistics by Party (All Elections):")
        print("-" * 60)
        print(df_summary.groupby("Party")[["Total_Votes", "Percent_Votes", "Seats"]].agg({
            "Total_Votes": ["mean", "min", "max"],
            "Percent_Votes": ["mean", "min", "max"],
            "Seats": ["mean", "min", "max"]
        }))
        
        print("\n\nAll Elections Summary:")
        print("-" * 60)
        # Show all elections with vote percentages and seats
        all_elections = df_summary.pivot_table(
            index="Year",
            columns="Code",
            values=["Percent_Votes", "Seats"],
            aggfunc="first"
        )
        if not all_elections.empty:
            print("\nVote Percentages by Election:")
            vote_pct_all = df_summary.pivot_table(
                index="Year",
                columns="Code",
                values="Percent_Votes",
                aggfunc="first"
            )
            print(vote_pct_all)
            print("\nSeats Won by Election:")
            seats_all = df_summary.pivot_table(
                index="Year",
                columns="Code",
                values="Seats",
                aggfunc="first"
            )
            print(seats_all)
        
        print("\n\nRecent Elections (2011-2021):")
        print("-" * 60)
        recent = df_summary[df_summary["Year"].str.startswith(("2011", "2016", "2021"))]
        if not recent.empty:
            # Pivot for vote percentages
            vote_pct = recent.pivot_table(
                index="Year",
                columns="Code",
                values="Percent_Votes",
                aggfunc="first"
            )
            seats = recent.pivot_table(
                index="Year",
                columns="Code",
                values="Seats",
                aggfunc="first"
            )
            print("\nVote Percentages:")
            print(vote_pct)
            print("\nSeats Won:")
            print(seats)
        else:
            print("Note: Summary data format differs for recent elections.")
            print("Check vote_distribution files for constituency-level data.")
    
    print("\n\nData loaded successfully! Ready for analysis.")
    
    # Analyze 2016 and 2021 elections by constituency
    analyze_constituency_breakdowns(election_data)
    
    return election_data, df_summary


def load_constituency_mapping() -> Dict[str, Dict]:
    """Load constituency mapping data from JSON file."""
    mapping_file = Path("data/constituency_maps/saint_lucia_constituencies.json")
    if not mapping_file.exists():
        print("Warning: Constituency mapping file not found.")
        return {}
    
    with open(mapping_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Create a mapping from constituency name to polling divisions
    mapping = {}
    for item in data:
        constituency_name = item.get("constituency", "")
        # Normalize constituency name (remove dashes, parentheses, etc. for matching)
        normalized = normalize_constituency_name(constituency_name)
        mapping[normalized] = {
            "original_name": constituency_name,
            "polling_divisions": item.get("polling_divisions", [])
        }
    
    return mapping


def normalize_constituency_name(name: str) -> str:
    """Normalize constituency name for matching."""
    if not name:
        return ""
    # Remove common variations
    name = name.upper()
    name = name.replace("–", "-").replace("—", "-")
    # Remove parenthetical codes like "(A)", "(B)", etc.
    import re
    name = re.sub(r'\s*\([A-Q]\)\s*', '', name)
    name = re.sub(r'\s*-\s*\([A-Q]\)\s*', '', name)
    # Remove extra whitespace
    name = " ".join(name.split())
    return name


def get_constituency_data_for_year(election_data: Dict, year: str) -> pd.DataFrame:
    """Get constituency-level data for a specific year."""
    # Look for summary results files with constituency data
    constituency_data = []
    
    for key, info in election_data.items():
        if info["year"] != year or info["type"] != "summary":
            continue
        
        data = info["data"]
        if isinstance(data, list) and len(data) > 0:
            first_item = data[0]
            if "Constituency" in first_item:
                # This is constituency-level data
                for row in data:
                    constituency = row.get("Constituency", "")
                    if constituency.upper() == "TOTAL":
                        continue
                    
                    # Extract vote counts for main parties
                    slp_votes = parse_vote_value(row.get("SLP", 0))
                    uwp_votes = parse_vote_value(row.get("UWP", 0))
                    ind_votes = parse_vote_value(row.get("IND", 0))
                    lpm_votes = parse_vote_value(row.get("LPM", 0))
                    ngp_votes = parse_vote_value(row.get("NGP", 0))
                    
                    # Use "Votes Cast" if available, otherwise sum all party votes
                    votes_cast = parse_vote_value(row.get("Votes Cast", 0))
                    if votes_cast > 0:
                        total_votes = votes_cast
                    else:
                        total_votes = slp_votes + uwp_votes + ind_votes + lpm_votes + ngp_votes
                    
                    if total_votes > 0:
                        vote_margin = abs(slp_votes - uwp_votes)  # Absolute vote difference
                        constituency_data.append({
                            "Constituency": constituency,
                            "SLP": slp_votes,
                            "UWP": uwp_votes,
                            "IND": ind_votes,
                            "LPM": lpm_votes,
                            "NGP": ngp_votes,
                            "Total_Votes": total_votes,
                            "SLP_Pct": (slp_votes / total_votes * 100) if total_votes > 0 else 0,
                            "UWP_Pct": (uwp_votes / total_votes * 100) if total_votes > 0 else 0,
                            "IND_Pct": (ind_votes / total_votes * 100) if total_votes > 0 else 0,
                            "LPM_Pct": (lpm_votes / total_votes * 100) if total_votes > 0 else 0,
                            "NGP_Pct": (ngp_votes / total_votes * 100) if total_votes > 0 else 0,
                            "Vote_Margin": vote_margin,
                        })
    
    return pd.DataFrame(constituency_data)


def parse_vote_value(val):
    """Parse vote value, handling strings, numbers, and null values."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return 0
    if isinstance(val, str):
        val = val.replace(",", "").replace("*", "").replace("–", "0").replace("\u2013", "0").strip()
        if not val or val == "NaN":
            return 0
        try:
            return float(val) if "." in val else int(val)
        except ValueError:
            return 0
    return float(val) if isinstance(val, float) else int(val)


def calculate_margin(row: pd.Series) -> float:
    """Calculate the margin between top two parties as a percentage."""
    slp_pct = row.get("SLP_Pct", 0)
    uwp_pct = row.get("UWP_Pct", 0)
    return abs(slp_pct - uwp_pct)


def analyze_constituency_breakdowns(election_data: Dict):
    """
    Analyze constituency breakdowns for 2016 and 2021 elections.
    
    Note: Vote data is available at the constituency level, not polling division level.
    This analysis shows constituency-level vote shares and identifies the narrowest margins.
    Polling division and community information is mapped from the constituency mapping data.
    """
    print("\n" + "=" * 60)
    print("CONSTITUENCY BREAKDOWN ANALYSIS (2016 & 2021)")
    print("=" * 60)
    print("\nNote: Analysis is at constituency level (finest granularity available in vote data)")
    
    # Load constituency mapping
    mapping = load_constituency_mapping()
    
    for year in ["2016", "2021"]:
        print(f"\n\n{year} ELECTION - Constituency Breakdown")
        print("-" * 60)
        
        df = get_constituency_data_for_year(election_data, year)
        if df.empty:
            print(f"No constituency data found for {year}")
            continue
        
        # Calculate margins
        df["Margin"] = df.apply(calculate_margin, axis=1)
        
        # Show full breakdown with percentages
        print(f"\nFull Constituency Breakdown - Vote Percentages ({year}):")
        print("-" * 80)
        display_df = df[["Constituency", "SLP_Pct", "UWP_Pct", "IND_Pct", "LPM_Pct", "NGP_Pct", "Total_Votes"]].copy()
        display_df["SLP_Pct"] = display_df["SLP_Pct"].round(2)
        display_df["UWP_Pct"] = display_df["UWP_Pct"].round(2)
        display_df["IND_Pct"] = display_df["IND_Pct"].round(2)
        display_df["LPM_Pct"] = display_df["LPM_Pct"].round(2)
        display_df["NGP_Pct"] = display_df["NGP_Pct"].round(2)
        display_df = display_df.sort_values("Constituency")
        print(display_df.to_string(index=False))
        
        # Sort by margin (narrowest first)
        df_sorted = df.sort_values("Margin").head(10)
        
        print(f"\n\nTop 10 Constituencies with Narrowest Margins ({year}):")
        print("-" * 80)
        margin_display = df_sorted[["Constituency", "SLP_Pct", "UWP_Pct", "Margin", "Vote_Margin", "Total_Votes"]].copy()
        margin_display["SLP_Pct"] = margin_display["SLP_Pct"].round(2)
        margin_display["UWP_Pct"] = margin_display["UWP_Pct"].round(2)
        margin_display["Margin"] = margin_display["Margin"].round(2)
        margin_display["Vote_Margin"] = margin_display["Vote_Margin"].astype(int)
        print(margin_display.to_string(index=False))
        
        # Create visualization
        create_margin_plot(df_sorted, year, mapping)
        
        # Analyze polling divisions for the 5 narrowest constituencies
        analyze_polling_divisions(year, df_sorted.head(5), mapping)
    
    print("\n" + "=" * 60)


def create_margin_plot(df: pd.DataFrame, year: str, mapping: Dict):
    """Create a plot showing margins and communities for the narrowest constituencies."""
    if df.empty:
        return
    
    # Set up the plot
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Create horizontal bar chart
    y_pos = range(len(df))
    margins = df["Margin"].values
    constituencies = df["Constituency"].values
    
    # Color bars based on which party won
    colors = []
    for _, row in df.iterrows():
        if row["SLP_Pct"] > row["UWP_Pct"]:
            colors.append("#FF6B6B")  # Red for SLP
        else:
            colors.append("#4ECDC4")  # Teal for UWP
    
    bars = ax.barh(y_pos, margins, color=colors, alpha=0.7)
    
    # Add value labels on bars
    for i, (bar, margin) in enumerate(zip(bars, margins)):
        ax.text(margin + 0.5, i, f'{margin:.1f}%', 
                va='center', fontweight='bold')
    
    # Customize the plot
    ax.set_yticks(y_pos)
    ax.set_yticklabels(constituencies, fontsize=9)
    ax.set_xlabel('Margin (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'Top 10 Narrowest Margins by Constituency - {year} Election\n'
                 f'(Red = SLP won, Teal = UWP won)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_xlim(0, max(margins) * 1.15)
    
    # Add communities information as text annotations
    for i, (_, row) in enumerate(df.iterrows()):
        constituency = row["Constituency"]
        normalized = normalize_constituency_name(constituency)
        
        # Find matching constituency in mapping
        matched_constituency = None
        for map_name, map_data in mapping.items():
            if normalized in map_name or map_name in normalized:
                matched_constituency = map_data
                break
        
        if matched_constituency:
            # Get all communities from polling divisions
            all_communities = []
            for div in matched_constituency.get("polling_divisions", []):
                div_name = div.get("division", "")
                if div_name.upper() not in ["POLLING DIVISION", "COMMUNITIES"]:
                    communities = div.get("communities", [])
                    all_communities.extend([c for c in communities if c.upper() not in ["COMMUNITIES"]])
            
            # Limit to first 5 communities for readability
            communities_str = ", ".join(all_communities[:5])
            if len(all_communities) > 5:
                communities_str += f" (+{len(all_communities) - 5} more)"
            
            # Add text annotation with polling divisions info
            num_divisions = len([d for d in matched_constituency.get("polling_divisions", []) 
                               if d.get("division", "").upper() not in ["POLLING DIVISION", "COMMUNITIES"]])
            
            info_text = f'  {num_divisions} polling divisions | Communities: {communities_str[:60]}...' if len(communities_str) > 60 else f'  {num_divisions} polling divisions | Communities: {communities_str}'
            ax.text(max(margins) * 1.05, i, info_text,
                   va='center', fontsize=7, style='italic', alpha=0.7)
    
    plt.tight_layout()
    
    # Save the plot
    output_dir = Path("analysis")
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f"narrowest_margins_{year}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_file}")
    plt.close()


def normalize_polling_division_name(div_name: str) -> str:
    """Normalize polling division name for matching (remove spaces, standardize format)."""
    if not div_name:
        return ""
    # Remove spaces, convert to uppercase, remove parentheses variations
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


def normalize_constituency_name_for_matching(name: str) -> str:
    """Normalize constituency name for matching between different data sources."""
    if not name:
        return ""
    name = name.upper().strip()
    # Remove common variations
    name = name.replace("–", "-").replace("—", "-")
    # Remove parenthetical codes
    import re
    name = re.sub(r'\s*\([A-Q]\)\s*', '', name)
    name = re.sub(r'\s*-\s*\([A-Q]\)\s*', '', name)
    # Remove extra whitespace
    name = " ".join(name.split())
    return name


def analyze_polling_divisions(year: str, top_constituencies: pd.DataFrame, mapping: Dict):
    """Analyze polling divisions for the top 5 narrowest constituencies."""
    print(f"\n\n{year} ELECTION - Polling Division Analysis (Top 5 Narrowest Constituencies)")
    print("=" * 80)
    
    # Load election night results
    night_results = load_election_night_results(year)
    if not night_results:
        print(f"No election night results found for {year}")
        return
    
    # Get constituency names from top 5
    constituency_names = top_constituencies["Constituency"].tolist()
    
    # Extract polling division data for these constituencies
    polling_division_data = []
    
    for constituency in constituency_names:
        # Normalize constituency name for matching
        normalized_const = normalize_constituency_name_for_matching(constituency)
        
        # Find matching records in election night results
        constituency_records = []
        for record in night_results:
            record_district = normalize_constituency_name_for_matching(record.get("district", ""))
            if normalized_const in record_district or record_district in normalized_const:
                # Check if this is a candidate record (has Party field)
                if "Party" in record and record.get("Party") in ["SLP", "UWP"]:
                    constituency_records.append(record)
        
        if not constituency_records:
            continue
        
        # Get all polling division columns (exclude metadata columns)
        exclude_cols = {"district", "District", "Candidate", "Party", "Total", "% Total", 
                       "summary_label", "summary_value"}
        
        # Find all polling division columns
        polling_divisions = set()
        for record in constituency_records:
            for key in record.keys():
                if key not in exclude_cols and key.strip():
                    polling_divisions.add(key)
        
        # Calculate margins for each polling division
        for div in polling_divisions:
            slp_votes = 0
            uwp_votes = 0
            
            for record in constituency_records:
                party = record.get("Party", "")
                votes_str = record.get(div, "0")
                try:
                    votes = int(str(votes_str).replace(",", "").strip() or "0")
                    if party == "SLP":
                        slp_votes += votes
                    elif party == "UWP":
                        uwp_votes += votes
                except (ValueError, TypeError):
                    pass
            
            total_votes = slp_votes + uwp_votes
            if total_votes > 0:
                slp_pct = (slp_votes / total_votes * 100)
                uwp_pct = (uwp_votes / total_votes * 100)
                margin = abs(slp_pct - uwp_pct)
                vote_margin = abs(slp_votes - uwp_votes)  # Absolute vote difference
                
                polling_division_data.append({
                    "Constituency": constituency,
                    "Polling_Division": div,
                    "SLP_Votes": slp_votes,
                    "UWP_Votes": uwp_votes,
                    "Total_Votes": total_votes,
                    "SLP_Pct": slp_pct,
                    "UWP_Pct": uwp_pct,
                    "Margin": margin,
                    "Vote_Margin": vote_margin
                })
    
    if not polling_division_data:
        print("No polling division data found for the selected constituencies")
        return
    
    # Create DataFrame and find top 10 narrowest
    df_pd = pd.DataFrame(polling_division_data)
    df_pd_sorted = df_pd.sort_values("Margin").head(10)
    
    print(f"\nTop 10 Narrowest Polling Divisions (from 5 narrowest constituencies):")
    print("-" * 80)
    display_cols = ["Constituency", "Polling_Division", "SLP_Pct", "UWP_Pct", "Margin", "Vote_Margin", "Total_Votes"]
    display_df = df_pd_sorted[display_cols].copy()
    display_df["SLP_Pct"] = display_df["SLP_Pct"].round(2)
    display_df["UWP_Pct"] = display_df["UWP_Pct"].round(2)
    display_df["Margin"] = display_df["Margin"].round(2)
    display_df["Vote_Margin"] = display_df["Vote_Margin"].astype(int)
    print(display_df.to_string(index=False))
    
    # Map polling divisions to communities
    map_divisions_to_communities(df_pd_sorted, mapping, year)


def map_divisions_to_communities(df: pd.DataFrame, mapping: Dict, year: str):
    """Map polling divisions to communities and create visualization."""
    # Build mapping of polling division to communities
    division_communities = {}
    
    for _, row in df.iterrows():
        constituency = row["Constituency"]
        polling_div = row["Polling_Division"]
        
        # Normalize constituency name
        normalized_const = normalize_constituency_name_for_matching(constituency)
        
        # Find matching constituency in mapping
        matched_constituency = None
        for map_name, map_data in mapping.items():
            if normalized_const in map_name or map_name in normalized_const:
                matched_constituency = map_data
                break
        
        if matched_constituency:
            # Normalize polling division name for matching
            normalized_div = normalize_polling_division_name(polling_div)
            
            # Find matching polling division
            communities_list = []
            for div_info in matched_constituency.get("polling_divisions", []):
                div_name = div_info.get("division", "")
                normalized_map_div = normalize_polling_division_name(div_name)
                
                if normalized_div == normalized_map_div or normalized_div in normalized_map_div:
                    communities = div_info.get("communities", [])
                    # Filter out header rows
                    communities_list = [c for c in communities 
                                      if c.upper() not in ["COMMUNITIES", "POLLING DIVISION"]]
                    break
            
            division_communities[polling_div] = {
                "constituency": constituency,
                "communities": communities_list
            }
    
    # Create visualization
    create_polling_division_plot(df, division_communities, year)


def create_polling_division_plot(df: pd.DataFrame, division_communities: Dict, year: str):
    """Create a plot showing polling division margins with community information."""
    if df.empty:
        return
    
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Create labels with constituency and polling division
    labels = [f"{row['Constituency']}\n{row['Polling_Division']}" for _, row in df.iterrows()]
    margins = df["Margin"].values
    
    # Color bars based on which party won
    colors = []
    for _, row in df.iterrows():
        if row["SLP_Pct"] > row["UWP_Pct"]:
            colors.append("#FF6B6B")  # Red for SLP
        else:
            colors.append("#4ECDC4")  # Teal for UWP
    
    y_pos = range(len(df))
    bars = ax.barh(y_pos, margins, color=colors, alpha=0.7)
    
    # Add value labels on bars
    for i, (bar, margin) in enumerate(zip(bars, margins)):
        ax.text(margin + 0.3, i, f'{margin:.1f}%', 
                va='center', fontweight='bold', fontsize=9)
    
    # Customize the plot
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel('Margin (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'Top 10 Narrowest Polling Division Margins - {year} Election\n'
                 f'(From 5 Narrowest Constituencies | Red = SLP won, Teal = UWP won)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_xlim(0, max(margins) * 1.2)
    
    # Add communities information
    for i, (_, row) in enumerate(df.iterrows()):
        polling_div = row["Polling_Division"]
        div_info = division_communities.get(polling_div, {})
        communities = div_info.get("communities", [])
        
        if communities:
            # Limit to first 5 communities for readability
            communities_str = ", ".join(communities[:5])
            if len(communities) > 5:
                communities_str += f" (+{len(communities) - 5} more)"
            
            # Add text annotation
            info_text = f'  Communities: {communities_str[:70]}...' if len(communities_str) > 70 else f'  Communities: {communities_str}'
            ax.text(max(margins) * 1.05, i, info_text,
                   va='center', fontsize=7, style='italic', alpha=0.7)
    
    plt.tight_layout()
    
    # Save the plot
    output_dir = Path("analysis")
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f"narrowest_polling_divisions_{year}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nPolling division plot saved to: {output_file}")
    plt.close()


if __name__ == "__main__":
    election_data, df_summary = main()
