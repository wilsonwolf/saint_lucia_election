#!/usr/bin/env python3
"""
Bellwether Constituency Analysis

Analyzes which constituencies have always voted for the party that won
the national election (i.e., constituencies that never break from the
national outcome).

Uses summary results data from 1979 onwards.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

# Elections to analyze (in chronological order)
ELECTIONS = [
    ("1979", "saint_lucia_1979_summary_results.json", "saint_lucia_1979_vote_distribution.json"),
    ("1982", "saint_lucia_1982_summary_results.json", "saint_lucia_1982_vote_distribution.json"),
    ("1987_apr6", "saint_lucia_1987_apr6_summary_results.json", "saint_lucia_1987_apr6_vote_distribution.json"),
    ("1987_apr30", "saint_lucia_1987_apr30_summary_results.json", None),  # No vote distribution for Apr 30
    ("1992", "saint_lucia_1992_summary_results.json", "saint_lucia_1992_vote_distribution.json"),
    ("1997", "saint_lucia_1997_summary_results.json", "saint_lucia_1997_vote_distribution.json"),
    ("2001", "saint_lucia_2001_summary_results.json", "saint_lucia_2001_vote_distribution.json"),
    ("2006", "saint_lucia_2006_summary_results.json", "saint_lucia_2006_vote_distribution.json"),
    ("2011", "saint_lucia_2011_summary_results.json", "saint_lucia_2011_vote_distribution.json"),
    ("2016", "saint_lucia_2016_summary_results.json", "saint_lucia_2016_vote_distribution.json"),
    ("2021", "saint_lucia_2021_summary_results.json", "saint_lucia_2021_vote_distribution.json"),
]

DATA_DIR = Path("data")


def parse_vote_value(val) -> int:
    """Parse vote value, handling various formats."""
    if val is None:
        return 0
    if isinstance(val, (int, float)):
        if val != val:  # NaN check
            return 0
        return int(val)
    if isinstance(val, str):
        val = val.replace(",", "").replace("–", "0").replace("\u2013", "0").replace("%", "").strip()
        if not val or val in ["NaN", "", "-"]:
            return 0
        try:
            return int(float(val))
        except ValueError:
            return 0
    return 0


def normalize_constituency_name(name: str) -> str:
    """Normalize constituency name for matching across elections."""
    if not name:
        return ""
    name = name.upper().strip()
    # Normalize variations
    name = name.replace("–", "-").replace("&", "/").replace(" AND ", "/")
    name = name.replace("ANSE LA RAYE", "ANSE-LA-RAYE")
    name = name.replace("ANSE-LA-RAYE / CANARIES", "ANSE-LA-RAYE/CANARIES")
    name = name.replace("ANSE-LA-RAYE/ CANARIES", "ANSE-LA-RAYE/CANARIES")
    name = name.replace("ANSE LA RAYE/CANARIES", "ANSE-LA-RAYE/CANARIES")
    name = name.replace("VIEUX FORT", "VIEUX-FORT")
    name = name.replace("VIEUX-FORT-", "VIEUX-FORT ")
    name = name.replace("V-FORT", "VIEUX-FORT")
    name = " ".join(name.split())
    return name


def normalize_party_name(party) -> str:
    """Normalize party names."""
    if party is None:
        return ""
    if isinstance(party, float):
        if party != party:  # NaN check
            return ""
        return ""
    if not isinstance(party, str):
        return ""
    if not party:
        return ""
    party = party.upper().strip()
    if "LABOUR" in party or party == "SLP":
        return "SLP"
    if "WORKERS" in party or party == "UWP":
        return "UWP"
    if party in ["IND", "INDEPENDENT"]:
        return "IND"
    return party


def load_vote_distribution(filepath: Path) -> Dict:
    """Load vote distribution and extract national winner by seats."""
    if not filepath.exists():
        return None

    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    parties = {}
    for row in data:
        # Handle different column naming
        party_name = row.get("Constituency", row.get("Party", ""))
        code = row.get("Code", "")
        seats = parse_vote_value(row.get("Seats", 0))
        votes = parse_vote_value(row.get("Total Votes", 0))

        if code and code not in ["NaN", ""]:
            normalized = normalize_party_name(code)
            if normalized and normalized != "NaN":
                parties[normalized] = {
                    "seats": seats,
                    "votes": votes
                }

    return parties


def load_summary_results(filepath: Path) -> List[Dict]:
    """Load constituency-level summary results."""
    if not filepath.exists():
        return None

    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return data


def get_national_winner(parties: Dict) -> Tuple[str, int]:
    """Determine national winner by seats won."""
    if not parties:
        return None, 0

    # Filter to main parties (SLP/UWP)
    main_parties = {k: v for k, v in parties.items() if k in ["SLP", "UWP"]}

    if not main_parties:
        return None, 0

    winner = max(main_parties.items(), key=lambda x: x[1]["seats"])
    return winner[0], winner[1]["seats"]


def get_constituency_winner(row: Dict, parties_in_election: List[str]) -> Tuple[str, int]:
    """Determine winner of a constituency."""
    # Get vote counts for each party
    party_votes = {}

    for party in ["SLP", "UWP", "IND", "LPM", "NGP", "PLP", "PDM", "NDP"]:
        votes = parse_vote_value(row.get(party, 0))
        if votes > 0:
            party_votes[party] = votes

    if not party_votes:
        return None, 0

    winner = max(party_votes.items(), key=lambda x: x[1])
    return normalize_party_name(winner[0]), winner[1]


def analyze_elections() -> Dict:
    """Analyze all elections and track constituency voting patterns."""

    # Track each constituency's voting history
    constituency_history = defaultdict(list)  # {constituency: [(year, const_winner, national_winner, matched), ...]}

    # Track national results
    national_results = []

    for year, summary_file, dist_file in ELECTIONS:
        summary_path = DATA_DIR / "summary_results" / summary_file

        # Load data
        summary_data = load_summary_results(summary_path)

        # Load vote distribution if available
        if dist_file:
            dist_path = DATA_DIR / "vote_distribution" / dist_file
            dist_data = load_vote_distribution(dist_path)
        else:
            # For elections without vote distribution, determine winner from summary results
            dist_data = None

        if not summary_data:
            print(f"Warning: Missing summary data for {year}")
            continue

        # If no vote distribution, calculate national winner from summary results
        if not dist_data:
            # Sum up votes from summary to determine winner
            party_totals = defaultdict(int)
            for row in summary_data:
                if row.get("Constituency", "").upper() == "TOTAL":
                    continue
                for party in ["SLP", "UWP", "PLP", "IND", "NGP", "LPM"]:
                    party_totals[party] += parse_vote_value(row.get(party, 0))
            # Create synthetic dist_data with lowercase keys to match load_vote_distribution format
            dist_data = {p: {"votes": v, "seats": 0} for p, v in party_totals.items() if v > 0}
            # Determine seats won (party with most votes in each constituency)
            for row in summary_data:
                if row.get("Constituency", "").upper() == "TOTAL":
                    continue
                max_party = None
                max_votes = 0
                for party in dist_data.keys():
                    votes = parse_vote_value(row.get(party, 0))
                    if votes > max_votes:
                        max_votes = votes
                        max_party = party
                if max_party:
                    dist_data[max_party]["seats"] = dist_data[max_party].get("seats", 0) + 1

        # Get national winner
        national_winner, national_seats = get_national_winner(dist_data)
        national_results.append({
            "year": year,
            "winner": national_winner,
            "seats": national_seats,
            "parties": dist_data
        })

        # Analyze each constituency
        for row in summary_data:
            const_name = row.get("Constituency", "")
            if not const_name or const_name.upper() == "TOTAL":
                continue

            normalized_name = normalize_constituency_name(const_name)
            const_winner, const_votes = get_constituency_winner(row, list(dist_data.keys()))

            if const_winner:
                matched = (const_winner == national_winner)
                constituency_history[normalized_name].append({
                    "year": year,
                    "constituency_winner": const_winner,
                    "national_winner": national_winner,
                    "matched": matched,
                    "votes": const_votes
                })

    return {
        "national_results": national_results,
        "constituency_history": dict(constituency_history)
    }


def find_bellwether_constituencies(analysis: Dict) -> Dict:
    """Find constituencies that always voted with the national winner."""

    constituency_history = analysis["constituency_history"]
    national_results = analysis["national_results"]

    total_elections = len(national_results)

    bellwethers = []
    near_bellwethers = []
    never_bellwethers = []

    for const_name, history in constituency_history.items():
        elections_participated = len(history)
        matches = sum(1 for h in history if h["matched"])
        misses = elections_participated - matches

        # Get the years they broke from national
        break_years = [h["year"] for h in history if not h["matched"]]
        break_details = [
            f"{h['year']}: voted {h['constituency_winner']} (national: {h['national_winner']})"
            for h in history if not h["matched"]
        ]

        result = {
            "constituency": const_name,
            "elections_participated": elections_participated,
            "matches": matches,
            "misses": misses,
            "match_rate": round(matches / elections_participated * 100, 1) if elections_participated > 0 else 0,
            "break_years": break_years,
            "break_details": break_details,
            "history": history
        }

        if misses == 0 and elections_participated >= total_elections - 1:
            bellwethers.append(result)
        elif misses <= 2:
            near_bellwethers.append(result)
        else:
            never_bellwethers.append(result)

    # Sort by match rate
    bellwethers.sort(key=lambda x: (-x["match_rate"], x["constituency"]))
    near_bellwethers.sort(key=lambda x: (-x["match_rate"], x["constituency"]))
    never_bellwethers.sort(key=lambda x: (-x["match_rate"], x["constituency"]))

    return {
        "bellwethers": bellwethers,
        "near_bellwethers": near_bellwethers,
        "others": never_bellwethers,
        "total_elections": total_elections
    }


def print_report(analysis: Dict, bellwether_results: Dict):
    """Print a formatted report."""

    print("=" * 80)
    print("BELLWETHER CONSTITUENCY ANALYSIS")
    print("Constituencies that vote with the national winner")
    print("=" * 80)
    print()

    # National election history
    print("NATIONAL ELECTION HISTORY")
    print("-" * 40)
    for result in analysis["national_results"]:
        slp_seats = result["parties"].get("SLP", {}).get("seats", 0)
        uwp_seats = result["parties"].get("UWP", {}).get("seats", 0)
        print(f"  {result['year']}: {result['winner']} won ({slp_seats} SLP vs {uwp_seats} UWP)")
    print()

    total_elections = bellwether_results["total_elections"]

    # Perfect bellwethers
    print("=" * 80)
    print(f"PERFECT BELLWETHERS (never broke from national outcome)")
    print("=" * 80)

    if bellwether_results["bellwethers"]:
        for b in bellwether_results["bellwethers"]:
            print(f"\n  {b['constituency']}")
            print(f"    Elections: {b['elections_participated']}/{total_elections}")
            print(f"    Match rate: {b['match_rate']}%")
    else:
        print("\n  None found - no constituency has always voted with the national winner")

    # Near bellwethers (1-2 misses)
    print()
    print("=" * 80)
    print("NEAR BELLWETHERS (1-2 misses)")
    print("=" * 80)

    for b in bellwether_results["near_bellwethers"]:
        print(f"\n  {b['constituency']}")
        print(f"    Elections: {b['elections_participated']}/{total_elections}")
        print(f"    Match rate: {b['match_rate']}% ({b['misses']} miss{'es' if b['misses'] != 1 else ''})")
        if b["break_details"]:
            print(f"    Breaks:")
            for detail in b["break_details"]:
                print(f"      - {detail}")

    # Summary table
    print()
    print("=" * 80)
    print("ALL CONSTITUENCIES RANKED BY MATCH RATE")
    print("=" * 80)
    print()
    print(f"{'Constituency':<30} {'Elections':<12} {'Matches':<10} {'Rate':<10}")
    print("-" * 62)

    all_constituencies = (
        bellwether_results["bellwethers"] +
        bellwether_results["near_bellwethers"] +
        bellwether_results["others"]
    )
    all_constituencies.sort(key=lambda x: (-x["match_rate"], x["constituency"]))

    for c in all_constituencies:
        print(f"{c['constituency']:<30} {c['elections_participated']:<12} {c['matches']:<10} {c['match_rate']:.1f}%")


def main():
    print("Loading election data from 1979-2021...")
    print()

    analysis = analyze_elections()
    bellwether_results = find_bellwether_constituencies(analysis)

    print_report(analysis, bellwether_results)

    # Save detailed results to JSON
    output_path = DATA_DIR / "bellwether_analysis.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            "national_results": analysis["national_results"],
            "bellwether_results": {
                "bellwethers": bellwether_results["bellwethers"],
                "near_bellwethers": bellwether_results["near_bellwethers"],
                "total_elections": bellwether_results["total_elections"]
            }
        }, f, indent=2, ensure_ascii=False)

    print()
    print(f"\nDetailed results saved to: {output_path}")


if __name__ == "__main__":
    main()
