#!/usr/bin/env python3
"""
Shared utilities for swing analysis scripts.

Functions for parsing election data, normalizing names, and loading results.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional


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


# Canonical constituency name mappings for consistent matching
CONSTITUENCY_ALIASES = {
    # Gros Islet variations
    "GROS ISLET": "GROS ISLET",
    "GROS - ISLET": "GROS ISLET",
    "GROS-ISLET": "GROS ISLET",
    "GROS – ISLET": "GROS ISLET",  # en-dash
    # Vieux Fort variations
    "VIEUX FORT NORTH": "VIEUX FORT NORTH",
    "V-FORT NORTH": "VIEUX FORT NORTH",
    "VFORT NORTH": "VIEUX FORT NORTH",
    "V FORT NORTH": "VIEUX FORT NORTH",
    "VIEUX FORT SOUTH": "VIEUX FORT SOUTH",
    "V-FORT SOUTH": "VIEUX FORT SOUTH",
    "VFORT SOUTH": "VIEUX FORT SOUTH",
    "V FORT SOUTH": "VIEUX FORT SOUTH",
    # Choiseul variations (common misspelling)
    "CHOISEUL": "CHOISEUL",
    "CHOISUEL": "CHOISEUL",
    # Anse La Raye variations
    "ANSE LA RAYE/CANARIES": "ANSE LA RAYE/CANARIES",
    "ANSE LA RAYE CANARIES": "ANSE LA RAYE/CANARIES",
    "ANSELA RAYE/CANARIES": "ANSE LA RAYE/CANARIES",
    "ANSE LA RAYECANARIES": "ANSE LA RAYE/CANARIES",
    # Dennery variations
    "DENNERY NORTH": "DENNERY NORTH",
    "DENNERYNORTH": "DENNERY NORTH",
    "DENNERY SOUTH": "DENNERY SOUTH",
    "DENNERYSOUTH": "DENNERY SOUTH",
    # Micoud variations
    "MICOUD NORTH": "MICOUD NORTH",
    "MICOUD SOUTH": "MICOUD SOUTH",
    # Castries variations
    "CASTRIES NORTH": "CASTRIES NORTH",
    "CASTRIES SOUTH": "CASTRIES SOUTH",
    "CASTRIES EAST": "CASTRIES EAST",
    "CASTRIES CENTRAL": "CASTRIES CENTRAL",
    "CASTRIES SOUTH EAST": "CASTRIES SOUTH EAST",
    "CASTRIES SOUTHEAST": "CASTRIES SOUTH EAST",
    # Others
    "BABONNEAU": "BABONNEAU",
    "SOUFRIERE": "SOUFRIERE",
    "LABORIE": "LABORIE",
}


def normalize_constituency_name(name: str) -> str:
    """
    Normalize constituency name for matching.

    Handles common variations:
    - "Gros - Islet" -> "GROS ISLET"
    - "V-Fort North" -> "VIEUX FORT NORTH"
    - "Choisuel" (misspelling) -> "CHOISEUL"
    """
    if not name:
        return ""
    # Basic cleaning
    name = name.replace("–", "-").replace("—", "-").strip()
    name = name.replace("AnseLa", "Anse La")
    name = re.sub(r'([a-z])([A-Z])', r'\1 \2', name)
    name = " ".join(name.split())
    name = name.upper()

    # Check for known aliases
    if name in CONSTITUENCY_ALIASES:
        return CONSTITUENCY_ALIASES[name]

    # Try removing hyphens and extra spaces for matching
    normalized_no_hyphen = name.replace("-", " ").replace("/", " ")
    normalized_no_hyphen = " ".join(normalized_no_hyphen.split())
    if normalized_no_hyphen in CONSTITUENCY_ALIASES:
        return CONSTITUENCY_ALIASES[normalized_no_hyphen]

    return name


def normalize_polling_division_name(div_name: str) -> str:
    """Normalize polling division name for matching."""
    if not div_name:
        return ""
    div_name = div_name.upper().strip()
    div_name = div_name.replace(" ", "").replace("(", "").replace(")", "")
    return div_name


def load_election_results(year: str, data_dir: str = "data") -> List[Dict]:
    """Load election night results for a specific year."""
    results_file = Path(data_dir) / "election_night_results" / f"st_lucia_{year}_full_results.json"
    if not results_file.exists():
        raise FileNotFoundError(f"Results file not found: {results_file}")
    with open(results_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def is_summary_record(record: Dict) -> bool:
    """Check if record is a summary (electors/votes/turnout/rejected)."""
    return "summary_label" in record


def is_candidate_record(record: Dict) -> bool:
    """Check if record is a candidate vote record."""
    return "Party" in record and record.get("Candidate")


def get_polling_division_columns(record: Dict) -> List[str]:
    """Extract polling division column names from a record."""
    exclude_cols = {
        "district", "District", "Candidate", "Party", "Total", "% Total",
        "%", "summary_label", "summary_value"
    }
    divisions = []
    for key in record.keys():
        if key not in exclude_cols and key.strip() and not key.startswith("col_"):
            divisions.append(key)
    return divisions


def get_all_constituencies(results: List[Dict]) -> List[str]:
    """
    Get list of all unique constituency names from results.
    Filters out summary labels.
    """
    summary_labels = {
        'No. of Electors', 'No. Of Electors', 'Votes Cast', 'Turnout %',
        'Rejected', '% Turnout', 'Turnout%', 'Total Votes Cast'
    }
    constituencies = set()
    for record in results:
        if is_candidate_record(record):
            district = record.get('district', '')
            if district and district not in summary_labels:
                constituencies.add(district)
    return sorted(constituencies)


def constituencies_match(name1: str, name2: str) -> bool:
    """
    Check if two constituency names refer to the same constituency.

    Handles variations like:
    - "Gros Islet" vs "Gros - Islet" vs "Gros – Islet"
    - "V-Fort North" vs "Vieux Fort North"
    - "Choisuel" vs "Choiseul"
    """
    norm1 = normalize_constituency_name(name1)
    norm2 = normalize_constituency_name(name2)
    if norm1 == norm2:
        return True

    # Also check without spaces, hyphens, and slashes for variations
    compact1 = norm1.replace(" ", "").replace("-", "").replace("/", "")
    compact2 = norm2.replace(" ", "").replace("-", "").replace("/", "")
    if compact1 == compact2:
        return True

    return False


def extract_constituency_data(results: List[Dict], constituency_name: str) -> Optional[Dict]:
    """
    Extract all data for a constituency from election results.

    Returns:
        {
            'original_name': str,
            'candidates': [
                {
                    'name': str,
                    'party': str,
                    'total_votes': int,
                    'pct': float,
                    'division_votes': {div_code: votes}
                }
            ],
            'polling_divisions': [list of division codes],
            'summary': {
                'electors': int,
                'votes_cast': int,
                'turnout_pct': float,
                'rejected': int
            }
        }
    """
    normalized_target = normalize_constituency_name(constituency_name)

    candidate_records = []
    summary_data = {}
    original_name = None
    found_constituency = False

    for record in results:
        district = record.get('district', '')

        if constituencies_match(district, constituency_name):
            found_constituency = True
            if is_candidate_record(record):
                candidate_records.append(record)
                if original_name is None:
                    original_name = district
            elif is_summary_record(record):
                label = record.get("summary_label", "")
                value = record.get("summary_value", "")
                if value and str(value).strip() and str(value).strip() not in ["", "–", "\u2013"]:
                    summary_data[label] = value
        elif found_constituency:
            if is_summary_record(record):
                label = record.get("summary_label", "")
                value = record.get("summary_value", "")
                if value and str(value).strip() and str(value).strip() not in ["", "–", "\u2013"]:
                    summary_data[label] = value
            elif is_candidate_record(record):
                # Hit next constituency
                break

    if not candidate_records:
        return None

    # Get all polling division columns
    all_divisions = set()
    for record in candidate_records:
        all_divisions.update(get_polling_division_columns(record))
    polling_divisions = sorted(all_divisions)

    # Build candidate data
    candidates = []
    for record in candidate_records:
        division_votes = {}
        for div in polling_divisions:
            votes = parse_vote_value(record.get(div, 0))
            division_votes[div] = votes

        total = parse_vote_value(record.get("Total", 0))
        pct_str = record.get("% Total", "0")
        try:
            pct = float(pct_str) if pct_str else 0.0
        except ValueError:
            pct = 0.0

        candidates.append({
            'name': record.get("Candidate", ""),
            'party': record.get("Party", "").strip(),
            'total_votes': total,
            'pct': pct,
            'division_votes': division_votes
        })

    # Parse summary data
    electors = parse_vote_value(summary_data.get("No. of Electors", 0))
    if electors == 0:
        electors = parse_vote_value(summary_data.get("No. Of Electors", 0))
    votes_cast = parse_vote_value(summary_data.get("Votes Cast", 0))
    rejected = parse_vote_value(summary_data.get("Rejected", 0))

    turnout_str = summary_data.get("Turnout %", "0")
    try:
        turnout_pct = float(turnout_str.replace("%", "")) if turnout_str else 0.0
    except ValueError:
        turnout_pct = 0.0

    return {
        'original_name': original_name,
        'candidates': candidates,
        'polling_divisions': polling_divisions,
        'summary': {
            'electors': electors,
            'votes_cast': votes_cast,
            'turnout_pct': turnout_pct,
            'rejected': rejected
        }
    }


def find_constituency_in_thresholds(
    constituency_name: str,
    thresholds: Dict
) -> Optional[Dict]:
    """
    Look up a constituency in the thresholds dictionary, handling name variations.

    Args:
        constituency_name: Name to look up (e.g., "Gros Islet", "V-Fort North")
        thresholds: Thresholds dictionary with 'constituencies' key

    Returns:
        The threshold data for the constituency, or None if not found
    """
    if not thresholds or 'constituencies' not in thresholds:
        return None

    constituencies = thresholds['constituencies']
    normalized_target = normalize_constituency_name(constituency_name)

    # Direct lookup
    if normalized_target in constituencies:
        return constituencies[normalized_target]

    # Try all keys in case of naming variations
    for key in constituencies:
        if normalize_constituency_name(key) == normalized_target:
            return constituencies[key]

    # Fuzzy match without spaces/hyphens
    normalized_compact = normalized_target.replace(" ", "").replace("-", "").replace("/", "")
    for key in constituencies:
        key_compact = normalize_constituency_name(key).replace(" ", "").replace("-", "").replace("/", "")
        if key_compact == normalized_compact:
            return constituencies[key]

    return None


def get_canonical_constituency_name(name: str) -> str:
    """
    Get the canonical (normalized) name for a constituency.

    This is useful for consistent display and file naming.
    """
    return normalize_constituency_name(name)
