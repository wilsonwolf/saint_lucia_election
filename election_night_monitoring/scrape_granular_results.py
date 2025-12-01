#!/usr/bin/env python3
"""
Granular Results Scraper - results.sluelectoral.com/district.php

Scrapes detailed surname-bucket level data from all 17 constituency pages.
This is the most granular data available and serves as a third redundancy source.

TERMINOLOGY:
- Polling Division: A voting location (e.g., "GROS ISLET PRIMARY SCHOOL") with code like A1(a)
- Ballot Box: A surname bucket WITHIN a polling division (e.g., "A-B", "C-D", "E-F")
  Each polling division has multiple ballot boxes, separated by voter surname ranges.

Features:
- Parallel fetching with rate limiting
- Raw granular data archival at surname-bucket (ballot box) level
- Ballot box counting progress tracking (which surname buckets counted, votes remaining)
- Aggregation to polling division level for compatibility with existing analysis

The granular data preserves:
- Surname buckets (A-B, C-D, etc.) = individual ballot boxes within each polling division
- Ballot box counting status per surname range
- Electors vs votes cast (to calculate remaining potential votes)

Usage:
    python scrape_granular_results.py [--once] [--output-dir PATH]
"""

import argparse
import hashlib
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
from bs4 import BeautifulSoup

# Configuration
BASE_URL = "https://results.sluelectoral.com/district.php?id="
DISTRICT_IDS = list(range(1, 18))  # 1-17
REQUEST_TIMEOUT = 60
MAX_RETRIES = 3
RETRY_DELAYS = [5, 15, 30]
PARALLEL_WORKERS = 3  # Conservative to avoid overwhelming server
DELAY_BETWEEN_REQUESTS = 1.0  # seconds

# District ID to constituency name mapping (will be populated from scraping)
DISTRICT_NAMES = {
    1: "Gros Islet",
    2: "Babonneau",
    3: "Castries North",
    4: "Castries East",
    5: "Castries Central",
    6: "Castries South East",
    7: "Castries South",
    8: "Anse La Raye/Canaries",
    9: "Soufriere",
    10: "Laborie",
    11: "Choisuel",
    12: "Vieux Fort South",
    13: "Vieux Fort North",
    14: "Micoud South",
    15: "Micoud North",
    16: "Dennery South",
    17: "Dennery North",
}


def clean_text(text: str) -> str:
    """Clean text by removing extra whitespace and special characters."""
    if not text:
        return ""
    text = text.replace("\xa0", " ").replace("\u2013", "-").replace("–", "-")
    return " ".join(text.split()).strip()


def parse_vote_value(val: str) -> int:
    """Parse vote value string to integer."""
    if not val:
        return 0
    val = val.replace(",", "").replace("%", "").strip()
    if not val or val == "-":
        return 0
    try:
        return int(float(val))
    except ValueError:
        return 0


def parse_pct_value(val: str) -> float:
    """Parse percentage value string to float."""
    if not val:
        return 0.0
    val = val.replace("%", "").strip()
    if not val or val == "-":
        return 0.0
    try:
        return float(val)
    except ValueError:
        return 0.0


def normalize_division_code(div_text: str) -> str:
    """
    Extract polling division code from section header.
    'A1 (a) - GROS ISLET PRIMARY SCHOOL' -> 'A1(a)'
    """
    if not div_text:
        return ""
    # Match pattern like 'A1 (a)', 'B2', 'J5', etc.
    match = re.match(r'^([A-Z]\d+)\s*(?:\(([a-z])\))?', div_text.strip())
    if match:
        base = match.group(1)
        suffix = match.group(2)
        if suffix:
            return f"{base}({suffix})"
        return base
    return ""


def extract_district_results(html: str, district_id: int) -> Optional[Dict]:
    """
    Parse a single district page into granular data structure with ballot box progress.

    TERMINOLOGY:
    - Polling Division: A location (e.g., "GROS ISLET PRIMARY SCHOOL") with code like A1(a)
    - Ballot Box: A surname bucket WITHIN a polling division (e.g., "A-B", "C-D")
    - Each polling division contains multiple ballot boxes separated by surname ranges

    Args:
        html: Raw HTML content
        district_id: District ID (1-17)

    Returns:
        {
            'district_id': int,
            'constituency': str,
            'candidates': [{'name': str, 'party': str}],
            'divisions': {
                'A1(a)': {
                    'name': 'GROS ISLET PRIMARY SCHOOL',
                    'buckets': [
                        {'range': 'A-B', 'electors': 397, 'votes': [0, 0], 'rejected': 0, 'votes_cast': 0}
                    ],
                    'electors': int,          # Total electors in this polling division
                    'votes_cast': int,        # Total votes cast in this polling division
                    'is_counted': bool,       # Whether any ballot box in this division is counted
                    'votes_remaining': int,   # Potential remaining votes (electors - votes_cast)
                    'buckets_total': int,     # Number of ballot boxes in this division
                    'buckets_counted': int    # Number of ballot boxes counted in this division
                }
            },
            'total_electors': int,
            'total_votes_cast': int,
            # Polling division level tracking
            'total_polling_divisions': int,       # Number of polling locations
            'polling_divisions_counted': int,     # Divisions with at least one counted ballot box
            'polling_divisions_remaining': int,   # Divisions with no counted ballot boxes
            'counted_divisions': [list],          # Division codes that have been counted
            'uncounted_divisions': [list],        # Division codes not yet counted
            # Ballot box (surname bucket) level tracking
            'total_ballot_boxes': int,            # Total surname buckets across all divisions
            'ballot_boxes_counted': int,          # Buckets with votes_cast > 0
            'ballot_boxes_remaining': int,        # Buckets not yet counted
            'counted_ballot_boxes': [list],       # List of "division:range" for counted boxes
            'uncounted_ballot_boxes': [list],     # List of "division:range" for uncounted boxes
            'votes_remaining': int                # Potential remaining votes constituency-wide
        }
    """
    soup = BeautifulSoup(html, "html.parser")
    tables = soup.find_all("table")

    if not tables:
        return None

    # Use the main data table (usually the largest one)
    main_table = max(tables, key=lambda t: len(t.find_all("tr")))
    rows = main_table.find_all("tr")

    if len(rows) < 3:
        return None

    # Parse header row to get candidate names
    candidates = []
    header_row = rows[0]
    header_cells = header_row.find_all(["th", "td"])

    for cell in header_cells[1:]:  # Skip first empty cell
        text = clean_text(cell.get_text())
        if text and "SLP" in text or "UWP" in text or "IND" in text or "NCP" in text:
            # Parse "Name, Full NamePARTY" format
            # Try to extract party code
            party_match = re.search(r'(SLP|UWP|IND|NCP|NGP)$', text)
            if party_match:
                party = party_match.group(1)
                name = text[:party_match.start()].strip().rstrip(',').strip()
                # Extract just the surname
                name_parts = name.split(',')
                surname = name_parts[0].strip() if name_parts else name
                candidates.append({'name': surname.upper(), 'party': party})

    # If we didn't find candidates in expected format, try alternate parsing
    if len(candidates) < 2:
        candidates = []
        for cell in header_cells:
            text = clean_text(cell.get_text())
            for party in ['SLP', 'UWP', 'IND', 'NCP', 'NGP']:
                if party in text:
                    # Extract name before party
                    idx = text.find(party)
                    name = text[:idx].strip().rstrip(',').strip()
                    if name:
                        name_parts = name.split(',')
                        surname = name_parts[0].strip() if name_parts else name
                        candidates.append({'name': surname.upper(), 'party': party})
                    break

    # Parse data rows
    divisions = {}
    current_division = None
    current_division_name = ""
    total_electors = 0

    for row in rows[2:]:  # Skip header rows
        cells = row.find_all(["th", "td"])
        if not cells:
            continue

        values = [clean_text(c.get_text()) for c in cells]
        if not values or not values[0]:
            continue

        first_val = values[0]

        # Check if this is a division header row (single cell spanning the row)
        if len(cells) == 1 or (len(values) >= 2 and not values[1]):
            # This might be a division header like "A1 (a) - SCHOOL NAME"
            div_code = normalize_division_code(first_val)
            if div_code:
                current_division = div_code
                # Extract full name (after the code)
                name_match = re.search(r'-\s*(.+)$', first_val)
                current_division_name = name_match.group(1).strip() if name_match else ""
                divisions[current_division] = {
                    'name': current_division_name,
                    'buckets': []
                }
            continue

        # Check if this is a bucket row (surname range like 'A-B', 'C-D')
        if current_division and re.match(r'^[A-Z]+-[A-Z]+$|^[A-Z]{3}-[A-Z]{3}$', first_val):
            # This is a surname bucket row
            # Expected columns: Range, Electors, Votes1, %1, Votes2, %2, Rejected, Rej%, VotesCast, Cast%, NotCast, NotCast%

            bucket = {
                'range': first_val,
                'electors': parse_vote_value(values[1]) if len(values) > 1 else 0,
                'votes': [],
                'rejected': 0,
                'votes_cast': 0
            }

            # Parse votes for each candidate
            # Columns are: Electors, [Votes, %] per candidate, Rejected, Rejected%, Votes Cast, ...
            col_idx = 2  # Start after Range and Electors
            for _ in candidates:
                if col_idx < len(values):
                    bucket['votes'].append(parse_vote_value(values[col_idx]))
                    col_idx += 2  # Skip percentage column
                else:
                    bucket['votes'].append(0)

            # Parse rejected (after candidate columns)
            if col_idx < len(values):
                bucket['rejected'] = parse_vote_value(values[col_idx])

            # Parse votes cast
            if col_idx + 2 < len(values):
                bucket['votes_cast'] = parse_vote_value(values[col_idx + 2])

            total_electors += bucket['electors']

            if current_division in divisions:
                divisions[current_division]['buckets'].append(bucket)

    # Get constituency name
    constituency = DISTRICT_NAMES.get(district_id, f"District {district_id}")

    # Try to get constituency from page title
    title = soup.find('title')
    if title:
        title_text = clean_text(title.get_text())
        # Extract constituency name from title if present

    # Calculate progress for each division and track ballot boxes (surname buckets)
    total_votes_cast = 0
    counted_divisions = []
    uncounted_divisions = []

    # Track ballot boxes (surname buckets) separately from polling divisions
    total_ballot_boxes = 0
    ballot_boxes_counted = 0
    counted_ballot_boxes = []  # List of (division_code, bucket_range) tuples
    uncounted_ballot_boxes = []

    for div_code, div_data in divisions.items():
        # Sum up the division totals from buckets
        div_electors = sum(b['electors'] for b in div_data['buckets'])
        div_votes_cast = sum(b['votes_cast'] for b in div_data['buckets'])

        # Count buckets (actual ballot boxes)
        div_buckets_total = len(div_data['buckets'])
        div_buckets_counted = 0

        for bucket in div_data['buckets']:
            total_ballot_boxes += 1
            if bucket['votes_cast'] > 0:
                ballot_boxes_counted += 1
                div_buckets_counted += 1
                counted_ballot_boxes.append(f"{div_code}:{bucket['range']}")
            else:
                uncounted_ballot_boxes.append(f"{div_code}:{bucket['range']}")

        div_data['electors'] = div_electors
        div_data['votes_cast'] = div_votes_cast
        div_data['is_counted'] = div_votes_cast > 0
        div_data['votes_remaining'] = max(0, div_electors - div_votes_cast)
        div_data['buckets_total'] = div_buckets_total
        div_data['buckets_counted'] = div_buckets_counted

        total_votes_cast += div_votes_cast

        if div_votes_cast > 0:
            counted_divisions.append(div_code)
        else:
            uncounted_divisions.append(div_code)

    return {
        'district_id': district_id,
        'constituency': constituency,
        'candidates': candidates,
        'divisions': divisions,
        'total_electors': total_electors,
        'total_votes_cast': total_votes_cast,
        # Polling division level tracking
        'total_polling_divisions': len(divisions),
        'polling_divisions_counted': len(counted_divisions),
        'polling_divisions_remaining': len(uncounted_divisions),
        'counted_divisions': sorted(counted_divisions),
        'uncounted_divisions': sorted(uncounted_divisions),
        # Ballot box (surname bucket) level tracking
        'total_ballot_boxes': total_ballot_boxes,
        'ballot_boxes_counted': ballot_boxes_counted,
        'ballot_boxes_remaining': total_ballot_boxes - ballot_boxes_counted,
        'counted_ballot_boxes': sorted(counted_ballot_boxes),
        'uncounted_ballot_boxes': sorted(uncounted_ballot_boxes),
        'votes_remaining': max(0, total_electors - total_votes_cast)
    }


def fetch_district(district_id: int) -> Tuple[int, Optional[Dict], Optional[str]]:
    """
    Fetch and parse a single district page.

    Returns:
        (district_id, data, error_message)
    """
    url = f"{BASE_URL}{district_id}"
    last_error = None

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(url, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()

            data = extract_district_results(response.text, district_id)
            if data:
                return district_id, data, None
            else:
                return district_id, None, "Failed to parse district data"

        except requests.exceptions.Timeout:
            last_error = f"Timeout after {REQUEST_TIMEOUT}s"
        except requests.exceptions.ConnectionError as e:
            last_error = f"Connection error: {e}"
        except requests.exceptions.HTTPError as e:
            last_error = f"HTTP error: {e}"
        except Exception as e:
            last_error = f"Error: {e}"

        if attempt < MAX_RETRIES - 1:
            time.sleep(RETRY_DELAYS[attempt])

    return district_id, None, last_error


def scrape_all_districts(
    district_ids: List[int] = None,
    parallel: bool = True
) -> Tuple[Dict[int, Dict], List[str]]:
    """
    Scrape all district pages.

    Args:
        district_ids: List of district IDs to scrape (default: 1-17)
        parallel: Whether to use parallel fetching

    Returns:
        ({district_id: data}, [error_messages])
    """
    if district_ids is None:
        district_ids = DISTRICT_IDS

    results = {}
    errors = []

    if parallel:
        with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as executor:
            futures = {}
            for district_id in district_ids:
                future = executor.submit(fetch_district, district_id)
                futures[future] = district_id
                time.sleep(DELAY_BETWEEN_REQUESTS)  # Rate limiting

            for future in as_completed(futures):
                district_id, data, error = future.result()
                if data:
                    results[district_id] = data
                if error:
                    errors.append(f"District {district_id}: {error}")
    else:
        for district_id in district_ids:
            print(f"  Fetching district {district_id}...")
            district_id, data, error = fetch_district(district_id)
            if data:
                results[district_id] = data
            if error:
                errors.append(f"District {district_id}: {error}")
            time.sleep(DELAY_BETWEEN_REQUESTS)

    return results, errors


def aggregate_to_election_format(granular_data: Dict[int, Dict]) -> List[Dict]:
    """
    Aggregate granular data to polling division level in standard election format.

    Sums surname buckets within each polling division to produce data compatible
    with existing election_night_results format.

    Args:
        granular_data: {district_id: district_data}

    Returns:
        List of records in standard election_night_results format
    """
    all_records = []

    for district_id in sorted(granular_data.keys()):
        data = granular_data[district_id]
        constituency = data['constituency']
        candidates = data['candidates']
        divisions = data['divisions']

        # Calculate totals per candidate across all divisions
        candidate_totals = [0] * len(candidates)
        total_votes = 0
        total_electors = 0
        total_rejected = 0

        # Aggregate each division
        division_votes = {}  # {div_code: {candidate_idx: votes}}

        for div_code, div_data in divisions.items():
            division_votes[div_code] = [0] * len(candidates)
            div_rejected = 0

            for bucket in div_data['buckets']:
                for i, votes in enumerate(bucket['votes']):
                    division_votes[div_code][i] += votes
                    candidate_totals[i] += votes
                    total_votes += votes
                div_rejected += bucket['rejected']
                total_electors += bucket['electors']

            total_rejected += div_rejected

        # Create candidate records
        first_candidate = True
        for i, candidate in enumerate(candidates):
            record = {
                'district': constituency,
                'District': constituency if first_candidate else '',
                'Candidate': candidate['name'],
                'Party': candidate['party']
            }

            # Add division votes
            for div_code in sorted(divisions.keys()):
                record[div_code] = str(division_votes[div_code][i])

            # Add totals
            record['Total'] = str(candidate_totals[i])
            pct = (candidate_totals[i] / total_votes * 100) if total_votes > 0 else 0
            record['% Total'] = f"{pct:.2f}"

            all_records.append(record)
            first_candidate = False

        # Add summary records
        all_records.append({
            'district': constituency,
            'summary_label': 'No. of Electors',
            'summary_value': str(total_electors)
        })
        all_records.append({
            'district': constituency,
            'summary_label': 'Votes Cast',
            'summary_value': str(total_votes + total_rejected)
        })
        turnout = ((total_votes + total_rejected) / total_electors * 100) if total_electors > 0 else 0
        all_records.append({
            'district': constituency,
            'summary_label': 'Turnout %',
            'summary_value': f"{turnout:.2f}"
        })
        all_records.append({
            'district': constituency,
            'summary_label': 'Rejected',
            'summary_value': str(total_rejected)
        })

    return all_records


def save_granular_snapshot(
    granular_data: Dict[int, Dict],
    output_dir: Path,
    timestamp: datetime
) -> str:
    """Save raw granular data as individual district files with ballot box progress."""
    snapshot_dir = output_dir / f"snapshot_{timestamp.strftime('%Y-%m-%d_%H-%M-%S')}"
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    for district_id, data in granular_data.items():
        filename = f"district_{district_id:02d}_{data['constituency'].lower().replace(' ', '_').replace('/', '_')}.json"
        filepath = snapshot_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    # Calculate aggregate ballot box progress
    total_ballot_boxes = sum(d.get('total_ballot_boxes', 0) for d in granular_data.values())
    ballot_boxes_counted = sum(d.get('ballot_boxes_counted', 0) for d in granular_data.values())
    total_polling_divisions = sum(d.get('total_polling_divisions', 0) for d in granular_data.values())
    polling_divisions_counted = sum(d.get('polling_divisions_counted', 0) for d in granular_data.values())
    total_votes_cast = sum(d.get('total_votes_cast', 0) for d in granular_data.values())
    total_electors = sum(d.get('total_electors', 0) for d in granular_data.values())

    # Save metadata with ballot box progress summary
    metadata = {
        'timestamp': timestamp.isoformat(),
        'districts_scraped': len(granular_data),
        'district_ids': sorted(granular_data.keys()),
        'progress': {
            'total_ballot_boxes': total_ballot_boxes,
            'ballot_boxes_counted': ballot_boxes_counted,
            'ballot_boxes_remaining': total_ballot_boxes - ballot_boxes_counted,
            'ballot_box_pct': round(ballot_boxes_counted / total_ballot_boxes * 100, 1) if total_ballot_boxes > 0 else 0,
            'total_polling_divisions': total_polling_divisions,
            'polling_divisions_counted': polling_divisions_counted,
            'total_electors': total_electors,
            'total_votes_cast': total_votes_cast,
            'turnout_pct': round(total_votes_cast / total_electors * 100, 1) if total_electors > 0 else 0
        },
        'per_constituency': {
            d['constituency']: {
                'ballot_boxes_counted': d.get('ballot_boxes_counted', 0),
                'ballot_boxes_total': d.get('total_ballot_boxes', 0),
                'votes_cast': d.get('total_votes_cast', 0),
                'electors': d.get('total_electors', 0)
            }
            for d in granular_data.values()
        }
    }
    with open(snapshot_dir / 'metadata.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)

    return str(snapshot_dir)


def save_aggregated_results(
    data: List[Dict],
    output_dir: Path,
    timestamp: datetime
) -> str:
    """Save aggregated results in standard format."""
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = f"results_{timestamp.strftime('%Y-%m-%d_%H-%M-%S')}.json"
    filepath = output_dir / filename

    output = {
        'metadata': {
            'scraped_at': timestamp.isoformat(),
            'source_type': 'GRANULAR',
            'source_url': BASE_URL,
            'record_count': len(data)
        },
        'results': data
    }

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    return str(filepath)


def main():
    parser = argparse.ArgumentParser(
        description="Scrape granular election results from all district pages"
    )
    parser.add_argument(
        "--output-dir",
        default="data/granular_snapshots",
        help="Output directory for granular data"
    )
    parser.add_argument(
        "--results-dir",
        default="data/live_election_results",
        help="Output directory for aggregated results"
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Scrape once and exit"
    )
    parser.add_argument(
        "--no-parallel",
        action="store_true",
        help="Disable parallel fetching"
    )
    parser.add_argument(
        "--districts",
        type=int,
        nargs="+",
        help="Specific district IDs to scrape"
    )

    args = parser.parse_args()

    district_ids = args.districts if args.districts else DISTRICT_IDS
    print(f"Scraping {len(district_ids)} district pages...")

    granular_data, errors = scrape_all_districts(
        district_ids=district_ids,
        parallel=not args.no_parallel
    )

    if errors:
        print(f"Errors: {errors}")

    timestamp = datetime.now()

    # Save granular snapshot
    snapshot_path = save_granular_snapshot(
        granular_data,
        Path(args.output_dir),
        timestamp
    )
    print(f"Saved granular snapshot to {snapshot_path}")

    # Aggregate and save
    aggregated = aggregate_to_election_format(granular_data)
    results_path = save_aggregated_results(
        aggregated,
        Path(args.results_dir),
        timestamp
    )
    print(f"Saved aggregated results ({len(aggregated)} records) to {results_path}")

    # Summary
    total_divisions = sum(d.get('total_polling_divisions', len(d['divisions'])) for d in granular_data.values())
    total_boxes = sum(d.get('total_ballot_boxes', 0) for d in granular_data.values())
    counted_boxes = sum(d.get('ballot_boxes_counted', 0) for d in granular_data.values())

    print(f"\nSummary:")
    print(f"  Districts scraped: {len(granular_data)}/{len(district_ids)}")
    print(f"  Total candidates: {sum(len(d['candidates']) for d in granular_data.values())}")
    print(f"  Total polling divisions: {total_divisions}")
    print(f"  Total ballot boxes (surname buckets): {total_boxes}")
    if total_boxes > 0:
        print(f"  Ballot boxes counted: {counted_boxes}/{total_boxes} ({counted_boxes/total_boxes*100:.1f}%)")

    return 0 if len(granular_data) == len(district_ids) else 1


if __name__ == "__main__":
    exit(main())
