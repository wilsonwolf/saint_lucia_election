#!/usr/bin/env python3
"""
Primary Source Scraper - results.sluelectoral.com/summary.php

Scrapes election results from the primary Electoral Department results page.
This is the preferred data source as it contains all constituencies in one page.

Output format matches existing election_night_results schema for compatibility.

Usage:
    python scrape_results_slu.py [--once] [--output-dir PATH]
"""

import argparse
import hashlib
import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
from bs4 import BeautifulSoup

# Configuration
PRIMARY_URL = "https://results.sluelectoral.com/summary.php"
REQUEST_TIMEOUT = 60
MAX_RETRIES = 3
RETRY_DELAYS = [10, 30, 60]


def clean_text(text: str) -> str:
    """Clean text by removing extra whitespace and special characters."""
    if not text:
        return ""
    text = text.replace("\xa0", " ").replace("\u2013", "-").replace("–", "-")
    return " ".join(text.split()).strip()


def normalize_polling_division(div: str) -> str:
    """
    Normalize polling division code for consistency.
    'A1 (a)' -> 'A1(a)'
    """
    if not div:
        return ""
    # Remove extra spaces around parentheses
    div = re.sub(r'\s*\(\s*', '(', div)
    div = re.sub(r'\s*\)\s*', ')', div)
    return div.strip()


def extract_summary_results(html: str) -> List[Dict]:
    """
    Parse summary.php HTML into election_night_results format.

    Each table represents one constituency. Converts to the same format
    as existing st_lucia_YYYY_full_results.json files.

    Args:
        html: Raw HTML content

    Returns:
        List of records (candidates and summaries) in standard format
    """
    soup = BeautifulSoup(html, "html.parser")
    tables = soup.find_all("table")

    all_data = []

    for table in tables:
        rows = table.find_all("tr")
        if len(rows) < 3:
            continue

        # Find header row (contains District, Candidate, Party, polling divisions)
        header_row = None
        header_idx = -1
        for idx, row in enumerate(rows[:3]):
            cells = row.find_all(["th", "td"])
            cell_texts = [clean_text(c.get_text()) for c in cells]
            if any("District" in t for t in cell_texts) and any("Candidate" in t for t in cell_texts):
                header_row = row
                header_idx = idx
                break

        if header_row is None:
            continue

        # Extract column headers
        header_cells = header_row.find_all(["th", "td"])
        headers = [clean_text(c.get_text()) for c in header_cells]

        # Find polling division columns (everything after Party, before Total)
        polling_divisions = []
        for i, h in enumerate(headers):
            if h not in ["", "District", "Candidate", "Party", "Total", "% Total", "%"]:
                # Normalize the polling division name
                normalized = normalize_polling_division(h)
                if normalized:
                    polling_divisions.append((i, normalized))

        # Find Total and % Total column indices
        total_idx = None
        pct_idx = None
        for i, h in enumerate(headers):
            if h == "Total":
                total_idx = i
            elif h in ["% Total", "%"]:
                pct_idx = i

        # Process candidate rows and summary rows
        current_district = None
        first_candidate = True

        for row in rows[header_idx + 1:]:
            cells = row.find_all(["th", "td"])
            if not cells:
                continue

            values = [clean_text(c.get_text()) for c in cells]
            if len(values) < 3:
                continue

            # Check if this is a summary row
            first_val = values[0] if values else ""
            if any(label in first_val for label in [
                "No. of Electors", "No. Of Electors",
                "Votes Cast", "Turnout", "Rejected"
            ]):
                # Find the value (usually in the second non-empty cell)
                summary_value = ""
                for v in values[1:]:
                    if v and v.strip() and v.strip() not in ["", "-"]:
                        summary_value = v.strip()
                        break

                all_data.append({
                    "district": current_district,
                    "summary_label": first_val,
                    "summary_value": summary_value
                })
                continue

            # Check if this is a candidate row
            # District name appears in first column only for first candidate
            if values[0] and values[0] not in ["", "-"]:
                current_district = values[0]
                first_candidate = True

            # Need at least candidate name and party
            candidate_name = values[1] if len(values) > 1 else ""
            party = values[2] if len(values) > 2 else ""

            if not candidate_name or not party:
                continue

            # Build candidate record
            record = {
                "district": current_district,
                "District": current_district if first_candidate else "",
                "Candidate": candidate_name.upper(),
                "Party": party.strip()
            }

            # Add polling division votes
            for col_idx, div_name in polling_divisions:
                if col_idx < len(values):
                    vote_val = values[col_idx]
                    # Clean the vote value
                    if vote_val and vote_val.strip() and vote_val.strip() not in ["-", ""]:
                        record[div_name] = vote_val.strip()
                    else:
                        record[div_name] = ""
                else:
                    record[div_name] = ""

            # Add Total and % Total
            if total_idx and total_idx < len(values):
                record["Total"] = values[total_idx] if values[total_idx] else ""
            else:
                record["Total"] = ""

            if pct_idx and pct_idx < len(values):
                record["% Total"] = values[pct_idx] if values[pct_idx] else ""
            else:
                record["% Total"] = ""

            all_data.append(record)
            first_candidate = False

    return all_data


def compute_content_hash(data: List[Dict]) -> str:
    """Compute MD5 hash of data for change detection."""
    content = json.dumps(data, sort_keys=True, ensure_ascii=False)
    return hashlib.md5(content.encode("utf-8")).hexdigest()


def fetch_with_retry(url: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Fetch URL content with retry logic.

    Returns:
        (html_content, error_message)
    """
    last_error = None

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(url, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            return response.text, None

        except requests.exceptions.Timeout:
            last_error = f"Timeout after {REQUEST_TIMEOUT}s"
        except requests.exceptions.ConnectionError as e:
            last_error = f"Connection error: {e}"
        except requests.exceptions.HTTPError as e:
            last_error = f"HTTP error: {e}"
        except Exception as e:
            last_error = f"Unexpected error: {e}"

        if attempt < MAX_RETRIES - 1:
            delay = RETRY_DELAYS[attempt]
            print(f"  [RETRY] Attempt {attempt + 1} failed: {last_error}")
            print(f"  [RETRY] Waiting {delay}s before retry...")
            time.sleep(delay)

    return None, last_error


def scrape_once(url: str = PRIMARY_URL) -> Tuple[Optional[List[Dict]], Optional[str]]:
    """
    Perform a single scrape operation.

    Returns:
        (data, error_message)
    """
    html, error = fetch_with_retry(url)

    if html is None:
        return None, error

    try:
        data = extract_summary_results(html)
        if not data:
            return None, "No data extracted (page may be empty or format changed)"
        return data, None
    except Exception as e:
        return None, f"Parsing error: {e}"


def validate_data(data: List[Dict]) -> Tuple[bool, List[str]]:
    """
    Validate scraped data has expected structure.

    Returns:
        (is_valid, list_of_issues)
    """
    issues = []

    if not data:
        issues.append("No data records")
        return False, issues

    # Check for candidate records
    candidate_records = [r for r in data if "Candidate" in r and r.get("Candidate")]
    if not candidate_records:
        issues.append("No candidate records found")

    # Check for multiple constituencies
    districts = set(r.get("district") for r in data if r.get("district"))
    if len(districts) < 10:
        issues.append(f"Only {len(districts)} constituencies found (expected ~17)")

    # Check for summary records
    summary_records = [r for r in data if "summary_label" in r]
    if len(summary_records) < 17:
        issues.append(f"Only {len(summary_records)} summary records found")

    return len(issues) == 0, issues


def save_results(
    data: List[Dict],
    output_dir: Path,
    timestamp: datetime,
    url: str
) -> str:
    """Save results to a timestamped JSON file."""
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = f"results_{timestamp.strftime('%Y-%m-%d_%H-%M-%S')}.json"
    filepath = output_dir / filename

    output = {
        "metadata": {
            "scraped_at": timestamp.isoformat(),
            "source_url": url,
            "source_type": "PRIMARY",
            "record_count": len(data)
        },
        "results": data
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    return str(filepath)


def main():
    parser = argparse.ArgumentParser(
        description="Scrape primary election results from results.sluelectoral.com"
    )
    parser.add_argument(
        "--url",
        default=PRIMARY_URL,
        help=f"URL to scrape (default: {PRIMARY_URL})"
    )
    parser.add_argument(
        "--output-dir",
        default="data/live_election_results",
        help="Output directory"
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Scrape once and exit"
    )

    args = parser.parse_args()
    output_dir = Path(args.output_dir)

    print(f"Scraping from {args.url}...")
    data, error = scrape_once(args.url)

    if error:
        print(f"ERROR: {error}")
        return 1

    # Validate
    is_valid, issues = validate_data(data)
    if issues:
        print(f"Validation issues: {issues}")

    # Save
    timestamp = datetime.now()
    filepath = save_results(data, output_dir, timestamp, args.url)
    print(f"Saved {len(data)} records to {filepath}")

    # Show sample
    candidates = [r for r in data if "Candidate" in r]
    print(f"  Candidate records: {len(candidates)}")
    print(f"  Summary records: {len(data) - len(candidates)}")

    districts = set(r.get("district") for r in data if r.get("district"))
    print(f"  Constituencies: {len(districts)}")

    return 0


if __name__ == "__main__":
    exit(main())
