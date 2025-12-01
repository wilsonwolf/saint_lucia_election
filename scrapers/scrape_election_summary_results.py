"""
Scrape election summary results and vote distribution from the Saint Lucia Electoral Department website.

This script extracts:
1. Summary results: Constituency-level vote breakdowns by party
2. Vote distribution: National totals by party (votes, percentage, candidates, seats)

Output files:
- data/summary_results/saint_lucia_{year}_summary_results.json
- data/vote_distribution/saint_lucia_{year}_vote_distribution.json

Usage:
    python scrapers/scrape_election_summary_results.py
"""

import requests
from bs4 import BeautifulSoup
import json
import re
import time
from pathlib import Path


# All election result pages to scrape
ELECTION_LINKS = [
    ("2021", "https://www.sluelectoral.com/2021-election-results/"),
    ("2016", "https://www.sluelectoral.com/past-results/2016-election-results/"),
    ("2011", "https://www.sluelectoral.com/past-results/2011-election-results/"),
    ("2006", "https://www.sluelectoral.com/past-results/2006-election-results/"),
    ("2001", "https://www.sluelectoral.com/past-results/2001-election-results/"),
    ("1997", "https://www.sluelectoral.com/past-results/1997-election-results/"),
    ("1992", "https://www.sluelectoral.com/past-results/election-results-1992/"),
    ("1987_apr6", "https://www.sluelectoral.com/past-results/election-results-1987/"),
    ("1987_apr30", "https://www.sluelectoral.com/past-results/1987-april-30-election-results/"),
    ("1982", "https://www.sluelectoral.com/past-results/election-results-1982/"),
    ("1979", "https://www.sluelectoral.com/past-results/election-results-1979/"),
]

# Known party codes that can appear in tables
KNOWN_PARTY_CODES = ["SLP", "UWP", "PLP", "LPM", "NGP", "IND", "PDM", "NDP", "SDP", "NPM"]


def clean_text(text: str) -> str:
    """Clean text by removing extra whitespace and special characters."""
    if not text:
        return ""
    text = text.replace("\xa0", " ").replace("\u2013", "-").replace("–", "-")
    return " ".join(text.split()).strip()


def parse_vote_value(val: str):
    """Parse a vote value from string to int or None."""
    if not val:
        return None
    val = clean_text(str(val))
    val = val.replace(",", "").replace("*", "").replace("-", "").strip()
    if not val or val.lower() in ["nan", "null", ""]:
        return None
    try:
        return int(float(val))
    except ValueError:
        return None


def parse_float_value(val: str):
    """Parse a float value from string (for turnout percentages)."""
    if not val:
        return None
    val = clean_text(str(val))
    val = val.replace(",", "").replace("%", "").strip()
    if not val or val.lower() in ["nan", "null", ""]:
        return None
    try:
        return float(val)
    except ValueError:
        return None


def get_soup(url: str) -> BeautifulSoup:
    """Fetch a page and return BeautifulSoup object."""
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return BeautifulSoup(response.text, "html.parser")


def is_constituency_table(headers: list[str]) -> bool:
    """Check if this table contains constituency-level results."""
    header_text = " ".join(headers).upper()
    return "CONSTITUENCY" in header_text and any(p in header_text for p in KNOWN_PARTY_CODES)


def is_vote_distribution_table(headers: list[str]) -> bool:
    """Check if this table contains vote distribution (party totals)."""
    header_text = " ".join(headers).upper()
    # Vote distribution tables typically have "SEATS" and "CANDIDATES" columns
    return ("SEATS" in header_text or "SEAT" in header_text) and \
           ("CANDIDATES" in header_text or "CANDIDATE" in header_text)


def extract_party_codes_from_headers(headers: list[str]) -> list[str]:
    """Extract party codes from table headers."""
    parties = []
    for h in headers:
        h_clean = clean_text(h).upper()
        for code in KNOWN_PARTY_CODES:
            if code == h_clean:
                parties.append(code)
                break
    return parties


def find_header_row(rows: list) -> tuple[int, list[str]]:
    """
    Find the header row in a table.
    Returns (row_index, headers).
    Some tables have a title row first, so headers may be in row 0 or row 1.
    """
    for i, row in enumerate(rows[:3]):  # Check first 3 rows
        cells = row.find_all(["th", "td"])
        headers = [clean_text(cell.get_text()) for cell in cells]

        # Header row has multiple columns and contains "Constituency"
        if len(headers) >= 3 and any("CONSTITUENCY" in h.upper() for h in headers):
            return i, headers

    return -1, []


def parse_constituency_table(table) -> list[dict]:
    """Parse a constituency results table."""
    rows = table.find_all("tr")
    if not rows:
        return []

    # Find the actual header row (may not be row 0)
    header_idx, headers = find_header_row(rows)
    if header_idx < 0 or not headers:
        return []

    # Check if this is actually a constituency table
    if not is_constituency_table(headers):
        return []

    # Map column indices to field names
    col_map = {}
    party_columns = []

    for i, h in enumerate(headers):
        h_upper = h.upper()
        if "CONSTITUENCY" in h_upper:
            col_map[i] = "Constituency"
        elif "REGISTERED" in h_upper:
            col_map[i] = "Registered Votes"
        elif "REJECTED" in h_upper:
            col_map[i] = "Rejected"
        elif "CAST" in h_upper:
            col_map[i] = "Votes Cast"
        elif "TURNOUT" in h_upper:
            col_map[i] = "Turnout"
        elif "TOTAL" in h_upper and "VOTE" in h_upper:
            col_map[i] = "Votes Cast"
        elif h_upper in KNOWN_PARTY_CODES:
            col_map[i] = h_upper
            party_columns.append(h_upper)

    # Parse data rows (start after header row)
    results = []
    for row in rows[header_idx + 1:]:
        cells = row.find_all(["th", "td"])
        if len(cells) < 2:
            continue

        values = [clean_text(cell.get_text()) for cell in cells]

        # Skip if no constituency name
        constituency = None
        for i, v in enumerate(values):
            if col_map.get(i) == "Constituency" and v:
                constituency = v
                break

        if not constituency:
            continue

        # Build row data
        row_data = {"Constituency": constituency}

        for i, v in enumerate(values):
            field = col_map.get(i)
            if not field or field == "Constituency":
                continue

            if field == "Turnout":
                row_data[field] = parse_float_value(v)
            elif field in KNOWN_PARTY_CODES:
                row_data[field] = parse_vote_value(v)
            else:
                row_data[field] = parse_vote_value(v)

        # Ensure all party columns exist (set to null if missing)
        for party in party_columns:
            if party not in row_data:
                row_data[party] = None

        results.append(row_data)

    return results


def find_vote_distribution_header_row(rows: list) -> tuple[int, list[str]]:
    """
    Find the header row in a vote distribution table.
    Returns (row_index, headers).
    """
    for i, row in enumerate(rows[:3]):  # Check first 3 rows
        cells = row.find_all(["th", "td"])
        headers = [clean_text(cell.get_text()) for cell in cells]

        # Header row has multiple columns and contains "Seats" or "Candidates"
        header_text = " ".join(headers).upper()
        if len(headers) >= 3 and ("SEATS" in header_text or "SEAT" in header_text):
            return i, headers

    return -1, []


def parse_vote_distribution_table(table) -> list[dict]:
    """Parse a vote distribution (party totals) table."""
    rows = table.find_all("tr")
    if not rows:
        return []

    # Find the actual header row
    header_idx, headers = find_vote_distribution_header_row(rows)
    if header_idx < 0 or not headers:
        return []

    # Check if this is actually a vote distribution table
    if not is_vote_distribution_table(headers):
        return []

    # Map column indices
    col_map = {}
    for i, h in enumerate(headers):
        h_upper = h.upper()
        if "CONSTITUENCY" in h_upper or "PARTY" in h_upper:
            col_map[i] = "Party"
        elif "CODE" in h_upper:
            col_map[i] = "Code"
        elif "TOTAL" in h_upper or ("VOTE" in h_upper and "%" not in h_upper):
            col_map[i] = "Total Votes"
        elif "%" in h_upper or "PERCENT" in h_upper:
            col_map[i] = "% Votes"
        elif "CANDIDATE" in h_upper:
            col_map[i] = "Candidates"
        elif "SEAT" in h_upper:
            col_map[i] = "Seats"

    # Parse data rows (start after header row)
    results = []
    for row in rows[header_idx + 1:]:
        cells = row.find_all(["th", "td"])
        if len(cells) < 2:
            continue

        values = [clean_text(cell.get_text()) for cell in cells]

        # Build row data
        row_data = {}
        for i, v in enumerate(values):
            field = col_map.get(i)
            if not field:
                continue

            if field == "Party":
                # Normalize party names
                party_name = v
                if "LABOUR" in party_name.upper():
                    party_name = "Saint Lucia Labour Party"
                row_data[field] = party_name
            elif field == "Code":
                row_data[field] = v.upper()
            elif field == "% Votes":
                row_data[field] = parse_float_value(v)
            else:
                row_data[field] = parse_vote_value(v)

        # Skip if no party name
        if not row_data.get("Party"):
            continue

        results.append(row_data)

    return results


def scrape_election(year: str, url: str) -> tuple[list[dict], list[dict]]:
    """
    Scrape a single election page.

    Returns:
        Tuple of (summary_results, vote_distribution)
    """
    print(f"  Fetching {url}...")
    soup = get_soup(url)

    tables = soup.find_all("table")

    summary_results = []
    vote_distribution = []

    for table in tables:
        # Try to parse as constituency table
        constituency_data = parse_constituency_table(table)
        if constituency_data:
            summary_results = constituency_data
            continue

        # Try to parse as vote distribution table
        distribution_data = parse_vote_distribution_table(table)
        if distribution_data:
            vote_distribution = distribution_data
            continue

    return summary_results, vote_distribution


def save_json(data: list, filepath: Path):
    """Save data to JSON file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main():
    print("=" * 60)
    print("SCRAPING ELECTION SUMMARY RESULTS")
    print("=" * 60)

    # Output directories
    summary_dir = Path("data/summary_results")
    vote_dist_dir = Path("data/vote_distribution")

    summary_dir.mkdir(parents=True, exist_ok=True)
    vote_dist_dir.mkdir(parents=True, exist_ok=True)

    errors = []
    success_count = 0

    for year, url in ELECTION_LINKS:
        print(f"\n[{year}] Scraping election results...")

        try:
            summary_results, vote_distribution = scrape_election(year, url)

            # Save summary results
            if summary_results:
                summary_file = summary_dir / f"saint_lucia_{year}_summary_results.json"
                save_json(summary_results, summary_file)
                print(f"  Saved summary results: {summary_file}")
                print(f"    -> {len(summary_results)} constituencies")
            else:
                print(f"  WARNING: No summary results found for {year}")

            # Save vote distribution
            if vote_distribution:
                vote_dist_file = vote_dist_dir / f"saint_lucia_{year}_vote_distribution.json"
                save_json(vote_distribution, vote_dist_file)
                print(f"  Saved vote distribution: {vote_dist_file}")
                print(f"    -> {len(vote_distribution)} parties")
            else:
                print(f"  WARNING: No vote distribution found for {year}")

            success_count += 1

        except Exception as e:
            error_msg = f"Error scraping {year}: {e}"
            print(f"  ERROR: {error_msg}")
            errors.append(error_msg)
            import traceback
            traceback.print_exc()

        # Be courteous to the server
        time.sleep(1)

    print("\n" + "=" * 60)
    print("SCRAPING COMPLETE")
    print("=" * 60)
    print(f"Successfully scraped: {success_count}/{len(ELECTION_LINKS)} elections")

    if errors:
        print(f"\nErrors encountered ({len(errors)}):")
        for e in errors:
            print(f"  - {e}")


if __name__ == "__main__":
    main()
