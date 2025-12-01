#!/usr/bin/env python3
"""
Live Election Results Scraper

Continuously scrapes election night results from the Saint Lucia Electoral
Department website, saving timestamped JSON files when new data is detected.

Features:
- Configurable polling interval (default: 5 minutes)
- Robust error handling with exponential backoff retries
- Change detection to avoid duplicate saves
- 60-second timeout for slow/overloaded website

Usage:
    python scrape_live_results.py [--interval 300] [--output-dir data/live_election_results]
"""

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
from bs4 import BeautifulSoup

# Default configuration
DEFAULT_URL = "https://www.sluelectoral.com/election-night-results-2026/"
DEFAULT_INTERVAL = 300  # 5 minutes
DEFAULT_OUTPUT_DIR = "data/live_election_results"
REQUEST_TIMEOUT = 60  # 60 seconds for slow website
MAX_RETRIES = 3
RETRY_DELAYS = [10, 30, 60]  # Exponential backoff delays in seconds


def clean_text(text: str) -> str:
    """Clean text by removing extra whitespace and special characters."""
    if not text:
        return ""
    return " ".join(text.replace("\xa0", " ").split()).strip()


def extract_results(html: str) -> List[Dict]:
    """
    Extract election results from HTML content.

    Uses the same parsing logic as the historical scraper to ensure
    consistent data format.

    Args:
        html: Raw HTML content from the election results page

    Returns:
        List of record dictionaries (candidates and summaries)
    """
    soup = BeautifulSoup(html, "html.parser")
    tables = soup.find_all("table")

    all_data = []

    for table in tables:
        rows = table.find_all("tr")
        column_headers = []
        current_district = None

        for tr in rows:
            cells = tr.find_all(["td", "th"])

            if not cells:
                continue

            values = [clean_text(c.get_text()) for c in cells]

            if len(values) < 2:
                continue

            # Header row detection
            if len(values) >= 2 and "District" in values[0] and "Candidate" in values[1]:
                column_headers = values
                continue

            # Process candidate/data rows
            if len(values) >= 3:
                # District field is only populated on first row of block
                if values[0] != "":
                    current_district = values[0]

                # Candidate row: has name and party
                if values[1] not in ["", None] and values[2] not in ["", None]:
                    entry = {"district": current_district}
                    for i, colname in enumerate(column_headers):
                        if colname == "":
                            colname = f"col_{i}"
                        entry[colname] = values[i] if i < len(values) else ""
                    all_data.append(entry)

            # Summary rows (No. of Electors, Votes Cast, etc.)
            if any(prefix in values[0] for prefix in [
                "No. of Electors", "No. Of Electors",
                "Votes Cast", "Turnout", "Turnout %", "Turnout%",
                "% Turnout", "Rejected"
            ]):
                summary_value = ""
                for val in values[1:]:
                    if val and val.strip() and val.strip() not in ["", "-", "\u2013"]:
                        summary_value = val.strip()
                        break

                all_data.append({
                    "district": current_district,
                    "summary_label": values[0],
                    "summary_value": summary_value
                })

    return all_data


def compute_content_hash(data: List[Dict]) -> str:
    """
    Compute MD5 hash of data for change detection.

    Args:
        data: List of record dictionaries

    Returns:
        MD5 hex digest string
    """
    content = json.dumps(data, sort_keys=True, ensure_ascii=False)
    return hashlib.md5(content.encode("utf-8")).hexdigest()


def fetch_with_retry(url: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Fetch URL content with retry logic.

    Args:
        url: URL to fetch

    Returns:
        Tuple of (html_content, error_message)
        - On success: (html, None)
        - On failure: (None, error_message)
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

        # If not the last attempt, wait before retrying
        if attempt < MAX_RETRIES - 1:
            delay = RETRY_DELAYS[attempt]
            print(f"  [RETRY] Attempt {attempt + 1} failed: {last_error}")
            print(f"  [RETRY] Waiting {delay}s before retry...")
            time.sleep(delay)

    return None, last_error


def scrape_once(url: str) -> Tuple[Optional[List[Dict]], Optional[str]]:
    """
    Perform a single scrape operation.

    Args:
        url: URL to scrape

    Returns:
        Tuple of (data, error_message)
        - On success: (list_of_records, None)
        - On failure: (None, error_message)
    """
    html, error = fetch_with_retry(url)

    if html is None:
        return None, error

    try:
        data = extract_results(html)
        if not data:
            return None, "No data extracted from page (page may be empty or format changed)"
        return data, None
    except Exception as e:
        return None, f"Parsing error: {e}"


def save_results(
    data: List[Dict],
    output_dir: Path,
    timestamp: datetime,
    url: str
) -> str:
    """
    Save results to a timestamped JSON file.

    Args:
        data: List of record dictionaries
        output_dir: Directory to save to
        timestamp: Timestamp for filename
        url: Source URL (included in metadata)

    Returns:
        Path to saved file
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = f"results_{timestamp.strftime('%Y-%m-%d_%H-%M-%S')}.json"
    filepath = output_dir / filename

    # Add metadata wrapper
    output = {
        "metadata": {
            "scraped_at": timestamp.isoformat(),
            "source_url": url,
            "record_count": len(data)
        },
        "results": data
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    return str(filepath)


def run_continuous(
    url: str,
    interval: int,
    output_dir: Path,
    callback: Optional[callable] = None
) -> None:
    """
    Run continuous scraping loop.

    Args:
        url: URL to scrape
        interval: Seconds between scrapes
        output_dir: Directory to save results
        callback: Optional function to call when new data is detected
                  Signature: callback(filepath, data, timestamp)
    """
    print(f"Starting continuous scraper")
    print(f"  URL: {url}")
    print(f"  Interval: {interval}s ({interval // 60} minutes)")
    print(f"  Output: {output_dir}")
    print("-" * 60)

    last_hash = None
    scrape_count = 0
    save_count = 0

    try:
        while True:
            scrape_count += 1
            timestamp = datetime.now()
            print(f"\n[{timestamp.strftime('%H:%M:%S')}] Scrape #{scrape_count}...")

            data, error = scrape_once(url)

            if error:
                print(f"  [ERROR] {error}")
                print(f"  [INFO] Will retry in {interval}s")
            else:
                current_hash = compute_content_hash(data)

                if current_hash == last_hash:
                    print(f"  [INFO] No changes detected ({len(data)} records)")
                else:
                    save_count += 1
                    filepath = save_results(data, output_dir, timestamp, url)
                    print(f"  [NEW] Saved {len(data)} records to {filepath}")

                    if callback:
                        try:
                            callback(filepath, data, timestamp)
                        except Exception as e:
                            print(f"  [CALLBACK ERROR] {e}")

                    last_hash = current_hash

            # Wait for next interval
            next_time = datetime.now().replace(microsecond=0)
            next_time = next_time.replace(
                minute=next_time.minute + interval // 60,
                second=next_time.second + interval % 60
            )
            print(f"  [WAIT] Next scrape at {next_time.strftime('%H:%M:%S')}")
            time.sleep(interval)

    except KeyboardInterrupt:
        print(f"\n\nStopped by user")
        print(f"  Total scrapes: {scrape_count}")
        print(f"  Files saved: {save_count}")


def main():
    parser = argparse.ArgumentParser(
        description="Continuously scrape live election results"
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_URL,
        help=f"URL to scrape (default: {DEFAULT_URL})"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=DEFAULT_INTERVAL,
        help=f"Seconds between scrapes (default: {DEFAULT_INTERVAL})"
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})"
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Scrape once and exit (for testing)"
    )

    args = parser.parse_args()
    output_dir = Path(args.output_dir)

    if args.once:
        # Single scrape mode for testing
        print(f"Single scrape from {args.url}...")
        data, error = scrape_once(args.url)

        if error:
            print(f"ERROR: {error}")
            sys.exit(1)

        timestamp = datetime.now()
        filepath = save_results(data, output_dir, timestamp, args.url)
        print(f"Saved {len(data)} records to {filepath}")
    else:
        # Continuous mode
        run_continuous(args.url, args.interval, output_dir)


if __name__ == "__main__":
    main()
