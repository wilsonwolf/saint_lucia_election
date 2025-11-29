"""
Scrape historical election night results from the Saint Lucia Electoral Department website.

This script extracts polling division-level vote breakdowns from election night results pages.
The data is saved to data/election_night_results/ as JSON files.

Election night results contain granular breakdowns of votes by polling division,
which provides more detailed data than the summary results.
"""

import requests
from bs4 import BeautifulSoup
import json
import re
from pathlib import Path

URLS = {
    "2021": "https://www.sluelectoral.com/election-night-results-2021/",
    "2016": "https://www.sluelectoral.com/past-results/election-night-results-2016/",
    "2011": "https://www.sluelectoral.com/candidates-parties/results/",
}

def clean(t):
    return " ".join(t.replace("\xa0", " ").split()).strip()

def extract_results(url):
    html = requests.get(url).text
    soup = BeautifulSoup(html, "html.parser")

    # Find all tables (each district block is a single giant table)
    tables = soup.find_all("table")

    all_data = []
    
    for table in tables:
        rows = table.find_all("tr")
        column_headers = []
        current_district = None

        for tr in rows:
            cells = tr.find_all(["td", "th"])

            # Skip empty rows
            if not cells:
                continue

            values = [clean(c.get_text()) for c in cells]

            # Skip rows with insufficient data
            if len(values) < 2:
                continue

            # Check if row is a header row for a district block
            # Header rows always contain "District", "Candidate", "Party"
            if len(values) >= 2 and "District" in values[0] and "Candidate" in values[1]:
                column_headers = values
                continue

            # Process candidate rows
            if len(values) >= 3:
                # Candidate row rule:
                # 1. District field is populated only on the first row of the block
                if values[0] != "":
                    current_district = values[0]

                # If candidate name exists, treat row as candidate data
                if values[1] not in ["", None] and values[2] not in ["", None]:
                    entry = {"district": current_district}
                    for i, colname in enumerate(column_headers):
                        if colname == "":
                            colname = f"col_{i}"
                        entry[colname] = values[i] if i < len(values) else ""

                    all_data.append(entry)

            # Process summary rows like:
            # No. of Electors / Votes Cast / Turnout / Rejected
            if any(prefix in values[0] for prefix in [
                "No. of Electors", "No. Of Electors",
                "Votes Cast", "Turnout", "Turnout %", "Turnout%",
                "Rejected"
            ]):
                # Find the first non-empty value after the label (value might be in column 2, 3, etc.)
                summary_value = ""
                for val in values[1:]:
                    if val and val.strip() and val.strip() not in ["", "–", "\u2013"]:
                        summary_value = val.strip()
                        break
                
                all_data.append({
                    "district": current_district,
                    "summary_label": values[0],
                    "summary_value": summary_value
                })

    return all_data


# MAIN LOOP
if __name__ == "__main__":
    # Create output directory if it doesn't exist
    output_dir = Path("data/election_night_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for year, url in URLS.items():
        print(f"Scraping {year} ...")
        try:
            results = extract_results(url)
            
            filename = output_dir / f"st_lucia_{year}_full_results.json"
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            print(f"Saved → {filename}")
            print(f"  Extracted {len(results)} records")
        except Exception as e:
            print(f"Error scraping {year}: {e}")
            import traceback
            traceback.print_exc()
