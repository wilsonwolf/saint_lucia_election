"""
Scrape constituency mapping data from the Saint Lucia Electoral Department website.

This script extracts constituency information including:
- Constituency names
- Polling divisions within each constituency
- Communities served by each polling division
- Polling station addresses

The output is saved to data/constituency_maps/saint_lucia_constituencies.json
"""

import requests
from bs4 import BeautifulSoup
import json
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# List of constituency pages to scrape (17 total).
# These were extracted from the Saint Lucia Electoral Department site.
# If the site structure changes, update these links accordingly.
constituency_links = [
    "https://www.sluelectoral.com/electoral/constituences/gros-islet-a-information/",
    "https://www.sluelectoral.com/electoral/constituences/babonneau-b-information/",
    "https://www.sluelectoral.com/electoral/constituences/castries-north-c/",
    "https://www.sluelectoral.com/electoral/constituences/castries-east-d-information/",
    "https://www.sluelectoral.com/electoral/constituences/castries-central-e-information/",
    "https://www.sluelectoral.com/electoral/constituences/castries-south-f/",
    "https://www.sluelectoral.com/electoral/constituences/anse-la-raye-canaries-g-information/",
    "https://www.sluelectoral.com/electoral/constituences/soufriere-h-information/",
    "https://www.sluelectoral.com/electoral/constituences/choiseul-i-information/",
    "https://www.sluelectoral.com/electoral/constituences/laborie-j-information/",
    "https://www.sluelectoral.com/electoral/constituences/vieux-fort-south-k-information/",
    # Note: the following URL is spelled "inforamtion" on the site
    "https://www.sluelectoral.com/electoral/constituences/vieux-fort-north-l-inforamtion/",
    "https://www.sluelectoral.com/electoral/constituences/micoud-south-m-information/",
    "https://www.sluelectoral.com/electoral/constituences/micoud-north-n-information/",
    "https://www.sluelectoral.com/electoral/constituences/dennery-south-o/",
    "https://www.sluelectoral.com/electoral/constituences/dennery-north-p-information/",
    "https://www.sluelectoral.com/electoral/constituences/castries-south-east-q-information/"
]

def get_soup(url: str) -> BeautifulSoup:
    """Fetch the page and return a BeautifulSoup object."""
    response = requests.get(url, timeout=15)
    if response.status_code == 404:
        raise requests.HTTPError(f"404 Not Found: {url}")
    response.raise_for_status()
    return BeautifulSoup(response.text, "html.parser")

def parse_constituency(url: str) -> dict:
    """
    Given a constituency URL, return a dictionary with:
    {
      "constituency": "Name",
      "polling_divisions": [
         {"division": "...", "communities": [...], "polling_station": "..."},
         ...
      ]
    }
    """
    soup = get_soup(url)

    # Constituency name is typically in the page title <h1> or <h2>.
    title_el = soup.find(["h1", "h2"])
    constituency_name = title_el.get_text(strip=True) if title_el else url.split("/")[-2]

    # Find the polling divisions table. It usually has headings: Polling Division, Communities, Polling Station Address.
    # This selector looks for a <table> element; if there are multiple tables, use the first.
    table = soup.find("table")
    polling_data = []

    if table:
        for row in table.find_all("tr"):
            cols = [c.get_text(" ", strip=True) for c in row.find_all(["th", "td"])]
            # We need at least 3 columns: division code, communities, and station address.
            if len(cols) >= 3:
                division_code = cols[0]
                communities_raw = cols[1]
                station_address = cols[2]

                # Split community names by comma or newline; remove empty items.
                communities = [c.strip() for c in communities_raw.replace("\n", ",").split(",") if c.strip()]

                polling_data.append({
                    "division": division_code,
                    "communities": communities,
                    "polling_station": station_address
                })

    return {
        "constituency": constituency_name,
        "polling_divisions": polling_data
    }

def scrape_all_links(links: list[str]) -> list[dict]:
    """
    Loop through the provided links, parse each constituency page,
    and return a list of constituency dictionaries.
    """
    all_data = []
    errors = []
    for idx, link in enumerate(links, start=1):
        print(f"Scraping {idx}/{len(links)}: {link}")
        try:
            constituency_info = parse_constituency(link)
            all_data.append(constituency_info)
            print(f"  ✓ Successfully scraped: {constituency_info['constituency']}")
        except Exception as exc:
            error_msg = f"Error scraping {link}: {exc}"
            print(f"  ✗ {error_msg}")
            errors.append(error_msg)
        # Be courteous to the remote server
        time.sleep(1)
    
    if errors:
        print(f"\nWarning: {len(errors)} error(s) occurred during scraping")
        for error in errors:
            print(f"  - {error}")
    
    return all_data

if __name__ == "__main__":
    # Create output directory if it doesn't exist
    output_dir = Path("data/constituency_maps")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    data = scrape_all_links(constituency_links)
    
    # Save the data in JSON format to the data/constituency_maps folder
    output_file = output_dir / "saint_lucia_constituencies.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"\nFinished! Data has been written to {output_file}")
    print(f"Successfully scraped {len(data)} constituencies")
