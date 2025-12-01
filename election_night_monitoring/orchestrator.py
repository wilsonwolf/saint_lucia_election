#!/usr/bin/env python3
"""
Election Night Monitoring Orchestrator - Multi-Source Version

Main entry point that coordinates all monitoring scripts with redundant data sources:
1. Tries PRIMARY source (results.sluelectoral.com/summary.php) first
2. Falls back to SECONDARY source (sluelectoral.com/election-night-results-2026)
3. Falls back to GRANULAR source (district.php pages aggregated)
4. Always saves granular snapshots for archive
5. Generates swing charts when new data is detected
6. Runs monitor_live_swings.py for console analysis
7. Sends alerts on state changes

Usage:
    python orchestrator.py [--interval 300] [--thresholds PATH] [--no-alerts]

Press Ctrl+C to stop monitoring.
"""

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Import local modules
from scrape_results_slu import scrape_once as scrape_primary, validate_data as validate_primary
from scrape_live_results import scrape_once as scrape_secondary, compute_content_hash
from scrape_granular_results import scrape_all_districts, aggregate_to_election_format, save_granular_snapshot
from generate_swing_charts import generate_all_charts
from alerts import check_and_alert, format_alert_summary, send_notification
from data_reconciler import (
    validate_data,
    reconcile_sources,
    save_reconciliation_report,
    format_reconciliation_summary
)

# Configuration - Data Source URLs
PRIMARY_URL = "https://results.sluelectoral.com/summary.php"
SECONDARY_URL = "https://www.sluelectoral.com/election-night-results-2026/"

DEFAULT_INTERVAL = 300  # 5 minutes
DEFAULT_THRESHOLDS = "data/swing_thresholds/swing_thresholds_2021.json"

# Output directories
RESULTS_DIR = "data/live_election_results"
GRANULAR_DIR = "data/granular_snapshots"
RECONCILIATION_DIR = "data/reconciliation_logs"
ANALYSIS_DIR = "analysis/live_2025"


def print_header(
    start_time: datetime,
    last_update: Optional[datetime],
    next_check: datetime,
    source_used: str = "N/A"
):
    """Print the monitoring header."""
    print("\033c", end="")  # Clear screen
    print("=" * 80)
    print("SAINT LUCIA ELECTION NIGHT MONITOR - MULTI-SOURCE")
    print(f"Started: {start_time.strftime('%H:%M:%S')} | ", end="")
    if last_update:
        print(f"Last Update: {last_update.strftime('%H:%M:%S')} ({source_used}) | ", end="")
    else:
        print("Last Update: -- | ", end="")
    print(f"Next Check: {next_check.strftime('%H:%M:%S')}")
    print("=" * 80)
    print()


def scrape_with_fallback(
    timestamp: datetime,
    save_granular: bool = True
) -> Tuple[Optional[List[Dict]], str, Optional[str], Dict]:
    """
    Try data sources in priority order with fallback.

    Priority:
    1. PRIMARY (results.sluelectoral.com/summary.php)
    2. SECONDARY (sluelectoral.com/election-night-results-2026)
    3. GRANULAR (district.php pages aggregated)

    Also saves granular snapshots in background regardless of which source is used.

    Args:
        timestamp: Current timestamp for file naming
        save_granular: Whether to save granular snapshots

    Returns:
        (data, source_used, results_filepath, metadata)
    """
    metadata = {
        'primary_status': None,
        'secondary_status': None,
        'granular_status': None,
        'granular_snapshot_path': None
    }

    primary_data = None
    secondary_data = None
    granular_data = None
    granular_raw = None
    granular_aggregated = None

    # Try PRIMARY source
    print(f"  [PRIMARY] Trying {PRIMARY_URL}...")
    primary_data, primary_error = scrape_primary(PRIMARY_URL)
    if primary_error:
        metadata['primary_status'] = f"FAILED: {primary_error}"
        print(f"  [PRIMARY] Failed: {primary_error}")
    else:
        is_valid, issues = validate_data(primary_data)
        if is_valid:
            metadata['primary_status'] = f"OK ({len(primary_data)} records)"
            print(f"  [PRIMARY] Success: {len(primary_data)} records")
        else:
            metadata['primary_status'] = f"INVALID: {issues}"
            print(f"  [PRIMARY] Invalid: {issues}")
            primary_data = None  # Don't use invalid data

    # Try SECONDARY source (only if primary failed)
    if primary_data is None:
        print(f"  [SECONDARY] Trying {SECONDARY_URL}...")
        secondary_data, secondary_error = scrape_secondary(SECONDARY_URL)
        if secondary_error:
            metadata['secondary_status'] = f"FAILED: {secondary_error}"
            print(f"  [SECONDARY] Failed: {secondary_error}")
        else:
            is_valid, issues = validate_data(secondary_data)
            if is_valid:
                metadata['secondary_status'] = f"OK ({len(secondary_data)} records)"
                print(f"  [SECONDARY] Success: {len(secondary_data)} records")
            else:
                metadata['secondary_status'] = f"INVALID: {issues}"
                print(f"  [SECONDARY] Invalid: {issues}")
                secondary_data = None

    # Try GRANULAR source (only if both primary and secondary failed)
    if primary_data is None and secondary_data is None:
        print(f"  [GRANULAR] Scraping 17 district pages...")
        granular_raw, granular_errors = scrape_all_districts()
        if granular_errors:
            metadata['granular_status'] = f"PARTIAL: {len(granular_errors)} errors"
            print(f"  [GRANULAR] Partial: {len(granular_errors)} errors")
        if granular_raw:
            granular_aggregated = aggregate_to_election_format(granular_raw)
            metadata['granular_status'] = f"OK ({len(granular_aggregated)} records)"
            print(f"  [GRANULAR] Success: {len(granular_aggregated)} records")
            granular_data = granular_aggregated

    # Save granular snapshots in background (regardless of which source we use)
    if save_granular:
        try:
            print(f"  [GRANULAR] Saving snapshot...")
            # Quick scrape for archival if we didn't already scrape
            if not granular_raw:
                granular_raw, _ = scrape_all_districts()
            if granular_raw:
                snapshot_path = save_granular_snapshot(
                    granular_raw,
                    Path(GRANULAR_DIR),
                    timestamp
                )
                metadata['granular_snapshot_path'] = snapshot_path
                print(f"  [GRANULAR] Snapshot saved to {snapshot_path}")
        except Exception as e:
            print(f"  [GRANULAR] Snapshot failed: {e}")

    # Determine which source to use and save results
    if primary_data:
        source = "PRIMARY"
        data = primary_data
    elif secondary_data:
        source = "SECONDARY"
        data = secondary_data
    elif granular_data:
        source = "GRANULAR"
        data = granular_data
    else:
        return None, "NONE", None, metadata

    # Save results
    results_path = save_results_file(data, source, timestamp)

    return data, source, results_path, metadata


def save_results_file(
    data: List[Dict],
    source: str,
    timestamp: datetime
) -> str:
    """Save results to timestamped JSON file."""
    output_dir = Path(RESULTS_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = f"results_{timestamp.strftime('%Y-%m-%d_%H-%M-%S')}.json"
    filepath = output_dir / filename

    output = {
        "metadata": {
            "scraped_at": timestamp.isoformat(),
            "source_type": source,
            "record_count": len(data)
        },
        "results": data
    }

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    return str(filepath)


def run_monitor_script(results_path: str, thresholds_path: str) -> Optional[str]:
    """
    Run the monitor_live_swings.py script and capture output.

    Returns the console output or None on error.
    """
    try:
        script_path = Path(__file__).parent / "monitor_live_swings.py"
        result = subprocess.run(
            [
                sys.executable,
                str(script_path),
                "--live-results", results_path,
                "--thresholds", thresholds_path
            ],
            capture_output=True,
            text=True,
            timeout=60
        )
        return result.stdout
    except subprocess.TimeoutExpired:
        print("[ERROR] monitor_live_swings.py timed out")
        return None
    except Exception as e:
        print(f"[ERROR] Failed to run monitor_live_swings.py: {e}")
        return None


def process_new_data(
    results_path: str,
    thresholds_path: str,
    timestamp: datetime,
    source_used: str,
    previous_analyses: Optional[Dict],
    enable_alerts: bool
) -> Dict:
    """
    Process newly scraped data.

    1. Generate swing charts
    2. Run monitor script
    3. Check for alerts

    Returns the new constituency analyses.
    """
    # Create timestamped output directory
    output_dir = Path(ANALYSIS_DIR) / timestamp.strftime('%Y-%m-%d_%H-%M-%S')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save source info
    source_file = output_dir / 'source_info.txt'
    with open(source_file, 'w') as f:
        f.write(f"Source: {source_used}\n")
        f.write(f"Timestamp: {timestamp.isoformat()}\n")
        f.write(f"Results: {results_path}\n")

    # Generate charts
    print(f"[{timestamp.strftime('%H:%M:%S')}] Generating swing charts...")
    analyses = generate_all_charts(results_path, thresholds_path, str(output_dir))
    print(f"  Generated {len(analyses)} constituency charts + dashboard")

    # Run monitor script
    print(f"[{timestamp.strftime('%H:%M:%S')}] Running swing analysis...")
    monitor_output = run_monitor_script(results_path, thresholds_path)

    # Save monitor output
    if monitor_output:
        output_file = output_dir / 'monitor_output.txt'
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(monitor_output)

        # Also print key sections
        print()
        for line in monitor_output.split('\n'):
            if 'PROJECTED SEATS' in line or 'STATUS SUMMARY' in line:
                print(line)
            elif line.startswith('  Holding:') or line.startswith('  At Risk:') or line.startswith('  Flip Projected:'):
                print(line)

    # Check for alerts
    if enable_alerts and previous_analyses is not None:
        alerts = check_and_alert(
            analyses,
            previous_analyses,
            enable_sound=True,
            enable_notification=True
        )
        if alerts:
            alert_summary = format_alert_summary(alerts)
            print(alert_summary)

            # Save alerts to file
            alerts_file = output_dir / 'alerts.json'
            with open(alerts_file, 'w', encoding='utf-8') as f:
                json.dump([
                    {'constituency': a[0], 'type': a[1], 'message': a[2]}
                    for a in alerts
                ], f, indent=2)

    print(f"\n[{timestamp.strftime('%H:%M:%S')}] Analysis saved to {output_dir}")

    return analyses


def main():
    parser = argparse.ArgumentParser(
        description="Election night monitoring orchestrator with multi-source redundancy"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=DEFAULT_INTERVAL,
        help=f"Seconds between scrapes (default: {DEFAULT_INTERVAL})"
    )
    parser.add_argument(
        "--thresholds",
        default=DEFAULT_THRESHOLDS,
        help=f"Path to thresholds JSON (default: {DEFAULT_THRESHOLDS})"
    )
    parser.add_argument(
        "--no-alerts",
        action="store_true",
        help="Disable desktop and sound alerts"
    )
    parser.add_argument(
        "--no-granular",
        action="store_true",
        help="Disable granular snapshot saving"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run once with test data and exit"
    )

    args = parser.parse_args()

    # Verify thresholds file exists
    thresholds_path = Path(args.thresholds)
    if not thresholds_path.exists():
        print(f"ERROR: Thresholds file not found: {thresholds_path}")
        print("Run generate_swing_thresholds.py first to create baseline thresholds.")
        sys.exit(1)

    # Create output directories
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    Path(GRANULAR_DIR).mkdir(parents=True, exist_ok=True)
    Path(RECONCILIATION_DIR).mkdir(parents=True, exist_ok=True)
    Path(ANALYSIS_DIR).mkdir(parents=True, exist_ok=True)

    # Send startup notification
    if not args.no_alerts:
        send_notification(
            "Election Monitor Started",
            f"Multi-source monitoring\nInterval: {args.interval // 60} minutes",
            sound=False
        )

    start_time = datetime.now()
    last_update = None
    last_hash = None
    last_source = "N/A"
    previous_analyses = None
    scrape_count = 0
    update_count = 0

    print(f"Starting election night monitor (multi-source)...")
    print(f"  PRIMARY: {PRIMARY_URL}")
    print(f"  SECONDARY: {SECONDARY_URL}")
    print(f"  GRANULAR: results.sluelectoral.com/district.php?id=1-17")
    print(f"  Interval: {args.interval}s ({args.interval // 60} minutes)")
    print(f"  Thresholds: {args.thresholds}")
    print(f"  Alerts: {'Disabled' if args.no_alerts else 'Enabled'}")
    print(f"  Granular snapshots: {'Disabled' if args.no_granular else 'Enabled'}")
    print("-" * 60)

    try:
        while True:
            scrape_count += 1
            timestamp = datetime.now()
            next_check = timestamp + timedelta(seconds=args.interval)

            print_header(start_time, last_update, next_check, last_source)
            print(f"[{timestamp.strftime('%H:%M:%S')}] Scraping election results (attempt #{scrape_count})...")

            # Scrape with fallback
            data, source, results_path, metadata = scrape_with_fallback(
                timestamp,
                save_granular=not args.no_granular
            )

            if data is None:
                print(f"  [ERROR] All sources failed")
                print(f"  Will retry in {args.interval}s")
            else:
                current_hash = compute_content_hash(data)

                if current_hash == last_hash:
                    print(f"  [INFO] No changes detected ({len(data)} records from {source})")
                else:
                    update_count += 1
                    print(f"  [NEW] Got {len(data)} records from {source}")

                    # Process the new data
                    previous_analyses = process_new_data(
                        results_path,
                        args.thresholds,
                        timestamp,
                        source,
                        previous_analyses,
                        enable_alerts=not args.no_alerts
                    )

                    last_update = timestamp
                    last_hash = current_hash
                    last_source = source

            # Test mode - exit after first scrape
            if args.test:
                print("\n[TEST MODE] Exiting after single run")
                break

            # Wait for next interval
            print(f"\n[WAIT] Next check at {next_check.strftime('%H:%M:%S')} (Press Ctrl+C to stop)")
            time.sleep(args.interval)

    except KeyboardInterrupt:
        print("\n\n" + "=" * 60)
        print("MONITORING STOPPED")
        print("=" * 60)
        print(f"  Total runtime: {datetime.now() - start_time}")
        print(f"  Total scrapes: {scrape_count}")
        print(f"  Data updates: {update_count}")
        print(f"  Last source: {last_source}")
        print()

        # Send shutdown notification
        if not args.no_alerts:
            send_notification(
                "Election Monitor Stopped",
                f"Total updates: {update_count}\nLast source: {last_source}",
                sound=False
            )


if __name__ == "__main__":
    main()
