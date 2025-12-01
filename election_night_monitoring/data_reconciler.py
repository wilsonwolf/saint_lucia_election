#!/usr/bin/env python3
"""
Data Reconciler - Multi-source validation and reconciliation

Provides functions to validate and reconcile election data from multiple sources:
- PRIMARY: results.sluelectoral.com/summary.php
- SECONDARY: sluelectoral.com/election-night-results-2026
- GRANULAR: results.sluelectoral.com/district.php (aggregated)

Features:
- Data validation (structure, completeness)
- Cross-source comparison
- Discrepancy detection and logging
- Best-source selection with fallback
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from swing_utils import (
    parse_vote_value,
    normalize_constituency_name,
    get_all_constituencies,
    extract_constituency_data,
)


def validate_data(data: List[Dict]) -> Tuple[bool, List[str]]:
    """
    Validate election data has expected structure and reasonable values.

    Checks:
    - Has records
    - Has candidate records with expected fields
    - Has multiple constituencies (expect 17)
    - Has summary records
    - Vote values are reasonable

    Args:
        data: List of election records

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

    # Check candidate records have required fields
    for i, record in enumerate(candidate_records[:5]):  # Check first 5
        if not record.get("Party"):
            issues.append(f"Record {i} missing Party field")
        if not record.get("district"):
            issues.append(f"Record {i} missing district field")

    # Check for multiple constituencies
    districts = set(r.get("district") for r in data if r.get("district"))
    summary_labels = {
        'No. of Electors', 'No. Of Electors', 'Votes Cast', 'Turnout %',
        'Rejected', '% Turnout', 'Turnout%'
    }
    valid_districts = {d for d in districts if d and d not in summary_labels}

    if len(valid_districts) < 10:
        issues.append(f"Only {len(valid_districts)} constituencies found (expected ~17)")
    elif len(valid_districts) < 17:
        issues.append(f"WARNING: Only {len(valid_districts)} constituencies (expected 17)")

    # Check for summary records
    summary_records = [r for r in data if "summary_label" in r]
    if len(summary_records) < 17:
        issues.append(f"Only {len(summary_records)} summary records (expected ~68)")

    # Check vote values are non-negative
    for record in candidate_records:
        total = parse_vote_value(record.get("Total", 0))
        if total < 0:
            issues.append(f"Negative vote total in {record.get('district')}")
            break

    # Determine if critical issues exist
    critical_issues = [i for i in issues if not i.startswith("WARNING")]
    is_valid = len(critical_issues) == 0

    return is_valid, issues


def has_actual_votes(data: List[Dict]) -> bool:
    """
    Check if data contains actual vote counts (not all zeros).

    Args:
        data: List of election records

    Returns:
        True if at least some votes are recorded
    """
    candidate_records = [r for r in data if "Candidate" in r]
    for record in candidate_records:
        total = parse_vote_value(record.get("Total", 0))
        if total > 0:
            return True
    return False


def count_votes_by_constituency(data: List[Dict]) -> Dict[str, Dict]:
    """
    Aggregate vote totals by constituency for comparison.

    Args:
        data: List of election records

    Returns:
        {constituency: {party: votes, 'total': total_votes}}
    """
    results = {}

    for record in data:
        if "Candidate" not in record:
            continue

        district = record.get("district", "")
        if not district:
            continue

        normalized = normalize_constituency_name(district)
        if normalized not in results:
            results[normalized] = {'total': 0}

        party = record.get("Party", "").strip()
        votes = parse_vote_value(record.get("Total", 0))

        if party:
            results[normalized][party] = results[normalized].get(party, 0) + votes
        results[normalized]['total'] += votes

    return results


def compare_sources(
    source1_data: List[Dict],
    source2_data: List[Dict],
    source1_name: str = "Source1",
    source2_name: str = "Source2",
    threshold_pct: float = 1.0
) -> Dict:
    """
    Compare vote totals between two data sources.

    Args:
        source1_data: First source data
        source2_data: Second source data
        source1_name: Name for first source
        source2_name: Name for second source
        threshold_pct: Percentage threshold for flagging discrepancies

    Returns:
        {
            'match': bool,
            'discrepancies': [{constituency, source1_votes, source2_votes, diff_pct}],
            'source1_only': [constituencies],
            'source2_only': [constituencies]
        }
    """
    source1_votes = count_votes_by_constituency(source1_data)
    source2_votes = count_votes_by_constituency(source2_data)

    all_constituencies = set(source1_votes.keys()) | set(source2_votes.keys())
    discrepancies = []
    source1_only = []
    source2_only = []

    for const in all_constituencies:
        s1 = source1_votes.get(const, {})
        s2 = source2_votes.get(const, {})

        s1_total = s1.get('total', 0)
        s2_total = s2.get('total', 0)

        if const not in source1_votes:
            source1_only.append(const)
            continue
        if const not in source2_votes:
            source2_only.append(const)
            continue

        # Compare totals
        if s1_total > 0 or s2_total > 0:
            max_total = max(s1_total, s2_total)
            if max_total > 0:
                diff_pct = abs(s1_total - s2_total) / max_total * 100
                if diff_pct > threshold_pct:
                    discrepancies.append({
                        'constituency': const,
                        f'{source1_name}_votes': s1_total,
                        f'{source2_name}_votes': s2_total,
                        'diff_pct': round(diff_pct, 2)
                    })

    return {
        'match': len(discrepancies) == 0 and len(source1_only) == 0 and len(source2_only) == 0,
        'discrepancies': discrepancies,
        f'{source1_name}_only': source1_only,
        f'{source2_name}_only': source2_only
    }


def reconcile_sources(
    primary_data: Optional[List[Dict]],
    secondary_data: Optional[List[Dict]],
    granular_data: Optional[List[Dict]]
) -> Tuple[Optional[List[Dict]], str, Dict]:
    """
    Reconcile data from multiple sources and select the best one.

    Priority:
    1. PRIMARY (if valid and has data)
    2. SECONDARY (if primary fails)
    3. GRANULAR (if both fail)

    Also performs cross-source validation when multiple sources succeed.

    Args:
        primary_data: Data from primary source (summary.php)
        secondary_data: Data from secondary source (sluelectoral.com)
        granular_data: Data from granular source (district.php aggregated)

    Returns:
        (best_data, source_used, reconciliation_report)
    """
    report = {
        'timestamp': datetime.now().isoformat(),
        'sources_available': [],
        'sources_valid': [],
        'source_used': None,
        'validation_results': {},
        'comparisons': {},
        'issues': []
    }

    # Validate each source
    sources = [
        ('PRIMARY', primary_data),
        ('SECONDARY', secondary_data),
        ('GRANULAR', granular_data)
    ]

    valid_sources = {}

    for name, data in sources:
        if data:
            report['sources_available'].append(name)
            is_valid, issues = validate_data(data)
            has_votes = has_actual_votes(data)
            report['validation_results'][name] = {
                'is_valid': is_valid,
                'has_votes': has_votes,
                'issues': issues,
                'record_count': len(data)
            }
            if is_valid:
                report['sources_valid'].append(name)
                valid_sources[name] = data

    # No valid sources
    if not valid_sources:
        report['source_used'] = 'NONE'
        report['issues'].append("No valid data sources available")
        return None, 'NONE', report

    # Cross-source comparison (if multiple valid sources)
    if len(valid_sources) >= 2:
        source_names = list(valid_sources.keys())
        for i in range(len(source_names)):
            for j in range(i + 1, len(source_names)):
                name1, name2 = source_names[i], source_names[j]
                comparison = compare_sources(
                    valid_sources[name1],
                    valid_sources[name2],
                    name1, name2
                )
                comparison_key = f"{name1}_vs_{name2}"
                report['comparisons'][comparison_key] = comparison

                if comparison['discrepancies']:
                    report['issues'].append(
                        f"Discrepancies between {name1} and {name2}: "
                        f"{len(comparison['discrepancies'])} constituencies differ"
                    )

    # Select best source (priority order)
    if 'PRIMARY' in valid_sources:
        report['source_used'] = 'PRIMARY'
        return valid_sources['PRIMARY'], 'PRIMARY', report
    elif 'SECONDARY' in valid_sources:
        report['source_used'] = 'SECONDARY'
        return valid_sources['SECONDARY'], 'SECONDARY', report
    elif 'GRANULAR' in valid_sources:
        report['source_used'] = 'GRANULAR'
        return valid_sources['GRANULAR'], 'GRANULAR', report

    # Shouldn't reach here, but fallback
    report['source_used'] = 'NONE'
    return None, 'NONE', report


def save_reconciliation_report(
    report: Dict,
    output_dir: Path,
    timestamp: datetime
) -> str:
    """Save reconciliation report to JSON file."""
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = f"reconciliation_{timestamp.strftime('%Y-%m-%d_%H-%M-%S')}.json"
    filepath = output_dir / filename

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    return str(filepath)


def format_reconciliation_summary(report: Dict) -> str:
    """Format a human-readable summary of reconciliation report."""
    lines = []
    lines.append("=" * 60)
    lines.append("DATA RECONCILIATION SUMMARY")
    lines.append("=" * 60)

    lines.append(f"\nSources Available: {', '.join(report['sources_available']) or 'None'}")
    lines.append(f"Sources Valid: {', '.join(report['sources_valid']) or 'None'}")
    lines.append(f"Source Used: {report['source_used']}")

    if report['validation_results']:
        lines.append("\nValidation Results:")
        for source, result in report['validation_results'].items():
            status = "VALID" if result['is_valid'] else "INVALID"
            votes = "has votes" if result['has_votes'] else "no votes"
            lines.append(f"  {source}: {status} ({votes}, {result['record_count']} records)")
            if result['issues']:
                for issue in result['issues'][:3]:  # Show first 3 issues
                    lines.append(f"    - {issue}")

    if report['comparisons']:
        lines.append("\nCross-Source Comparisons:")
        for comp_name, comp in report['comparisons'].items():
            match_status = "MATCH" if comp['match'] else "MISMATCH"
            lines.append(f"  {comp_name}: {match_status}")
            if comp['discrepancies']:
                lines.append(f"    Discrepancies: {len(comp['discrepancies'])}")

    if report['issues']:
        lines.append("\nIssues:")
        for issue in report['issues']:
            lines.append(f"  - {issue}")

    return "\n".join(lines)


if __name__ == "__main__":
    # Test with sample data
    print("Data Reconciler - Test Mode")
    print("Use this module as an import for actual reconciliation")
