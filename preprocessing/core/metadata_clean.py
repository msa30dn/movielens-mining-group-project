"""Phase 2: Metadata Generation for Clean Data

MovieLens Data Mining
Nguyen Sy Hung
2026

This module builds metadata_clean.json containing:
- Cleaning statistics (rows processed, transformations applied)
- Validation results (schema, domain, join coverage)
- Audit information for provenance tracking

The metadata_clean.json complements metadata_raw.json to provide
full lineage from raw → clean.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import pandas as pd


def build_clean_metadata(
    run_id: str,
    raw_counts: Dict[str, int],
    clean_counts: Dict[str, int],
    validation_results: Dict[str, Any],
) -> Dict[str, Any]:
    """Build metadata dictionary for clean phase.
    
    Args:
        run_id: Run identifier
        raw_counts: Row counts from raw data
        clean_counts: Row counts from clean data
        validation_results: Validation stats from validate module
        
    Returns:
        Metadata dictionary ready for JSON serialization
    """
    metadata = {
        'run_id': run_id,
        'phase': 'clean',
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        
        'raw_counts': raw_counts,
        'clean_counts': clean_counts,
        
        'validation': validation_results,
    }
    
    return metadata


def write_metadata_clean(metadata: Dict[str, Any], path: str | Path) -> None:
    """Write metadata_clean.json to disk.
    
    Args:
        metadata: Metadata dictionary
        path: Output file path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def compute_cleaning_stats(
    raw_counts: Dict[str, int],
    clean_counts: Dict[str, int]
) -> Dict[str, Any]:
    """Compute statistics about the cleaning process.
    
    Args:
        raw_counts: Row counts before cleaning
        clean_counts: Row counts after cleaning
        
    Returns:
        Dict with cleaning statistics
    """
    stats = {}
    
    for table in clean_counts.keys():
        raw_n = raw_counts.get(table, 0)
        clean_n = clean_counts.get(table, 0)
        
        stats[table] = {
            'raw_rows': raw_n,
            'clean_rows': clean_n,
            'rows_dropped': raw_n - clean_n,
            'retention_rate': round(clean_n / raw_n, 6) if raw_n > 0 else 0.0,
        }
    
    return stats


def summarize_validation_results(validation_results: Dict[str, Any]) -> Dict[str, Any]:
    """Create a summary of validation results for quick inspection.
    
    Args:
        validation_results: Full validation results
        
    Returns:
        Summary dict with key metrics
    """
    summary = {
        'all_validations_passed': True,
        'warnings': [],
        'errors': [],
    }
    
    # Check each table's validation
    for table, results in validation_results.items():
        if 'error' in results:
            summary['all_validations_passed'] = False
            summary['errors'].append({
                'table': table,
                'error': results['error']
            })
        
        if 'warning' in results:
            summary['warnings'].append({
                'table': table,
                'warning': results['warning']
            })
    
    return summary
