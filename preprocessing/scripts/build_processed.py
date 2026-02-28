#!/usr/bin/env python3
# MovieLens Data Mining
# 
# Nguyen Sy Hung
# 2026
"""Build processing pipeline (Phase 2+): staging → clean → split → features.

This script assumes raw staging Parquet files already exist.
Run `build_raw_staging.py` first if they don't.

Usage:
    python scripts/build_processed.py [--staging data/raw_staging] [--out data/processed]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.pipeline import run_pipeline


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Build processing pipeline: staging → clean → split → features",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--staging",
        default="data/raw_staging", 
        help="Path to raw staging directory (Parquet files)"
    )
    parser.add_argument(
        "--out", 
        default="data/processed", 
        help="Output root for processed runs"
    )
    parser.add_argument(
        "--run-tag", 
        default=None, 
        help="Optional run_id override (default: auto-timestamp)"
    )
    parser.add_argument(
        "--schema-version", 
        default="0.1.0", 
        help="Schema version string"
    )
    parser.add_argument(
        "--split-policy",
        default="global",
        choices=["global", "per_user"],
        help="Temporal split policy (global cutoff vs per-user cutoff)"
    )
    parser.add_argument(
        "--test-frac",
        type=float,
        default=0.1,
        help="Test fraction for temporal split"
    )
    parser.add_argument(
        "--cutoff-date",
        default=None,
        help="Explicit global cutoff datetime (only used when --split-policy=global)"
    )
    return parser.parse_args()


def main() -> None:
    """Entrypoint for processing pipeline builder."""
    args = parse_args()
    
    try:
        print("Starting processing pipeline...")
        print(f"Raw staging directory: {args.staging}")
        print(f"Output directory: {args.out}")
        print()
        
        run_dir = run_pipeline(
            raw_staging_dir=args.staging,
            out_root=args.out,
            run_tag=args.run_tag,
            schema_version=args.schema_version,
            split_policy=args.split_policy,
            test_frac=args.test_frac,
            cutoff_date=args.cutoff_date,
        )
        
        print()
        print("="*60)
        print(f"✓ Pipeline completed successfully")
        print(f"✓ Output: {run_dir}")
        print("="*60)
        
    except KeyboardInterrupt:
        print()
        print("="*60)
        print("✗ Pipeline interrupted by user")
        print("="*60)
        sys.exit(130)
    except Exception as e:
        print()
        print("="*60)
        print(f"✗ Pipeline failed: {e}")
        print("="*60)
        sys.exit(1)


if __name__ == "__main__":
    main()
