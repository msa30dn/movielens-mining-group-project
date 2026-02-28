#!/usr/bin/env python3
# MovieLens Data Mining
# Nguyen Sy Hung
# 2026
"""PHASE 0-1: INGEST - Build raw staging layer (CSV → Parquet).

This script should be run ONCE to create the shared raw staging Parquet files.
These files are then reused by all subsequent runs of build_processed.py.

Usage:
    python scripts/build_raw_staging.py [--raw data/raw/ml-latest] [--staging data/raw_staging]
"""
from __future__ import annotations

import argparse
import sys
import time
import logging
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core import ingest
from core.metadata_raw import build_raw_metadata, get_git_commit_hash, write_metadata_raw


def _setup_logger() -> logging.Logger:
    """Configure console logger."""
    logger = logging.getLogger("staging")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    
    return logger


def build_raw_staging(
    raw_dir: str | Path,
    staging_dir: str | Path = "data/raw_staging",
    force: bool = False
) -> Path:
    """Build raw staging Parquet files from CSV sources.
    
    Args:
        raw_dir: Path to raw ml-latest directory containing CSVs
        staging_dir: Path to output staging directory (default: data/raw_staging)
        force: If True, rebuild even if staging files already exist
        
    Returns:
        Path to the staging directory
        
    Raises:
        FileNotFoundError: If raw directory or required CSVs don't exist
    """
    start_time = time.time()
    logger = _setup_logger()
    
    raw_path = Path(raw_dir)
    staging_path = Path(staging_dir)
    
    # Validate raw directory exists
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw directory not found: {raw_path}")
    
    # Create staging directory
    staging_path.mkdir(parents=True, exist_ok=True)
    
    # Check if staging already exists
    required_files = [
        "movies_raw.parquet",
        "ratings_raw.parquet",
        "tags_raw.parquet",
        "links_raw.parquet",
    ]
    staging_exists = all((staging_path / f).exists() for f in required_files)
    
    if staging_exists and not force:
        logger.info("="*60)
        logger.info("PHASE 0-1: Raw staging already exists at: %s", staging_path)
        logger.info("Use --force to rebuild")
        logger.info("="*60)
        return staging_path
    
    logger.info("="*60)
    logger.info("PHASE 0-1: INGEST (CSV → Parquet Staging)")
    logger.info("="*60)
    logger.info("Raw directory: %s", raw_path)
    logger.info("Staging directory: %s", staging_path)
    logger.info("")
    
    # Read and write each CSV → Parquet
    logger.info("Reading movies.csv...")
    movies = ingest.read_movies(raw_path)
    movies_count = len(movies)
    movies.to_parquet(staging_path / "movies_raw.parquet", index=False)
    logger.info("  ✓ Written: movies_raw.parquet (%s rows)", f"{movies_count:,}")
    del movies
    
    logger.info("Reading ratings.csv...")
    ratings = ingest.read_ratings(raw_path)
    ratings_count = len(ratings)
    ratings.to_parquet(staging_path / "ratings_raw.parquet", index=False)
    logger.info("  ✓ Written: ratings_raw.parquet (%s rows)", f"{ratings_count:,}")
    del ratings
    
    logger.info("Reading tags.csv...")
    tags = ingest.read_tags(raw_path)
    tags_count = len(tags)
    tags.to_parquet(staging_path / "tags_raw.parquet", index=False)
    logger.info("  ✓ Written: tags_raw.parquet (%s rows)", f"{tags_count:,}")
    del tags
    
    logger.info("Reading links.csv...")
    links = ingest.read_links(raw_path)
    links_count = len(links)
    links.to_parquet(staging_path / "links_raw.parquet", index=False)
    logger.info("  ✓ Written: links_raw.parquet (%s rows)", f"{links_count:,}")
    del links
    
    # Optional genome files (check if they exist)
    genome_tags_csv = raw_path / "genome-tags.csv"
    genome_scores_csv = raw_path / "genome-scores.csv"
    
    genome_tags_count = 0
    genome_scores_count = 0
    
    if genome_tags_csv.exists():
        logger.info("Reading genome-tags.csv...")
        genome_tags = ingest.read_genome_tags(raw_path)
        genome_tags_count = len(genome_tags)
        genome_tags.to_parquet(staging_path / "genome_tags_raw.parquet", index=False)
        logger.info("  ✓ Written: genome_tags_raw.parquet (%s rows)", f"{genome_tags_count:,}")
        del genome_tags
    
    if genome_scores_csv.exists():
        logger.info("Reading genome-scores.csv...")
        genome_scores = ingest.read_genome_scores(raw_path)
        genome_scores_count = len(genome_scores)
        genome_scores.to_parquet(staging_path / "genome_scores_raw.parquet", index=False)
        logger.info("  ✓ Written: genome_scores_raw.parquet (%s rows)", f"{genome_scores_count:,}")
        del genome_scores
    
    logger.info("")
    
    # Generate provenance metadata
    logger.info("Generating provenance metadata (hashes + row counts)...")
    raw_metadata = build_raw_metadata(
        raw_dir=raw_path,
        run_id="staging",  # Special identifier for shared staging
        schema_version="0.1.0",
        pipeline_version=get_git_commit_hash(Path(__file__).parent.parent),
    )
    
    metadata_path = staging_path / "metadata_raw.json"
    write_metadata_raw(raw_metadata, metadata_path)
    logger.info("  ✓ Written: metadata_raw.json")
    
    elapsed = time.time() - start_time
    logger.info("")
    logger.info("="*60)
    logger.info("✓ PHASE 0-1 (INGEST) COMPLETED in %.2f seconds", elapsed)
    logger.info("✓ Output: %s", staging_path)
    logger.info("="*60)
    
    return staging_path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="PHASE 0-1: INGEST raw staging layer (CSV → Parquet) - run this ONCE",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--raw",
        default="data/raw/ml-latest",
        help="Path to raw ml-latest folder containing CSVs"
    )
    parser.add_argument(
        "--staging",
        default="data/raw_staging",
        help="Output directory for staging Parquet files"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force rebuild even if staging files exist"
    )
    return parser.parse_args()


def main() -> None:
    """Entrypoint for raw staging builder."""
    args = parse_args()
    
    try:
        staging_path = build_raw_staging(
            raw_dir=args.raw,
            staging_dir=args.staging,
            force=args.force,
        )
        sys.exit(0)
        
    except KeyboardInterrupt:
        print()
        print("="*60)
        print("✗ Staging build interrupted by user")
        print("="*60)
        sys.exit(130)
    except Exception as e:
        print()
        print("="*60)
        print(f"✗ Staging build failed: {e}")
        print("="*60)
        sys.exit(1)


if __name__ == "__main__":
    main()
