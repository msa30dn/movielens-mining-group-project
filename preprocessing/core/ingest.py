"""Ingest helpers for reading raw MovieLens CSV files.

MovieLens Data Mining
Nguyen Sy Hung
2026
"""

from pathlib import Path
import logging

import pandas as pd


logger = logging.getLogger(__name__)


REQUIRED_COLUMNS = {
    "movies.csv": {"movieId", "title", "genres"},
    "ratings.csv": {"userId", "movieId", "rating", "timestamp"},
    "tags.csv": {"userId", "movieId", "tag", "timestamp"},
    "links.csv": {"movieId", "imdbId", "tmdbId"},
}

# Add genome validation
GENOME_COLUMNS = {
    "genome-tags.csv": {"tagId", "tag"},
    "genome-scores.csv": {"movieId", "tagId", "relevance"},
}


def _read_csv_strict(
    path: Path, 
    dtype: dict[str, str], 
    required_columns: set[str],
    log_progress: bool = True
) -> pd.DataFrame:
    """Read CSV with strict schema validation and optional logging.
    
    Args:
        path: Path to CSV file
        dtype: Column dtype specifications
        required_columns: Set of required column names
        log_progress: Whether to log read progress
        
    Returns:
        DataFrame with validated schema
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If required columns are missing
    """
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    
    if log_progress:
        logger.info(f"Reading {path.name}...")
    
    frame = pd.read_csv(path, dtype=dtype)
    
    if log_progress:
        logger.info(f"  Loaded {len(frame):,} rows × {len(frame.columns)} columns")
    
    missing = sorted(required_columns - set(frame.columns))
    if missing:
        raise ValueError(f"{path.name}: missing required columns: {missing}")
    
    return frame


def read_movies(raw_dir: Path) -> pd.DataFrame:
    return _read_csv_strict(
        raw_dir / "movies.csv",
        dtype={"movieId": "int32", "title": "string", "genres": "string"},
        required_columns=REQUIRED_COLUMNS["movies.csv"],
    )


def read_ratings(raw_dir: Path) -> pd.DataFrame:
    return _read_csv_strict(
        raw_dir / "ratings.csv",
        dtype={"userId": "int32", "movieId": "int32", "rating": "float32", "timestamp": "int64"},
        required_columns=REQUIRED_COLUMNS["ratings.csv"],
    )


def read_tags(raw_dir: Path) -> pd.DataFrame:
    return _read_csv_strict(
        raw_dir / "tags.csv",
        dtype={"userId": "int32", "movieId": "int32", "tag": "string", "timestamp": "int64"},
        required_columns=REQUIRED_COLUMNS["tags.csv"],
    )


def read_links(raw_dir: Path) -> pd.DataFrame:
    return _read_csv_strict(
        raw_dir / "links.csv",
        dtype={"movieId": "int32", "imdbId": "Int64", "tmdbId": "Int64"},
        required_columns=REQUIRED_COLUMNS["links.csv"],
    )


def read_genome_tags(raw_dir: Path) -> pd.DataFrame | None:
    """Read genome tags file with validation (optional file).
    
    Args:
        raw_dir: Path to raw data directory
        
    Returns:
        DataFrame or None if file doesn't exist
        
    Raises:
        ValueError: If required columns are missing
    """
    path = raw_dir / "genome-tags.csv"
    if not path.exists():
        logger.info("genome-tags.csv not found (optional)")
        return None
    
    logger.info(f"Reading {path.name}...")
    df = pd.read_csv(path, dtype={"tagId": "int32", "tag": "string"})
    
    # Validate columns
    missing = GENOME_COLUMNS["genome-tags.csv"] - set(df.columns)
    if missing:
        raise ValueError(f"genome-tags.csv: missing required columns: {sorted(missing)}")
    
    logger.info(f"  Loaded {len(df):,} genome tags")
    return df


def read_genome_scores(raw_dir: Path) -> pd.DataFrame | None:
    """Read genome scores file with validation (optional file).
    
    Args:
        raw_dir: Path to raw data directory
        
    Returns:
        DataFrame or None if file doesn't exist
        
    Raises:
        ValueError: If required columns are missing
    """
    path = raw_dir / "genome-scores.csv"
    if not path.exists():
        logger.info("genome-scores.csv not found (optional)")
        return None
    
    logger.info(f"Reading {path.name}...")
    df = pd.read_csv(path, dtype={"movieId": "int32", "tagId": "int32", "relevance": "float32"})
    
    # Validate columns
    missing = GENOME_COLUMNS["genome-scores.csv"] - set(df.columns)
    if missing:
        raise ValueError(f"genome-scores.csv: missing required columns: {sorted(missing)}")
    
    logger.info(f"  Loaded {len(df):,} genome scores")
    return df
