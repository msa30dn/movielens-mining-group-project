"""
Phase 2: Data Validation & Quality Checks

This module validates cleaned DataFrames against schema and domain constraints.

Validation categories:
- Schema: Required columns, data types
- Domain: Value ranges, non-null constraints
- Referential: Join coverage (foreign keys)
- Duplicates: Duplicate detection and reporting

Validation philosophy:
- Hard gates: Violations that fail the pipeline (e.g., missing required columns)
- Soft gates: Violations that are reported but don't fail (e.g., low join coverage)
"""

# MovieLens Data Mining
#
# Nguyen Sy Hung
# 2026


from __future__ import annotations

from typing import Any, Dict

import pandas as pd


class ValidationError(Exception):
    """Raised when a hard validation gate fails."""
    pass


def validate_schema(
    df: pd.DataFrame,
    required_cols: list[str],
    name: str
) -> None:
    """Validate that DataFrame has required columns.
    
    Args:
        df: DataFrame to validate
        required_cols: List of required column names
        name: Table name for error messages
        
    Raises:
        ValidationError: If required columns are missing
    """
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValidationError(
            f"{name}: Missing required columns: {sorted(missing)}"
        )


def validate_rating_domain(df: pd.DataFrame) -> Dict[str, Any]:
    """Validate rating domain constraints.
    
    Checks:
    - rating in [0.5, 5.0]
    - userId, movieId are non-null
    
    Args:
        df: Ratings DataFrame
        
    Returns:
        Dict with validation stats
        
    Raises:
        ValidationError: If hard constraints violated
    """
    stats = {}
    
    # Check for nulls in key columns
    null_users = df['userId'].isna().sum()
    null_movies = df['movieId'].isna().sum()
    null_ratings = df['rating'].isna().sum()
    
    stats['null_users'] = int(null_users)
    stats['null_movies'] = int(null_movies)
    stats['null_ratings'] = int(null_ratings)
    
    if null_users > 0:
        raise ValidationError(f"Ratings: {null_users} null userIds found (not allowed)")
    if null_movies > 0:
        raise ValidationError(f"Ratings: {null_movies} null movieIds found (not allowed)")
    
    # Check rating range
    min_rating = df['rating'].min()
    max_rating = df['rating'].max()
    
    stats['min_rating'] = float(min_rating)
    stats['max_rating'] = float(max_rating)
    
    if min_rating < 0.5 or max_rating > 5.0:
        raise ValidationError(
            f"Ratings: rating out of range [0.5, 5.0]: min={min_rating}, max={max_rating}"
        )
    
    # Check for invalid ratings (not in valid MovieLens scale)
    valid_ratings = {0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0}
    unique_ratings = set(df['rating'].unique())
    invalid_ratings = unique_ratings - valid_ratings
    
    if invalid_ratings:
        stats['invalid_ratings'] = sorted(list(invalid_ratings))
        # This is a soft gate - report but don't fail
        stats['warning'] = f"Found ratings not on standard scale: {sorted(invalid_ratings)}"
    
    return stats


def validate_join_coverage(
    fact_df: pd.DataFrame,
    dim_df: pd.DataFrame,
    fact_name: str,
    fact_key: str,
    dim_key: str,
    threshold: float = 0.95
) -> Dict[str, Any]:
    """Validate join coverage between fact and dimension tables.
    
    Args:
        fact_df: Fact table (e.g., ratings)
        dim_df: Dimension table (e.g., movies)
        fact_name: Name of fact table for reporting
        fact_key: Foreign key column in fact table
        dim_key: Primary key column in dimension table
        threshold: Minimum acceptable coverage (default: 95%)
        
    Returns:
        Dict with coverage stats
        
    Raises:
        ValidationError: If coverage below threshold
    """
    fact_keys = set(fact_df[fact_key].dropna().unique())
    dim_keys = set(dim_df[dim_key].dropna().unique())
    
    matched_keys = fact_keys & dim_keys
    unmatched_keys = fact_keys - dim_keys
    
    n_fact = len(fact_df)
    n_matched = len(fact_df[fact_df[fact_key].isin(matched_keys)])
    n_unmatched = n_fact - n_matched
    
    coverage = n_matched / n_fact if n_fact > 0 else 0.0
    
    stats = {
        'n_fact_rows': int(n_fact),
        'n_matched_rows': int(n_matched),
        'n_unmatched_rows': int(n_unmatched),
        'coverage': round(coverage, 6),
        'n_unique_fact_keys': len(fact_keys),
        'n_unique_dim_keys': len(dim_keys),
        'n_matched_keys': len(matched_keys),
        'n_unmatched_keys': len(unmatched_keys),
    }
    
    if coverage < threshold:
        raise ValidationError(
            f"{fact_name} → dimension: Coverage {coverage:.2%} below threshold {threshold:.2%} "
            f"({n_unmatched:,} rows unmatched)"
        )
    
    return stats


def check_duplicates(
    df: pd.DataFrame,
    subset: list[str] | None = None,
    name: str = "table"
) -> Dict[str, Any]:
    """Check for duplicate rows.
    
    Args:
        df: DataFrame to check
        subset: Columns to check for duplicates (None = all columns)
        name: Table name for reporting
        
    Returns:
        Dict with duplicate stats
    """
    if subset:
        # Only check duplicates on specified columns
        subset_cols = [c for c in subset if c in df.columns]
        n_dupes = int(df.duplicated(subset=subset_cols, keep='first').sum())
        n_unique = int(df[subset_cols].drop_duplicates().shape[0])
    else:
        n_dupes = int(df.duplicated(keep='first').sum())
        n_unique = len(df) - n_dupes
    
    stats = {
        'name': name,
        'n_total': len(df),
        'n_duplicates': n_dupes,
        'n_unique': n_unique,
        'duplicate_rate': round(n_dupes / len(df), 6) if len(df) > 0 else 0.0,
    }
    
    return stats


def validate_movies(df: pd.DataFrame) -> Dict[str, Any]:
    """Validate movies table.
    
    Args:
        df: Clean movies DataFrame
        
    Returns:
        Dict with validation stats
    """
    stats = {}
    
    # Schema validation
    validate_schema(df, ['movieId', 'title', 'year', 'genres', 'genres_list'], 'movies')
    
    # Check for null movieIds (primary key)
    null_ids = df['movieId'].isna().sum()
    stats['null_movieIds'] = int(null_ids)
    
    if null_ids > 0:
        raise ValidationError(f"Movies: {null_ids} null movieIds found (not allowed)")
    
    # Check for duplicate movieIds
    dupe_stats = check_duplicates(df, subset=['movieId'], name='movies')
    stats['duplicates'] = dupe_stats
    
    if dupe_stats['n_duplicates'] > 0:
        raise ValidationError(
            f"Movies: {dupe_stats['n_duplicates']} duplicate movieIds found (not allowed)"
        )
    
    # Year stats (soft check)
    year_null = df['year'].isna().sum()
    stats['year_null'] = int(year_null)
    stats['year_null_pct'] = round(year_null / len(df), 6) if len(df) > 0 else 0.0
    
    if not df['year'].isna().all():
        stats['year_min'] = int(df['year'].min())
        stats['year_max'] = int(df['year'].max())
    
    # Genre stats
    stats['no_genres'] = int((df['genres_list'].apply(len) == 0).sum())
    stats['no_genres_pct'] = round(stats['no_genres'] / len(df), 6) if len(df) > 0 else 0.0
    
    return stats


def validate_tags(df: pd.DataFrame) -> Dict[str, Any]:
    """Validate tags table.
    
    Args:
        df: Clean tags DataFrame
        
    Returns:
        Dict with validation stats
    """
    stats = {}
    
    # Schema validation
    validate_schema(df, ['userId', 'movieId', 'tag_raw', 'tag_norm', 'timestamp', 'tag_dt_utc'], 'tags')
    
    # Check for nulls in key columns
    null_users = df['userId'].isna().sum()
    null_movies = df['movieId'].isna().sum()
    
    stats['null_users'] = int(null_users)
    stats['null_movies'] = int(null_movies)
    
    if null_users > 0:
        raise ValidationError(f"Tags: {null_users} null userIds found (not allowed)")
    if null_movies > 0:
        raise ValidationError(f"Tags: {null_movies} null movieIds found (not allowed)")

    # Normalized tag must be present and non-empty (hard gate)
    null_tag_norm = int(df['tag_norm'].isna().sum())
    empty_tag_norm = int((df['tag_norm'].fillna('').astype(str).str.len() == 0).sum())
    stats['null_tag_norm'] = null_tag_norm
    stats['empty_tag_norm'] = empty_tag_norm
    if null_tag_norm > 0 or empty_tag_norm > 0:
        raise ValidationError(
            f"Tags: invalid tag_norm values found (null={null_tag_norm}, empty={empty_tag_norm})"
        )
    
    # Duplicate stats (soft check - tags can have duplicates)
    dupe_stats = check_duplicates(df, subset=['userId', 'movieId', 'tag_norm', 'timestamp'], name='tags')
    stats['duplicates'] = dupe_stats
    
    # Tag normalization impact
    n_unique_raw = df['tag_raw'].nunique()
    n_unique_norm = df['tag_norm'].nunique()
    stats['unique_tags_raw'] = n_unique_raw
    stats['unique_tags_norm'] = n_unique_norm
    stats['normalization_reduction'] = n_unique_raw - n_unique_norm
    
    return stats


def validate_links(df: pd.DataFrame) -> Dict[str, Any]:
    """Validate links table.
    
    Args:
        df: Clean links DataFrame
        
    Returns:
        Dict with validation stats
    """
    stats = {}
    
    # Schema validation
    validate_schema(df, ['movieId', 'imdbId', 'tmdbId'], 'links')
    
    # Check for null movieIds
    null_ids = df['movieId'].isna().sum()
    stats['null_movieIds'] = int(null_ids)
    
    if null_ids > 0:
        raise ValidationError(f"Links: {null_ids} null movieIds found (not allowed)")
    
    # Check duplicates
    dupe_stats = check_duplicates(df, subset=['movieId'], name='links')
    stats['duplicates'] = dupe_stats
    
    if dupe_stats['n_duplicates'] > 0:
        raise ValidationError(
            f"Links: {dupe_stats['n_duplicates']} duplicate movieIds found (not allowed)"
        )
    
    # Coverage stats (soft check)
    stats['imdbId_missing'] = int(df['imdbId'].isna().sum())
    stats['tmdbId_missing'] = int(df['tmdbId'].isna().sum())
    stats['imdbId_coverage'] = round(1 - stats['imdbId_missing'] / len(df), 6) if len(df) > 0 else 0.0
    stats['tmdbId_coverage'] = round(1 - stats['tmdbId_missing'] / len(df), 6) if len(df) > 0 else 0.0
    
    return stats
