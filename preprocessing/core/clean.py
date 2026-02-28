"""Phase 2: Data Cleaning & Normalization

MovieLens Data Mining
Nguyen Sy Hung
2026

This module transforms raw DataFrames into clean, normalized tables ready for analysis.

Transformations:
- Movies: Extract year from title, split genres string into list
- Ratings: Convert Unix timestamp to UTC datetime
- Tags: Normalize tag text, convert timestamp to UTC datetime
- Links: Keep structure, ensure proper types
- Genome: Keep structure, ensure proper types

All transformations are deterministic and documented.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def extract_year_from_title(title: str) -> tuple[str, int | None]:
    """Extract year from movie title if present.
    
    Expected format: "Movie Name (YYYY)" or "Movie Name, The (YYYY)"
    
    Args:
        title: Movie title string
        
    Returns:
        Tuple of (canonical_title, year_or_none), where canonical_title
        preserves the original MovieLens title text (including "(YYYY)").
        
    Examples:
        >>> extract_year_from_title("Toy Story (1995)")
        ('Toy Story (1995)', 1995)
        >>> extract_year_from_title("Movie without year")
        ('Movie without year', None)
    """
    if not isinstance(title, str):
        return str(title), None
    
    title = title.strip()

    # Match (YYYY) at the end of the string
    match = re.search(r'\((\d{4})\)\s*$', title)
    if match:
        year_str = match.group(1)
        year = int(year_str)
        # Keep canonical MovieLens title unchanged; only extract year.
        return title, year
    
    return title, None


def split_genres(genres_str: str) -> list[str]:
    """Split pipe-separated genres into list.
    
    Args:
        genres_str: Genres string like "Action|Adventure|Sci-Fi"
        
    Returns:
        List of genre strings, empty list if no genres
        
    Examples:
        >>> split_genres("Action|Adventure|Sci-Fi")
        ['Action', 'Adventure', 'Sci-Fi']
        >>> split_genres("(no genres listed)")
        []
    """
    if not isinstance(genres_str, str) or genres_str.strip() == "" or genres_str == "(no genres listed)":
        return []
    
    return [g.strip() for g in genres_str.split("|") if g.strip()]


def normalize_tag(tag: str) -> str:
    """Normalize tag text for consistency.
    
    Normalization:
    - Convert to lowercase
    - Collapse multiple spaces to single space
    - Strip leading/trailing whitespace
    
    Args:
        tag: Raw tag string
        
    Returns:
        Normalized tag string
        
    Examples:
        >>> normalize_tag("  Sci-Fi  ")
        'sci-fi'
        >>> normalize_tag("GREAT   Movie")
        'great movie'
    """
    # Treat missing values as empty so they can be dropped later.
    # Important: pandas missing scalars (pd.NA/NaN) stringify to '<NA>'/'nan'.
    if tag is None or pd.isna(tag):
        return ""
    if not isinstance(tag, str):
        tag = str(tag)
    
    # Lowercase, collapse spaces, strip
    normalized = tag.lower()
    normalized = re.sub(r'\s+', ' ', normalized)
    normalized = normalized.strip()
    
    return normalized


def timestamp_to_utc(timestamp: int) -> datetime:
    """Convert Unix timestamp to UTC datetime.
    
    Args:
        timestamp: Unix timestamp (seconds since epoch)
        
    Returns:
        UTC datetime object
        
    Examples:
        >>> timestamp_to_utc(1225734739)
        datetime.datetime(2008, 11, 3, 17, 52, 19, tzinfo=datetime.timezone.utc)
    """
    return datetime.fromtimestamp(timestamp, tz=timezone.utc)


def clean_movies(df: pd.DataFrame) -> pd.DataFrame:
    """Clean movies table: extract year, split genres.
    
    Input columns: movieId, title, genres
    Output columns: movieId, title, year, genres, genres_list

    Notes:
        - ``title`` preserves canonical MovieLens value (e.g. ``"Toy Story (1995)"``).
        - ``year`` is extracted into a separate nullable integer column.
    
    Args:
        df: Raw movies DataFrame
        
    Returns:
        Clean movies DataFrame with year and genres_list columns
    """
    df = df.copy()
    
    # Extract year from title
    df[['title', 'year']] = df['title'].apply(
        lambda t: pd.Series(extract_year_from_title(t))
    )
    
    # Split genres into list
    df['genres_list'] = df['genres'].apply(split_genres)
    
    # Ensure proper types
    df = df.astype({
        'movieId': 'int32',
        'title': 'string',
        'year': 'Int32',  # Nullable integer
        'genres': 'string',
    })
    
    return df[['movieId', 'title', 'year', 'genres', 'genres_list']]


def clean_ratings(df: pd.DataFrame) -> pd.DataFrame:
    """Clean ratings table: convert timestamp to datetime.
    
    Input columns: userId, movieId, rating, timestamp
    Output columns: userId, movieId, rating, timestamp, rating_dt_utc
    
    Args:
        df: Raw ratings DataFrame
        
    Returns:
        Clean ratings DataFrame with rating_dt_utc column
    """
    df = df.copy()
    
    # Convert timestamp to UTC datetime
    df['rating_dt_utc'] = df['timestamp'].apply(timestamp_to_utc)
    
    # Ensure proper types
    df = df.astype({
        'userId': 'int32',
        'movieId': 'int32',
        'rating': 'float32',
        'timestamp': 'int64',
    })
    df['rating_dt_utc'] = pd.to_datetime(df['rating_dt_utc'], utc=True)
    
    return df[['userId', 'movieId', 'rating', 'timestamp', 'rating_dt_utc']]


def clean_tags(df: pd.DataFrame) -> pd.DataFrame:
    """Clean tags table: normalize tag text, convert timestamp to datetime.
    
    Input columns: userId, movieId, tag, timestamp
    Output columns: userId, movieId, tag_raw, tag_norm, timestamp, tag_dt_utc
    
    Args:
        df: Raw tags DataFrame
        
    Returns:
        Clean tags DataFrame with tag_norm and tag_dt_utc columns
    """
    df = df.copy()
    
    # Rename tag to tag_raw and create normalized version
    df = df.rename(columns={'tag': 'tag_raw'})
    df['tag_norm'] = df['tag_raw'].apply(normalize_tag)

    # Drop empty / missing tags after normalization.
    # (E.g., whitespace-only tags and nulls should not survive into the contract.)
    df['tag_norm'] = df['tag_norm'].replace('', pd.NA)
    df = df.dropna(subset=['tag_norm']).copy()
    
    # Convert timestamp to UTC datetime
    df['tag_dt_utc'] = df['timestamp'].apply(timestamp_to_utc)
    
    # Ensure proper types
    df = df.astype({
        'userId': 'int32',
        'movieId': 'int32',
        'tag_raw': 'string',
        'tag_norm': 'string',
        'timestamp': 'int64',
    })
    df['tag_dt_utc'] = pd.to_datetime(df['tag_dt_utc'], utc=True)
    
    return df[['userId', 'movieId', 'tag_raw', 'tag_norm', 'timestamp', 'tag_dt_utc']]


def clean_links(df: pd.DataFrame) -> pd.DataFrame:
    """Clean links table: ensure proper types.
    
    Input columns: movieId, imdbId, tmdbId
    Output columns: movieId, imdbId, tmdbId
    
    Args:
        df: Raw links DataFrame
        
    Returns:
        Clean links DataFrame
    """
    df = df.copy()
    
    # Ensure proper types
    df = df.astype({
        'movieId': 'int32',
        'imdbId': 'Int64',
        'tmdbId': 'Int64',
    })
    
    return df[['movieId', 'imdbId', 'tmdbId']]


def clean_genome_tags(df: pd.DataFrame) -> pd.DataFrame:
    """Clean genome tags table: ensure proper types.
    
    Input columns: tagId, tag
    Output columns: tagId, tag
    
    Args:
        df: Raw genome tags DataFrame
        
    Returns:
        Clean genome tags DataFrame
    """
    df = df.copy()
    
    # Ensure proper types
    df = df.astype({
        'tagId': 'int32',
        'tag': 'string',
    })
    
    return df[['tagId', 'tag']]


def clean_genome_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Clean genome scores table: ensure proper types.
    
    Input columns: movieId, tagId, relevance
    Output columns: movieId, tagId, relevance
    
    Args:
        df: Raw genome scores DataFrame
        
    Returns:
        Clean genome scores DataFrame
    """
    df = df.copy()
    
    # Ensure proper types
    df = df.astype({
        'movieId': 'int32',
        'tagId': 'int32',
        'relevance': 'float32',
    })
    
    return df[['movieId', 'tagId', 'relevance']]
