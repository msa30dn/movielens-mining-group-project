# MovieLens Data Mining
# Nguyen Sy Hung
# 2026

"""Temporal split module for MovieLens dataset.

This module implements leakage-safe temporal splitting strategies for creating
train/test sets. All splits enforce temporal ordering to prevent data leakage.

Key principles:
- Train data must temporally precede test data
- No overlap between train and test
- Split policy and cutoff stored in metadata for reproducibility
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, Tuple

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def global_time_split(
    ratings_clean: pd.DataFrame,
    cutoff_date: str | datetime | None = None,
    test_fraction: float = 0.1,
    datetime_col: str = "rating_dt_utc",
) -> Tuple[pd.DataFrame, pd.DataFrame, datetime]:
    """Split ratings into train/test based on global temporal cutoff.
    
    This implements a simple and leakage-safe split: all ratings before the cutoff
    go to train, all ratings after go to test. This ensures temporal ordering and
    is appropriate for time-series evaluation.
    
    Args:
        ratings_clean: Clean ratings DataFrame with datetime column
        cutoff_date: Explicit cutoff datetime (or None to auto-compute)
        test_fraction: Fraction of timeline for test (default 0.1 = 10%)
        datetime_col: Name of datetime column (default: rating_dt_utc)
        
    Returns:
        Tuple of (train_df, test_df, cutoff_datetime)
        
    Raises:
        ValueError: If datetime column missing or invalid test_fraction
        
    Examples:
        >>> train, test, cutoff = global_time_split(ratings, test_fraction=0.1)
        >>> assert train[datetime_col].max() <= test[datetime_col].min()
    """
    if datetime_col not in ratings_clean.columns:
        raise ValueError(f"Column '{datetime_col}' not found in ratings DataFrame")
    
    if not 0 < test_fraction < 1:
        raise ValueError(f"test_fraction must be in (0, 1), got {test_fraction}")
    
    # Determine cutoff date
    if cutoff_date is None:
        # Auto-compute: use test_fraction of the timeline
        min_date = ratings_clean[datetime_col].min()
        max_date = ratings_clean[datetime_col].max()
        
        # Compute cutoff as a point on the timeline
        timeline_span = (max_date - min_date).total_seconds()
        cutoff_seconds = timeline_span * (1 - test_fraction)
        cutoff = min_date + pd.Timedelta(seconds=cutoff_seconds)
        
        logger.info(f"Auto-computed cutoff from test_fraction={test_fraction:.2%}")
        logger.info(f"  Timeline: {min_date} → {max_date}")
        logger.info(f"  Cutoff: {cutoff}")
    else:
        # Use explicit cutoff
        if isinstance(cutoff_date, str):
            cutoff = pd.to_datetime(cutoff_date, utc=True)
        else:
            cutoff = cutoff_date
        
        logger.info(f"Using explicit cutoff: {cutoff}")
    
    # Split by cutoff
    train_mask = ratings_clean[datetime_col] < cutoff
    test_mask = ratings_clean[datetime_col] >= cutoff
    
    train_df = ratings_clean[train_mask].copy()
    test_df = ratings_clean[test_mask].copy()
    
    logger.info(f"Split results:")
    logger.info(f"  Train: {len(train_df):,} ratings ({len(train_df)/len(ratings_clean):.1%})")
    logger.info(f"  Test:  {len(test_df):,} ratings ({len(test_df)/len(ratings_clean):.1%})")
    
    return train_df, test_df, cutoff


def per_user_time_split(
    ratings_clean: pd.DataFrame,
    test_fraction: float = 0.1,
    datetime_col: str = "rating_dt_utc",
    user_col: str = "userId",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split ratings into train/test per user based on each user's timeline.

    This reduces user cold-start in the test set vs a global cutoff.

    Returns:
        (train_df, test_df, user_cutoffs_df)
        where user_cutoffs_df has columns [userId, user_cutoff_dt_utc] for users
        that have at least one test row.
    """
    if datetime_col not in ratings_clean.columns:
        raise ValueError(f"Column '{datetime_col}' not found in ratings DataFrame")
    if user_col not in ratings_clean.columns:
        raise ValueError(f"Column '{user_col}' not found in ratings DataFrame")
    if "timestamp" not in ratings_clean.columns:
        raise ValueError("Column 'timestamp' not found in ratings DataFrame")
    if not 0 < test_fraction < 1:
        raise ValueError(f"test_fraction must be in (0, 1), got {test_fraction}")

    df = ratings_clean.copy()
    df = df.sort_values([user_col, datetime_col, "timestamp"], ascending=[True, True, True])

    sizes = df.groupby(user_col, observed=True).size().rename("n")
    df = df.join(sizes, on=user_col)
    df["pos"] = df.groupby(user_col, observed=True).cumcount()

    # Split index per user (exclusive upper bound for train)
    k = ((1.0 - test_fraction) * df["n"]).astype("int64")
    # Clamp for users with n>=2: 1 <= k <= n-1
    k = k.where(df["n"] < 2, k.clip(lower=1))
    k = k.where(df["n"] < 2, k.clip(upper=df["n"] - 1))
    df["k"] = k

    train_mask = df["pos"] < df["k"]
    test_mask = (~train_mask) & (df["n"] >= 2)

    train_df = df[train_mask].drop(columns=["n", "pos", "k"]).copy()
    test_df = df[test_mask].drop(columns=["n", "pos", "k"]).copy()

    if test_df.empty:
        cutoffs = pd.DataFrame(columns=[user_col, "user_cutoff_dt_utc"])
    else:
        cutoffs = (
            test_df.groupby(user_col, observed=True)[datetime_col]
            .min()
            .rename("user_cutoff_dt_utc")
            .reset_index()
        )

    return train_df, test_df, cutoffs


def validate_split(
    train: pd.DataFrame,
    test: pd.DataFrame,
    cutoff_date: datetime | None,
    datetime_col: str = "rating_dt_utc",
    split_policy: str = "global",
) -> Dict[str, Any]:
    """Validate temporal split constraints and compute quality metrics.
    
    This function performs comprehensive validation of a train/test split to ensure:
    1. No temporal paradox (train after test)
    2. No overlap between sets
    3. Reasonable cold-start statistics
    4. Distribution preservation checks
    
    Args:
        train: Training DataFrame
        test: Test DataFrame
        cutoff_date: Split cutoff datetime
        datetime_col: Name of datetime column
        
    Returns:
        Dictionary with validation results and metrics
        
    Raises:
        ValueError: If hard constraints are violated (overlap, time paradox)
    """
    if split_policy not in {"global", "per_user"}:
        raise ValueError(f"split_policy must be 'global' or 'per_user', got {split_policy}")

    results = {
        "cutoff_date": str(cutoff_date) if cutoff_date is not None else None,
        "datetime_col": datetime_col,
        "split_policy": split_policy,
        "checks": {},
        "stats": {},
        "warnings": [],
    }

    # Check 1: Temporal ordering (HARD GATE)
    if split_policy == "global":
        train_max = train[datetime_col].max()
        test_min = test[datetime_col].min()

        temporal_ok = train_max <= test_min
        results["checks"]["temporal_ordering"] = {
            "passed": temporal_ok,
            "policy": "global",
            "train_max": str(train_max),
            "test_min": str(test_min),
            "gap_seconds": (test_min - train_max).total_seconds() if temporal_ok else None,
        }

        if not temporal_ok:
            raise ValueError(
                f"Temporal paradox detected! Train max ({train_max}) > Test min ({test_min})"
            )
        logger.info("✓ Global temporal ordering valid")
    else:
        train_max_u = train.groupby("userId", observed=True)[datetime_col].max()
        test_min_u = test.groupby("userId", observed=True)[datetime_col].min()
        common = train_max_u.index.intersection(test_min_u.index)

        if len(common) == 0:
            temporal_ok = True
            violations = 0
        else:
            temporal_ok_mask = train_max_u.loc[common] <= test_min_u.loc[common]
            violations = int((~temporal_ok_mask).sum())
            temporal_ok = violations == 0

        results["checks"]["temporal_ordering"] = {
            "passed": temporal_ok,
            "policy": "per_user",
            "users_in_test": int(test["userId"].nunique()),
            "violating_users": violations,
        }
        if not temporal_ok:
            raise ValueError(f"Per-user temporal paradox for {violations} users")
        logger.info("✓ Per-user temporal ordering valid")
    
    # Check 2: No overlap (HARD GATE)
    # Use a composite key for overlap check (userId, movieId, timestamp)
    train_keys = set(zip(train["userId"], train["movieId"], train["timestamp"]))
    test_keys = set(zip(test["userId"], test["movieId"], test["timestamp"]))
    
    overlap = train_keys & test_keys
    overlap_ok = len(overlap) == 0
    
    results["checks"]["overlap"] = {
        "passed": overlap_ok,
        "overlap_count": len(overlap),
    }
    
    if not overlap_ok:
        raise ValueError(f"Overlap detected: {len(overlap)} rows appear in both train and test")
    
    logger.info(f"✓ No overlap detected")
    
    # Check 3: Cold-start statistics (SOFT GATE - report only)
    train_users = set(train["userId"].unique())
    test_users = set(test["userId"].unique())
    train_movies = set(train["movieId"].unique())
    test_movies = set(test["movieId"].unique())
    
    cold_start_users = test_users - train_users
    cold_start_movies = test_movies - train_movies
    
    user_cold_start_rate = len(cold_start_users) / len(test_users) if test_users else 0
    movie_cold_start_rate = len(cold_start_movies) / len(test_movies) if test_movies else 0
    
    results["stats"]["cold_start"] = {
        "users": {
            "train_unique": len(train_users),
            "test_unique": len(test_users),
            "cold_start_count": len(cold_start_users),
            "cold_start_rate": round(user_cold_start_rate, 4),
        },
        "movies": {
            "train_unique": len(train_movies),
            "test_unique": len(test_movies),
            "cold_start_count": len(cold_start_movies),
            "cold_start_rate": round(movie_cold_start_rate, 4),
        },
    }
    
    logger.info(f"Cold-start statistics:")
    logger.info(f"  Users: {len(cold_start_users):,} cold-start ({user_cold_start_rate:.1%})")
    logger.info(f"  Movies: {len(cold_start_movies):,} cold-start ({movie_cold_start_rate:.1%})")
    
    if user_cold_start_rate > 0.5:
        warning = f"High user cold-start rate: {user_cold_start_rate:.1%}"
        results["warnings"].append(warning)
        logger.warning(warning)
    
    if movie_cold_start_rate > 0.5:
        warning = f"High movie cold-start rate: {movie_cold_start_rate:.1%}"
        results["warnings"].append(warning)
        logger.warning(warning)
    
    # Check 4: Distribution preservation (rating values)
    train_rating_dist = train["rating"].value_counts(normalize=True).sort_index()
    test_rating_dist = test["rating"].value_counts(normalize=True).sort_index()
    
    # Compute KL divergence (simple distribution similarity check)
    common_ratings = train_rating_dist.index.intersection(test_rating_dist.index)
    if len(common_ratings) > 0:
        train_probs = train_rating_dist[common_ratings].values
        test_probs = test_rating_dist[common_ratings].values
        
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        kl_div = np.sum(test_probs * np.log((test_probs + epsilon) / (train_probs + epsilon)))
        
        results["stats"]["distribution"] = {
            "rating_kl_divergence": round(kl_div, 6),
            "train_mean": round(train["rating"].mean(), 3),
            "test_mean": round(test["rating"].mean(), 3),
            "train_std": round(train["rating"].std(), 3),
            "test_std": round(test["rating"].std(), 3),
        }
        
        logger.info(f"Rating distribution similarity (KL div): {kl_div:.6f}")
        
        if kl_div > 0.1:
            warning = f"High KL divergence ({kl_div:.4f}) - distributions may differ significantly"
            results["warnings"].append(warning)
            logger.warning(warning)
    
    # Check 5: Size statistics
    results["stats"]["sizes"] = {
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "total_rows": int(len(train) + len(test)),
        "test_fraction": round(len(test) / (len(train) + len(test)), 4),
    }
    
    return results


def build_split_report(
    validation_results: Dict[str, Any],
    cutoff_date: datetime | None,
    config: Any,
    run_id: str,
) -> Dict[str, Any]:
    """Build comprehensive split report for metadata.
    
    Args:
        validation_results: Output from validate_split()
        cutoff_date: Split cutoff datetime
        config: Pipeline configuration object
        run_id: Run identifier
        
    Returns:
        Complete split report dictionary
    """
    report = {
        "run_id": run_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "split_policy": getattr(config, "split_policy", "global"),
        "cutoff_date": str(cutoff_date) if cutoff_date is not None else None,
        "config": {
            "test_fraction": config.test_frac,
            "datetime_column": "rating_dt_utc",
        },
        "validation": validation_results,
        "summary": {
            "passed_all_checks": len(validation_results.get("warnings", [])) == 0,
            "warning_count": len(validation_results.get("warnings", [])),
        },
    }
    
    return report


def write_split_report(report: Dict[str, Any], path: str) -> None:
    """Write split report to JSON file.
    
    Args:
        report: Split report dictionary
        path: Output file path
    """
    from pathlib import Path
    import json
    
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    def _json_default(obj: Any) -> Any:
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (pd.Timestamp, datetime)):
            return obj.isoformat()
        return str(obj)

    with open(path_obj, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, default=_json_default)
    
    logger.info(f"Split report written to {path_obj}")
