"""User feature builders for train-only data.

MovieLens Data Mining
Nguyen Sy Hung
2026
"""

from __future__ import annotations

import re
from typing import Iterable

import pandas as pd


def _slugify(value: str) -> str:
    token = re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")
    return token or "unknown"


def _prepare_movie_genres(movies_clean: pd.DataFrame) -> pd.DataFrame:
    genre_rows = (
        movies_clean[["movieId", "genres_list"]]
        .explode("genres_list")
        .rename(columns={"genres_list": "genre"})
    )
    genre_rows = genre_rows[genre_rows["genre"].notna()]
    genre_rows["genre"] = genre_rows["genre"].astype("string")
    return genre_rows


def _build_user_genre_preferences(
    interactions_train: pd.DataFrame,
    movies_clean: pd.DataFrame,
) -> pd.DataFrame:
    genre_rows = _prepare_movie_genres(movies_clean)
    if genre_rows.empty:
        return pd.DataFrame(columns=["userId"])

    joined = interactions_train[["userId", "movieId", "rating"]].merge(
        genre_rows,
        on="movieId",
        how="inner",
    )
    if joined.empty:
        return pd.DataFrame(columns=["userId"])

    user_genre = (
        joined.groupby(["userId", "genre"], observed=True)["rating"]
        .mean()
        .unstack(fill_value=0.0)
        .reset_index()
    )

    rename_map = {
        col: f"genre_pref__{_slugify(col)}"
        for col in user_genre.columns
        if col != "userId"
    }
    user_genre = user_genre.rename(columns=rename_map)
    return user_genre


def _build_user_tag_stats(tags_train: pd.DataFrame | None) -> pd.DataFrame:
    if tags_train is None or tags_train.empty:
        return pd.DataFrame(columns=["userId", "n_tag_events"])

    out = (
        tags_train.groupby("userId", observed=True)
        .agg(
            n_tag_events=("tag_norm", "count"),
        )
        .reset_index()
    )
    return out


def build_user_features_train(
    interactions_train: pd.DataFrame,
    movies_clean: pd.DataFrame,
    tags_train: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build train-only user feature table.

    Expected output columns include baseline behavior metrics, activity window,
    genre preference vectors, and tag usage counts.
    """
    ratings = interactions_train.copy()
    ratings["rating_date"] = ratings["rating_dt_utc"].dt.date

    user_base = (
        ratings.groupby("userId", observed=True)
        .agg(
            n_ratings=("rating", "count"),
            rating_mean=("rating", "mean"),
            rating_std=("rating", "std"),
            rating_min=("rating", "min"),
            rating_max=("rating", "max"),
            first_dt=("rating_dt_utc", "min"),
            last_dt=("rating_dt_utc", "max"),
            active_days=("rating_date", "nunique"),
        )
        .reset_index()
    )
    user_base["rating_std"] = user_base["rating_std"].fillna(0.0)

    user_genre = _build_user_genre_preferences(interactions_train, movies_clean)
    user_tags = _build_user_tag_stats(tags_train)

    features = user_base.merge(user_genre, on="userId", how="left").merge(
        user_tags,
        on="userId",
        how="left",
    )

    if "n_tag_events" not in features.columns:
        features["n_tag_events"] = 0

    features["n_tag_events"] = features["n_tag_events"].fillna(0).astype("int32")

    numeric_cols: Iterable[str] = [
        "n_ratings",
        "rating_mean",
        "rating_std",
        "rating_min",
        "rating_max",
        "active_days",
    ]
    for col in numeric_cols:
        features[col] = pd.to_numeric(features[col], errors="coerce")

    features["userId"] = features["userId"].astype("int32")
    features["n_ratings"] = features["n_ratings"].fillna(0).astype("int32")
    features["active_days"] = features["active_days"].fillna(0).astype("int32")
    for col in ["rating_mean", "rating_std", "rating_min", "rating_max"]:
        features[col] = features[col].astype("float32")

    genre_cols = [c for c in features.columns if c.startswith("genre_pref__")]
    for col in genre_cols:
        features[col] = features[col].fillna(0.0).astype("float32")

    ordered_prefix = [
        "userId",
        "n_ratings",
        "rating_mean",
        "rating_std",
        "rating_min",
        "rating_max",
        "first_dt",
        "last_dt",
        "active_days",
        "n_tag_events",
    ]
    ordered = ordered_prefix + sorted(genre_cols)
    return features[ordered]
