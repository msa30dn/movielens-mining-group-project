"""Movie feature builders for train-only data.

MovieLens Data Mining
Nguyen Sy Hung
2026
"""

from __future__ import annotations

import math
import re

import pandas as pd


def _slugify(value: str) -> str:
    token = re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")
    return token or "unknown"


def _rating_entropy(series: pd.Series) -> float:
    probs = series.value_counts(normalize=True)
    if probs.empty:
        return 0.0
    entropy = float(-(probs * probs.map(math.log)).sum())
    max_entropy = math.log(len(probs)) if len(probs) > 1 else 1.0
    return entropy / max_entropy if max_entropy > 0 else 0.0


def _movie_genre_flags(movies_clean: pd.DataFrame) -> pd.DataFrame:
    genre_rows = (
        movies_clean[["movieId", "genres_list"]]
        .explode("genres_list")
        .rename(columns={"genres_list": "genre"})
    )
    genre_rows = genre_rows[genre_rows["genre"].notna()]
    if genre_rows.empty:
        return pd.DataFrame(columns=["movieId"])

    genre_rows["value"] = 1
    out = (
        genre_rows.pivot_table(
            index="movieId",
            columns="genre",
            values="value",
            aggfunc="max",
            fill_value=0,
        )
        .reset_index()
    )
    rename_map = {
        col: f"genre_flag__{_slugify(col)}"
        for col in out.columns
        if col != "movieId"
    }
    out = out.rename(columns=rename_map)
    return out


def _movie_tag_stats(tags_train: pd.DataFrame | None, min_tag_freq: int) -> pd.DataFrame:
    if tags_train is None or tags_train.empty:
        return pd.DataFrame(columns=["movieId", "n_tag_events", "top_tags"])

    tag_counts = tags_train["tag_norm"].value_counts()
    allowed_tags = set(tag_counts[tag_counts >= min_tag_freq].index)

    filtered = tags_train[tags_train["tag_norm"].isin(allowed_tags)].copy()
    if filtered.empty:
        summary = (
            tags_train.groupby("movieId", observed=True)
            .agg(
                n_tag_events=("tag_norm", "count"),
            )
            .reset_index()
        )
        summary["top_tags"] = ""
        return summary

    summary = (
        filtered.groupby("movieId", observed=True)
        .agg(
            n_tag_events=("tag_norm", "count"),
        )
        .reset_index()
    )

    ranked = (
        filtered.groupby(["movieId", "tag_norm"], observed=True)
        .size()
        .rename("freq")
        .reset_index()
        .sort_values(["movieId", "freq", "tag_norm"], ascending=[True, False, True])
    )
    ranked["rank"] = ranked.groupby("movieId", observed=True).cumcount() + 1
    top3 = ranked[ranked["rank"] <= 3]
    top_tags = (
        top3.groupby("movieId", observed=True)["tag_norm"]
        .agg(lambda xs: "|".join(map(str, xs)))
        .rename("top_tags")
        .reset_index()
    )

    return summary.merge(top_tags, on="movieId", how="left")


def build_movie_features_train(
    interactions_train: pd.DataFrame,
    movies_clean: pd.DataFrame,
    tags_train: pd.DataFrame | None = None,
    min_tag_freq: int = 10,
) -> pd.DataFrame:
    """Build train-only movie feature table."""
    movie_base = (
        interactions_train.groupby("movieId", observed=True)
        .agg(
            n_ratings=("rating", "count"),
            rating_mean=("rating", "mean"),
            rating_std=("rating", "std"),
            rating_min=("rating", "min"),
            rating_max=("rating", "max"),
        )
        .reset_index()
    )
    movie_base["rating_std"] = movie_base["rating_std"].fillna(0.0)

    entropy_df = (
        interactions_train.groupby("movieId", observed=True)["rating"]
        .apply(_rating_entropy)
        .rename("rating_entropy")
        .reset_index()
    )

    movie_base = movie_base.merge(entropy_df, on="movieId", how="left")
    movie_base["rating_entropy"] = movie_base["rating_entropy"].fillna(0.0)
    movie_base["polarization_score"] = movie_base["rating_std"] * (
        1.0 - movie_base["rating_entropy"]
    )
    movie_base["polarization_score"] = movie_base["polarization_score"].clip(lower=0.0)

    genre_flags = _movie_genre_flags(movies_clean)
    tag_stats = _movie_tag_stats(tags_train, min_tag_freq=min_tag_freq)

    features = movie_base.merge(genre_flags, on="movieId", how="left").merge(
        tag_stats,
        on="movieId",
        how="left",
    )

    for col in ["n_tag_events"]:
        if col not in features.columns:
            features[col] = 0
        features[col] = features[col].fillna(0).astype("int32")

    if "top_tags" not in features.columns:
        features["top_tags"] = ""
    features["top_tags"] = features["top_tags"].fillna("").astype("string")

    features["movieId"] = features["movieId"].astype("int32")
    features["n_ratings"] = features["n_ratings"].fillna(0).astype("int32")
    for col in [
        "rating_mean",
        "rating_std",
        "rating_min",
        "rating_max",
        "rating_entropy",
        "polarization_score",
    ]:
        features[col] = features[col].fillna(0.0).astype("float32")

    genre_cols = [c for c in features.columns if c.startswith("genre_flag__")]
    for col in genre_cols:
        features[col] = features[col].fillna(0).astype("int8")

    ordered_prefix = [
        "movieId",
        "n_ratings",
        "rating_mean",
        "rating_std",
        "rating_min",
        "rating_max",
        "rating_entropy",
        "polarization_score",
        "n_tag_events",
        "top_tags",
    ]
    ordered = ordered_prefix + sorted(genre_cols)
    return features[ordered]
