# MovieLens Data Mining
#
# Nguyen Sy Hung
# 2026

from __future__ import annotations

import re

import pandas as pd


def _slugify(value: str) -> str:
    token = re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")
    return token or "unknown"


def _tokenize_movie(movie_id: int) -> str:
    return f"movie:{int(movie_id)}"


def _tokenize_genre(genre: str) -> str:
    return f"genre:{_slugify(genre)}"


def _tokenize_tag(tag: str) -> str:
    return f"tag:{_slugify(tag)}"


def build_transactions_train(
    interactions_train: pd.DataFrame,
    movies_clean: pd.DataFrame,
    tags_train: pd.DataFrame | None = None,
    like_threshold: float = 4.0,
    min_tag_freq: int = 10,
    include_movie_tokens: bool = True,
    include_genre_tokens: bool = True,
    include_tag_tokens: bool = False,
) -> pd.DataFrame:
    """Build per-user basket transactions from train interactions only."""
    liked = interactions_train[interactions_train["rating"] >= like_threshold][
        ["userId", "movieId"]
    ].drop_duplicates()

    token_frames: list[pd.DataFrame] = []

    if include_movie_tokens and not liked.empty:
        movie_tokens = liked.copy()
        movie_tokens["item"] = movie_tokens["movieId"].map(_tokenize_movie)
        token_frames.append(movie_tokens[["userId", "item"]])

    if include_genre_tokens and not liked.empty:
        genre_lookup = (
            movies_clean[["movieId", "genres_list"]]
            .explode("genres_list")
            .rename(columns={"genres_list": "genre"})
        )
        genre_lookup = genre_lookup[genre_lookup["genre"].notna()]
        if not genre_lookup.empty:
            genre_tokens = liked.merge(genre_lookup, on="movieId", how="inner")
            if not genre_tokens.empty:
                genre_tokens["item"] = genre_tokens["genre"].map(_tokenize_genre)
                token_frames.append(genre_tokens[["userId", "item"]])

    if include_tag_tokens and tags_train is not None and not tags_train.empty and not liked.empty:
        tag_counts = tags_train["tag_norm"].value_counts()
        allowed_tags = set(tag_counts[tag_counts >= min_tag_freq].index)
        filtered_tags = tags_train[tags_train["tag_norm"].isin(allowed_tags)][
            ["userId", "movieId", "tag_norm"]
        ].drop_duplicates()
        if not filtered_tags.empty:
            user_movie_tags = liked.merge(filtered_tags, on=["userId", "movieId"], how="inner")
            if not user_movie_tags.empty:
                user_movie_tags["item"] = user_movie_tags["tag_norm"].map(_tokenize_tag)
                token_frames.append(user_movie_tags[["userId", "item"]])

    if not token_frames:
        empty_out = pd.DataFrame(columns=["userId", "items", "n_items"])
        empty_out["userId"] = empty_out["userId"].astype("int32")
        empty_out["n_items"] = empty_out["n_items"].astype("int32")
        return empty_out

    tokens = pd.concat(token_frames, ignore_index=True).drop_duplicates()
    transactions = (
        tokens.groupby("userId", observed=True)["item"]
        .agg(lambda xs: sorted(set(xs)))
        .rename("items")
        .reset_index()
    )
    transactions["n_items"] = transactions["items"].map(len).astype("int32")
    transactions["userId"] = transactions["userId"].astype("int32")

    return transactions[["userId", "items", "n_items"]]
