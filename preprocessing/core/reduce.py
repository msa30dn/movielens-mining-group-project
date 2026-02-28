"""
core/reduce.py
==============
Vocabulary-based reduction of ``transactions_train.parquet`` for Story B.1
(Frequent Patterns / Association Rules).

Design rationale (from docs/Reduction-guide.md):
- Token/vocabulary pruning belongs in the backbone: it defines *what the
    transaction database is*, keeps the contract consistent, and makes results
    reproducible across story modules.
- Mining knobs (min_support, min_confidence) remain in Story B.1.

Public API
----------
reduce_transactions(transactions, ...)       -> (reduced_df, manifest_dict)
build_token_stats(transactions, reduced)     -> token_stats_df
build_basket_stats(transactions, reduced)    -> basket_stats_dict
"""
# MovieLens Data Mining
# Nguyen Sy Hung
# 2026
from __future__ import annotations

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# ── token family helpers ──────────────────────────────────────────────────────

def _token_family(token: str) -> str:
    """Return the prefix family of a token string, e.g. 'genre', 'movie', 'tag'."""
    if ":" in token:
        return token.split(":", 1)[0]
    return "unknown"


def _explode_items(transactions: pd.DataFrame) -> pd.DataFrame:
    """Explode the items list column into one (userId, token) row per token."""
    if transactions.empty:
        return pd.DataFrame(columns=["userId", "token"])
    exploded = transactions[["userId", "items"]].explode("items").rename(columns={"items": "token"})
    exploded = exploded[exploded["token"].notna() & (exploded["token"] != "")]
    return exploded.reset_index(drop=True)


# ── main reduction function ───────────────────────────────────────────────────

def reduce_transactions(
    transactions: pd.DataFrame,
    token_families: tuple[str, ...] | list[str] = ("genre", "movie", "tag"),
    min_token_df: int = 5,
    top_k_movies: int | None = 5_000,
    top_k_tags: int | None = 500,
    min_items_per_basket: int = 2,
    max_items_per_basket: int | None = None,
    sample_frac: float | None = None,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Reduce transactions_train by vocabulary pruning and optional basket filtering.

    Reduction knobs (in order of ROI, per Reduction-guide.md):

    1. ``token_families``  — only keep tokens whose prefix is in this set.
       Use ``("genre",)`` for cheapest run; ``("genre", "movie", "tag")`` for full.
    2. ``min_token_df``    — drop tokens that appear in fewer than this many baskets
       (document frequency). Applied after family filtering.
    3. ``top_k_movies``    — after df filtering, keep only the top-K ``movie:*``
       tokens by document frequency. ``None`` = no cap.
    4. ``top_k_tags``      — same for ``tag:*`` tokens. ``None`` = no cap.
    5. ``min_items_per_basket`` — drop baskets that become too short after vocab
       filtering (default 2 to remove singletons).
    6. ``max_items_per_basket`` — cap basket size by keeping genres first, then
       movies, then tags, ranked by global document frequency. ``None`` = no cap.
    7. ``sample_frac``     — randomly sample this fraction of baskets **after**
       all filtering. Intended for dev/exploration only; ``None`` = keep all.

    Parameters
    ----------
    transactions : pd.DataFrame
        Must have columns ``userId`` (int) and ``items`` (list[str]).
    token_families : tuple[str, ...]
        Token family prefixes to retain (e.g. ``"genre"``, ``"movie"``, ``"tag"``).
    min_token_df : int
        Minimum number of baskets a token must appear in to be kept.
    top_k_movies : int | None
        Keep only the top-K most frequent ``movie:*`` tokens. No cap if ``None``.
    top_k_tags : int | None
        Keep only the top-K most frequent ``tag:*`` tokens. No cap if ``None``.
    min_items_per_basket : int
        Drop baskets with fewer tokens than this after vocab filtering.
    max_items_per_basket : int | None
        Cap basket size. When capping, tokens are kept in order:
        genres first, movies second, tags last (within each family: most frequent
        globally → least frequent).
    sample_frac : float | None
        Fraction of baskets to sample (0 < f ≤ 1). ``None`` = no sampling.
    random_seed : int
        Random seed for reproducibility.

    Returns
    -------
    reduced : pd.DataFrame
        Same schema as input (``userId``, ``items``, ``n_items``), reduced.
    manifest : dict
        All reduction parameters + before/after counts for provenance.
    """
    n_baskets_in = int(len(transactions))
    n_tokens_in = int(transactions["n_items"].sum()) if "n_items" in transactions.columns else \
        int(transactions["items"].map(len).sum())
    vocab_in = int(transactions["items"].explode().nunique())
    vocab_in_set = set(transactions["items"].explode().dropna().unique())

    logger.info(
        "reduce_transactions: %d baskets, %d total tokens, %d unique vocab in",
        n_baskets_in, n_tokens_in, vocab_in,
    )

    if transactions.empty:
        manifest = _build_manifest(
            token_families=token_families,
            min_token_df=min_token_df,
            top_k_movies=top_k_movies,
            top_k_tags=top_k_tags,
            min_items_per_basket=min_items_per_basket,
            max_items_per_basket=max_items_per_basket,
            sample_frac=sample_frac,
            random_seed=random_seed,
            n_baskets_in=0, n_baskets_out=0,
            n_tokens_in=0, n_tokens_out=0,
            vocab_in=0, vocab_out=0,
            vocab_removed=[],
        )
        empty = pd.DataFrame(columns=["userId", "items", "n_items"])
        empty["userId"] = empty["userId"].astype("int32")
        empty["n_items"] = empty["n_items"].astype("int32")
        return empty, manifest

    # ── Step 1: explode ─────────────────────────────────────────────────────
    long = _explode_items(transactions)
    long["family"] = long["token"].map(_token_family)

    # ── Step 2: family filter ────────────────────────────────────────────────
    allowed_families = set(token_families)
    long = long[long["family"].isin(allowed_families)].copy()

    # ── Step 3: document frequency (df = number of baskets containing token) ─
    doc_freq = long.groupby("token")["userId"].nunique().rename("df")
    long = long.join(doc_freq, on="token")
    long = long[long["df"] >= min_token_df].drop(columns=["df"])

    # ── Step 4: top-K restrictions per family ────────────────────────────────
    # Recompute df on the min_token_df-filtered set
    doc_freq2 = long.groupby("token")["userId"].nunique().rename("df")

    allowed_tokens: set[str] = set(doc_freq2.index)

    if top_k_movies is not None:
        movie_tokens = doc_freq2[doc_freq2.index.str.startswith("movie:")].nlargest(top_k_movies)
        non_movie = doc_freq2[~doc_freq2.index.str.startswith("movie:")].index
        allowed_tokens = (set(movie_tokens.index) | set(non_movie)) & allowed_tokens

    if top_k_tags is not None:
        tag_tokens = doc_freq2[doc_freq2.index.str.startswith("tag:")].nlargest(top_k_tags)
        non_tag = doc_freq2[~doc_freq2.index.str.startswith("tag:")].index
        allowed_tokens = (set(tag_tokens.index) | set(non_tag)) & allowed_tokens

    vocab_removed = sorted(set(doc_freq2.index) - allowed_tokens)
    long = long[long["token"].isin(allowed_tokens)].copy()

    logger.info(
        "  After vocab filter: %d unique tokens remain (%d removed)",
        len(allowed_tokens), len(vocab_removed),
    )

    # ── Step 5: re-aggregate baskets ────────────────────────────────────────
    if long.empty:
        vocab_removed_all = sorted(vocab_in_set)
        reduced = pd.DataFrame(columns=["userId", "items", "n_items"])
        reduced["userId"] = reduced["userId"].astype("int32")
        reduced["n_items"] = reduced["n_items"].astype("int32")
        manifest = _build_manifest(
            token_families=token_families,
            min_token_df=min_token_df,
            top_k_movies=top_k_movies,
            top_k_tags=top_k_tags,
            min_items_per_basket=min_items_per_basket,
            max_items_per_basket=max_items_per_basket,
            sample_frac=sample_frac,
            random_seed=random_seed,
            n_baskets_in=n_baskets_in, n_baskets_out=0,
            n_tokens_in=n_tokens_in, n_tokens_out=0,
            vocab_in=vocab_in, vocab_out=0,
            vocab_removed=vocab_removed_all,
        )
        return reduced, manifest

    reduced = (
        long.groupby("userId", observed=True)["token"]
        .agg(lambda xs: sorted(set(xs)))
        .rename("items")
        .reset_index()
    )
    reduced["n_items"] = reduced["items"].map(len).astype("int32")
    reduced["userId"] = reduced["userId"].astype("int32")

    # ── Step 6: min_items_per_basket ─────────────────────────────────────────
    before_min = len(reduced)
    reduced = reduced[reduced["n_items"] >= min_items_per_basket].reset_index(drop=True)
    logger.info(
        "  After min_items_per_basket=%d: %d baskets remain (%d dropped)",
        min_items_per_basket, len(reduced), before_min - len(reduced),
    )

    # ── Step 7: max_items_per_basket (cap, keeping genres first) ─────────────
    if max_items_per_basket is not None and not reduced.empty:
        # Global token ranking: genre > movie > tag, then by doc_freq desc
        family_rank = {"genre": 0, "movie": 1, "tag": 2}
        token_rank = (
            doc_freq2.rename("df")
            .reset_index()
            .rename(columns={"index": "token"})
        )
        token_rank["family"] = token_rank["token"].map(_token_family)
        token_rank["family_rank"] = token_rank["family"].map(lambda f: family_rank.get(f, 99))
        # Sort by family_rank asc, df desc → assigning a global priority rank
        token_rank = token_rank.sort_values(
            ["family_rank", "df"], ascending=[True, False]
        )
        token_priority: dict[str, int] = {
            tok: i for i, tok in enumerate(token_rank["token"])
        }

        def _cap(items: list[str]) -> list[str]:
            ranked = sorted(items, key=lambda t: token_priority.get(t, 999_999))
            return ranked[:max_items_per_basket]

        reduced["items"] = reduced["items"].map(_cap)
        reduced["n_items"] = reduced["items"].map(len).astype("int32")
        reduced = reduced[reduced["n_items"] >= min_items_per_basket].reset_index(drop=True)
        logger.info("  After max_items_per_basket=%d: %d baskets", max_items_per_basket, len(reduced))

    # ── Step 8: sampling ─────────────────────────────────────────────────────
    if sample_frac is not None and not reduced.empty:
        n_before = len(reduced)
        reduced = reduced.sample(frac=sample_frac, random_state=random_seed).reset_index(drop=True)
        logger.info(
            "  After sample_frac=%.3f: %d baskets (was %d)",
            sample_frac, len(reduced), n_before,
        )

    # ── Final stats ──────────────────────────────────────────────────────────
    n_baskets_out = int(len(reduced))
    n_tokens_out = int(reduced["n_items"].sum()) if not reduced.empty else 0
    vocab_out = int(reduced["items"].explode().nunique()) if not reduced.empty else 0
    vocab_out_set = set(reduced["items"].explode().dropna().unique()) if not reduced.empty else set()
    vocab_removed_all = sorted(vocab_in_set - vocab_out_set)

    logger.info(
        "reduce_transactions done: %d → %d baskets, vocab %d → %d",
        n_baskets_in, n_baskets_out, vocab_in, vocab_out,
    )

    manifest = _build_manifest(
        token_families=token_families,
        min_token_df=min_token_df,
        top_k_movies=top_k_movies,
        top_k_tags=top_k_tags,
        min_items_per_basket=min_items_per_basket,
        max_items_per_basket=max_items_per_basket,
        sample_frac=sample_frac,
        random_seed=random_seed,
        n_baskets_in=n_baskets_in,
        n_baskets_out=n_baskets_out,
        n_tokens_in=n_tokens_in,
        n_tokens_out=n_tokens_out,
        vocab_in=vocab_in,
        vocab_out=vocab_out,
        vocab_removed=vocab_removed_all,
    )
    return reduced, manifest


# ── companion artefact builders ───────────────────────────────────────────────

def build_token_stats(
    transactions: pd.DataFrame,
    reduced: pd.DataFrame,
) -> pd.DataFrame:
    """Build a token-level stats table comparing before/after reduction.

    Returns a DataFrame with columns:
        token, family, df_before, df_after, kept
    """
    def _df(txn: pd.DataFrame, col: str = "df") -> pd.Series:
        if txn.empty:
            return pd.Series(dtype="int64", name=col)
        return (
            txn[["userId", "items"]]
            .explode("items")
            .rename(columns={"items": "token"})
            .groupby("token")["userId"]
            .nunique()
            .rename(col)
        )

    df_before = _df(transactions, "df_before")
    df_after = _df(reduced, "df_after")

    stats = (
        df_before.to_frame()
        .join(df_after, how="outer")
        .fillna(0)
        .astype(int)
        .reset_index()
    )
    stats.columns = ["token", "df_before", "df_after"]
    stats["family"] = stats["token"].map(_token_family)
    stats["kept"] = stats["df_after"] > 0
    stats = stats[["token", "family", "df_before", "df_after", "kept"]].sort_values(
        ["family", "df_before"], ascending=[True, False]
    ).reset_index(drop=True)
    return stats


def build_basket_stats(
    transactions: pd.DataFrame,
    reduced: pd.DataFrame,
) -> dict[str, Any]:
    """Build a summary dict comparing basket size distribution before/after."""
    def _stats(txn: pd.DataFrame, prefix: str) -> dict[str, Any]:
        if txn.empty:
            return {f"{prefix}_n_baskets": 0, f"{prefix}_avg_items": 0.0,
                    f"{prefix}_median_items": 0.0, f"{prefix}_total_tokens": 0}
        sizes = txn["n_items"] if "n_items" in txn.columns else txn["items"].map(len)
        return {
            f"{prefix}_n_baskets": int(len(txn)),
            f"{prefix}_avg_items": round(float(sizes.mean()), 4),
            f"{prefix}_median_items": round(float(sizes.median()), 4),
            f"{prefix}_total_tokens": int(sizes.sum()),
        }

    before = _stats(transactions, "before")
    after = _stats(reduced, "after")

    reduction_rate = (
        1.0 - after["after_n_baskets"] / before["before_n_baskets"]
        if before["before_n_baskets"] > 0 else 0.0
    )
    token_reduction_rate = (
        1.0 - after["after_total_tokens"] / before["before_total_tokens"]
        if before["before_total_tokens"] > 0 else 0.0
    )

    return {
        **before,
        **after,
        "basket_reduction_rate": round(reduction_rate, 6),
        "token_reduction_rate": round(token_reduction_rate, 6),
    }


# ── internal helpers ──────────────────────────────────────────────────────────

def _build_manifest(
    *,
    token_families: tuple[str, ...] | list[str],
    min_token_df: int,
    top_k_movies: int | None,
    top_k_tags: int | None,
    min_items_per_basket: int,
    max_items_per_basket: int | None,
    sample_frac: float | None,
    random_seed: int,
    n_baskets_in: int,
    n_baskets_out: int,
    n_tokens_in: int,
    n_tokens_out: int,
    vocab_in: int,
    vocab_out: int,
    vocab_removed: list[str],
) -> dict[str, Any]:
    """Build the reduction_manifest dict."""
    import pandas as _pd

    return {
        "generated_at_utc": _pd.Timestamp.utcnow().isoformat(),
        "config": {
            "token_families": list(token_families),
            "min_token_df": min_token_df,
            "top_k_movies": top_k_movies,
            "top_k_tags": top_k_tags,
            "min_items_per_basket": min_items_per_basket,
            "max_items_per_basket": max_items_per_basket,
            "sample_frac": sample_frac,
            "random_seed": random_seed,
        },
        "before": {
            "n_baskets": n_baskets_in,
            "n_total_tokens": n_tokens_in,
            "vocab_size": vocab_in,
        },
        "after": {
            "n_baskets": n_baskets_out,
            "n_total_tokens": n_tokens_out,
            "vocab_size": vocab_out,
        },
        "reduction": {
            "baskets_removed": n_baskets_in - n_baskets_out,
            "basket_reduction_rate": round(
                (n_baskets_in - n_baskets_out) / n_baskets_in, 6
            ) if n_baskets_in > 0 else 0.0,
            "tokens_removed": n_tokens_in - n_tokens_out,
            "token_reduction_rate": round(
                (n_tokens_in - n_tokens_out) / n_tokens_in, 6
            ) if n_tokens_in > 0 else 0.0,
            "vocab_removed": vocab_in - vocab_out,
            "vocab_reduction_rate": round(
                (vocab_in - vocab_out) / vocab_in, 6
            ) if vocab_in > 0 else 0.0,
            "n_vocab_tokens_removed": len(vocab_removed),
            # Only store a sample (up to 200) to keep manifest readable
            "vocab_removed_sample": vocab_removed[:200],
        },
    }
