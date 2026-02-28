"""Story B.1 Explainable Recommender – Streamlit demo app.

Consumes **all three** Story B.1 preset outputs:

- **Preset A** (movie_only): movie→movie association rules — the core
  recommendation engine.
- **Preset B** (controlled_movie_tag): movie↔tag rules — "movie DNA"
  explanations that surface thematic tags.
- **Preset C** (genre_tag): genre↔tag semantic associations — a stable
  backdrop of genre-level patterns.

Directory layout (Feb 2026):

    data/story-module-outputs/story_b/
      preset_A_movie_only_full/<run_id>/tables/association_rules.parquet
      preset_B_controlled_movie_tag_full/<run_id>/tables/association_rules.parquet
      preset_C_genre_tag_full/<run_id>/tables/association_rules.parquet

The notebook generates folder names as ``preset_{PRESET}_{TOKEN_FAMILY_MODE}_{TRANSACTIONS_INPUT}``
(e.g. ``preset_A_movie_only_full`` when TRANSACTIONS_INPUT='full').
The app tries both the ``_full`` and ``_reduced`` suffix variants when
discovering preset directories.
"""

# MovieLens Data Mining
# Nguyen Sy Hung
# 2026

from __future__ import annotations

import ast
import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


APP_NAME = "story_b_explainable_recommender"

# Notebook output pattern: preset_{PRESET}_{TOKEN_FAMILY_MODE}_{TRANSACTIONS_INPUT}
# e.g. preset_A_movie_only_full (when TRANSACTIONS_INPUT='full')
# We list _full first (current default), then _reduced as fallback.
_PRESET_A_CANDIDATES = ["preset_A_movie_only_full", "preset_A_movie_only_reduced", "preset_A_movie_only"]
_PRESET_B_CANDIDATES = ["preset_B_controlled_movie_tag_full", "preset_B_controlled_movie_tag_reduced", "preset_B_controlled_movie_tag"]
_PRESET_C_CANDIDATES = ["preset_C_genre_tag_full", "preset_C_genre_tag_reduced", "preset_C_genre_tag"]


@dataclass(frozen=True)
class AppConfig:
    data_root: Path

    @property
    def preprocessed_tables(self) -> Path:
        return self.data_root / "data" / "preprocessed-data" / "tables"

    @property
    def story_b_root(self) -> Path:
        return self.data_root / "data" / "story-module-outputs" / "story_b"

    @property
    def app_out_root(self) -> Path:
        return self.data_root / "data" / "demo_apps" / APP_NAME


def _load_config() -> AppConfig:
    cfg_path = Path(__file__).resolve().parent / "app_config.json"
    obj = json.loads(cfg_path.read_text(encoding="utf-8"))
    raw_root = obj.get("data_root", "../..")
    data_root = (cfg_path.parent / raw_root).resolve()
    return AppConfig(data_root=data_root)


def _read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing Parquet: {path}")
    return pd.read_parquet(path)


def _discover_runs_for_preset(
    story_b_root: Path,
    preset_candidates: list[str] | str,
) -> tuple[str, list[tuple[str, Path]]]:
    """Return ``(resolved_dir_name, [(run_id, run_dir_path), ...])`` sorted latest-first.

    *preset_candidates* is tried in order; the first existing directory wins.
    """
    if isinstance(preset_candidates, str):
        preset_candidates = [preset_candidates]

    for preset_dir in preset_candidates:
        preset_path = story_b_root / preset_dir
        if not preset_path.is_dir():
            continue

        runs: list[tuple[str, Path]] = []
        for candidate in sorted(preset_path.iterdir(), reverse=True):
            if not candidate.is_dir():
                continue
            rules_file = candidate / "tables" / "association_rules.parquet"
            if rules_file.exists():
                runs.append((candidate.name, candidate))
        if runs:
            return preset_dir, runs

    return preset_candidates[0], []


def _resolve_rules_path(runs: list[tuple[str, Path]], selected_run_id: str) -> tuple[str | None, Path | None]:
    for run_id, run_dir in runs:
        if run_id == selected_run_id:
            return run_id, run_dir / "tables" / "association_rules.parquet"
    return None, None


def _parse_token_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, float) and pd.isna(value):
        return []
    if isinstance(value, (list, tuple, set, frozenset)):
        return [str(v) for v in value]
    if hasattr(value, "tolist") and not isinstance(value, str):
        converted = value.tolist()
        if isinstance(converted, list):
            return [str(v) for v in converted]
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return []
        if raw.startswith(("[", "(", "{")):
            try:
                parsed = ast.literal_eval(raw)
                if isinstance(parsed, (list, tuple, set, frozenset)):
                    return [str(v) for v in parsed]
            except Exception:
                pass
        return [raw]
    return [str(value)]


def _extract_first_movie_token(tokens: list[str]) -> str | None:
    for token in tokens:
        if isinstance(token, str) and token.startswith("movie:"):
            return token
    return None


def _ensure_rules_schema(rules: pd.DataFrame) -> pd.DataFrame:
    df = rules.copy()
    if "antecedent" not in df.columns and "antecedents" in df.columns:
        df["antecedent"] = df["antecedents"]
    if "consequent" not in df.columns and "consequents" in df.columns:
        df["consequent"] = df["consequents"]

    required = ["antecedent", "consequent", "support", "confidence", "lift"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Rules table missing required columns: {missing}")

    df["antecedent_tokens"] = df["antecedent"].map(_parse_token_list)
    df["consequent_tokens"] = df["consequent"].map(_parse_token_list)
    df["rec_movie_token"] = df["consequent_tokens"].map(_extract_first_movie_token)
    for metric in ["support", "confidence", "lift"]:
        df[metric] = pd.to_numeric(df[metric], errors="coerce")
    df = df.dropna(subset=["support", "confidence", "lift"]).copy()
    return df


def _build_movie_lookup(movies: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    if "movieId" not in movies.columns:
        raise ValueError("dim_movies_clean.parquet must include movieId column")

    title_col = "title"
    if title_col not in movies.columns:
        candidates = [col for col in movies.columns if col != "movieId"]
        if not candidates:
            raise ValueError("dim_movies_clean.parquet must include a title-like column")
        title_col = candidates[0]

    df = movies[["movieId", title_col]].copy()
    if "genres" in movies.columns:
        df["genres"] = movies["genres"].astype(str)

    df["movieId"] = pd.to_numeric(df["movieId"], errors="coerce")
    df = df.dropna(subset=["movieId"])
    df["movieId"] = df["movieId"].astype(int)
    df["title"] = df[title_col].astype(str)
    df["movie_token"] = df["movieId"].map(lambda mid: f"movie:{mid}")
    df = df.drop_duplicates(subset=["movie_token"])

    lookup = dict(zip(df["movie_token"], df["title"]))
    return df, lookup


def _decode_token(token: str, movie_lookup: dict[str, str]) -> str:
    if token.startswith("movie:"):
        return movie_lookup.get(token, token)
    if token.startswith("genre:"):
        return token.replace("genre:", "", 1).replace("_", " ").title()
    if token.startswith("tag:"):
        return token.replace("tag:", "", 1)
    return token


def _decode_token_list(tokens: list[str], movie_lookup: dict[str, str]) -> str:
    if not tokens:
        return "[]"
    return " + ".join(_decode_token(token, movie_lookup) for token in tokens)


def _compute_recommendations(
    rules: pd.DataFrame,
    *,
    seed_token: str,
    min_lift: float,
    min_confidence: float,
    top_k: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fired = rules[
        rules["antecedent_tokens"].map(lambda tokens: seed_token in tokens)
        & rules["rec_movie_token"].notna()
        & (rules["rec_movie_token"] != seed_token)
        & (rules["lift"] >= min_lift)
        & (rules["confidence"] >= min_confidence)
    ].copy()

    if fired.empty:
        empty_recs = pd.DataFrame(columns=["rec_movie_token", "lift", "confidence", "support", "evidence_count", "rank"])
        return empty_recs, fired

    recs = (
        fired.groupby("rec_movie_token", as_index=False)
        .agg(
            lift=("lift", "max"),
            confidence=("confidence", "max"),
            support=("support", "max"),
            evidence_count=("rec_movie_token", "size"),
        )
        .sort_values(["lift", "confidence", "support"], ascending=[False, False, False])
        .head(top_k)
        .reset_index(drop=True)
    )
    recs["rank"] = recs.index + 1
    return recs, fired


def _get_movie_tag_dna(
    rules_b: pd.DataFrame | None,
    movie_token: str,
    *,
    top_k: int,
) -> pd.DataFrame:
    """Return top tag tokens associated with a movie via Preset B rules."""
    if rules_b is None or rules_b.empty:
        return pd.DataFrame()

    hits = rules_b[
        rules_b["antecedent_tokens"].map(lambda toks: movie_token in toks)
        | rules_b["consequent_tokens"].map(lambda toks: movie_token in toks)
    ].copy()
    if hits.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for _, row in hits.iterrows():
        tokens = list(row["antecedent_tokens"]) + list(row["consequent_tokens"])
        for tok in tokens:
            if tok.startswith("tag:"):
                rows.append(
                    {
                        "tag_token": tok,
                        "lift": float(row["lift"]),
                        "confidence": float(row["confidence"]),
                        "support": float(row["support"]),
                    }
                )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    return (
        df.groupby("tag_token", as_index=False)
        .agg(lift=("lift", "max"), confidence=("confidence", "max"), support=("support", "max"))
        .sort_values(["lift", "confidence", "support"], ascending=[False, False, False])
        .head(top_k)
        .reset_index(drop=True)
    )


def _get_genre_semantics(
    rules_c: pd.DataFrame | None,
    genre_tokens: list[str],
    *,
    top_k: int,
) -> pd.DataFrame:
    """Return top genre↔tag (or genre↔genre) rules that touch any input genre token."""
    if rules_c is None or rules_c.empty or not genre_tokens:
        return pd.DataFrame()

    genre_set = set(genre_tokens)
    mask = rules_c["antecedent_tokens"].map(lambda toks: bool(genre_set & set(toks))) | rules_c["consequent_tokens"].map(
        lambda toks: bool(genre_set & set(toks))
    )
    hits = rules_c[mask].copy()
    if hits.empty:
        return pd.DataFrame()

    return hits.sort_values(["lift", "confidence", "support"], ascending=[False, False, False]).head(top_k).reset_index(drop=True)


def _make_out_dirs(root: Path) -> dict[str, Path]:
    tables = root / "tables"
    reports = root / "reports"
    figures = root / "figures"
    for p in (tables, reports, figures):
        p.mkdir(parents=True, exist_ok=True)
    return {"root": root, "tables": tables, "reports": reports, "figures": figures}


def _write_manifest(out: dict[str, Path], payload: dict) -> Path:
    payload = dict(payload)
    now_utc = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    payload.setdefault("generated_at_utc", now_utc)
    p = out["reports"] / "run_manifest.json"
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return p


def _write_markdown(out: dict[str, Path], name: str, text: str) -> Path:
    p = out["reports"] / f"{name}.md"
    p.write_text(text, encoding="utf-8")
    return p


def _write_table(out: dict[str, Path], name: str, df: pd.DataFrame) -> Path:
    p = out["tables"] / f"{name}.parquet"
    df.to_parquet(p, index=False)
    return p


@st.cache_data(show_spinner=False)
def _load_movies(path: str) -> pd.DataFrame:
    return _read_parquet(Path(path))


@st.cache_data(show_spinner=False)
def _load_rules(path: str) -> pd.DataFrame:
    raw = _read_parquet(Path(path))
    return _ensure_rules_schema(raw)


def main() -> None:
    st.set_page_config(
        page_title="Story B.1 Explainable Recommender · MovieLens",
        layout="wide",
    )

    st.title("Story B.1 Explainable Recommender")
    st.caption(
        "Because you liked X → here are Y₁…Yₖ, with the rules X→Y along with the rule metrics (support, confidence, lift) that justify each recommendation"
    )

    cfg = _load_config()

    st.sidebar.subheader("Data roots")
    st.sidebar.code(str(cfg.data_root), language="text")
    st.sidebar.code(str(cfg.story_b_root), language="text")

    dir_a, runs_a = _discover_runs_for_preset(cfg.story_b_root, _PRESET_A_CANDIDATES)
    dir_b, runs_b = _discover_runs_for_preset(cfg.story_b_root, _PRESET_B_CANDIDATES)
    dir_c, runs_c = _discover_runs_for_preset(cfg.story_b_root, _PRESET_C_CANDIDATES)

    st.sidebar.subheader("Preset run selectors")

    options_a = [rid for rid, _ in runs_a] or ["(none)"]
    options_b = [rid for rid, _ in runs_b] or ["(none)"]
    options_c = [rid for rid, _ in runs_c] or ["(none)"]

    sel_a = st.sidebar.selectbox("Preset A (movie→movie)", options=options_a, index=0)
    sel_b = st.sidebar.selectbox("Preset B (movie↔tag)", options=options_b, index=0)
    sel_c = st.sidebar.selectbox("Preset C (genre↔tag)", options=options_c, index=0)

    run_a_id, run_a_rules_path = _resolve_rules_path(runs_a, sel_a)
    run_b_id, run_b_rules_path = _resolve_rules_path(runs_b, sel_b)
    run_c_id, run_c_rules_path = _resolve_rules_path(runs_c, sel_c)

    dim_movies_path = cfg.preprocessed_tables / "dim_movies_clean.parquet"

    with st.expander("Expected input files"):
        st.markdown("**Required**")
        st.code(str(dim_movies_path), language="text")
        st.code(str(run_a_rules_path) if run_a_rules_path else "(Preset A missing)", language="text")
        st.markdown("**Optional**")
        st.code(str(run_b_rules_path) if run_b_rules_path else "(Preset B missing)", language="text")
        st.code(str(run_c_rules_path) if run_c_rules_path else "(Preset C missing)", language="text")

    missing_required: list[str] = []
    if not dim_movies_path.exists():
        missing_required.append(str(dim_movies_path))
    if run_a_rules_path is None or not run_a_rules_path.exists():
        missing_required.append(str(run_a_rules_path) if run_a_rules_path else f"No Preset A runs under {cfg.story_b_root / dir_a}")

    if missing_required:
        st.error("Missing required input artifacts:")
        for p in missing_required:
            st.code(p, language="text")
        st.info("Generate Story B.1 Preset A (RUN_PRESET='A') outputs first, then rerun this app.")
        return

    assert run_a_rules_path is not None

    try:
        movies_raw = _load_movies(str(dim_movies_path))
        movies_df, movie_lookup = _build_movie_lookup(movies_raw)
        rules_a = _load_rules(str(run_a_rules_path))
    except Exception as exc:
        st.error(f"Failed to load required data: {exc}")
        return

    rules_b: pd.DataFrame | None = None
    rules_c: pd.DataFrame | None = None
    preset_b_status: str
    preset_c_status: str

    # Preset B (optional)
    if run_b_rules_path is None:
        preset_b_status = "missing (no runs discovered)"
    elif not run_b_rules_path.exists():
        preset_b_status = "missing file"
    else:
        try:
            rules_b_loaded = _load_rules(str(run_b_rules_path))
            preset_b_status = f"`{run_b_id or '—'}`: {len(rules_b_loaded):,} rules" + (
                " (file present, 0 rules)" if rules_b_loaded.empty else ""
            )
            if not rules_b_loaded.empty:
                rules_b = rules_b_loaded
        except Exception as exc:
            preset_b_status = f"`{run_b_id or '—'}`: failed to load ({type(exc).__name__})"

    # Preset C (optional)
    if run_c_rules_path is None:
        preset_c_status = "missing (no runs discovered)"
    elif not run_c_rules_path.exists():
        preset_c_status = "missing file"
    else:
        try:
            rules_c_loaded = _load_rules(str(run_c_rules_path))
            preset_c_status = f"`{run_c_id or '—'}`: {len(rules_c_loaded):,} rules" + (
                " (file present, 0 rules)" if rules_c_loaded.empty else ""
            )
            if not rules_c_loaded.empty:
                rules_c = rules_c_loaded
        except Exception as exc:
            preset_c_status = f"`{run_c_id or '—'}`: failed to load ({type(exc).__name__})"

    covered_seed_tokens = {
        tok
        for toks in rules_a["antecedent_tokens"]
        for tok in toks
        if isinstance(tok, str) and tok.startswith("movie:")
    }

    st.sidebar.subheader("Data status")
    st.sidebar.write(f"- dim_movies: {len(movies_df):,} rows")
    st.sidebar.write(f"- Preset A `{run_a_id}`: {len(rules_a):,} rules")
    st.sidebar.write(f"- Preset B: {preset_b_status}")
    st.sidebar.write(f"- Preset C: {preset_c_status}")
    n_seed_covered = len(covered_seed_tokens)
    n_movies_total = len(movies_df)
    seed_coverage_pct = (100.0 * n_seed_covered / n_movies_total) if n_movies_total else 0.0
    st.sidebar.write(f"- Seed coverage (Preset A antecedents): {n_seed_covered:,}/{n_movies_total:,} ({seed_coverage_pct:.2f}%)")

    with st.sidebar:
        st.subheader("Ranking controls")
        min_lift = st.slider("Min lift", min_value=1.0, max_value=5.0, value=1.1, step=0.1)
        min_conf = st.slider("Min confidence", min_value=0.0, max_value=1.0, value=0.4, step=0.01)
        top_k = st.slider("Top-k", min_value=3, max_value=30, value=10, step=1)
        max_evidence_per_rec = st.slider("Evidence rules per recommendation", min_value=1, max_value=5, value=3, step=1)
        top_tags_per_movie = st.slider("Preset B: tags per movie", min_value=0, max_value=10, value=5, step=1)
        top_genre_rules = st.slider("Preset C: rules shown", min_value=0, max_value=30, value=10, step=1)

    all_options = movies_df.sort_values("title")["movie_token"].tolist()
    covered_options = movies_df[movies_df["movie_token"].isin(covered_seed_tokens)].sort_values("title")["movie_token"].tolist()

    if not all_options:
        st.error("No movie options available from dim_movies_clean.parquet")
        return

    seed_mode = st.radio(
        "Seed selection",
        options=[
            "Covered movies only (guaranteed recs)",
            "All movies (may have no recs)",
        ],
        index=0,
    )
    options = covered_options if seed_mode.startswith("Covered") else all_options

    if not options:
        st.error("No seed movies available for the current selection mode")
        return

    preferred_default_seeds = ["movie:318", "movie:296", "movie:356", "movie:593"]
    default_seed = next((tok for tok in preferred_default_seeds if tok in set(options)), options[0])

    if "seed_token" not in st.session_state or st.session_state["seed_token"] not in set(options):
        st.session_state["seed_token"] = default_seed

    seed_token = st.selectbox(
        "Seed movie (X)",
        options=options,
        index=options.index(st.session_state["seed_token"]),
        format_func=lambda tok: f"{movie_lookup.get(tok, tok)} [{tok}]",
    )
    st.session_state["seed_token"] = seed_token
    seed_title = movie_lookup.get(seed_token, seed_token)

    # Seed genres for preset C
    seed_row = movies_df[movies_df["movie_token"] == seed_token]
    seed_genres: list[str] = []
    if not seed_row.empty and "genres" in seed_row.columns:
        raw_genres = seed_row.iloc[0].get("genres", "")
        if pd.notna(raw_genres) and str(raw_genres).strip():
            seed_genres = [
                f"genre:{g.strip().lower().replace(' ', '_').replace('-', '_')}"
                for g in str(raw_genres).split("|")
                if g.strip() and g.strip() != "(no genres listed)"
            ]

    recs, fired_rules = _compute_recommendations(
        rules_a,
        seed_token=seed_token,
        min_lift=min_lift,
        min_confidence=min_conf,
        top_k=top_k,
    )

    st.markdown(f"### Because you liked **{seed_title}**")

    # Seed DNA panel (Preset B)
    if top_tags_per_movie > 0:
        seed_dna = _get_movie_tag_dna(rules_b, seed_token, top_k=top_tags_per_movie)
        if not seed_dna.empty:
            st.markdown("**Preset B (movie↔tag) — seed movie DNA**")
            st.dataframe(
                seed_dna.assign(tag=lambda df: df["tag_token"].map(lambda t: _decode_token(t, movie_lookup)))[
                    ["tag", "tag_token", "lift", "confidence", "support"]
                ],
                width="stretch",
            )

    if recs.empty:
        if seed_token not in covered_seed_tokens:
            st.warning(
                "This seed movie is outside Preset A rule coverage, so no recommendations can fire for it. "
                "Try switching to 'Covered movies only' or pick a different seed."
            )
        else:
            st.warning("No recommendations matched the current filters. Try lowering min lift/confidence.")
        return

    recs = recs.copy()
    recs["rec_title"] = recs["rec_movie_token"].map(lambda tok: movie_lookup.get(tok, tok))

    display_df = recs[["rank", "rec_title", "lift", "confidence", "support", "evidence_count"]].rename(
        columns={
            "rec_title": "recommended_movie",
            "evidence_count": "rules_supporting_this_recommendation",
        }
    )
    st.dataframe(display_df, width="stretch")

    pivot_token = st.selectbox(
        "Pivot to recommendation",
        options=recs["rec_movie_token"].tolist(),
        format_func=lambda tok: movie_lookup.get(tok, tok),
    )
    if st.button("Use selected recommendation as new seed"):
        st.session_state["seed_token"] = pivot_token
        st.rerun()

    evidence_rows: list[dict[str, Any]] = []
    for rec_token in recs["rec_movie_token"].tolist():
        subset = fired_rules[fired_rules["rec_movie_token"] == rec_token].copy()
        subset = subset.sort_values(["lift", "confidence", "support"], ascending=[False, False, False]).head(max_evidence_per_rec)
        rec_title = movie_lookup.get(rec_token, rec_token)

        with st.expander(f"Evidence: Because you liked {seed_title} → {rec_title}"):
            st.markdown("**Preset A evidence (movie→movie)**")
            for _, row in subset.iterrows():
                ante_tokens = row["antecedent_tokens"]
                cons_tokens = row["consequent_tokens"]
                antecedent_text = _decode_token_list(ante_tokens, movie_lookup)
                consequent_text = _decode_token_list(cons_tokens, movie_lookup)
                st.markdown(f"**Rule:** {antecedent_text} → {consequent_text}")
                c1, c2, c3 = st.columns(3)
                c1.metric("Support", f"{row['support']:.4f}")
                c2.metric("Confidence", f"{row['confidence']:.4f}")
                c3.metric("Lift", f"{row['lift']:.4f}")

                evidence_rows.append(
                    {
                        "seed_movie_token": seed_token,
                        "seed_title": seed_title,
                        "rec_movie_token": rec_token,
                        "rec_title": rec_title,
                        "antecedent": ante_tokens,
                        "consequent": cons_tokens,
                        "support": float(row["support"]),
                        "confidence": float(row["confidence"]),
                        "lift": float(row["lift"]),
                    }
                )

            if top_tags_per_movie > 0:
                dna = _get_movie_tag_dna(rules_b, rec_token, top_k=top_tags_per_movie)
                if not dna.empty:
                    st.markdown("**Preset B enrichment (movie↔tag DNA)**")
                    for _, r in dna.iterrows():
                        st.markdown(
                            f"- {_decode_token(r['tag_token'], movie_lookup)} "
                            f"(`{r['tag_token']}`) — lift {r['lift']:.2f}, conf {r['confidence']:.2f}"
                        )

    if top_genre_rules > 0 and rules_c is not None and seed_genres:
        st.markdown("---")
        st.markdown("### Preset C (genre↔tag) — genre semantics")
        st.caption("Seed genres: " + ", ".join(f"`{g}`" for g in seed_genres))

        preset_c_genre_vocab = {
            tok
            for col in ["antecedent_tokens", "consequent_tokens"]
            for toks in rules_c[col]
            for tok in toks
            if isinstance(tok, str) and tok.startswith("genre:")
        }
        covered_seed_genres = [g for g in seed_genres if g in preset_c_genre_vocab]
        st.caption(f"Preset C genre coverage for this seed: {len(covered_seed_genres)}/{len(seed_genres)}")

        genre_hits = _get_genre_semantics(rules_c, seed_genres, top_k=top_genre_rules)
        if genre_hits.empty:
            st.info("No Preset C rules matched the seed genres. Showing global Preset C semantic fallback.")
            genre_hits = rules_c.sort_values(["lift", "confidence", "support"], ascending=[False, False, False]).head(top_genre_rules).reset_index(drop=True)
            if genre_hits.empty:
                st.info("Preset C has no available rules in the selected run.")
            else:
                show = pd.DataFrame(
                    {
                        "antecedent": genre_hits["antecedent_tokens"].map(lambda toks: _decode_token_list(toks, movie_lookup)),
                        "consequent": genre_hits["consequent_tokens"].map(lambda toks: _decode_token_list(toks, movie_lookup)),
                        "support": genre_hits["support"],
                        "confidence": genre_hits["confidence"],
                        "lift": genre_hits["lift"],
                    }
                )
                st.dataframe(show, width="stretch")
        else:
            show = pd.DataFrame(
                {
                    "antecedent": genre_hits["antecedent_tokens"].map(lambda toks: _decode_token_list(toks, movie_lookup)),
                    "consequent": genre_hits["consequent_tokens"].map(lambda toks: _decode_token_list(toks, movie_lookup)),
                    "support": genre_hits["support"],
                    "confidence": genre_hits["confidence"],
                    "lift": genre_hits["lift"],
                }
            )
            st.dataframe(show, width="stretch")

    evidence_df = pd.DataFrame(
        evidence_rows,
        columns=[
            "seed_movie_token",
            "seed_title",
            "rec_movie_token",
            "rec_title",
            "antecedent",
            "consequent",
            "support",
            "confidence",
            "lift",
        ],
    )

    st.markdown("---")
    if st.button("Export current evidence artifacts"):
        out = _make_out_dirs(cfg.app_out_root)
        _write_manifest(
            out,
            {
                "app": APP_NAME,
                "caption": "Because you liked X → here are Y₁…Yₖ, with the rules X→Y along with the rule metrics (support, confidence, lift) that justify each recommendation",
                "inputs": {
                    "dim_movies": str(dim_movies_path),
                    "preset_a": {"run_id": run_a_id, "rules": str(run_a_rules_path)},
                    "preset_b": {"run_id": run_b_id, "rules": str(run_b_rules_path) if run_b_rules_path else None},
                    "preset_c": {"run_id": run_c_id, "rules": str(run_c_rules_path) if run_c_rules_path else None},
                },
                "parameters": {
                    "seed_token": seed_token,
                    "seed_genres": seed_genres,
                    "min_lift": min_lift,
                    "min_confidence": min_conf,
                    "top_k": top_k,
                    "max_evidence_per_recommendation": max_evidence_per_rec,
                    "preset_b_top_tags": top_tags_per_movie,
                    "preset_c_top_rules": top_genre_rules,
                },
                "outputs": {
                    "recommendations_count": int(len(recs)),
                    "evidence_rows": int(len(evidence_df)),
                },
            },
        )
        _write_table(out, "rule_evidence", evidence_df)
        _write_markdown(
            out,
            "summary",
            "\n".join(
                [
                    "# Story B.1 Explainable Recommender — Export Summary",
                    f"- Seed movie: {seed_title} ({seed_token})",
                    f"- Recommendations shown: {len(recs)}",
                    f"- Rule evidence rows exported: {len(evidence_df)}",
                    f"- Preset A run_id: {run_a_id}",
                    f"- Preset B run_id: {run_b_id}",
                    f"- Preset C run_id: {run_c_id}",
                ]
            ),
        )
        st.success(f"Wrote artifacts to: {out['root']}")


if __name__ == "__main__":
    main()
