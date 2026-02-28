# MovieLens Data Mining
#
# Nguyen Sy Hung
# 2026

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd


def _collect_validation_warnings(validation_results: Dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    for section, payload in validation_results.items():
        if isinstance(payload, dict) and payload.get("warning"):
            warnings.append(f"{section}: {payload['warning']}")
    return warnings


def _safe_get(dct: Dict[str, Any], keys: list[str], default: Any = None) -> Any:
    cursor: Any = dct
    for key in keys:
        if not isinstance(cursor, dict) or key not in cursor:
            return default
        cursor = cursor[key]
    return cursor


def build_audit_report(
    run_id: str,
    schema_version: str,
    clean_counts: Dict[str, int],
    validation_results: Dict[str, Any],
    split_validation: Dict[str, Any],
    phase4_report: Dict[str, Any],
) -> Dict[str, Any]:
    """Build Phase 5 audit report from prior phase outputs.

    The report is intentionally compact and focused on run health, leakage safety,
    and downstream readiness.
    """
    ratings_join_coverage = float(_safe_get(validation_results, ["ratings_movie_join", "coverage"], 0.0))
    tags_join_coverage = float(_safe_get(validation_results, ["tags_movie_join", "coverage"], 0.0))

    temporal_ok = bool(_safe_get(split_validation, ["checks", "temporal_ordering", "passed"], False))
    overlap_ok = bool(_safe_get(split_validation, ["checks", "overlap", "passed"], False))

    train_rows = int(_safe_get(phase4_report, ["coverage", "train_rows"], 0))
    train_users = int(_safe_get(phase4_report, ["coverage", "train_unique_users"], 0))
    train_movies = int(_safe_get(phase4_report, ["coverage", "train_unique_movies"], 0))
    user_feature_rows = int(_safe_get(phase4_report, ["coverage", "user_features_rows"], 0))
    movie_feature_rows = int(_safe_get(phase4_report, ["coverage", "movie_features_rows"], 0))
    transactions_rows = int(_safe_get(phase4_report, ["coverage", "transactions_rows"], 0))

    user_feature_coverage = float(_safe_get(phase4_report, ["coverage", "user_feature_coverage"], 0.0))
    movie_feature_coverage = float(_safe_get(phase4_report, ["coverage", "movie_feature_coverage"], 0.0))

    warnings: list[str] = []
    warnings.extend(_collect_validation_warnings(validation_results))
    warnings.extend(split_validation.get("warnings", []))

    data_integrity_ok = (
        int(_safe_get(validation_results, ["ratings", "null_users"], 0)) == 0
        and int(_safe_get(validation_results, ["ratings", "null_movies"], 0)) == 0
        and int(_safe_get(validation_results, ["movies", "duplicates", "n_duplicates"], 0)) == 0
        and int(_safe_get(validation_results, ["links", "duplicates", "n_duplicates"], 0)) == 0
        and ratings_join_coverage >= 0.95
        and tags_join_coverage >= 0.95
    )

    split_sanity_ok = temporal_ok and overlap_ok
    feature_coverage_ok = (
        user_feature_coverage >= 0.99
        and movie_feature_coverage >= 0.99
        and user_feature_rows > 0
        and movie_feature_rows > 0
    )
    method_readiness_ok = transactions_rows > 0 and train_rows > 0

    gate_status = {
        "data_integrity_ok": data_integrity_ok,
        "split_sanity_ok": split_sanity_ok,
        "feature_coverage_ok": feature_coverage_ok,
        "method_readiness_ok": method_readiness_ok,
    }
    passed_all = all(gate_status.values())

    report: Dict[str, Any] = {
        "run_id": run_id,
        "phase": "audit",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "schema_version": schema_version,
        "summary": {
            "passed_all_gates": passed_all,
            "warning_count": len(warnings),
        },
        "gates": gate_status,
        "counts": {
            "clean_tables": clean_counts,
            "split": {
                "train_rows": train_rows,
                "test_rows": int(_safe_get(split_validation, ["stats", "sizes", "test_rows"], 0)),
                "total_rows": int(_safe_get(split_validation, ["stats", "sizes", "total_rows"], 0)),
            },
            "features": {
                "train_unique_users": train_users,
                "train_unique_movies": train_movies,
                "user_features_rows": user_feature_rows,
                "movie_features_rows": movie_feature_rows,
                "transactions_rows": transactions_rows,
            },
        },
        "join_coverage": {
            "ratings_to_movies": ratings_join_coverage,
            "tags_to_movies": tags_join_coverage,
        },
        "missingness": {
            "ratings": {
                "null_users": int(_safe_get(validation_results, ["ratings", "null_users"], 0)),
                "null_movies": int(_safe_get(validation_results, ["ratings", "null_movies"], 0)),
                "null_ratings": int(_safe_get(validation_results, ["ratings", "null_ratings"], 0)),
            },
            "movies": {
                "year_null": int(_safe_get(validation_results, ["movies", "year_null"], 0)),
                "no_genres": int(_safe_get(validation_results, ["movies", "no_genres"], 0)),
            },
            "links": {
                "imdbId_missing": int(_safe_get(validation_results, ["links", "imdbId_missing"], 0)),
                "tmdbId_missing": int(_safe_get(validation_results, ["links", "tmdbId_missing"], 0)),
            },
        },
        "split_sanity": {
            "temporal_ordering": _safe_get(split_validation, ["checks", "temporal_ordering"], {}),
            "overlap": _safe_get(split_validation, ["checks", "overlap"], {}),
            "cold_start": _safe_get(split_validation, ["stats", "cold_start"], {}),
            "distribution": _safe_get(split_validation, ["stats", "distribution"], {}),
        },
        "feature_coverage": {
            "user_feature_coverage": user_feature_coverage,
            "movie_feature_coverage": movie_feature_coverage,
        },
        "warnings": warnings,
    }
    return report


def write_audit_report(report: Dict[str, Any], path: str | Path) -> None:
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

    path_obj.write_text(json.dumps(report, indent=2, default=_json_default), encoding="utf-8")
