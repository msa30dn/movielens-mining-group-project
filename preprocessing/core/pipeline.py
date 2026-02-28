# MovieLens Data Mining
#
# Nguyen Sy Hung
# 2026

from __future__ import annotations

from pathlib import Path
import logging
import time
import json
from typing import Any, Dict

from .config import PipelineConfig
from . import ingest
from . import clean
from . import validate
from . import split
from . import features_user
from . import features_movie
from . import transactions
from . import reduce
from . import audit
from .metadata_raw import build_raw_metadata, get_git_commit_hash, write_metadata_raw
from .metadata_clean import build_clean_metadata, write_metadata_clean
from .paths import make_run_paths


def _configure_logger(log_file: Path, name: str = "backbone") -> logging.Logger:
    """Configure file and console logging.
    
    Args:
        log_file: Path to log file
        name: Logger name
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    
    return logger


def run_pipeline(
    out_root: str, 
    raw_staging_dir: str = "data/raw_staging",
    run_tag: str | None = None, 
    schema_version: str = "0.1.0",
    split_policy: str = "global",
    test_frac: float = 0.1,
    cutoff_date: str | None = None,
) -> Path:
    """Run processing pipeline: Phase 2 (Clean) + future phases.
    
    NOTE: This assumes raw staging Parquet files already exist.
    Run `scripts/build_raw_staging.py` first if they don't.
    
    Phase 2: Clean, normalize, validate, and write contract tables
    Future: Phase 3 (Split), Phase 4 (Features), etc.
    
    Args:
        out_root: Path to processed output root
        raw_staging_dir: Path to raw staging Parquet directory (default: data/raw_staging)
        run_tag: Optional custom run identifier (default: auto-timestamp)
        schema_version: Schema version string
    Returns:
        Path to the run directory containing tables/ and reports/
        
    Raises:
        FileNotFoundError: If raw_staging_dir or required Parquet files don't exist
        ValueError: If Parquet schema validation fails
        validate.ValidationError: If data validation fails
        Exception: For other errors during pipeline execution
    """
    start_time = time.time()
    logger = None  # Initialize logger variable
    
    try:
        # Setup paths and logging (use dummy raw_dir since we're using staging)
        staging_path = Path(raw_staging_dir)
        run_paths = make_run_paths(
            raw_dir=staging_path.parent / "raw" / "ml-latest",  # Dummy, not used
            out_root=out_root, 
            run_tag=run_tag,
            validate_raw=False,  # Don't validate raw_dir since we use staging
            raw_staging_root=staging_path
        )
        logger = _configure_logger(run_paths.reports_dir / "build.log")
        
        config = PipelineConfig(
            raw_dir=staging_path,  # Point to staging instead
            out_root=Path(out_root),
            run_tag=run_tag,
            schema_version=schema_version,
            split_policy=split_policy,
            test_frac=test_frac,
            cutoff_date=cutoff_date,
        )

        logger.info("="*60)
        logger.info("PROCESSING PIPELINE (Phase 2+)")
        logger.info("="*60)
        logger.info("Run ID: %s", run_paths.run_dir.name)
        logger.info("Raw staging: %s", staging_path)
        logger.info("Output directory: %s", run_paths.run_dir)
        logger.info("Schema version: %s", config.schema_version)
        logger.info("Test fraction: %.1f%%", config.test_frac * 100)
        logger.info("Like threshold: %.1f", config.like_threshold)
        logger.info("Reduction token families: %s", list(config.reduction_token_families))
        logger.info("Reduction min_token_df: %d", config.reduction_min_token_df)
        logger.info("Reduction top_k_movies: %s", config.reduction_top_k_movies)
        logger.info("Reduction top_k_tags:   %s", config.reduction_top_k_tags)
        logger.info("")

        # Validate raw staging directory exists
        if not staging_path.exists():
            raise FileNotFoundError(
                f"Raw staging directory not found: {staging_path}\n"
                f"Run 'python scripts/build_raw_staging.py' first to create staging files."
            )

        # Validate required staging files exist
        required_files = [
            "movies_raw.parquet",
            "ratings_raw.parquet",
            "tags_raw.parquet",
            "links_raw.parquet",
        ]
        missing_files = [f for f in required_files if not (staging_path / f).exists()]
        if missing_files:
            raise FileNotFoundError(
                f"Missing required staging files: {missing_files}\n"
                f"Run 'python scripts/build_raw_staging.py' first."
            )

        # ================================================================
        # PHASE 2: CLEAN + NORMALIZE + VALIDATE
        # ================================================================
        logger.info("="*60)
        logger.info("PHASE 2: CLEAN + NORMALIZE + VALIDATE")
        logger.info("="*60)
        
        phase2_start = time.time()
        logger.info("Loading raw staging Parquet files from %s...", staging_path)
        import pandas as pd
        movies_raw = pd.read_parquet(staging_path / "movies_raw.parquet")
        ratings_raw = pd.read_parquet(staging_path / "ratings_raw.parquet")
        tags_raw = pd.read_parquet(staging_path / "tags_raw.parquet")
        links_raw = pd.read_parquet(staging_path / "links_raw.parquet")
        
        genome_tags_path = staging_path / "genome_tags_raw.parquet"
        genome_tags_raw = pd.read_parquet(genome_tags_path) if genome_tags_path.exists() else None
        
        genome_scores_path = staging_path / "genome_scores_raw.parquet"
        genome_scores_raw = pd.read_parquet(genome_scores_path) if genome_scores_path.exists() else None
        
        # Store raw counts for metadata
        raw_counts = {
            'movies': len(movies_raw),
            'ratings': len(ratings_raw),
            'tags': len(tags_raw),
            'links': len(links_raw),
        }
        if genome_tags_raw is not None:
            raw_counts['genome_tags'] = len(genome_tags_raw)
        if genome_scores_raw is not None:
            raw_counts['genome_scores'] = len(genome_scores_raw)
        
        # Clean tables
        logger.info("Cleaning tables...")
        logger.info("  Cleaning movies...")
        movies_clean = clean.clean_movies(movies_raw)
        
        logger.info("  Cleaning ratings...")
        ratings_clean = clean.clean_ratings(ratings_raw)
        
        logger.info("  Cleaning tags...")
        tags_clean = clean.clean_tags(tags_raw)
        
        logger.info("  Cleaning links...")
        links_clean = clean.clean_links(links_raw)
        
        if genome_tags_raw is not None:
            logger.info("  Cleaning genome_tags...")
            genome_tags_clean = clean.clean_genome_tags(genome_tags_raw)
        else:
            genome_tags_clean = None
            
        if genome_scores_raw is not None:
            logger.info("  Cleaning genome_scores...")
            genome_scores_clean = clean.clean_genome_scores(genome_scores_raw)
        else:
            genome_scores_clean = None
        
        # Free raw data memory
        del movies_raw, ratings_raw, tags_raw, links_raw, genome_tags_raw, genome_scores_raw
        
        # Store clean counts for metadata
        clean_counts = {
            'movies': len(movies_clean),
            'ratings': len(ratings_clean),
            'tags': len(tags_clean),
            'links': len(links_clean),
        }
        if genome_tags_clean is not None:
            clean_counts['genome_tags'] = len(genome_tags_clean)
        if genome_scores_clean is not None:
            clean_counts['genome_scores'] = len(genome_scores_clean)
        
        logger.info("Clean table counts:")
        logger.info("  movies_clean:        %12s rows", f"{clean_counts['movies']:,}")
        logger.info("  ratings_clean:       %12s rows", f"{clean_counts['ratings']:,}")
        logger.info("  tags_clean:          %12s rows", f"{clean_counts['tags']:,}")
        logger.info("  links_clean:         %12s rows", f"{clean_counts['links']:,}")
        if genome_tags_clean is not None:
            logger.info("  genome_tags_clean:   %12s rows", f"{clean_counts['genome_tags']:,}")
        if genome_scores_clean is not None:
            logger.info("  genome_scores_clean: %12s rows", f"{clean_counts['genome_scores']:,}")
        
        # Validate tables
        logger.info("Validating tables...")
        validation_results: Dict[str, Any] = {}
        
        try:
            logger.info("  Validating movies...")
            validation_results['movies'] = validate.validate_movies(movies_clean)
            
            logger.info("  Validating ratings...")
            validation_results['ratings'] = validate.validate_rating_domain(ratings_clean)
            
            logger.info("  Validating tags...")
            validation_results['tags'] = validate.validate_tags(tags_clean)
            
            logger.info("  Validating links...")
            validation_results['links'] = validate.validate_links(links_clean)
            
            # Validate join coverage: ratings → movies
            logger.info("  Validating ratings → movies join coverage...")
            validation_results['ratings_movie_join'] = validate.validate_join_coverage(
                fact_df=ratings_clean,
                dim_df=movies_clean,
                fact_name='ratings',
                fact_key='movieId',
                dim_key='movieId',
                threshold=0.95
            )
            
            # Validate join coverage: tags → movies
            logger.info("  Validating tags → movies join coverage...")
            validation_results['tags_movie_join'] = validate.validate_join_coverage(
                fact_df=tags_clean,
                dim_df=movies_clean,
                fact_name='tags',
                fact_key='movieId',
                dim_key='movieId',
                threshold=0.95
            )
            
            logger.info("✓ All validations passed")
            
        except validate.ValidationError as e:
            logger.error("✗ Validation failed: %s", e)
            raise
        
        # Write clean tables to parquet
        logger.info("Writing clean tables to parquet...")
        movies_clean.to_parquet(run_paths.tables_dir / "dim_movies_clean.parquet", index=False)
        ratings_clean.to_parquet(run_paths.tables_dir / "fact_ratings_clean.parquet", index=False)
        tags_clean.to_parquet(run_paths.tables_dir / "dim_tags_clean.parquet", index=False)
        links_clean.to_parquet(run_paths.tables_dir / "dim_links_clean.parquet", index=False)
        
        if genome_tags_clean is not None:
            genome_tags_clean.to_parquet(run_paths.tables_dir / "dim_genome_tags_clean.parquet", index=False)
        if genome_scores_clean is not None:
            genome_scores_clean.to_parquet(run_paths.tables_dir / "fact_genome_scores_clean.parquet", index=False)
        
        logger.info("✓ Clean tables written to %s", run_paths.tables_dir)
        
        # Free clean data memory (keep ratings_clean for Phase 3)
        del movies_clean, tags_clean, links_clean
        if genome_tags_clean is not None:
            del genome_tags_clean
        if genome_scores_clean is not None:
            del genome_scores_clean
        
        # Generate and write metadata_clean.json
        logger.info("Generating metadata_clean.json...")
        metadata_clean = build_clean_metadata(
            run_id=run_paths.run_dir.name,
            raw_counts=raw_counts,
            clean_counts=clean_counts,
            validation_results=validation_results,
        )
        
        metadata_clean_path = run_paths.reports_dir / "metadata_clean.json"
        write_metadata_clean(metadata_clean, metadata_clean_path)
        logger.info("✓ metadata_clean.json written")
        
        phase2_elapsed = time.time() - phase2_start
        logger.info("="*60)
        logger.info("PHASE 2 COMPLETED in %.2f seconds", phase2_elapsed)
        logger.info("="*60)
        logger.info("")

        # ================================================================
        # PHASE 3: TEMPORAL SPLIT (TRAIN/TEST)
        # ================================================================
        phase3_start = time.time()
        logger.info("="*60)
        logger.info("PHASE 3: TEMPORAL SPLIT")
        logger.info("="*60)
        logger.info("Split policy: %s", config.split_policy)
        logger.info("Test fraction: %.1f%%", config.test_frac * 100)
        if config.cutoff_date and config.split_policy == "global":
            logger.info("Explicit cutoff: %s", config.cutoff_date)
        logger.info("")

        # Perform temporal split on ratings_clean
        user_cutoffs = None
        if config.split_policy == "global":
            logger.info("Splitting ratings_clean by global temporal cutoff...")
            train_df, test_df, cutoff_datetime = split.global_time_split(
                ratings_clean,
                cutoff_date=config.cutoff_date,
                test_fraction=config.test_frac,
            )
            logger.info("  Train: %d rows (before %s)", len(train_df), cutoff_datetime)
            logger.info("  Test:  %d rows (at/after %s)", len(test_df), cutoff_datetime)
        else:
            logger.info("Splitting ratings_clean per-user by temporal cutoff...")
            train_df, test_df, user_cutoffs = split.per_user_time_split(
                ratings_clean,
                test_fraction=config.test_frac,
            )
            cutoff_datetime = None
            logger.info("  Train: %d rows", len(train_df))
            logger.info("  Test:  %d rows", len(test_df))
            logger.info(
                "  Users with test rows: %d",
                int(user_cutoffs["userId"].nunique()) if user_cutoffs is not None else 0,
            )
        logger.info("")

        # Validate split (hard gates + soft warnings)
        logger.info("Validating temporal split...")
        validation = split.validate_split(
            train_df,
            test_df,
            cutoff_datetime,
            split_policy=config.split_policy,
        )
        
        # Log validation results
        checks = validation["checks"]
        logger.info("  Temporal ordering: %s", "✓" if checks["temporal_ordering"]["passed"] else "✗ FAILED")
        logger.info("  Overlap check: %s", "✓" if checks["overlap"]["passed"] else "✗ FAILED")
        logger.info("  Cold-start users: %d (%.1f%%)", 
               validation["stats"]["cold_start"]["users"]["cold_start_count"], 
               validation["stats"]["cold_start"]["users"]["cold_start_rate"] * 100)
        logger.info("  Cold-start items: %d (%.1f%%)", 
               validation["stats"]["cold_start"]["movies"]["cold_start_count"],
               validation["stats"]["cold_start"]["movies"]["cold_start_rate"] * 100)
        logger.info("  Rating distribution KL divergence: %.4f", validation["stats"]["distribution"]["rating_kl_divergence"])
        
        # Check warnings
        if validation["warnings"]:
            logger.warning("Split validation warnings:")
            for warning in validation["warnings"]:
                logger.warning("  - %s", warning)
        logger.info("")

        # Write train/test splits
        train_path = run_paths.tables_dir / "interactions_train.parquet"
        test_path = run_paths.tables_dir / "interactions_test.parquet"
        train_df.to_parquet(train_path, index=False)
        test_df.to_parquet(test_path, index=False)
        logger.info("✓ interactions_train.parquet written (%d rows)", len(train_df))
        logger.info("✓ interactions_test.parquet written (%d rows)", len(test_df))
        logger.info("")

        # Build and write split metadata report
        split_report = split.build_split_report(
            validation_results=validation,
            cutoff_date=cutoff_datetime,
            config=config,
            run_id=run_paths.run_dir.name,
        )
        split_report_path = run_paths.reports_dir / "split_report.json"
        split.write_split_report(split_report, split_report_path)
        logger.info("✓ split_report.json written")
        logger.info("")

        phase3_elapsed = time.time() - phase3_start
        logger.info("="*60)
        logger.info("PHASE 3 COMPLETED in %.2f seconds", phase3_elapsed)
        logger.info("="*60)
        logger.info("")

        # ================================================================
        # PHASE 4: FEATURE FACTORY + TRANSACTIONS VIEW
        # ================================================================
        phase4_start = time.time()
        logger.info("="*60)
        logger.info("PHASE 4: FEATURE FACTORY + TRANSACTIONS")
        logger.info("="*60)

        import pandas as pd

        movies_for_features = pd.read_parquet(run_paths.tables_dir / "dim_movies_clean.parquet")
        tags_for_features = pd.read_parquet(run_paths.tables_dir / "dim_tags_clean.parquet")
        if config.split_policy == "global":
            tags_train = tags_for_features[tags_for_features["tag_dt_utc"] < cutoff_datetime].copy()
        else:
            # Leakage-safe tag filtering for per-user splits.
            # Users without test rows are train-only → keep all their tags.
            cutoffs = user_cutoffs.copy() if user_cutoffs is not None else pd.DataFrame(
                columns=["userId", "user_cutoff_dt_utc"]
            )
            tags_train = tags_for_features.merge(cutoffs, on="userId", how="left")
            tags_train = tags_train[
                (tags_train["user_cutoff_dt_utc"].isna())
                | (tags_train["tag_dt_utc"] < tags_train["user_cutoff_dt_utc"])
            ].copy()
            tags_train = tags_train.drop(columns=["user_cutoff_dt_utc"])

        logger.info("Building user_features_train...")
        user_features_train = features_user.build_user_features_train(
            interactions_train=train_df,
            movies_clean=movies_for_features,
            tags_train=tags_train,
        )

        logger.info("Building movie_features_train...")
        movie_features_train = features_movie.build_movie_features_train(
            interactions_train=train_df,
            movies_clean=movies_for_features,
            tags_train=tags_train,
            min_tag_freq=config.min_tag_freq,
        )

        logger.info("Building transactions_train...")
        transactions_train = transactions.build_transactions_train(
            interactions_train=train_df,
            movies_clean=movies_for_features,
            tags_train=tags_train,
            like_threshold=config.like_threshold,
            min_tag_freq=config.min_tag_freq,
            include_movie_tokens=True,
            include_genre_tokens=True,
            include_tag_tokens=True,
        )

        user_features_path = run_paths.tables_dir / "user_features_train.parquet"
        movie_features_path = run_paths.tables_dir / "movie_features_train.parquet"
        transactions_path = run_paths.tables_dir / "transactions_train.parquet"

        user_features_train.to_parquet(user_features_path, index=False)
        movie_features_train.to_parquet(movie_features_path, index=False)
        transactions_train.to_parquet(transactions_path, index=False)

        logger.info("✓ user_features_train.parquet written (%d rows)", len(user_features_train))
        logger.info("✓ movie_features_train.parquet written (%d rows)", len(movie_features_train))
        logger.info("✓ transactions_train.parquet written (%d rows)", len(transactions_train))

        phase4_report = {
            "run_id": run_paths.run_dir.name,
            "phase": "feature_factory",
            "generated_at_utc": pd.Timestamp.utcnow().isoformat(),
            "config": {
                "like_threshold": config.like_threshold,
                "min_tag_freq": config.min_tag_freq,
                "transactions_tokenization": {
                    "include_movie_tokens": True,
                    "include_genre_tokens": True,
                    "include_tag_tokens": True,
                },
            },
            "coverage": {
                "train_rows": int(len(train_df)),
                "train_unique_users": int(train_df["userId"].nunique()),
                "train_unique_movies": int(train_df["movieId"].nunique()),
                "tags_train_rows": int(len(tags_train)),
                "user_features_rows": int(len(user_features_train)),
                "movie_features_rows": int(len(movie_features_train)),
                "transactions_rows": int(len(transactions_train)),
                "user_feature_coverage": round(
                    float(len(user_features_train)) / float(train_df["userId"].nunique()),
                    6,
                ) if train_df["userId"].nunique() > 0 else 0.0,
                "movie_feature_coverage": round(
                    float(len(movie_features_train)) / float(train_df["movieId"].nunique()),
                    6,
                ) if train_df["movieId"].nunique() > 0 else 0.0,
            },
        }
        phase4_report_path = run_paths.reports_dir / "phase4_report.json"
        phase4_report_path.write_text(json.dumps(phase4_report, indent=2), encoding="utf-8")
        logger.info("✓ phase4_report.json written")
        logger.info("")

        del movies_for_features, tags_for_features, tags_train
        del user_features_train, movie_features_train
        # transactions_train kept alive for Phase 4.5

        phase4_elapsed = time.time() - phase4_start
        logger.info("="*60)
        logger.info("PHASE 4 COMPLETED in %.2f seconds", phase4_elapsed)
        logger.info("="*60)
        logger.info("")

        # ================================================================
        # PHASE 4.5: REDUCTION (transactions_train → transactions_train_reduced)
        # ================================================================
        phase45_start = time.time()
        logger.info("="*60)
        logger.info("PHASE 4.5: REDUCTION (Story B.1 prep)")
        logger.info("="*60)

        reduced_transactions, reduction_manifest = reduce.reduce_transactions(
            transactions=transactions_train,
            token_families=config.reduction_token_families,
            min_token_df=config.reduction_min_token_df,
            top_k_movies=config.reduction_top_k_movies,
            top_k_tags=config.reduction_top_k_tags,
            min_items_per_basket=config.reduction_min_items_per_basket,
            max_items_per_basket=config.reduction_max_items_per_basket,
            sample_frac=config.reduction_sample_frac,
            random_seed=config.random_seed,
        )

        # token stats (document frequency before vs after)
        token_stats = reduce.build_token_stats(transactions_train, reduced_transactions)
        basket_stats = reduce.build_basket_stats(transactions_train, reduced_transactions)

        # Write artefacts
        reduced_txn_path = run_paths.tables_dir / "transactions_train_reduced.parquet"
        reduced_transactions.to_parquet(reduced_txn_path, index=False)

        token_stats_path = run_paths.tables_dir / "token_stats.parquet"
        token_stats.to_parquet(token_stats_path, index=False)
        token_stats_csv_path = run_paths.reports_dir / "token_stats.csv"
        token_stats.to_csv(token_stats_csv_path, index=False)

        reduction_manifest_path = run_paths.reports_dir / "reduction_manifest.json"
        reduction_manifest_path.write_text(
            json.dumps(reduction_manifest, indent=2), encoding="utf-8"
        )
        basket_stats_path = run_paths.reports_dir / "basket_stats.json"
        basket_stats_path.write_text(
            json.dumps(basket_stats, indent=2), encoding="utf-8"
        )

        logger.info(
            "✓ transactions_train_reduced.parquet written (%d baskets)",
            len(reduced_transactions),
        )
        logger.info(
            "  Baskets:  %d → %d (%.1f%% kept)",
            reduction_manifest["before"]["n_baskets"],
            reduction_manifest["after"]["n_baskets"],
            (1.0 - reduction_manifest["reduction"]["basket_reduction_rate"]) * 100,
        )
        logger.info(
            "  Vocab:    %d → %d tokens (%.1f%% kept)",
            reduction_manifest["before"]["vocab_size"],
            reduction_manifest["after"]["vocab_size"],
            (1.0 - reduction_manifest["reduction"]["vocab_reduction_rate"]) * 100,
        )
        logger.info("✓ token_stats.parquet + token_stats.csv written")
        logger.info("✓ reduction_manifest.json written")
        logger.info("✓ basket_stats.json written")

        # Enrich phase4_report with reduction summary and re-write
        phase4_report["reduction"] = reduction_manifest
        phase4_report_path.write_text(json.dumps(phase4_report, indent=2), encoding="utf-8")
        logger.info("✓ phase4_report.json updated with reduction stats")

        del transactions_train, reduced_transactions, token_stats

        phase45_elapsed = time.time() - phase45_start
        logger.info("="*60)
        logger.info("PHASE 4.5 COMPLETED in %.2f seconds", phase45_elapsed)
        logger.info("="*60)
        logger.info("")

        # ================================================================
        # PHASE 5: AUDIT REPORT
        # ================================================================
        phase5_start = time.time()
        logger.info("="*60)
        logger.info("PHASE 5: AUDIT REPORT")
        logger.info("="*60)

        audit_report = audit.build_audit_report(
            run_id=run_paths.run_dir.name,
            schema_version=config.schema_version,
            clean_counts=clean_counts,
            validation_results=validation_results,
            split_validation=validation,
            phase4_report=phase4_report,
        )
        audit_report_path = run_paths.reports_dir / "audit_report.json"
        audit.write_audit_report(audit_report, audit_report_path)
        logger.info("✓ audit_report.json written")
        logger.info(
            "Audit summary: passed_all_gates=%s, warnings=%d",
            audit_report["summary"]["passed_all_gates"],
            audit_report["summary"]["warning_count"],
        )

        phase5_elapsed = time.time() - phase5_start
        logger.info("="*60)
        logger.info("PHASE 5 COMPLETED in %.2f seconds", phase5_elapsed)
        logger.info("="*60)
        logger.info("")

        # Free memory (split data no longer needed)
        del ratings_clean, train_df, test_df

        # Final summary
        elapsed = time.time() - start_time
        logger.info("="*60)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY in %.2f seconds", elapsed)
        logger.info("  Phase 2 (Clean):     %.2f seconds", phase2_elapsed)
        logger.info("  Phase 3 (Split):     %.2f seconds", phase3_elapsed)
        logger.info("  Phase 4 (Features):  %.2f seconds", phase4_elapsed)
        logger.info("  Phase 4.5 (Reduce):  %.2f seconds", phase45_elapsed)
        logger.info("  Phase 5 (Audit):     %.2f seconds", phase5_elapsed)
        logger.info("="*60)
        
        return run_paths.run_dir
        
    except FileNotFoundError as e:
        if logger:
            logger.error("File not found: %s", e)
            logger.error("Please ensure raw data directory exists with required CSV files")
        raise
    except ValueError as e:
        if logger:
            logger.error("Schema validation failed: %s", e)
            logger.error("Please check that CSV files have expected columns and formats")
        raise
    except validate.ValidationError as e:
        if logger:
            logger.error("Data validation failed: %s", e)
            logger.error("Please check data quality constraints and thresholds")
        raise
    except Exception as e:
        if logger:
            logger.error("PIPELINE FAILED: %s", e)
            logger.exception("Full traceback:")
        raise
