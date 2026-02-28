"""Pipeline configuration types and defaults.

MovieLens Data Mining
Nguyen Sy Hung
2026
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PipelineConfig:
    """Pipeline configuration with validation."""
    raw_dir: Path
    out_root: Path
    run_tag: str | None = None
    schema_version: str = "0.1.0"
    
    # Phase 3: Temporal split parameters
    test_frac: float = 0.1  # Fraction of timeline for test (0.1 = 10%)
    split_policy: str = "global"  # "global" or "per_user" (future)
    cutoff_date: str | None = None  # Explicit cutoff (or None for auto)
    
    # Phase 4: Feature engineering parameters
    like_threshold: float = 4.0  # For transaction baskets
    min_tag_freq: int = 10  # Tag vocabulary filtering
    random_seed: int = 42  # Reproducibility

    # Phase 4.5: Reduction knobs (for transactions_train_reduced.parquet / Story B.1)
    # Token families to retain in reduced transactions.
    # Use ("genre",) for cheapest run; ("genre", "movie", "tag") for full.
    reduction_token_families: tuple[str, ...] = ("genre", "movie", "tag")
    # Drop tokens that appear in fewer than this many baskets.
    reduction_min_token_df: int = 5
    # Keep only the top-K movie:* tokens by document frequency. None = no cap.
    reduction_top_k_movies: int | None = 5_000
    # Keep only the top-K tag:* tokens by document frequency. None = no cap.
    reduction_top_k_tags: int | None = 500
    # Drop baskets that become shorter than this after vocab filtering.
    reduction_min_items_per_basket: int = 2
    # Cap basket size (genres kept first, then movies, then tags). None = no cap.
    reduction_max_items_per_basket: int | None = None
    # Sample this fraction of baskets after all filtering. None = keep all.
    # Intended for dev/exploration only — label clearly if used.
    reduction_sample_frac: float | None = None

    def __post_init__(self):
        """Validate configuration on creation."""
        # Convert to Path if needed (frozen dataclass workaround)
        if not isinstance(self.raw_dir, Path):
            object.__setattr__(self, 'raw_dir', Path(self.raw_dir))
        if not isinstance(self.out_root, Path):
            object.__setattr__(self, 'out_root', Path(self.out_root))
        
        # Validate ranges
        if not (0.0 < self.test_frac < 1.0):
            raise ValueError(f"test_frac must be in (0, 1), got {self.test_frac}")
        if self.split_policy not in ["global", "per_user"]:
            raise ValueError(f"split_policy must be 'global' or 'per_user', got {self.split_policy}")
        if self.like_threshold < 0:
            raise ValueError(f"like_threshold must be >= 0, got {self.like_threshold}")
        if self.min_tag_freq < 1:
            raise ValueError(f"min_tag_freq must be >= 1, got {self.min_tag_freq}")
        if self.reduction_min_token_df < 1:
            raise ValueError(f"reduction_min_token_df must be >= 1, got {self.reduction_min_token_df}")
        if self.reduction_top_k_movies is not None and self.reduction_top_k_movies < 1:
            raise ValueError(f"reduction_top_k_movies must be >= 1 or None, got {self.reduction_top_k_movies}")
        if self.reduction_top_k_tags is not None and self.reduction_top_k_tags < 1:
            raise ValueError(f"reduction_top_k_tags must be >= 1 or None, got {self.reduction_top_k_tags}")
        if self.reduction_min_items_per_basket < 1:
            raise ValueError(f"reduction_min_items_per_basket must be >= 1, got {self.reduction_min_items_per_basket}")
        if self.reduction_max_items_per_basket is not None and self.reduction_max_items_per_basket < 1:
            raise ValueError(f"reduction_max_items_per_basket must be >= 1 or None, got {self.reduction_max_items_per_basket}")
        if self.reduction_sample_frac is not None and not (0.0 < self.reduction_sample_frac <= 1.0):
            raise ValueError(f"reduction_sample_frac must be in (0, 1] or None, got {self.reduction_sample_frac}")
