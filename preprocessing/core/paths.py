"""Path helpers for pipeline run management.

MovieLens Data Mining
Nguyen Sy Hung
2026
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List


@dataclass(frozen=True)
class RunPaths:
    """Container for pipeline directory paths."""
    raw_dir: Path
    run_dir: Path
    raw_staging_dir: Path
    tables_dir: Path
    reports_dir: Path


def make_run_paths(
    raw_dir: str | Path, 
    out_root: str | Path, 
    run_tag: str | None = None,
    validate_raw: bool = True,
    raw_staging_root: str | Path = "data/raw_staging"
) -> RunPaths:
    """Create run directory structure with validation.
    
    Args:
        raw_dir: Path to raw data directory
        out_root: Root path for processed outputs
        run_tag: Optional custom run identifier
        validate_raw: Whether to validate raw directory exists
        raw_staging_root: Root path for shared raw staging Parquets (default: data/raw_staging)
        
    Returns:
        RunPaths object with all directory paths
        
    Raises:
        FileNotFoundError: If raw_dir doesn't exist and validate_raw=True
    """
    raw_dir_path = Path(raw_dir)
    out_root_path = Path(out_root)
    raw_staging_dir = Path(raw_staging_root)

    # Validate raw directory exists (unless explicitly skipped)
    if validate_raw and not raw_dir_path.exists():
        raise FileNotFoundError(f"Raw directory does not exist: {raw_dir_path}")

    # Add milliseconds to prevent collisions (format: YYYYMMDD_HHMMSS_mmm)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")[:-3]
    run_id = run_tag or f"run_{ts}"

    run_dir = out_root_path / run_id
    tables_dir = run_dir / "tables"
    reports_dir = run_dir / "reports"

    # Create directories (raw_staging is shared, not per-run)
    raw_staging_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    return RunPaths(
        raw_dir=raw_dir_path,
        run_dir=run_dir,
        raw_staging_dir=raw_staging_dir,
        tables_dir=tables_dir,
        reports_dir=reports_dir,
    )


def list_runs(out_root: str | Path) -> List[str]:
    """List all existing run_ids in chronological order.
    
    Args:
        out_root: Root path for processed outputs
        
    Returns:
        Sorted list of run_id strings
    """
    out_path = Path(out_root)
    if not out_path.exists():
        return []
    runs = [d.name for d in out_path.iterdir() if d.is_dir() and d.name.startswith("run_")]
    return sorted(runs)


def get_latest_run(out_root: str | Path) -> Path | None:
    """Get the most recent run directory.
    
    Args:
        out_root: Root path for processed outputs
        
    Returns:
        Path to latest run directory, or None if no runs exist
    """
    runs = list_runs(out_root)
    return Path(out_root) / runs[-1] if runs else None
