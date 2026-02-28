"""Metadata helpers for raw data files.

MovieLens Data Mining
Nguyen Sy Hung
2026
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import logging
import subprocess
import sys


logger = logging.getLogger(__name__)


REQUIRED_FILES = ["movies.csv", "ratings.csv", "tags.csv", "links.csv"]
OPTIONAL_FILES = ["genome-tags.csv", "genome-scores.csv"]


@dataclass
class FileMeta:
    name: str
    bytes: int
    sha256: str
    rows: int


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute SHA-256 hash of a file.
    
    Args:
        path: Path to file
        chunk_size: Size of chunks to read
        
    Returns:
        Hex digest of SHA-256 hash
        
    Raises:
        Exception: If file cannot be read
    """
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(chunk_size), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except Exception as e:
        logger.error(f"Failed to hash {path.name}: {e}")
        raise


def count_rows_csv(path: Path) -> int:
    """Count CSV rows (excluding header).
    
    Warning: Assumes well-formed CSV. Embedded newlines in quoted fields
    will cause overcounting.
    
    Args:
        path: Path to CSV file
        
    Returns:
        Number of data rows (excluding header)
        
    Raises:
        Exception: If file cannot be read
    """
    try:
        with path.open("rb") as handle:
            lines = sum(1 for _ in handle)
        return max(0, lines - 1)
    except Exception as e:
        logger.error(f"Failed to count rows in {path.name}: {e}")
        raise


def get_git_commit_hash(project_root: Path) -> str | None:
    """Get short git commit hash if repo exists.
    
    Args:
        project_root: Path to git repository root
        
    Returns:
        Short commit hash or None if unavailable
    """
    try:
        result = subprocess.run(
            ["git", "-C", str(project_root), "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,  # Prevent hanging
        )
        value = result.stdout.strip()
        return value or None
    except subprocess.TimeoutExpired:
        logger.warning("Git command timed out")
        return None
    except subprocess.CalledProcessError as e:
        logger.warning(f"Git command failed: {e}")
        return None
    except FileNotFoundError:
        logger.warning("Git executable not found")
        return None
    except Exception as e:
        logger.warning(f"Unexpected error getting git hash: {e}")
        return None


def _build_file_meta(raw_dir: Path, file_name: str) -> dict:
    """Build metadata for a single file.
    
    Args:
        raw_dir: Path to raw data directory
        file_name: Name of file to process
        
    Returns:
        Dictionary with file metadata or {"missing": True}
    """
    file_path = raw_dir / file_name
    if not file_path.exists():
        return {"missing": True}

    logger.info(f"Computing metadata for {file_name}...")
    meta = FileMeta(
        name=file_name,
        bytes=file_path.stat().st_size,
        sha256=sha256_file(file_path),
        rows=count_rows_csv(file_path),
    )
    return asdict(meta)


def build_raw_metadata(
    raw_dir: Path,
    run_id: str,
    schema_version: str,
    pipeline_version: str | None = None,
) -> dict:
    """Build complete raw metadata snapshot.
    
    Args:
        raw_dir: Path to raw data directory
        run_id: Unique run identifier
        schema_version: Schema version string
        pipeline_version: Git commit hash or version string
        
    Returns:
        Dictionary with complete metadata
    """
    now_iso = datetime.now(timezone.utc).isoformat()
    metadata = {
        "run_id": run_id,
        "generated_at_utc": now_iso,
        "schema_version": schema_version,
        "pipeline_version": pipeline_version,
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "raw_dir": str(raw_dir),
        "files": {},
    }

    for file_name in REQUIRED_FILES + OPTIONAL_FILES:
        metadata["files"][file_name] = _build_file_meta(raw_dir, file_name)

    return metadata


def write_metadata_raw(metadata: dict, out_path: Path) -> None:
    """Write metadata to JSON file.
    
    Args:
        metadata: Metadata dictionary
        out_path: Path to output JSON file
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    logger.info(f"Wrote metadata: {out_path}")
