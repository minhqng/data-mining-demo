"""Small, dependency-light utilities shared by the BTL11 pipeline.

Only deterministic writers are provided: JSON/text/CSV/Parquet writes go to a
temporary sibling and are atomically moved into place.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import tempfile
import uuid
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Time / hashing
# --------------------------------------------------------------------------- #
def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp with a trailing ``Z``."""
    return (
        _dt.datetime.now(_dt.UTC)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def json_sanitize(value: Any) -> Any:
    """Convert numpy/pandas values into JSON-safe Python types.

    Non-finite floats become ``None``; callers must attach an explicit reason
    when a metric is undefined instead of relying on this conversion.
    """
    if value is None:
        return None
    if isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        numeric = float(value)
        return numeric if np.isfinite(numeric) else None
    if isinstance(value, np.ndarray):
        return [json_sanitize(item) for item in value.tolist()]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): json_sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_sanitize(item) for item in value]
    if isinstance(value, (pd.Timestamp, _dt.datetime, _dt.date)):
        return value.isoformat()
    if value is pd.NA:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return str(value)


def canonical_json(value: Any) -> str:
    return json.dumps(
        json_sanitize(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def sha256_obj(value: Any) -> str:
    return sha256_text(canonical_json(value))


def short_hash(value: str, length: int = 8) -> str:
    return value[:length]


def safe_slug(text: str, max_length: int = 56) -> str:
    allowed = [character if character.isalnum() or character in "-_." else "-" for character in text]
    slug = "".join(allowed).strip("-._") or "run"
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug[:max_length]


_RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,199}$")


def validate_run_id(run_id: str) -> str:
    """Reject path traversal or non-artifact run identifiers."""
    if not isinstance(run_id, str) or not _RUN_ID_PATTERN.fullmatch(run_id) or ".." in run_id:
        raise ValueError(
            "Invalid run_id. Expected a single artifact-directory name using "
            "letters, digits, dot, underscore or hyphen (no path separators)."
        )
    return run_id


def new_run_id(config_sha256: str, dataset_sha256: str, tag: str) -> str:
    stamp = _dt.datetime.now(_dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    return "_".join(
        [
            stamp,
            short_hash(config_sha256, 8),
            short_hash(dataset_sha256, 8),
            safe_slug(tag),
        ]
    )


def allocate_run_dir(output_dir: str | Path, base_run_id: str) -> tuple[str, Path]:
    """Create a new run directory without ever reusing an existing one."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate = base_run_id
    counter = 2
    while True:
        path = output_dir / candidate
        try:
            path.mkdir(parents=False, exist_ok=False)
            return candidate, path
        except FileExistsError:
            candidate = f"{base_run_id}-{counter}"
            counter += 1


# --------------------------------------------------------------------------- #
# Atomic writers
# --------------------------------------------------------------------------- #
def _temporary_path(target: Path, suffix: str = "") -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{target.name}.{uuid.uuid4().hex[:8]}",
        suffix=suffix,
        dir=target.parent,
    )
    os.close(descriptor)
    return Path(name)


def atomic_write_text(path: str | Path, text: str) -> Path:
    target = Path(path)
    temporary = _temporary_path(target)
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def atomic_write_json(path: str | Path, payload: Any) -> Path:
    return atomic_write_text(
        path,
        json.dumps(
            json_sanitize(payload),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
    )


def atomic_write_csv(path: str | Path, frame: pd.DataFrame) -> Path:
    target = Path(path)
    temporary = _temporary_path(target)
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def atomic_write_parquet(path: str | Path, frame: pd.DataFrame) -> Path:
    target = Path(path)
    temporary = _temporary_path(target, suffix=".parquet")
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# Runtime context
# --------------------------------------------------------------------------- #
def peak_rss_mib() -> float | None:
    """Return this process's high-water RSS if the platform exposes it."""
    try:
        status = Path("/proc/self/status").read_text(encoding="utf-8")
        for line in status.splitlines():
            if line.startswith("VmHWM:"):
                return round(int(line.split()[1]) / 1024.0, 2)
    except (OSError, ValueError, IndexError):
        pass
    try:
        import psutil

        process = psutil.Process(os.getpid())
        return round(process.memory_info().rss / (1024.0 * 1024.0), 2)
    except Exception:  # pragma: no cover - platform-specific fallback
        return None


def cpu_model() -> str | None:
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or None


def total_ram_gib() -> float | None:
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                return round(int(line.split()[1]) / (1024.0 * 1024.0), 2)
    except (OSError, ValueError, IndexError):
        pass
    return None


def host_info() -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cpu_model": cpu_model(),
        "cpu_count": os.cpu_count(),
        "ram_total_gib": total_ram_gib(),
    }


def library_versions(packages: Iterable[str]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def git_info(repo_root: str | Path = ".") -> dict[str, Any]:
    root = Path(repo_root)
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            ).stdout.strip()
        )
        return {"available": True, "commit": commit, "dirty_worktree": dirty}
    except (OSError, subprocess.SubprocessError):
        return {
            "available": False,
            "commit": None,
            "dirty_worktree": None,
            "note": "No git repository/commit was created in this runtime.",
        }


def count_lines(path: str | Path) -> int:
    with open(path, "rb") as handle:
        return sum(1 for _ in handle)
