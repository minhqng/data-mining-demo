"""Raw dataset audit, provenance, unique-track preparation and stable splits.

The real snapshot is read-only. Derived files are written atomically and are
never allowed to hide a schema, duplicate, conflict or licensing problem.
"""

from __future__ import annotations

import datetime as _dt
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import Config
from .errors import ConflictingFeaturesError, DataError, SchemaError
from .features import F9_FEATURES, validate_feature_names
from .utils import (
    atomic_write_csv,
    atomic_write_json,
    atomic_write_parquet,
    read_json,
    sha256_file,
    sha256_obj,
    utc_now,
)

REQUIRED_COLUMNS: tuple[str, ...] = ("track_id", *F9_FEATURES, "track_genre")
METADATA_COLUMNS: tuple[str, ...] = (
    "artists",
    "album_name",
    "track_name",
    "popularity",
    "duration_ms",
    "explicit",
    "key",
    "mode",
    "time_signature",
)
ALLOWED_RAW_COLUMNS: frozenset[str] = frozenset((*REQUIRED_COLUMNS, *METADATA_COLUMNS))
UNIT01_FEATURES: tuple[str, ...] = (
    "acousticness",
    "danceability",
    "energy",
    "valence",
    "speechiness",
    "instrumentalness",
    "liveness",
)
AUDIT_VERSION = "btl11.audit.v1"
MANIFEST_VERSION = "btl11.manifest.v1"


# --------------------------------------------------------------------------- #
# Raw reading
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RawDataset:
    frame: pd.DataFrame
    raw_path: Path
    raw_sha256: str
    nbytes: int
    file_mtime_utc: str | None
    dropped_columns: tuple[str, ...]
    original_columns: tuple[str, ...]

    @property
    def n_rows(self) -> int:
        return len(self.frame)


def read_raw_dataset(csv_path: str | Path, config: Config) -> RawDataset:
    """Read and type-check the exact raw snapshot without mutating it."""
    path = Path(csv_path)
    if not path.exists():
        raise DataError(
            f"Raw dataset not found at {path}. Place the exact Kaggle Spotify Tracks "
            "Dataset CSV there (see README) or update project.raw_path. Mirrors and "
            "alternative datasets are not acceptable substitutes."
        )
    if not path.is_file():
        raise DataError(f"Raw dataset path is not a file: {path}")
    try:
        frame = pd.read_csv(path)
    except Exception as exc:  # pandas raises several parser exceptions
        raise SchemaError(f"Cannot parse CSV {path}: {exc}") from exc
    if frame.empty:
        raise SchemaError(f"Raw dataset {path} is empty.")

    original_columns = tuple(str(column) for column in frame.columns)
    dropped = tuple(
        str(column)
        for column in frame.columns
        if str(column).strip() == "" or str(column).startswith("Unnamed:")
    )
    frame = frame.drop(columns=list(dropped), errors="ignore")

    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise SchemaError(
            f"Raw dataset is missing mandatory columns: {missing}. "
            f"Required: {list(REQUIRED_COLUMNS)}"
        )
    unknown = sorted(str(column) for column in frame.columns if str(column) not in ALLOWED_RAW_COLUMNS)
    if unknown:
        raise SchemaError(
            f"Raw dataset contains undocumented columns {unknown}. The schema is fail-fast; "
            "add a documented contract decision instead of silently ignoring new fields."
        )
    duplicated_columns = frame.columns[frame.columns.duplicated()].tolist()
    if duplicated_columns:
        raise SchemaError(f"Raw dataset contains duplicate column names: {duplicated_columns}")

    frame = frame.copy()
    frame["_source_row"] = np.arange(len(frame), dtype=np.int64)

    track_ids = frame["track_id"].astype("string").str.strip()
    frame["track_id"] = track_ids

    for feature in F9_FEATURES:
        original = frame[feature]
        converted = pd.to_numeric(original, errors="coerce")
        invalid_type = original.notna() & converted.isna()
        if invalid_type.any():
            examples = frame.loc[invalid_type, ["_source_row", feature]].head(5)
            raise SchemaError(
                f"Feature '{feature}' contains non-numeric values; "
                f"examples:\n{examples.to_string(index=False)}"
            )
        frame[feature] = converted.astype("float64")

    frame["track_genre"] = frame["track_genre"].astype("string").str.strip()
    frame.loc[frame["track_genre"] == "", "track_genre"] = pd.NA

    for column in METADATA_COLUMNS:
        if column in frame.columns and column not in {"popularity", "duration_ms"}:
            frame[column] = frame[column].astype("string")
    if "track_genre" in frame.columns:
        # Kept only for auditing/bridging; never enters the feature matrix.
        pass

    stat = path.stat()
    mtime = (
        _dt.datetime.fromtimestamp(stat.st_mtime, tz=_dt.UTC)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )
    return RawDataset(
        frame=frame,
        raw_path=path,
        raw_sha256=sha256_file(path),
        nbytes=stat.st_size,
        file_mtime_utc=mtime,
        dropped_columns=dropped,
        original_columns=original_columns,
    )


# --------------------------------------------------------------------------- #
# Conflicts
# --------------------------------------------------------------------------- #
def detect_conflicting_ids(
    frame: pd.DataFrame,
    feature_names: Sequence[str],
    atol: float,
    rtol: float,
) -> tuple[list[str], pd.DataFrame]:
    """Find track IDs whose valid audio vectors disagree beyond tolerance."""
    names = list(feature_names)
    valid = frame.loc[
        frame["track_id"].notna() & (frame["track_id"].astype("string").str.len() > 0),
        ["track_id", "_source_row", *names],
    ].copy()
    if valid.empty:
        return [], pd.DataFrame(
            columns=["track_id", "n_rows", "max_feature_range", "worst_feature"]
        )
    grouped = valid.groupby("track_id", sort=True, dropna=True)[names]
    maxima = grouped.transform("max")
    minima = grouped.transform("min")
    reference = grouped.transform("median").abs()
    spread = maxima - minima
    tolerance = atol + rtol * reference
    conflicting_cells = spread.gt(tolerance) & maxima.notna()
    conflicting_ids = conflicting_cells.any(axis=1)
    if not conflicting_ids.any():
        return [], pd.DataFrame(
            columns=["track_id", "n_rows", "max_feature_range", "worst_feature"]
        )
    rows: list[dict[str, Any]] = []
    for track_id, group in valid.loc[conflicting_ids].groupby("track_id", sort=True):
        group_spread = group[names].max(numeric_only=True) - group[names].min(
            numeric_only=True
        )
        worst_feature = str(group_spread.idxmax())
        rows.append(
            {
                "track_id": str(track_id),
                "n_rows": int(len(group)),
                "max_feature_range": float(group_spread.max()),
                "worst_feature": worst_feature,
            }
        )
    details = pd.DataFrame(rows).sort_values("track_id").reset_index(drop=True)
    return details["track_id"].astype(str).tolist(), details


# --------------------------------------------------------------------------- #
# Audit
# --------------------------------------------------------------------------- #
def build_audit(raw: RawDataset, config: Config) -> dict[str, Any]:
    frame = raw.frame
    feature_names = list(F9_FEATURES)
    ids = frame["track_id"].dropna().astype(str)
    ids = ids[ids.str.len() > 0]
    unique_ids = ids.unique()
    rows_per_id = ids.value_counts()
    genre_per_id = (
        frame.dropna(subset=["track_id", "track_genre"])
        .groupby(frame["track_id"].astype("string"))["track_genre"]
        .nunique()
    )
    multi_genre_ids = genre_per_id[genre_per_id > 1].index.astype(str).tolist()
    conflict_ids, conflict_details = detect_conflicting_ids(
        frame,
        feature_names,
        atol=float(config.data["feature_tolerance_atol"]),
        rtol=float(config.data["feature_tolerance_rtol"]),
    )

    features = frame[feature_names]
    null_counts = {str(column): int(frame[column].isna().sum()) for column in frame.columns}
    invalid_ids = int(frame["track_id"].isna().sum() + (frame["track_id"].astype("string").str.len() == 0).sum())
    nonfinite = int((~np.isfinite(features.to_numpy(dtype="float64"))).sum())
    feature_nulls = {name: int(features[name].isna().sum()) for name in feature_names}
    range_rows: list[dict[str, Any]] = []
    quantiles = features.quantile([0.0, 0.01, 0.25, 0.5, 0.75, 0.99, 1.0])
    for name in feature_names:
        series = features[name].dropna()
        q25, q75 = float(series.quantile(0.25)), float(series.quantile(0.75))
        iqr = q75 - q25
        lower, upper = q25 - 1.5 * iqr, q75 + 1.5 * iqr
        outliers = int(((series < lower) | (series > upper)).sum()) if iqr > 0 else 0
        range_rows.append(
            {
                "feature": name,
                "min": float(series.min()),
                "p01": float(quantiles.loc[0.01, name]),
                "p25": q25,
                "median": float(series.median()),
                "mean": float(series.mean()),
                "std": float(series.std(ddof=0)),
                "p75": q75,
                "p99": float(quantiles.loc[0.99, name]),
                "max": float(series.max()),
                "skew": float(series.skew()),
                "iqr_outliers": outliers,
                "iqr_outlier_share": round(outliers / max(len(series), 1), 6),
            }
        )
    out_of_unit = {
        name: int(((features[name] < 0) | (features[name] > 1)).sum())
        for name in UNIT01_FEATURES
    }
    flags = {
        "tempo_equals_zero": int((features["tempo"] == 0).sum()),
        "loudness_gt_zero": int((features["loudness"] > 0).sum()),
        "tempo_lt_zero": int((features["tempo"] < 0).sum()),
        "unit01_out_of_range": out_of_unit,
        "nonfinite_feature_cells": nonfinite,
        "feature_null_cells": int(sum(feature_nulls.values())),
    }
    genre_counts = (
        frame["track_genre"].dropna().astype(str).value_counts().sort_index().to_dict()
    )
    exact_duplicates = int(
        frame.drop(columns=["_source_row"]).duplicated(keep=False).sum()
    )
    exact_duplicate_groups = int(
        frame.drop(columns=["_source_row"])
        .duplicated(keep="first")
        .sum()
    )
    audit = {
        "audit_version": AUDIT_VERSION,
        "generated_utc": utc_now(),
        "raw_path": str(raw.raw_path),
        "raw_sha256": raw.raw_sha256,
        "raw_bytes": raw.nbytes,
        "file_mtime_utc": raw.file_mtime_utc,
        "dataset_identity": {
            "source": "Kaggle Spotify Tracks Dataset",
            "kaggle_dataset_id": config.compliance["kaggle_dataset_id"],
            "canonical_url": config.compliance["canonical_kaggle_url"],
        },
        "schema": {
            "original_columns": list(raw.original_columns),
            "dropped_export_index_columns": list(raw.dropped_columns),
            "required_columns": list(REQUIRED_COLUMNS),
            "n_rows": int(len(frame)),
            "n_columns_after_drop": int(frame.shape[1] - 1),
            "schema_hash": sha256_obj({str(k): str(v) for k, v in frame.dtypes.items()}),
        },
        "counts": {
            "n_rows": int(len(frame)),
            "n_unique_track_ids": int(len(unique_ids)),
            "n_rows_with_duplicate_track_id": int(rows_per_id[rows_per_id > 1].sum()),
            "n_duplicate_track_ids": int((rows_per_id > 1).sum()),
            "max_rows_per_track_id": int(rows_per_id.max()) if len(rows_per_id) else 0,
            "rows_per_track_id_distribution": {
                str(int(key)): int(value)
                for key, value in rows_per_id.value_counts().sort_index().items()
            },
            "n_tracks_with_multiple_genres": int(len(multi_genre_ids)),
            "n_exact_duplicate_rows": exact_duplicates,
            "n_exact_duplicate_extra_rows": exact_duplicate_groups,
            "n_conflicting_track_ids": int(len(conflict_ids)),
            "n_invalid_track_ids": invalid_ids,
            "n_track_genres": int(frame["track_genre"].nunique()),
        },
        "missing": {
            "by_column": {k: v for k, v in null_counts.items() if v},
            "feature_nulls": feature_nulls,
            "track_genre_nulls": int(frame["track_genre"].isna().sum()),
        },
        "invalid": {
            "nonfinite_feature_cells": nonfinite,
            "track_id_nulls_or_blank": invalid_ids,
        },
        "feature_ranges": range_rows,
        "quantiles": {
            str(index): {name: float(value) for name, value in row.items()}
            for index, row in quantiles.iterrows()
        },
        "flags": flags,
        "genre_counts": {str(key): int(value) for key, value in genre_counts.items()},
        "conflicts": {
            "tolerance": {
                "atol": float(config.data["feature_tolerance_atol"]),
                "rtol": float(config.data["feature_tolerance_rtol"]),
            },
            "ids": conflict_ids,
            "details": conflict_details.to_dict(orient="records"),
        },
        "multi_genre_ids_examples": sorted(multi_genre_ids)[:10],
        "cleaning_protocol": {
            "unit": "unique track_id",
            "conflicting_policy": "fail_primary_stage",
            "missing_numeric_policy": config.data["missing_numeric"],
            "outliers": "reported, not automatically removed or winsorized",
            "genre_role": "post-hoc multi-label validation only; never a model input",
        },
    }
    return audit


def build_manifest(
    raw: RawDataset,
    audit: dict[str, Any],
    config: Config,
    *,
    existing: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the audit manifest while preserving a human license decision.

    Re-auditing must never silently turn a prior `rejected`/approved human review
    back into `pending`.
    """
    schema_hash = audit["schema"]["schema_hash"]
    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "source": "Kaggle — Spotify Tracks Dataset",
        "kaggle_dataset_id": config.compliance["kaggle_dataset_id"],
        "canonical_url": config.compliance["canonical_kaggle_url"],
        "file": raw.raw_path.name,
        "local_path": str(raw.raw_path),
        "sha256": raw.raw_sha256,
        "bytes": raw.nbytes,
        "rows": audit["counts"]["n_rows"],
        "columns": audit["schema"]["n_columns_after_drop"],
        "track_id_unique": audit["counts"]["n_unique_track_ids"],
        "track_genre_nunique": audit["counts"]["n_track_genres"],
        "schema_hash": schema_hash,
        "downloaded_utc": None,
        "downloaded_utc_note": (
            "The supplier did not record an authoritative retrieval timestamp; "
            "file_mtime_utc is filesystem metadata, not proof of download time."
        ),
        "file_mtime_utc": raw.file_mtime_utc,
        "audited_utc": utc_now(),
        "auditor_version": AUDIT_VERSION,
        "license_review_status": "pending_human_verification",
        "redistribution_allowed": False,
        "local_analysis_authorized_by_owner": bool(
            config.compliance["allow_pending_license_local_analysis"]
        ),
        "license_metadata_observed": {
            "kaggle_page_license": (
                "ODbL 1.0 (Open Data Commons Open Database License 1.0), "
                "copyright original authors"
            ),
            "huggingface_mirror_license": "BSD",
            "discrepancy": True,
            "observed_utc": "2026-10-27",
            "sources": [
                config.compliance["canonical_kaggle_url"],
                "https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset",
            ],
            "provenance_risk": (
                "The mirror states the data was collected via Spotify's Web API. "
                "Spotify Developer Policy may restrict ML/AI reuse of Platform/Content; "
                "this observation is not legal advice and does not clear redistribution."
            ),
        },
        "license_note": (
            "Local academic analysis was authorized by the project owner for the assigned "
            "BTL11 coursework. Kaggle lists ODbL 1.0 while the Hugging Face mirror lists "
            "BSD; the discrepancy and Spotify API provenance remain unresolved. Public "
            "redistribution and Spotify Platform/Content ML/AI reuse are NOT cleared and "
            "remain a human-verification gate. Raw data is gitignored and must not be published."
        ),
        "license_review_owner_action": (
            "Verify Kaggle terms and institutional policy in docs/DATASET_PROVENANCE.md "
            "before any public release, deployment or redistribution."
        ),
    }
    if existing:
        preserved_keys = (
            "license_review_status",
            "redistribution_allowed",
            "license_review_owner_action",
            "license_review_owner",
            "license_reviewed_utc",
        )
        hash_matches = existing.get("sha256") == raw.raw_sha256
        manifest["license_review_previous"] = {
            key: existing.get(key) for key in preserved_keys
        }
        manifest["license_review_previous"]["raw_sha256_matched"] = hash_matches
        if hash_matches:
            for key in preserved_keys:
                if key in existing:
                    manifest[key] = existing[key]
            manifest["license_review_preserved_from_previous_audit"] = True
        else:
            manifest["license_review_preserved_from_previous_audit"] = False
            manifest["license_note"] += (
                " A previous license decision was NOT carried over because it was bound "
                "to different raw bytes; the new snapshot requires human review."
            )
    return manifest


def _load_existing_manifest(raw_path: str | Path) -> dict[str, Any] | None:
    path = Path(raw_path).parent / "MANIFEST.json"
    if not path.exists():
        return None
    try:
        payload = read_json(path)
    except (OSError, ValueError) as exc:
        raise DataError(
            f"Existing manifest {path} is unreadable. Refusing to overwrite a potentially "
            "human-edited compliance decision; repair or explicitly remove it first."
        ) from exc
    if not isinstance(payload, dict):
        raise DataError(f"Existing manifest {path} is not a JSON object.")
    return payload


# --------------------------------------------------------------------------- #
# Splits
# --------------------------------------------------------------------------- #
def _hash_bucket(value: str) -> int:
    digest = int.from_bytes(bytes.fromhex(sha256_obj(value)), "big")
    return digest % 1_000_000


def make_id_splits(
    track_ids: Iterable[str], seed: int, dev_fraction: float
) -> dict[str, list[str]]:
    ids = sorted({str(track_id) for track_id in track_ids})
    if len(ids) < 4:
        raise DataError("At least four unique tracks are required to make disjoint splits.")
    dev: list[str] = []
    holdout: list[str] = []
    cutoff = int(round(dev_fraction * 1_000_000))
    for track_id in ids:
        if _hash_bucket(f"{seed}:split:{track_id}") < cutoff:
            dev.append(track_id)
        else:
            holdout.append(track_id)
    if len(dev) < 4 or len(holdout) < 2:
        raise DataError(
            f"Hash split produced dev={len(dev)}, holdout={len(holdout)}; adjust "
            "dev_fraction or provide more unique tracks."
        )
    return {"dev": dev, "holdout": holdout}


def select_ids_by_hash(
    ids: Iterable[str], size: int, namespace: str, seed: int
) -> list[str]:
    unique = sorted({str(track_id) for track_id in ids})
    if size >= len(unique):
        return unique
    if size < 2:
        raise DataError("Evaluation cohort size must be >= 2.")
    ranked = sorted(unique, key=lambda track_id: (sha256_obj(f"{seed}:{namespace}:{track_id}"), track_id))
    return sorted(ranked[:size])


def build_splits(
    tracks: pd.DataFrame, config: Config
) -> tuple[dict[str, list[str]], dict[str, str], pd.DataFrame]:
    seed = int(config.data["split_seed"])
    split = make_id_splits(
        tracks["track_id"].astype(str).tolist(),
        seed=seed,
        dev_fraction=float(config.data["dev_fraction"]),
    )
    dev, holdout = split["dev"], split["holdout"]
    eval_size = min(int(config.data["eval_cohort_size"]), max(2, len(dev) - 2))
    eval_ids = select_ids_by_hash(dev, eval_size, "eval", int(config.data["eval_seed"]))
    eval_set = set(eval_ids)
    fit_ids = sorted(track_id for track_id in dev if track_id not in eval_set)
    if len(fit_ids) < 2:
        raise DataError("Development fit subset has fewer than two tracks.")
    assert not (set(dev) & set(holdout))
    assert not (set(fit_ids) & eval_set)
    assert set(fit_ids) | eval_set == set(dev)
    splits = {
        "dev": dev,
        "holdout": holdout,
        "eval": eval_ids,
        "fit": fit_ids,
    }
    split_hashes = {name: sha256_obj(list(values)) for name, values in splits.items()}
    split_hashes["combined"] = sha256_obj(
        {
            "splits": split_hashes,
            "split_seed": seed,
            "eval_seed": int(config.data["eval_seed"]),
            "dev_fraction": float(config.data["dev_fraction"]),
        }
    )
    split_table = pd.DataFrame(
        [
            {"split": name, "n_unique_track_ids": len(values)}
            for name, values in splits.items()
        ]
    )
    return splits, split_hashes, split_table


# --------------------------------------------------------------------------- #
# Preparation
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PreparedData:
    tracks: pd.DataFrame
    track_genres: pd.DataFrame
    splits: dict[str, list[str]]
    split_hashes: dict[str, str]
    dataset_sha256: str
    processed_sha256: str
    manifest: dict[str, Any] = field(default_factory=dict)
    audit: dict[str, Any] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)
    _positions: pd.Series = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        positions = pd.Series(
            np.arange(len(self.tracks), dtype=np.int64),
            index=self.tracks["track_id"].astype(str),
        )
        if positions.index.has_duplicates:
            raise DataError("Prepared tracks artifact contains duplicate track_id values.")
        object.__setattr__(self, "_positions", positions)
        object.__setattr__(self, "splits", {k: list(v) for k, v in self.splits.items()})

    @property
    def n_tracks(self) -> int:
        return len(self.tracks)

    @property
    def all_ids(self) -> list[str]:
        return self.tracks["track_id"].astype(str).tolist()

    def positions_for(self, ids: Sequence[str]) -> np.ndarray:
        missing = [track_id for track_id in ids if track_id not in self._positions.index]
        if missing:
            raise DataError(
                f"{len(missing)} requested track IDs are absent from tracks.parquet; "
                f"examples: {missing[:5]}"
            )
        return self._positions.loc[list(ids)].to_numpy(dtype=np.int64)

    def subset_tracks(self, ids: Sequence[str], preserve_order: bool = True) -> pd.DataFrame:
        unique_ids = list(dict.fromkeys(str(track_id) for track_id in ids))
        subset = self.tracks.iloc[self.positions_for(unique_ids)].copy()
        if preserve_order:
            subset = subset.set_index("track_id").loc[unique_ids].reset_index()
        return subset

    def feature_matrix(self, ids: Sequence[str], feature_names: Sequence[str]) -> np.ndarray:
        names = validate_feature_names(feature_names)
        subset = self.tracks.iloc[self.positions_for(list(ids))]
        return subset.loc[:, names].to_numpy(dtype=np.float64, copy=True)

    def genre_bridge(self) -> pd.DataFrame:
        return self.track_genres.copy()


def prepare_dataset(config: Config) -> PreparedData:
    raw = read_raw_dataset(config.raw_path, config)
    audit = build_audit(raw, config)
    existing = _load_existing_manifest(config.raw_path)
    manifest = build_manifest(raw, audit, config, existing=existing)
    _write_manifest_and_audit(manifest, audit, config)
    # Fail closed if the owner has rejected the data or analysis is not authorized.
    config.license_operational_status(manifest)

    features = list(F9_FEATURES)
    frame = raw.frame.copy()
    valid_id = frame["track_id"].notna() & (frame["track_id"].astype("string").str.len() > 0)
    finite_features = np.isfinite(frame[features].to_numpy(dtype="float64")).all(axis=1)
    invalid_mask = ~(valid_id & finite_features)
    quarantine_parts: list[pd.DataFrame] = []
    if invalid_mask.any():
        invalid = frame.loc[invalid_mask, ["track_id", "_source_row", *features]].copy()
        invalid["reason"] = np.where(
            ~valid_id.loc[invalid_mask], "invalid_track_id", "missing_or_nonfinite_feature"
        )
        quarantine_parts.append(invalid)
    valid = frame.loc[~invalid_mask].copy()

    conflict_ids, conflict_details = detect_conflicting_ids(
        valid,
        features,
        atol=float(config.data["feature_tolerance_atol"]),
        rtol=float(config.data["feature_tolerance_rtol"]),
    )
    if conflict_ids:
        conflict_path = Path("data/processed/conflicting_track_ids.csv")
        atomic_write_csv(conflict_path, conflict_details)
        raise ConflictingFeaturesError(
            f"{len(conflict_ids)} track_id values carry contradicting audio vectors. "
            f"Details written to {conflict_path}. The primary stage fails closed; resolve "
            "with a documented data rule instead of dropping/averaging silently."
        )

    metadata_present = [column for column in METADATA_COLUMNS if column in valid.columns]
    order_columns = ["track_id", "track_genre", "_source_row"]
    valid = valid.sort_values(order_columns, kind="mergesort", na_position="last")
    representatives = valid.drop_duplicates(subset=["track_id"], keep="first").copy()

    tracks = representatives.loc[:, ["track_id", *features, *metadata_present, "_source_row"]].copy()
    tracks = tracks.sort_values("track_id", kind="mergesort").reset_index(drop=True)
    tracks["track_id"] = tracks["track_id"].astype(str)
    if tracks["track_id"].duplicated().any():
        raise DataError("Deduplication failed: tracks still contain duplicate track_id.")

    genre_bridge = (
        valid[["track_id", "track_genre"]]
        .dropna(subset=["track_id", "track_genre"])
        .rename(columns={"track_genre": "genre"})
        .copy()
    )
    genre_bridge["track_id"] = genre_bridge["track_id"].astype(str)
    genre_bridge["genre"] = genre_bridge["genre"].astype(str)
    valid_track_ids = set(tracks["track_id"])
    genre_bridge = genre_bridge[genre_bridge["track_id"].isin(valid_track_ids)]
    genre_bridge = (
        genre_bridge.drop_duplicates(subset=["track_id", "genre"])
        .sort_values(["track_id", "genre"], kind="mergesort")
        .reset_index(drop=True)
    )
    if genre_bridge.duplicated(subset=["track_id", "genre"]).any():
        raise DataError("Genre bridge violates the unique (track_id, genre) contract.")

    splits, split_hashes, split_table = build_splits(tracks, config)

    processed_dir = Path("data/processed")
    split_dir = Path("data/splits")
    processed_dir.mkdir(parents=True, exist_ok=True)
    split_dir.mkdir(parents=True, exist_ok=True)
    tracks_path = atomic_write_parquet(processed_dir / "tracks.parquet", tracks)
    genres_path = atomic_write_parquet(processed_dir / "track_genres.parquet", genre_bridge)
    for split_name, ids in splits.items():
        atomic_write_csv(
            split_dir / f"{split_name}_ids.csv",
            pd.DataFrame({"track_id": ids}),
        )
    atomic_write_csv(processed_dir / "split_summary.csv", split_table)

    quarantine_frame = (
        pd.concat(quarantine_parts, ignore_index=True)
        if quarantine_parts
        else pd.DataFrame(columns=["track_id", "_source_row", *features, "reason"])
    )
    atomic_write_csv(processed_dir / "quarantine_rows.csv", quarantine_frame)

    tracks_sha = sha256_file(tracks_path)
    genres_sha = sha256_file(genres_path)
    processed_sha = sha256_obj({"tracks_parquet": tracks_sha, "track_genres_parquet": genres_sha})
    split_files = {
        name: sha256_file(split_dir / f"{name}_ids.csv") for name in splits
    }
    split_sha = sha256_obj({"combined": split_hashes["combined"], "files": split_files})

    n_rows_with_multi_genre = int(genre_bridge.groupby("track_id")["genre"].nunique().gt(1).sum())
    summary = {
        "prepare_version": "btl11.prepare.v1",
        "generated_utc": utc_now(),
        "raw_sha256": raw.raw_sha256,
        "raw_rows": int(len(frame)),
        "unique_track_ids": int(len(tracks)),
        "dropped_rows": int(invalid_mask.sum()),
        "quarantined_row_ids": int(invalid_mask.sum()),
        "dropped_track_ids_without_valid_row": int(
            len(set(frame.loc[invalid_mask, "track_id"].dropna().astype(str)) - valid_track_ids)
        ),
        "conflicting_track_ids": 0,
        "tracks_with_multiple_genres": n_rows_with_multi_genre,
        "genre_bridge_rows": int(len(genre_bridge)),
        "n_track_genres": int(genre_bridge["genre"].nunique()),
        "tracks_without_genre": int(
            tracks["track_id"].isin(set(genre_bridge["track_id"])).eq(False).sum()
        ),
        "tracks_parquet_sha256": tracks_sha,
        "track_genres_parquet_sha256": genres_sha,
        "processed_sha256": processed_sha,
        "split_sha256": split_sha,
        "split_hashes": split_hashes,
        "split_files_sha256": split_files,
        "splits": {name: len(ids) for name, ids in splits.items()},
        "feature_order": features,
        "genre_in_feature_matrix": False,
        "license_operational_status": config.license_operational_status(manifest),
    }
    atomic_write_json(processed_dir / "prepare_summary.json", summary)
    # Attach audit counts computed before quarantine.
    audit["prepare_summary"] = summary
    atomic_write_json(processed_dir / "audit.json", audit)
    return PreparedData(
        tracks=tracks,
        track_genres=genre_bridge,
        splits=splits,
        split_hashes=split_hashes,
        dataset_sha256=raw.raw_sha256,
        processed_sha256=processed_sha,
        manifest=manifest,
        audit=audit,
        summary=summary,
    )


def _write_manifest_and_audit(
    manifest: dict[str, Any], audit: dict[str, Any], config: Config
) -> None:
    raw_dir = Path(config.raw_path).parent
    processed_dir = Path("data/processed")
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(raw_dir / "MANIFEST.json", manifest)
    atomic_write_json(processed_dir / "audit.json", audit)


def audit_only(config: Config, input_path: str | Path | None = None) -> dict[str, Any]:
    requested = Path(input_path) if input_path is not None else Path(config.raw_path)
    if requested.resolve() != Path(config.raw_path).resolve():
        raise DataError(
            f"Audit input {requested} differs from project.raw_path {config.raw_path}. "
            "Audit and prepare must refer to the exact same snapshot."
        )
    raw = read_raw_dataset(requested, config)
    audit = build_audit(raw, config)
    existing = _load_existing_manifest(requested)
    manifest = build_manifest(raw, audit, config, existing=existing)
    _write_manifest_and_audit(manifest, audit, config)
    config.license_operational_status(manifest)
    return {"manifest": manifest, "audit": audit}


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_prepared(config: Config, enforce_license: bool = True) -> PreparedData:
    processed_dir = Path("data/processed")
    tracks_path = processed_dir / "tracks.parquet"
    genres_path = processed_dir / "track_genres.parquet"
    summary_path = processed_dir / "prepare_summary.json"
    manifest_path = Path(config.raw_path).parent / "MANIFEST.json"
    missing = [
        str(path)
        for path in (tracks_path, genres_path, summary_path, manifest_path)
        if not path.exists()
    ]
    if missing:
        raise DataError(
            f"Prepared artifacts are missing: {missing}. Run `btl11 audit` and "
            "`btl11 prepare` first."
        )
    import json

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    audit = json.loads((processed_dir / "audit.json").read_text(encoding="utf-8"))
    if summary.get("raw_sha256") != sha256_file(config.raw_path):
        raise DataError(
            "Raw dataset bytes changed after prepare; rerun audit/prepare before empirical work."
        )
    if manifest.get("sha256") != summary.get("raw_sha256"):
        raise DataError("MANIFEST.json and prepare_summary.json disagree on the raw hash.")
    if enforce_license:
        config.license_operational_status(manifest)

    tracks = pd.read_parquet(tracks_path)
    genres = pd.read_parquet(genres_path)
    tracks["track_id"] = tracks["track_id"].astype(str)
    genres["track_id"] = genres["track_id"].astype(str)
    genres["genre"] = genres["genre"].astype(str)
    if not tracks["track_id"].is_monotonic_increasing or tracks["track_id"].duplicated().any():
        raise DataError("tracks.parquet must be sorted by unique track_id.")
    if genres.duplicated(subset=["track_id", "genre"]).any():
        raise DataError("track_genres.parquet violates unique (track_id, genre).")
    if not set(genres["track_id"]).issubset(set(tracks["track_id"])):
        raise DataError("track_genres.parquet contains orphan track IDs.")
    tracks_sha = sha256_file(tracks_path)
    genres_sha = sha256_file(genres_path)
    if tracks_sha != summary.get("tracks_parquet_sha256"):
        raise DataError("tracks.parquet hash differs from prepare_summary.json.")
    if genres_sha != summary.get("track_genres_parquet_sha256"):
        raise DataError("track_genres.parquet hash differs from prepare_summary.json.")
    processed_sha = sha256_obj({"tracks_parquet": tracks_sha, "track_genres_parquet": genres_sha})
    if processed_sha != summary.get("processed_sha256"):
        raise DataError("Combined processed hash mismatch.")

    split_dir = Path("data/splits")
    splits: dict[str, list[str]] = {}
    for name in ("dev", "holdout", "eval", "fit"):
        path = split_dir / f"{name}_ids.csv"
        if not path.exists():
            raise DataError(f"Split file missing: {path}. Rerun prepare.")
        values = pd.read_csv(path)["track_id"].astype(str).tolist()
        if len(values) != len(set(values)):
            raise DataError(f"Split {name} contains duplicate IDs.")
        splits[name] = values
        if sha256_file(path) != summary["split_files_sha256"].get(name):
            raise DataError(f"Split file hash mismatch for {name}.")
    if set(splits["dev"]) & set(splits["holdout"]):
        raise DataError("Development/holdout splits overlap.")
    if not set(splits["eval"]) <= set(splits["dev"]):
        raise DataError("Evaluation cohort is not a subset of development.")
    if set(splits["fit"]) | set(splits["eval"]) != set(splits["dev"]):
        raise DataError("fit + eval must equal dev.")
    if set(splits["fit"]) & set(splits["eval"]):
        raise DataError("fit and eval overlap.")
    if set(splits["dev"]) | set(splits["holdout"]) != set(tracks["track_id"]):
        raise DataError("dev + holdout does not cover every unique track.")

    # Reconstruct split hashes from loaded IDs and enforce the stored combined hash.
    split_hashes = {name: sha256_obj(list(values)) for name, values in splits.items()}
    split_hashes["combined"] = summary["split_hashes"]["combined"]
    file_hashes = {
        name: sha256_file(split_dir / f"{name}_ids.csv") for name in splits
    }
    split_sha = sha256_obj({"combined": split_hashes["combined"], "files": file_hashes})
    if split_sha != summary.get("split_sha256"):
        raise DataError("Combined split hash mismatch.")
    return PreparedData(
        tracks=tracks,
        track_genres=genres,
        splits=splits,
        split_hashes=split_hashes,
        dataset_sha256=manifest["sha256"],
        processed_sha256=processed_sha,
        manifest=manifest,
        audit=audit,
        summary=summary,
    )
