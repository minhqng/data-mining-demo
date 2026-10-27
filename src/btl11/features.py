"""Feature allowlists and strict numeric-matrix construction."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from .errors import SchemaError

F9_FEATURES: tuple[str, ...] = (
    "acousticness",
    "danceability",
    "energy",
    "valence",
    "tempo",
    "loudness",
    "speechiness",
    "instrumentalness",
    "liveness",
)
F5_FEATURES: tuple[str, ...] = (
    "acousticness",
    "danceability",
    "energy",
    "valence",
    "tempo",
)
FORBIDDEN_PRIMARY: tuple[str, ...] = (
    "track_genre",
    "track_id",
    "track_name",
    "artists",
    "album_name",
    "popularity",
    "duration_ms",
    "explicit",
    "key",
    "mode",
    "time_signature",
)
FEATURE_SET_NAMES = ("f5", "f9")


def feature_names_for_set(name: str, primary: Sequence[str], ablation: Sequence[str]) -> list[str]:
    """Resolve ``f9``/``f5`` to the configured, ordered allowlist."""
    normalized = str(name).lower()
    if normalized == "f9":
        return list(primary)
    if normalized == "f5":
        return list(ablation)
    raise SchemaError(f"Unknown feature_set {name!r}; expected one of {FEATURE_SET_NAMES}.")


def validate_feature_names(feature_names: Sequence[str]) -> list[str]:
    """Reject metadata/labels and require exactly the F5 or F9 allowlist."""
    names = list(feature_names)
    forbidden = sorted(set(names) & set(FORBIDDEN_PRIMARY))
    if forbidden:
        raise SchemaError(f"Metadata/label columns are not allowed in X: {forbidden}")
    if tuple(names) == F9_FEATURES:
        return names
    if tuple(names) == F5_FEATURES:
        return names
    raise SchemaError(
        "The clustering feature list must equal the locked F9 or F5 allowlist in order; "
        f"received {names}."
    )


def to_feature_matrix(tracks: pd.DataFrame, feature_names: Sequence[str]) -> np.ndarray:
    """Return a finite float64 matrix in exactly the requested column order.

    No ``track_id`` or raw index is returned. Missing/non-numeric/infinite
    values fail loudly instead of being silently imputed.
    """
    names = validate_feature_names(feature_names)
    missing = [name for name in names if name not in tracks.columns]
    if missing:
        raise SchemaError(f"Feature columns missing from the tracks artifact: {missing}")
    subset = tracks.loc[:, names]
    non_numeric = [
        column
        for column in subset.columns
        if not pd.api.types.is_numeric_dtype(subset[column])
    ]
    if non_numeric:
        raise SchemaError(f"Feature columns must be numeric; non-numeric: {non_numeric}")
    matrix = subset.to_numpy(dtype=np.float64, copy=True)
    if matrix.shape[1] != len(names):
        raise SchemaError(f"Unexpected feature width {matrix.shape[1]} != {len(names)}")
    if not np.isfinite(matrix).all():
        bad_rows = np.flatnonzero(~np.isfinite(matrix).all(axis=1))[:5].tolist()
        raise SchemaError(f"Feature matrix contains NaN/inf; example row positions: {bad_rows}")
    return matrix


def feature_frame(tracks: pd.DataFrame, feature_names: Sequence[str]) -> pd.DataFrame:
    """Typed view used for raw-unit profiling (keeps track_id out of X)."""
    names = validate_feature_names(feature_names)
    return tracks.loc[:, names].astype(np.float64).copy()
