"""Shared test-only synthetic fixtures. No real Kaggle data is used here."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.datasets import make_blobs

from btl11.config import Config
from btl11.data import prepare_dataset
from btl11.features import F9_FEATURES

REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = REPO_ROOT / "configs" / "base.yaml"


def synthetic_tracks(n_ids: int = 48, seed: int = 7) -> pd.DataFrame:
    """Well-separated 4-cluster synthetic audio frame with multi-label rows."""
    X, _ = make_blobs(
        n_samples=n_ids,
        centers=4,
        n_features=9,
        cluster_std=0.55,
        random_state=seed,
    )
    scaled = np.empty_like(X)
    for column in range(X.shape[1]):
        values = X[:, column]
        scaled[:, column] = (values - values.min()) / max(values.max() - values.min(), 1e-9)
    unit_columns = ["acousticness", "danceability", "energy", "valence", "speechiness", "instrumentalness", "liveness"]
    unit_values = 0.02 + 0.96 * scaled[:, [0, 1, 2, 3, 6, 7, 8]]
    tempo = 60.0 + 170.0 * scaled[:, 4]
    loudness = -32.0 + 27.0 * scaled[:, 5]
    features = pd.DataFrame(unit_values, columns=unit_columns)
    features.insert(4, "tempo", tempo)
    features.insert(5, "loudness", loudness)
    features = features.loc[:, list(F9_FEATURES)]
    genres = ["genre_a", "genre_b", "genre_c", "genre_d", "genre_e", "genre_f"]
    rows: list[dict[str, Any]] = []
    for index in range(n_ids):
        track_id = f"trk_{index:03d}"
        row = {
            "track_id": track_id,
            "artists": f"Artist {index}",
            "album_name": f"Album {index}",
            "track_name": f"Track {index}",
            "popularity": int(20 + index),
            "duration_ms": int(120000 + index * 1000),
            "explicit": "False",
            "key": index % 12,
            "mode": index % 2,
            "time_signature": 4,
            **{feature: float(features.iloc[index][feature]) for feature in F9_FEATURES},
            "track_genre": genres[index % len(genres)],
        }
        rows.append(row)
    # Multi-label duplicates: identical audio vectors, different genre.
    for index in range(12):
        duplicate = dict(rows[index])
        duplicate["track_genre"] = genres[(index + 2) % len(genres)]
        rows.append(duplicate)
    # Exact duplicate rows within the same genre (must not double-weight the track).
    for index in range(2):
        rows.append(dict(rows[index]))
    frame = pd.DataFrame(rows)
    frame.insert(0, "Unnamed: 0", np.arange(len(frame)))
    return frame


def write_raw(
    tmp_path: Path,
    *,
    conflict: bool = False,
    missing_feature: bool = False,
    wrong_type: bool = False,
    n_ids: int = 48,
) -> Path:
    frame = synthetic_tracks(n_ids=n_ids)
    if conflict:
        duplicate = dict(frame.iloc[0])
        duplicate["energy"] = min(0.99, float(duplicate["energy"]) + 0.3)
        duplicate["track_genre"] = "genre_z"
        frame = pd.concat([frame, pd.DataFrame([duplicate])], ignore_index=True)
    if missing_feature:
        # Pick a single-row track so the whole track is quarantined.
        frame.loc[20, "energy"] = np.nan
    if wrong_type:
        frame["danceability"] = frame["danceability"].astype(object)
        frame.loc[4, "danceability"] = "not-a-number"
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / "dataset.csv"
    frame.to_csv(path, index=False)
    return path


def make_config(tmp_path: Path, **overrides: Any) -> Config:
    base = Config.load(BASE_CONFIG)
    merged = base.merged(
        {
            "project": {
                "raw_path": "data/raw/dataset.csv",
                "output_dir": "artifacts/runs",
            },
            "data": {
                "dev_fraction": 0.75,
                "holdout_fraction": 0.25,
                "eval_cohort_size": 8,
                "eval_seed": 42,
                "split_seed": 42,
            },
            "search": {
                "quick_k": [2],
                "quick_seeds": [42],
                "stability_seeds": [11, 23, 37],
                "silhouette_sample_size": 8,
                "max_fit_count_warning": 50,
            },
            "gmm": {
                "n_init_quick": 1,
                "n_init_final": 1,
                "max_iter": 200,
            },
            "visualization": {
                "tsne_sample_size": 32,
                "tsne_perplexity": 8,
                "scatter_sample_size": 32,
            },
            "hierarchical": {"sample_size": 40},
            "selection": {
                "shortlist_k": [2],
                "finalists": [],
                "frozen": [],
                "selected": None,
                "stability_bootstrap": 1,
                "stability_n_init_gmm": 1,
            },
        }
    )
    if overrides:
        merged = merged.merged(overrides)
    return merged


def write_config(config: Config, tmp_path: Path) -> Path:
    config_dir = tmp_path / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    path = config_dir / "base.yaml"
    path.write_text(yaml.safe_dump(config.raw, sort_keys=True, allow_unicode=True), encoding="utf-8")
    return path


@pytest.fixture
def fixture_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    write_raw(tmp_path)
    config = make_config(tmp_path)
    config_path = write_config(config, tmp_path)
    return tmp_path, config, config_path


@pytest.fixture
def prepared_env(fixture_env):
    tmp_path, config, config_path = fixture_env
    prepared = prepare_dataset(config)
    return tmp_path, config, config_path, prepared


def descriptor(k: int = 2, algorithm: str = "kmeans", covariance: str | None = None) -> dict:
    return {
        "algorithm": algorithm,
        "k": k,
        "covariance_type": covariance,
        "feature_set": "f9",
        "scaler": "standard",
        "pca": None,
        "seed": 42,
    }
