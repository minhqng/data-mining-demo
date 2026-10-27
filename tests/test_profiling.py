"""Raw-unit profiles and multi-label post-hoc genre behavior."""

from __future__ import annotations

import pandas as pd
import pytest

from btl11.errors import DataError, SchemaError
from btl11.features import F9_FEATURES
from btl11.profiling import (
    generate_genre_posthoc,
    generate_profiles,
    profile_clusters,
)


def test_profile_counts_reconcile_and_raw_units(prepared_env):
    tmp_path, config, _, prepared = prepared_env
    assignments = pd.DataFrame(
        {
            "track_id": prepared.tracks["track_id"],
            "cluster": [index % 2 for index in range(prepared.n_tracks)],
        }
    )
    wide = profile_clusters(prepared.tracks, assignments, prepared.tracks, F9_FEATURES)
    assert int(wide["n"].sum()) == prepared.n_tracks
    assert abs(float(wide["share"].sum()) - 1.0) < 1e-12
    assert set(wide["cluster"]) == {0, 1}
    for feature in F9_FEATURES:
        assert f"{feature}_mean" in wide.columns
        assert f"{feature}_median" in wide.columns
        assert f"{feature}_iqr" in wide.columns
        assert f"{feature}_z" in wide.columns
    assert wide["interpretation_hypothesis"].str.contains("không phải genre").all()
    duplicate = pd.concat([assignments, assignments.iloc[[0]]], ignore_index=True)
    with pytest.raises(SchemaError, match="one row"):
        profile_clusters(prepared.tracks, duplicate, prepared.tracks, F9_FEATURES)
    unknown = assignments.copy()
    unknown.loc[0, "track_id"] = "missing"
    with pytest.raises(DataError, match="lost rows"):
        profile_clusters(prepared.tracks, unknown, prepared.tracks, F9_FEATURES)


def test_profiles_files_without_bundle(prepared_env):
    tmp_path, config, _, prepared = prepared_env
    assignments = pd.DataFrame(
        {"track_id": prepared.tracks["track_id"], "cluster": [i % 2 for i in range(prepared.n_tracks)]}
    )
    run_path = tmp_path / "artifacts" / "runs" / "synthetic_profile"
    run_path.mkdir(parents=True)
    result = generate_profiles(
        config, prepared, run_path, "synthetic_profile", assignments, bundle=None
    )
    assert (run_path / "profiles.csv").exists()
    assert (run_path / "tables" / "profiles_long.csv").exists()
    assert (run_path / "figures" / "profile_heatmap.png").exists()
    assert result["n_assigned"] == prepared.n_tracks


def test_profile_zero_iqr_feature_yields_zero_deviation():
    tracks = pd.DataFrame(
        {
            "track_id": [f"t{index}" for index in range(6)],
            "energy": [0.5] * 6,
            "danceability": [0.1, 0.2, 0.3, 0.7, 0.8, 0.9],
        }
    )
    assignments = pd.DataFrame(
        {"track_id": tracks["track_id"], "cluster": [0, 0, 0, 1, 1, 1]}
    )
    wide = profile_clusters(tracks, assignments, tracks, ["energy"])
    assert wide["energy_iqr"].tolist() == [0.0, 0.0]
    assert wide["energy_z"].tolist() == [0.0, 0.0]


def test_multilabel_incidence_can_sum_above_100pct(prepared_env, tmp_path):
    tmp_path, config, _, prepared = prepared_env
    # Put every unique track in one cluster; tracks with two genres must let
    # incidence shares sum above 100% without being treated as an error.
    assignments = pd.DataFrame(
        {"track_id": prepared.tracks["track_id"], "cluster": 0}
    )
    run_path = tmp_path / "artifacts" / "runs" / "genre_probe"
    run_path.mkdir(parents=True)
    result = generate_genre_posthoc(
        config, prepared, run_path, "genre_probe", assignments
    )
    incidence = pd.read_csv(run_path / "tables" / "genre_cluster_incidence.csv")
    share_sum = incidence.groupby("cluster")["share_of_cluster_tracks"].sum()
    assert share_sum.max() > 1.0
    assert (run_path / "figures" / "genre_cluster_heatmap.png").exists()
    external = result["single_genre_metrics"]
    assert external["n_single_genre_eval"] > 0
    assert external["n_omitted_not_single_label"] == 12
    assert external["adjusted_rand_vs_genre"]["value"] is None
    assert external["adjusted_rand_vs_genre"]["reason"]
