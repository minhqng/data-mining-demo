"""Data-contract, duplicate and provenance gates."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from btl11.data import (
    load_prepared,
    prepare_dataset,
    read_raw_dataset,
    select_ids_by_hash,
)
from btl11.errors import ConflictingFeaturesError, DataError, SchemaError
from btl11.features import F5_FEATURES, F9_FEATURES, to_feature_matrix

from .conftest import make_config, synthetic_tracks, write_config, write_raw


def test_audit_and_prepare_unique_tracks_multilabel(prepared_env):
    tmp_path, config, config_path, prepared = prepared_env
    assert prepared.n_tracks == 48
    assert prepared.tracks["track_id"].is_unique
    assert prepared.tracks["track_id"].is_monotonic_increasing
    assert "track_genre" not in prepared.tracks.columns
    assert len(prepared.track_genres) == 60  # 48 primary + 12 secondary genre rows
    assert not prepared.track_genres.duplicated(subset=["track_id", "genre"]).any()
    multi = prepared.track_genres.groupby("track_id")["genre"].nunique().gt(1).sum()
    assert int(multi) == 12
    assert prepared.summary["unique_track_ids"] == 48
    assert prepared.summary["dropped_rows"] == 0
    assert prepared.summary["conflicting_track_ids"] == 0
    assert prepared.processed_sha256
    assert prepared.split_hashes["combined"]
    manifest = json.loads((tmp_path / "data" / "raw" / "MANIFEST.json").read_text())
    for key in (
        "sha256",
        "bytes",
        "rows",
        "track_id_unique",
        "track_genre_nunique",
        "license_review_status",
        "redistribution_allowed",
        "local_analysis_authorized_by_owner",
    ):
        assert key in manifest
    assert manifest["redistribution_allowed"] is False
    assert manifest["license_review_status"] == "pending_human_verification"
    audit = json.loads((tmp_path / "data" / "processed" / "audit.json").read_text())
    assert audit["counts"]["n_unique_track_ids"] == 48
    assert audit["counts"]["n_tracks_with_multiple_genres"] == 12
    assert audit["counts"]["n_exact_duplicate_rows"] >= 4
    assert audit["counts"]["n_conflicting_track_ids"] == 0


def test_exact_duplicate_rows_do_not_double_weight(prepared_env):
    _, _, _, prepared = prepared_env
    # The exact duplicate rows are collapsed to one unique track and the genre
    # bridge is unique; repeated identical metadata rows cannot change centroids.
    assert prepared.n_tracks == 48
    assert prepared.track_genres["track_id"].value_counts().max() == 2


def test_splits_are_id_disjoint_stable_and_cover_all(prepared_env):
    _, config, _, prepared = prepared_env
    dev = set(prepared.splits["dev"])
    holdout = set(prepared.splits["holdout"])
    eval_ids = set(prepared.splits["eval"])
    fit = set(prepared.splits["fit"])
    assert not (dev & holdout)
    assert dev | holdout == set(prepared.tracks["track_id"])
    assert eval_ids <= dev
    assert not (fit & eval_ids)
    assert fit | eval_ids == dev
    assert prepared.splits["eval"] == select_ids_by_hash(
        prepared.splits["dev"], len(prepared.splits["eval"]), "eval", config.data["eval_seed"]
    )
    # Re-running prepare on the same bytes yields the same processed and split hashes.
    first_processed = prepared.processed_sha256
    first_split = prepared.summary["split_sha256"]
    second = prepare_dataset(config)
    assert second.processed_sha256 == first_processed
    assert second.summary["split_sha256"] == first_split
    assert second.splits == prepared.splits


def test_conflicting_audio_vectors_fail_closed(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    write_raw(tmp_path, conflict=True)
    config = make_config(tmp_path)
    write_config(config, tmp_path)
    with pytest.raises(ConflictingFeaturesError, match="contradicting"):
        prepare_dataset(config)
    conflict_file = tmp_path / "data" / "processed" / "conflicting_track_ids.csv"
    assert conflict_file.exists()
    assert "trk_000" in conflict_file.read_text()


def test_missing_feature_row_is_quarantined_not_imputed(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    write_raw(tmp_path, missing_feature=True)
    config = make_config(tmp_path)
    prepared = prepare_dataset(config)
    assert prepared.summary["dropped_rows"] == 1
    assert prepared.n_tracks == 47
    quarantine = pd.read_csv(tmp_path / "data" / "processed" / "quarantine_rows.csv")
    assert len(quarantine) == 1
    assert quarantine.iloc[0]["reason"] == "missing_or_nonfinite_feature"
    assert len(prepared.tracks) == 47
    assert (prepared.tracks[list(F9_FEATURES)].isna().sum().sum()) == 0


def test_wrong_type_and_missing_columns_fail_with_useful_errors(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    write_raw(tmp_path, wrong_type=True)
    config = make_config(tmp_path)
    with pytest.raises(SchemaError, match="non-numeric"):
        read_raw_dataset("data/raw/dataset.csv", config)
    # Missing mandatory column.
    frame = synthetic_tracks(24)
    frame = frame.drop(columns=["energy"])
    frame.to_csv(tmp_path / "data" / "raw" / "dataset.csv", index=False)
    with pytest.raises(SchemaError, match="missing mandatory columns"):
        read_raw_dataset("data/raw/dataset.csv", config)


def test_feature_matrix_order_and_no_metadata(prepared_env):
    _, _, _, prepared = prepared_env
    f9 = to_feature_matrix(prepared.tracks, F9_FEATURES)
    f5 = to_feature_matrix(prepared.tracks, F5_FEATURES)
    assert f9.shape == (48, 9)
    assert f5.shape == (48, 5)
    with pytest.raises(SchemaError, match="Metadata/label"):
        to_feature_matrix(prepared.tracks, ["danceability", "track_genre"])
    with pytest.raises(SchemaError, match="locked F9 or F5"):
        to_feature_matrix(prepared.tracks, ["danceability", "energy"])
    reordered = list(F9_FEATURES)
    reordered[0], reordered[1] = reordered[1], reordered[0]
    with pytest.raises(SchemaError, match="locked F9 or F5"):
        to_feature_matrix(prepared.tracks, reordered)
    tracks_missing = prepared.tracks.drop(columns=["tempo"])
    with pytest.raises(SchemaError, match="missing"):
        to_feature_matrix(tracks_missing, F9_FEATURES)


def test_undocumented_extra_column_fails_fast(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    write_raw(tmp_path)
    config = make_config(tmp_path)
    path = tmp_path / "data" / "raw" / "dataset.csv"
    frame = pd.read_csv(path)
    frame["mystery_feature"] = 1.0
    frame.to_csv(path, index=False)
    with pytest.raises(SchemaError, match="undocumented columns"):
        read_raw_dataset(path, config)


def test_missing_raw_csv_and_missing_manifest_are_actionable(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = make_config(tmp_path)
    with pytest.raises(DataError, match="Raw dataset not found"):
        prepare_dataset(config)
    with pytest.raises(DataError, match="Prepared artifacts are missing"):
        load_prepared(config)


def test_rejected_license_fails_closed(fixture_env):
    _, config, _ = fixture_env
    with pytest.raises(Exception, match="rejected"):
        config.license_operational_status({"license_review_status": "rejected"})
    # Pending is allowed only because the owner explicitly authorized local analysis.
    allowed = config.license_operational_status(
        {"license_review_status": "pending_human_verification"}
    )
    assert allowed == "local_academic_analysis_allowed_redistribution_pending"
    strict = config.merged(
        {"compliance": {"allow_pending_license_local_analysis": False,
                        "local_analysis_basis": "not authorized",
                        "canonical_kaggle_url": config.compliance["canonical_kaggle_url"],
                        "kaggle_dataset_id": config.compliance["kaggle_dataset_id"],
                        "redistribution_allowed": False}}
    )
    with pytest.raises(Exception, match="not cleared"):
        strict.license_operational_status(
            {"license_review_status": "pending_human_verification"}
        )


def test_audit_preserves_rejected_human_license_decision(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    write_raw(tmp_path)
    config = make_config(tmp_path)
    manifest_path = tmp_path / "data" / "raw" / "MANIFEST.json"
    from btl11.utils import sha256_file

    raw_sha = sha256_file(tmp_path / "data" / "raw" / "dataset.csv")
    manifest_path.write_text(
        json.dumps(
            {
                "sha256": raw_sha,
                "license_review_status": "rejected",
                "redistribution_allowed": False,
                "license_review_owner_action": "Do not run or publish.",
            }
        ),
        encoding="utf-8",
    )
    from btl11.data import audit_only

    with pytest.raises(Exception, match="rejected"):
        audit_only(config)
    preserved = json.loads(manifest_path.read_text())
    assert preserved["license_review_status"] == "rejected"
    assert preserved["license_review_owner_action"] == "Do not run or publish."
    assert preserved["license_review_preserved_from_previous_audit"] is True


def test_config_locks_dataset_and_feature_order(fixture_env):
    _, config, _ = fixture_env
    with pytest.raises(Exception, match="mandated Kaggle"):
        config.merged(
            {
                "compliance": {
                    "canonical_kaggle_url": "https://www.kaggle.com/datasets/other/dataset",
                    "kaggle_dataset_id": config.compliance["kaggle_dataset_id"],
                    "allow_pending_license_local_analysis": True,
                    "local_analysis_basis": "x",
                    "redistribution_allowed": False,
                }
            }
        )
    wrong_features = list(F9_FEATURES)
    wrong_features[0], wrong_features[1] = wrong_features[1], wrong_features[0]
    with pytest.raises(Exception, match="locked F9"):
        config.merged({"features": {"primary": wrong_features}})
    with pytest.raises(Exception, match="silhouette_sample_size"):
        config.merged({"search": {"silhouette_sample_size": 9}})
    with pytest.raises(Exception, match="silhouette_seed"):
        config.merged({"search": {"silhouette_seed": 7}})
    with pytest.raises(Exception, match="not implemented"):
        config.merged({"data": {"missing_numeric": "impute_dev_only"}})


def test_license_decision_is_bound_to_raw_bytes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    write_raw(tmp_path)
    config = make_config(tmp_path)
    from btl11.data import audit_only

    audit_only(config)
    manifest_path = tmp_path / "data" / "raw" / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["license_review_status"] = "approved_for_coursework"
    manifest["license_review_owner_action"] = "Approved for the exact audited bytes."
    manifest["redistribution_allowed"] = False
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    # Replace the raw bytes without changing schema: the old approval must not
    # silently apply to a different snapshot.
    frame = pd.read_csv(tmp_path / "data" / "raw" / "dataset.csv")
    frame.loc[0, "energy"] = float(frame.loc[0, "energy"]) + 0.01
    frame.to_csv(tmp_path / "data" / "raw" / "dataset.csv", index=False)
    audit_only(config)
    updated = json.loads(manifest_path.read_text())
    assert updated["license_review_status"] == "pending_human_verification"
    assert updated["license_review_preserved_from_previous_audit"] is False
    assert updated["license_review_previous"]["raw_sha256_matched"] is False
