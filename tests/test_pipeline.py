"""TEST-ONLY synthetic end-to-end proof. It does not satisfy real-data gates."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from btl11.errors import ConfigError, VerificationError
from btl11.experiment import (
    ablate,
    compare_experiment_set,
    find_matching_run,
    iter_run_docs,
    load_run_doc,
    refit_all_tracks,
    run_candidate,
    run_dir,
    stability,
    validate_holdout,
    ward_diagnostic,
)
from btl11.models import load_bundle
from btl11.reporting import run_report
from btl11.run_visuals import visualize_run
from btl11.utils import sha256_obj
from btl11.verification import verify_run

from .conftest import descriptor


def _selection_config(config, desc):
    return config.merged(
        {
            "selection": {
                "baseline_experiment_set": "baseline_f9_standard",
                "shortlist_k": [2],
                "finalists": [desc],
                "frozen": [desc],
                "selected": desc,
                "ablation_algorithms": ["kmeans"],
                "ablation_covariance": "diag",
                "stability_bootstrap": 1,
                "stability_n_init_gmm": 1,
                "notes": "synthetic end-to-end evidence only",
            }
        }
    )


def test_tiny_synthetic_pipeline_end_to_end(prepared_env):
    tmp_path, config, _, prepared = prepared_env
    candidate = run_candidate(
        config,
        prepared,
        algorithm="kmeans",
        k=2,
        seed=42,
        experiment_set="baseline_f9_standard",
        experiment_note="TEST-ONLY synthetic baseline",
    )
    assert candidate.status == "completed"
    doc = load_run_doc(config, candidate.run_id)
    assert doc["run_type"] == "candidate"
    assert doc["has_model"] is True
    assert (run_dir(config, candidate.run_id) / "model.joblib").exists()
    assert (run_dir(config, candidate.run_id) / "assignments.parquet").exists()
    verified = verify_run(config, candidate.run_id)
    assert verified["status"] == "verified"
    comparison = compare_experiment_set(config, "baseline_f9_standard")
    assert Path(comparison["comparison_csv"]).exists()
    assert "bic_rank_within_space" in pd.read_csv(comparison["comparison_csv"]).columns

    desc = descriptor()
    selected_config = _selection_config(config, desc)
    ablation = ablate(selected_config, prepared, "scaler")
    assert Path(ablation["comparison_csv"]).exists()
    assert {"standard", "robust", "none_unscaled_E00_diagnostic"} <= set(
        pd.read_csv(ablation["comparison_csv"])["variant"]
    )
    stability_result = stability(selected_config, prepared, "finalists")
    assert stability_result["run_id"]
    stability_doc = load_run_doc(selected_config, stability_result["run_id"])
    assert stability_doc["n_fits_failed"] == 0
    assert verify_run(selected_config, stability_result["run_id"])["status"] == "verified"
    holdout = validate_holdout(selected_config, prepared)
    assert holdout["run_id"]
    assert verify_run(selected_config, holdout["run_id"])["status"] == "verified"
    refit = refit_all_tracks(selected_config, prepared, "selected_final")
    assert refit["run_id"] != candidate.run_id
    final_doc = load_run_doc(selected_config, refit["run_id"])
    assert final_doc["run_type"] == "final_refit"
    assert final_doc["evaluation_model_id"] == candidate.run_id
    assert final_doc["holdout_run_id"] == holdout["run_id"]
    assert final_doc["final_refit_model_id"] == refit["run_id"]
    assert final_doc["protocol_note"]
    assert (run_dir(selected_config, refit["run_id"]) / "profiles.csv").exists()
    assert (run_dir(selected_config, refit["run_id"]) / "tables" / "genre_cluster_incidence.csv").exists()
    assert verify_run(selected_config, refit["run_id"])["status"] == "verified"

    visuals = visualize_run(
        selected_config, prepared, refit["run_id"], ["pca", "profile", "genre"]
    )
    assert Path(visuals["outputs"]["pca"]["figure"]).exists()
    assert Path(visuals["outputs"]["pca"]["coordinates"]).exists()
    ward = ward_diagnostic(selected_config, prepared)
    assert ward["run_id"]
    assert verify_run(selected_config, ward["run_id"])["status"] == "verified"
    report = run_report(
        selected_config, prepared, refit["run_id"], tmp_path / "artifacts" / "report"
    )
    assert Path(report["report_tables_md"]).exists()
    assert (tmp_path / "artifacts" / "report" / "final_cluster_profiles.csv").exists()
    assert (tmp_path / "artifacts" / "report" / "figures_index.csv").exists()
    # The final refit model is a distinct artifact identity from the eval model.
    base_bundle = load_bundle(run_dir(selected_config, candidate.run_id) / "model.joblib")
    final_bundle = load_bundle(run_dir(selected_config, refit["run_id"]) / "model.joblib")
    assert base_bundle.n_fit != final_bundle.n_fit
    assert base_bundle.trained_on_ids_sha256 != final_bundle.trained_on_ids_sha256
    # No genre/metadata column entered the fitted feature space.
    assert "track_genre" not in final_bundle.feature_names
    assert "track_id" not in final_bundle.feature_names


def test_failed_candidate_is_recorded_as_failed(prepared_env):
    tmp_path, config, _, prepared = prepared_env
    with pytest.raises(Exception, match="E00"):
        run_candidate(
            config,
            prepared,
            algorithm="kmeans",
            k=2,
            scaler="none",
            pca_variance=0.95,
            experiment_set="failure_probe",
        )
    failed = [
        doc for _, doc in iter_run_docs(config) if doc.get("experiment_set") == "failure_probe"
    ]
    assert len(failed) == 1
    assert failed[0]["status"] == "failed"
    assert failed[0]["error_type"]
    assert failed[0]["has_model"] is False
    assert not (run_dir(config, failed[0]["run_id"]) / "model.joblib").exists()
    index = pd.read_csv(tmp_path / "artifacts" / "INDEX.csv")
    assert (index["status"] == "failed").any()


def test_holdout_is_one_time_per_frozen_descriptor(prepared_env):
    _, config, _, prepared = prepared_env
    desc = descriptor()
    selected_config = _selection_config(config, desc)
    run_candidate(
        config, prepared, algorithm="kmeans", k=2, seed=42,
        experiment_set="baseline_f9_standard",
    )
    validate_holdout(selected_config, prepared)
    # A brand-new candidate with the same frozen descriptor must not reset the
    # one-time holdout guard.
    run_candidate(
        config, prepared, algorithm="kmeans", k=2, seed=42,
        experiment_set="baseline_second",
    )
    with pytest.raises(ConfigError, match="already used"):
        validate_holdout(selected_config, prepared)


def test_refit_requires_frozen_winner_and_completed_holdout(prepared_env):
    _, config, _, prepared = prepared_env
    desc = descriptor()
    run_candidate(
        config, prepared, algorithm="kmeans", k=2, seed=42,
        experiment_set="baseline_f9_standard",
    )
    no_frozen = config.merged(
        {"selection": {"selected": desc, "frozen": [], "shortlist_k": [2]}}
    )
    with pytest.raises(ConfigError, match="frozen"):
        refit_all_tracks(no_frozen, prepared)
    mismatched = config.merged(
        {
            "selection": {
                "selected": desc,
                "frozen": [descriptor(k=4)],
                "shortlist_k": [2],
            }
        }
    )
    with pytest.raises(ConfigError, match="differs"):
        refit_all_tracks(mismatched, prepared)
    frozen_no_holdout = _selection_config(config, desc)
    with pytest.raises(ConfigError, match="holdout"):
        refit_all_tracks(frozen_no_holdout, prepared)


def test_reuse_rejects_changed_model_hyperparameters(prepared_env):
    _, config, _, prepared = prepared_env
    desc = descriptor()
    run_candidate(
        config, prepared, algorithm="kmeans", k=2, seed=42,
        experiment_set="baseline_f9_standard",
    )
    kwargs = {
        "run_type": "candidate",
        "dataset_sha256": prepared.dataset_sha256,
        "fit_ids_sha256": sha256_obj(list(prepared.splits["fit"])),
        "eval_ids_sha256": sha256_obj(list(prepared.splits["eval"])),
    }
    assert find_matching_run(config, desc, **kwargs) is not None
    changed = config.merged({"kmeans": {"n_init": 2}})
    assert find_matching_run(changed, desc, **kwargs) is None


def test_gmm_and_pca_candidates_verify(prepared_env):
    _, config, _, prepared = prepared_env
    gmm = run_candidate(
        config, prepared, algorithm="gmm", k=2, covariance_type="diag",
        n_init=1, experiment_set="gmm_verify_probe",
    )
    assert gmm.status == "completed"
    assert verify_run(config, gmm.run_id)["status"] == "verified"
    pca = run_candidate(
        config, prepared, algorithm="kmeans", k=2, pca_variance=0.95,
        experiment_set="pca_verify_probe",
    )
    assert pca.status == "completed"
    assert verify_run(config, pca.run_id)["status"] == "verified"


def test_verify_rejects_tampered_descriptor_and_path_traversal(prepared_env):
    tmp_path, config, _, prepared = prepared_env
    candidate = run_candidate(
        config, prepared, algorithm="kmeans", k=2, seed=42,
        experiment_set="baseline_f9_standard",
    )
    source = run_dir(config, candidate.run_id)
    target = tmp_path / "artifacts" / "runs" / "tampered"
    shutil.copytree(source, target)
    doc = json.loads((target / "run.json").read_text())
    doc["run_id"] = "tampered"
    doc["scaler"] = "robust"  # bundle says standard
    (target / "run.json").write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(VerificationError, match="bundle_scaler"):
        verify_run(config, "tampered")
    with pytest.raises(VerificationError, match="Invalid run_id"):
        verify_run(config, "../escape")


def test_refit_uses_the_holdout_validated_frozen_run(prepared_env):
    _, config, _, prepared = prepared_env
    desc = descriptor()
    first = run_candidate(
        config, prepared, algorithm="kmeans", k=2, seed=42,
        experiment_set="baseline_f9_standard",
    )
    selected_config = _selection_config(config, desc)
    holdout = validate_holdout(selected_config, prepared)
    second = run_candidate(
        config, prepared, algorithm="kmeans", k=2, seed=42,
        experiment_set="post_holdout_newer_candidate",
    )
    assert second.run_id != first.run_id
    refit = refit_all_tracks(selected_config, prepared, "selected_final")
    final_doc = load_run_doc(selected_config, refit["run_id"])
    assert final_doc["evaluation_model_id"] == first.run_id
    assert final_doc["evaluation_model_id"] == holdout["frozen_run_id"]
    assert final_doc["evaluation_model_id"] != second.run_id


def test_gmm_final_refit_verifies_with_final_n_init(prepared_env):
    _, config, _, prepared = prepared_env
    config = config.merged({"gmm": {"n_init_final": 2}})
    desc = descriptor(algorithm="gmm", covariance="diag")
    candidate = run_candidate(
        config, prepared, algorithm="gmm", k=2, covariance_type="diag",
        n_init=1, experiment_set="gmm_final_probe",
    )
    selected_config = _selection_config(config, desc)
    validate_holdout(selected_config, prepared)
    refit = refit_all_tracks(selected_config, prepared, "selected_final")
    assert verify_run(selected_config, refit["run_id"])["status"] == "verified"
    final_doc = load_run_doc(selected_config, refit["run_id"])
    assert final_doc["evaluation_model_id"] == candidate.run_id


def test_verify_rejects_tampered_artifact_hash(prepared_env):
    tmp_path, config, _, prepared = prepared_env
    candidate = run_candidate(
        config, prepared, algorithm="kmeans", k=2, seed=42,
        experiment_set="baseline_f9_standard",
    )
    source = run_dir(config, candidate.run_id)
    target = tmp_path / "artifacts" / "runs" / "tampered_hash"
    shutil.copytree(source, target)
    doc = json.loads((target / "run.json").read_text())
    doc["run_id"] = "tampered_hash"
    (target / "run.json").write_text(json.dumps(doc), encoding="utf-8")
    figure = target / "figures" / "cluster_sizes_fit.png"
    figure.write_bytes(figure.read_bytes() + b"tamper")
    with pytest.raises(VerificationError, match="artifact_hash"):
        verify_run(config, "tampered_hash")


def test_verify_rejects_broken_final_refit_chain(prepared_env):
    tmp_path, config, _, prepared = prepared_env
    desc = descriptor()
    run_candidate(
        config, prepared, algorithm="kmeans", k=2, seed=42,
        experiment_set="baseline_f9_standard",
    )
    selected_config = _selection_config(config, desc)
    holdout = validate_holdout(selected_config, prepared)
    refit = refit_all_tracks(selected_config, prepared, "selected_final")
    source = run_dir(selected_config, refit["run_id"])
    target = tmp_path / "artifacts" / "runs" / "broken_final"
    shutil.copytree(source, target)
    doc = json.loads((target / "run.json").read_text())
    doc["run_id"] = "broken_final"
    doc["final_refit_model_id"] = "broken_final"
    # Point the refit at the holdout run instead of the candidate that the
    # holdout actually validated.
    doc["evaluation_model_id"] = holdout["run_id"]
    (target / "run.json").write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(VerificationError, match="holdout validated"):
        verify_run(selected_config, "broken_final")
