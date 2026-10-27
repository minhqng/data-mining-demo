"""Independent verification of a persisted run against its declared artifacts."""

from __future__ import annotations

import math
from collections.abc import Mapping
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from .config import Config
from .data import load_prepared, select_ids_by_hash
from .errors import VerificationError
from .evaluation import (
    ari_between_same_ids,
    gmm_information_criteria,
    inertia_from_labels,
    score_clustering,
)
from .features import FORBIDDEN_PRIMARY
from .models import load_bundle
from .utils import read_json, sha256_file, sha256_obj, validate_run_id


def _check(checks: list[dict[str, Any]], name: str, condition: bool, detail: str) -> None:
    checks.append({"check": name, "status": "PASS" if condition else "FAIL", "detail": detail})
    if not condition:
        raise VerificationError(f"Verification failed at {name}: {detail}")


def _compare_metric(
    checks: list[dict[str, Any]],
    name: str,
    stored: Any,
    recomputed: Any,
    *,
    stored_reason: str | None = None,
    rel_tol: float = 1e-6,
    abs_tol: float = 1e-6,
) -> None:
    if stored is None:
        _check(
            checks,
            f"metric:{name}:stored_null",
            bool(stored_reason) and recomputed is None,
            "stored metric is null; it must carry a non-empty reason and the independent "
            "recomputation must also be undefined",
        )
        return
    if recomputed is None:
        _check(checks, f"metric:{name}", False, "metric could be recomputed but stored value exists")
        return
    ok = math.isclose(float(stored), float(recomputed), rel_tol=rel_tol, abs_tol=abs_tol)
    _check(
        checks,
        f"metric:{name}",
        ok,
        f"stored={float(stored):.12g}, recomputed={float(recomputed):.12g}",
    )


def _check_cluster_sizes(
    checks: list[dict[str, Any]],
    name: str,
    stored_sizes: list[Mapping[str, Any]] | list[dict[str, Any]],
    assignments: pd.DataFrame,
    expected_ids: set[str],
    expected_k: int,
) -> None:
    stored = {int(row["cluster"]): int(row["count"]) for row in stored_sizes}
    observed_series = (
        assignments.loc[assignments["track_id"].isin(expected_ids), "cluster"]
        .value_counts()
        .sort_index()
    )
    observed = {int(cluster): int(count) for cluster, count in observed_series.items()}
    expected = {cluster: observed.get(cluster, 0) for cluster in range(expected_k)}
    _check(
        checks,
        name,
        stored == expected,
        f"stored={stored}, observed={expected}",
    )


def _load_reference_doc(output_dir: Path, run_id: Any) -> dict[str, Any]:
    if not isinstance(run_id, str) or not run_id:
        raise VerificationError("A required frozen/evaluation run id is missing.")
    try:
        safe_id = validate_run_id(run_id)
    except ValueError as exc:
        raise VerificationError(str(exc)) from exc
    base = output_dir.resolve()
    target = (base / safe_id / "run.json").resolve()
    if base not in target.parents:
        raise VerificationError("Reference run path escapes the configured output directory.")
    if not target.exists():
        raise VerificationError(f"Reference run.json is missing: {target}")
    return read_json(target)


def _check_run_chain(
    checks: list[dict[str, Any]],
    config: Config,
    doc: Mapping[str, Any],
    run_type: str,
    prepared: Any,
) -> None:
    """Verify that freeze -> holdout -> refit references are mutually consistent."""
    from .experiment import descriptor_matches

    if run_type not in {"candidate", "holdout", "final_refit"}:
        return
    if run_type == "candidate":
        _check(
            checks,
            "candidate_fit_scope",
            doc.get("fit_scope") == "development_minus_fixed_eval_cohort"
            and doc.get("assignment_scope") == "development",
            "candidate fit/assignment scopes exclude eval from fitting",
        )
        return

    base_dir = Path(config.output_dir)
    if run_type == "holdout":
        frozen_descriptor = doc.get("frozen_descriptor")
        legacy_descriptor = frozen_descriptor is None
        if legacy_descriptor:
            # Older immutable runs stored the same payload under ``descriptor``
            # before ``frozen_descriptor`` was introduced.
            frozen_descriptor = doc.get("descriptor")
        _check(
            checks,
            "holdout_used_flag",
            doc.get("holdout_used") is True and doc.get("holdout_reused") is not True,
            "holdout is marked used exactly once (not a repeated use)",
        )
        _check(
            checks,
            "holdout_evaluation_model_id",
            doc.get("evaluation_model_id") == doc.get("frozen_run_id"),
            f"evaluation_model_id={doc.get('evaluation_model_id')}",
        )
        _check(
            checks,
            "holdout_descriptor_present",
            isinstance(frozen_descriptor, Mapping),
            "frozen descriptor must be recorded (legacy: descriptor field)",
        )
        _check(
            checks,
            "holdout_descriptor_hash",
            isinstance(frozen_descriptor, Mapping)
            and (
                legacy_descriptor
                or sha256_obj(frozen_descriptor) == doc.get("frozen_descriptor_sha256")
            ),
            "legacy run: descriptor used because frozen_descriptor is absent"
            if legacy_descriptor
            else "frozen_descriptor hashes to frozen_descriptor_sha256",
        )
        holdout_ids = list(prepared.splits["holdout"])
        holdout_eval_ids = select_ids_by_hash(
            holdout_ids,
            size=min(int(config.data["eval_cohort_size"]), len(holdout_ids)),
            namespace="holdout_eval",
            seed=int(config.data["eval_seed"]),
        )
        _check(
            checks,
            "holdout_n_holdout",
            int(doc.get("n_holdout", -1)) == len(holdout_ids),
            f"n_holdout={doc.get('n_holdout')}, expected={len(holdout_ids)}",
        )
        _check(
            checks,
            "holdout_ids_hash",
            doc.get("holdout_ids_sha256") == sha256_obj(holdout_ids),
            "holdout ID hash matches the current split",
        )
        _check(
            checks,
            "holdout_eval_ids_hash",
            doc.get("holdout_eval_ids_sha256") == sha256_obj(holdout_eval_ids),
            "holdout eval-cohort hash matches the current split",
        )
        _check(
            checks,
            "holdout_descriptor_matches_its_run",
            isinstance(frozen_descriptor, Mapping)
            and doc.get("descriptor") == frozen_descriptor,
            "holdout descriptor equals its frozen descriptor",
        )
        ref_doc = _load_reference_doc(base_dir, doc.get("frozen_run_id"))
        _check(
            checks,
            "holdout_frozen_run_type",
            ref_doc.get("run_type") == "candidate" and ref_doc.get("status") == "completed",
            f"frozen run type/status={ref_doc.get('run_type')}/{ref_doc.get('status')}",
        )
        _check(
            checks,
            "holdout_frozen_descriptor_matches",
            isinstance(frozen_descriptor, Mapping)
            and descriptor_matches(ref_doc, frozen_descriptor),
            "frozen candidate descriptor matches the holdout descriptor",
        )
        for field in ("dataset_sha256", "split_sha256", "fit_ids_sha256", "eval_ids_sha256"):
            _check(
                checks,
                f"holdout_frozen_run_{field}",
                ref_doc.get(field) == doc.get(field),
                f"{field}={ref_doc.get(field)}",
            )
        return

    # final_refit
    descriptor = doc.get("descriptor")
    frozen_descriptor = doc.get("frozen_descriptor")
    legacy_descriptor = frozen_descriptor is None
    if legacy_descriptor:
        frozen_descriptor = descriptor
    _check(
        checks,
        "final_refit_model_id",
        doc.get("final_refit_model_id") == doc.get("run_id"),
        "final_refit_model_id equals this run",
    )
    _check(
        checks,
        "final_refit_holdout_used",
        doc.get("holdout_used") is True,
        "full refit records that the holdout was already used",
    )
    _check(
        checks,
        "final_refit_descriptor_present",
        isinstance(descriptor, Mapping) and isinstance(frozen_descriptor, Mapping),
        "descriptor and frozen descriptor must be recorded (legacy: descriptor field)",
    )
    _check(
        checks,
        "final_refit_descriptor_equals_frozen",
        descriptor == frozen_descriptor,
        "selection did not change after freeze",
    )
    _check(
        checks,
        "final_refit_descriptor_hash",
        isinstance(frozen_descriptor, Mapping)
        and (
            legacy_descriptor
            or sha256_obj(frozen_descriptor) == doc.get("frozen_descriptor_sha256")
        ),
        "legacy run: descriptor used because frozen_descriptor is absent"
        if legacy_descriptor
        else "frozen_descriptor hashes to frozen_descriptor_sha256",
    )
    all_ids = prepared.all_ids
    _check(
        checks,
        "final_refit_scope",
        doc.get("fit_scope") == "all_unique_tracks"
        and doc.get("assignment_scope") == "all_unique_tracks",
        "full refit scope is all unique tracks",
    )
    _check(
        checks,
        "final_refit_n_fit",
        int(doc.get("n_fit", -1)) == len(all_ids),
        f"n_fit={doc.get('n_fit')}, expected={len(all_ids)}",
    )
    _check(
        checks,
        "final_refit_fit_ids_hash",
        doc.get("fit_ids_sha256") == sha256_obj(sorted(all_ids)),
        "fit ID hash covers every unique track",
    )
    verification_ids = select_ids_by_hash(
        all_ids,
        int(config.data["eval_cohort_size"]),
        "final_verification",
        int(config.data["eval_seed"]),
    )
    _check(
        checks,
        "final_refit_n_eval",
        int(doc.get("n_eval", -1)) == len(verification_ids),
        f"n_eval={doc.get('n_eval')}, expected={len(verification_ids)}",
    )
    holdout_doc = _load_reference_doc(base_dir, doc.get("holdout_run_id"))
    _check(
        checks,
        "final_refit_holdout_type",
        holdout_doc.get("run_type") == "holdout"
        and holdout_doc.get("status") == "completed",
        f"holdout type/status={holdout_doc.get('run_type')}/{holdout_doc.get('status')}",
    )
    _check(
        checks,
        "final_refit_holdout_frozen_run",
        holdout_doc.get("frozen_run_id") == doc.get("evaluation_model_id"),
        "holdout validated the exact evaluation model cited by the refit",
    )
    _check(
        checks,
        "final_refit_holdout_descriptor",
        isinstance(descriptor, Mapping)
        and (
            holdout_doc.get("frozen_descriptor") or holdout_doc.get("descriptor")
        )
        == descriptor,
        "holdout and refit share the same frozen descriptor",
    )
    for field in ("dataset_sha256", "split_sha256"):
        _check(
            checks,
            f"final_refit_holdout_{field}",
            holdout_doc.get(field) == doc.get(field),
            f"{field}={holdout_doc.get(field)}",
        )
    ref_doc = _load_reference_doc(base_dir, doc.get("evaluation_model_id"))
    _check(
        checks,
        "final_refit_evaluation_model_type",
        ref_doc.get("run_type") == "candidate" and ref_doc.get("status") == "completed",
        f"evaluation model type/status={ref_doc.get('run_type')}/{ref_doc.get('status')}",
    )
    _check(
        checks,
        "final_refit_evaluation_model_descriptor",
        isinstance(descriptor, Mapping) and descriptor_matches(ref_doc, descriptor),
        "evaluation model descriptor matches the refit descriptor",
    )
    for field in ("dataset_sha256", "split_sha256"):
        _check(
            checks,
            f"final_refit_evaluation_model_{field}",
            ref_doc.get(field) == doc.get(field),
            f"{field}={ref_doc.get(field)}",
        )


def verify_run(config: Config, run_id: str) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    try:
        safe_run_id = validate_run_id(run_id)
    except ValueError as exc:
        raise VerificationError(str(exc)) from exc
    base = Path(config.output_dir).resolve()
    path = (base / safe_run_id).resolve()
    if base not in path.parents:
        raise VerificationError("Run path escapes the configured output directory.")
    run_json_path = path / "run.json"
    _check(checks, "run_directory", path.exists(), str(path))
    _check(checks, "run_json_exists", run_json_path.exists(), str(run_json_path))
    doc = read_json(run_json_path)
    _check(checks, "run_id_matches", doc.get("run_id") == run_id, str(doc.get("run_id")))
    _check(checks, "status_completed", doc.get("status") == "completed",
           f"status={doc.get('status')!r} (failed runs are not verified as successes)")
    run_type = doc.get("run_type")
    _check(checks, "run_type_known", run_type in {"candidate", "final_refit", "holdout", "stability", "ward"},
           str(run_type))
    reference_doc = dict(doc)
    if run_type in {"holdout", "final_refit"}:
        reference_id = doc.get("frozen_run_id") or doc.get("evaluation_model_id")
        base_doc = _load_reference_doc(Path(config.output_dir), reference_id)
        # Fill only fields absent from older run schemas. A final refit's own
        # n_init/reg_covar (e.g. n_init_final) must not be replaced by the
        # candidate's quick-sweep values.
        for key in ("n_init", "reg_covar"):
            if reference_doc.get(key) is None:
                reference_doc[key] = base_doc.get(key)

    # Config snapshot and immutable data identities.
    snapshot = path / "config.snapshot.yaml"
    _check(checks, "config_snapshot_exists", snapshot.exists(), str(snapshot))
    _check(
        checks,
        "config_snapshot_hash",
        sha256_file(snapshot) == doc.get("config_snapshot_sha256"),
        "config.snapshot.yaml matches run.json",
    )
    snapshot_config = Config.from_snapshot(snapshot)
    _check(
        checks,
        "config_canonical_hash",
        snapshot_config.sha256() == doc.get("config_sha256"),
        "snapshot content hashes to run.json config_sha256",
    )
    prepared = load_prepared(config, enforce_license=False)
    _check(
        checks,
        "raw_dataset_hash",
        prepared.dataset_sha256 == doc.get("dataset_sha256"),
        f"dataset={prepared.dataset_sha256}",
    )
    _check(
        checks,
        "processed_hash",
        prepared.processed_sha256 == doc.get("processed_sha256"),
        f"processed={prepared.processed_sha256}",
    )
    _check(
        checks,
        "split_hash",
        prepared.summary.get("split_sha256") == doc.get("split_sha256"),
        "current split files match the run's split hash",
    )
    for artifact in doc.get("artifacts", []):
        artifact_path = (path / artifact["path"]).resolve()
        if path.resolve() not in artifact_path.parents:
            _check(
                checks,
                f"artifact_containment:{artifact['path']}",
                False,
                "artifact path escapes the run directory",
            )
        if not artifact_path.exists():
            _check(checks, f"artifact_exists:{artifact['path']}", False, "missing artifact")
        actual = sha256_file(artifact_path)
        _check(
            checks,
            f"artifact_hash:{artifact['path']}",
            actual == artifact["sha256"],
            "sha256 matches run manifest",
        )
    metrics_path = path / "metrics.json"
    _check(checks, "metrics_exists", metrics_path.exists(), str(metrics_path))
    metrics_doc = read_json(metrics_path)
    _check(
        checks,
        "metrics_self_hash",
        sha256_file(metrics_path) == doc.get("metrics_sha256"),
        "metrics.json matches run.json metrics_sha256",
    )
    _check(
        checks,
        "metrics_schema",
        metrics_doc.get("schema", "").startswith("btl11.metrics."),
        str(metrics_doc.get("schema")),
    )

    # Split membership of the model run.
    fit_ids = list(prepared.splits["fit"])
    eval_ids = list(prepared.splits["eval"])
    holdout_ids = list(prepared.splits["holdout"])
    all_ids = prepared.all_ids
    _check_run_chain(checks, config, doc, run_type, prepared)
    if run_type in {"candidate", "holdout", "final_refit"}:
        assignment_path = path / "assignments.parquet"
        _check(checks, "assignments_exists", assignment_path.exists(), str(assignment_path))
        assignments = pd.read_parquet(assignment_path)
        _check(
            checks,
            "assignments_schema",
            {"track_id", "cluster"} <= set(assignments.columns),
            str(list(assignments.columns)),
        )
        assignments["track_id"] = assignments["track_id"].astype(str)
        _check(
            checks,
            "assignments_unique_ids",
            not assignments["track_id"].duplicated().any(),
            "one row per track_id",
        )
        if run_type == "candidate":
            expected_ids = set(prepared.splits["dev"])
            _check(checks, "candidate_assignments_scope", set(assignments["track_id"]) == expected_ids,
                   f"assignments={len(assignments)}, dev={len(expected_ids)}")
            _check(
                checks,
                "candidate_fit_ids_hash",
                sha256_obj(fit_ids) == doc.get("fit_ids_sha256"),
                "fit cohort unchanged",
            )
            _check(
                checks,
                "candidate_eval_ids_hash",
                sha256_obj(eval_ids) == doc.get("eval_ids_sha256"),
                "eval cohort unchanged",
            )
        elif run_type == "holdout":
            _check(checks, "holdout_assignments_scope",
                   set(assignments["track_id"]) == set(holdout_ids),
                   f"assignments={len(assignments)}, holdout={len(holdout_ids)}")
        else:
            _check(checks, "final_refit_assignments_scope",
                   set(assignments["track_id"]) == set(all_ids),
                   f"assignments={len(assignments)}, all={len(all_ids)}")
        _check(
            checks,
            "assignments_checksum",
            sha256_file(assignment_path) == doc.get("checksums", {}).get("assignments_sha256"),
            "assignments.parquet matches run.json",
        )
        model_path = path / "model.joblib"
        _check(checks, "model_exists", model_path.exists(), str(model_path))
        _check(
            checks,
            "model_checksum",
            sha256_file(model_path) == doc.get("checksums", {}).get("model_sha256"),
            "model.joblib matches run.json",
        )
        bundle = load_bundle(model_path)
        feature_names = list(doc.get("feature_order") or [])
        _check(checks, "feature_order_declared", len(feature_names) in {5, 9}, str(feature_names))
        _check(checks, "no_metadata_in_features",
               not (set(feature_names) & set(FORBIDDEN_PRIMARY)),
               "no genre/ID/metadata in the model feature list")
        _check(checks, "bundle_features_match_run",
               list(bundle.feature_names) == feature_names,
               "bundle feature order equals run.json")
        _check(checks, "bundle_algorithm_k",
               bundle.algorithm == doc.get("algorithm") and int(bundle.k) == int(doc.get("k")),
               f"{bundle.algorithm} K={bundle.k}")
        _check(checks, "bundle_pca_components",
               bundle.pca_actual_components == doc.get("pca_actual_components"),
               f"{bundle.pca_actual_components}")
        expected_fit_hash = doc.get("fit_ids_sha256")
        _check(checks, "bundle_training_ids",
               bundle.trained_on_ids_sha256 == expected_fit_hash,
               "bundle trained_on hash equals run fit_ids_sha256")
        _check(checks, "bundle_scaler", bundle.scaler == doc.get("scaler"),
               f"bundle={bundle.scaler}, doc={doc.get('scaler')}")
        doc_pca = doc.get("pca_variance")
        pca_ok = (bundle.pca_variance is None and doc_pca is None) or (
            bundle.pca_variance is not None
            and doc_pca is not None
            and math.isclose(float(bundle.pca_variance), float(doc_pca), abs_tol=1e-12)
        )
        _check(checks, "bundle_pca_variance", pca_ok,
               f"bundle={bundle.pca_variance}, doc={doc_pca}")
        _check(checks, "bundle_covariance", bundle.covariance_type == doc.get("covariance_type"),
               f"bundle={bundle.covariance_type}, doc={doc.get('covariance_type')}")
        declared_space = doc.get("feature_space_id") or metrics_doc.get("context", {}).get(
            "feature_space_id"
        )
        _check(checks, "bundle_feature_space", bundle.feature_space_id == declared_space,
               f"bundle={bundle.feature_space_id}, doc={declared_space}")
        _check(checks, "bundle_n_fit", int(bundle.n_fit) == int(doc.get("n_fit")),
               f"bundle={bundle.n_fit}, doc={doc.get('n_fit')}")
        _check(checks, "bundle_seed", int(bundle.seed) == int(doc.get("seed")),
               f"bundle={bundle.seed}, doc={doc.get('seed')}")
        # Estimator hyperparameters must match the *snapshot used at fit time*.
        estimator = bundle.estimator
        if bundle.algorithm == "kmeans":
            expected_kmeans = snapshot_config.kmeans
            _check(checks, "estimator_kmeans_params",
                   int(estimator.n_clusters) == int(doc["k"])
                   and estimator.init == expected_kmeans["init"]
                   and int(estimator.n_init) == int(reference_doc.get("n_init"))
                   and int(estimator.max_iter) == int(expected_kmeans["max_iter"])
                   and math.isclose(float(estimator.tol), float(expected_kmeans["tol"])),
                   "KMeans params match run.json/snapshot")
        else:
            expected_gmm = snapshot_config.gmm
            reg_covar_doc = reference_doc.get("reg_covar")
            _check(checks, "estimator_gmm_params",
                   int(estimator.n_components) == int(doc["k"])
                   and estimator.covariance_type == doc.get("covariance_type")
                   and int(estimator.n_init) == int(reference_doc.get("n_init"))
                   and int(estimator.max_iter) == int(expected_gmm["max_iter"])
                   and estimator.init_params == expected_gmm["init_params"]
                   and (
                       (reg_covar_doc is None and math.isclose(float(estimator.reg_covar), float(expected_gmm["reg_covar_quick"])))
                       or (
                           reg_covar_doc is not None
                           and math.isclose(float(estimator.reg_covar), float(reg_covar_doc), abs_tol=1e-15)
                       )
                   ),
                   "GMM params match run.json/snapshot")
        k = int(doc["k"])
        _check(checks, "cluster_ids_in_range",
               assignments["cluster"].between(0, k - 1).all(),
               f"clusters within [0, {k - 1}]")
        if run_type == "candidate":
            cohort_ids = eval_ids
            cohort_file = path / "eval_ids.csv"
            _check(checks, "eval_file_exists", cohort_file.exists(), str(cohort_file))
            stored_cohort = pd.read_csv(cohort_file)["track_id"].astype(str).tolist()
            _check(checks, "eval_file_matches_split", stored_cohort == eval_ids,
                   "stored eval cohort equals split eval IDs")
            if "cluster_sizes_eval" in metrics_doc:
                _check_cluster_sizes(
                    checks,
                    "eval_cluster_sizes_reconcile",
                    metrics_doc["cluster_sizes_eval"],
                    assignments,
                    set(eval_ids),
                    k,
                )
        elif run_type == "holdout":
            cohort_ids = select_ids_by_hash(
                holdout_ids,
                size=min(int(config.data["eval_cohort_size"]), len(holdout_ids)),
                namespace="holdout_eval",
                seed=int(config.data["eval_seed"]),
            )
            cohort_file = path / "eval_ids.csv"
            _check(checks, "holdout_eval_file_exists", cohort_file.exists(), str(cohort_file))
            stored_cohort = pd.read_csv(cohort_file)["track_id"].astype(str).tolist()
            _check(checks, "holdout_eval_cohort_matches", stored_cohort == cohort_ids,
                   "holdout eval cohort is reproducible")
            if "cluster_sizes_holdout" in metrics_doc:
                _check_cluster_sizes(
                    checks,
                    "holdout_cluster_sizes_reconcile",
                    metrics_doc["cluster_sizes_holdout"],
                    assignments,
                    set(holdout_ids),
                    k,
                )
        else:
            cohort_ids = select_ids_by_hash(
                all_ids,
                int(config.data["eval_cohort_size"]),
                "final_verification",
                int(config.data["eval_seed"]),
            )
            cohort_file = path / "verification_ids.csv"
            _check(checks, "verification_file_exists", cohort_file.exists(), str(cohort_file))
            stored_cohort = pd.read_csv(cohort_file)["track_id"].astype(str).tolist()
            _check(checks, "verification_cohort_matches", stored_cohort == cohort_ids,
                   "final verification cohort is reproducible")
            if "cluster_sizes_all" in metrics_doc:
                _check_cluster_sizes(
                    checks,
                    "final_cluster_sizes_reconcile",
                    metrics_doc["cluster_sizes_all"],
                    assignments,
                    set(all_ids),
                    k,
                )
            if not (path / "profiles.csv").exists():
                _check(checks, "final_profiles_exist", False, "profiles.csv missing")
            _check(checks, "final_profiles_exist", (path / "profiles.csv").exists(),
                   "profiles.csv exists")
            if not (path / "tables" / "genre_cluster_incidence.csv").exists():
                _check(checks, "final_genre_table_exists", False,
                       "post-hoc genre incidence missing")
            _check(checks, "final_genre_table_exists",
                   (path / "tables" / "genre_cluster_incidence.csv").exists(),
                   "post-hoc genre incidence exists")

        # Recompute predictions and validity metrics from raw bytes + fitted bundle.
        X_cohort = prepared.feature_matrix(cohort_ids, feature_names)
        predicted = bundle.predict(X_cohort)
        stored_labels = (
            assignments.set_index("track_id").reindex(cohort_ids)["cluster"].to_numpy(dtype=int)
        )
        _check(
            checks,
            "predictions_match_assignments",
            np.array_equal(predicted, stored_labels),
            "re-prediction reproduces saved labels on the same cohort",
        )
        Xt_cohort = bundle.transform(X_cohort)
        recomputed = score_clustering(
            Xt_cohort,
            stored_labels,
            evaluation_ids=cohort_ids,
            metric_space_id=bundle.feature_space_id,
            expected_k=k,
        )
        stored_metrics = metrics_doc.get("metrics", {})
        for metric in ("silhouette", "davies_bouldin", "calinski_harabasz"):
            entry = stored_metrics.get(metric, {})
            _compare_metric(
                checks,
                metric,
                entry.get("value") if isinstance(entry, Mapping) else None,
                recomputed[metric]["value"],
                stored_reason=entry.get("reason") if isinstance(entry, Mapping) else None,
            )
            _check(
                checks,
                f"metric_space:{metric}",
                isinstance(entry, Mapping)
                and entry.get("space") == recomputed[metric]["space"],
                "metric space identifier matches",
            )
        _check(
            checks,
            "evaluation_ids_hash",
            recomputed["evaluation_ids_sha256"]
            == metrics_doc.get("context", {}).get("evaluation_ids_sha256"),
            "evaluation cohort hash matches",
        )
        # Recompute one fit-space statistic independently, not just assignment metrics.
        if bundle.algorithm == "kmeans":
            inertia_entry = stored_metrics.get("inertia_fit")
            if isinstance(inertia_entry, Mapping) and inertia_entry.get("value") is not None:
                X_fit_raw = prepared.feature_matrix(fit_ids, feature_names)
                Xt_fit = bundle.transform(X_fit_raw)
                fit_labels = np.asarray(bundle.estimator.predict(Xt_fit), dtype=np.int64)
                recomputed_inertia = inertia_from_labels(
                    Xt_fit, fit_labels, np.asarray(bundle.estimator.cluster_centers_)
                )
                _compare_metric(
                    checks,
                    "inertia_fit",
                    inertia_entry.get("value"),
                    recomputed_inertia,
                    stored_reason=inertia_entry.get("reason"),
                )
        else:
            bic_entry = stored_metrics.get("bic")
            aic_entry = stored_metrics.get("aic")
            if (
                isinstance(bic_entry, Mapping)
                and bic_entry.get("value") is not None
                and isinstance(aic_entry, Mapping)
                and aic_entry.get("value") is not None
            ):
                X_fit_raw = prepared.feature_matrix(fit_ids, feature_names)
                Xt_fit = bundle.transform(X_fit_raw)
                info = gmm_information_criteria(
                    bundle.estimator,
                    Xt_fit,
                    k=k,
                    d=int(Xt_fit.shape[1]),
                    covariance_type=str(doc.get("covariance_type")),
                )
                _compare_metric(
                    checks, "bic", bic_entry.get("value"), info["bic"],
                    stored_reason=bic_entry.get("reason"),
                )
                _compare_metric(
                    checks, "aic", aic_entry.get("value"), info["aic"],
                    stored_reason=aic_entry.get("reason"),
                )
                n_param_entry = stored_metrics.get("n_parameters")
                if isinstance(n_param_entry, Mapping) and n_param_entry.get("value") is not None:
                    _compare_metric(
                        checks, "n_parameters", n_param_entry.get("value"),
                        float(info["n_parameters"]), stored_reason=n_param_entry.get("reason"),
                    )
        if bundle.is_gmm:
            probabilities = bundle.predict_proba(X_cohort[: min(200, len(cohort_ids))])
            _check(
                checks,
                "gmm_row_sums_one",
                np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-8),
                "posterior rows sum to one on a sample",
            )
            sample_labels = np.argmax(probabilities, axis=1)
            _check(
                checks,
                "gmm_argmax_labels",
                np.array_equal(sample_labels, predicted[: len(sample_labels)]),
                "hard labels equal argmax(predict_proba)",
            )
    elif run_type == "stability":
        summary_path = path / "tables" / "stability_summary.csv"
        pairs_path = path / "tables" / "ari_pairs.csv"
        _check(checks, "stability_summary_exists", summary_path.exists(), str(summary_path))
        _check(checks, "stability_pairs_exists", pairs_path.exists(), str(pairs_path))
        summary = pd.read_csv(summary_path)
        pairs = pd.read_csv(pairs_path)
        _check(checks, "stability_reference_n", int(summary["n_reference_ids"].iloc[0]) > 0,
               "reference cohort non-empty")
        _check(
            checks,
            "stability_ari_range",
            pairs["ari"].dropna().between(-1.0000001, 1.0000001).all(),
            "ARI values within [-1, 1]",
        )
        label_files = sorted((path / "tables").glob("stability_labels_*.csv"))
        _check(checks, "stability_labels_exist", len(label_files) >= 1,
               f"{len(label_files)} label tables")
        for label_file in label_files:
            frame = pd.read_csv(label_file)
            ids = frame["track_id"].astype(str).tolist()
            base_columns = [column for column in frame.columns if column.endswith("/base")]
            _check(
                checks,
                f"stability_base_column:{label_file.stem}",
                len(base_columns) == 1,
                f"base columns={base_columns}",
            )
            if len(base_columns) != 1:
                continue
            base_column = base_columns[0]
            finalist = base_column[: -len("/base")]
            summary_rows = summary.loc[summary["finalist"] == finalist]
            _check(
                checks,
                f"stability_summary_row:{label_file.stem}",
                len(summary_rows) == 1,
                f"summary rows for finalist {finalist!r}: {len(summary_rows)}",
            )
            if len(summary_rows) != 1:
                continue
            row = summary_rows.iloc[0]
            seed_columns = [column for column in frame.columns if "/seed=" in column]
            bootstrap_columns = [
                column for column in frame.columns if "/bootstrap=" in column
            ]
            seed_values = [
                ari_between_same_ids(frame[base_column], frame[column], ids, ids)
                for column in seed_columns
            ]
            bootstrap_values = [
                ari_between_same_ids(frame[base_column], frame[column], ids, ids)
                for column in bootstrap_columns
            ]
            if seed_values:
                _compare_metric(
                    checks,
                    f"stability_seed_ari_mean:{label_file.stem}",
                    row["seed_ari_mean"],
                    float(np.mean(seed_values)),
                )
            if bootstrap_values:
                _compare_metric(
                    checks,
                    f"stability_bootstrap_ari_mean:{label_file.stem}",
                    row["bootstrap_ari_mean"],
                    float(np.mean(bootstrap_values)),
                )
            reference_columns = seed_columns + bootstrap_columns
            pairwise_values = [
                ari_between_same_ids(frame[left], frame[right], ids, ids)
                for left, right in combinations(reference_columns, 2)
            ]
            if pairwise_values:
                _compare_metric(
                    checks,
                    f"stability_pairwise_ari_mean:{label_file.stem}",
                    row["pairwise_ari_mean"],
                    float(np.mean(pairwise_values)),
                )
            _check(
                checks,
                f"stability_reference_n:{label_file.stem}",
                int(row["n_reference_ids"]) == len(ids),
                f"n_reference_ids={row['n_reference_ids']}, labels={len(ids)}",
            )
    else:  # ward
        ward_assignments = path / "tables" / "ward_assignments.csv"
        _check(checks, "ward_assignments_exists", ward_assignments.exists(), str(ward_assignments))
        frame = pd.read_csv(ward_assignments)
        sample_ids = frame["track_id"].astype(str).tolist()
        _check(checks, "ward_ids_unique", not frame["track_id"].duplicated().any(),
               "sample IDs unique")
        _check(
            checks,
            "ward_sample_within_fit_ids",
            set(sample_ids) <= set(prepared.splits["fit"]),
            "Ward sample is a subset of the development fit IDs",
        )
        _check(
            checks,
            "ward_n_sample",
            len(sample_ids) == int(doc.get("n_eval", -1)),
            f"n_eval={doc.get('n_eval')}, rows={len(sample_ids)}",
        )
        _check(
            checks,
            "ward_sample_hash",
            doc.get("fit_ids_sha256") == sha256_obj(sorted(sample_ids)),
            "Ward sample ID hash matches run.json",
        )
        _check(
            checks,
            "ward_metrics_n_sample",
            int(metrics_doc.get("context", {}).get("n_sample", -1)) == len(sample_ids),
            "metrics context n_sample matches the assignment table",
        )
        _check(checks, "ward_k", frame["ward_cluster"].nunique() == int(doc["k"]),
               "Ward cluster count matches declared K")
        recomputed_ari = float(
            adjusted_rand_score(frame["ward_cluster"], frame["selected_model_cluster"])
        )
        entry = metrics_doc.get("metrics", {}).get("ari_vs_selected_model_same_sample", {})
        _compare_metric(checks, "ward_ari_vs_selected", entry.get("value"), recomputed_ari)
        _check(checks, "ward_figure_exists",
               (path / "figures" / "ward_dendrogram.png").exists(),
               "dendrogram exists")
        descriptor = doc.get("descriptor")
        _check(
            checks,
            "ward_descriptor_present",
            isinstance(descriptor, Mapping),
            "Ward run records the selected descriptor",
        )
        from .experiment import descriptor_matches

        selected_doc = _load_reference_doc(
            Path(config.output_dir),
            doc.get("selected_run_id") or doc.get("evaluation_model_id"),
        )
        _check(
            checks,
            "ward_selected_run",
            selected_doc.get("status") == "completed"
            and selected_doc.get("run_type") == "candidate"
            and isinstance(descriptor, Mapping)
            and descriptor_matches(selected_doc, descriptor),
            "Ward references a completed candidate with the same descriptor",
        )
    _check(
        checks,
        "license_metadata_present",
        bool(doc.get("license_status")),
        f"license_status={doc.get('license_status')}",
    )
    return {
        "run_id": run_id,
        "run_type": run_type,
        "status": "verified",
        "n_checks": len(checks),
        "checks": checks,
        "path": str(path),
    }
