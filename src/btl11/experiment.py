"""Run lifecycle, staged sweeps, ablations, stability, holdout and final refit."""

from __future__ import annotations

import json
import time
import traceback
import warnings
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering

from .config import Config
from .data import PreparedData, select_ids_by_hash
from .errors import (
    ArtifactError,
    ConfigError,
    RunError,
    SchemaError,
)
from .evaluation import (
    ari_between_same_ids,
    cluster_size_table,
    cohort_identifier,
    gmm_information_criteria,
    inertia_from_labels,
    pairwise_ari,
    rank_bic_within_space,
    score_clustering,
)
from .features import feature_names_for_set
from .models import (
    assignments_frame,
    fit_clustering_bundle,
    gmm_uncertainty,
    load_bundle,
    save_bundle,
    write_bundle_metadata,
)
from .utils import (
    allocate_run_dir,
    atomic_write_csv,
    atomic_write_json,
    atomic_write_parquet,
    atomic_write_text,
    git_info,
    host_info,
    library_versions,
    new_run_id,
    peak_rss_mib,
    read_json,
    safe_slug,
    sha256_file,
    sha256_obj,
    utc_now,
    validate_run_id,
)
from .visualization import (
    plot_ablation,
    plot_cluster_sizes,
    plot_gmm_information,
    plot_stability,
    plot_sweep_elbow,
    plot_sweep_metrics,
)

LIBRARIES = (
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "matplotlib",
    "seaborn",
    "pyarrow",
    "joblib",
)
INDEX_COLUMNS = [
    "run_id",
    "created_utc",
    "run_type",
    "experiment_set",
    "algorithm",
    "k",
    "covariance_type",
    "feature_set",
    "scaler",
    "pca_variance",
    "pca_actual_components",
    "seed",
    "status",
    "n_fit",
    "n_eval",
    "silhouette",
    "davies_bouldin",
    "calinski_harabasz",
    "bic",
    "aic",
    "model_path",
    "metrics_path",
    "config_sha256",
    "dataset_sha256",
    "experiment_note",
]


# --------------------------------------------------------------------------- #
# Run discovery / lifecycle
# --------------------------------------------------------------------------- #
@dataclass
class RunResult:
    run_id: str
    path: Path
    status: str
    metrics: dict[str, Any]
    reused: bool = False
    doc: dict[str, Any] | None = None


def iter_run_docs(config: Config) -> list[tuple[Path, dict[str, Any]]]:
    output = Path(config.output_dir)
    if not output.exists():
        return []
    docs: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(output.glob("*/run.json")):
        try:
            docs.append((path.parent, read_json(path)))
        except (OSError, json.JSONDecodeError):
            continue
    return docs


def run_dir(config: Config, run_id: str) -> Path:
    try:
        safe_id = validate_run_id(run_id)
    except ValueError as exc:
        raise ArtifactError(str(exc)) from exc
    base = Path(config.output_dir).resolve()
    path = (base / safe_id).resolve()
    if base not in path.parents:
        raise ArtifactError("Run path escapes the configured output directory.")
    if not path.exists():
        raise ArtifactError(
            f"Run {run_id!r} not found under {config.output_dir}. Use artifacts/INDEX.csv "
            "to list actual runs."
        )
    return path


def load_run_doc(config: Config, run_id: str) -> dict[str, Any]:
    path = run_dir(config, run_id) / "run.json"
    if not path.exists():
        raise ArtifactError(f"run.json missing for run {run_id}.")
    return read_json(path)


def descriptor_matches(doc: Mapping[str, Any], descriptor: Mapping[str, Any]) -> bool:
    if doc.get("algorithm") != descriptor.get("algorithm"):
        return False
    if int(doc.get("k", -1)) != int(descriptor.get("k", -2)):
        return False
    if doc.get("feature_set") != descriptor.get("feature_set"):
        return False
    if doc.get("scaler") != descriptor.get("scaler"):
        return False
    doc_pca = doc.get("pca_variance")
    wanted_pca = descriptor.get("pca")
    if (doc_pca is None) != (wanted_pca is None):
        return False
    if doc_pca is not None and abs(float(doc_pca) - float(wanted_pca)) > 1e-12:
        return False
    if int(doc.get("seed", -1)) != int(descriptor.get("seed", -2)):
        return False
    return not (
        descriptor.get("algorithm") == "gmm"
        and doc.get("covariance_type") != descriptor.get("covariance_type")
    )


def _snapshot_matches_model_config(path: Path, config: Config) -> bool:
    """Reuse a run only if model-relevant hyperparameters did not change.

    Selection fields legitimately change after runs are created; kmeans/gmm
    settings and PCA whitening do not.
    """
    try:
        snapshot = yaml.safe_load((path / "config.snapshot.yaml").read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return False
    if not isinstance(snapshot, dict):
        return False
    if snapshot.get("kmeans") != config.raw.get("kmeans"):
        return False
    if snapshot.get("gmm") != config.raw.get("gmm"):
        return False
    snapshot_preprocess = snapshot.get("preprocess") or {}
    return snapshot_preprocess.get("pca_whiten") == config.preprocess.get("pca_whiten")


def find_matching_run(
    config: Config,
    descriptor: Mapping[str, Any],
    *,
    run_type: str = "candidate",
    dataset_sha256: str | None = None,
    fit_ids_sha256: str | None = None,
    eval_ids_sha256: str | None = None,
) -> RunResult | None:
    matches: list[tuple[Path, dict[str, Any]]] = []
    for path, doc in iter_run_docs(config):
        if doc.get("status") != "completed" or doc.get("run_type") != run_type:
            continue
        if not descriptor_matches(doc, descriptor):
            continue
        if dataset_sha256 and doc.get("dataset_sha256") != dataset_sha256:
            continue
        if fit_ids_sha256 and doc.get("fit_ids_sha256") != fit_ids_sha256:
            continue
        if eval_ids_sha256 and doc.get("eval_ids_sha256") != eval_ids_sha256:
            continue
        if not _snapshot_matches_model_config(path, config):
            continue
        matches.append((path, doc))
    if not matches:
        return None
    matches.sort(key=lambda item: item[1].get("created_utc", ""), reverse=True)
    path, doc = matches[0]
    metrics_path = path / "metrics.json"
    metrics = read_json(metrics_path) if metrics_path.exists() else {}
    return RunResult(
        run_id=doc["run_id"],
        path=path,
        status=doc["status"],
        metrics=metrics,
        reused=True,
        doc=doc,
    )


def update_index(config: Config, doc: Mapping[str, Any], metrics: Mapping[str, Any] | None) -> Path:
    index_path = Path("artifacts/INDEX.csv")
    index_path.parent.mkdir(parents=True, exist_ok=True)
    summary = (metrics or {}).get("metrics", {}) if metrics else {}
    row = {
        "run_id": doc.get("run_id"),
        "created_utc": doc.get("created_utc"),
        "run_type": doc.get("run_type"),
        "experiment_set": doc.get("experiment_set"),
        "algorithm": doc.get("algorithm"),
        "k": doc.get("k"),
        "covariance_type": doc.get("covariance_type"),
        "feature_set": doc.get("feature_set"),
        "scaler": doc.get("scaler"),
        "pca_variance": doc.get("pca_variance"),
        "pca_actual_components": doc.get("pca_actual_components"),
        "seed": doc.get("seed"),
        "status": doc.get("status"),
        "n_fit": doc.get("n_fit"),
        "n_eval": doc.get("n_eval"),
        "silhouette": _metric_value(summary, "silhouette"),
        "davies_bouldin": _metric_value(summary, "davies_bouldin"),
        "calinski_harabasz": _metric_value(summary, "calinski_harabasz"),
        "bic": _metric_value(summary, "bic"),
        "aic": _metric_value(summary, "aic"),
        "model_path": str(Path(config.output_dir) / str(doc.get("run_id")) / "model.joblib")
        if doc.get("has_model")
        else None,
        "metrics_path": str(Path(config.output_dir) / str(doc.get("run_id")) / "metrics.json"),
        "config_sha256": doc.get("config_sha256"),
        "dataset_sha256": doc.get("dataset_sha256"),
        "experiment_note": doc.get("experiment_note"),
    }
    if index_path.exists():
        frame = pd.read_csv(index_path)
        frame = frame[frame["run_id"].astype(str) != str(doc.get("run_id"))]
    else:
        frame = pd.DataFrame(columns=INDEX_COLUMNS)
    frame = pd.concat([frame, pd.DataFrame([row])], ignore_index=True)
    for column in INDEX_COLUMNS:
        if column not in frame.columns:
            frame[column] = None
    frame = frame.loc[:, INDEX_COLUMNS].sort_values(["created_utc", "run_id"])
    atomic_write_csv(index_path, frame)
    return index_path


def _metric_value(metrics: Mapping[str, Any], name: str) -> Any:
    entry = metrics.get(name)
    if isinstance(entry, Mapping):
        return entry.get("value")
    return None


class RunSession:
    """Create an immutable run directory and honestly journal its lifecycle."""

    def __init__(
        self,
        config: Config,
        *,
        run_type: str,
        tag: str,
        prepared: PreparedData,
        base_fields: Mapping[str, Any],
    ) -> None:
        self.config = config
        self.run_type = run_type
        self.prepared = prepared
        self.config_sha256 = config.sha256()
        self.started_utc = utc_now()
        base = new_run_id(self.config_sha256, prepared.dataset_sha256, tag)
        self.run_id, self.path = allocate_run_dir(config.output_dir, base)
        (self.path / "tables").mkdir()
        (self.path / "figures").mkdir()
        snapshot_path = self.path / "config.snapshot.yaml"
        atomic_write_text(snapshot_path, config.snapshot_yaml())
        self.snapshot_sha256 = sha256_file(snapshot_path)
        self.doc: dict[str, Any] = {
            "run_id": self.run_id,
            "run_type": run_type,
            "status": "running",
            "created_utc": self.started_utc,
            "started_utc": self.started_utc,
            "finished_utc": None,
            "config_sha256": self.config_sha256,
            "config_snapshot": "config.snapshot.yaml",
            "config_snapshot_sha256": self.snapshot_sha256,
            "dataset_sha256": prepared.dataset_sha256,
            "processed_sha256": prepared.processed_sha256,
            "split_sha256": prepared.summary.get("split_sha256"),
            "license_status": prepared.manifest.get("license_review_status"),
            "license_operational_status": config.license_operational_status(
                prepared.manifest
            ),
            "python_version": host_info()["python_version"],
            "library_versions": library_versions(LIBRARIES),
            "host": host_info(),
            "git": git_info(),
            "warnings": [],
            "has_model": False,
            "artifacts": [],
            **dict(base_fields),
        }
        atomic_write_json(self.path / "run.json", self.doc)
        update_index(config, self.doc, None)
        self._completed = False

    def complete(
        self,
        *,
        metrics: Mapping[str, Any],
        updates: Mapping[str, Any],
    ) -> dict[str, Any]:
        metrics_path = atomic_write_json(self.path / "metrics.json", metrics)
        doc = {
            **self.doc,
            **dict(updates),
            "status": "completed",
            "finished_utc": utc_now(),
            "metrics_path": "metrics.json",
            "metrics_sha256": sha256_file(metrics_path),
        }
        artifacts = _artifact_manifest(self.path)
        doc["artifacts"] = artifacts
        doc["summary_metrics"] = {
            name: _metric_value(metrics.get("metrics", {}), name)
            for name in (
                "silhouette",
                "davies_bouldin",
                "calinski_harabasz",
                "bic",
                "aic",
                "inertia_fit",
                "inertia_eval",
            )
        }
        atomic_write_json(self.path / "run.json", doc)
        self.doc = doc
        self._completed = True
        update_index(self.config, doc, metrics)
        return doc

    def fail(self, exc: BaseException) -> dict[str, Any]:
        if self._completed:
            return self.doc
        error_text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        atomic_write_text(self.path / "error.log", error_text)
        doc = {
            **self.doc,
            "status": "failed",
            "finished_utc": utc_now(),
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "error_log": "error.log",
            "artifacts": _artifact_manifest(self.path),
        }
        atomic_write_json(self.path / "run.json", doc)
        self.doc = doc
        update_index(self.config, doc, None)
        return doc


def _artifact_manifest(path: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for file in sorted(path.rglob("*")):
        if not file.is_file() or file.name in {"run.json"}:
            continue
        entries.append(
            {
                "path": str(file.relative_to(path)),
                "sha256": sha256_file(file),
                "bytes": file.stat().st_size,
            }
        )
    return entries


# --------------------------------------------------------------------------- #
# Candidate fits
# --------------------------------------------------------------------------- #
def _metric_entry(
    value: float | None,
    *,
    direction: str,
    space: str,
    n: int,
    method: str,
    reason: str | None = None,
) -> dict[str, Any]:
    if value is not None and not np.isfinite(float(value)):
        value = None
        reason = "non_finite_metric"
    return {
        "value": None if value is None else float(value),
        "direction": direction,
        "space": space,
        "n_eval": int(n),
        "method": method,
        "reason": reason if value is None else None,
    }


def _captured_warnings(caught: Iterable[warnings.WarningMessage]) -> list[dict[str, str]]:
    return [
        {"category": entry.category.__name__, "message": str(entry.message)}
        for entry in caught
    ]


def _candidate_tag(
    algorithm: str,
    k: int,
    feature_set: str,
    scaler: str,
    pca_variance: float | None,
    covariance_type: str | None,
    seed: int,
) -> str:
    pca_label = "nopca" if pca_variance is None else f"pca{int(float(pca_variance) * 100)}"
    cov = f"_cov-{covariance_type}" if covariance_type else ""
    return safe_slug(
        f"{algorithm}_{feature_set}_{scaler}_{pca_label}{cov}_k{k}_s{seed}",
        max_length=80,
    )


def _feature_names(config: Config, feature_set: str) -> list[str]:
    return feature_names_for_set(
        feature_set, config.feature_primary, config.feature_ablation
    )


def run_candidate(
    config: Config,
    prepared: PreparedData,
    *,
    algorithm: str,
    k: int,
    feature_set: str = "f9",
    scaler: str = "standard",
    pca_variance: float | None = None,
    seed: int | None = None,
    covariance_type: str | None = None,
    n_init: int | None = None,
    reg_covar: float | None = None,
    experiment_set: str,
    experiment_note: str = "",
    reuse: bool = False,
) -> RunResult:
    """Fit one candidate on fit IDs and evaluate it on the shared cohort."""
    seed = int(seed if seed is not None else config.seed)
    if algorithm == "gmm":
        covariance_type = covariance_type or "diag"
    feature_names = _feature_names(config, feature_set)
    fit_ids = list(prepared.splits["fit"])
    eval_ids = list(prepared.splits["eval"])
    descriptor = {
        "algorithm": algorithm,
        "k": int(k),
        "covariance_type": covariance_type if algorithm == "gmm" else None,
        "feature_set": feature_set,
        "scaler": scaler,
        "pca": pca_variance,
        "seed": seed,
    }
    fit_sha = sha256_obj(fit_ids)
    eval_sha = sha256_obj(eval_ids)
    if reuse and algorithm != "gmm":
        existing = find_matching_run(
            config,
            descriptor,
            run_type="candidate",
            dataset_sha256=prepared.dataset_sha256,
            fit_ids_sha256=fit_sha,
            eval_ids_sha256=eval_sha,
        )
        if existing is not None:
            return existing
    if reuse and algorithm == "gmm":
        existing = find_matching_run(
            config,
            descriptor,
            run_type="candidate",
            dataset_sha256=prepared.dataset_sha256,
            fit_ids_sha256=fit_sha,
            eval_ids_sha256=eval_sha,
        )
        if existing is not None:
            return existing

    tag = _candidate_tag(algorithm, k, feature_set, scaler, pca_variance, covariance_type, seed)
    session = RunSession(
        config,
        run_type="candidate",
        tag=tag,
        prepared=prepared,
        base_fields={
            "experiment_set": experiment_set,
            "experiment_note": experiment_note or None,
            "algorithm": algorithm,
            "k": int(k),
            "covariance_type": covariance_type if algorithm == "gmm" else None,
            "feature_set": feature_set,
            "feature_order": feature_names,
            "scaler": scaler,
            "pca_variance": None if pca_variance is None else float(pca_variance),
            "pca_actual_components": None,
            "seed": seed,
            "n_init": None,
            "reg_covar": reg_covar,
            "n_fit": len(fit_ids),
            "n_eval": len(eval_ids),
            "fit_ids_sha256": fit_sha,
            "eval_ids_sha256": eval_sha,
            "fit_scope": "development_minus_fixed_eval_cohort",
            "assignment_scope": "development",
            "evaluation_model_id": None,
            "final_refit_model_id": None,
            "holdout_used": False,
        },
    )
    try:
        overall_started = time.perf_counter()
        X_fit = prepared.feature_matrix(fit_ids, feature_names)
        X_eval = prepared.feature_matrix(eval_ids, feature_names)
        started = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            bundle, Xt_fit = fit_clustering_bundle(
                X_fit,
                fit_ids,
                algorithm=algorithm,
                k=k,
                config=config,
                seed=seed,
                feature_names=feature_names,
                scaler=scaler,
                pca_variance=pca_variance,
                covariance_type=covariance_type,
                n_init=n_init,
                reg_covar=reg_covar,
            )
            fit_seconds = time.perf_counter() - started
            Xt_eval = bundle.transform(X_eval)
            labels_fit = np.asarray(bundle.estimator.predict(Xt_fit), dtype=np.int64)
            labels_eval = np.asarray(bundle.estimator.predict(Xt_eval), dtype=np.int64)
            probabilities_eval = (
                bundle.predict_proba(X_eval) if bundle.is_gmm else None
            )
        eval_score = score_clustering(
            Xt_eval,
            labels_eval,
            evaluation_ids=eval_ids,
            metric_space_id=bundle.feature_space_id,
            expected_k=int(k),
        )
        sizes_fit = cluster_size_table(labels_fit, expected_k=int(k))
        sizes_eval = cluster_size_table(labels_eval, expected_k=int(k))
        empty_clusters_fit = int((sizes_fit["count"] == 0).sum())
        empty_clusters_eval = int((sizes_eval["count"] == 0).sum())
        min_share_fit = float(sizes_fit["share"].min())
        max_share_fit = float(sizes_fit["share"].max())
        space = bundle.feature_space_id
        metrics_map: dict[str, Any] = {
            **{name: eval_score[name] for name in ("silhouette", "davies_bouldin", "calinski_harabasz")},
        }
        converged = False
        n_iter = None
        gmm_details: dict[str, Any] = {}
        uncertainty_summary: dict[str, Any] = {}
        if algorithm == "kmeans":
            centers = np.asarray(bundle.estimator.cluster_centers_, dtype=np.float64)
            inertia_fit = float(bundle.estimator.inertia_)
            inertia_eval = inertia_from_labels(Xt_eval, labels_eval, centers)
            n_iter = int(bundle.estimator.n_iter_)
            converged = bool(n_iter < int(config.kmeans["max_iter"]))
            metrics_map["inertia_fit"] = _metric_entry(
                inertia_fit,
                direction="minimize",
                space=space,
                n=len(fit_ids),
                method="sklearn KMeans.inertia_ (fit cohort)",
            )
            metrics_map["inertia_eval"] = _metric_entry(
                inertia_eval,
                direction="minimize",
                space=space,
                n=len(eval_ids),
                method="sum of squared distances to assigned centroid (eval cohort)",
            )
        else:
            assert probabilities_eval is not None
            uncertainty = gmm_uncertainty(probabilities_eval)
            if not np.array_equal(uncertainty["hard_labels"], labels_eval):
                raise SchemaError("GMM hard labels differ from argmax(predict_proba).")
            weights = np.asarray(bundle.estimator.weights_, dtype=np.float64)
            info = gmm_information_criteria(
                bundle.estimator,
                Xt_fit,
                k=int(k),
                d=int(Xt_fit.shape[1]),
                covariance_type=str(bundle.covariance_type),
            )
            converged = bool(info["converged"])
            n_iter = int(info["n_iter"])
            if bool(config.gmm["require_convergence"]) and not converged:
                raise RunError(
                    f"GMM K={k} covariance={covariance_type} did not converge within "
                    f"{config.gmm['max_iter']} iterations; gmm.require_convergence=true "
                    "forbids accepting this fit."
                )
            mean_ll_eval = float(bundle.estimator.score(Xt_eval))
            metrics_map["bic"] = _metric_entry(
                info["bic"], direction="minimize", space=space, n=len(fit_ids),
                method="GaussianMixture.bic on fit cohort; within-space only",
            )
            metrics_map["aic"] = _metric_entry(
                info["aic"], direction="minimize", space=space, n=len(fit_ids),
                method="GaussianMixture.aic on fit cohort; within-space only",
            )
            metrics_map["mean_log_likelihood_fit"] = _metric_entry(
                info["mean_log_likelihood"], direction="maximize", space=space,
                n=len(fit_ids), method="GaussianMixture.score on fit cohort",
            )
            metrics_map["mean_log_likelihood_eval"] = _metric_entry(
                mean_ll_eval, direction="maximize", space=space, n=len(eval_ids),
                method="GaussianMixture.score on fixed eval cohort",
            )
            metrics_map["n_parameters"] = _metric_entry(
                float(info["n_parameters"]), direction="context", space=space,
                n=len(fit_ids), method="closed-form GMM free-parameter count",
            )
            gmm_details = {
                **info,
                "weights": weights.tolist(),
                "min_weight": float(weights.min()),
                "max_weight": float(weights.max()),
                "near_empty_components_weight_lt_1e-3": int((weights < 1e-3).sum()),
                "mean_log_likelihood_eval": mean_ll_eval,
            }
            uncertainty_summary = {
                "confidence_mean": float(uncertainty["confidence"].mean()),
                "confidence_p10": float(np.quantile(uncertainty["confidence"], 0.10)),
                "confidence_min": float(uncertainty["confidence"].min()),
                "normalized_entropy_mean": float(uncertainty["normalized_entropy"].mean()),
                "normalized_entropy_p90": float(
                    np.quantile(uncertainty["normalized_entropy"], 0.90)
                ),
                "borderline_definition": "lowest 10% confidence within this eval cohort",
            }
            metrics_map["gmm_confidence_mean"] = _metric_entry(
                uncertainty_summary["confidence_mean"],
                direction="context",
                space=space,
                n=len(eval_ids),
                method="mean max_k responsibilities on fixed eval cohort",
            )
            metrics_map["gmm_normalized_entropy_mean"] = _metric_entry(
                uncertainty_summary["normalized_entropy_mean"],
                direction="minimize",
                space=space,
                n=len(eval_ids),
                method="mean normalized posterior entropy on fixed eval cohort",
            )
        health = {
            "empty_clusters_fit": empty_clusters_fit,
            "empty_clusters_eval": empty_clusters_eval,
            "min_cluster_share_fit": min_share_fit,
            "max_cluster_share_fit": max_share_fit,
            "min_cluster_share_eval": eval_score["min_cluster_share"],
            "max_cluster_share_eval": eval_score["max_cluster_share"],
            "degenerate_fit_warning": bool(empty_clusters_fit > 0 or min_share_fit < 0.01),
            "tiny_cluster_threshold_share": 0.01,
            "note": (
                "Health flags are diagnostic; no composite quality score is computed."
            ),
        }
        metrics_map["min_cluster_share_fit"] = _metric_entry(
            min_share_fit, direction="maximize", space="raw cluster sizes on fit cohort",
            n=len(fit_ids), method="min cluster share",
        )
        metrics_map["empty_clusters_fit"] = _metric_entry(
            float(empty_clusters_fit), direction="minimize",
            space="raw cluster sizes on fit cohort", n=len(fit_ids),
            method="count of expected clusters with zero members",
        )
        runtime = {
            "fit_seconds": float(fit_seconds),
            "total_seconds": float(time.perf_counter() - overall_started),
            "peak_rss_mib_process": peak_rss_mib(),
            "peak_rss_note": "process high-water mark, not an isolated per-fit measurement",
        }
        # Persist the model and assignments after successful fitting.
        model_path = save_bundle(bundle, session.path / "model.joblib")
        write_bundle_metadata(bundle, session.path / "model_metadata.json")
        dev_ids = list(prepared.splits["dev"])
        X_dev = prepared.feature_matrix(dev_ids, feature_names)
        dev_proba = bundle.predict_proba(X_dev) if bundle.is_gmm else None
        dev_labels = np.asarray(bundle.estimator.predict(bundle.transform(X_dev)), dtype=np.int64)
        assignments = assignments_frame(dev_ids, dev_labels, probabilities=dev_proba)
        assignments_path = atomic_write_parquet(session.path / "assignments.parquet", assignments)
        if algorithm == "gmm":
            dev_uncertainty = gmm_uncertainty(dev_proba) if dev_proba is not None else None
            if dev_uncertainty is not None:
                contributions = {
                    "confidence_mean_dev": float(dev_uncertainty["confidence"].mean()),
                    "normalized_entropy_mean_dev": float(
                        dev_uncertainty["normalized_entropy"].mean()
                    ),
                }
            else:  # pragma: no cover - defensive
                contributions = {}
        else:
            contributions = {}
        atomic_write_csv(session.path / "tables" / "cluster_sizes_fit.csv", sizes_fit)
        atomic_write_csv(session.path / "tables" / "cluster_sizes_eval.csv", sizes_eval)
        atomic_write_csv(session.path / "eval_ids.csv", pd.DataFrame({"track_id": eval_ids}))
        one_row = {
            "run_id": session.run_id,
            "algorithm": algorithm,
            "k": int(k),
            "covariance_type": covariance_type if algorithm == "gmm" else None,
            "feature_set": feature_set,
            "scaler": scaler,
            "pca_variance": pca_variance,
            "feature_space_id": space,
            "silhouette": metrics_map["silhouette"]["value"],
            "davies_bouldin": metrics_map["davies_bouldin"]["value"],
            "calinski_harabasz": metrics_map["calinski_harabasz"]["value"],
            "bic": metrics_map.get("bic", {}).get("value"),
            "aic": metrics_map.get("aic", {}).get("value"),
            "inertia_fit": metrics_map.get("inertia_fit", {}).get("value"),
            "inertia_eval": metrics_map.get("inertia_eval", {}).get("value"),
            "min_cluster_share_fit": min_share_fit,
            "empty_clusters_fit": empty_clusters_fit,
            "converged": converged,
            "fit_seconds": fit_seconds,
            "peak_rss_mib_process": runtime["peak_rss_mib_process"],
        }
        atomic_write_csv(session.path / "tables" / "candidate_metrics.csv", pd.DataFrame([one_row]))
        pca_info = None
        if bundle.pca_actual_components is not None:
            from .preprocess import pca_diagnostics, pca_loadings_frame, pca_variance_frame

            pca_info = pca_diagnostics(bundle.transformer, feature_names)
            variance_frame = pca_variance_frame(pca_info)
            loadings = pca_loadings_frame(pca_info)
            atomic_write_csv(session.path / "tables" / "pca_explained_variance.csv", variance_frame)
            atomic_write_csv(session.path / "tables" / "pca_loadings.csv", loadings)
        plot_cluster_sizes(
            sizes_fit,
            session.path / "figures" / "cluster_sizes_fit.png",
            title=(
                f"{algorithm.upper()} K={k} | {space} | fit n={len(fit_ids)} | "
                f"run {session.run_id}"
            ),
        )
        warnings_list = _captured_warnings(caught)
        metrics_doc = {
            "schema": "btl11.metrics.candidate.v1",
            "run_id": session.run_id,
            "context": {
                "dataset_sha256": prepared.dataset_sha256,
                "processed_sha256": prepared.processed_sha256,
                "split_sha256": prepared.summary.get("split_sha256"),
                "fit_ids_sha256": fit_sha,
                "eval_ids_sha256": eval_sha,
                "evaluation_ids_sha256": eval_score["evaluation_ids_sha256"],
                "feature_space_id": space,
                "feature_order": feature_names,
                "algorithm": algorithm,
                "k": int(k),
                "covariance_type": covariance_type if algorithm == "gmm" else None,
                "scaler": scaler,
                "pca_variance": pca_variance,
                "pca_actual_components": bundle.pca_actual_components,
                "seed": seed,
                "n_fit": len(fit_ids),
                "n_eval": len(eval_ids),
                "fit_scope": "development_minus_fixed_eval_cohort",
                "metric_scope": "fixed evaluation cohort excluded from fit",
            },
            "metrics": metrics_map,
            "cluster_sizes_fit": sizes_fit.to_dict(orient="records"),
            "cluster_sizes_eval": sizes_eval.to_dict(orient="records"),
            "health": health,
            "gmm": gmm_details or None,
            "gmm_uncertainty": uncertainty_summary or None,
            "assignment_uncertainty_dev": contributions or None,
            "runtime": runtime,
            "warnings": warnings_list,
            "notes": [
                "Validity metrics use one fixed ID cohort excluded from fitting.",
                "Silhouette/DB/CH are contextual to the shown feature space.",
                "No composite quality score is fabricated.",
            ],
        }
        runtime["total_seconds"] = float(time.perf_counter() - overall_started)
        updates = {
            "feature_space_id": space,
            "pca_actual_components": bundle.pca_actual_components,
            "n_init": int(
                n_init
                or (
                    config.kmeans["n_init"]
                    if algorithm == "kmeans"
                    else config.gmm["n_init_quick"]
                )
            ),
            "converged": converged,
            "n_iter": n_iter,
            "warnings": warnings_list,
            "warnings_count": len(warnings_list),
            "health": health,
            "gmm": gmm_details or None,
            "gmm_uncertainty": uncertainty_summary or None,
            "pca_diagnostics": pca_info,
            "summary_metrics": {
                "silhouette": metrics_map["silhouette"]["value"],
                "davies_bouldin": metrics_map["davies_bouldin"]["value"],
                "calinski_harabasz": metrics_map["calinski_harabasz"]["value"],
                "bic": metrics_map.get("bic", {}).get("value"),
                "aic": metrics_map.get("aic", {}).get("value"),
            },
            "has_model": True,
            "descriptor": descriptor,
            "model_path": "model.joblib",
            "assignments_path": "assignments.parquet",
            "checksums": {
                "model_sha256": sha256_file(model_path),
                "assignments_sha256": sha256_file(assignments_path),
            },
            "fit_seconds": fit_seconds,
            "peak_rss_mib": runtime["peak_rss_mib_process"],
        }
        doc = session.complete(metrics=metrics_doc, updates=updates)
        return RunResult(
            run_id=session.run_id,
            path=session.path,
            status=doc["status"],
            metrics=metrics_doc,
            doc=doc,
        )
    except Exception as exc:
        session.fail(exc)
        raise


# --------------------------------------------------------------------------- #
# Baseline sweeps
# --------------------------------------------------------------------------- #
def build_sweep_jobs(
    config: Config,
    algorithm: str,
    *,
    feature_set: str = "f9",
    scaler: str | None = None,
    pca_variance: float | None = None,
) -> list[dict[str, Any]]:
    scaler = scaler or str(config.preprocess["scaler_primary"])
    covariances: list[str | None] = (
        list(config.gmm["covariance_candidates"]) if algorithm == "gmm" else [None]
    )
    jobs: list[dict[str, Any]] = []
    for seed in config.search["quick_seeds"]:
        for k in config.search["quick_k"]:
            for covariance in covariances:
                jobs.append(
                    {
                        "algorithm": algorithm,
                        "k": int(k),
                        "covariance_type": covariance,
                        "feature_set": feature_set,
                        "scaler": scaler,
                        "pca_variance": pca_variance,
                        "seed": int(seed),
                        "n_init": (
                            int(config.gmm["n_init_quick"])
                            if algorithm == "gmm"
                            else int(config.kmeans["n_init"])
                        ),
                        "reg_covar": (
                            float(config.gmm["reg_covar_quick"])
                            if algorithm == "gmm"
                            else None
                        ),
                    }
                )
    return jobs


def dry_run_sweep(config: Config, algorithm: str) -> dict[str, Any]:
    jobs = build_sweep_jobs(config, algorithm)
    return {
        "algorithm": algorithm,
        "n_jobs": len(jobs),
        "jobs": jobs,
        "fit_count_limit": int(config.search["max_fit_count_warning"]),
        "data_required": str(config.raw_path),
        "prepared_required": "data/processed/tracks.parquet",
        "note": "Dry run only: no directory, raw file or result is created.",
    }


def run_sweep(
    config: Config,
    prepared: PreparedData,
    algorithm: str,
    *,
    allow_many_fits: bool = False,
) -> dict[str, Any]:
    if algorithm not in {"kmeans", "gmm"}:
        raise ConfigError("sweep --algorithm must be kmeans or gmm.")
    jobs = build_sweep_jobs(config, algorithm)
    if len(jobs) > int(config.search["max_fit_count_warning"]) and not allow_many_fits:
        raise RunError(
            f"Sweep would run {len(jobs)} fits, above the configured warning threshold "
            f"{config.search['max_fit_count_warning']}. Use a staged design or explicitly "
            "pass --allow-many-fits."
        )
    experiment_set = str(config.selection["baseline_experiment_set"])
    sweep_time = utc_now().replace(":", "").replace("-", "")
    sweep_dir = Path("artifacts/sweeps") / f"{experiment_set}_{algorithm}_{sweep_time}"
    sweep_dir.mkdir(parents=True, exist_ok=False)
    results: list[RunResult] = []
    failures: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for index, job in enumerate(jobs, start=1):
        print(
            f"[{index}/{len(jobs)}] {algorithm} K={job['k']} "
            f"cov={job['covariance_type']} seed={job['seed']}",
            flush=True,
        )
        try:
            result = run_candidate(
                config,
                prepared,
                algorithm=job["algorithm"],
                k=job["k"],
                feature_set=job["feature_set"],
                scaler=job["scaler"],
                pca_variance=job["pca_variance"],
                seed=job["seed"],
                covariance_type=job["covariance_type"],
                n_init=job["n_init"],
                reg_covar=job["reg_covar"],
                experiment_set=experiment_set,
                experiment_note=f"baseline sweep {algorithm}",
            )
            results.append(result)
            row = _result_row(result)
            rows.append(row)
        except Exception as exc:  # preserve the failure in the evidence trail
            failures.append(
                {
                    "job": job,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            print(f"  FAILED: {type(exc).__name__}: {exc}", flush=True)
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values(["k", "covariance_type"], na_position="first")
        atomic_write_csv(sweep_dir / "candidates.csv", frame)
    if not failures:
        atomic_write_csv(
            sweep_dir / "failures.csv",
            pd.DataFrame(columns=["algorithm", "k", "covariance_type", "error_type", "error"]),
        )
    else:
        failure_rows = [
            {
                "algorithm": failure["job"]["algorithm"],
                "k": failure["job"]["k"],
                "covariance_type": failure["job"]["covariance_type"],
                "error_type": failure["error_type"],
                "error": failure["error"],
            }
            for failure in failures
        ]
        atomic_write_csv(sweep_dir / "failures.csv", pd.DataFrame(failure_rows))
    if not frame.empty:
        try:
            if algorithm == "kmeans":
                plot_sweep_elbow(frame, sweep_dir / "elbow.png")
                plot_sweep_metrics(frame, sweep_dir / "kmeans_metrics.png")
            else:
                plot_gmm_information(frame, sweep_dir / "gmm_bic_aic.png")
        except Exception as exc:  # plotting failure must be visible, not fatal to evidence
            failures.append(
                {"job": {"algorithm": algorithm, "k": None, "covariance_type": None},
                 "error_type": type(exc).__name__, "error": f"plotting failed: {exc}"}
            )
    summary = {
        "sweep_id": sweep_dir.name,
        "created_utc": utc_now(),
        "algorithm": algorithm,
        "experiment_set": experiment_set,
        "config_sha256": config.sha256(),
        "dataset_sha256": prepared.dataset_sha256,
        "n_jobs": len(jobs),
        "n_completed": len(results),
        "n_failed": len(failures),
        "run_ids": [result.run_id for result in results],
        "failures": failures,
        "baseline_fit_note": (
            "Fits use development-minus-eval IDs; metrics use the fixed eval cohort. "
            "Raw BIC is never ranked across different feature spaces."
        ),
    }
    atomic_write_json(sweep_dir / "sweep.json", summary)
    return summary


def _result_row(result: RunResult) -> dict[str, Any]:
    doc = result.doc or {}
    metrics = result.metrics.get("metrics", {})
    return {
        "run_id": result.run_id,
        "algorithm": doc.get("algorithm"),
        "k": doc.get("k"),
        "covariance_type": doc.get("covariance_type"),
        "feature_set": doc.get("feature_set"),
        "scaler": doc.get("scaler"),
        "pca_variance": doc.get("pca_variance"),
        "pca_actual_components": doc.get("pca_actual_components"),
        "feature_space_id": doc.get("feature_space_id")
        or result.metrics.get("context", {}).get("feature_space_id"),
        "seed": doc.get("seed"),
        "silhouette": _metric_value(metrics, "silhouette"),
        "davies_bouldin": _metric_value(metrics, "davies_bouldin"),
        "calinski_harabasz": _metric_value(metrics, "calinski_harabasz"),
        "bic": _metric_value(metrics, "bic"),
        "aic": _metric_value(metrics, "aic"),
        "inertia_fit": _metric_value(metrics, "inertia_fit"),
        "inertia_eval": _metric_value(metrics, "inertia_eval"),
        "min_cluster_share_fit": _metric_value(metrics, "min_cluster_share_fit"),
        "empty_clusters_fit": _metric_value(metrics, "empty_clusters_fit"),
        "converged": doc.get("converged"),
        "fit_seconds": doc.get("fit_seconds"),
        "status": result.status,
        "run_path": str(result.path),
    }


# --------------------------------------------------------------------------- #
# Comparison
# --------------------------------------------------------------------------- #
def compare_experiment_set(config: Config, experiment_set: str) -> dict[str, Any]:
    docs = [
        (path, doc)
        for path, doc in iter_run_docs(config)
        if doc.get("experiment_set") == experiment_set
    ]
    if not docs:
        raise ArtifactError(
            f"No runs found for experiment_set {experiment_set!r}. Run `btl11 sweep` first."
        )
    completed = [(path, doc) for path, doc in docs if doc.get("status") == "completed"]
    failed = [(path, doc) for path, doc in docs if doc.get("status") == "failed"]
    rows: list[dict[str, Any]] = []
    for path, doc in completed:
        metrics_path = path / "metrics.json"
        if not metrics_path.exists():
            continue
        metrics = read_json(metrics_path)
        rows.append(
            {
                "run_id": doc["run_id"],
                "algorithm": doc.get("algorithm"),
                "k": doc.get("k"),
                "covariance_type": doc.get("covariance_type"),
                "feature_set": doc.get("feature_set"),
                "scaler": doc.get("scaler"),
                "pca_variance": doc.get("pca_variance"),
                "pca_actual_components": doc.get("pca_actual_components"),
                "feature_space_id": doc.get("feature_space_id")
                or metrics.get("context", {}).get("feature_space_id"),
                "dataset_sha256": doc.get("dataset_sha256"),
                "fit_ids_sha256": doc.get("fit_ids_sha256"),
                "eval_ids_sha256": doc.get("eval_ids_sha256"),
                "cohort_id": cohort_identifier(
                    doc.get("dataset_sha256"),
                    doc.get("fit_ids_sha256"),
                    doc.get("eval_ids_sha256"),
                ),
                "seed": doc.get("seed"),
                "n_fit": doc.get("n_fit"),
                "n_eval": doc.get("n_eval"),
                "converged": doc.get("converged"),
                "silhouette": _metric_value(metrics.get("metrics", {}), "silhouette"),
                "davies_bouldin": _metric_value(metrics.get("metrics", {}), "davies_bouldin"),
                "calinski_harabasz": _metric_value(metrics.get("metrics", {}), "calinski_harabasz"),
                "bic": _metric_value(metrics.get("metrics", {}), "bic"),
                "aic": _metric_value(metrics.get("metrics", {}), "aic"),
                "inertia_fit": _metric_value(metrics.get("metrics", {}), "inertia_fit"),
                "inertia_eval": _metric_value(metrics.get("metrics", {}), "inertia_eval"),
                "min_cluster_share_fit": _metric_value(
                    metrics.get("metrics", {}), "min_cluster_share_fit"
                ),
                "empty_clusters_fit": _metric_value(
                    metrics.get("metrics", {}), "empty_clusters_fit"
                ),
                "fit_seconds": doc.get("fit_seconds"),
                "run_path": str(path),
            }
        )
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = rank_bic_within_space(frame)
        frame = frame.sort_values(["algorithm", "k", "covariance_type"]).reset_index(drop=True)
    comparison_time = utc_now().replace(":", "").replace("-", "")
    outdir = (
        Path("artifacts/comparisons")
        / safe_slug(experiment_set)
        / comparison_time
    )
    outdir.mkdir(parents=True, exist_ok=False)
    atomic_write_csv(outdir / "comparison.csv", frame)
    failed_frame = pd.DataFrame(
        [
            {
                "run_id": doc.get("run_id"),
                "algorithm": doc.get("algorithm"),
                "k": doc.get("k"),
                "covariance_type": doc.get("covariance_type"),
                "error_type": doc.get("error_type"),
                "error_message": doc.get("error_message"),
                "path": str(path),
            }
            for path, doc in failed
        ]
    )
    if failed_frame.empty:
        failed_frame = pd.DataFrame(
            columns=["run_id", "algorithm", "k", "covariance_type", "error_type", "error_message", "path"]
        )
    atomic_write_csv(outdir / "failed_runs.csv", failed_frame)
    if not frame.empty:
        kmeans_rows = frame[frame["algorithm"] == "kmeans"]
        gmm_rows = frame[frame["algorithm"] == "gmm"]
        try:
            if not kmeans_rows.empty:
                plot_sweep_elbow(kmeans_rows, outdir / "elbow.png")
                plot_sweep_metrics(kmeans_rows, outdir / "kmeans_metrics.png")
            if not gmm_rows.empty:
                plot_gmm_information(gmm_rows, outdir / "gmm_bic_aic.png")
        except Exception as exc:
            atomic_write_text(outdir / "plot_error.txt", str(exc))
    markdown = build_comparison_markdown(
        frame, failed_frame, experiment_set, config
    )
    atomic_write_text(outdir / "comparison.md", markdown)
    summary = {
        "experiment_set": experiment_set,
        "output_dir": str(outdir),
        "n_completed": len(completed),
        "n_failed": len(failed),
        "comparison_csv": str(outdir / "comparison.csv"),
        "comparison_md": str(outdir / "comparison.md"),
    }
    return summary


def _md_table(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    if frame.empty:
        return "_No rows._"
    subset = frame.loc[:, [column for column in columns if column in frame.columns]].copy()
    subset = subset.astype(object).where(subset.notna(), "")
    header = "| " + " | ".join(str(column) for column in subset.columns) + " |"
    separator = "| " + " | ".join("---" for _ in subset.columns) + " |"
    lines = [header, separator]
    for _, row in subset.iterrows():
        values = []
        for value in row.tolist():
            text = f"{value:.6g}" if isinstance(value, float) else str(value)
            values.append(text.replace("|", "\\|").replace("\n", " "))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def build_comparison_markdown(
    frame: pd.DataFrame,
    failed: pd.DataFrame,
    experiment_set: str,
    config: Config,
) -> str:
    lines = [
        f"# So sánh experiment set `{experiment_set}`",
        "",
        f"- Dataset SHA-256: `{config.raw_path}` (xem MANIFEST)",
        "- Tất cả metric hợp lệ dùng cùng một fixed eval cohort (ID tách khỏi fit).",
        "- **Không xếp hạng BIC thô giữa các feature space/scaler/PCA khác nhau.** "
        "`bic_rank_within_space` chỉ có nghĩa trong cùng `feature_space_id`.",
        "- Silhouette/DB/CH phụ thuộc không gian đo; so sánh có bối cảnh.",
        "",
    ]
    if frame.empty:
        lines.append("Không có run completed.")
    else:
        kmeans = frame[frame["algorithm"] == "kmeans"]
        gmm = frame[frame["algorithm"] == "gmm"]
        lines += ["## K-Means", "", _md_table(kmeans, [
            "run_id", "k", "feature_space_id", "seed", "silhouette",
            "davies_bouldin", "calinski_harabasz", "inertia_fit", "inertia_eval",
            "min_cluster_share_fit", "empty_clusters_fit", "converged",
        ]), ""]
        lines += ["## GMM (BIC chỉ so trong cùng representation + cohort)", "", _md_table(gmm, [
            "run_id", "k", "covariance_type", "feature_space_id", "cohort_id", "seed", "bic",
            "bic_rank_within_space", "aic", "silhouette", "davies_bouldin",
            "calinski_harabasz", "converged",
        ]), ""]
    if not failed.empty:
        lines += [
            "## Failed runs (không bị ẩn khỏi lịch sử)",
            "",
            _md_table(failed, ["run_id", "algorithm", "k", "covariance_type", "error_type", "error_message"]),
            "",
        ]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# Ablations
# --------------------------------------------------------------------------- #
def ablate(
    config: Config,
    prepared: PreparedData,
    factor: str,
    *,
    reuse_existing: bool = True,
) -> dict[str, Any]:
    if factor not in {"scaler", "feature_set", "pca"}:
        raise ConfigError("ablate --factor must be scaler, feature_set or pca.")
    shortlist = [int(k) for k in config.selection["shortlist_k"]]
    if not shortlist:
        raise ConfigError(
            "selection.shortlist_k is empty. Run the baseline sweeps, inspect the comparison, "
            "then set a small evidence-based shortlist in configs/base.yaml."
        )
    algorithms = list(config.selection["ablation_algorithms"])
    covariance = str(config.selection["ablation_covariance"])
    variants: list[dict[str, Any]] = []
    if factor == "scaler":
        for scaler in ("standard", "robust", "none"):
            variants.append({"scaler": scaler, "pca_variance": None, "feature_set": "f9",
                             "label": ("none_unscaled_E00_diagnostic" if scaler == "none" else scaler)})
    elif factor == "feature_set":
        variants = [
            {"scaler": "standard", "pca_variance": None, "feature_set": "f9", "label": "f9"},
            {"scaler": "standard", "pca_variance": None, "feature_set": "f5", "label": "f5"},
        ]
    else:
        variants = [
            {"scaler": "standard", "pca_variance": None, "feature_set": "f9", "label": "no_pca"},
            {"scaler": "standard", "pca_variance": 0.90, "feature_set": "f9", "label": "pca90"},
            {"scaler": "standard", "pca_variance": 0.95, "feature_set": "f9", "label": "pca95"},
        ]
    new_job_count = 0
    planned: list[dict[str, Any]] = []
    for variant in variants:
        for algorithm in algorithms:
            if factor == "scaler" and variant["scaler"] == "none" and algorithm != "kmeans":
                continue
            for k in shortlist:
                planned.append(
                    {
                        **variant,
                        "algorithm": algorithm,
                        "k": k,
                        "covariance_type": covariance if algorithm == "gmm" else None,
                        "seed": config.seed,
                    }
                )
    if reuse_existing:
        for job in planned:
            descriptor = {
                "algorithm": job["algorithm"],
                "k": job["k"],
                "covariance_type": job["covariance_type"],
                "feature_set": job["feature_set"],
                "scaler": job["scaler"],
                "pca": job["pca_variance"],
                "seed": job["seed"],
            }
            if find_matching_run(
                config,
                descriptor,
                run_type="candidate",
                dataset_sha256=prepared.dataset_sha256,
                fit_ids_sha256=sha256_obj(list(prepared.splits["fit"])),
                eval_ids_sha256=sha256_obj(list(prepared.splits["eval"])),
            ) is None:
                new_job_count += 1
    else:
        new_job_count = len(planned)
    if new_job_count > int(config.search["max_fit_count_warning"]):
        raise RunError(
            f"Ablation would add {new_job_count} fits, above the configured warning threshold. "
            "Reduce the shortlist or use a staged design."
        )
    experiment_set = f"ablation_{factor}"
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for index, job in enumerate(planned, start=1):
        descriptor = {
            "algorithm": job["algorithm"],
            "k": job["k"],
            "covariance_type": job["covariance_type"],
            "feature_set": job["feature_set"],
            "scaler": job["scaler"],
            "pca": job["pca_variance"],
            "seed": job["seed"],
        }
        print(
            f"[{index}/{len(planned)}] {factor} {job['label']} {job['algorithm']} K={job['k']}",
            flush=True,
        )
        try:
            existing = (
                find_matching_run(
                    config,
                    descriptor,
                    run_type="candidate",
                    dataset_sha256=prepared.dataset_sha256,
                    fit_ids_sha256=sha256_obj(list(prepared.splits["fit"])),
                    eval_ids_sha256=sha256_obj(list(prepared.splits["eval"])),
                )
                if reuse_existing
                else None
            )
            if existing is not None:
                result = existing
            else:
                result = run_candidate(
                    config,
                    prepared,
                    algorithm=job["algorithm"],
                    k=job["k"],
                    feature_set=job["feature_set"],
                    scaler=job["scaler"],
                    pca_variance=job["pca_variance"],
                    seed=job["seed"],
                    covariance_type=job["covariance_type"],
                    n_init=(
                        config.gmm["n_init_quick"]
                        if job["algorithm"] == "gmm"
                        else config.kmeans["n_init"]
                    ),
                    experiment_set=experiment_set,
                    experiment_note=f"ablation {factor}: {job['label']}",
                )
            row = _result_row(result)
            row["variant"] = job["label"]
            row["factor"] = factor
            row["reused_existing_run"] = bool(existing is not None)
            rows.append(row)
        except Exception as exc:
            failures.append(
                {
                    "variant": job["label"],
                    **descriptor,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            print(f"  FAILED: {type(exc).__name__}: {exc}", flush=True)
    stamp = utc_now().replace(":", "").replace("-", "")
    outdir = Path("artifacts/ablations") / factor / stamp
    outdir.mkdir(parents=True, exist_ok=False)
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values(["algorithm", "k", "variant"]).reset_index(drop=True)
    atomic_write_csv(outdir / "comparison.csv", frame)
    failure_frame = pd.DataFrame(failures)
    if failure_frame.empty:
        failure_frame = pd.DataFrame(columns=["variant", "algorithm", "k", "error_type", "error"])
    atomic_write_csv(outdir / "failures.csv", failure_frame)
    if not frame.empty:
        try:
            plot_ablation(frame, factor, outdir / "metrics_by_variant.png")
        except Exception as exc:
            atomic_write_text(outdir / "plot_error.txt", str(exc))
    lines = [
        f"# Ablation `{factor}`",
        "",
        "- Chỉ số được tính trên cùng fixed eval cohort; khác biểu diễn/scaler/PCA nên "
        "không phải đại lượng cùng thang đo tuyệt đối.",
        "- `none_unscaled_E00_diagnostic` chỉ minh họa tác động của scale (E00), "
        "không phải model được khuyến nghị và không được chọn final.",
        "- BIC chỉ có nghĩa trong cùng feature space; bảng này không xếp hạng BIC chéo.",
        "",
        _md_table(frame, [
            "variant", "algorithm", "k", "covariance_type", "feature_space_id",
            "silhouette", "davies_bouldin", "calinski_harabasz", "bic",
            "min_cluster_share_fit", "converged", "run_id", "reused_existing_run",
        ]),
        "",
    ]
    if not failure_frame.empty:
        lines += ["## Failures", "", _md_table(failure_frame, list(failure_frame.columns)), ""]
    atomic_write_text(outdir / "summary.md", "\n".join(lines))
    summary = {
        "factor": factor,
        "output_dir": str(outdir),
        "n_rows": len(rows),
        "n_failed": len(failures),
        "comparison_csv": str(outdir / "comparison.csv"),
        "summary_md": str(outdir / "summary.md"),
    }
    atomic_write_json(outdir / "ablation.json", summary)
    return summary


# --------------------------------------------------------------------------- #
# Stability
# --------------------------------------------------------------------------- #
def stability(
    config: Config,
    prepared: PreparedData,
    selector: str = "finalists",
    *,
    allow_bootstrap: bool = True,
) -> dict[str, Any]:
    if selector not in {"finalists", "frozen_finalists"}:
        raise ConfigError("stability --model-selector must be finalists/frozen_finalists.")
    descriptors = (
        list(config.selection["finalists"])
        if selector == "finalists"
        else list(config.selection["frozen"])
    )
    if not descriptors:
        raise ConfigError(
            f"selection.{selector} is empty. Populate it from the baseline comparison evidence "
            "before running stability."
        )
    fit_ids = list(prepared.splits["fit"])
    eval_ids = list(prepared.splits["eval"])
    X_fit_by_fs = {}
    X_eval_by_fs = {}
    base_runs: dict[str, RunResult] = {}
    for entry in descriptors:
        descriptor = config.descriptor(entry)
        base = find_matching_run(
            config,
            descriptor,
            run_type="candidate",
            dataset_sha256=prepared.dataset_sha256,
            fit_ids_sha256=sha256_obj(fit_ids),
            eval_ids_sha256=sha256_obj(eval_ids),
        )
        if base is None:
            raise ArtifactError(
                "No completed candidate run matches finalist descriptor "
                f"{descriptor}; run the baseline/ablation fit first."
            )
        base_runs[_descriptor_key(descriptor)] = base
        fs = descriptor["feature_set"]
        if fs not in X_fit_by_fs:
            names = _feature_names(config, fs)
            X_fit_by_fs[fs] = prepared.feature_matrix(fit_ids, names)
            X_eval_by_fs[fs] = prepared.feature_matrix(eval_ids, names)
    session = RunSession(
        config,
        run_type="stability",
        tag="stability",
        prepared=prepared,
        base_fields={
            "experiment_set": "stability",
            "algorithm": "mixed",
            "k": None,
            "covariance_type": None,
            "feature_set": "mixed",
            "feature_order": None,
            "scaler": "mixed",
            "pca_variance": None,
            "n_fit": len(fit_ids),
            "n_eval": len(eval_ids),
            "fit_ids_sha256": sha256_obj(fit_ids),
            "eval_ids_sha256": sha256_obj(eval_ids),
            "fit_scope": "development_minus_fixed_eval_cohort",
            "assignment_scope": "none",
            "evaluation_model_id": None,
            "final_refit_model_id": None,
            "holdout_used": False,
            "has_model": False,
            "finalists": [config.descriptor(entry) for entry in descriptors],
            "base_run_ids": {key: run.run_id for key, run in base_runs.items()},
        },
    )
    try:
        seeds = [int(seed) for seed in config.search["stability_seeds"]]
        n_bootstrap = int(config.selection["stability_bootstrap"]) if allow_bootstrap else 0
        reference_ids = list(eval_ids)
        run_rows: list[dict[str, Any]] = []
        pair_rows: list[dict[str, Any]] = []
        summary_rows: list[dict[str, Any]] = []
        failures: list[dict[str, Any]] = []
        for key, base in base_runs.items():
            descriptor = config.descriptor(
                next(entry for entry in descriptors if _descriptor_key(config.descriptor(entry)) == key)
            )
            base_assignments = pd.read_parquet(base.path / "assignments.parquet")
            base_assignments["track_id"] = base_assignments["track_id"].astype(str)
            base_eval = (
                base_assignments.set_index("track_id")
                .reindex(reference_ids)["cluster"]
                .to_numpy(dtype=np.int64)
            )
            if np.isnan(base_eval.astype(float)).any():
                raise ArtifactError("Base run assignments do not cover the eval cohort.")
            labels_for_key: dict[str, np.ndarray] = {}
            run_rows.append(
                {
                    "finalist": key,
                    "variant": "base-run",
                    "seed": descriptor["seed"],
                    "n_fit": len(fit_ids),
                    "fit_seconds": None,
                    "converged": True,
                    "ari_vs_base": 1.0,
                    "status": "completed",
                }
            )
            names = _feature_names(config, descriptor["feature_set"])
            for seed in seeds:
                print(f"[stability] {key} seed={seed}", flush=True)
                try:
                    fit_started = time.perf_counter()
                    bundle, _ = fit_clustering_bundle(
                        X_fit_by_fs[descriptor["feature_set"]],
                        fit_ids,
                        algorithm=descriptor["algorithm"],
                        k=descriptor["k"],
                        config=config,
                        seed=seed,
                        feature_names=names,
                        scaler=descriptor["scaler"],
                        pca_variance=descriptor["pca"],
                        covariance_type=descriptor["covariance_type"],
                        n_init=(
                            int(config.selection["stability_n_init_gmm"])
                            if descriptor["algorithm"] == "gmm"
                            else int(config.kmeans["n_init"])
                        ),
                    )
                    fit_seconds = time.perf_counter() - fit_started
                    labels = bundle.predict(X_eval_by_fs[descriptor["feature_set"]])
                    ari = ari_between_same_ids(base_eval, labels, reference_ids, reference_ids)
                    labels_for_key[f"{key}/seed={seed}"] = labels
                    run_rows.append(
                        {
                            "finalist": key,
                            "variant": "seed",
                            "seed": seed,
                            "n_fit": len(fit_ids),
                            "fit_seconds": float(fit_seconds),
                            "converged": bool(
                                getattr(bundle.estimator, "converged_", True)
                            ),
                            "ari_vs_base": float(ari),
                            "status": "completed",
                        }
                    )
                except Exception as exc:
                    failures.append(
                        {"finalist": key, "variant": "seed", "seed": seed,
                         "error_type": type(exc).__name__, "error": str(exc)}
                    )
                    run_rows.append(
                        {"finalist": key, "variant": "seed", "seed": seed,
                         "n_fit": len(fit_ids), "fit_seconds": None,
                         "converged": None, "ari_vs_base": None, "status": "failed"}
                    )
            if n_bootstrap > 0:
                fit_positions = np.arange(len(fit_ids), dtype=np.int64)
                for bootstrap_index in range(n_bootstrap):
                    print(f"[stability] {key} bootstrap={bootstrap_index + 1}/{n_bootstrap}", flush=True)
                    rng = np.random.default_rng(descriptor["seed"] + 1000 + bootstrap_index)
                    sampled = rng.choice(fit_positions, size=len(fit_positions), replace=True)
                    X_boot = X_fit_by_fs[descriptor["feature_set"]][sampled]
                    try:
                        fit_started = time.perf_counter()
                        bundle, _ = fit_clustering_bundle(
                            X_boot,
                            [fit_ids[index] for index in sampled],
                            algorithm=descriptor["algorithm"],
                            k=descriptor["k"],
                            config=config,
                            seed=descriptor["seed"] + bootstrap_index,
                            feature_names=names,
                            scaler=descriptor["scaler"],
                            pca_variance=descriptor["pca"],
                            covariance_type=descriptor["covariance_type"],
                            n_init=(
                                int(config.selection["stability_n_init_gmm"])
                                if descriptor["algorithm"] == "gmm"
                                else int(config.kmeans["n_init"])
                            ),
                        )
                        fit_seconds = time.perf_counter() - fit_started
                        labels = bundle.predict(X_eval_by_fs[descriptor["feature_set"]])
                        ari = ari_between_same_ids(base_eval, labels, reference_ids, reference_ids)
                        labels_for_key[f"{key}/bootstrap={bootstrap_index}"] = labels
                        run_rows.append(
                            {
                                "finalist": key,
                                "variant": "bootstrap",
                                "seed": descriptor["seed"] + bootstrap_index,
                                "n_fit": len(fit_ids),
                                "fit_seconds": float(fit_seconds),
                                "converged": bool(
                                    getattr(bundle.estimator, "converged_", True)
                                ),
                                "ari_vs_base": float(ari),
                                "status": "completed",
                            }
                        )
                    except Exception as exc:
                        failures.append(
                            {"finalist": key, "variant": "bootstrap",
                             "seed": descriptor["seed"] + bootstrap_index,
                             "error_type": type(exc).__name__, "error": str(exc)}
                        )
                        run_rows.append(
                            {"finalist": key, "variant": "bootstrap",
                             "seed": descriptor["seed"] + bootstrap_index,
                             "n_fit": len(fit_ids), "fit_seconds": None,
                             "converged": None, "ari_vs_base": None, "status": "failed"}
                        )
            pair_table, pair_summary = pairwise_ari(labels_for_key, reference_ids)
            pair_table.insert(0, "finalist", key)
            pair_rows.extend(pair_table.to_dict(orient="records"))
            labels_df = pd.DataFrame({"track_id": reference_ids})
            labels_df[f"{key}/base"] = base_eval
            for name, labels in labels_for_key.items():
                labels_df[name] = labels
            atomic_write_csv(
                session.path / "tables" / f"stability_labels_{safe_slug(key)}.csv",
                labels_df,
            )
            seed_values = [
                row["ari_vs_base"]
                for row in run_rows
                if row["finalist"] == key and row["variant"] == "seed" and row["ari_vs_base"] is not None
            ]
            bootstrap_values = [
                row["ari_vs_base"]
                for row in run_rows
                if row["finalist"] == key and row["variant"] == "bootstrap" and row["ari_vs_base"] is not None
            ]
            summary_rows.append(
                {
                    "finalist": key,
                    "algorithm": descriptor["algorithm"],
                    "k": descriptor["k"],
                    "covariance_type": descriptor["covariance_type"],
                    "feature_space_id": (
                        (base.doc or {}).get("feature_space_id")
                        or base.metrics.get("context", {}).get("feature_space_id")
                    ),
                    "n_seed_comparisons": len(seed_values),
                    "seed_ari_mean": float(np.mean(seed_values)) if seed_values else None,
                    "seed_ari_std": float(np.std(seed_values, ddof=0)) if seed_values else None,
                    "seed_ari_min": float(np.min(seed_values)) if seed_values else None,
                    "bootstrap_n": len(bootstrap_values),
                    "bootstrap_ari_mean": float(np.mean(bootstrap_values)) if bootstrap_values else None,
                    "bootstrap_ari_std": float(np.std(bootstrap_values, ddof=0)) if bootstrap_values else None,
                    "bootstrap_ari_min": float(np.min(bootstrap_values)) if bootstrap_values else None,
                    "pairwise_ari_mean": pair_summary["mean"],
                    "pairwise_ari_std": pair_summary["std"],
                    "n_reference_ids": pair_summary["reference_n"],
                    "reference_ids_sha256": pair_summary["reference_ids_sha256"],
                }
            )
        run_frame = pd.DataFrame(run_rows)
        pair_frame = pd.DataFrame(pair_rows)
        summary_frame = pd.DataFrame(summary_rows)
        atomic_write_csv(session.path / "tables" / "stability_runs.csv", run_frame)
        atomic_write_csv(session.path / "tables" / "ari_pairs.csv", pair_frame)
        atomic_write_csv(session.path / "tables" / "stability_summary.csv", summary_frame)
        try:
            plot_stability(summary_frame, session.path / "figures" / "stability.png")
        except Exception as exc:
            atomic_write_text(session.path / "figures" / "plot_error.txt", str(exc))
        metrics_map: dict[str, Any] = {}
        for _, row in summary_frame.iterrows():
            key = str(row["finalist"])
            metrics_map[f"{key}__seed_ari_mean"] = _metric_entry(
                row["seed_ari_mean"], direction="maximize",
                space=f"{key} fixed eval cohort", n=int(row["n_reference_ids"]),
                method="mean ARI vs base assignment across distinct random seeds",
                reason=None if row["seed_ari_mean"] is not None else "no_successful_seed_fits",
            )
            metrics_map[f"{key}__bootstrap_ari_mean"] = _metric_entry(
                row["bootstrap_ari_mean"], direction="maximize",
                space=f"{key} fixed eval cohort", n=int(row["n_reference_ids"]),
                method="mean ARI vs base assignment across bootstrap refits",
                reason=None if row["bootstrap_ari_mean"] is not None else "no_successful_bootstrap_fits",
            )
        metrics_doc = {
            "schema": "btl11.metrics.stability.v1",
            "run_id": session.run_id,
            "context": {
                "n_fit": len(fit_ids),
                "n_eval": len(eval_ids),
                "reference_ids_sha256": sha256_obj(eval_ids),
                "seeds": seeds,
                "bootstrap": n_bootstrap,
                "fit_ids_sha256": sha256_obj(fit_ids),
            },
            "metrics": metrics_map,
            "summary": summary_frame.to_dict(orient="records"),
            "failures": failures,
            "protocol": (
                "All refits predict the *same* eval ID cohort. Seed stability isolates "
                "estimator randomness; bootstrap stability resamples fit rows but never "
                "compares labels from non-identical reference rows. ARI is permutation invariant."
            ),
        }
        session.complete(
            metrics=metrics_doc,
            updates={
                "n_fits_attempted": int(
                    len(descriptors) * (len(seeds) + n_bootstrap)
                ),
                "n_fits_failed": len(failures),
                "summary_metrics": {
                    "silhouette": None,
                    "davies_bouldin": None,
                    "calinski_harabasz": None,
                    "bic": None,
                    "aic": None,
                },
                "has_model": False,
                "stability_summary": summary_frame.to_dict(orient="records"),
            },
        )
        return {
            "run_id": session.run_id,
            "path": str(session.path),
            "summary": summary_frame.to_dict(orient="records"),
            "failures": failures,
        }
    except Exception as exc:
        session.fail(exc)
        raise


def _descriptor_key(descriptor: Mapping[str, Any]) -> str:
    pca = "none" if descriptor.get("pca") is None else f"{float(descriptor['pca']):.2f}"
    cov = descriptor.get("covariance_type") or "n/a"
    return f"{descriptor['algorithm']}-k{descriptor['k']}-{descriptor['feature_set']}-{descriptor['scaler']}-{pca}-{cov}-s{descriptor['seed']}"


# --------------------------------------------------------------------------- #
# Holdout validation
# --------------------------------------------------------------------------- #
def validate_holdout(
    config: Config,
    prepared: PreparedData,
    *,
    allow_repeat: bool = False,
) -> dict[str, Any]:
    frozen = list(config.selection["frozen"])
    if len(frozen) != 1:
        raise ConfigError(
            "selection.frozen must contain exactly one signed-off winner before the one-time "
            "holdout check. Do not choose the winner by peeking at holdout metrics."
        )
    descriptor = config.descriptor(frozen[0])
    base = find_matching_run(
        config,
        descriptor,
        run_type="candidate",
        dataset_sha256=prepared.dataset_sha256,
        fit_ids_sha256=sha256_obj(list(prepared.splits["fit"])),
        eval_ids_sha256=sha256_obj(list(prepared.splits["eval"])),
    )
    if base is None:
        raise ArtifactError(
            "No completed candidate run matches the frozen descriptor. Refit/run it before "
            "holdout validation."
        )
    previous_doc = _find_holdout_for_descriptor(config, prepared, descriptor)
    if previous_doc is not None and not allow_repeat:
        raise ConfigError(
            f"Holdout was already used for this frozen descriptor at "
            f"{previous_doc.get('run_id')}. The SSOT requires a one-time check; creating a "
            "new candidate with the same descriptor does not reset it. Passing --allow-repeat "
            "would contaminate its interpretation and is recorded only if explicitly requested."
        )
    base_doc = base.doc or load_run_doc(config, base.run_id)
    model_path = base.path / "model.joblib"
    expected_model_sha = base_doc.get("checksums", {}).get("model_sha256")
    if not model_path.exists() or sha256_file(model_path) != expected_model_sha:
        raise ArtifactError("Frozen model hash mismatch; refusing to reuse a changed artifact.")
    bundle = load_bundle(model_path)
    holdout_ids = list(prepared.splits["holdout"])
    holdout_eval_ids = select_ids_by_hash(
        holdout_ids,
        size=min(int(config.data["eval_cohort_size"]), len(holdout_ids)),
        namespace="holdout_eval",
        seed=int(config.data["eval_seed"]),
    )
    feature_names = list(base_doc["feature_order"])
    session = RunSession(
        config,
        run_type="holdout",
        tag=f"holdout_{base.run_id}",
        prepared=prepared,
        base_fields={
            "experiment_set": "holdout",
            "algorithm": base_doc["algorithm"],
            "k": base_doc["k"],
            "covariance_type": base_doc.get("covariance_type"),
            "feature_set": base_doc["feature_set"],
            "feature_order": feature_names,
            "scaler": base_doc["scaler"],
            "pca_variance": base_doc.get("pca_variance"),
            "pca_actual_components": base_doc.get("pca_actual_components"),
            "seed": base_doc["seed"],
            "n_init": base_doc.get("n_init"),
            "reg_covar": base_doc.get("reg_covar"),
            "n_fit": base_doc.get("n_fit"),
            "n_eval": len(holdout_eval_ids),
            "n_holdout": len(holdout_ids),
            "holdout_ids_sha256": sha256_obj(holdout_ids),
            "holdout_eval_ids_sha256": sha256_obj(holdout_eval_ids),
            "fit_ids_sha256": base_doc.get("fit_ids_sha256"),
            "eval_ids_sha256": base_doc.get("eval_ids_sha256"),
            "fit_scope": "frozen_development_model",
            "assignment_scope": "holdout",
            "evaluation_model_id": base.run_id,
            "final_refit_model_id": None,
            "holdout_used": True,
            "holdout_reused": previous_doc is not None,
            "frozen_run_id": base.run_id,
            "frozen_descriptor": descriptor,
            "frozen_descriptor_sha256": sha256_obj(descriptor),
            "has_model": True,
            "descriptor": descriptor,
        },
    )
    try:
        X_holdout = prepared.feature_matrix(holdout_ids, feature_names)
        started = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            labels_all = bundle.predict(X_holdout)
            probabilities_all = bundle.predict_proba(X_holdout) if bundle.is_gmm else None
        runtime_seconds = time.perf_counter() - started
        position = {track_id: index for index, track_id in enumerate(holdout_ids)}
        eval_positions = np.array([position[track_id] for track_id in holdout_eval_ids], dtype=np.int64)
        Xt_eval = bundle.transform(X_holdout[eval_positions])
        labels_eval = labels_all[eval_positions]
        eval_score = score_clustering(
            Xt_eval,
            labels_eval,
            evaluation_ids=holdout_eval_ids,
            metric_space_id=bundle.feature_space_id,
            expected_k=int(base_doc["k"]),
        )
        assignments = assignments_frame(
            holdout_ids, labels_all, probabilities=probabilities_all
        )
        atomic_write_parquet(session.path / "assignments.parquet", assignments)
        atomic_write_csv(
            session.path / "eval_ids.csv", pd.DataFrame({"track_id": holdout_eval_ids})
        )
        save_bundle(bundle, session.path / "model.joblib")
        write_bundle_metadata(bundle, session.path / "model_metadata.json")
        sizes_all = cluster_size_table(labels_all, expected_k=int(base_doc["k"]))
        sizes_eval = cluster_size_table(labels_eval, expected_k=int(base_doc["k"]))
        atomic_write_csv(session.path / "tables" / "cluster_sizes_holdout.csv", sizes_all)
        atomic_write_csv(session.path / "tables" / "cluster_sizes_holdout_eval.csv", sizes_eval)
        plot_cluster_sizes(
            sizes_all,
            session.path / "figures" / "cluster_sizes_holdout.png",
            title=(
                f"Frozen {base_doc['algorithm'].upper()} K={base_doc['k']} | holdout "
                f"n={len(holdout_ids)} | eval n={len(holdout_eval_ids)} | run {session.run_id}"
            ),
        )
        dev_sil = _metric_value(base.metrics.get("metrics", {}), "silhouette")
        dev_db = _metric_value(base.metrics.get("metrics", {}), "davies_bouldin")
        dev_ch = _metric_value(base.metrics.get("metrics", {}), "calinski_harabasz")
        metrics_map = {
            **{name: eval_score[name] for name in ("silhouette", "davies_bouldin", "calinski_harabasz")},
        }
        metrics_map["silhouette_delta_vs_dev"] = _metric_entry(
            (eval_score["silhouette"]["value"] - dev_sil)
            if eval_score["silhouette"]["value"] is not None and dev_sil is not None
            else None,
            direction="context",
            space="delta: holdout fixed cohort minus development fixed cohort "
            "(same representation, different tracks)",
            n=len(holdout_eval_ids),
            method="holdout silhouette - development silhouette",
            reason=None
            if eval_score["silhouette"]["value"] is not None and dev_sil is not None
            else "silhouette_undefined_on_one_side",
        )
        metrics_map["davies_bouldin_delta_vs_dev"] = _metric_entry(
            (eval_score["davies_bouldin"]["value"] - dev_db)
            if eval_score["davies_bouldin"]["value"] is not None and dev_db is not None
            else None,
            direction="context",
            space="delta: holdout minus development (same representation)",
            n=len(holdout_eval_ids),
            method="holdout DB - development DB",
            reason=None
            if eval_score["davies_bouldin"]["value"] is not None and dev_db is not None
            else "db_undefined_on_one_side",
        )
        metrics_map["calinski_harabasz_delta_vs_dev"] = _metric_entry(
            (eval_score["calinski_harabasz"]["value"] - dev_ch)
            if eval_score["calinski_harabasz"]["value"] is not None and dev_ch is not None
            else None,
            direction="context",
            space="delta: holdout minus development (same representation)",
            n=len(holdout_eval_ids),
            method="holdout CH - development CH",
            reason=None
            if eval_score["calinski_harabasz"]["value"] is not None and dev_ch is not None
            else "ch_undefined_on_one_side",
        )
        metrics_doc = {
            "schema": "btl11.metrics.holdout.v1",
            "run_id": session.run_id,
            "context": {
                "frozen_run_id": base.run_id,
                "dataset_sha256": prepared.dataset_sha256,
                "holdout_ids_sha256": sha256_obj(holdout_ids),
                "holdout_eval_ids_sha256": sha256_obj(holdout_eval_ids),
                "evaluation_ids_sha256": sha256_obj(holdout_eval_ids),
                "feature_space_id": bundle.feature_space_id,
                "algorithm": base_doc["algorithm"],
                "k": base_doc["k"],
                "covariance_type": base_doc.get("covariance_type"),
                "n_holdout": len(holdout_ids),
                "n_eval": len(holdout_eval_ids),
                "holdout_protocol": (
                    "one-time, frozen development model, no refit; not used for selection"
                ),
            },
            "metrics": metrics_map,
            "dev_reference": {
                "silhouette": dev_sil,
                "davies_bouldin": dev_db,
                "calinski_harabasz": dev_ch,
                "run_id": base.run_id,
            },
            "cluster_sizes_holdout": sizes_all.to_dict(orient="records"),
            "cluster_sizes_holdout_eval": sizes_eval.to_dict(orient="records"),
            "warnings": _captured_warnings(caught),
            "runtime_seconds": runtime_seconds,
            "note": (
                "Clustering has no ground-truth labels. Holdout here tests assignment/geometry "
                "sanity out-of-sample, not classification accuracy."
            ),
        }
        session.complete(
            metrics=metrics_doc,
            updates={
                "summary_metrics": {
                    "silhouette": eval_score["silhouette"]["value"],
                    "davies_bouldin": eval_score["davies_bouldin"]["value"],
                    "calinski_harabasz": eval_score["calinski_harabasz"]["value"],
                    "bic": None,
                    "aic": None,
                },
                "checksums": {
                    "model_sha256": sha256_file(session.path / "model.joblib"),
                    "assignments_sha256": sha256_file(session.path / "assignments.parquet"),
                },
                "frozen_run_id": base.run_id,
                "holdout_protocol": "one-time; no refit; no selection use",
                "warnings": _captured_warnings(caught),
                "warnings_count": len(caught),
            },
        )
        return {
            "run_id": session.run_id,
            "path": str(session.path),
            "frozen_run_id": base.run_id,
            "metrics": metrics_map,
            "reused": previous_doc is not None,
        }
    except Exception as exc:
        session.fail(exc)
        raise


def _find_holdout_for(frozen_run_id: str, config: Config) -> str | None:
    matches = [
        doc.get("run_id")
        for _, doc in iter_run_docs(config)
        if doc.get("run_type") == "holdout"
        and doc.get("frozen_run_id") == frozen_run_id
        and doc.get("status") == "completed"
    ]
    return matches[-1] if matches else None


def _find_holdout_for_descriptor(
    config: Config, prepared: PreparedData, descriptor: Mapping[str, Any]
) -> dict[str, Any] | None:
    """Find a completed one-time holdout for this descriptor and data/split identity.

    Keying only on a candidate run_id would let a re-run with the same descriptor
    bypass the one-time rule, so the guard is descriptor- and data-scoped.
    """
    target = sha256_obj(descriptor)
    matches: list[dict[str, Any]] = []
    for _, doc in iter_run_docs(config):
        if doc.get("run_type") != "holdout" or doc.get("status") != "completed":
            continue
        if doc.get("dataset_sha256") != prepared.dataset_sha256:
            continue
        if doc.get("split_sha256") != prepared.summary.get("split_sha256"):
            continue
        stored_descriptor = doc.get("frozen_descriptor")
        if stored_descriptor is not None:
            if sha256_obj(stored_descriptor) == target:
                matches.append(doc)
            continue
        base_id = doc.get("frozen_run_id")
        if not base_id:
            continue
        try:
            base_doc = load_run_doc(config, base_id)
        except ArtifactError:
            continue
        if descriptor_matches(base_doc, descriptor):
            matches.append(doc)
    if not matches:
        return None
    matches.sort(key=lambda item: item.get("created_utc", ""), reverse=True)
    return matches[0]


# --------------------------------------------------------------------------- #
# Final refit
# --------------------------------------------------------------------------- #
def refit_all_tracks(
    config: Config,
    prepared: PreparedData,
    selector: str = "selected_final",
) -> dict[str, Any]:
    if selector not in {"selected_final"}:
        raise ConfigError("refit --model-selector must be selected_final.")
    selected = config.selection.get("selected")
    frozen = list(config.selection.get("frozen") or [])
    if not selected:
        raise ConfigError(
            "selection.selected is empty. Freeze a winner from development evidence first."
        )
    if len(frozen) != 1:
        raise ConfigError(
            "selection.frozen must contain exactly one winner before refit; "
            "do not refit an unfrozen or post-holdout-changed selection."
        )
    descriptor = config.descriptor(selected)
    frozen_descriptor = config.descriptor(frozen[0])
    if descriptor != frozen_descriptor:
        raise ConfigError(
            "selection.selected differs from the frozen winner. Refusing to refit a model "
            "changed after the holdout freeze."
        )
    holdout_doc = _find_holdout_for_descriptor(config, prepared, frozen_descriptor)
    if holdout_doc is None:
        raise ConfigError(
            "The one-time holdout for the frozen descriptor has not been completed. "
            "Run `btl11 validate-holdout` before the full-data refit."
        )
    # The evaluation model must be exactly the run that holdout validated,
    # not merely a newer candidate with the same descriptor.
    frozen_run_id = holdout_doc.get("frozen_run_id")
    if not frozen_run_id:
        raise ArtifactError("The completed holdout does not identify its frozen run.")
    frozen_path = run_dir(config, str(frozen_run_id))
    frozen_doc = load_run_doc(config, str(frozen_run_id))
    if frozen_doc.get("status") != "completed" or not descriptor_matches(
        frozen_doc, descriptor
    ):
        raise ArtifactError(
            "The holdout's frozen run is missing, incomplete, or has a different descriptor."
        )
    metrics_path = frozen_path / "metrics.json"
    base = RunResult(
        run_id=str(frozen_run_id),
        path=frozen_path,
        status="completed",
        metrics=read_json(metrics_path) if metrics_path.exists() else {},
        doc=frozen_doc,
    )
    holdout_run = holdout_doc.get("run_id")
    feature_names = _feature_names(config, descriptor["feature_set"])
    all_ids = prepared.all_ids
    X_all = prepared.feature_matrix(all_ids, feature_names)
    session = RunSession(
        config,
        run_type="final_refit",
        tag=f"final_refit_{descriptor['algorithm']}_k{descriptor['k']}",
        prepared=prepared,
        base_fields={
            "experiment_set": "final_refit",
            "algorithm": descriptor["algorithm"],
            "k": descriptor["k"],
            "covariance_type": descriptor["covariance_type"],
            "feature_set": descriptor["feature_set"],
            "feature_order": feature_names,
            "scaler": descriptor["scaler"],
            "pca_variance": descriptor["pca"],
            "pca_actual_components": None,
            "seed": descriptor["seed"],
            "n_init": int(
                config.gmm["n_init_final"]
                if descriptor["algorithm"] == "gmm"
                else config.kmeans["n_init"]
            ),
            "reg_covar": (
                float(config.gmm["reg_covar_quick"])
                if descriptor["algorithm"] == "gmm"
                else None
            ),
            "n_fit": len(all_ids),
            "n_eval": None,
            "fit_ids_sha256": sha256_obj(sorted(all_ids)),
            "eval_ids_sha256": None,
            "fit_scope": "all_unique_tracks",
            "assignment_scope": "all_unique_tracks",
            "evaluation_model_id": base.run_id,
            "final_refit_model_id": None,
            "holdout_used": True,
            "holdout_run_id": holdout_run,
            "frozen_descriptor": frozen_descriptor,
            "frozen_descriptor_sha256": sha256_obj(frozen_descriptor),
            "has_model": True,
            "descriptor": descriptor,
        },
    )
    try:
        started = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            bundle, Xt_all = fit_clustering_bundle(
                X_all,
                all_ids,
                algorithm=descriptor["algorithm"],
                k=descriptor["k"],
                config=config,
                seed=descriptor["seed"],
                feature_names=feature_names,
                scaler=descriptor["scaler"],
                pca_variance=descriptor["pca"],
                covariance_type=descriptor["covariance_type"],
                n_init=(
                    int(config.gmm["n_init_final"])
                    if descriptor["algorithm"] == "gmm"
                    else int(config.kmeans["n_init"])
                ),
            )
            fit_seconds = time.perf_counter() - started
            labels_all = np.asarray(bundle.estimator.predict(Xt_all), dtype=np.int64)
            probabilities_all = bundle.predict_proba(X_all) if bundle.is_gmm else None
        verification_ids = select_ids_by_hash(
            all_ids, int(config.data["eval_cohort_size"]), "final_verification",
            int(config.data["eval_seed"]),
        )
        position = {track_id: index for index, track_id in enumerate(all_ids)}
        verify_positions = np.array(
            [position[track_id] for track_id in verification_ids], dtype=np.int64
        )
        Xt_verify = Xt_all[verify_positions]
        labels_verify = labels_all[verify_positions]
        verify_score = score_clustering(
            Xt_verify,
            labels_verify,
            evaluation_ids=verification_ids,
            metric_space_id=bundle.feature_space_id,
            expected_k=int(descriptor["k"]),
        )
        assignments = assignments_frame(
            all_ids, labels_all, probabilities=probabilities_all
        )
        model_path = save_bundle(bundle, session.path / "model.joblib")
        write_bundle_metadata(bundle, session.path / "model_metadata.json")
        assignments_path = atomic_write_parquet(
            session.path / "assignments.parquet", assignments
        )
        atomic_write_csv(
            session.path / "verification_ids.csv",
            pd.DataFrame({"track_id": verification_ids}),
        )
        sizes_all = cluster_size_table(labels_all, expected_k=int(descriptor["k"]))
        atomic_write_csv(session.path / "tables" / "cluster_sizes_all.csv", sizes_all)
        plot_cluster_sizes(
            sizes_all,
            session.path / "figures" / "cluster_sizes_all.png",
            title=(
                f"Final refit {descriptor['algorithm'].upper()} K={descriptor['k']} | "
                f"all unique tracks n={len(all_ids)} | run {session.run_id}"
            ),
        )
        metrics_map = {
            **{name: verify_score[name] for name in ("silhouette", "davies_bouldin", "calinski_harabasz")}
        }
        metrics_map["silhouette_descriptive"] = _metric_entry(
            verify_score["silhouette"]["value"],
            direction="context",
            space=bundle.feature_space_id,
            n=len(verification_ids),
            method=(
                "silhouette on a deterministic 3000-track sample of all tracks; "
                "in-sample descriptive, NOT model-selection evidence"
            ),
            reason=verify_score["silhouette"]["reason"],
        )
        if bundle.is_gmm:
            probs = np.asarray(probabilities_all, dtype=np.float64)
            unc = gmm_uncertainty(probs)
            metrics_map["gmm_confidence_mean_all"] = _metric_entry(
                float(unc["confidence"].mean()), direction="context",
                space=bundle.feature_space_id, n=len(all_ids),
                method="mean posterior confidence on all refit tracks",
            )
        # Profiles, representative tracks and post-hoc genre tables.
        from .profiling import generate_genre_posthoc, generate_profiles

        profile_info = generate_profiles(
            config, prepared, session.path, session.run_id, assignments,
            bundle=bundle, all_tracks_scope=True,
        )
        genre_info = generate_genre_posthoc(
            config, prepared, session.path, session.run_id, assignments, all_tracks_scope=True
        )
        metrics_doc = {
            "schema": "btl11.metrics.final_refit.v1",
            "run_id": session.run_id,
            "context": {
                "dataset_sha256": prepared.dataset_sha256,
                "feature_space_id": bundle.feature_space_id,
                "algorithm": descriptor["algorithm"],
                "k": descriptor["k"],
                "covariance_type": descriptor["covariance_type"],
                "scaler": descriptor["scaler"],
                "pca_variance": descriptor["pca"],
                "pca_actual_components": bundle.pca_actual_components,
                "n_all_tracks": len(all_ids),
                "verification_n": len(verification_ids),
                "verification_ids_sha256": sha256_obj(verification_ids),
                "evaluation_ids_sha256": sha256_obj(verification_ids),
                "evaluation_model_id": base.run_id,
                "holdout_run_id": holdout_run,
                "protocol_note": (
                    "Full-data refit is a distinct model identity. Descriptive verification "
                    "metrics are in-sample and do not inherit holdout metrics."
                ),
            },
            "metrics": metrics_map,
            "cluster_sizes_all": sizes_all.to_dict(orient="records"),
            "profile_summary": profile_info,
            "genre_posthoc": genre_info,
            "warnings": _captured_warnings(caught),
            "runtime": {
                "fit_seconds": fit_seconds,
                "peak_rss_mib_process": peak_rss_mib(),
            },
        }
        doc = session.complete(
            metrics=metrics_doc,
            updates={
                "feature_space_id": bundle.feature_space_id,
                "pca_actual_components": bundle.pca_actual_components,
                "n_eval": len(verification_ids),
                "n_init": int(
                    config.gmm["n_init_final"]
                    if descriptor["algorithm"] == "gmm"
                    else config.kmeans["n_init"]
                ),
                "final_refit_model_id": session.run_id,
                "evaluation_model_id": base.run_id,
                "holdout_run_id": holdout_run,
                "converged": bool(getattr(bundle.estimator, "converged_", True)),
                "n_iter": int(getattr(bundle.estimator, "n_iter_", -1)),
                "warnings": _captured_warnings(caught),
                "warnings_count": len(caught),
                "summary_metrics": {
                    "silhouette": verify_score["silhouette"]["value"],
                    "davies_bouldin": verify_score["davies_bouldin"]["value"],
                    "calinski_harabasz": verify_score["calinski_harabasz"]["value"],
                    "bic": None,
                    "aic": None,
                },
                "checksums": {
                    "model_sha256": sha256_file(model_path),
                    "assignments_sha256": sha256_file(assignments_path),
                },
                "fit_seconds": fit_seconds,
                "profile_files": profile_info.get("files", {}),
                "genre_files": genre_info.get("files", {}),
                "protocol_note": (
                    "Distinct full-data refit identity; descriptive metrics only. "
                    "Holdout metrics belong to the frozen evaluation model."
                ),
            },
        )
        selection_record = {
            "selected": descriptor,
            "evaluation_run_id": base.run_id,
            "holdout_run_id": holdout_run,
            "final_refit_run_id": session.run_id,
            "created_utc": utc_now(),
            "note": (
                "Selection was based on development evidence. Holdout was a one-time check; "
                "the full-data refit is a separate model identity."
            ),
        }
        selection_dir = Path("artifacts/selection")
        selection_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(selection_dir / "final_selection.json", selection_record)
        return {
            "run_id": session.run_id,
            "path": str(session.path),
            "evaluation_run_id": base.run_id,
            "proto": doc["status"],
            "profile_info": profile_info,
            "genre_info": genre_info,
        }
    except Exception as exc:
        session.fail(exc)
        raise


# --------------------------------------------------------------------------- #
# Ward diagnostic
# --------------------------------------------------------------------------- #
def ward_diagnostic(config: Config, prepared: PreparedData) -> dict[str, Any]:
    selected = config.selection.get("selected")
    if not selected:
        raise ConfigError("selection.selected is empty; run ward only after freezing the winner.")
    descriptor = config.descriptor(selected)
    base = find_matching_run(
        config,
        descriptor,
        run_type="candidate",
        dataset_sha256=prepared.dataset_sha256,
        fit_ids_sha256=sha256_obj(list(prepared.splits["fit"])),
        eval_ids_sha256=sha256_obj(list(prepared.splits["eval"])),
    )
    if base is None:
        raise ArtifactError("No completed candidate run matches selection.selected for Ward.")
    base_doc = base.doc or load_run_doc(config, base.run_id)
    model_path = base.path / "model.joblib"
    expected_model_sha = base_doc.get("checksums", {}).get("model_sha256")
    if not model_path.exists() or sha256_file(model_path) != expected_model_sha:
        raise ArtifactError(
            "Selected model hash mismatch; refusing to run Ward against a changed artifact."
        )
    bundle = load_bundle(model_path)
    feature_names = _feature_names(config, descriptor["feature_set"])
    if list(bundle.feature_names) != feature_names:
        raise ArtifactError("Selected model feature order does not match the Ward descriptor.")
    if bundle.algorithm != descriptor["algorithm"] or int(bundle.k) != int(descriptor["k"]):
        raise ArtifactError("Selected model algorithm/K does not match the Ward descriptor.")
    sample_size = min(int(config.hierarchical["sample_size"]), len(prepared.splits["fit"]))
    sample_ids = select_ids_by_hash(
        prepared.splits["fit"], sample_size, "ward_sample", config.seed
    )
    X_sample_raw = prepared.feature_matrix(sample_ids, feature_names)
    session = RunSession(
        config,
        run_type="ward",
        tag="ward_diagnostic_frozen_repr",
        prepared=prepared,
        base_fields={
            "experiment_set": "ward_diagnostic",
            "algorithm": "ward",
            "k": descriptor["k"],
            "covariance_type": None,
            "feature_set": descriptor["feature_set"],
            "feature_order": feature_names,
            "feature_space_id": bundle.feature_space_id,
            "scaler": descriptor["scaler"],
            "pca_variance": descriptor["pca"],
            "pca_actual_components": bundle.pca_actual_components,
            "seed": config.seed,
            "n_fit": len(sample_ids),
            "n_eval": len(sample_ids),
            "fit_ids_sha256": sha256_obj(sample_ids),
            "eval_ids_sha256": sha256_obj(sample_ids),
            "fit_scope": "bounded_deterministic_sample_of_fit_ids",
            "assignment_scope": "bounded_sample_only",
            "evaluation_model_id": base.run_id,
            "selected_run_id": base.run_id,
            "final_refit_model_id": None,
            "holdout_used": False,
            "has_model": False,
            "descriptor": descriptor,
            "selected_model_sha256": expected_model_sha,
            "ward_note": (
                "Ward is a sample-only structural diagnostic; it does not predict arbitrary "
                "new tracks and is not compared as a generalising classifier."
            ),
        },
    )
    try:
        # Use the frozen selected model's fitted transformer so the Ward-vs-model
        # ARI isolates the algorithm instead of also refitting scaling statistics.
        X_sample = bundle.transform(X_sample_raw)
        started = time.perf_counter()
        ward = AgglomerativeClustering(
            n_clusters=int(descriptor["k"]),
            linkage=config.hierarchical["linkage"],
            metric=config.hierarchical["metric"],
        )
        ward_labels = np.asarray(ward.fit_predict(X_sample), dtype=np.int64)
        ward_seconds = time.perf_counter() - started
        ward_score = score_clustering(
            np.asarray(X_sample, dtype=np.float64),
            ward_labels,
            evaluation_ids=sample_ids,
            metric_space_id=f"{bundle.feature_space_id}|ward_sample",
            expected_k=int(descriptor["k"]),
        )
        base_assignments = pd.read_parquet(base.path / "assignments.parquet")
        base_assignments["track_id"] = base_assignments["track_id"].astype(str)
        base_labels = (
            base_assignments.set_index("track_id").reindex(sample_ids)["cluster"].to_numpy(
                dtype=np.int64
            )
        )
        if np.isnan(base_labels.astype(float)).any():
            raise ArtifactError("Selected run does not assign every Ward sample track.")
        ari = ari_between_same_ids(ward_labels, base_labels, sample_ids, sample_ids)
        atomic_write_csv(
            session.path / "tables" / "ward_assignments.csv",
            pd.DataFrame({"track_id": sample_ids, "ward_cluster": ward_labels,
                          "selected_model_cluster": base_labels}),
        )
        sizes = cluster_size_table(ward_labels, expected_k=int(descriptor["k"]))
        atomic_write_csv(session.path / "tables" / "ward_cluster_sizes.csv", sizes)
        # Dendrogram from Ward linkage on the same sample representation.
        Z = linkage(np.asarray(X_sample, dtype=np.float64), method="ward")
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 5))
        dendrogram(Z, truncate_mode="lastp", p=30, ax=ax, show_leaf_counts=True)
        ax.set_title(
            f"Ward dendrogram (truncated) | sample n={len(sample_ids)} | "
            f"{descriptor['feature_set']}/{descriptor['scaler']} | run {session.run_id}"
        )
        ax.set_xlabel("Merged clusters (leaf counts in parentheses)")
        ax.set_ylabel("Ward linkage distance")
        fig.tight_layout()
        fig.savefig(session.path / "figures" / "ward_dendrogram.png", dpi=160)
        plt.close(fig)
        metrics_doc = {
            "schema": "btl11.metrics.ward.v1",
            "run_id": session.run_id,
            "context": {
                "sample_ids_sha256": sha256_obj(sample_ids),
                "n_sample": len(sample_ids),
                "k": descriptor["k"],
                "linkage": config.hierarchical["linkage"],
                "metric": config.hierarchical["metric"],
                "selected_run_id": base.run_id,
                "selected_model_sha256": expected_model_sha,
                "feature_space_id": bundle.feature_space_id,
                "representation": "frozen selected-model transformer (fit on fit IDs)",
                "sample_scope": "deterministic bounded sample of development fit IDs",
            },
            "metrics": {
                **{name: ward_score[name] for name in ("silhouette", "davies_bouldin", "calinski_harabasz")},
                "ari_vs_selected_model_same_sample": _metric_entry(
                    ari, direction="maximize", space="Ward sample labels vs selected model labels",
                    n=len(sample_ids), method="Adjusted Rand Index on identical sample IDs",
                ),
            },
            "cluster_sizes": sizes.to_dict(orient="records"),
            "runtime": {"ward_seconds": ward_seconds, "note": "O(n^2) sample-only"},
            "caveat": (
                "Ward cannot predict new points; this is not a generalisation metric. The "
                "transformer is the frozen selected model's fitted transformer, so the ARI "
                "isolates the clustering algorithm on a bounded sample; validity metrics are "
                "sample-only and not comparable to the fixed evaluation cohort."
            ),
        }
        session.complete(
            metrics=metrics_doc,
            updates={
                "summary_metrics": {
                    "silhouette": ward_score["silhouette"]["value"],
                    "davies_bouldin": ward_score["davies_bouldin"]["value"],
                    "calinski_harabasz": ward_score["calinski_harabasz"]["value"],
                    "bic": None,
                    "aic": None,
                },
                "ward_ari_vs_selected": ari,
                "has_model": False,
                "ward_seconds": ward_seconds,
                "feature_space_id": bundle.feature_space_id,
                "selected_model_sha256": expected_model_sha,
            },
        )
        return {
            "run_id": session.run_id,
            "path": str(session.path),
            "ari_vs_selected": ari,
            "silhouette": ward_score["silhouette"]["value"],
        }
    except Exception as exc:
        session.fail(exc)
        raise
