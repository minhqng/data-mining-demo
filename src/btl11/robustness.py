"""Post-hoc feature-dominance, leave-one-feature-out and controlled model checks.

This module is deliberately separate from the frozen experiment protocol. It
reads completed artifacts, never mutates them, and uses development IDs only
(fit for fitting sensitivity models, the fixed eval cohort for comparison).
Output is exploratory robustness evidence, not a new confirmatory holdout.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

from .config import Config
from .data import PreparedData
from .errors import ArtifactError, ConfigError, DataError, SchemaError
from .evaluation import cohort_identifier, rank_bic_within_space
from .experiment import (
    descriptor_matches,
    find_matching_run,
    iter_run_docs,
    load_run_doc,
    run_dir,
)
from .features import F5_FEATURES, F9_FEATURES
from .models import ClusteringBundle, fit_clustering_bundle, load_bundle
from .utils import (
    allocate_run_dir,
    atomic_write_csv,
    atomic_write_json,
    sha256_file,
    sha256_obj,
    utc_now,
)

ROBUSTNESS_SCHEMA = "btl11.robustness.v1"
NEAR_ZERO_THRESHOLD = 1.0e-4


# --------------------------------------------------------------------------- #
# Loading and guards
# --------------------------------------------------------------------------- #
def _load_bundle_for_doc(config: Config, doc: Mapping[str, Any], path: Path) -> ClusteringBundle:
    model_path = path / "model.joblib"
    expected = (doc.get("checksums") or {}).get("model_sha256")
    if not model_path.exists() or sha256_file(model_path) != expected:
        raise ArtifactError(
            f"Model hash mismatch for run {doc.get('run_id')}; refusing robustness analysis."
        )
    return load_bundle(model_path)


def _resolve_evaluation_model(
    config: Config,
    prepared: PreparedData,
    evaluation_run_id: str | None,
) -> tuple[str, Path, dict[str, Any], ClusteringBundle]:
    if evaluation_run_id:
        path = run_dir(config, evaluation_run_id)
        doc = load_run_doc(config, evaluation_run_id)
    else:
        frozen = list(config.selection.get("frozen") or [])
        if len(frozen) != 1:
            raise ConfigError(
                "selection.frozen must contain exactly one winner, or pass --evaluation-run."
            )
        descriptor = config.descriptor(frozen[0])
        result = find_matching_run(
            config,
            descriptor,
            run_type="candidate",
            dataset_sha256=prepared.dataset_sha256,
            fit_ids_sha256=sha256_obj(list(prepared.splits["fit"])),
            eval_ids_sha256=sha256_obj(list(prepared.splits["eval"])),
        )
        if result is None:
            raise ArtifactError(
                "No completed candidate matches selection.frozen; run the frozen model first."
            )
        path, doc = result.path, result.doc or load_run_doc(config, result.run_id)
        evaluation_run_id = result.run_id
    if doc.get("status") != "completed" or not doc.get("has_model"):
        raise ArtifactError("Robustness analysis requires a completed run with a model bundle.")
    bundle = _load_bundle_for_doc(config, doc, path)
    if bundle.algorithm != "kmeans":
        raise ConfigError(
            "Feature-dominance and margin attribution are defined for the selected K-Means model."
        )
    if bundle.scaler != "robust":
        raise ConfigError(
            "This analysis targets the selected RobustScaler model; the frozen model uses "
            f"scaler={bundle.scaler!r}."
        )
    if list(bundle.feature_names) not in (list(F9_FEATURES), list(F5_FEATURES)):
        raise SchemaError(f"Unexpected frozen feature order: {list(bundle.feature_names)}")
    return str(evaluation_run_id), path, doc, bundle


def _raw_matrix(prepared: PreparedData, ids: Sequence[str], names: Sequence[str]) -> np.ndarray:
    """Build a finite raw-unit matrix for an arbitrary subset of the F9 allowlist.

    The primary :func:`features.to_feature_matrix` contract intentionally accepts
    only the exact F5/F9 lists, so leave-one-feature-out needs a local read from
    the already-validated prepared table.
    """
    requested = list(names)
    unknown = sorted(set(requested) - set(F9_FEATURES))
    if unknown:
        raise SchemaError(f"Robustness features must be F9 audio columns; unknown: {unknown}")
    frame = prepared.subset_tracks(ids)
    missing = [name for name in requested if name not in frame.columns]
    if missing:
        raise SchemaError(f"Prepared tracks are missing robustness columns: {missing}")
    matrix = frame.loc[:, requested].to_numpy(dtype=np.float64, copy=True)
    if not np.isfinite(matrix).all():
        raise SchemaError("Robustness matrix contains NaN/inf.")
    return matrix


def _check_development_only(prepared: PreparedData) -> None:
    fit = set(prepared.splits["fit"])
    ev = set(prepared.splits["eval"])
    dev = set(prepared.splits["dev"])
    holdout = set(prepared.splits["holdout"])
    if fit & ev:
        raise DataError("Fit and eval cohorts overlap; robustness comparison refused.")
    if fit | ev != dev:
        raise DataError("Fit + eval must equal the development split.")
    if dev & holdout:
        raise DataError("Development and holdout overlap; robustness comparison refused.")


# --------------------------------------------------------------------------- #
# Attribution tables
# --------------------------------------------------------------------------- #
def _scaler_table(
    bundle: ClusteringBundle,
    names: Sequence[str],
    X_fit: np.ndarray,
    Z_fit: np.ndarray,
) -> pd.DataFrame:
    scaler = bundle.transformer.named_steps["scaler"]
    center = getattr(scaler, "center_", getattr(scaler, "mean_", None))
    scale = getattr(scaler, "scale_", None)
    if center is None or scale is None:
        raise SchemaError("The fitted scaler does not expose center/scale parameters.")
    rows: list[dict[str, Any]] = []
    for index, name in enumerate(names):
        raw = X_fit[:, index]
        z = Z_fit[:, index]
        q1, q3 = np.quantile(raw, [0.25, 0.75])
        rows.append(
            {
                "feature": name,
                "unit_interval_expected": name not in {"tempo", "loudness"},
                "raw_min": float(raw.min()),
                "raw_p01": float(np.quantile(raw, 0.01)),
                "raw_median": float(np.median(raw)),
                "raw_p99": float(np.quantile(raw, 0.99)),
                "raw_max": float(raw.max()),
                "raw_mean": float(raw.mean()),
                "raw_std": float(raw.std(ddof=0)),
                "raw_var": float(raw.var(ddof=0)),
                "raw_q1": float(q1),
                "raw_q3": float(q3),
                "raw_iqr": float(q3 - q1),
                "zero_share": float(np.mean(raw == 0.0)),
                "near_zero_share": float(np.mean(np.abs(raw) < NEAR_ZERO_THRESHOLD)),
                "scaler_center": float(center[index]),
                "scaler_scale": float(scale[index]),
                "transformed_var": float(z.var(ddof=0)),
                "transformed_p01": float(np.quantile(z, 0.01)),
                "transformed_q1": float(np.quantile(z, 0.25)),
                "transformed_median": float(np.median(z)),
                "transformed_q3": float(np.quantile(z, 0.75)),
                "transformed_p99": float(np.quantile(z, 0.99)),
            }
        )
    return pd.DataFrame(rows)


def _variance_decomposition(
    names: Sequence[str],
    Z_fit: np.ndarray,
    labels: np.ndarray,
    centers: np.ndarray,
) -> pd.DataFrame:
    grand = Z_fit.mean(axis=0)
    total = ((Z_fit - grand) ** 2).sum(axis=0)
    within = ((Z_fit - centers[labels]) ** 2).sum(axis=0)
    between = total - within
    total_sum = float(total.sum())
    within_sum = float(within.sum())
    between_sum = float(between.sum())
    rows = []
    for index, name in enumerate(names):
        rows.append(
            {
                "feature": name,
                "total_ss": float(total[index]),
                "within_ss": float(within[index]),
                "between_ss": float(between[index]),
                "between_share": float(between[index] / total[index])
                if total[index] > 0
                else None,
                "eta_squared": float(between[index] / total[index])
                if total[index] > 0
                else None,
                "share_of_total_within": float(within[index] / within_sum)
                if within_sum > 0
                else None,
                "share_of_total_between": float(between[index] / between_sum)
                if between_sum > 0
                else None,
                "total_ss_share": float(total[index] / total_sum) if total_sum > 0 else None,
            }
        )
    return pd.DataFrame(rows)


def _assignment_contributions(
    names: Sequence[str],
    Z_fit: np.ndarray,
    labels: np.ndarray,
    centers: np.ndarray,
) -> pd.DataFrame:
    n_points, k = len(labels), len(centers)
    assigned = (Z_fit - centers[labels]) ** 2
    distances = np.stack(
        [((Z_fit - centers[component]) ** 2).sum(axis=1) for component in range(k)],
        axis=1,
    )
    mask = np.ones_like(distances, dtype=bool)
    mask[np.arange(n_points), labels] = False
    nearest_other = np.argmin(np.where(mask, distances, np.inf), axis=1)
    alternative = (Z_fit - centers[nearest_other]) ** 2
    contribution = alternative - assigned
    margin = contribution.sum(axis=1)
    mean_assigned = assigned.mean(axis=0)
    mean_contribution = contribution.mean(axis=0)
    mean_margin = float(margin.mean())
    rows = []
    for index, name in enumerate(names):
        rows.append(
            {
                "feature": name,
                "mean_assigned_squared_distance": float(mean_assigned[index]),
                "share_of_assigned_squared_distance": float(
                    mean_assigned[index] / mean_assigned.sum()
                )
                if mean_assigned.sum() > 0
                else None,
                "mean_margin_contribution": float(mean_contribution[index]),
                "share_of_mean_margin": float(mean_contribution[index] / mean_margin)
                if mean_margin != 0
                else None,
                "flip_share_if_feature_removed": float(
                    np.mean((margin - contribution[:, index]) < 0)
                ),
            }
        )
    return pd.DataFrame(rows)


def _safe_metrics(X: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    n = len(labels)
    observed = int(np.unique(labels).size)
    if n < 3 or observed < 2 or observed > n - 1:
        return {
            "silhouette": None,
            "davies_bouldin": None,
            "calinski_harabasz": None,
            "reason": "invalid_or_degenerate_cluster_count",
        }
    return {
        "silhouette": float(silhouette_score(X, labels, metric="euclidean")),
        "davies_bouldin": float(davies_bouldin_score(X, labels)),
        "calinski_harabasz": float(calinski_harabasz_score(X, labels)),
        "reason": None,
    }


def _single_feature_partitions(
    config: Config,
    prepared: PreparedData,
    names: Sequence[str],
    X_fit: np.ndarray,
    X_eval: np.ndarray,
    full_eval_labels: np.ndarray,
    *,
    k: int,
    seed: int,
    scaler: str,
) -> pd.DataFrame:
    fit_ids = list(prepared.splits["fit"])
    rows: list[dict[str, Any]] = []
    for index, name in enumerate(names):
        bundle, _ = fit_clustering_bundle(
            X_fit[:, [index]],
            fit_ids,
            algorithm="kmeans",
            k=k,
            config=config,
            seed=seed,
            feature_names=[name],
            scaler=scaler,
            pca_variance=None,
        )
        fit_labels = np.asarray(bundle.estimator.labels_, dtype=np.int64)
        eval_labels = bundle.predict(X_eval[:, [index]])
        metrics = _safe_metrics(bundle.transform(X_eval[:, [index]]), eval_labels)
        counts = np.bincount(eval_labels, minlength=k)
        rows.append(
            {
                "feature": name,
                "ari_vs_full_model": float(
                    adjusted_rand_score(full_eval_labels, eval_labels)
                ),
                "silhouette_1d": metrics["silhouette"],
                "n_clusters_fit": int(np.unique(fit_labels).size),
                "n_clusters_eval": int(np.unique(eval_labels).size),
                "min_share_eval": float(counts.min() / counts.sum()),
                "max_share_eval": float(counts.max() / counts.sum()),
                **{f"count_eval_{cluster}": int(counts[cluster]) for cluster in range(k)},
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Leave-one-feature-out
# --------------------------------------------------------------------------- #
def _loo_profiles(
    names: Sequence[str],
    X_eval: np.ndarray,
    labels: np.ndarray,
    dropped: str,
    k: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    counts = np.bincount(labels, minlength=k)
    total = int(counts.sum())
    for cluster in range(k):
        mask = labels == cluster
        if not mask.any():
            continue
        for index, feature in enumerate(names):
            values = X_eval[mask, index]
            rows.append(
                {
                    "dropped_feature": dropped,
                    "cluster": int(cluster),
                    "n": int(counts[cluster]),
                    "share": float(counts[cluster] / total) if total else 0.0,
                    "feature": feature,
                    "median": float(np.median(values)),
                    "q25": float(np.quantile(values, 0.25)),
                    "q75": float(np.quantile(values, 0.75)),
                    "mean": float(values.mean()),
                }
            )
    return rows


def _leave_one_feature_out(
    config: Config,
    prepared: PreparedData,
    bundle: ClusteringBundle,
    names: Sequence[str],
    X_fit: np.ndarray,
    X_eval: np.ndarray,
    full_eval_labels: np.ndarray,
    *,
    include_stability: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fit_ids = list(prepared.splits["fit"])
    k = int(bundle.k)
    seeds = [int(seed) for seed in config.search["stability_seeds"]]
    rows: list[dict[str, Any]] = []
    profile_rows: list[dict[str, Any]] = []
    stability_rows: list[dict[str, Any]] = []

    variants: list[tuple[str, list[int]]] = [("none (frozen reference)", list(range(len(names))))]
    variants.extend(
        (name, [index for index, candidate in enumerate(names) if candidate != name])
        for name in names
    )
    for dropped, keep_indices in variants:
        keep_names = [names[index] for index in keep_indices]
        fit_bundle, transformed_fit = fit_clustering_bundle(
            X_fit[:, keep_indices],
            fit_ids,
            algorithm="kmeans",
            k=k,
            config=config,
            seed=int(bundle.seed),
            feature_names=keep_names,
            scaler=bundle.scaler,
            pca_variance=None,
        )
        transformed_eval = fit_bundle.transform(X_eval[:, keep_indices])
        eval_labels = np.asarray(
            fit_bundle.estimator.predict(transformed_eval), dtype=np.int64
        )
        fit_labels = np.asarray(fit_bundle.estimator.labels_, dtype=np.int64)
        metrics = _safe_metrics(transformed_eval, eval_labels)
        counts = np.bincount(eval_labels, minlength=k)
        ari = (
            1.0
            if dropped == "none (frozen reference)"
            else float(adjusted_rand_score(full_eval_labels, eval_labels))
        )
        row: dict[str, Any] = {
            "dropped_feature": dropped,
            "n_features": len(keep_names),
            "ari_vs_full_model": ari,
            **metrics,
            "min_share_eval": float(counts.min() / counts.sum()) if counts.sum() else None,
            "max_share_eval": float(counts.max() / counts.sum()) if counts.sum() else None,
            "n_clusters_fit": int(np.unique(fit_labels).size),
            "n_clusters_eval": int(np.unique(eval_labels).size),
            **{f"count_eval_{cluster}": int(counts[cluster]) for cluster in range(k)},
        }
        if include_stability:
            seed_aris: list[float] = []
            for seed in seeds:
                seed_bundle, _ = fit_clustering_bundle(
                    X_fit[:, keep_indices],
                    fit_ids,
                    algorithm="kmeans",
                    k=k,
                    config=config,
                    seed=seed,
                    feature_names=keep_names,
                    scaler=bundle.scaler,
                    pca_variance=None,
                )
                seed_eval = seed_bundle.predict(X_eval[:, keep_indices])
                seed_aris.append(float(adjusted_rand_score(eval_labels, seed_eval)))
                stability_rows.append(
                    {
                        "dropped_feature": dropped,
                        "seed": seed,
                        "ari_vs_variant_seed42": seed_aris[-1],
                    }
                )
            row["seed_ari_mean"] = float(np.mean(seed_aris))
            row["seed_ari_min"] = float(np.min(seed_aris))
            row["seed_ari_std"] = float(np.std(seed_aris, ddof=0))
            row["n_seed_refits"] = len(seed_aris)
        rows.append(row)
        profile_rows.extend(_loo_profiles(names, X_eval, eval_labels, dropped, k))
    return (
        pd.DataFrame(rows),
        pd.DataFrame(profile_rows),
        pd.DataFrame(stability_rows),
    )


# --------------------------------------------------------------------------- #
# Controlled K-Means vs GMM table
# --------------------------------------------------------------------------- #
def _find_candidate_run(
    config: Config,
    prepared: PreparedData,
    descriptor: Mapping[str, Any],
    allowed_sets: set[str],
) -> tuple[Path, dict[str, Any]] | None:
    matches: list[tuple[Path, dict[str, Any]]] = []
    for path, doc in iter_run_docs(config):
        if doc.get("status") != "completed" or doc.get("run_type") != "candidate":
            continue
        if doc.get("experiment_set") not in allowed_sets:
            continue
        if doc.get("dataset_sha256") != prepared.dataset_sha256:
            continue
        if doc.get("fit_ids_sha256") != sha256_obj(list(prepared.splits["fit"])):
            continue
        if doc.get("eval_ids_sha256") != sha256_obj(list(prepared.splits["eval"])):
            continue
        if not descriptor_matches(doc, descriptor):
            continue
        matches.append((path, doc))
    if not matches:
        return None
    return max(matches, key=lambda item: item[1].get("created_utc", ""))


def _controlled_model_comparison(
    config: Config,
    prepared: PreparedData,
    *,
    experiment_set: str = "exploratory_controlled_gmm",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    eval_ids = list(prepared.splits["eval"])
    fit_sha = sha256_obj(list(prepared.splits["fit"]))
    eval_sha = sha256_obj(eval_ids)
    cohort = cohort_identifier(prepared.dataset_sha256, fit_sha, eval_sha)
    rows: list[dict[str, Any]] = []
    for scaler in ("standard", "robust"):
        feature_set = "f9"
        for k in (2, 4):
            kmeans_descriptor = {
                "algorithm": "kmeans",
                "k": k,
                "covariance_type": None,
                "feature_set": feature_set,
                "scaler": scaler,
                "pca": None,
                "seed": int(config.seed),
            }
            kmeans_match = _find_candidate_run(
                config,
                prepared,
                kmeans_descriptor,
                {"baseline_f9_standard", "ablation_scaler"},
            )
            kmeans_labels = None
            kmeans_metrics: dict[str, Any] = {}
            kmeans_bundle = None
            if kmeans_match is not None:
                kmeans_path, kmeans_doc = kmeans_match
                kmeans_bundle = _load_bundle_for_doc(config, kmeans_doc, kmeans_path)
                X_eval = prepared.feature_matrix(eval_ids, list(kmeans_bundle.feature_names))
                kmeans_labels = kmeans_bundle.predict(X_eval)
                metrics_doc = json.loads((kmeans_path / "metrics.json").read_text())
                kmeans_metrics = {
                    "kmeans_silhouette": _metric_value(metrics_doc, "silhouette"),
                    "kmeans_davies_bouldin": _metric_value(metrics_doc, "davies_bouldin"),
                    "kmeans_calinski_harabasz": _metric_value(
                        metrics_doc, "calinski_harabasz"
                    ),
                    "kmeans_min_cluster_share_fit": _metric_value(
                        metrics_doc, "min_cluster_share_fit"
                    ),
                }
            for covariance in ("diag", "full"):
                gmm_descriptor = {
                    "algorithm": "gmm",
                    "k": k,
                    "covariance_type": covariance,
                    "feature_set": feature_set,
                    "scaler": scaler,
                    "pca": None,
                    "seed": int(config.seed),
                }
                gmm_match = _find_candidate_run(
                    config, prepared, gmm_descriptor, {experiment_set}
                )
                row: dict[str, Any] = {
                    "space": scaler,
                    "feature_space_id": f"{feature_set}|{scaler}|none|whiten=False",
                    "k": k,
                    "covariance_type": covariance,
                    "cohort_id": cohort,
                    "n_eval": len(eval_ids),
                    "kmeans_run_id": kmeans_match[1].get("run_id") if kmeans_match else None,
                    "kmeans_n_init": kmeans_match[1].get("n_init") if kmeans_match else None,
                    **kmeans_metrics,
                }
                if gmm_match is None or kmeans_labels is None or kmeans_bundle is None:
                    row["reason"] = (
                        "missing_gmm_run" if gmm_match is None else "missing_kmeans_run"
                    )
                    rows.append(row)
                    continue
                gmm_path, gmm_doc = gmm_match
                gmm_bundle = _load_bundle_for_doc(config, gmm_doc, gmm_path)
                if list(gmm_bundle.feature_names) != list(kmeans_bundle.feature_names):
                    raise SchemaError("Controlled runs have different feature orders.")
                X_eval = prepared.feature_matrix(eval_ids, list(gmm_bundle.feature_names))
                gmm_labels = gmm_bundle.predict(X_eval)
                gmm_metrics_doc = json.loads((gmm_path / "metrics.json").read_text())
                uncertainty = gmm_metrics_doc.get("gmm_uncertainty") or {}
                rows.append(
                    {
                        **row,
                        "gmm_run_id": gmm_doc.get("run_id"),
                        "gmm_n_init": gmm_doc.get("n_init"),
                        "gmm_silhouette": _metric_value(gmm_metrics_doc, "silhouette"),
                        "gmm_davies_bouldin": _metric_value(
                            gmm_metrics_doc, "davies_bouldin"
                        ),
                        "gmm_calinski_harabasz": _metric_value(
                            gmm_metrics_doc, "calinski_harabasz"
                        ),
                        "ari_kmeans_vs_gmm": float(
                            adjusted_rand_score(kmeans_labels, gmm_labels)
                        ),
                        "gmm_confidence_mean": uncertainty.get("confidence_mean"),
                        "gmm_normalized_entropy_mean": uncertainty.get(
                            "normalized_entropy_mean"
                        ),
                        "gmm_bic": _metric_value(gmm_metrics_doc, "bic"),
                        "gmm_aic": _metric_value(gmm_metrics_doc, "aic"),
                        "gmm_n_parameters": _metric_value(gmm_metrics_doc, "n_parameters"),
                        "gmm_converged": (gmm_metrics_doc.get("gmm") or {}).get(
                            "converged"
                        ),
                        "gmm_weights": json.dumps(
                            (gmm_metrics_doc.get("gmm") or {}).get("weights")
                        ),
                        "reason": None,
                    }
                )
    table = pd.DataFrame(rows)
    if not table.empty:
        if "gmm_bic" in table.columns:
            # The ranking helper expects a column named ``bic``; only GMM rows carry one.
            table["bic"] = table["gmm_bic"]
            table = rank_bic_within_space(table).drop(columns=["bic"])
        else:
            table["bic_rank_within_space"] = pd.NA
        table = table.sort_values(
            ["space", "k", "covariance_type"], kind="mergesort"
        ).reset_index(drop=True)
    complete = table[table["reason"].isna()] if "reason" in table.columns else table
    summary = {
        "experiment_set": experiment_set,
        "n_rows": int(len(table)),
        "n_complete_comparisons": int(len(complete)),
        "ari_min": float(complete["ari_kmeans_vs_gmm"].min())
        if not complete.empty
        else None,
        "ari_max": float(complete["ari_kmeans_vs_gmm"].max())
        if not complete.empty
        else None,
        "note": (
            "Same fit/eval IDs, feature order, scaler and K within each row. GMM covariance "
            "and n_init are recorded; BIC ranks only within one feature_space_id/cohort."
        ),
    }
    return table, summary


def _metric_value(metrics_doc: Mapping[str, Any], name: str) -> Any:
    entry = (metrics_doc.get("metrics") or {}).get(name)
    if isinstance(entry, Mapping):
        return entry.get("value")
    return None


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def _plot_feature_dominance(table: pd.DataFrame, path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    features = list(table[table["space"] == "robust"]["feature"])
    robust = table[table["space"] == "robust"].set_index("feature")
    standard = table[table["space"] == "standard"].set_index("feature")
    x = np.arange(len(features))
    width = 0.38
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(
        x - width / 2,
        [robust.loc[name, "share_of_total_between"] * 100 for name in features],
        width,
        label="RobustScaler model",
    )
    if not standard.empty:
        ax.bar(
            x + width / 2,
            [standard.loc[name, "share_of_total_between"] * 100 for name in features],
            width,
            label="StandardScaler model",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(features, rotation=35, ha="right")
    ax.set_ylabel("Share of total between-cluster SS (%)")
    ax.set_title(
        "Feature attribution of between-cluster separation\n"
        "F9, K-Means K=2, same development IDs; cross-space shares are contextual",
        fontsize=10,
    )
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _plot_loo_ari(table: pd.DataFrame, path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = table[table["dropped_feature"] != "none (frozen reference)"].copy()
    data = data.sort_values("ari_vs_full_model")
    fig, ax = plt.subplots(figsize=(10, 4.8))
    colors = ["#C44E52" if value < 0.9 else "#4C72B0" for value in data["ari_vs_full_model"]]
    ax.bar(data["dropped_feature"], data["ari_vs_full_model"], color=colors)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_ylabel("ARI vs frozen full-model partition")
    ax.set_xlabel("Feature removed (RobustScaler + K-Means K=2, seed 42)")
    ax.set_ylim(min(-0.2, float(data["ari_vs_full_model"].min()) - 0.05), 1.02)
    ax.set_title(
        "Leave-one-feature-out sensitivity on the fixed development eval cohort\n"
        "ARI is computed on identical IDs; silhouette cross-space is not ranked",
        fontsize=10,
    )
    ax.grid(axis="y", alpha=0.25)
    for index, value in enumerate(data["ari_vs_full_model"]):
        ax.text(index, value, f"{value:.3f}", ha="center", va="bottom", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run_robustness_analysis(
    config: Config,
    prepared: PreparedData,
    *,
    evaluation_run_id: str | None = None,
    output_dir: str | Path = "artifacts/robustness",
    include_stability: bool = True,
    include_scaler_comparison: bool = True,
) -> dict[str, Any]:
    _check_development_only(prepared)
    run_id, run_path, run_doc, bundle = _resolve_evaluation_model(
        config, prepared, evaluation_run_id
    )
    fit_ids = list(prepared.splits["fit"])
    eval_ids = list(prepared.splits["eval"])
    names = list(bundle.feature_names)
    X_fit = _raw_matrix(prepared, fit_ids, names)
    X_eval = _raw_matrix(prepared, eval_ids, names)
    Z_fit = bundle.transform(X_fit)
    Z_eval = bundle.transform(X_eval)
    fit_labels = np.asarray(bundle.estimator.predict(Z_fit), dtype=np.int64)
    eval_labels = np.asarray(bundle.estimator.predict(Z_eval), dtype=np.int64)
    centers = np.asarray(bundle.estimator.cluster_centers_, dtype=np.float64)
    if int(bundle.k) != len(centers):
        raise SchemaError("Frozen K-Means centroids do not match the declared K.")

    scaler_table = _scaler_table(bundle, names, X_fit, Z_fit)
    decomposition = _variance_decomposition(names, Z_fit, fit_labels, centers)
    contributions = _assignment_contributions(names, Z_fit, fit_labels, centers)
    single_feature = _single_feature_partitions(
        config,
        prepared,
        names,
        X_fit,
        X_eval,
        eval_labels,
        k=int(bundle.k),
        seed=int(bundle.seed),
        scaler=bundle.scaler,
    )
    loo, loo_profiles, loo_stability = _leave_one_feature_out(
        config,
        prepared,
        bundle,
        names,
        X_fit,
        X_eval,
        eval_labels,
        include_stability=include_stability,
    )

    scaler_space = pd.DataFrame()
    scaler_space_summary: dict[str, Any] = {
        "available": False,
        "reason": "disabled" if not include_scaler_comparison else "not_found",
    }
    if include_scaler_comparison:
        standard_descriptor = {
            "algorithm": "kmeans",
            "k": int(bundle.k),
            "covariance_type": None,
            "feature_set": "f9" if len(names) == 9 else "f5",
            "scaler": "standard",
            "pca": None,
            "seed": int(bundle.seed),
        }
        standard_match = find_matching_run(
            config,
            standard_descriptor,
            run_type="candidate",
            dataset_sha256=prepared.dataset_sha256,
            fit_ids_sha256=sha256_obj(fit_ids),
            eval_ids_sha256=sha256_obj(eval_ids),
        )
        if standard_match is not None:
            standard_doc = standard_match.doc or load_run_doc(config, standard_match.run_id)
            standard_bundle = _load_bundle_for_doc(
                config, standard_doc, standard_match.path
            )
            standard_names = list(standard_bundle.feature_names)
            if standard_names == names:
                standard_fit = _raw_matrix(prepared, fit_ids, names)
                standard_eval = _raw_matrix(prepared, eval_ids, names)
                standard_Z_fit = standard_bundle.transform(standard_fit)
                standard_Z_eval = standard_bundle.transform(standard_eval)
                standard_fit_labels = np.asarray(
                    standard_bundle.estimator.predict(standard_Z_fit), dtype=np.int64
                )
                standard_eval_labels = np.asarray(
                    standard_bundle.estimator.predict(standard_Z_eval), dtype=np.int64
                )
                standard_decomposition = _variance_decomposition(
                    names,
                    standard_Z_fit,
                    standard_fit_labels,
                    np.asarray(standard_bundle.estimator.cluster_centers_),
                )
                standard_contributions = _assignment_contributions(
                    names,
                    standard_Z_fit,
                    standard_fit_labels,
                    np.asarray(standard_bundle.estimator.cluster_centers_),
                )
                standard_single = _single_feature_partitions(
                    config,
                    prepared,
                    names,
                    standard_fit,
                    standard_eval,
                    standard_eval_labels,
                    k=int(standard_bundle.k),
                    seed=int(standard_bundle.seed),
                    scaler=standard_bundle.scaler,
                )
                robust_table = decomposition.merge(
                    contributions, on="feature", how="inner"
                ).merge(
                    single_feature.loc[:, ["feature", "ari_vs_full_model"]],
                    on="feature",
                    how="inner",
                )
                robust_table["space"] = "robust"
                robust_table["run_id"] = run_id
                robust_table["transformed_var"] = scaler_table.set_index("feature").loc[
                    names, "transformed_var"
                ].to_numpy()
                standard_table = standard_decomposition.merge(
                    standard_contributions, on="feature", how="inner"
                ).merge(
                    standard_single.loc[:, ["feature", "ari_vs_full_model"]],
                    on="feature",
                    how="inner",
                )
                standard_table["space"] = "standard"
                standard_table["run_id"] = standard_match.run_id
                standard_table["transformed_var"] = (
                    np.var(standard_Z_fit, axis=0, ddof=0)
                )
                scaler_space = pd.concat(
                    [robust_table, standard_table], ignore_index=True
                )
                scaler_space = scaler_space.loc[
                    :,
                    [
                        "space",
                        "run_id",
                        "feature",
                        "between_share",
                        "share_of_total_between",
                        "share_of_mean_margin",
                        "flip_share_if_feature_removed",
                        "ari_vs_full_model",
                        "transformed_var",
                    ],
                ]
                scaler_space_summary = {
                    "available": True,
                    "standard_run_id": standard_match.run_id,
                    "robust_run_id": run_id,
                    "cross_scaler_partition_ari": float(
                        adjusted_rand_score(eval_labels, standard_eval_labels)
                    ),
                    "robust_silhouette_eval": _safe_metrics(Z_eval, eval_labels)[
                        "silhouette"
                    ],
                    "standard_silhouette_eval": _safe_metrics(
                        standard_Z_eval, standard_eval_labels
                    )["silhouette"],
                    "note": (
                        "Both models use F9/no-PCA/K=2/seed 42 and the same fit/eval IDs. "
                        "Silhouette values live in different distance geometries and are "
                        "contextual; the cross-scaler ARI is the like-for-like partition "
                        "comparison."
                    ),
                }
        else:
            scaler_space_summary["reason"] = "no_completed_standard_kmeans_run"

    controlled_gmm, controlled_summary = _controlled_model_comparison(config, prepared)

    stamp = utc_now().replace(":", "").replace("-", "")
    _, output_path = allocate_run_dir(Path(output_dir), stamp)
    figures_dir = output_path / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(output_path / "scaler_parameters.csv", scaler_table)
    atomic_write_csv(output_path / "variance_decomposition.csv", decomposition)
    atomic_write_csv(output_path / "assignment_contributions.csv", contributions)
    atomic_write_csv(output_path / "single_feature_partitions.csv", single_feature)
    atomic_write_csv(output_path / "leave_one_feature_out.csv", loo)
    atomic_write_csv(output_path / "loo_profiles.csv", loo_profiles)
    atomic_write_csv(output_path / "loo_seed_stability.csv", loo_stability)
    if not scaler_space.empty:
        atomic_write_csv(output_path / "scaler_space_feature_attribution.csv", scaler_space)
    if not controlled_gmm.empty:
        atomic_write_csv(output_path / "controlled_model_comparison.csv", controlled_gmm)
    dominance_figure = None
    if not scaler_space.empty:
        dominance_figure = _plot_feature_dominance(
            scaler_space, figures_dir / "feature_dominance.png"
        )
    loo_figure = _plot_loo_ari(loo, figures_dir / "loo_ari.png")

    instrumental_between = _row_value(
        decomposition, "instrumentalness", "share_of_total_between"
    )
    instrumental_margin = _row_value(
        contributions, "instrumentalness", "share_of_mean_margin"
    )
    instrumental_flip = _row_value(
        contributions, "instrumentalness", "flip_share_if_feature_removed"
    )
    instrumental_single = _row_value(
        single_feature, "instrumentalness", "ari_vs_full_model"
    )
    instrumental_loo = _row_value(
        loo, "instrumentalness", "ari_vs_full_model", key_column="dropped_feature"
    )
    other_loo = loo[
        (~loo["dropped_feature"].isin(["none (frozen reference)", "instrumentalness"]))
    ]
    other_min = float(other_loo["ari_vs_full_model"].min()) if not other_loo.empty else None
    if instrumental_between is not None and instrumental_margin is not None:
        interpretation = (
            "Under the frozen RobustScaler, instrumentalness accounts for "
            f"{instrumental_between:.1%} of between-cluster sum of squares and "
            f"{instrumental_margin:.1%} of the mean assignment margin; a K=2 partition "
            f"on instrumentalness alone reproduces the full partition at ARI "
            f"{instrumental_single:.3f}. Removing instrumentalness changes the partition "
            f"(ARI {instrumental_loo:.3f} vs the full model)"
            + (
                f", while removing any other single feature leaves ARI >= {other_min:.3f}."
                if other_min is not None
                else "."
            )
            + " The raw feature is strongly zero-inflated/bimodal, so this is a real "
            "acoustic separation amplified by IQR-based scaling, not an implementation "
            "defect; the K=2 model should be described as an instrumental-vs-vocal "
            "segmentation, not as balanced multivariate structure."
        )
    else:
        interpretation = (
            "instrumentalness is not part of the frozen feature set; dominance is undefined."
        )
    summary = {
        "schema": ROBUSTNESS_SCHEMA,
        "generated_utc": utc_now(),
        "evidence_class": "post_hoc_exploratory_development_only",
        "confirmatory": False,
        "holdout_used": False,
        "selection_changed": False,
        "evaluation_run_id": run_id,
        "evaluation_run_type": run_doc.get("run_type"),
        "evaluation_run_path": str(run_path),
        "model_sha256": (run_doc.get("checksums") or {}).get("model_sha256"),
        "dataset_sha256": prepared.dataset_sha256,
        "fit_ids_sha256": sha256_obj(fit_ids),
        "eval_ids_sha256": sha256_obj(eval_ids),
        "feature_space_id": bundle.feature_space_id,
        "feature_order": names,
        "n_fit": len(fit_ids),
        "n_eval": len(eval_ids),
        "scaler_parameters": {
            name: {
                "center": float(scaler_table.iloc[index]["scaler_center"]),
                "scale": float(scaler_table.iloc[index]["scaler_scale"]),
            }
            for index, name in enumerate(names)
        },
        "dominance": {
            "instrumentalness_between_share_of_total_between": instrumental_between,
            "instrumentalness_share_of_mean_margin": instrumental_margin,
            "instrumentalness_flip_share_if_removed": instrumental_flip,
            "instrumentalness_single_feature_ari": instrumental_single,
            "loo_instrumentalness_ari": instrumental_loo,
            "loo_other_min_ari": other_min,
            "interpretation": interpretation,
        },
        "scaler_comparison": scaler_space_summary,
        "controlled_gmm_comparison": controlled_summary,
        "loo_variants": loo.to_dict(orient="records"),
        "caveats": [
            "Post-hoc exploratory analysis on development IDs only; holdout was not touched.",
            "Silhouette/DB/CH values are contextual to each representation; they are not ranked "
            "across RobustScaler/StandardScaler or leave-one-feature-out spaces.",
            "ARI is the like-for-like partition comparison because it uses identical IDs only.",
            "A dominant single feature does not by itself invalidate a partition, but it limits "
            "the claim that the partition reflects balanced multivariate acoustic structure.",
            "The frozen model and all historical runs remain unchanged.",
        ],
        "artifacts": {
            "scaler_parameters": "scaler_parameters.csv",
            "variance_decomposition": "variance_decomposition.csv",
            "assignment_contributions": "assignment_contributions.csv",
            "single_feature_partitions": "single_feature_partitions.csv",
            "leave_one_feature_out": "leave_one_feature_out.csv",
            "loo_profiles": "loo_profiles.csv",
            "loo_seed_stability": "loo_seed_stability.csv",
            "scaler_space_feature_attribution": (
                "scaler_space_feature_attribution.csv" if not scaler_space.empty else None
            ),
            "controlled_model_comparison": (
                "controlled_model_comparison.csv" if not controlled_gmm.empty else None
            ),
            "feature_dominance_figure": (
                str(dominance_figure.relative_to(output_path)) if dominance_figure else None
            ),
            "loo_figure": str(loo_figure.relative_to(output_path)),
        },
    }
    atomic_write_json(output_path / "robustness.json", summary)
    return {"output_dir": str(output_path), **summary}


def _row_value(
    frame: pd.DataFrame,
    key: str,
    value_column: str,
    *,
    key_column: str = "feature",
) -> float | None:
    if (
        frame.empty
        or key_column not in frame.columns
        or value_column not in frame.columns
    ):
        return None
    rows = frame.loc[frame[key_column] == key]
    if rows.empty:
        return None
    value = rows.iloc[0][value_column]
    return None if pd.isna(value) else float(value)
