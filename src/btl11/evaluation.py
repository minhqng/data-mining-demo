"""Guarded intrinsic metrics, GMM information criteria and label-stability ARI."""

from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture

from .errors import SchemaError
from .models import parameter_count
from .utils import sha256_obj


def _metric_entry(
    value: float | None,
    *,
    direction: str,
    space: str,
    n_eval: int,
    method: str,
    reason: str | None = None,
) -> dict[str, Any]:
    numeric = None if value is None else float(value)
    if numeric is not None and not np.isfinite(numeric):
        numeric = None
        reason = "non_finite_metric"
    return {
        "value": numeric,
        "direction": direction,
        "space": space,
        "n_eval": int(n_eval),
        "method": method,
        "reason": reason if numeric is None else None,
    }


def cluster_size_table(
    labels: Sequence[int],
    *,
    expected_k: int | None = None,
    cluster_column: str = "cluster",
) -> pd.DataFrame:
    array = np.asarray(labels, dtype=np.int64)
    if array.ndim != 1:
        raise SchemaError("Labels must be one-dimensional.")
    values, counts = np.unique(array, return_counts=True)
    mapping = {int(value): int(count) for value, count in zip(values, counts, strict=True)}
    cluster_ids = list(range(int(expected_k))) if expected_k is not None else sorted(mapping)
    total = int(len(array))
    return pd.DataFrame(
        {
            cluster_column: cluster_ids,
            "count": [mapping.get(cluster, 0) for cluster in cluster_ids],
            "share": [mapping.get(cluster, 0) / total if total else 0.0 for cluster in cluster_ids],
        }
    )


def score_clustering(
    X: np.ndarray,
    labels: Sequence[int],
    *,
    evaluation_ids: Sequence[str] | None,
    metric_space_id: str,
    expected_k: int | None = None,
) -> dict[str, Any]:
    """Compute validity metrics on the caller-supplied evaluation cohort.

    Silhouette is *not* silently defined for degenerate label counts. Invalid
    cases store ``value=null`` plus an explicit reason.
    """
    matrix = np.asarray(X, dtype=np.float64)
    labels_array = np.asarray(labels, dtype=np.int64)
    if matrix.ndim != 2:
        raise SchemaError("Evaluation X must be two-dimensional.")
    if len(matrix) != len(labels_array):
        raise SchemaError("Evaluation X and labels have different lengths.")
    if not np.isfinite(matrix).all():
        raise SchemaError("Evaluation X contains NaN/inf.")
    n = int(len(labels_array))
    n_clusters = int(np.unique(labels_array).size)
    sizes = cluster_size_table(labels_array, expected_k=expected_k)
    missing_clusters = (
        sorted(set(range(int(expected_k))) - set(np.unique(labels_array).tolist()))
        if expected_k is not None
        else []
    )
    # Use the expected-k table so an empty cluster counts as share 0.0 instead of
    # disappearing from the observed-count minimum.
    min_share = float(sizes["share"].min()) if len(sizes) else None
    max_share = float(sizes["share"].max()) if len(sizes) else None

    sil_reason = None
    sil_value = None
    if n < 3:
        sil_reason = "fewer_than_three_evaluation_points"
    elif n_clusters < 2:
        sil_reason = "fewer_than_two_observed_clusters"
    elif n_clusters > n - 1:
        sil_reason = "each_point_is_its_own_cluster"
    else:
        try:
            sil_value = float(silhouette_score(matrix, labels_array, metric="euclidean"))
        except ValueError as exc:
            sil_reason = f"sklearn_silhouette_error:{exc}"
    silhouette = _metric_entry(
        sil_value,
        direction="maximize",
        space=metric_space_id,
        n_eval=n,
        method="sklearn.metrics.silhouette_score(metric='euclidean')",
        reason=sil_reason,
    )

    db_value, db_reason = None, None
    ch_value, ch_reason = None, None
    if n_clusters < 2:
        db_reason = ch_reason = "fewer_than_two_observed_clusters"
    elif n_clusters > n - 1:
        db_reason = ch_reason = "each_point_is_its_own_cluster"
    else:
        try:
            db_value = float(davies_bouldin_score(matrix, labels_array))
        except ValueError as exc:
            db_reason = f"sklearn_davies_bouldin_error:{exc}"
        try:
            ch_value = float(calinski_harabasz_score(matrix, labels_array))
        except ValueError as exc:
            ch_reason = f"sklearn_calinski_harabasz_error:{exc}"
    davies_bouldin = _metric_entry(
        db_value,
        direction="minimize",
        space=metric_space_id,
        n_eval=n,
        method="sklearn.metrics.davies_bouldin_score",
        reason=db_reason,
    )
    calinski = _metric_entry(
        ch_value,
        direction="maximize",
        space=metric_space_id,
        n_eval=n,
        method="sklearn.metrics.calinski_harabasz_score",
        reason=ch_reason,
    )

    return {
        "n_eval": n,
        "n_clusters_observed": n_clusters,
        "missing_clusters": missing_clusters,
        "cluster_sizes": sizes.to_dict(orient="records"),
        "min_cluster_share": min_share,
        "max_cluster_share": max_share,
        "silhouette": silhouette,
        "davies_bouldin": davies_bouldin,
        "calinski_harabasz": calinski,
        "metric_space_id": metric_space_id,
        "evaluation_ids_sha256": sha256_obj(sorted(map(str, evaluation_ids)))
        if evaluation_ids is not None
        else None,
    }


def inertia_from_labels(
    X: np.ndarray, labels: Sequence[int], centers: np.ndarray
) -> float:
    matrix = np.asarray(X, dtype=np.float64)
    labels_array = np.asarray(labels, dtype=np.int64)
    centers_array = np.asarray(centers, dtype=np.float64)
    if matrix.shape[0] != labels_array.shape[0]:
        raise SchemaError("X and labels have different lengths.")
    assigned = centers_array[labels_array]
    squared = np.sum((matrix - assigned) ** 2, axis=1)
    return float(squared.sum())


def gmm_information_criteria(
    model: GaussianMixture,
    X_fit: np.ndarray,
    *,
    k: int,
    d: int,
    covariance_type: str,
) -> dict[str, Any]:
    """BIC/AIC/LL on the exact fitted representation and cohort."""
    matrix = np.asarray(X_fit, dtype=np.float64)
    bic = float(model.bic(matrix))
    aic = float(model.aic(matrix))
    average_ll = float(model.score(matrix))
    total_ll = average_ll * matrix.shape[0]
    n_parameters = parameter_count(covariance_type, k, d)
    converged = bool(getattr(model, "converged_", False))
    n_iter = int(getattr(model, "n_iter_", -1))
    lower_bound = float(getattr(model, "lower_bound_", np.nan))
    if not np.isfinite(bic) or not np.isfinite(aic) or not np.isfinite(total_ll):
        raise SchemaError("GMM information criteria/log-likelihood are non-finite.")
    return {
        "bic": bic,
        "aic": aic,
        "log_likelihood_total": total_ll,
        "mean_log_likelihood": average_ll,
        "n_parameters": n_parameters,
        "covariance_type": covariance_type,
        "converged": converged,
        "n_iter": n_iter,
        "lower_bound": None if not np.isfinite(lower_bound) else lower_bound,
    }


def ari_between_same_ids(
    labels_a: Sequence[int],
    labels_b: Sequence[int],
    ids_a: Sequence[str],
    ids_b: Sequence[str],
) -> float:
    """ARI for two label vectors defined on exactly the same reference IDs."""
    ids_a = [str(value) for value in ids_a]
    ids_b = [str(value) for value in ids_b]
    if ids_a != ids_b:
        raise SchemaError(
            "ARI requires identical ID order/sets; refusing to compare non-identical rows."
        )
    if len(labels_a) != len(labels_b) or len(labels_a) != len(ids_a):
        raise SchemaError("ARI label vectors/IDs have inconsistent lengths.")
    return float(adjusted_rand_score(np.asarray(labels_a), np.asarray(labels_b)))


def pairwise_ari(
    labels_by_run: dict[str, Sequence[int]], reference_ids: Sequence[str]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """All-pairs ARI on one fixed reference cohort (label-permutation invariant)."""
    ids = [str(value) for value in reference_ids]
    rows: list[dict[str, Any]] = []
    names = list(labels_by_run)
    for name, labels in labels_by_run.items():
        if len(labels) != len(ids):
            raise SchemaError(f"Run {name} labels do not match the reference ID cohort.")
    for left, right in combinations(names, 2):
        score = ari_between_same_ids(labels_by_run[left], labels_by_run[right], ids, ids)
        rows.append({"run_a": left, "run_b": right, "ari": score})
    table = pd.DataFrame(rows, columns=["run_a", "run_b", "ari"])
    values = table["ari"].to_numpy(dtype=float) if not table.empty else np.array([])
    summary = {
        "n_comparisons": int(values.size),
        "mean": float(values.mean()) if values.size else None,
        "std": float(values.std(ddof=0)) if values.size else None,
        "min": float(values.min()) if values.size else None,
        "max": float(values.max()) if values.size else None,
        "reference_ids_sha256": sha256_obj(sorted(ids)),
        "reference_n": int(len(ids)),
        "labels_by_run": {name: [int(value) for value in labels] for name, labels in labels_by_run.items()},
    }
    return table, summary


def cohort_identifier(
    dataset_sha256: str | None, fit_ids_sha256: str | None, eval_ids_sha256: str | None
) -> str:
    """Stable identity for a fair metric/BIC comparison cohort."""
    return sha256_obj(
        {
            "dataset_sha256": dataset_sha256,
            "fit_ids_sha256": fit_ids_sha256,
            "eval_ids_sha256": eval_ids_sha256,
        }
    )[:16]


def rank_bic_within_space(table: pd.DataFrame) -> pd.DataFrame:
    """Rank GMM BIC only inside the same representation **and cohort**.

    Re-running a sweep after changing the raw file or split seed creates a new
    cohort; its BIC values must never be pooled with the old ranking.
    """
    required = {"feature_space_id", "bic"}
    if not required <= set(table.columns):
        raise SchemaError(f"BIC ranking requires columns {sorted(required)}.")
    ranked = table.copy()
    if "cohort_id" not in ranked.columns:
        identity = {"dataset_sha256", "fit_ids_sha256", "eval_ids_sha256"}
        if not identity <= set(ranked.columns):
            raise SchemaError(
                "BIC ranking requires cohort_id or dataset/fit/eval identity columns; "
                "raw BIC must not be pooled across cohorts."
            )
        ranked["cohort_id"] = [
            cohort_identifier(dataset, fit, eval_ids)
            for dataset, fit, eval_ids in zip(
                ranked["dataset_sha256"],
                ranked["fit_ids_sha256"],
                ranked["eval_ids_sha256"],
                strict=True,
            )
        ]
    ranked["bic_rank_within_space"] = (
        ranked.groupby(["feature_space_id", "cohort_id"], dropna=False)["bic"]
        .rank(method="min", ascending=True, na_option="keep")
        .astype("Int64")
    )
    return ranked
