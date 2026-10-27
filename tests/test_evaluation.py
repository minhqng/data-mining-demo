"""Guarded metrics, BIC comparability and label-permutation stability."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score

from btl11.data import select_ids_by_hash
from btl11.errors import SchemaError
from btl11.evaluation import (
    ari_between_same_ids,
    cluster_size_table,
    inertia_from_labels,
    pairwise_ari,
    rank_bic_within_space,
    score_clustering,
)


def test_silhouette_with_one_cluster_is_null_with_reason():
    X = np.arange(20, dtype=float).reshape(10, 2)
    labels = np.zeros(10, dtype=int)
    result = score_clustering(X, labels, evaluation_ids=[str(i) for i in range(10)],
                              metric_space_id="test", expected_k=2)
    assert result["silhouette"]["value"] is None
    assert result["silhouette"]["reason"] == "fewer_than_two_observed_clusters"
    assert result["n_clusters_observed"] == 1
    assert result["missing_clusters"] == [1]
    assert result["davies_bouldin"]["value"] is None
    assert result["calinski_harabasz"]["value"] is None
    assert result["cluster_sizes"][0]["count"] == 10
    assert result["cluster_sizes"][1]["count"] == 0


def test_valid_metrics_and_cluster_size_reconciliation():
    X, y = make_blobs(n_samples=200, centers=3, n_features=5, random_state=1)
    result = score_clustering(X, y, evaluation_ids=[f"id{i}" for i in range(len(y))],
                              metric_space_id="test", expected_k=3)
    assert result["silhouette"]["value"] is not None
    assert -1 <= result["silhouette"]["value"] <= 1
    assert result["davies_bouldin"]["value"] is not None
    assert result["calinski_harabasz"]["value"] is not None
    assert sum(row["count"] for row in result["cluster_sizes"]) == len(y)
    assert abs(sum(row["share"] for row in result["cluster_sizes"]) - 1.0) < 1e-12


def test_silhouette_cohort_cap_is_deterministic():
    ids = [f"id{i:05d}" for i in range(1000)]
    sample_a = select_ids_by_hash(ids, 50, "eval", 42)
    sample_b = select_ids_by_hash(ids, 50, "eval", 42)
    assert sample_a == sample_b
    assert len(sample_a) == 50
    assert sample_a == sorted(sample_a)
    assert len(select_ids_by_hash(ids, 2000, "eval", 42)) == 1000


def test_bic_is_ranked_only_within_representation():
    table = pd.DataFrame(
        {
            "feature_space_id": ["f9|standard|none"] * 3 + ["f5|standard|none"] * 3,
            "cohort_id": ["cohortA"] * 3 + ["cohortB"] * 3,
            "bic": [300.0, 100.0, 200.0, 50.0, 10.0, 20.0],
        }
    )
    ranked = rank_bic_within_space(table)
    assert "bic_rank_within_space" in ranked.columns
    assert list(ranked["bic_rank_within_space"]) == [3, 1, 2, 3, 1, 2]
    assert "bic_rank_global" not in ranked.columns
    # Same feature space but different cohorts must not be pooled.
    split = pd.DataFrame(
        {
            "feature_space_id": ["f9|standard|none"] * 4,
            "cohort_id": ["A", "A", "B", "B"],
            "bic": [30.0, 10.0, 20.0, 5.0],
        }
    )
    split_ranked = rank_bic_within_space(split)
    assert list(split_ranked["bic_rank_within_space"]) == [2, 1, 2, 1]
    with pytest.raises(SchemaError, match="cohort"):
        rank_bic_within_space(pd.DataFrame({"feature_space_id": ["f9"], "bic": [1.0]}))


def test_ari_is_permutation_invariant_and_requires_same_ids():
    ids = [f"t{i}" for i in range(6)]
    labels_a = [0, 0, 1, 1, 2, 2]
    labels_b = [2, 2, 0, 0, 1, 1]  # same partition, renamed clusters
    assert ari_between_same_ids(labels_a, labels_b, ids, ids) == pytest.approx(1.0)
    assert ari_between_same_ids(labels_a, labels_a, ids, ids) == pytest.approx(1.0)
    with pytest.raises(SchemaError, match="identical ID"):
        ari_between_same_ids(labels_a, labels_b, ids, list(reversed(ids)))
    labels_c = [0, 1, 0, 1, 2, 2]
    expected = adjusted_rand_score(labels_a, labels_c)
    table, summary = pairwise_ari({"a": labels_a, "b": labels_b, "c": labels_c}, ids)
    assert set(table["ari"]).issubset({1.0, expected})
    assert summary["reference_n"] == 6
    assert summary["mean"] is not None
    with pytest.raises(SchemaError):
        pairwise_ari({"bad": labels_a[:3]}, ids)


def test_inertia_and_cluster_size_helpers():
    X = np.array([[0.0, 0.0], [0.0, 1.0], [10.0, 10.0], [10.0, 11.0]])
    labels = np.array([0, 0, 1, 1])
    centers = np.array([[0.0, 0.5], [10.0, 10.5]])
    inertia = inertia_from_labels(X, labels, centers)
    assert inertia == pytest.approx(1.0)
    sizes = cluster_size_table(labels, expected_k=3)
    assert sizes["count"].tolist() == [2, 2, 0]
    assert sizes["share"].tolist() == [0.5, 0.5, 0.0]


def test_empty_expected_cluster_counts_as_zero_min_share():
    X = np.array(
        [[0.0, 0.0], [0.1, 0.0], [10.0, 10.0], [10.1, 10.0]], dtype=float
    )
    labels = np.array([0, 0, 1, 1])
    result = score_clustering(
        X,
        labels,
        evaluation_ids=["a", "b", "c", "d"],
        metric_space_id="test",
        expected_k=3,
    )
    assert result["cluster_sizes"][2]["count"] == 0
    assert result["min_cluster_share"] == 0.0
    assert result["max_cluster_share"] == pytest.approx(0.5)
