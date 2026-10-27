"""Preprocessing, K-Means, GMM and bundle contracts."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

from btl11.errors import SchemaError
from btl11.evaluation import gmm_information_criteria
from btl11.features import F9_FEATURES
from btl11.models import (
    ClusteringBundle,
    assignments_frame,
    create_clusterer,
    fit_clustering_bundle,
    gmm_uncertainty,
    load_bundle,
    parameter_count,
    save_bundle,
)
from btl11.preprocess import (
    make_feature_transformer,
    pca_diagnostics,
    pca_variance_frame,
    transformed_feature_names,
)


def _blobs(seed: int = 3, n: int = 300, centers: int = 3) -> tuple[np.ndarray, np.ndarray]:
    X, y = make_blobs(n_samples=n, centers=centers, n_features=9, cluster_std=0.55,
                      random_state=seed)
    return StandardScaler().fit_transform(X), y


def test_scaler_is_fit_only_on_fit_cohort_and_holdout_is_shifted():
    rng = np.random.default_rng(4)
    X_fit = rng.normal(0.0, 1.0, size=(200, 9))
    X_holdout = rng.normal(20.0, 1.0, size=(100, 9))
    transformer = make_feature_transformer("standard", None)
    transformer.fit(X_fit)
    assert np.allclose(transformer.named_steps["scaler"].mean_, 0.0, atol=0.25)
    transformed_holdout = transformer.transform(X_holdout)
    # A leaked fit on holdout would center it near zero; it must remain near +20.
    assert transformed_holdout.mean() > 15.0
    train_mean = transformer.named_steps["scaler"].mean_.copy()
    transformer.transform(X_holdout)
    assert np.array_equal(train_mean, transformer.named_steps["scaler"].mean_)
    with pytest.raises(SchemaError, match="whiten"):
        make_feature_transformer("standard", 0.95, whiten=True)


def test_pca_threshold_components_and_no_pca_identity_path():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(400, 9))
    transformer = make_feature_transformer("standard", 0.95)
    transformer.fit(X)
    ratios = transformer.named_steps["pca"].explained_variance_ratio_
    assert ratios.sum() >= 0.95
    diagnostics = pca_diagnostics(transformer, [f"f{i}" for i in range(9)])
    assert diagnostics["pca_present"] is True
    assert diagnostics["pca_actual_components"] == len(ratios)
    variance = pca_variance_frame(diagnostics)
    assert variance["cumulative_explained_variance"].iloc[-1] >= 0.95
    names = transformed_feature_names(transformer, [f"f{i}" for i in range(9)])
    assert names == [f"PC{i + 1}" for i in range(len(ratios))]
    identity = make_feature_transformer("standard", None)
    identity.fit(X)
    transformed = identity.transform(X)
    assert transformed.shape == X.shape
    assert np.isfinite(transformed).all()
    assert transformed_feature_names(identity, [f"f{i}" for i in range(9)]) == [f"f{i}" for i in range(9)]
    with pytest.raises(SchemaError, match="unscaled"):
        make_feature_transformer("none", 0.95)


def test_kmeans_determinism_nonnegative_inertia_and_monotonic_sanity(fixture_env):
    _, config, _ = fixture_env
    X, _ = _blobs()
    model_a = create_clusterer("kmeans", 3, config, seed=42)
    model_b = create_clusterer("kmeans", 3, config, seed=42)
    labels_a = model_a.fit_predict(X)
    labels_b = model_b.fit_predict(X)
    assert np.array_equal(labels_a, labels_b)
    assert model_a.inertia_ >= 0
    inertia_2 = create_clusterer("kmeans", 2, config, seed=42).fit(X).inertia_
    inertia_3 = create_clusterer("kmeans", 3, config, seed=42).fit(X).inertia_
    assert inertia_2 > inertia_3 >= 0


def test_gmm_posteriors_entropy_parameters_and_finiteness(fixture_env):
    _, config, _ = fixture_env
    X, _ = _blobs(seed=9)
    gmm = create_clusterer("gmm", 3, config, seed=42, covariance_type="diag", n_init=2)
    gmm.fit(X)
    probabilities = gmm.predict_proba(X)
    assert np.isfinite(probabilities).all()
    assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-8)
    assert np.isfinite(gmm.score(X))
    uncertainty = gmm_uncertainty(probabilities)
    assert np.all(uncertainty["confidence"] >= 1.0 / 3.0)
    assert np.all(uncertainty["normalized_entropy"] >= 0)
    assert np.all(uncertainty["normalized_entropy"] <= 1.0)
    assert np.array_equal(uncertainty["hard_labels"], probabilities.argmax(axis=1))
    assert parameter_count("diag", 12, 9) == 227
    assert parameter_count("full", 12, 9) == 659
    assert isinstance(gmm.converged_, bool)
    with pytest.raises(SchemaError):
        assignments_frame(["a", "b"], uncertainty["hard_labels"][:1])


def test_bundle_roundtrip_predictions_and_metadata(fixture_env, tmp_path):
    _, config, _ = fixture_env
    X, _ = _blobs(n=120, centers=3)
    ids = [f"t{i:03d}" for i in range(len(X))]
    bundle, transformed = fit_clustering_bundle(
        X,
        ids,
        algorithm="kmeans",
        k=3,
        config=config,
        seed=42,
        feature_names=[
            "acousticness", "danceability", "energy", "valence", "tempo",
            "loudness", "speechiness", "instrumentalness", "liveness",
        ],
        scaler="standard",
        pca_variance=None,
    )
    path = save_bundle(bundle, tmp_path / "model.joblib")
    loaded = load_bundle(path)
    assert isinstance(loaded, ClusteringBundle)
    assert loaded.algorithm == "kmeans"
    assert np.array_equal(loaded.predict(X), bundle.predict(X))
    assert loaded.trained_on_ids_sha256 == bundle.trained_on_ids_sha256
    assert np.isfinite(transformed).all()


def test_robust_scaler_handles_zero_iqr_feature():
    rng = np.random.default_rng(11)
    X = rng.normal(size=(100, 9))
    X[:, 3] = 0.5  # constant column: Q1 == Q3, IQR == 0
    transformer = make_feature_transformer("robust", None)
    transformer.fit(X)
    scale = transformer.named_steps["scaler"].scale_
    assert scale[3] == pytest.approx(1.0)
    transformed = transformer.transform(X)
    assert np.isfinite(transformed).all()
    assert np.allclose(transformed[:, 3], 0.0)


def test_gmm_default_covariance_is_recorded_and_criteria_are_finite(fixture_env):
    _, config, _ = fixture_env
    X, _ = _blobs(seed=13)
    ids = [f"t{i:03d}" for i in range(len(X))]
    bundle, transformed = fit_clustering_bundle(
        X,
        ids,
        algorithm="gmm",
        k=3,
        config=config,
        seed=42,
        feature_names=list(F9_FEATURES),
        scaler="standard",
        pca_variance=None,
        covariance_type=None,
        n_init=1,
    )
    assert bundle.covariance_type == "diag"
    info = gmm_information_criteria(
        bundle.estimator,
        transformed,
        k=3,
        d=transformed.shape[1],
        covariance_type=bundle.covariance_type,
    )
    assert np.isfinite(info["bic"])
    assert np.isfinite(info["aic"])
