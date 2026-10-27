"""K-Means / GMM estimators and a serialisable preprocessing+model bundle."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline

from .config import Config
from .errors import ArtifactError, SchemaError
from .preprocess import make_feature_transformer, transform_finite
from .utils import atomic_write_text, library_versions, sha256_obj

BUNDLE_SCHEMA = "btl11.bundle.v1"
Algorithm = Literal["kmeans", "gmm"]


def create_clusterer(
    algorithm: Algorithm,
    k: int,
    config: Config,
    seed: int,
    *,
    covariance_type: str | None = None,
    n_init: int | None = None,
    reg_covar: float | None = None,
) -> KMeans | GaussianMixture:
    if k < 2:
        raise SchemaError("The number of clusters/components must be >= 2.")
    if algorithm == "kmeans":
        return KMeans(
            n_clusters=int(k),
            init=config.kmeans["init"],
            n_init=int(n_init or config.kmeans["n_init"]),
            max_iter=int(config.kmeans["max_iter"]),
            tol=float(config.kmeans["tol"]),
            random_state=int(seed),
            algorithm="lloyd",
        )
    if algorithm == "gmm":
        covariance = covariance_type or "diag"
        if covariance not in {"diag", "full"}:
            raise SchemaError("GMM covariance_type must be diag/full.")
        return GaussianMixture(
            n_components=int(k),
            covariance_type=covariance,
            tol=1e-3,
            reg_covar=float(reg_covar if reg_covar is not None else config.gmm["reg_covar_quick"]),
            max_iter=int(config.gmm["max_iter"]),
            n_init=int(n_init or config.gmm["n_init_quick"]),
            init_params=config.gmm["init_params"],
            random_state=int(seed),
        )
    raise SchemaError(f"Unknown algorithm {algorithm!r}; expected kmeans/gmm.")


@dataclass
class ClusteringBundle:
    """Fitted transformer + estimator plus immutable provenance metadata.

    Only load bundles produced locally by this project. ``joblib`` uses pickle,
    so a bundle from an untrusted source is executable code.
    """

    transformer: Pipeline
    estimator: KMeans | GaussianMixture
    algorithm: str
    k: int
    feature_names: tuple[str, ...]
    scaler: str
    pca_variance: float | None
    pca_actual_components: int | None
    seed: int
    covariance_type: str | None
    feature_space_id: str
    trained_on_ids_sha256: str
    n_fit: int
    bundle_schema: str = BUNDLE_SCHEMA
    library_versions_: dict[str, str | None] = field(default_factory=dict)

    def transform(self, X_raw: np.ndarray) -> np.ndarray:
        return transform_finite(self.transformer, X_raw)

    def predict(self, X_raw: np.ndarray) -> np.ndarray:
        transformed = self.transform(X_raw)
        return np.asarray(self.estimator.predict(transformed), dtype=np.int64)

    def predict_proba(self, X_raw: np.ndarray) -> np.ndarray:
        if not isinstance(self.estimator, GaussianMixture):
            raise SchemaError("predict_proba is only defined for the GMM bundle.")
        transformed = self.transform(X_raw)
        probabilities = np.asarray(self.estimator.predict_proba(transformed), dtype=np.float64)
        if not np.isfinite(probabilities).all():
            raise SchemaError("GMM posterior contains NaN/inf.")
        return probabilities

    def labels_on_transformed(self, transformed: np.ndarray) -> np.ndarray:
        return np.asarray(self.estimator.predict(transformed), dtype=np.int64)

    @property
    def is_gmm(self) -> bool:
        return isinstance(self.estimator, GaussianMixture)

    def describe(self) -> dict[str, Any]:
        return {
            "bundle_schema": self.bundle_schema,
            "algorithm": self.algorithm,
            "k": int(self.k),
            "feature_names": list(self.feature_names),
            "scaler": self.scaler,
            "pca_variance": self.pca_variance,
            "pca_actual_components": self.pca_actual_components,
            "seed": int(self.seed),
            "covariance_type": self.covariance_type,
            "feature_space_id": self.feature_space_id,
            "trained_on_ids_sha256": self.trained_on_ids_sha256,
            "n_fit": int(self.n_fit),
            "library_versions": dict(self.library_versions_),
        }


def fit_clustering_bundle(
    X_fit: np.ndarray,
    fit_ids: Sequence[str],
    *,
    algorithm: Algorithm,
    k: int,
    config: Config,
    seed: int,
    feature_names: Sequence[str],
    scaler: str,
    pca_variance: float | None,
    covariance_type: str | None = None,
    n_init: int | None = None,
    reg_covar: float | None = None,
) -> tuple[ClusteringBundle, np.ndarray]:
    transformer = make_feature_transformer(
        scaler=scaler,
        pca_variance=pca_variance,
        whiten=bool(config.preprocess["pca_whiten"]),
    )
    transformed = np.asarray(transformer.fit_transform(X_fit), dtype=np.float64)
    if not np.isfinite(transformed).all():
        raise SchemaError("Scaler/PCA produced non-finite values on the fit cohort.")
    resolved_covariance = (
        (covariance_type or "diag") if algorithm == "gmm" else None
    )
    estimator = create_clusterer(
        algorithm,
        k,
        config,
        seed,
        covariance_type=resolved_covariance,
        n_init=n_init,
        reg_covar=reg_covar,
    )
    estimator.fit(transformed)
    pca = transformer.named_steps.get("pca")
    pca_components = int(pca.n_components_) if pca is not None else None
    feature_space_id = _feature_space_id(feature_names, scaler, pca_variance)
    bundle = ClusteringBundle(
        transformer=transformer,
        estimator=estimator,
        algorithm=algorithm,
        k=int(k),
        feature_names=tuple(feature_names),
        scaler=scaler,
        pca_variance=None if pca_variance is None else float(pca_variance),
        pca_actual_components=pca_components,
        seed=int(seed),
        covariance_type=resolved_covariance,
        feature_space_id=feature_space_id,
        trained_on_ids_sha256=sha256_obj(sorted(str(track_id) for track_id in fit_ids)),
        n_fit=int(len(fit_ids)),
        library_versions_=library_versions(["numpy", "pandas", "scikit-learn", "scipy"]),
    )
    return bundle, transformed


def _feature_space_id(
    feature_names: Sequence[str], scaler: str, pca_variance: float | None
) -> str:
    feature_set = "f9" if len(feature_names) == 9 else "f5" if len(feature_names) == 5 else "custom"
    pca_label = "none" if pca_variance is None else f"pca{int(float(pca_variance) * 100)}"
    return f"{feature_set}|{scaler}|{pca_label}|whiten=False"


def save_bundle(bundle: ClusteringBundle, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(f".{target.name}.tmp")
    joblib.dump(bundle, temp)
    temp.replace(target)
    return target


def load_bundle(path: str | Path) -> ClusteringBundle:
    """Load a trusted local bundle and validate its schema/class identity."""
    source = Path(path)
    if not source.exists():
        raise ArtifactError(f"Model bundle not found: {source}")
    bundle = joblib.load(source)
    if not isinstance(bundle, ClusteringBundle):
        raise ArtifactError(f"{source} did not contain a ClusteringBundle.")
    if bundle.bundle_schema != BUNDLE_SCHEMA:
        raise ArtifactError(
            f"Unsupported bundle schema {bundle.bundle_schema!r}; expected {BUNDLE_SCHEMA}."
        )
    if bundle.algorithm == "kmeans" and not isinstance(bundle.estimator, KMeans):
        raise ArtifactError("Bundle algorithm/class mismatch for K-Means.")
    if bundle.algorithm == "gmm" and not isinstance(bundle.estimator, GaussianMixture):
        raise ArtifactError("Bundle algorithm/class mismatch for GMM.")
    if not isinstance(bundle.transformer, Pipeline):
        raise ArtifactError("Bundle transformer is not a scikit-learn Pipeline.")
    return bundle


def parameter_count(covariance_type: str, k: int, d: int) -> int:
    """Number of free GMM parameters, per the SSOT formulas."""
    if k < 1 or d < 1:
        raise SchemaError("k and d must be positive.")
    if covariance_type == "diag":
        return int(k * (2 * d + 1) - 1)
    if covariance_type == "full":
        return int((k - 1) + k * d + k * d * (d + 1) / 2)
    raise SchemaError("Only diag/full parameter counting is supported.")


def gmm_uncertainty(probabilities: np.ndarray) -> dict[str, np.ndarray]:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 2:
        raise SchemaError("Posterior must be a 2-D array.")
    k = probabilities.shape[1]
    row_sums = probabilities.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-8):
        raise SchemaError("GMM responsibilities do not sum to one.")
    clipped = np.clip(probabilities, 1e-300, 1.0)
    entropy = -np.sum(probabilities * np.log(clipped), axis=1)
    normalized = entropy / np.log(k) if k >= 2 else np.zeros_like(entropy)
    confidence = probabilities.max(axis=1)
    hard_labels = probabilities.argmax(axis=1).astype(np.int64)
    return {
        "confidence": confidence,
        "entropy": entropy,
        "normalized_entropy": np.clip(normalized, 0.0, 1.0),
        "hard_labels": hard_labels,
    }


def assignments_frame(
    ids: Sequence[str],
    labels: np.ndarray,
    *,
    probabilities: np.ndarray | None = None,
) -> pd.DataFrame:
    ids = [str(track_id) for track_id in ids]
    labels = np.asarray(labels, dtype=np.int64)
    if len(ids) != len(labels):
        raise SchemaError("Assignment IDs and labels have different lengths.")
    frame = pd.DataFrame({"track_id": ids, "cluster": labels})
    if probabilities is not None:
        uncertainty = gmm_uncertainty(probabilities)
        if not np.array_equal(uncertainty["hard_labels"], labels):
            raise SchemaError("Hard labels must equal argmax(predict_proba).")
        frame["confidence"] = uncertainty["confidence"]
        frame["entropy"] = uncertainty["entropy"]
        frame["normalized_entropy"] = uncertainty["normalized_entropy"]
    if frame["track_id"].duplicated().any():
        raise SchemaError("Assignments contain duplicate track IDs.")
    return frame.sort_values("track_id", kind="mergesort").reset_index(drop=True)


def write_bundle_metadata(bundle: ClusteringBundle, path: str | Path) -> Path:
    import json

    return atomic_write_text(
        path,
        json.dumps(bundle.describe(), indent=2, sort_keys=True) + "\n",
    )
