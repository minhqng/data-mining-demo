"""Feature transformations that are fitted on one explicit cohort only."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, RobustScaler, StandardScaler

from .errors import SchemaError

ScalerName = Literal["standard", "robust", "none"]


def make_feature_transformer(
    scaler: ScalerName,
    pca_variance: float | None,
    *,
    whiten: bool = False,
) -> Pipeline:
    """Build a scaler (optionally followed by PCA).

    The returned object is *not* fitted here. Call ``.fit``/``.fit_transform``
    only with the development fit cohort. PCA always follows scaling because
    scikit-learn PCA centers but does not standardize features.
    """
    if scaler == "standard":
        steps: list[tuple[str, Any]] = [("scaler", StandardScaler())]
    elif scaler == "robust":
        steps = [("scaler", RobustScaler())]
    elif scaler == "none":
        # Intentionally unscaled diagnostic (E00) only. Never a selectable model.
        steps = [("scaler", FunctionTransformer(validate=False))]
    else:
        raise SchemaError(f"Unknown scaler {scaler!r}; expected standard/robust/none.")
    if scaler == "none" and pca_variance is not None:
        raise SchemaError("The unscaled E00 diagnostic does not support PCA.")
    if pca_variance is not None:
        variance = float(pca_variance)
        if not 0.0 < variance < 1.0:
            raise SchemaError("PCA variance must be in (0, 1) or None.")
        if whiten:
            raise SchemaError(
                "PCA whitening is not part of the primary protocol; do not enable it silently."
            )
        steps.append(
            (
                "pca",
                PCA(
                    n_components=variance,
                    whiten=False,
                    svd_solver="full",
                ),
            )
        )
    return Pipeline(steps)


def fit_transform_finite(
    transformer: Pipeline, X_fit: np.ndarray
) -> tuple[Pipeline, np.ndarray]:
    transformed = transformer.fit_transform(X_fit)
    matrix = np.asarray(transformed, dtype=np.float64)
    if not np.isfinite(matrix).all():
        raise SchemaError("Fitted transformer produced NaN/inf values.")
    return transformer, matrix


def transform_finite(transformer: Pipeline, X: np.ndarray) -> np.ndarray:
    matrix = np.asarray(transformer.transform(X), dtype=np.float64)
    if not np.isfinite(matrix).all():
        raise SchemaError("Transformed matrix contains NaN/inf; refusing to model it.")
    return matrix


def transformed_feature_names(
    transformer: Pipeline, input_feature_names: Sequence[str]
) -> list[str]:
    steps = transformer.named_steps
    if "pca" in steps:
        n_components = int(steps["pca"].n_components_)
        return [f"PC{index + 1}" for index in range(n_components)]
    return list(input_feature_names)


def pca_diagnostics(
    transformer: Pipeline, input_feature_names: Sequence[str]
) -> dict[str, Any]:
    """Return actual PCA components, variance curve and loadings, if present."""
    steps = transformer.named_steps
    names = list(input_feature_names)
    if "pca" not in steps:
        scaler = steps["scaler"]
        variance = np.asarray(getattr(scaler, "var_", []), dtype=float)
        return {
            "pca_present": False,
            "pca_variance_requested": None,
            "pca_actual_components": None,
            "explained_variance_ratio": None,
            "cumulative_explained_variance": None,
            "loadings": None,
            "input_features": names,
            "scaler_variances": variance.tolist() if variance.size else None,
        }
    pca: PCA = steps["pca"]
    ratios = np.asarray(pca.explained_variance_ratio_, dtype=float)
    loadings = pd.DataFrame(
        pca.components_,
        index=[f"PC{index + 1}" for index in range(pca.components_.shape[0])],
        columns=names,
    )
    return {
        "pca_present": True,
        "pca_variance_requested": None,
        "pca_actual_components": int(pca.n_components_),
        "explained_variance_ratio": ratios.tolist(),
        "cumulative_explained_variance": np.cumsum(ratios).tolist(),
        "loadings": loadings.reset_index(names="component").to_dict(orient="records"),
        "singular_values": np.asarray(pca.singular_values_, dtype=float).tolist(),
        "input_features": names,
    }


def pca_loadings_frame(diagnostics: dict[str, Any]) -> pd.DataFrame:
    loadings = diagnostics.get("loadings")
    if not loadings:
        return pd.DataFrame()
    return pd.DataFrame(loadings)


def pca_variance_frame(diagnostics: dict[str, Any]) -> pd.DataFrame:
    ratios = diagnostics.get("explained_variance_ratio")
    if not ratios:
        return pd.DataFrame(columns=["component", "explained_variance_ratio", "cumulative"])
    cumulative = diagnostics["cumulative_explained_variance"]
    return pd.DataFrame(
        {
            "component": [f"PC{index + 1}" for index in range(len(ratios))],
            "explained_variance_ratio": [float(value) for value in ratios],
            "cumulative_explained_variance": [float(value) for value in cumulative],
        }
    )
