"""PCA diagnostics fitted only on the development fit cohort."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import Config
from .data import PreparedData
from .preprocess import (
    make_feature_transformer,
    pca_diagnostics,
    pca_loadings_frame,
    pca_variance_frame,
    transformed_feature_names,
)
from .utils import atomic_write_csv, atomic_write_json, utc_now
from .visualization import plot_pca_diagnostics


def run_pca_diagnostics(config: Config, prepared: PreparedData) -> dict[str, Any]:
    stamp = utc_now().replace(":", "").replace("-", "")
    outdir = Path("artifacts/pca") / stamp
    (outdir / "tables").mkdir(parents=True, exist_ok=False)
    (outdir / "figures").mkdir(parents=True, exist_ok=True)
    fit_ids = list(prepared.splits["fit"])
    feature_names = list(config.feature_primary)
    X_fit = prepared.feature_matrix(fit_ids, feature_names)
    results: dict[str, Any] = {}
    for candidate in config.preprocess["pca_candidates"]:
        label = "none" if candidate in (None, "none") else f"pca{int(float(candidate) * 100)}"
        if candidate in (None, "none"):
            transformer = make_feature_transformer(
                config.preprocess["scaler_primary"], None,
                whiten=bool(config.preprocess["pca_whiten"]),
            )
            transformer.fit(X_fit)
            transformed_names = transformed_feature_names(transformer, feature_names)
            results[label] = {
                "pca_present": False,
                "n_components": len(transformed_names),
                "feature_names": transformed_names,
            }
            continue
        transformer = make_feature_transformer(
            config.preprocess["scaler_primary"],
            float(candidate),
            whiten=bool(config.preprocess["pca_whiten"]),
        )
        transformer.fit(X_fit)
        diagnostics = pca_diagnostics(transformer, feature_names)
        variance = pca_variance_frame(diagnostics)
        loadings = pca_loadings_frame(diagnostics)
        atomic_write_csv(outdir / "tables" / f"explained_variance_{label}.csv", variance)
        atomic_write_csv(outdir / "tables" / f"loadings_{label}.csv", loadings)
        figure = plot_pca_diagnostics(
            diagnostics, outdir / "figures" / f"pca_diagnostics_{label}.png"
        )
        results[label] = {
            "pca_present": True,
            "variance_requested": float(candidate),
            "n_components": int(diagnostics["pca_actual_components"]),
            "cumulative_final": float(diagnostics["cumulative_explained_variance"][-1]),
            "explained_variance_ratio": diagnostics["explained_variance_ratio"],
            "cumulative_explained_variance": diagnostics["cumulative_explained_variance"],
            "loadings_table": str(outdir / "tables" / f"loadings_{label}.csv"),
            "variance_table": str(outdir / "tables" / f"explained_variance_{label}.csv"),
            "figure": str(figure),
        }
    payload = {
        "generated_utc": utc_now(),
        "fit_scope": "development fit cohort only; eval and holdout never used",
        "n_fit": len(fit_ids),
        "fit_ids_sha256": prepared.split_hashes.get("fit"),
        "dataset_sha256": prepared.dataset_sha256,
        "scaler": config.preprocess["scaler_primary"],
        "whiten": bool(config.preprocess["pca_whiten"]),
        "results": results,
        "note": (
            "PCA is a linear variance-retention transform; PCA95 is not 95% cluster accuracy. "
            "Visualization PCA2D is not the default training representation."
        ),
    }
    atomic_write_json(outdir / "pca_diagnostics.json", payload)
    return {"output_dir": str(outdir), **payload}
