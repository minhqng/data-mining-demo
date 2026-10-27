"""Post-hoc run visualizations: PCA2D, t-SNE, profile, uncertainty, genre."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from .config import Config
from .data import PreparedData, select_ids_by_hash
from .errors import ArtifactError, ConfigError, DataError
from .experiment import load_run_doc, run_dir
from .models import load_bundle
from .profiling import generate_genre_posthoc, generate_profiles
from .utils import atomic_write_csv, atomic_write_json, sha256_obj, utc_now
from .visualization import plot_pca_scatter, plot_tsne


def visualize_run(
    config: Config,
    prepared: PreparedData,
    run_id: str,
    kinds: Sequence[str],
) -> dict[str, Any]:
    requested = [kind.strip().lower() for kind in kinds if kind.strip()]
    allowed = {"pca", "tsne", "profile", "genre"}
    unknown = sorted(set(requested) - allowed)
    if unknown:
        raise ConfigError(f"Unknown visualization kinds {unknown}; allowed: {sorted(allowed)}.")
    doc = load_run_doc(config, run_id)
    if doc.get("status") != "completed":
        raise ArtifactError(f"Run {run_id} is not completed; refusing to visualize.")
    path = run_dir(config, run_id)
    assignments_path = path / "assignments.parquet"
    if not assignments_path.exists():
        raise ArtifactError(f"Run {run_id} has no assignments.parquet.")
    assignments = pd.read_parquet(assignments_path)
    assignments["track_id"] = assignments["track_id"].astype(str)
    feature_names = list(doc.get("feature_order") or config.feature_primary)
    bundle = load_bundle(path / "model.joblib") if doc.get("has_model") else None
    if bundle is None and ({"pca", "tsne", "profile"} & set(requested)):
        raise ArtifactError(
            f"Run {run_id} has no trusted model bundle; PCA/t-SNE/profile need the fitted "
            "transformer. (Diagnostic runs such as stability have no single model.)"
        )
    outputs: dict[str, Any] = {}
    posthoc_dir = Path("artifacts/posthoc") / run_id
    (posthoc_dir / "figures").mkdir(parents=True, exist_ok=True)
    (posthoc_dir / "tables").mkdir(parents=True, exist_ok=True)
    if "pca" in requested:
        outputs["pca"] = _visualize_pca(
            config, prepared, doc, assignments, bundle, feature_names, run_id, posthoc_dir
        )
    if "tsne" in requested:
        outputs["tsne"] = _visualize_tsne(
            config, prepared, doc, assignments, bundle, feature_names, run_id, posthoc_dir
        )
    if "profile" in requested:
        if (path / "profiles.csv").exists():
            outputs["profile"] = {
                "skipped": "profiles.csv already exists; immutable run artifacts are not overwritten"
            }
        else:
            outputs["profile"] = generate_profiles(
                config,
                prepared,
                posthoc_dir,
                run_id,
                assignments,
                bundle=bundle,
                all_tracks_scope=doc.get("run_type") == "final_refit",
                feature_names=feature_names,
            )
    if "genre" in requested:
        if doc.get("run_type") != "final_refit":
            raise ConfigError(
                "The multi-label genre post-hoc view is unlocked only for the final refit "
                "after the model is frozen."
            )
        genre_table = path / "tables" / "genre_cluster_incidence.csv"
        if genre_table.exists():
            outputs["genre"] = {
                "skipped": "genre tables already exist; immutable run artifacts are not overwritten"
            }
        else:
            outputs["genre"] = generate_genre_posthoc(
                config,
                prepared,
                posthoc_dir,
                run_id,
                assignments,
                all_tracks_scope=True,
            )
    payload = {
        "run_id": run_id,
        "generated_utc": utc_now(),
        "kinds": requested,
        "feature_space_id": doc.get("feature_space_id"),
        "n_assignments": int(len(assignments)),
        "outputs": outputs,
        "visual_caveats": {
            "pca2": "PCA2D is a display projection, not the default clustering representation.",
            "tsne": "t-SNE is visual only; distances, densities and K must not be read from it.",
            "sampling": "Deterministic bounded samples; underlying coordinates are exported as CSV.",
            "location": "Post-hoc views are written under artifacts/posthoc/<run_id>/ and do not mutate run artifacts.",
        },
    }
    atomic_write_json(posthoc_dir / "visualization.json", payload)
    return payload


def _visualize_pca(
    config: Config,
    prepared: PreparedData,
    doc: dict[str, Any],
    assignments: pd.DataFrame,
    bundle: Any,
    feature_names: Sequence[str],
    run_id: str,
    outdir: Path,
) -> dict[str, Any]:
    ids = assignments["track_id"].tolist()
    X = prepared.feature_matrix(ids, feature_names)
    Xt = bundle.transform(np.asarray(X, dtype=np.float64))
    pca2 = PCA(n_components=2, svd_solver="full", random_state=int(config.seed))
    coords = pca2.fit_transform(Xt)
    sample_size = min(int(config.visualization["scatter_sample_size"]), len(ids))
    sample_ids = select_ids_by_hash(ids, sample_size, "pca_scatter", int(config.seed))
    position = {track_id: index for index, track_id in enumerate(ids)}
    sample_positions = np.array([position[track_id] for track_id in sample_ids], dtype=np.int64)
    coordinate_frame = pd.DataFrame(
        {
            "track_id": sample_ids,
            "cluster": assignments.set_index("track_id").reindex(sample_ids)["cluster"].to_numpy(dtype=int),
            "PC1": coords[sample_positions, 0],
            "PC2": coords[sample_positions, 1],
        }
    )
    table_path = atomic_write_csv(outdir / "tables" / "pca2_coordinates.csv", coordinate_frame)
    figure = plot_pca_scatter(
        coordinate_frame,
        outdir / "figures" / "pca2_scatter.png",
        title=(
            f"PCA2D display | {doc.get('algorithm', '').upper()} K={doc.get('k')} | "
            f"{doc.get('feature_space_id')} | sample n={len(sample_ids)} | run {run_id}"
        ),
        explained=pca2.explained_variance_ratio_,
    )
    return {
        "figure": str(figure),
        "coordinates": str(table_path),
        "sample_n": int(len(sample_ids)),
        "explained_variance_ratio": pca2.explained_variance_ratio_.tolist(),
        "sample_ids_sha256": sha256_obj(sample_ids),
        "note": (
            "PCA(2) is re-fitted on the fitted model's transformed feature space for display. "
            "It does not replace the model representation."
        ),
    }


def _visualize_tsne(
    config: Config,
    prepared: PreparedData,
    doc: dict[str, Any],
    assignments: pd.DataFrame,
    bundle: Any,
    feature_names: Sequence[str],
    run_id: str,
    outdir: Path,
) -> dict[str, Any]:
    ids = assignments["track_id"].tolist()
    n = min(int(config.visualization["tsne_sample_size"]), len(ids))
    if n < 10:
        raise DataError("t-SNE needs at least 10 assignment points.")
    sample_ids = select_ids_by_hash(ids, n, "tsne_sample", int(config.visualization["tsne_random_state"]))
    X = prepared.feature_matrix(sample_ids, feature_names)
    Xt = bundle.transform(np.asarray(X, dtype=np.float64))
    perplexity = float(config.visualization["tsne_perplexity"])
    perplexity = min(perplexity, max(1.0, (len(sample_ids) - 1) / 3.0))
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        max_iter=1000,
        random_state=int(config.visualization["tsne_random_state"]),
    )
    embedding = tsne.fit_transform(Xt)
    labels = assignments.set_index("track_id").reindex(sample_ids)["cluster"].to_numpy(dtype=int)
    frame = pd.DataFrame(
        {
            "track_id": sample_ids,
            "cluster": labels,
            "tsne_1": embedding[:, 0],
            "tsne_2": embedding[:, 1],
        }
    )
    table_path = atomic_write_csv(outdir / "tables" / "tsne_coordinates.csv", frame)
    figure = plot_tsne(
        frame,
        outdir / "figures" / "tsne2d.png",
        title=(
            f"t-SNE display | {doc.get('algorithm', '').upper()} K={doc.get('k')} | "
            f"{doc.get('feature_space_id')} | sample n={len(sample_ids)} | run {run_id}"
        ),
        perplexity=perplexity,
    )
    return {
        "figure": str(figure),
        "coordinates": str(table_path),
        "sample_n": int(len(sample_ids)),
        "perplexity": perplexity,
        "random_state": int(config.visualization["tsne_random_state"]),
        "sample_ids_sha256": sha256_obj(sample_ids),
        "note": "Visual only: not used to select K, models, or to compute validity metrics.",
    }
