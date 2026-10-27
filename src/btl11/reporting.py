"""Machine-generated report tables and figure index from actual run artifacts."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from .config import Config
from .data import PreparedData
from .errors import ArtifactError
from .evaluation import cohort_identifier, rank_bic_within_space
from .experiment import iter_run_docs, load_run_doc, run_dir
from .utils import (
    atomic_write_csv,
    atomic_write_json,
    atomic_write_text,
    read_json,
    utc_now,
)


def _metric_rows(metrics_doc: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, entry in (metrics_doc.get("metrics") or {}).items():
        if not isinstance(entry, Mapping):
            continue
        rows.append(
            {
                "metric": name,
                "value": entry.get("value"),
                "direction": entry.get("direction"),
                "space": entry.get("space"),
                "n_eval": entry.get("n_eval"),
                "method": entry.get("method"),
                "reason": entry.get("reason"),
            }
        )
    return rows


def _md_table(frame: pd.DataFrame, columns: Sequence[str] | None = None, limit: int | None = None) -> str:
    if frame.empty:
        return "_NOT RUN / no rows._"
    subset = frame.copy()
    if columns:
        subset = subset.loc[:, [column for column in columns if column in subset.columns]]
    if limit is not None:
        subset = subset.head(limit)
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


def _latest(pattern: str) -> Path | None:
    matches = sorted(Path().glob(pattern))
    return matches[-1] if matches else None


def _display_path(path: Path) -> str:
    """Prefer a repo-relative POSIX path; keep the path when it lies outside the repo."""
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _run_table(config: Config, experiment_set: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path, doc in iter_run_docs(config):
        if doc.get("experiment_set") != experiment_set or doc.get("status") != "completed":
            continue
        metrics_path = path / "metrics.json"
        if not metrics_path.exists():
            continue
        metrics = read_json(metrics_path)
        flat = {row["metric"]: row["value"] for row in _metric_rows(metrics)}
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
                "feature_space_id": metrics.get("context", {}).get("feature_space_id")
                or doc.get("feature_space_id"),
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
                "silhouette": flat.get("silhouette"),
                "davies_bouldin": flat.get("davies_bouldin"),
                "calinski_harabasz": flat.get("calinski_harabasz"),
                "bic": flat.get("bic"),
                "aic": flat.get("aic"),
                "inertia_fit": flat.get("inertia_fit"),
                "inertia_eval": flat.get("inertia_eval"),
                "min_cluster_share_fit": flat.get("min_cluster_share_fit"),
                "empty_clusters_fit": flat.get("empty_clusters_fit"),
                "fit_seconds": doc.get("fit_seconds"),
                "run_path": str(path),
            }
        )
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = rank_bic_within_space(frame)
        frame = frame.sort_values(["algorithm", "k", "covariance_type"]).reset_index(drop=True)
    return frame


def run_report(
    config: Config,
    prepared: PreparedData,
    selected_run_id: str,
    output_dir: str | Path,
) -> dict[str, Any]:
    doc = load_run_doc(config, selected_run_id)
    if doc.get("status") != "completed":
        raise ArtifactError("Selected report run is not completed.")
    if doc.get("run_type") != "final_refit":
        raise ArtifactError(
            "The final report must cite a final_refit run, not an evaluation candidate. "
            f"Run {selected_run_id} is {doc.get('run_type')!r}."
        )
    if doc.get("dataset_sha256") != prepared.dataset_sha256:
        raise ArtifactError(
            "The selected final refit was produced from different dataset bytes than the "
            "current prepared snapshot; regenerate the refit before reporting."
        )
    if not doc.get("holdout_run_id"):
        raise ArtifactError(
            "The final refit run has no one-time holdout reference. Refit only after the "
            "frozen model has passed `btl11 validate-holdout`."
        )
    holdout_doc = load_run_doc(config, str(doc["holdout_run_id"]))
    if holdout_doc.get("status") != "completed" or holdout_doc.get("run_type") != "holdout":
        raise ArtifactError("The referenced holdout run is not a completed holdout.")
    if holdout_doc.get("dataset_sha256") != prepared.dataset_sha256:
        raise ArtifactError("The referenced holdout was produced from different dataset bytes.")
    if holdout_doc.get("frozen_run_id") != doc.get("evaluation_model_id"):
        raise ArtifactError(
            "The final refit did not use the exact evaluation model that the holdout "
            "validated; its provenance chain is broken."
        )
    evaluation_doc = load_run_doc(config, str(doc["evaluation_model_id"]))
    if (
        evaluation_doc.get("status") != "completed"
        or evaluation_doc.get("run_type") != "candidate"
        or evaluation_doc.get("dataset_sha256") != prepared.dataset_sha256
    ):
        raise ArtifactError(
            "The evaluation model cited by the final refit is missing, incomplete or "
            "belongs to a different dataset snapshot."
        )
    frozen_descriptor = doc.get("frozen_descriptor")
    if frozen_descriptor is not None and doc.get("descriptor") != frozen_descriptor:
        raise ArtifactError(
            "The final refit descriptor differs from its frozen descriptor; "
            "selection changed after freeze."
        )
    path = run_dir(config, selected_run_id)
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    manifest = prepared.manifest
    summary = prepared.summary
    audit = prepared.audit
    dataset_rows = pd.DataFrame(
        [
            {
                "dataset_sha256": manifest.get("sha256"),
                "source": manifest.get("source"),
                "kaggle_dataset_id": manifest.get("kaggle_dataset_id"),
                "raw_bytes": manifest.get("bytes"),
                "raw_rows": audit.get("counts", {}).get("n_rows"),
                "unique_tracks": summary.get("unique_track_ids"),
                "n_genres": summary.get("n_track_genres"),
                "tracks_with_multiple_genres": summary.get("tracks_with_multiple_genres"),
                "genre_bridge_rows": summary.get("genre_bridge_rows"),
                "n_exact_duplicate_rows": audit.get("counts", {}).get("n_exact_duplicate_rows"),
                "n_duplicate_track_ids": audit.get("counts", {}).get("n_duplicate_track_ids"),
                "n_conflicting_track_ids": summary.get("conflicting_track_ids"),
                "n_quarantined_rows": summary.get("dropped_rows"),
                "license_review_status": manifest.get("license_review_status"),
                "redistribution_allowed": manifest.get("redistribution_allowed"),
                "local_analysis_authorized": manifest.get("local_analysis_authorized_by_owner"),
                "kaggle_page_license": manifest.get("license_metadata_observed", {}).get(
                    "kaggle_page_license"
                ),
                "huggingface_mirror_license": manifest.get("license_metadata_observed", {}).get(
                    "huggingface_mirror_license"
                ),
                "license_metadata_discrepancy": manifest.get("license_metadata_observed", {}).get(
                    "discrepancy"
                ),
                "dev_tracks": summary.get("splits", {}).get("dev"),
                "fit_tracks": summary.get("splits", {}).get("fit"),
                "eval_tracks": summary.get("splits", {}).get("eval"),
                "holdout_tracks": summary.get("splits", {}).get("holdout"),
            }
        ]
    )
    atomic_write_csv(outdir / "dataset_summary.csv", dataset_rows)
    baseline_set = str(config.selection.get("baseline_experiment_set", "baseline_f9_standard"))
    baseline = _run_table(config, baseline_set)
    if baseline.empty:
        atomic_write_csv(outdir / "baseline_all.csv", pd.DataFrame())
        atomic_write_csv(outdir / "kmeans_baseline.csv", pd.DataFrame())
        atomic_write_csv(outdir / "gmm_baseline.csv", pd.DataFrame())
    else:
        atomic_write_csv(outdir / "baseline_all.csv", baseline)
        atomic_write_csv(
            outdir / "kmeans_baseline.csv",
            baseline[baseline["algorithm"] == "kmeans"],
        )
        atomic_write_csv(
            outdir / "gmm_baseline.csv",
            baseline[baseline["algorithm"] == "gmm"],
        )
    ablation_tables: dict[str, pd.DataFrame] = {}
    for factor in ("scaler", "feature_set", "pca"):
        latest = _latest(f"artifacts/ablations/{factor}/*/comparison.csv")
        if latest is not None:
            frame = pd.read_csv(latest)
            ablation_tables[factor] = frame
            atomic_write_csv(outdir / f"ablation_{factor}.csv", frame)
        else:
            ablation_tables[factor] = pd.DataFrame()
            atomic_write_csv(outdir / f"ablation_{factor}.csv", pd.DataFrame())
    stability_summary = pd.DataFrame()
    stability_pairs = pd.DataFrame()
    stability_run = None
    stability_candidates = [
        (path, doc)
        for path, doc in iter_run_docs(config)
        if doc.get("run_type") == "stability" and doc.get("status") == "completed"
    ]
    if stability_candidates:
        stability_candidates.sort(key=lambda item: item[1].get("created_utc", ""))
        stability_path, stability_doc = stability_candidates[-1]
        stability_run = stability_doc.get("run_id")
        summary_path = stability_path / "tables" / "stability_summary.csv"
        pairs_path = stability_path / "tables" / "ari_pairs.csv"
        if summary_path.exists():
            stability_summary = pd.read_csv(summary_path)
            atomic_write_csv(outdir / "stability_summary.csv", stability_summary)
        if pairs_path.exists():
            stability_pairs = pd.read_csv(pairs_path)
            atomic_write_csv(outdir / "stability_ari_pairs.csv", stability_pairs)
    holdout_summary = pd.DataFrame()
    holdout_run_id = doc.get("holdout_run_id")
    if holdout_run_id:
        holdout_path = Path(config.output_dir) / holdout_run_id
        holdout_metrics_path = holdout_path / "metrics.json"
        if holdout_metrics_path.exists():
            holdout = read_json(holdout_metrics_path)
            holdout_summary = pd.DataFrame(_metric_rows(holdout))
            atomic_write_csv(outdir / "holdout_metrics.csv", holdout_summary)
    profiles = pd.DataFrame()
    profiles_path = path / "profiles.csv"
    if profiles_path.exists():
        profiles = pd.read_csv(profiles_path)
        atomic_write_csv(outdir / "final_cluster_profiles.csv", profiles)
    representative_path = path / "tables" / "representative_tracks.csv"
    if representative_path.exists():
        atomic_write_csv(
            outdir / "representative_tracks.csv",
            pd.read_csv(representative_path),
        )
    genre_incidence = pd.DataFrame()
    genre_dominant = pd.DataFrame()
    genre_top10 = pd.DataFrame()
    genre_external: dict[str, Any] | None = None
    incidence_path = path / "tables" / "genre_cluster_incidence.csv"
    if incidence_path.exists():
        genre_incidence = pd.read_csv(incidence_path)
        atomic_write_csv(outdir / "genre_cluster_incidence.csv", genre_incidence)
        genre_top10 = (
            genre_incidence.sort_values(["cluster", "n_tracks_with_genre"], ascending=[True, False])
            .groupby("cluster", sort=True)
            .head(10)
            .reset_index(drop=True)
        )
        atomic_write_csv(outdir / "genre_top10_per_cluster.csv", genre_top10)
    dominant_path = path / "tables" / "genre_dominant_by_cluster.csv"
    if dominant_path.exists():
        genre_dominant = pd.read_csv(dominant_path)
        atomic_write_csv(outdir / "genre_dominant_by_cluster.csv", genre_dominant)
    external_path = path / "tables" / "genre_external_metrics.json"
    if external_path.exists():
        genre_external = read_json(external_path)
        atomic_write_json(outdir / "genre_external_metrics.json", genre_external)
    final_metrics_path = path / "metrics.json"
    final_metrics_rows = (
        pd.DataFrame(_metric_rows(read_json(final_metrics_path)))
        if final_metrics_path.exists()
        else pd.DataFrame()
    )
    atomic_write_csv(outdir / "final_refit_metrics.csv", final_metrics_rows)
    figures: list[dict[str, Any]] = []
    posthoc_figures = sorted(
        (Path("artifacts/posthoc") / selected_run_id / "figures").glob("*.png")
    )
    posthoc_names = {figure.name for figure in posthoc_figures}
    for figure in sorted(path.glob("figures/*.png")):
        if figure.name in posthoc_names:
            continue
        figures.append(
            {
                "figure": _display_path(figure),
                "scope": "final_refit",
                "run_id": selected_run_id,
                "caption": figure.stem.replace("_", " "),
            }
        )
    for figure in posthoc_figures:
        figures.append(
            {
                "figure": _display_path(figure),
                "scope": "posthoc_visual",
                "run_id": selected_run_id,
                "caption": figure.stem.replace("_", " "),
            }
        )
    if not baseline.empty:
        comparison_dirs = sorted(Path("artifacts/comparisons").glob(f"{baseline_set}/*/"))
        if comparison_dirs:
            for figure in sorted(comparison_dirs[-1].glob("*.png")):
                figures.append(
                    {
                        "figure": _display_path(figure),
                        "scope": "baseline_comparison",
                        "run_id": "multiple",
                        "caption": figure.stem.replace("_", " "),
                    }
                )
    for factor in ("scaler", "feature_set", "pca"):
        latest_dir = _latest(f"artifacts/ablations/{factor}/*/")
        if latest_dir:
            for figure in sorted(Path(latest_dir).glob("*.png")):
                figures.append(
                    {
                        "figure": _display_path(figure),
                        "scope": f"ablation_{factor}",
                        "run_id": "multiple",
                        "caption": figure.stem.replace("_", " "),
                    }
                )
    if stability_run:
        for figure in sorted((Path(config.output_dir) / stability_run / "figures").glob("*.png")):
            figures.append(
                {
                    "figure": _display_path(figure),
                    "scope": "stability",
                    "run_id": stability_run,
                    "caption": figure.stem.replace("_", " "),
                }
            )
    figure_frame = pd.DataFrame(figures)
    atomic_write_csv(outdir / "figures_index.csv", figure_frame)
    index_path = Path("artifacts/INDEX.csv")
    if index_path.exists():
        runs = pd.read_csv(index_path)
        atomic_write_csv(outdir / "all_runs_index.csv", runs)
    profiles_compact_columns = [
        column
        for column in (
            "cluster",
            "n",
            "share",
            "acousticness_median",
            "danceability_median",
            "energy_median",
            "instrumentalness_median",
            "speechiness_median",
            "valence_median",
            "tempo_median",
            "interpretation_hypothesis",
            "top_positive_features",
            "top_negative_features",
        )
        if column in profiles.columns
    ]
    lines = [
        "# BTL11 — Machine-generated report tables",
        "",
        f"- Generated: `{utc_now()}`",
        f"- Selected final refit run: `{selected_run_id}`",
        f"- Evaluation model run: `{doc.get('evaluation_model_id')}`",
        f"- Holdout run: `{holdout_run_id}`",
        f"- Dataset SHA-256: `{manifest.get('sha256')}`",
        f"- License status: `{manifest.get('license_review_status')}` "
        f"(redistribution allowed: `{manifest.get('redistribution_allowed')}`)",
        "",
        "> Every empirical value below comes from the run artifacts in this repository. "
        "Sections without a completed run are marked NOT RUN.",
        "",
        "## 1. Dataset",
        "",
        _md_table(dataset_rows),
        "",
        "## 2. Baseline K-Means (F9/Standard/No-PCA unless shown)",
        "",
        _md_table(
            baseline[baseline["algorithm"] == "kmeans"] if not baseline.empty else baseline,
            [
                "run_id", "k", "feature_space_id", "n_fit", "n_eval", "silhouette",
                "davies_bouldin", "calinski_harabasz", "inertia_fit", "inertia_eval",
                "min_cluster_share_fit", "empty_clusters_fit", "converged",
            ],
        ),
        "",
        "## 3. Baseline GMM (BIC/AIC only comparable within one representation)",
        "",
        _md_table(
            baseline[baseline["algorithm"] == "gmm"] if not baseline.empty else baseline,
            [
                "run_id", "k", "covariance_type", "feature_space_id", "cohort_id", "bic",
                "bic_rank_within_space", "aic", "silhouette", "davies_bouldin",
                "calinski_harabasz", "converged",
            ],
        ),
        "",
        "## 4. Ablations",
        "",
    ]
    for factor in ("scaler", "feature_set", "pca"):
        lines += [
            f"### {factor}",
            "",
            _md_table(
                ablation_tables[factor],
                [
                    "variant", "algorithm", "k", "covariance_type", "feature_space_id",
                    "silhouette", "davies_bouldin", "calinski_harabasz", "bic",
                    "min_cluster_share_fit", "converged", "run_id",
                ],
            ),
            "",
        ]
    lines += [
        "> Cross-representation metrics are contextual; only BIC ranks inside the same "
        "feature-space identifier.",
        "> E00 `none_unscaled` is a scale diagnostic, never the recommended model.",
        "",
        "## 5. Stability (same fixed reference IDs; ARI permutation invariant)",
        "",
        _md_table(
            stability_summary,
            [
                "finalist", "algorithm", "k", "covariance_type", "n_seed_comparisons",
                "seed_ari_mean", "seed_ari_std", "bootstrap_n", "bootstrap_ari_mean",
                "bootstrap_ari_std", "pairwise_ari_mean", "n_reference_ids",
            ],
        ),
        "",
        "## 6. One-time holdout check (frozen evaluation model)",
        "",
        _md_table(holdout_summary, ["metric", "value", "direction", "space", "n_eval", "reason"]),
        "",
        "## 7. Final refit profiles (raw units; acoustic hypotheses, not genres)",
        "",
        _md_table(profiles, profiles_compact_columns),
        "",
        "## 8. Post-hoc multi-label genre evidence (top 10 per cluster)",
        "",
        _md_table(
            genre_top10,
            ["cluster", "genre", "n_tracks_with_genre", "cluster_n", "share_of_cluster_tracks"],
        ),
        "",
        "> Denominator = unique tracks per cluster. Shares can sum >100% because tracks can "
        "carry multiple genres. Genre was never a training/tuning input.",
        "",
        json.dumps(genre_external, indent=2, ensure_ascii=False) if genre_external else "_NOT RUN._",
        "",
        "## 9. Final refit metrics (in-sample descriptive verification sample)",
        "",
        _md_table(final_metrics_rows, ["metric", "value", "direction", "space", "n_eval", "reason"]),
        "",
        "## 10. Figures",
        "",
        _md_table(figure_frame, ["figure", "scope", "run_id", "caption"]),
        "",
    ]
    md_path = outdir / "REPORT_TABLES.md"
    atomic_write_text(md_path, "\n".join(lines))
    metadata = {
        "generated_utc": utc_now(),
        "selected_run_id": selected_run_id,
        "evaluation_model_id": doc.get("evaluation_model_id"),
        "holdout_run_id": holdout_run_id,
        "baseline_experiment_set": baseline_set,
        "stability_run_id": stability_run,
        "dataset_sha256": manifest.get("sha256"),
        "n_baseline_runs": int(len(baseline)),
        "n_figures": int(len(figure_frame)),
        "output_dir": str(outdir),
        "report_tables_md": str(md_path),
    }
    atomic_write_json(outdir / "report_metadata.json", metadata)
    return metadata
