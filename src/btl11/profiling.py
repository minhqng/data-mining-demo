"""Raw-unit cluster profiling, representative local tracks and post-hoc genre checks."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from .config import Config
from .data import PreparedData
from .errors import DataError, SchemaError
from .features import F9_FEATURES
from .models import ClusteringBundle
from .utils import atomic_write_csv, atomic_write_json, sha256_obj
from .visualization import plot_genre_heatmap, plot_gmm_uncertainty, plot_profile_heatmap


def _validate_assignments(assignments: pd.DataFrame) -> pd.DataFrame:
    required = {"track_id", "cluster"}
    if not required <= set(assignments.columns):
        raise SchemaError(f"Assignments must contain {sorted(required)}.")
    frame = assignments.copy()
    frame["track_id"] = frame["track_id"].astype(str)
    if frame["track_id"].duplicated().any():
        raise SchemaError("Assignments must have one row per track_id.")
    frame["cluster"] = frame["cluster"].astype(int)
    return frame


def _hypothesis(row: pd.Series) -> str:
    energy = float(row.get("energy_median", np.nan))
    dance = float(row.get("danceability_median", np.nan))
    acoustic = float(row.get("acousticness_median", np.nan))
    instrumental = float(row.get("instrumentalness_median", np.nan))
    speech = float(row.get("speechiness_median", np.nan))
    valence = float(row.get("valence_median", np.nan))
    tempo = float(row.get("tempo_median", np.nan))
    if instrumental >= 0.5:
        return "Instrumental/ambient (giả thuyết âm học, không phải genre)"
    if speech >= 0.33:
        return "Speech-heavy / rap-like (giả thuyết âm học, không phải genre)"
    if acoustic >= 0.6 and energy <= 0.4:
        return "Acoustic / năng lượng thấp (giả thuyết âm học, không phải genre)"
    if instrumental <= 0.05 and energy >= 0.6:
        if dance >= 0.65 and valence >= 0.55:
            return "Vocal/dance, năng lượng cao (giả thuyết âm học, không phải genre)"
        return (
            "Vocal/non-instrumental, năng lượng trung bình–cao "
            "(giả thuyết âm học, không phải genre)"
        )
    if energy <= 0.35 and valence <= 0.30:
        return "U tối / nhẹ, ít năng lượng (giả thuyết âm học, không phải genre)"
    if tempo >= 140 and energy >= 0.55:
        return "Tempo cao, năng lượng (giả thuyết âm học, không phải genre)"
    return "Hỗn hợp / chưa đủ đặc trưng để đặt tên (giả thuyết âm học, không phải genre)"


def profile_clusters(
    raw_features: pd.DataFrame,
    assignments: pd.DataFrame,
    global_reference: pd.DataFrame,
    feature_names: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Build the wide raw-unit profile table with one row per cluster.

    The join is strictly by ``track_id``; counts must reconcile exactly with
    the assignments supplied.
    """
    assignments = _validate_assignments(assignments)
    feature_names = list(feature_names or [f for f in F9_FEATURES if f in raw_features.columns])
    if not feature_names:
        raise SchemaError("No profiling feature columns were supplied.")
    for column in ("track_id", *feature_names):
        if column not in raw_features.columns:
            raise SchemaError(f"Profiling frame is missing {column!r}.")
        if column not in global_reference.columns:
            raise SchemaError(f"Global reference frame is missing {column!r}.")
    tracks = raw_features.copy()
    tracks["track_id"] = tracks["track_id"].astype(str)
    if tracks["track_id"].duplicated().any():
        raise SchemaError("Raw profiling frame contains duplicate track IDs.")
    merged = tracks.merge(
        assignments.loc[:, ["track_id", "cluster"]],
        on="track_id",
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(assignments):
        missing = sorted(set(assignments["track_id"]) - set(tracks["track_id"]))[:5]
        raise DataError(
            f"Profile join lost rows: assignments={len(assignments)}, joined={len(merged)}. "
            f"Examples of unknown IDs: {missing}"
        )
    for feature in feature_names:
        merged[feature] = pd.to_numeric(merged[feature], errors="raise").astype(float)
        if not np.isfinite(merged[feature]).all():
            raise DataError(f"Profiling feature {feature!r} contains NaN/inf.")
    global_stats: dict[str, dict[str, float]] = {}
    for feature in feature_names:
        values = pd.to_numeric(global_reference[feature], errors="coerce").dropna().astype(float)
        q25, q75 = float(values.quantile(0.25)), float(values.quantile(0.75))
        std = float(values.std(ddof=0))
        global_stats[feature] = {
            "mean": float(values.mean()),
            "std": std if std > 0 else 0.0,
            "median": float(values.median()),
            "iqr": max(q75 - q25, 0.0),
        }
    total = int(len(merged))
    rows: list[dict[str, Any]] = []
    for cluster, group in merged.groupby("cluster", sort=True):
        row: dict[str, Any] = {
            "cluster": int(cluster),
            "n": int(len(group)),
            "share": float(len(group) / total),
        }
        z_values: dict[str, float] = {}
        for feature in feature_names:
            values = group[feature].astype(float)
            mean = float(values.mean())
            median = float(values.median())
            q25, q75 = float(values.quantile(0.25)), float(values.quantile(0.75))
            stats = global_stats[feature]
            z = (mean - stats["mean"]) / stats["std"] if stats["std"] > 0 else 0.0
            z_values[feature] = z
            row[f"{feature}_mean"] = mean
            row[f"{feature}_median"] = median
            row[f"{feature}_std"] = float(values.std(ddof=0))
            row[f"{feature}_q25"] = q25
            row[f"{feature}_q75"] = q75
            row[f"{feature}_iqr"] = max(q75 - q25, 0.0)
            row[f"{feature}_z"] = z
        ordered = sorted(z_values.items(), key=lambda item: item[1], reverse=True)
        row["top_positive_features"] = ", ".join(
            f"{name} ({value:+.2f})" for name, value in ordered[:3]
        )
        row["top_negative_features"] = ", ".join(
            f"{name} ({value:+.2f})" for name, value in ordered[-3:][::-1]
        )
        row["interpretation_hypothesis"] = _hypothesis(pd.Series(row))
        row["interpretation_note"] = (
            "Hypothesis from acoustic statistics only; not a genre and not verified by listening."
        )
        rows.append(row)
    wide = pd.DataFrame(rows).sort_values("cluster").reset_index(drop=True)
    if int(wide["n"].sum()) != total:
        raise DataError("Cluster profile counts do not reconcile with assignments.")
    return wide


def _profiles_long(
    wide: pd.DataFrame, feature_names: Sequence[str], global_reference: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for feature in feature_names:
        values = pd.to_numeric(global_reference[feature], errors="coerce").dropna().astype(float)
        global_mean = float(values.mean())
        global_median = float(values.median())
        global_std = float(values.std(ddof=0))
        q25, q75 = float(values.quantile(0.25)), float(values.quantile(0.75))
        global_iqr = q75 - q25
        for _, row in wide.iterrows():
            median = float(row[f"{feature}_median"])
            rows.append(
                {
                    "cluster": int(row["cluster"]),
                    "n": int(row["n"]),
                    "share": float(row["share"]),
                    "feature": feature,
                    "mean": float(row[f"{feature}_mean"]),
                    "median": median,
                    "std": float(row[f"{feature}_std"]),
                    "q25": float(row[f"{feature}_q25"]),
                    "q75": float(row[f"{feature}_q75"]),
                    "iqr": float(row[f"{feature}_iqr"]),
                    "global_mean": global_mean,
                    "global_median": global_median,
                    "global_std": global_std,
                    "z_vs_global_mean": float(row[f"{feature}_z"]),
                    "robust_deviation_vs_global_median": (
                        (median - global_median) / global_iqr if global_iqr > 0 else 0.0
                    ),
                }
            )
    return pd.DataFrame(rows)


def _representative_tracks(
    prepared: PreparedData,
    assignments: pd.DataFrame,
    bundle: ClusteringBundle | None,
    *,
    feature_names: Sequence[str],
) -> pd.DataFrame:
    assignments = _validate_assignments(assignments)
    ids = assignments["track_id"].tolist()
    metadata_columns = [
        column
        for column in ("track_name", "artists", "album_name", "popularity", "duration_ms")
        if column in prepared.tracks.columns
    ]
    feature_columns = list(feature_names)
    metadata = prepared.subset_tracks(ids).loc[
        :, ["track_id", *metadata_columns, *feature_columns]
    ].copy()
    metadata["track_id"] = metadata["track_id"].astype(str)
    cluster_by_id = assignments.set_index("track_id")["cluster"]
    metadata["cluster"] = metadata["track_id"].map(cluster_by_id).astype(int)
    if bundle is None:
        return pd.DataFrame(
            columns=[
                "track_id",
                "cluster",
                "selection_kind",
                "rank",
                *metadata_columns,
                *feature_columns,
            ]
        )
    X = prepared.feature_matrix(ids, feature_names)
    Xt = bundle.transform(X)
    labels = assignments.set_index("track_id").reindex(ids)["cluster"].to_numpy(dtype=np.int64)
    confidence = (
        assignments.set_index("track_id").reindex(ids)["confidence"].to_numpy(dtype=float)
        if "confidence" in assignments.columns
        else None
    )
    rows: list[dict[str, Any]] = []
    for cluster in sorted(np.unique(labels)):
        positions = np.flatnonzero(labels == cluster)
        points = Xt[positions]
        centroid = points.mean(axis=0)
        distances = np.linalg.norm(points - centroid, axis=1)
        if bundle.is_gmm and confidence is not None:
            cluster_confidence = confidence[positions]
            order_high = positions[np.argsort(-cluster_confidence)[:3]]
            order_low = positions[np.argsort(cluster_confidence)[:2]]
            for rank, position in enumerate(order_high, start=1):
                rows.append(
                    {
                        "track_id": ids[position],
                        "cluster": int(cluster),
                        "selection_kind": "high_posterior",
                        "rank": rank,
                        "score_name": "gmm_confidence",
                        "score": float(confidence[position]),
                        "distance_to_centroid": float(distances[np.where(positions == position)[0][0]]),
                    }
                )
            for rank, position in enumerate(order_low, start=1):
                rows.append(
                    {
                        "track_id": ids[position],
                        "cluster": int(cluster),
                        "selection_kind": "borderline",
                        "rank": rank,
                        "score_name": "gmm_confidence",
                        "score": float(confidence[position]),
                        "distance_to_centroid": float(distances[np.where(positions == position)[0][0]]),
                    }
                )
        else:
            order_close = positions[np.argsort(distances)[:3]]
            order_far = positions[np.argsort(-distances)[:2]]
            for rank, position in enumerate(order_close, start=1):
                rows.append(
                    {
                        "track_id": ids[position],
                        "cluster": int(cluster),
                        "selection_kind": "near_centroid",
                        "rank": rank,
                        "score_name": "euclidean_distance_to_centroid",
                        "score": float(distances[np.where(positions == position)[0][0]]),
                        "distance_to_centroid": float(distances[np.where(positions == position)[0][0]]),
                    }
                )
            for rank, position in enumerate(order_far, start=1):
                rows.append(
                    {
                        "track_id": ids[position],
                        "cluster": int(cluster),
                        "selection_kind": "boundary_far",
                        "rank": rank,
                        "score_name": "euclidean_distance_to_centroid",
                        "score": float(distances[np.where(positions == position)[0][0]]),
                        "distance_to_centroid": float(distances[np.where(positions == position)[0][0]]),
                    }
                )
    examples = pd.DataFrame(rows).merge(metadata, on=["track_id", "cluster"], how="left")
    examples["note"] = (
        "Selected from local metadata/geometry only; no audio was listened to by the pipeline."
    )
    return examples.sort_values(["cluster", "selection_kind", "rank"]).reset_index(drop=True)


def generate_profiles(
    config: Config,
    prepared: PreparedData,
    run_path: str | Path,
    run_id: str,
    assignments: pd.DataFrame,
    *,
    bundle: ClusteringBundle | None = None,
    all_tracks_scope: bool = False,
    feature_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    run_path = Path(run_path)
    (run_path / "tables").mkdir(parents=True, exist_ok=True)
    (run_path / "figures").mkdir(parents=True, exist_ok=True)
    assignments = _validate_assignments(assignments)
    names = list(feature_names or (bundle.feature_names if bundle else config.feature_primary))
    assigned_ids = assignments["track_id"].tolist()
    raw = prepared.subset_tracks(assigned_ids)
    # Candidate/holdout profiles must not use holdout features as a reference
    # scale before the model is frozen; only the final full refit sees all tracks.
    reference = (
        prepared.tracks
        if all_tracks_scope
        else prepared.subset_tracks(prepared.splits["dev"])
    )
    wide = profile_clusters(raw, assignments, reference, names)
    long = _profiles_long(wide, names, reference)
    profiles_path = atomic_write_csv(run_path / "profiles.csv", wide)
    long_path = atomic_write_csv(run_path / "tables" / "profiles_long.csv", long)
    atomic_write_json(
        run_path / "tables" / "profile_metadata.json",
        {
            "run_id": run_id,
            "n_assigned": len(assignments),
            "n_clusters": int(wide["cluster"].nunique()),
            "scope": "all_unique_tracks" if all_tracks_scope else "assignments_scope",
            "feature_names": names,
            "global_reference": (
                "all unique tracks in tracks.parquet"
                if all_tracks_scope
                else "development split (fit + eval) in tracks.parquet"
            ),
            "units": "raw audio-feature units (not standardized)",
            "deviation_definition": "(cluster mean - global mean) / global SD",
            "counts_reconcile": int(wide["n"].sum()) == len(assignments),
            "hypothesis_caveat": "Cluster names are acoustic hypotheses, never verified genres.",
            "assignments_sha256": sha256_obj(assignments.sort_values("track_id").to_dict("records")),
        },
    )
    profile_figure = plot_profile_heatmap(
        wide, names, run_path / "figures" / "profile_heatmap.png"
    )
    examples = _representative_tracks(
        prepared, assignments, bundle, feature_names=names
    )
    examples_path = atomic_write_csv(
        run_path / "tables" / "representative_tracks.csv", examples
    )
    uncertainty_info = None
    if {"confidence", "entropy", "normalized_entropy"} <= set(assignments.columns):
        uncertainty = assignments.loc[:, ["track_id", "cluster", "confidence", "entropy", "normalized_entropy"]].copy()
        uncertainty_path = atomic_write_csv(
            run_path / "tables" / "gmm_uncertainty_per_track.csv", uncertainty
        )
        k = int(assignments["cluster"].nunique())
        uncertainty_figure = plot_gmm_uncertainty(
            uncertainty.dropna(),
            run_path / "figures" / "gmm_uncertainty.png",
            k=k,
            run_id=run_id,
        )
        uncertainty_info = {
            "table": str(uncertainty_path),
            "figure": str(uncertainty_figure),
            "mean_confidence": float(uncertainty["confidence"].mean()),
            "p10_confidence": float(uncertainty["confidence"].quantile(0.10)),
            "mean_normalized_entropy": float(uncertainty["normalized_entropy"].mean()),
            "borderline_definition": "lowest 10% confidence in this assignment scope",
        }
    return {
        "files": {
            "profiles": str(profiles_path),
            "profiles_long": str(long_path),
            "profile_figure": str(profile_figure),
            "representative_tracks": str(examples_path),
        },
        "n_clusters": int(wide["cluster"].nunique()),
        "n_assigned": int(len(assignments)),
        "cluster_sizes": wide.loc[:, ["cluster", "n", "share"]].to_dict(orient="records"),
        "interpretations": wide.loc[
            :, ["cluster", "interpretation_hypothesis", "top_positive_features", "top_negative_features"]
        ].to_dict(orient="records"),
        "uncertainty": uncertainty_info,
        "representative_examples": int(len(examples)),
    }


def generate_genre_posthoc(
    config: Config,
    prepared: PreparedData,
    run_path: str | Path,
    run_id: str,
    assignments: pd.DataFrame,
    *,
    all_tracks_scope: bool = False,
    top_n: int = 20,
) -> dict[str, Any]:
    """Multi-label incidence heatmap and a clearly limited single-label check."""
    run_path = Path(run_path)
    assignments = _validate_assignments(assignments)
    bridge = prepared.track_genres.copy()
    bridge["track_id"] = bridge["track_id"].astype(str)
    assigned_ids = set(assignments["track_id"])
    bridge = bridge[bridge["track_id"].isin(assigned_ids)]
    joined = assignments.loc[:, ["track_id", "cluster"]].merge(
        bridge, on="track_id", how="left", validate="one_to_many"
    )
    cluster_sizes = assignments.groupby("cluster")["track_id"].nunique().sort_index()
    incidence_rows: list[dict[str, Any]] = []
    genre_counts = joined.dropna(subset=["genre"]).groupby(["cluster", "genre"]).size()
    for (cluster, genre), count in genre_counts.items():
        cluster_n = int(cluster_sizes.loc[cluster])
        incidence_rows.append(
            {
                "cluster": int(cluster),
                "genre": str(genre),
                "n_tracks_with_genre": int(count),
                "cluster_n": cluster_n,
                "share_of_cluster_tracks": float(count / cluster_n) if cluster_n else None,
            }
        )
    incidence = pd.DataFrame(incidence_rows).sort_values(
        ["cluster", "n_tracks_with_genre", "genre"], ascending=[True, False, True]
    )
    incidence_path = atomic_write_csv(
        run_path / "tables" / "genre_cluster_incidence.csv", incidence
    )
    dominant_rows: list[dict[str, Any]] = []
    for cluster, group in incidence.groupby("cluster", sort=True):
        for rank, (_, row) in enumerate(group.head(3).iterrows(), start=1):
            dominant_rows.append(
                {
                    "cluster": int(cluster),
                    "rank": rank,
                    "genre": row["genre"],
                    "n_tracks_with_genre": int(row["n_tracks_with_genre"]),
                    "cluster_n": int(row["cluster_n"]),
                    "share_of_cluster_tracks": float(row["share_of_cluster_tracks"]),
                }
            )
    dominant = pd.DataFrame(dominant_rows)
    dominant_path = atomic_write_csv(
        run_path / "tables" / "genre_dominant_by_cluster.csv", dominant
    )
    # Top genres by incidence across the assigned scope (multi-label, so shares can sum >100%).
    totals = (
        joined.dropna(subset=["genre"])
        .groupby("genre")["track_id"]
        .nunique()
        .sort_values(ascending=False)
    )
    top_genres = totals.head(top_n).index.tolist()
    matrix = (
        incidence[incidence["genre"].isin(top_genres)]
        .pivot(index="cluster", columns="genre", values="share_of_cluster_tracks")
        .reindex(columns=top_genres)
        .fillna(0.0)
        * 100.0
    )
    matrix = matrix.sort_index()
    heatmap_path = plot_genre_heatmap(
        matrix,
        run_path / "figures" / "genre_cluster_heatmap.png",
        title=(
            f"Post-hoc multi-label genre incidence by cluster (top {len(top_genres)} genres) "
            f"| run {run_id}"
        ),
        note=(
            "Denominator = unique tracks in each cluster. Rows can sum >100% because a track "
            "may carry multiple genres. Genre never entered training or model selection."
        ),
    )
    # Single-genre-only external check: explicitly declared cohort, no cherry-picking.
    genre_counts_per_track = bridge.groupby("track_id")["genre"].nunique()
    single_ids = set(genre_counts_per_track[genre_counts_per_track == 1].index)
    single_bridge = bridge[bridge["track_id"].isin(single_ids)].drop_duplicates("track_id")
    single = assignments.loc[:, ["track_id", "cluster"]].merge(
        single_bridge, on="track_id", how="inner"
    )
    n_single = int(len(single))
    n_omitted = int(len(assignments) - n_single)
    ari_value = None
    nmi_value = None
    reason = None
    if n_single < 10 or single["cluster"].nunique() < 2 or single["genre"].nunique() < 2:
        reason = "insufficient_single_genre_or_cluster_diversity"
    else:
        try:
            ari_value = float(adjusted_rand_score(single["genre"], single["cluster"]))
            nmi_value = float(normalized_mutual_info_score(single["genre"], single["cluster"]))
        except ValueError as exc:
            reason = f"sklearn_error:{exc}"
    external = {
        "schema": "btl11.genre_posthoc.v1",
        "run_id": run_id,
        "scope": "all_unique_tracks" if all_tracks_scope else "assignments_scope",
        "n_assigned": int(len(assignments)),
        "n_single_genre_eval": n_single,
        "n_omitted_not_single_label": n_omitted,
        "cluster_n": int(assignments["cluster"].nunique()),
        "genre_n_single_cohort": int(single["genre"].nunique()) if n_single else 0,
        "adjusted_rand_vs_genre": {"value": ari_value, "reason": reason},
        "normalized_mutual_info_vs_genre": {"value": nmi_value, "reason": reason},
        "cohort_rule": (
            "Single-genre-only tracks (exactly one unique genre in track_genres). "
            "The cohort is declared before computing ARI/NMI; no labels were used for tuning."
        ),
        "caveat": (
            "Genre taxonomy does not define acoustic personas. Low/high agreement is diagnostic, "
            "not a target and not proof of cluster quality."
        ),
    }
    external_path = atomic_write_json(
        run_path / "tables" / "genre_external_metrics.json", external
    )
    return {
        "files": {
            "incidence": str(incidence_path),
            "dominant": str(dominant_path),
            "heatmap": str(heatmap_path),
            "external_metrics": str(external_path),
        },
        "n_genres": int(joined["genre"].nunique()),
        "top_genres": top_genres,
        "single_genre_metrics": external,
        "cluster_sizes": {str(k): int(v) for k, v in cluster_sizes.items()},
    }
