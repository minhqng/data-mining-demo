"""Exploratory data analysis generated from prepared artifacts only."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .config import Config
from .data import PreparedData
from .features import F9_FEATURES
from .utils import (
    atomic_write_csv,
    atomic_write_json,
    sha256_file,
    utc_now,
)
from .visualization import plot_correlation, plot_eda_distributions


def run_eda(config: Config, prepared: PreparedData) -> dict[str, Any]:
    stamp = utc_now().replace(":", "").replace("-", "")
    outdir = Path("artifacts/eda") / stamp
    (outdir / "tables").mkdir(parents=True, exist_ok=False)
    (outdir / "figures").mkdir(parents=True, exist_ok=True)
    tracks = prepared.tracks
    feature_names = list(F9_FEATURES)
    rows: list[dict[str, Any]] = []
    for feature in feature_names:
        values = pd.to_numeric(tracks[feature], errors="coerce").dropna().astype(float)
        q01, q25, q75, q99 = (
            float(values.quantile(0.01)),
            float(values.quantile(0.25)),
            float(values.quantile(0.75)),
            float(values.quantile(0.99)),
        )
        iqr = q75 - q25
        lower, upper = q25 - 1.5 * iqr, q75 + 1.5 * iqr
        outliers = int(((values < lower) | (values > upper)).sum()) if iqr > 0 else 0
        rows.append(
            {
                "feature": feature,
                "n": int(values.size),
                "missing": int(tracks[feature].isna().sum()),
                "mean": float(values.mean()),
                "std": float(values.std(ddof=0)),
                "min": float(values.min()),
                "p01": q01,
                "q25": q25,
                "median": float(values.median()),
                "q75": q75,
                "p99": q99,
                "max": float(values.max()),
                "skew": float(values.skew()),
                "iqr": iqr,
                "iqr_outliers": outliers,
                "iqr_outlier_share": float(outliers / max(len(values), 1)),
                "unit_interval_expected": feature
                in {
                    "acousticness",
                    "danceability",
                    "energy",
                    "valence",
                    "speechiness",
                    "instrumentalness",
                    "liveness",
                },
            }
        )
    summary = pd.DataFrame(rows)
    atomic_write_csv(outdir / "tables" / "feature_summary.csv", summary)
    correlation = tracks.loc[:, feature_names].corr(method="pearson")
    atomic_write_csv(
        outdir / "tables" / "feature_correlation.csv",
        correlation.reset_index(names="feature"),
    )
    missingness = pd.DataFrame(
        [
            {
                "column": column,
                "n_missing": int(tracks[column].isna().sum()),
                "share": float(tracks[column].isna().mean()),
            }
            for column in tracks.columns
        ]
    )
    atomic_write_csv(outdir / "tables" / "missingness.csv", missingness)
    genre_counts = (
        prepared.track_genres.groupby("genre")["track_id"]
        .nunique()
        .sort_values(ascending=False)
        .reset_index(name="n_tracks")
    )
    atomic_write_csv(outdir / "tables" / "genre_track_counts.csv", genre_counts)
    top_genres = genre_counts.head(30)
    atomic_write_csv(outdir / "tables" / "genre_top30.csv", top_genres)
    # Descriptive genre figure; genre is post-hoc metadata, never a training input.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(top_genres["genre"], top_genres["n_tracks"], color="#4C72B0")
    ax.set_title(
        "Top 30 genres by unique tracks (descriptive only; multi-label incidence, "
        "not a model target)"
    )
    ax.set_xlabel("Genre")
    ax.set_ylabel("Unique tracks containing genre")
    ax.tick_params(axis="x", rotation=75)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / "figures" / "genre_top30.png", dpi=160)
    plt.close(fig)
    distributions = plot_eda_distributions(
        {"tracks": tracks},
        feature_names,
        outdir / "figures" / "feature_distributions.png",
        title=(
            f"Audio feature distributions — {prepared.n_tracks:,} unique tracks | "
            f"dataset {prepared.dataset_sha256[:12]}"
        ),
    )
    corr_figure = plot_correlation(
        correlation,
        outdir / "figures" / "feature_correlation.png",
        title=(
            "Pearson correlation of raw audio features | "
            f"{prepared.n_tracks:,} unique tracks | dataset {prepared.dataset_sha256[:12]}"
        ),
    )
    # Save the exact data behind the correlation figure is already a CSV.
    metadata = {
        "generated_utc": utc_now(),
        "dataset_sha256": prepared.dataset_sha256,
        "processed_sha256": prepared.processed_sha256,
        "n_tracks": prepared.n_tracks,
        "n_genres": int(prepared.track_genres["genre"].nunique()),
        "n_multi_genre_tracks": int(
            prepared.track_genres.groupby("track_id")["genre"].nunique().gt(1).sum()
        ),
        "feature_names": feature_names,
        "figures": {
            "feature_distributions": str(distributions),
            "feature_correlation": str(corr_figure),
            "genre_top30": str(outdir / "figures" / "genre_top30.png"),
        },
        "tables": {
            "feature_summary": str(outdir / "tables" / "feature_summary.csv"),
            "feature_correlation": str(outdir / "tables" / "feature_correlation.csv"),
            "genre_track_counts": str(outdir / "tables" / "genre_track_counts.csv"),
        },
        "audit": {
            "raw_rows": prepared.summary.get("raw_rows"),
            "unique_tracks": prepared.summary.get("unique_track_ids"),
            "dedup_dropped_rows": prepared.summary.get("dropped_rows"),
            "n_track_genres": prepared.summary.get("n_track_genres"),
            "splits": prepared.summary.get("splits"),
        },
        "note": "EDA is descriptive. Outliers are reported, never automatically removed.",
    }
    atomic_write_json(outdir / "eda.json", metadata)
    atomic_write_json(
        outdir / "checksums.json",
        {
            name: sha256_file(path)
            for name, path in {
                "feature_summary": outdir / "tables" / "feature_summary.csv",
                "correlation": outdir / "tables" / "feature_correlation.csv",
                "distributions": distributions,
                "corr_figure": corr_figure,
            }.items()
        },
    )
    return {"output_dir": str(outdir), **metadata}
