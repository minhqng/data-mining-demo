"""Deterministic, accessible figures. Plotting never participates in model selection."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PALETTE = "tab20"
FIG_DPI = 160


def _finish(fig: plt.Figure, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(target, dpi=FIG_DPI)
    plt.close(fig)
    return target


def plot_cluster_sizes(sizes: pd.DataFrame, path: str | Path, *, title: str) -> Path:
    fig, ax = plt.subplots(figsize=(9, 4.8))
    clusters = sizes["cluster"].astype(int).astype(str)
    colors = plt.get_cmap(PALETTE)(np.arange(len(sizes)) % 20)
    ax.bar(clusters, sizes["count"], color=colors)
    for index, (count, share) in enumerate(zip(sizes["count"], sizes["share"], strict=True)):
        ax.text(index, count, f"{count}\n({share:.1%})", ha="center", va="bottom", fontsize=8)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Number of tracks")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    return _finish(fig, path)


def plot_sweep_elbow(frame: pd.DataFrame, path: str | Path) -> Path:
    data = frame.sort_values("k")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(data["k"], data["inertia_fit"], marker="o", linewidth=2, label="Fit-cohort inertia (SSE)")
    if "inertia_eval" in data and data["inertia_eval"].notna().any():
        ax.plot(data["k"], data["inertia_eval"], marker="s", linestyle="--", label="Eval-cohort SSE")
    ax.set_title(
        "K-Means elbow — exploratory only; K is not chosen from the elbow alone\n"
        f"{data['feature_space_id'].iloc[0] if len(data) else ''}",
        fontsize=10,
    )
    ax.set_xlabel("K (number of clusters)")
    ax.set_ylabel("Inertia / SSE (squared feature units)")
    ax.set_xticks(sorted(data["k"].dropna().astype(int).unique()))
    ax.legend()
    ax.grid(alpha=0.25)
    return _finish(fig, path)


def plot_sweep_metrics(frame: pd.DataFrame, path: str | Path) -> Path:
    data = frame.sort_values("k")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    specs = [
        ("silhouette", "Silhouette ↑"),
        ("davies_bouldin", "Davies–Bouldin ↓"),
        ("calinski_harabasz", "Calinski–Harabasz ↑"),
    ]
    for ax, (column, label) in zip(axes, specs, strict=True):
        ax.plot(data["k"], pd.to_numeric(data[column], errors="coerce"), marker="o")
        ax.set_title(label)
        ax.set_xlabel("K")
        ax.set_ylabel(label)
        ax.grid(alpha=0.25)
        ax.set_xticks(sorted(data["k"].dropna().astype(int).unique()))
    fig.suptitle(
        "Fixed-cohort validity metrics (same IDs across candidates). "
        "No single metric selects the model.",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return _finish(fig, path)


def plot_gmm_information(frame: pd.DataFrame, path: str | Path) -> Path:
    data = frame.sort_values("k")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for covariance, group in data.groupby("covariance_type", dropna=False):
        label = str(covariance)
        axes[0].plot(group["k"], group["bic"], marker="o", label=label)
        axes[1].plot(group["k"], group["aic"], marker="o", label=label)
    spaces = sorted(data["feature_space_id"].dropna().unique())
    axes[0].set_title("GMM BIC ↓ (within one representation only)")
    axes[1].set_title("GMM AIC ↓ (within one representation only)")
    for ax in axes:
        ax.set_xlabel("K components")
        ax.set_ylabel("Information criterion")
        ax.grid(alpha=0.25)
        ax.legend(title="covariance")
    fig.suptitle(
        "GMM information criteria — do not compare raw BIC across feature/scaler/PCA spaces\n"
        + " | ".join(spaces),
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return _finish(fig, path)


def plot_ablation(frame: pd.DataFrame, factor: str, path: str | Path) -> Path:
    if frame.empty:
        raise ValueError("Cannot plot an empty ablation table.")
    data = frame.copy()
    data["variant"] = data["variant"].astype(str)
    algorithms = list(data["algorithm"].dropna().unique())
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    metrics = [
        ("silhouette", "Silhouette ↑"),
        ("davies_bouldin", "Davies–Bouldin ↓"),
        ("calinski_harabasz", "Calinski–Harabasz ↑"),
    ]
    variants = list(dict.fromkeys(data["variant"]))
    width = 0.8 / max(len(algorithms), 1)
    for ax, (column, label) in zip(axes, metrics, strict=True):
        for offset, algorithm in enumerate(algorithms):
            subset = data[data["algorithm"] == algorithm]
            positions = np.arange(len(variants)) + offset * width
            values = []
            for variant in variants:
                rows = subset[subset["variant"] == variant]
                values.append(float(rows[column].mean()) if not rows.empty and rows[column].notna().any() else np.nan)
            ax.bar(positions, values, width=width, label=algorithm)
        ax.set_xticks(np.arange(len(variants)) + width * (len(algorithms) - 1) / 2)
        ax.set_xticklabels(variants, rotation=25, ha="right", fontsize=8)
        ax.set_title(label)
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.suptitle(
        f"Ablation: {factor} — metrics are contextual to each representation; "
        "not a cross-space ranking",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return _finish(fig, path)


def plot_stability(summary: pd.DataFrame, path: str | Path) -> Path:
    if summary.empty:
        raise ValueError("Cannot plot empty stability results.")
    labels = summary["finalist"].astype(str).tolist()
    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10, 4.8))
    seed_mean = summary["seed_ari_mean"].astype(float).fillna(0).to_numpy()
    seed_std = summary["seed_ari_std"].astype(float).fillna(0).to_numpy()
    boot_mean = summary["bootstrap_ari_mean"].astype(float).fillna(0).to_numpy()
    boot_std = summary["bootstrap_ari_std"].astype(float).fillna(0).to_numpy()
    ax.bar(x - width / 2, seed_mean, width, yerr=seed_std, capsize=3, label="Seed refits vs base")
    ax.bar(x + width / 2, boot_mean, width, yerr=boot_std, capsize=3, label="Bootstrap refits vs base")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=8)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Adjusted Rand Index vs baseline (same eval IDs)")
    ax.set_title(
        "Assignment stability — ARI is label-permutation invariant; "
        "all predictions use the same fixed reference IDs",
        fontsize=9,
    )
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    return _finish(fig, path)


def plot_pca_diagnostics(diagnostics: dict[str, Any], path: str | Path) -> Path:
    ratios = diagnostics.get("explained_variance_ratio")
    if not ratios:
        raise ValueError("No PCA explained variance to plot.")
    cumulative = diagnostics["cumulative_explained_variance"]
    components = [f"PC{i + 1}" for i in range(len(ratios))]
    fig, ax1 = plt.subplots(figsize=(8, 4.8))
    ax1.bar(components, ratios, color="#4C72B0", alpha=0.85, label="Explained variance ratio")
    ax1.set_xlabel("Principal component")
    ax1.set_ylabel("Explained variance ratio")
    ax1.tick_params(axis="x", rotation=45)
    ax2 = ax1.twinx()
    ax2.plot(components, cumulative, color="#C44E52", marker="o", label="Cumulative")
    ax2.set_ylabel("Cumulative explained variance")
    ax2.set_ylim(0, 1.02)
    ax1.set_title(
        f"PCA diagnostics — {diagnostics.get('pca_actual_components')} components retained "
        "(fit on development fit cohort only)",
        fontsize=10,
    )
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right", fontsize=8)
    ax1.grid(axis="y", alpha=0.25)
    return _finish(fig, path)


def plot_pca_scatter(
    coordinates: pd.DataFrame,
    path: str | Path,
    *,
    title: str,
    explained: Sequence[float] | None = None,
) -> Path:
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    clusters = sorted(coordinates["cluster"].unique())
    cmap = plt.get_cmap(PALETTE)
    for index, cluster in enumerate(clusters):
        subset = coordinates[coordinates["cluster"] == cluster]
        ax.scatter(
            subset["PC1"], subset["PC2"], s=9, alpha=0.65,
            color=cmap(index % 20), label=f"C{cluster} (n={len(subset)})",
        )
    axis_text = (
        f"PC1 ({explained[0]:.1%})"
        if explained is not None and len(explained) > 0
        else "PC1"
    )
    y_text = (
        f"PC2 ({explained[1]:.1%})"
        if explained is not None and len(explained) > 1
        else "PC2"
    )
    ax.set_xlabel(f"{axis_text} — standardized audio-feature units")
    ax.set_ylabel(f"{y_text} — standardized audio-feature units")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7, markerscale=2, ncol=2)
    ax.grid(alpha=0.2)
    return _finish(fig, path)


def plot_tsne(coordinates: pd.DataFrame, path: str | Path, *, title: str, perplexity: float) -> Path:
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    clusters = sorted(coordinates["cluster"].unique())
    cmap = plt.get_cmap(PALETTE)
    for index, cluster in enumerate(clusters):
        subset = coordinates[coordinates["cluster"] == cluster]
        ax.scatter(
            subset["tsne_1"], subset["tsne_2"], s=8, alpha=0.65,
            color=cmap(index % 20), label=f"C{cluster} (n={len(subset)})",
        )
    ax.set_xlabel("t-SNE dimension 1 (arbitrary units)")
    ax.set_ylabel("t-SNE dimension 2 (arbitrary units)")
    ax.set_title(
        title + f"\nperplexity={perplexity:g}; VISUAL ONLY — do not infer distances, "
        "density or K from this embedding.",
        fontsize=9,
    )
    ax.legend(fontsize=7, markerscale=2, ncol=2)
    ax.grid(alpha=0.15)
    return _finish(fig, path)


def plot_profile_heatmap(profiles: pd.DataFrame, feature_names: Sequence[str], path: str | Path) -> Path:
    z_columns = [f"{feature}_z" for feature in feature_names]
    if not set(z_columns) <= set(profiles.columns):
        raise ValueError("Profiles are missing standardized-deviation columns.")
    matrix = profiles.loc[:, z_columns].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(11, max(4.5, 0.5 * len(profiles) + 2)))
    limit = np.nanmax(np.abs(matrix)) if np.isfinite(matrix).any() else 1.0
    image = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit)
    ax.set_xticks(range(len(feature_names)))
    ax.set_xticklabels(feature_names, rotation=35, ha="right")
    ax.set_yticks(range(len(profiles)))
    ax.set_yticklabels(
        [f"C{int(c)} (n={int(n)})" for c, n in zip(profiles["cluster"], profiles["n"], strict=True)]
    )
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            if np.isfinite(value):
                ax.text(column, row, f"{value:+.2f}", ha="center", va="center", fontsize=7,
                        color="white" if abs(value) > 0.6 * limit else "black")
    ax.set_title(
        "Cluster profile — deviation of raw-unit cluster mean vs global mean, in global SD units\n"
        "These are statistical descriptions/hypotheses, not genre labels.",
        fontsize=9,
    )
    fig.colorbar(image, ax=ax, label="Standardized deviation (SD)")
    return _finish(fig, path)


def plot_gmm_uncertainty(
    uncertainty: pd.DataFrame, path: str | Path, *, k: int, run_id: str
) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].hist(uncertainty["confidence"], bins=40, color="#4C72B0")
    axes[0].axvline(
        float(np.quantile(uncertainty["confidence"], 0.10)),
        color="#C44E52", linestyle="--", label="10th percentile (borderline)",
    )
    axes[0].set_title(f"GMM max posterior confidence | K={k}")
    axes[0].set_xlabel("max_k responsibility")
    axes[0].set_ylabel("Tracks")
    axes[0].legend(fontsize=8)
    axes[1].hist(uncertainty["normalized_entropy"], bins=40, color="#55A868")
    axes[1].set_title("Normalized posterior entropy")
    axes[1].set_xlabel("H(posterior) / log K")
    axes[1].set_ylabel("Tracks")
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.suptitle(
        f"Mixture-membership uncertainty conditional on the fitted GMM — not a genre probability "
        f"(run {run_id})",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return _finish(fig, path)


def plot_genre_heatmap(
    matrix: pd.DataFrame,
    path: str | Path,
    *,
    title: str,
    note: str,
) -> Path:
    values = matrix.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(max(10, 0.32 * matrix.shape[1] + 4), max(4.5, 0.5 * matrix.shape[0] + 2)))
    image = ax.imshow(values, aspect="auto", cmap="viridis")
    ax.set_xticks(range(matrix.shape[1]))
    ax.set_xticklabels(matrix.columns, rotation=55, ha="right", fontsize=7)
    ax.set_yticks(range(matrix.shape[0]))
    ax.set_yticklabels([f"C{index}" for index in matrix.index])
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            if np.isfinite(value) and value > 0:
                ax.text(column, row, f"{value:.0f}", ha="center", va="center", fontsize=5.5,
                        color="white" if value < np.nanmax(values) * 0.6 else "black")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("Genre (multi-label; a track can contain several)")
    ax.set_ylabel("Cluster")
    fig.colorbar(image, ax=ax, label="% unique tracks in cluster containing genre")
    fig.text(0.01, 0.005, note, fontsize=7, color="#444444")
    return _finish(fig, path)


def plot_eda_distributions(
    frames: dict[str, pd.DataFrame],
    feature_names: Sequence[str],
    path: str | Path,
    *,
    title: str,
) -> Path:
    source = frames.get("tracks")
    if source is None:
        raise ValueError("EDA distributions require the tracks frame.")
    n = len(feature_names)
    columns = 3
    rows = int(np.ceil(n / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(4.2 * columns, 3.2 * rows))
    axes_array = np.asarray(axes).reshape(-1)
    for ax, feature in zip(axes_array, feature_names, strict=False):
        ax.hist(source[feature].dropna(), bins=50, color="#4C72B0", alpha=0.85)
        ax.set_title(feature)
        ax.set_xlabel("value")
        ax.set_ylabel("tracks")
        ax.grid(alpha=0.2)
    for ax in axes_array[len(feature_names):]:
        ax.axis("off")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _finish(fig, path)


def plot_correlation(matrix: pd.DataFrame, path: str | Path, *, title: str) -> Path:
    fig, ax = plt.subplots(figsize=(8.5, 7))
    image = ax.imshow(matrix.to_numpy(dtype=float), cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix.iloc[row, column]
            ax.text(column, row, f"{value:.2f}", ha="center", va="center", fontsize=7)
    ax.set_title(title, fontsize=10)
    fig.colorbar(image, ax=ax, label="Pearson correlation")
    return _finish(fig, path)
