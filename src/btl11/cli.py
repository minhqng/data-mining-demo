"""Command-line interface with explicit, actionable failures and exit codes."""

from __future__ import annotations

import argparse
import sys
import traceback
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from . import __version__
from .config import Config
from .data import audit_only, load_prepared, prepare_dataset
from .eda import run_eda
from .errors import BTL11Error, ConfigError
from .experiment import (
    ablate,
    compare_experiment_set,
    dry_run_sweep,
    load_run_doc,
    refit_all_tracks,
    run_dir,
    run_sweep,
    stability,
    validate_holdout,
    ward_diagnostic,
)
from .models import load_bundle
from .profiling import generate_profiles
from .reduction import run_pca_diagnostics
from .reporting import run_report
from .robustness import run_robustness_analysis
from .run_visuals import visualize_run
from .utils import atomic_write_json, safe_slug
from .verification import verify_run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="btl11",
        description=(
            "BTL11 — Music Genre & Audio Track Segmentation. Reproducible unsupervised "
            "acoustic clustering on the mandated Kaggle Spotify Tracks Dataset."
        ),
    )
    parser.add_argument("--version", action="version", version=f"btl11 {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    def with_config(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--config",
            default="configs/base.yaml",
            help="YAML configuration (default: configs/base.yaml). Run from the repo root.",
        )

    p = sub.add_parser("audit", help="Audit the exact raw CSV, schema, duplicates and provenance.")
    p.add_argument("--input", default=None, help="Raw CSV; must equal project.raw_path.")
    with_config(p)
    p.set_defaults(func=cmd_audit)

    p = sub.add_parser(
        "prepare",
        help="Deduplicate unique tracks, preserve multi-label genres, write stable splits.",
    )
    with_config(p)
    p.set_defaults(func=cmd_prepare)

    p = sub.add_parser("eda", help="Generate EDA tables and figures from prepared tracks.")
    with_config(p)
    p.set_defaults(func=cmd_eda)

    p = sub.add_parser("pca", help="Fit PCA 90/95 on the development fit cohort and report loadings.")
    with_config(p)
    p.set_defaults(func=cmd_pca)

    p = sub.add_parser("sweep", help="Staged baseline K-Means or GMM sweep.")
    p.add_argument("--algorithm", required=True, choices=["kmeans", "gmm"])
    p.add_argument("--dry-run", action="store_true", help="Print exact jobs; create nothing.")
    p.add_argument("--allow-many-fits", action="store_true", help="Explicitly raise the fit guard.")
    with_config(p)
    p.set_defaults(func=cmd_sweep)

    p = sub.add_parser("compare", help="Aggregate completed runs of one experiment set.")
    p.add_argument("--experiment-set", required=True)
    with_config(p)
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("ablate", help="Scaler / feature-set / PCA ablation on the shortlist.")
    p.add_argument("--factor", required=True, choices=["scaler", "feature_set", "pca"])
    with_config(p)
    p.set_defaults(func=cmd_ablate)

    p = sub.add_parser(
        "stability",
        help="Multi-seed and bootstrap ARI on one fixed reference cohort.",
    )
    p.add_argument("--model-selector", default="finalists", choices=["finalists", "frozen_finalists"])
    with_config(p)
    p.set_defaults(func=cmd_stability)

    p = sub.add_parser("validate-holdout", help="One-time frozen-model holdout check.")
    p.add_argument("--model-selector", default="frozen_finalists", choices=["frozen_finalists"])
    p.add_argument(
        "--allow-repeat",
        action="store_true",
        help="Explicitly record a contaminated repeated holdout use (not recommended).",
    )
    with_config(p)
    p.set_defaults(func=cmd_validate_holdout)

    p = sub.add_parser("refit", help="Refit the selected configuration on all unique tracks.")
    p.add_argument("--model-selector", default="selected_final", choices=["selected_final"])
    p.add_argument(
        "--all-tracks",
        action="store_true",
        help="Required safety flag: confirm the intentional full-data refit.",
    )
    with_config(p)
    p.set_defaults(func=cmd_refit)

    p = sub.add_parser("profile", help="Raw-unit cluster profiles and representative local tracks.")
    p.add_argument("--run-id", required=True)
    p.add_argument("--force", action="store_true", help="Regenerate an existing profiles.csv.")
    with_config(p)
    p.set_defaults(func=cmd_profile)

    p = sub.add_parser("visualize", help="PCA2D / t-SNE / profile / post-hoc genre views.")
    p.add_argument("--run-id", required=True)
    p.add_argument("--kind", default="pca,tsne,profile")
    with_config(p)
    p.set_defaults(func=cmd_visualize)

    p = sub.add_parser("report", help="Generate machine report tables from a final refit run.")
    p.add_argument("--selected-run", required=True)
    p.add_argument("--output", default="artifacts/final-report-tables")
    with_config(p)
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("verify", help="Independently verify a run's manifest, hashes and metrics.")
    p.add_argument("--run-id", required=True)
    p.add_argument("--verbose", action="store_true", help="Print every check.")
    with_config(p)
    p.set_defaults(func=cmd_verify)

    p = sub.add_parser("ward", help="Bounded Ward diagnostic against the frozen selected model.")
    with_config(p)
    p.set_defaults(func=cmd_ward)

    p = sub.add_parser(
        "robustness",
        help="Post-hoc feature-dominance and leave-one-feature-out sensitivity (development only).",
    )
    p.add_argument(
        "--evaluation-run",
        default=None,
        help="Completed K-Means run to analyse; default resolves selection.frozen.",
    )
    p.add_argument("--output-dir", default="artifacts/robustness")
    p.add_argument(
        "--skip-stability",
        action="store_true",
        help="Skip the multi-seed refits for every leave-one-feature-out variant.",
    )
    p.add_argument(
        "--skip-scaler-comparison",
        action="store_true",
        help="Skip the StandardScaler feature-attribution comparison.",
    )
    with_config(p)
    p.set_defaults(func=cmd_robustness)
    return parser


def _config(args: argparse.Namespace) -> Config:
    return Config.load(args.config)


def _prepared(config: Config):
    return load_prepared(config, enforce_license=True)


def cmd_audit(args: argparse.Namespace) -> int:
    config = _config(args)
    result = audit_only(config, args.input)
    audit = result["audit"]
    manifest = result["manifest"]
    print("AUDIT PASS")
    print(f"  raw sha256 : {manifest['sha256']}")
    print(f"  rows       : {audit['counts']['n_rows']:,}")
    print(f"  unique IDs : {audit['counts']['n_unique_track_ids']:,}")
    print(f"  genres     : {audit['counts']['n_track_genres']}")
    print(f"  multi-genre tracks: {audit['counts']['n_tracks_with_multiple_genres']:,}")
    print(f"  conflicting IDs   : {audit['counts']['n_conflicting_track_ids']}")
    print(f"  license status    : {manifest['license_review_status']}")
    print(f"  manifest          : {Path(config.raw_path).parent / 'MANIFEST.json'}")
    print("  audit             : data/processed/audit.json")
    return 0


def cmd_prepare(args: argparse.Namespace) -> int:
    config = _config(args)
    prepared = prepare_dataset(config)
    summary = prepared.summary
    print("PREPARE PASS")
    print(f"  raw rows                 : {summary['raw_rows']:,}")
    print(f"  unique tracks            : {summary['unique_track_ids']:,}")
    print(f"  multi-genre tracks       : {summary['tracks_with_multiple_genres']:,}")
    print(f"  genre bridge rows        : {summary['genre_bridge_rows']:,}")
    print(f"  conflicting IDs          : {summary['conflicting_track_ids']}")
    print(f"  processed sha256         : {summary['processed_sha256']}")
    print(f"  split sha256             : {summary['split_sha256']}")
    print(f"  splits                   : {summary['splits']}")
    print("  outputs                  : data/processed/, data/splits/")
    return 0


def cmd_eda(args: argparse.Namespace) -> int:
    config = _config(args)
    result = run_eda(config, _prepared(config))
    print("EDA PASS")
    print(f"  output: {result['output_dir']}")
    print(f"  figures: {len(result['figures'])}")
    return 0


def cmd_pca(args: argparse.Namespace) -> int:
    config = _config(args)
    result = run_pca_diagnostics(config, _prepared(config))
    print("PCA DIAGNOSTICS PASS")
    print(f"  output: {result['output_dir']}")
    for name, entry in result["results"].items():
        if entry.get("pca_present"):
            print(
                f"  {name}: {entry['n_components']} components, "
                f"cumulative variance={entry['cumulative_final']:.6f}"
            )
        else:
            print(f"  {name}: identity path ({entry['n_components']} features)")
    return 0


def cmd_sweep(args: argparse.Namespace) -> int:
    config = _config(args)
    if args.dry_run:
        plan = dry_run_sweep(config, args.algorithm)
        print("DRY RUN — no files, no fits, no successful-run artifact.")
        print(f"  algorithm : {plan['algorithm']}")
        print(f"  n_jobs    : {plan['n_jobs']} (guard {plan['fit_count_limit']})")
        frame = pd.DataFrame(plan["jobs"])
        print(frame.to_string(index=False))
        print(f"  raw input : {plan['data_required']}")
        print(f"  prepared  : {plan['prepared_required']}")
        print(plan["note"])
        return 0
    prepared = _prepared(config)
    result = run_sweep(config, prepared, args.algorithm, allow_many_fits=args.allow_many_fits)
    print("SWEEP COMPLETED" if result["n_failed"] == 0 else "SWEEP COMPLETED WITH FAILURES")
    print(f"  output     : {Path(result['sweep_id']).as_posix()}")
    print(f"  completed  : {result['n_completed']}")
    print(f"  failed     : {result['n_failed']}")
    print(f"  run IDs    : {', '.join(result['run_ids'][:6])}{' ...' if len(result['run_ids']) > 6 else ''}")
    print(f"  artifacts  : artifacts/sweeps/{result['sweep_id']}/")
    return 0 if result["n_failed"] == 0 else 3


def cmd_compare(args: argparse.Namespace) -> int:
    config = _config(args)
    result = compare_experiment_set(config, args.experiment_set)
    print("COMPARE PASS")
    print(f"  completed: {result['n_completed']} | failed: {result['n_failed']}")
    print(f"  table    : {result['comparison_csv']}")
    print(f"  markdown : {result['comparison_md']}")
    return 0


def cmd_ablate(args: argparse.Namespace) -> int:
    config = _config(args)
    result = ablate(config, _prepared(config), args.factor)
    print("ABLATION PASS" if result["n_failed"] == 0 else "ABLATION WITH FAILURES")
    print(f"  output: {result['output_dir']}")
    print(f"  rows  : {result['n_rows']} | failed fits: {result['n_failed']}")
    return 0 if result["n_failed"] == 0 else 3


def cmd_stability(args: argparse.Namespace) -> int:
    config = _config(args)
    result = stability(config, _prepared(config), args.model_selector)
    print("STABILITY PASS")
    print(f"  run   : {result['run_id']}")
    print(f"  output: {result['path']}")
    for row in result["summary"]:
        print(
            f"  {row['finalist']}: seed ARI mean={row['seed_ari_mean']}, "
            f"bootstrap ARI mean={row['bootstrap_ari_mean']}, "
            f"pairwise mean={row['pairwise_ari_mean']}"
        )
    if result["failures"]:
        print(f"  failed refits: {len(result['failures'])} (see tables/stability_runs.csv)")
        return 3
    return 0


def cmd_validate_holdout(args: argparse.Namespace) -> int:
    config = _config(args)
    result = validate_holdout(config, _prepared(config), allow_repeat=args.allow_repeat)
    undefined = [
        name
        for name in ("silhouette", "davies_bouldin", "calinski_harabasz")
        if result["metrics"][name]["value"] is None
    ]
    if undefined:
        print("HOLDOUT COMPLETED WITH UNDEFINED METRIC(S): " + ", ".join(undefined))
    elif result["reused"]:
        print("HOLDOUT REPEATED (contaminated by explicit --allow-repeat)")
    else:
        print("HOLDOUT VALIDATION PASS")
    print(f"  run          : {result['run_id']}")
    print(f"  frozen model : {result['frozen_run_id']}")
    print(f"  output       : {result['path']}")
    for name in ("silhouette", "davies_bouldin", "calinski_harabasz"):
        entry = result["metrics"][name]
        print(f"  {name}: {entry['value']} (reason={entry['reason']})")
    return 0 if not undefined else 3


def cmd_refit(args: argparse.Namespace) -> int:
    if not args.all_tracks:
        raise ConfigError(
            "refit requires the explicit safety flag --all-tracks; the full-data refit is a "
            "distinct model identity and must be intentional."
        )
    config = _config(args)
    result = refit_all_tracks(config, _prepared(config), args.model_selector)
    print("FINAL REFIT PASS")
    print(f"  run              : {result['run_id']}")
    print(f"  evaluation model : {result['evaluation_run_id']}")
    print(f"  output           : {result['path']}")
    print(f"  profiles         : {result['profile_info']['files']}")
    print(f"  genre post-hoc   : {result['genre_info']['files']}")
    return 0


def cmd_profile(args: argparse.Namespace) -> int:
    config = _config(args)
    prepared = _prepared(config)
    doc = load_run_doc(config, args.run_id)
    if doc.get("status") != "completed":
        raise ConfigError(f"Run {args.run_id} is not completed.")
    path = run_dir(config, args.run_id)
    if (path / "profiles.csv").exists() and not args.force:
        print(f"PROFILE already exists: {path / 'profiles.csv'} (use --force to regenerate)")
        return 0
    assignments = pd.read_parquet(path / "assignments.parquet")
    feature_names = list(doc.get("feature_order") or config.feature_primary)
    bundle = load_bundle(path / "model.joblib") if doc.get("has_model") else None
    posthoc_path = Path("artifacts/posthoc") / args.run_id
    posthoc_path.mkdir(parents=True, exist_ok=True)
    result = generate_profiles(
        config,
        prepared,
        posthoc_path,
        args.run_id,
        assignments,
        bundle=bundle,
        all_tracks_scope=doc.get("run_type") == "final_refit",
        feature_names=feature_names,
    )
    print("PROFILE PASS")
    print(f"  output: {result['files']}")
    print(f"  clusters: {result['n_clusters']} | assigned tracks: {result['n_assigned']}")
    return 0


def cmd_visualize(args: argparse.Namespace) -> int:
    config = _config(args)
    prepared = _prepared(config)
    kinds = [kind.strip() for kind in args.kind.split(",") if kind.strip()]
    result = visualize_run(config, prepared, args.run_id, kinds)
    print("VISUALIZE PASS")
    print(f"  run  : {result['run_id']}")
    print(f"  kinds: {', '.join(result['kinds'])}")
    for name, payload in result["outputs"].items():
        print(f"  {name}: {payload.get('figure') or payload.get('skipped') or payload}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    config = _config(args)
    prepared = _prepared(config)
    result = run_report(config, prepared, args.selected_run, args.output)
    print("REPORT PASS")
    print(f"  output: {result['output_dir']}")
    print(f"  tables: {result['report_tables_md']}")
    print(f"  figures indexed: {result['n_figures']}")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    config = _config(args)
    result = verify_run(config, args.run_id)
    log_dir = Path("artifacts/verification")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = atomic_write_json(
        log_dir / f"{safe_slug(result['run_id'], 180)}.json", result
    )
    print("VERIFY PASS")
    print(f"  run      : {result['run_id']} ({result['run_type']})")
    print(f"  checks   : {result['n_checks']} passed")
    print(f"  path     : {result['path']}")
    print(f"  log      : {log_path}")
    if args.verbose:
        for check in result["checks"]:
            print(f"    [{check['status']}] {check['check']}: {check['detail']}")
    return 0


def cmd_ward(args: argparse.Namespace) -> int:
    config = _config(args)
    result = ward_diagnostic(config, _prepared(config))
    print("WARD DIAGNOSTIC PASS")
    print(f"  run              : {result['run_id']}")
    print(f"  output           : {result['path']}")
    print(f"  ARI vs selected  : {result['ari_vs_selected']}")
    print(f"  silhouette       : {result['silhouette']}")
    return 0


def cmd_robustness(args: argparse.Namespace) -> int:
    config = _config(args)
    result = run_robustness_analysis(
        config,
        _prepared(config),
        evaluation_run_id=args.evaluation_run,
        output_dir=args.output_dir,
        include_stability=not args.skip_stability,
        include_scaler_comparison=not args.skip_scaler_comparison,
    )
    dominance = result["dominance"]
    print("ROBUSTNESS ANALYSIS PASS")
    print(f"  output: {result['output_dir']}")
    print(f"  frozen run: {result['evaluation_run_id']}")
    print(
        "  instrumentalness: "
        f"between-SS share={dominance['instrumentalness_between_share_of_total_between']}, "
        f"margin share={dominance['instrumentalness_share_of_mean_margin']}, "
        f"single-feature ARI={dominance['instrumentalness_single_feature_ari']}, "
        f"LOO ARI={dominance['loo_instrumentalness_ari']}"
    )
    scaler = result["scaler_comparison"]
    if scaler.get("available"):
        print(
            "  standard-vs-robust partition ARI="
            f"{scaler['cross_scaler_partition_ari']:.6f}"
        )
    gmm = result["controlled_gmm_comparison"]
    print(
        f"  controlled GMM comparisons: {gmm['n_complete_comparisons']}/{gmm['n_rows']} "
        f"(ARI range {gmm['ari_min']}..{gmm['ari_max']})"
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except BTL11Error as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    except Exception:
        print("UNEXPECTED ERROR — full traceback follows:", file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
