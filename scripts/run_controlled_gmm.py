"""Run the exploratory controlled GMM comparison used by the robustness audit.

F9 / no-PCA, StandardScaler vs RobustScaler, K=2/4, covariance diag/full,
seed 42, GMM n_init = gmm.n_init_final. Each fit is a normal candidate run in a
separate experiment set so it never enters the baseline selection sweep.
"""

from __future__ import annotations

import argparse

from btl11.config import Config
from btl11.data import load_prepared
from btl11.experiment import (
    compare_experiment_set,
    descriptor_matches,
    iter_run_docs,
    run_candidate,
)
from btl11.utils import sha256_obj


def _existing_run_id(config: Config, prepared, descriptor: dict, experiment_set: str) -> str | None:
    fit_sha = sha256_obj(list(prepared.splits["fit"]))
    eval_sha = sha256_obj(list(prepared.splits["eval"]))
    matches = []
    for _, doc in iter_run_docs(config):
        if doc.get("status") != "completed" or doc.get("run_type") != "candidate":
            continue
        if doc.get("experiment_set") != experiment_set:
            continue
        if doc.get("dataset_sha256") != prepared.dataset_sha256:
            continue
        if doc.get("fit_ids_sha256") != fit_sha or doc.get("eval_ids_sha256") != eval_sha:
            continue
        if descriptor_matches(doc, descriptor):
            matches.append(doc)
    if not matches:
        return None
    return max(matches, key=lambda item: item.get("created_utc", "")).get("run_id")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--experiment-set", default="exploratory_controlled_gmm")
    parser.add_argument(
        "--n-init",
        type=int,
        default=None,
        help="GMM n_init; default uses gmm.n_init_final from the config.",
    )
    args = parser.parse_args()
    config = Config.load(args.config)
    prepared = load_prepared(config, enforce_license=True)
    n_init = int(args.n_init if args.n_init is not None else config.gmm["n_init_final"])
    run_ids: list[str] = []
    for scaler in ("standard", "robust"):
        for k in (2, 4):
            for covariance in ("diag", "full"):
                descriptor = {
                    "algorithm": "gmm",
                    "k": k,
                    "covariance_type": covariance,
                    "feature_set": "f9",
                    "scaler": scaler,
                    "pca": None,
                    "seed": int(config.seed),
                }
                existing = _existing_run_id(
                    config, prepared, descriptor, args.experiment_set
                )
                if existing is not None:
                    print(f"reuse {existing}")
                    run_ids.append(existing)
                    continue
                result = run_candidate(
                    config,
                    prepared,
                    algorithm="gmm",
                    k=k,
                    feature_set="f9",
                    scaler=scaler,
                    pca_variance=None,
                    seed=int(config.seed),
                    covariance_type=covariance,
                    n_init=n_init,
                    experiment_set=args.experiment_set,
                    experiment_note=(
                        f"EXPLORATORY controlled GMM comparison: {scaler} K={k} "
                        f"cov={covariance} n_init={n_init}; not selection evidence"
                    ),
                )
                print(result.run_id)
                run_ids.append(result.run_id)
    comparison = compare_experiment_set(config, args.experiment_set)
    print(f"comparison: {comparison['comparison_csv']}")
    print(f"n runs: {len(run_ids)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
