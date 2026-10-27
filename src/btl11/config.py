"""YAML configuration loading and validation against locked BTL11 decisions."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from .errors import ConfigError
from .features import F5_FEATURES, F9_FEATURES, FORBIDDEN_PRIMARY
from .utils import canonical_json, read_json, sha256_obj

CANONICAL_KAGGLE_URL = (
    "https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset"
)
CANONICAL_KAGGLE_ID = "maharshipandya/-spotify-tracks-dataset"

TOP_LEVEL_KEYS = {
    "project",
    "data",
    "features",
    "preprocess",
    "search",
    "kmeans",
    "gmm",
    "visualization",
    "hierarchical",
    "compliance",
    "selection",
    "extensions",
}
SCALERS = ("standard", "robust")
COVARIANCE_TYPES = ("diag", "full")


def _deep_merge(base: dict[str, Any], updates: Mapping[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(dict(merged[key]), value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


class Config:
    """Validated view over the nested YAML configuration."""

    def __init__(self, data: dict[str, Any], path: Path | None = None) -> None:
        if not isinstance(data, dict):
            raise ConfigError("Configuration root must be a YAML mapping.")
        self._data = copy.deepcopy(data)
        self.path = path
        self.validate()

    # ------------------------------------------------------------------ load
    @classmethod
    def load(cls, path: str | Path) -> Config:
        config_path = Path(path)
        if not config_path.exists():
            raise ConfigError(
                f"Configuration file not found: {config_path}. "
                "Use configs/base.yaml or pass --config explicitly."
            )
        try:
            raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise ConfigError(f"Invalid YAML in {config_path}: {exc}") from exc
        return cls(raw, path=config_path)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any], path: Path | None = None) -> Config:
        return cls(dict(data), path=path)

    @classmethod
    def from_snapshot(cls, path: str | Path) -> Config:
        snapshot_path = Path(path)
        try:
            raw = yaml.safe_load(snapshot_path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            raise ConfigError(f"Cannot read config snapshot {snapshot_path}: {exc}") from exc
        return cls(raw, path=snapshot_path)

    def merged(self, updates: Mapping[str, Any]) -> Config:
        return Config(_deep_merge(self._data, updates), path=self.path)

    # ------------------------------------------------------------- properties
    @property
    def raw(self) -> dict[str, Any]:
        return copy.deepcopy(self._data)

    @property
    def project(self) -> dict[str, Any]:
        return self._data["project"]

    @property
    def data(self) -> dict[str, Any]:
        return self._data["data"]

    @property
    def features(self) -> dict[str, Any]:
        return self._data["features"]

    @property
    def preprocess(self) -> dict[str, Any]:
        return self._data["preprocess"]

    @property
    def search(self) -> dict[str, Any]:
        return self._data["search"]

    @property
    def kmeans(self) -> dict[str, Any]:
        return self._data["kmeans"]

    @property
    def gmm(self) -> dict[str, Any]:
        return self._data["gmm"]

    @property
    def visualization(self) -> dict[str, Any]:
        return self._data["visualization"]

    @property
    def hierarchical(self) -> dict[str, Any]:
        return self._data["hierarchical"]

    @property
    def compliance(self) -> dict[str, Any]:
        return self._data["compliance"]

    @property
    def selection(self) -> dict[str, Any]:
        return self._data["selection"]

    @property
    def extensions(self) -> dict[str, Any]:
        return self._data["extensions"]

    @property
    def seed(self) -> int:
        return int(self.project["seed"])

    @property
    def raw_path(self) -> Path:
        return Path(self.project["raw_path"])

    @property
    def output_dir(self) -> Path:
        return Path(self.project["output_dir"])

    @property
    def feature_primary(self) -> list[str]:
        return list(self.features["primary"])

    @property
    def feature_ablation(self) -> list[str]:
        return list(self.features["required_ablation"])

    # ---------------------------------------------------------------- hashing
    def canonical(self) -> str:
        return canonical_json(self._data)

    def sha256(self) -> str:
        return sha256_obj(self._data)

    def snapshot_yaml(self) -> str:
        return yaml.safe_dump(self._data, sort_keys=True, allow_unicode=True)

    # -------------------------------------------------------------- selection
    def descriptor(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Normalize and validate one candidate-model descriptor."""
        pca_value = payload.get("pca")
        if isinstance(pca_value, str) and pca_value.lower() == "none":
            pca_value = None
        descriptor = {
            "algorithm": str(payload.get("algorithm", "")).lower(),
            "k": payload.get("k"),
            "covariance_type": payload.get("covariance_type"),
            "feature_set": str(payload.get("feature_set", "f9")).lower(),
            "scaler": str(payload.get("scaler", self.preprocess["scaler_primary"])).lower(),
            "pca": pca_value,
            "seed": payload.get("seed", self.seed),
        }
        validate_descriptor(descriptor, self)
        return descriptor

    # ------------------------------------------------------------- validation
    def validate(self) -> None:
        missing = sorted(TOP_LEVEL_KEYS - set(self._data))
        if missing:
            raise ConfigError(f"Configuration is missing top-level sections: {missing}")
        unknown = sorted(set(self._data) - TOP_LEVEL_KEYS)
        if unknown:
            raise ConfigError(f"Unknown top-level configuration sections: {unknown}")

        project = self.project
        for key in ("name", "raw_path", "output_dir"):
            if not isinstance(project.get(key), str) or not project[key]:
                raise ConfigError(f"project.{key} must be a non-empty string.")
        _require_int(project, "seed", minimum=0)
        if not isinstance(project.get("license_review_required"), bool):
            raise ConfigError("project.license_review_required must be boolean.")

        data = self.data
        if data.get("unit") != "unique_track_id":
            raise ConfigError("data.unit is locked to 'unique_track_id'.")
        if data.get("on_conflicting_track_features") != "fail":
            raise ConfigError(
                "data.on_conflicting_track_features is locked to 'fail' until a documented ADR."
            )
        if data.get("missing_numeric") != "drop_after_audit":
            raise ConfigError(
                "data.missing_numeric='impute_dev_only' is a declared SSOT option but is not "
                "implemented; primary runs must use 'drop_after_audit' (quarantine invalid rows)."
            )
        if data.get("split_method") != "stable_id_hash":
            raise ConfigError("data.split_method must be stable_id_hash.")
        dev_fraction = _require_number(data, "dev_fraction", minimum=0.0, maximum=1.0)
        holdout_fraction = _require_number(
            data, "holdout_fraction", minimum=0.0, maximum=1.0
        )
        if abs(dev_fraction + holdout_fraction - 1.0) > 1e-9:
            raise ConfigError("data.dev_fraction + data.holdout_fraction must equal 1.0.")
        _require_int(data, "split_seed", minimum=0)
        _require_int(data, "eval_cohort_size", minimum=2)
        _require_int(data, "eval_seed", minimum=0)
        _require_number(data, "feature_tolerance_atol", minimum=0.0)
        _require_number(data, "feature_tolerance_rtol", minimum=0.0)

        features = self.features
        if list(features.get("primary", [])) != list(F9_FEATURES):
            raise ConfigError(
                "features.primary must equal the locked F9 allowlist in order: "
                f"{list(F9_FEATURES)}"
            )
        if list(features.get("required_ablation", [])) != list(F5_FEATURES):
            raise ConfigError(
                "features.required_ablation must equal the locked F5 allowlist in order: "
                f"{list(F5_FEATURES)}"
            )
        forbidden = features.get("forbidden_primary", [])
        missing_forbidden = set(FORBIDDEN_PRIMARY) - set(forbidden)
        if missing_forbidden:
            raise ConfigError(
                f"features.forbidden_primary must include metadata/label guardrails: "
                f"{sorted(missing_forbidden)}"
            )
        if set(F9_FEATURES) & set(forbidden):
            raise ConfigError("A primary audio feature cannot also be forbidden.")

        preprocess = self.preprocess
        if preprocess.get("scaler_primary") != "standard":
            raise ConfigError("preprocess.scaler_primary is locked to 'standard'.")
        if preprocess.get("scaler_ablation") != "robust":
            raise ConfigError("preprocess.scaler_ablation is locked to 'robust'.")
        if not isinstance(preprocess.get("pca_whiten"), bool):
            raise ConfigError("preprocess.pca_whiten must be boolean.")
        candidates = preprocess.get("pca_candidates")
        if not isinstance(candidates, list) or not candidates:
            raise ConfigError("preprocess.pca_candidates must be a non-empty list.")
        for candidate in candidates:
            if candidate is None or candidate == "none":
                continue
            if not isinstance(candidate, (int, float)) or not 0.0 < float(candidate) < 1.0:
                raise ConfigError(
                    "Each PCA candidate must be null/'none' or a variance in (0, 1)."
                )

        search = self.search
        quick_k = search.get("quick_k")
        if not isinstance(quick_k, list) or len(quick_k) < 1:
            raise ConfigError("search.quick_k must be a non-empty list.")
        if any(not isinstance(k, int) or isinstance(k, bool) or k < 2 for k in quick_k):
            raise ConfigError("search.quick_k must contain integers >= 2.")
        if quick_k != sorted(set(quick_k)):
            raise ConfigError("search.quick_k must be sorted and unique.")
        for key in ("quick_seeds", "stability_seeds"):
            seeds = search.get(key)
            if not isinstance(seeds, list) or not seeds:
                raise ConfigError(f"search.{key} must be a non-empty list.")
            if any(not isinstance(seed, int) or isinstance(seed, bool) for seed in seeds):
                raise ConfigError(f"search.{key} must contain integers.")
        _require_int(search, "silhouette_sample_size", minimum=2)
        _require_int(search, "silhouette_seed", minimum=0)
        if int(search["silhouette_sample_size"]) != int(data["eval_cohort_size"]):
            raise ConfigError(
                "search.silhouette_sample_size must equal data.eval_cohort_size: the fixed "
                "evaluation cohort is the scored cohort, and a divergence would silently "
                "score a different sample than the one reported."
            )
        if int(search["silhouette_seed"]) != int(data["eval_seed"]):
            raise ConfigError(
                "search.silhouette_seed must equal data.eval_seed for the same fixed cohort."
            )
        _require_int(search, "max_fit_count_warning", minimum=1)

        kmeans = self.kmeans
        if kmeans.get("init") not in {"k-means++", "random"}:
            raise ConfigError("kmeans.init must be 'k-means++' or 'random'.")
        _require_int(kmeans, "n_init", minimum=1)
        _require_int(kmeans, "max_iter", minimum=1)
        _require_number(kmeans, "tol", minimum=0.0, exclusive=True)

        gmm = self.gmm
        covariances = gmm.get("covariance_candidates")
        if not isinstance(covariances, list) or not covariances:
            raise ConfigError("gmm.covariance_candidates must be a non-empty list.")
        if not set(covariances) <= set(COVARIANCE_TYPES):
            raise ConfigError("gmm.covariance_candidates may only contain diag/full.")
        _require_int(gmm, "n_init_quick", minimum=1)
        _require_int(gmm, "n_init_final", minimum=1)
        _require_int(gmm, "max_iter", minimum=1)
        _require_number(gmm, "reg_covar_quick", minimum=0.0, exclusive=True)
        fallbacks = gmm.get("reg_covar_fallback", [])
        if not isinstance(fallbacks, list) or any(
            not isinstance(value, (int, float)) or value <= 0 for value in fallbacks
        ):
            raise ConfigError("gmm.reg_covar_fallback must be a list of positive numbers.")
        if gmm.get("init_params") not in {"kmeans", "k-means++", "random", "random_from_data"}:
            raise ConfigError("gmm.init_params is not a supported scikit-learn option.")
        if not isinstance(gmm.get("require_convergence"), bool):
            raise ConfigError("gmm.require_convergence must be boolean.")

        visualization = self.visualization
        _require_int(visualization, "pca_plot_components", minimum=2)
        _require_int(visualization, "tsne_sample_size", minimum=10)
        _require_number(visualization, "tsne_perplexity", minimum=1.0)
        _require_int(visualization, "tsne_random_state", minimum=0)
        _require_int(visualization, "scatter_sample_size", minimum=10)

        hierarchical = self.hierarchical
        if hierarchical.get("enabled") not in {"conditional", True, False}:
            raise ConfigError("hierarchical.enabled must be conditional/true/false.")
        _require_int(hierarchical, "sample_size", minimum=10)
        if hierarchical.get("linkage") != "ward":
            raise ConfigError("hierarchical.linkage is locked to ward.")
        if hierarchical.get("metric") != "euclidean":
            raise ConfigError("hierarchical.metric is locked to euclidean.")

        compliance = self.compliance
        if compliance.get("canonical_kaggle_url") != CANONICAL_KAGGLE_URL:
            raise ConfigError(
                "compliance.canonical_kaggle_url must be the mandated Kaggle dataset URL."
            )
        if compliance.get("kaggle_dataset_id") != CANONICAL_KAGGLE_ID:
            raise ConfigError(
                "compliance.kaggle_dataset_id must be "
                f"{CANONICAL_KAGGLE_ID!r}; other datasets are forbidden."
            )
        for key in ("allow_pending_license_local_analysis", "redistribution_allowed"):
            if not isinstance(compliance.get(key), bool):
                raise ConfigError(f"compliance.{key} must be boolean.")
        if compliance["redistribution_allowed"]:
            raise ConfigError(
                "This project has not cleared public redistribution; "
                "compliance.redistribution_allowed must stay false."
            )

        selection = self.selection
        for key in ("shortlist_k", "finalists", "frozen"):
            value = selection.get(key)
            if not isinstance(value, list):
                raise ConfigError(f"selection.{key} must be a list.")
        if any(not isinstance(k, int) or k < 2 for k in selection["shortlist_k"]):
            raise ConfigError("selection.shortlist_k must contain integers >= 2.")
        algorithms = selection.get("ablation_algorithms", [])
        if not set(algorithms) <= {"kmeans", "gmm"}:
            raise ConfigError("selection.ablation_algorithms may only contain kmeans/gmm.")
        if selection.get("ablation_covariance") not in COVARIANCE_TYPES:
            raise ConfigError("selection.ablation_covariance must be diag/full.")
        _require_int(selection, "stability_bootstrap", minimum=1)
        _require_int(selection, "stability_n_init_gmm", minimum=1)
        for entry in selection["finalists"] + selection["frozen"]:
            self.descriptor(entry)
        if selection.get("selected") is not None:
            self.descriptor(selection["selected"])

        extensions = self.extensions
        for key, value in extensions.items():
            if not isinstance(value, bool):
                raise ConfigError(f"extensions.{key} must be boolean.")

    # ------------------------------------------------------------- provenance
    def license_operational_status(self, manifest: Mapping[str, Any]) -> str:
        """Classify whether local empirical analysis is allowed."""
        status = str(manifest.get("license_review_status", "missing"))
        local_authorized = bool(
            manifest.get("local_analysis_authorized_by_owner")
            or self.compliance.get("allow_pending_license_local_analysis")
        )
        if status == "rejected":
            raise ConfigError(
                "License gate: the dataset review status is 'rejected'. "
                "Do not run empirical analysis; obtain a permissible data source decision."
            )
        if (
            status in {"missing", "unknown"}
            or status.startswith("pending")
        ) and not local_authorized:
            raise ConfigError(
                "License gate: license review is not cleared and local analysis is not "
                "authorized. Complete the human review in docs/DATASET_PROVENANCE.md and "
                "record it in data/raw/MANIFEST.json."
            )
        if status.startswith("pending") and local_authorized:
            return "local_academic_analysis_allowed_redistribution_pending"
        return status


def _require_int(section: Mapping[str, Any], key: str, minimum: int) -> int:
    value = section.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ConfigError(f"{key} must be an integer >= {minimum}.")
    return value


def _require_number(
    section: Mapping[str, Any],
    key: str,
    minimum: float,
    maximum: float | None = None,
    exclusive: bool = False,
) -> float:
    value = section.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ConfigError(f"{key} must be numeric.")
    numeric = float(value)
    if exclusive and numeric <= minimum:
        raise ConfigError(f"{key} must be > {minimum}.")
    if not exclusive and numeric < minimum:
        raise ConfigError(f"{key} must be >= {minimum}.")
    if maximum is not None and numeric > maximum:
        raise ConfigError(f"{key} must be <= {maximum}.")
    return numeric


def validate_descriptor(descriptor: Mapping[str, Any], config: Config) -> None:
    algorithm = descriptor.get("algorithm")
    if algorithm not in {"kmeans", "gmm"}:
        raise ConfigError(f"selection algorithm must be kmeans/gmm; got {algorithm!r}.")
    k = descriptor.get("k")
    if not isinstance(k, int) or isinstance(k, bool) or k < 2:
        raise ConfigError(f"selection k must be an integer >= 2; got {k!r}.")
    if descriptor.get("feature_set") not in {"f5", "f9"}:
        raise ConfigError("selection feature_set must be f5/f9.")
    if descriptor.get("scaler") not in SCALERS:
        raise ConfigError("selection scaler must be standard/robust.")
    pca = descriptor.get("pca")
    if pca is not None and (not isinstance(pca, (int, float)) or not 0.0 < float(pca) < 1.0):
        raise ConfigError("selection pca must be null or a variance in (0, 1).")
    seed = descriptor.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ConfigError("selection seed must be a non-negative integer.")
    covariance = descriptor.get("covariance_type")
    if algorithm == "gmm":
        if covariance not in COVARIANCE_TYPES:
            raise ConfigError("A GMM descriptor requires covariance_type diag/full.")
    elif covariance not in (None, ""):
        raise ConfigError("A K-Means descriptor must not set covariance_type.")


def read_selection_state(path: str | Path) -> dict[str, Any]:
    """Read a previously written selection state, if present."""
    candidate = Path(path)
    if not candidate.exists():
        return {}
    return read_json(candidate)
