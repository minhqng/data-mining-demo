"""Synthetic regression tests for the exploratory robustness analysis."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from btl11.errors import ConfigError
from btl11.experiment import run_candidate
from btl11.features import F9_FEATURES
from btl11.robustness import run_robustness_analysis


def _build(prepared_env, tmp_path: Path, scaler: str = "robust"):
    _, config, _, prepared = prepared_env
    candidate = run_candidate(
        config,
        prepared,
        algorithm="kmeans",
        k=2,
        scaler=scaler,
        seed=42,
        experiment_set="robustness_probe",
    )
    result = run_robustness_analysis(
        config,
        prepared,
        evaluation_run_id=candidate.run_id,
        output_dir=tmp_path / "artifacts" / "robustness",
        include_stability=False,
        include_scaler_comparison=False,
    )
    return config, prepared, candidate, result


def test_robustness_outputs_and_decomposition(prepared_env, tmp_path):
    _, _, _, result = _build(prepared_env, tmp_path)
    out = Path(result["output_dir"])
    assert (out / "robustness.json").exists()
    scaler = pd.read_csv(out / "scaler_parameters.csv")
    decomposition = pd.read_csv(out / "variance_decomposition.csv")
    contributions = pd.read_csv(out / "assignment_contributions.csv")
    single = pd.read_csv(out / "single_feature_partitions.csv")
    loo = pd.read_csv(out / "leave_one_feature_out.csv")
    profiles = pd.read_csv(out / "loo_profiles.csv")
    assert len(scaler) == 9
    assert len(decomposition) == 9
    assert len(contributions) == 9
    assert len(single) == 9
    assert len(loo) == 10  # frozen reference + nine leave-one-out variants
    assert not profiles.empty
    np.testing.assert_allclose(
        decomposition["within_ss"] + decomposition["between_ss"],
        decomposition["total_ss"],
        rtol=1e-9,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        contributions["share_of_assigned_squared_distance"].sum(), 1.0, rtol=1e-9
    )
    assert set(loo["dropped_feature"]) == {"none (frozen reference)", *F9_FEATURES}
    assert (out / "figures" / "loo_ari.png").exists()
    summary = json.loads((out / "robustness.json").read_text())
    assert summary["evidence_class"] == "post_hoc_exploratory_development_only"
    assert summary["confirmatory"] is False
    assert summary["holdout_used"] is False
    assert summary["selection_changed"] is False
    assert summary["controlled_gmm_comparison"]["n_rows"] == 8
    assert summary["dominance"]["instrumentalness_between_share_of_total_between"] is not None
    assert summary["dominance"]["interpretation"]


def test_robustness_rejects_non_robust_frozen_model(prepared_env, tmp_path):
    _, config, _, prepared = prepared_env
    candidate = run_candidate(
        config,
        prepared,
        algorithm="kmeans",
        k=2,
        scaler="standard",
        seed=42,
        experiment_set="robustness_probe",
    )
    with pytest.raises(ConfigError, match="RobustScaler"):
        run_robustness_analysis(
            config,
            prepared,
            evaluation_run_id=candidate.run_id,
            output_dir=tmp_path / "artifacts" / "robustness",
            include_stability=False,
            include_scaler_comparison=False,
        )
