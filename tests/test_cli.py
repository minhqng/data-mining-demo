"""CLI smoke tests and actionable failure modes."""

from __future__ import annotations

import pytest

from btl11.cli import main

from .conftest import make_config, write_config


def test_cli_help_smoke():
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0


def test_cli_dry_run_creates_no_run_artifacts(fixture_env):
    tmp_path, _, config_path = fixture_env
    assert main(["sweep", "--algorithm", "kmeans", "--config", str(config_path), "--dry-run"]) == 0
    assert not (tmp_path / "artifacts" / "runs").exists()
    assert not (tmp_path / "data" / "processed").exists()
    assert main(["sweep", "--algorithm", "gmm", "--config", str(config_path), "--dry-run"]) == 0


def test_cli_missing_dataset_explains_remediation(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    make_config(tmp_path)
    config = make_config(tmp_path)
    config_path = write_config(config, tmp_path)
    code = main(["prepare", "--config", str(config_path)])
    assert code == 2


def test_cli_audit_and_prepare_on_synthetic_fixture(fixture_env):
    tmp_path, _, config_path = fixture_env
    assert main(["audit", "--config", str(config_path)]) == 0
    assert (tmp_path / "data" / "raw" / "MANIFEST.json").exists()
    assert main(["prepare", "--config", str(config_path)]) == 0
    assert (tmp_path / "data" / "processed" / "tracks.parquet").exists()
    assert (tmp_path / "data" / "splits" / "dev_ids.csv").exists()
