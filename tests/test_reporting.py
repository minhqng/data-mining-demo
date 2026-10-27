"""Focused tests for report table formatting."""

from pathlib import Path

from btl11.reporting import _display_path


def test_display_path_prefers_repo_relative_paths() -> None:
    figure = Path.cwd() / "artifacts" / "runs" / "run" / "figures" / "plot.png"
    assert _display_path(figure) == "artifacts/runs/run/figures/plot.png"


def test_display_path_keeps_paths_outside_the_repository(tmp_path: Path) -> None:
    figure = tmp_path / "plot.png"
    assert _display_path(figure) == figure.as_posix()
