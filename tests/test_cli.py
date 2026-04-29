"""Tests for the command-line simulation workflow."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("biorbd")

from skating_aerial_alignment.cli import main  # noqa: E402


def test_cli_single_simulation_saves_parameters_summary_and_timeseries(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A single CLI simulation writes the expected output files."""

    output_dir = tmp_path / "single_run"
    exit_code = main(
        [
            "simulate",
            "--output-dir",
            str(output_dir),
            "--sigma-rps",
            "0.0",
            "0.2",
            "2.8",
            "--vertical-velocity",
            "2.6",
            "--backward-velocity",
            "1.4",
            "--print-summary",
        ]
    )

    assert exit_code == 0
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    parameters = json.loads((output_dir / "parameters.json").read_text(encoding="utf-8"))
    timeseries = np.load(output_dir / "timeseries.npz")
    stdout = capsys.readouterr().out

    assert "twist_turns" in stdout
    assert summary["backward_velocity_m_s"] == pytest.approx(1.4)
    assert parameters["angular_velocity_rps"] == [0.0, 0.2, 2.8]
    assert timeseries["time"].ndim == 1
    assert timeseries["q"].shape[0] == timeseries["time"].shape[0]


def test_cli_batch_simulation_saves_run_directories_and_batch_summary(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A batch configuration produces one folder per scenario plus a summary index."""

    config_path = tmp_path / "batch.json"
    config_path.write_text(
        json.dumps(
            {
                "scenarios": [
                    {
                        "label": "baseline",
                        "angular_velocity_rps": [0.0, 0.0, 3.0],
                    },
                    {
                        "label": "tilted",
                        "angular_velocity_rps": [0.0, 0.4, 3.0],
                        "inward_tilt_deg": -10.0,
                        "backward_horizontal_velocity": 1.2,
                    },
                ]
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "batch_runs"

    exit_code = main(["batch", "--config", str(config_path), "--output-dir", str(output_dir)])

    assert exit_code == 0
    assert capsys.readouterr().out.strip() == str(output_dir)
    batch_summary = json.loads((output_dir / "batch_summary.json").read_text(encoding="utf-8"))
    batch_summary_csv = (output_dir / "batch_summary.csv").read_text(encoding="utf-8")

    assert len(batch_summary) == 2
    assert "run_001_baseline" in batch_summary[0]["run_directory"]
    assert "twist_turns" in batch_summary_csv
    assert (output_dir / "run_001_baseline" / "summary.json").exists()
    assert (output_dir / "run_002_tilted" / "timeseries.npz").exists()
