"""Smoke tests for the project bootstrap."""

import os
import subprocess
import sys
from pathlib import Path

from skating_aerial_alignment import __version__


def test_package_version_is_defined() -> None:
    """The package exposes a semantic version string."""

    assert __version__ == "0.1.0"


def test_gui_launcher_can_bootstrap_the_local_src_directory() -> None:
    """The checkout launcher works without relying on a PYTHONPATH environment variable."""

    project_root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    script = (
        "import runpy; "
        "namespace = runpy.run_path('skating_aerial_alignment_gui.py'); "
        "namespace['_ensure_local_src_on_path'](); "
        "import skating_aerial_alignment; "
        "print(skating_aerial_alignment.__version__)"
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=project_root,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == "0.1.0"
