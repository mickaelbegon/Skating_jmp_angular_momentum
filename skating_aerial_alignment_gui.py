"""Convenience launcher for the skating aerial-alignment GUI."""

from __future__ import annotations

import sys
from pathlib import Path


def _src_directory() -> Path:
    """Return the local source directory for checkout-based launches."""

    return Path(__file__).resolve().parent / "src"


def _ensure_local_src_on_path() -> None:
    """Make `python skating_aerial_alignment_gui.py` work without PYTHONPATH."""

    src_directory = _src_directory()
    if src_directory.exists():
        src_path = str(src_directory)
        if src_path not in sys.path:
            sys.path.insert(0, src_path)


def main() -> None:
    """Launch the interactive GUI."""

    _ensure_local_src_on_path()
    from skating_aerial_alignment.visualization import launch_app

    launch_app()


if __name__ == "__main__":
    main()
