"""Smoke test for the matplotlib GUI initialization."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pytest

pytest.importorskip("biorbd")

from skating_aerial_alignment.visualization import SkatingAerialAlignmentApp  # noqa: E402


def test_gui_builds_without_display_side_effects() -> None:
    """The GUI can be instantiated under a non-interactive backend."""

    app = SkatingAerialAlignmentApp()
    try:
        assert len(app.sliders) == 6
        assert app.result.time.size > 0
    finally:
        app.animation._draw_was_started = True
        plt.close(app.figure)
