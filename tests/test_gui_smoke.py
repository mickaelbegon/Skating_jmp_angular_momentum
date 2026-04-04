"""Smoke test for the matplotlib GUI initialization."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest

pytest.importorskip("biorbd")

from skating_aerial_alignment.visualization import SkatingAerialAlignmentApp  # noqa: E402


def test_gui_builds_without_display_side_effects() -> None:
    """The GUI can be instantiated under a non-interactive backend."""

    app = SkatingAerialAlignmentApp()
    try:
        assert len(app.sliders) == 7
        assert app.result.time.size > 0
        assert app.sliders["salto_rps"].valmin == 0.0
        assert app.sliders["salto_rps"].valmax == 0.25
        assert app.sliders["backward_velocity"].valmin == 0.0
        assert len(app.principal_moment_lines) == 3
        assert len(app.time_cursors) == 5
        assert app.ground_surface is not None
        assert app.playback_selector.value_selected == "100%"
        assert app.frames_per_animation_step >= 1
    finally:
        app.animation._draw_was_started = True
        plt.close(app.figure)


def test_face_mode_recomputes_display_markers_with_zero_twist() -> None:
    """The front-view mode redraws the avatar with the twist coordinate set to zero."""

    app = SkatingAerialAlignmentApp()
    try:
        app.stabilization_checkbox.set_active(1)
        displayed_markers, _ = app._display_kinematics(0)
        expected_q = app.result.q[0].copy()
        expected_q[5] = 0.0
        expected_markers = app.simulator.markers(expected_q)

        assert np.allclose(displayed_markers, expected_markers)
    finally:
        app.animation._draw_was_started = True
        plt.close(app.figure)
