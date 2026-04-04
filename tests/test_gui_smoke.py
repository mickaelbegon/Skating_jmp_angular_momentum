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
        assert app.sliders["backward_travel"].valmin == 1.0
        assert app.sliders["backward_travel"].valmax == 5.0
        assert app.sliders["flight_time"].valmin == 0.4
        assert app.sliders["flight_time"].valmax == 0.8
        assert app.twist_inertia_line is not None
        assert app.com_trajectory_line is not None
        assert app.com_point is not None
        assert app.com_trajectory_line.get_alpha() == pytest.approx(0.28)
        x_data, y_data, z_data = app.com_trajectory_line.get_data_3d()
        assert len(x_data) == app.result.time.size
        assert len(y_data) == app.result.time.size
        assert len(z_data) == app.result.time.size
        assert len(app.time_cursors) == 5
        assert app.ground_surface is not None
        assert app.precession_cone_line is not None
        assert app.playback_selector.value_selected == "100%"
        assert app.playback_menu_visible is False
        assert app.playback_menu_axis.get_visible() is False
        assert app.speed_button.label.get_text() == "Vitesse 100%"
        assert len(app.control_section_titles) == 4
        assert app.control_section_titles[0].get_text() == "Moment cinetique global"
        assert len(app.stabilization_checkbox.get_status()) == 3
        assert app._inward_tilt_optimization_enabled() is False
        assert app.time_slider.valmin == 0.0
        assert app.time_slider.valmax == pytest.approx(app.result.flight_time)
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


def test_time_slider_moves_to_the_requested_frame_and_pauses_animation() -> None:
    """Scrubbing in time updates the displayed frame and pauses the animation."""

    app = SkatingAerialAlignmentApp()
    try:
        target_frame = len(app.result.time) // 2
        target_time = float(app.result.time[target_frame])

        app.time_slider.set_val(target_time)

        assert app.is_paused is True
        assert app.pause_button.label.get_text() == "Play"
        assert app.frame_index == target_frame
        assert app.time_slider.val == pytest.approx(target_time)
    finally:
        app.animation._draw_was_started = True
        plt.close(app.figure)


def test_enabling_stabilization_runs_automatic_pd_tuning() -> None:
    """Turning on stabilization automatically tunes and applies the PD gains."""

    app = SkatingAerialAlignmentApp()
    try:
        initial_controller = app.parameters.controller

        app.stabilization_checkbox.set_active(0)

        assert app.parameters.stabilize_trunk is True
        assert app.optimization_result is not None
        assert app.parameters.controller == app.optimization_result.controller
        assert app.parameters.controller != initial_controller
    finally:
        app.animation._draw_was_started = True
        plt.close(app.figure)


def test_enabling_inward_tilt_optimization_updates_the_slider_and_result() -> None:
    """Turning on inward-tilt optimization applies the optimized tilt to the scenario."""

    app = SkatingAerialAlignmentApp()
    try:
        app.stabilization_checkbox.set_active(2)

        assert app._inward_tilt_optimization_enabled() is True
        assert app.inward_tilt_optimization_result is not None
        assert app.parameters.inward_tilt_deg == pytest.approx(
            app.inward_tilt_optimization_result.inward_tilt_deg
        )
        assert app.sliders["inward_tilt"].val == pytest.approx(app.parameters.inward_tilt_deg)
    finally:
        app.animation._draw_was_started = True
        plt.close(app.figure)
