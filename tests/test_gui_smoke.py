"""Smoke test for the matplotlib GUI initialization."""

from __future__ import annotations

from types import SimpleNamespace

import matplotlib as mpl
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
        assert app.sliders["backward_velocity"].valmin == 1.0
        assert app.sliders["backward_velocity"].valmax == 5.0
        assert app.sliders["backward_velocity"].label.get_text() == "V_arr (m/s)"
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
        assert len(app.time_cursors) == 7
        assert app.ground_surface is not None
        assert app.precession_cone_line is not None
        assert app.playback_selector.value_selected == "100%"
        assert app.playback_menu_visible is False
        assert app.playback_menu_axis.get_visible() is False
        assert app.retune_pd_button.label.get_text() == "Retuner PD"
        assert app.speed_button.label.get_text() == "Vitesse 100%"
        assert len(app.control_section_titles) == 4
        assert app.control_section_titles[0].get_text() == "Moment cinetique global"
        assert len(app.stabilization_checkbox.get_status()) == 4
        assert app._inward_tilt_optimization_enabled() is False
        assert app._alignment_optimization_enabled() is False
        assert app.time_slider.valmin == 0.0
        assert app.time_slider.valmax == pytest.approx(app.result.flight_time)
        assert app.sliders["salto_rps"].ax.get_position().height < 0.009
        assert app.time_slider.ax.get_position().height < 0.009
        assert app.control_panels["momentum"].get_position().height > 0.19
        assert app.ax_rotation.get_position().x1 < 0.96
        assert app.ax_rotation.get_ylabel() == "Vrille (deg)"
        assert app.ax_rotation_salto.get_ylabel() == "Salto (deg)"
        assert app.ax_inertia_twist_speed.get_ylabel() == "omega_twist (deg/s)"
        assert app.twist_speed_line is not None
        assert app.ax_3d._axis_names[app.ax_3d._vertical_axis] == "z"
        assert mpl.rcParams["axes3d.mouserotationstyle"] == "azel"
        assert app.ax_inertia.get_xlabel() == "Temps (s)"
        assert all(label.get_visible() is False for label in app.ax_alignment.get_xticklabels())
        assert all(label.get_visible() is False for label in app.ax_rotation.get_xticklabels())
        assert app.frames_per_animation_step >= 1
        assert app.animation.event_source.interval == app.ANIMATION_TIMER_INTERVAL_MS
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


def test_animation_rewinds_to_the_start_when_playback_reaches_the_end() -> None:
    """Playback pauses and rewinds once the final frame has been displayed."""

    app = SkatingAerialAlignmentApp()
    try:
        app.frame_index = len(app.result.time) - 2
        app.frames_per_animation_step = 3
        app.is_paused = False
        app.pause_button.label.set_text("Pause")

        app._animate(0)

        assert app.frame_index == 0
        assert app.is_paused is True
        assert app.pause_button.label.get_text() == "Play"
        assert app.time_slider.val == pytest.approx(0.0)
    finally:
        app.animation._draw_was_started = True
        plt.close(app.figure)


def test_enabling_stabilization_does_not_auto_tune_pd_controller() -> None:
    """Turning on stabilization keeps the current gains until explicit retuning."""

    app = SkatingAerialAlignmentApp()
    try:
        initial_controller = app.parameters.controller

        app.stabilization_checkbox.set_active(0)

        assert app.parameters.stabilize_trunk is True
        assert app.optimization_result is None
        assert app.parameters.controller == initial_controller
    finally:
        app.animation._draw_was_started = True
        plt.close(app.figure)


def test_retune_pd_button_runs_explicit_pd_tuning(tmp_path) -> None:
    """The explicit retune button calibrates and applies the PD controller."""

    app = SkatingAerialAlignmentApp(pd_cache_path=tmp_path / "pd_tuning_cache.json")
    try:
        initial_controller = app.parameters.controller
        app.stabilization_checkbox.set_active(0)

        app._retune_pd_controller(None)

        assert app.parameters.stabilize_trunk is True
        assert app.optimization_result is not None
        assert app.parameters.controller == app.optimization_result.controller
        assert app.parameters.controller != initial_controller
    finally:
        app.animation._draw_was_started = True
        plt.close(app.figure)


def test_parameter_change_invalidates_previous_pd_tuning_result() -> None:
    """Changing the scenario clears the displayed PD optimization result."""

    app = SkatingAerialAlignmentApp()
    try:
        app.stabilization_checkbox.set_active(0)
        app._retune_pd_controller(None)

        assert app.optimization_result is not None

        app.sliders["twist_rps"].set_val(app.sliders["twist_rps"].val + 0.1)

        assert app.optimization_result is None
    finally:
        app.animation._draw_was_started = True
        plt.close(app.figure)


def test_pd_tuning_cache_is_reloaded_on_startup(tmp_path) -> None:
    """A tuned PD controller is persisted and reused when the GUI starts again."""

    cache_path = tmp_path / "pd_tuning_cache.json"

    first_app = SkatingAerialAlignmentApp(pd_cache_path=cache_path)
    try:
        first_app.stabilization_checkbox.set_active(0)
        first_app._retune_pd_controller(None)
        tuned_controller = first_app.parameters.controller

        assert cache_path.exists()
    finally:
        first_app.animation._draw_was_started = True
        plt.close(first_app.figure)

    second_app = SkatingAerialAlignmentApp(pd_cache_path=cache_path)
    try:
        assert second_app.parameters.controller == tuned_controller
        assert len(second_app._pd_tuning_cache) >= 1
    finally:
        second_app.animation._draw_was_started = True
        plt.close(second_app.figure)


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


def test_enabling_alignment_optimization_updates_the_slider_and_result() -> None:
    """Turning on alignment optimization applies the optimized tilt to the scenario."""

    app = SkatingAerialAlignmentApp()
    try:
        app.stabilization_checkbox.set_active(3)

        assert app._alignment_optimization_enabled() is True
        assert app.alignment_optimization_result is not None
        assert app.inward_tilt_optimization_result is None
        assert app.parameters.inward_tilt_deg == pytest.approx(
            app.alignment_optimization_result.inward_tilt_deg
        )
        assert app.sliders["inward_tilt"].val == pytest.approx(app.parameters.inward_tilt_deg)
    finally:
        app.animation._draw_was_started = True
        plt.close(app.figure)


def test_alignment_and_twist_optimization_modes_are_mutually_exclusive() -> None:
    """Selecting one inward-tilt optimization mode disables the other one."""

    app = SkatingAerialAlignmentApp()
    try:
        app.stabilization_checkbox.set_active(2)
        assert app._inward_tilt_optimization_enabled() is True
        assert app._alignment_optimization_enabled() is False

        app.stabilization_checkbox.set_active(3)

        assert app._alignment_optimization_enabled() is True
        assert app._inward_tilt_optimization_enabled() is False
    finally:
        app.animation._draw_was_started = True
        plt.close(app.figure)


def test_checkbox_click_tolerates_missing_frame_index_payload() -> None:
    """Malformed Matplotlib hit-test payloads on checkboxes are ignored safely."""

    app = SkatingAerialAlignmentApp()
    try:
        checkbox = app.stabilization_checkbox
        initial_status = checkbox.get_status()
        checkbox.ax.contains = lambda event: (True, {})
        checkbox._frames.contains = lambda event: (True, {})
        checkbox.labels = []

        event = SimpleNamespace(button=1, x=0.0, y=0.0)
        checkbox._clicked(event)

        assert checkbox.get_status() == initial_status
    finally:
        app.animation._draw_was_started = True
        plt.close(app.figure)


def test_radio_button_click_tolerates_missing_button_index_payload() -> None:
    """Malformed Matplotlib hit-test payloads on radio buttons are ignored safely."""

    app = SkatingAerialAlignmentApp()
    try:
        selector = app.playback_selector
        initial_label = selector.value_selected
        selector.ax.contains = lambda event: (True, {})
        selector._buttons.contains = lambda event: (True, {})
        selector.labels = []

        event = SimpleNamespace(button=1, x=0.0, y=0.0)
        selector._clicked(event)

        assert selector.value_selected == initial_label
    finally:
        app.animation._draw_was_started = True
        plt.close(app.figure)
