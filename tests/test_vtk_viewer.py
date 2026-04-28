"""Smoke tests for the dedicated VTK viewer."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("biorbd")
pytest.importorskip("vtk")

from skating_aerial_alignment.simulation import (  # noqa: E402
    FlightSimulationParameters,
    SkaterFlightSimulator,
)
from skating_aerial_alignment.visualization import VtkAvatarScene, VtkSkaterAnimator  # noqa: E402


def test_vtk_avatar_scene_builds_expected_actor_groups() -> None:
    """The VTK scene creates the expected avatar and guide actors."""

    simulator = SkaterFlightSimulator()
    result = simulator.simulate(FlightSimulationParameters())

    scene = VtkAvatarScene(simulator, result)

    assert len(scene.segment_actors) >= 18
    assert len(scene.ellipsoid_actors) == 7
    assert scene.body_axis_source is not None
    assert scene.sigma_source is not None
    assert scene.com_path_points is not None
    assert scene.precession_points is not None
    assert scene.com_sphere_actor is not None


def test_vtk_avatar_scene_updates_sigma_line_and_face_view() -> None:
    """The VTK scene updates line geometry and supports the face-view override."""

    simulator = SkaterFlightSimulator()
    parameters = FlightSimulationParameters(angular_velocity_rps=(0.0, 0.4, 2.5))
    result = simulator.simulate(parameters)
    scene = VtkAvatarScene(simulator, result, face_view=True)

    frame_index = min(10, result.time.size - 1)
    scene.update_frame(frame_index)

    sigma_end = np.array(scene.sigma_source.GetPoint2())
    sigma_start = np.array(scene.sigma_source.GetPoint1())
    assert np.linalg.norm(sigma_end - sigma_start) > 0.0

    display_markers, _ = scene.marker_cloud_for_frame(frame_index)
    display_q = result.q[frame_index].copy()
    display_q[5] = 0.0
    expected_markers = simulator.markers(display_q)
    assert np.allclose(display_markers, expected_markers)


def test_vtk_animator_updates_playback_controls_offscreen() -> None:
    """The VTK animator supports headless stepping and playback state changes."""

    animator = VtkSkaterAnimator(offscreen=True, render_on_update=False)

    animator.set_playback_speed(0.5)
    animator.step_to(5)
    animator.toggle_pause()

    assert animator.playback_speed == pytest.approx(0.5)
    assert animator.scene.frame_index == 5
    assert animator.is_paused is True
    assert animator.frame_step() > 0.0
