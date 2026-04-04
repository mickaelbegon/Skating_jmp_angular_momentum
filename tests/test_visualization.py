"""Tests for visualization helpers that do not require a GUI backend."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from skating_aerial_alignment.simulation import (
    FlightSimulationParameters,
    PDControllerConfiguration,
)
from skating_aerial_alignment.visualization.app import (
    format_inertia_and_controller_text,
    format_status_text,
    skeleton_connections,
)


class _DummySimulator:
    """Minimal simulator stub exposing the formatting dependency."""

    @staticmethod
    def angular_momentum_from_rps(_angular_velocity_rps):
        """Return a deterministic pseudo angular momentum for the formatter test."""

        return np.array([1.0, 2.0, 3.0])


def test_skeleton_connections_cover_head_upper_limbs_and_lower_limbs() -> None:
    """The 3D skeleton contains the expected anatomical links."""

    connections = skeleton_connections()

    assert ("thorax_top", "head_top") in connections
    assert ("shoulder_left", "elbow_left") in connections
    assert ("hip_right", "knee_right") in connections
    assert ("hip_left", "pelvis_thorax_joint_center") in connections
    assert ("shoulder_right", "pelvis_thorax_joint_center") in connections


def test_status_text_mentions_flight_time_velocity_and_controller_state() -> None:
    """The status text exposes the main physical settings shown in the GUI."""

    text = format_status_text(
        FlightSimulationParameters(
            angular_velocity_rps=(0.0, 0.0, 3.0),
            takeoff_vertical_velocity=0.62,
            stabilize_trunk=True,
        ),
        SimpleNamespace(flight_time=0.1264, initial_body_axis_alignment_deg=18.5),
        _DummySimulator(),
    )

    assert "Temps de vol" in text
    assert "0.62 m/s" in text
    assert "Angle initial" in text
    assert "PD actif" in text


def test_inertia_text_mentions_inertias_and_controller_gains() -> None:
    """The secondary status line exposes inertias and current controller gains."""

    simulator = SimpleNamespace(
        biomod_builder=SimpleNamespace(principal_moments=lambda: np.array([1.0, 2.0, 3.0]))
    )
    parameters = FlightSimulationParameters(
        stabilize_trunk=True,
        controller=PDControllerConfiguration(
            proportional_gains=(10.0, 20.0, 30.0),
            derivative_gains=(1.0, 2.0, 3.0),
        ),
    )

    text = format_inertia_and_controller_text(parameters, simulator, optimization_result=None)

    assert "I = [1.00, 2.00, 3.00]" in text
    assert "Kp = [10.0, 20.0, 30.0]" in text
    assert "Kd = [1.0, 2.0, 3.0]" in text
