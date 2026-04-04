"""Tests for visualization helpers that do not require a GUI backend."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from skating_aerial_alignment.simulation import FlightSimulationParameters
from skating_aerial_alignment.visualization.app import format_status_text, skeleton_connections


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


def test_status_text_mentions_flight_time_velocity_and_controller_state() -> None:
    """The status text exposes the main physical settings shown in the GUI."""

    text = format_status_text(
        FlightSimulationParameters(
            angular_velocity_rps=(0.0, 0.0, 3.0),
            takeoff_vertical_velocity=0.62,
            stabilize_trunk=True,
        ),
        SimpleNamespace(flight_time=0.1264),
        _DummySimulator(),
    )

    assert "Temps de vol" in text
    assert "0.62 m/s" in text
    assert "PD actif" in text
