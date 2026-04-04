"""Tests for the skater aerial-flight simulation helpers."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("biorbd")

from skating_aerial_alignment.simulation import (  # noqa: E402
    FlightSimulationParameters,
    PDControllerConfiguration,
    SkaterFlightSimulator,
)


def test_flight_time_and_takeoff_velocity_are_inverse_relations() -> None:
    """Ballistic flight time and takeoff velocity conversions are consistent."""

    flight_time = SkaterFlightSimulator.flight_time_from_takeoff_velocity(0.62)
    reconstructed_velocity = SkaterFlightSimulator.takeoff_velocity_from_flight_time(flight_time)

    assert reconstructed_velocity == pytest.approx(0.62)


def test_controller_torques_oppose_trunk_deviation() -> None:
    """The PD controller generates restoring trunk torques."""

    simulator = SkaterFlightSimulator()
    torques = simulator.controller_torques(
        q_trunk=np.array([0.10, -0.05, 0.02]),
        qdot_trunk=np.array([0.0, 0.0, 0.0]),
        configuration=PDControllerConfiguration(
            proportional_gains=(100.0, 100.0, 100.0),
            derivative_gains=(0.0, 0.0, 0.0),
            torque_limits=(500.0, 500.0, 500.0),
        ),
        enabled=True,
    )

    assert np.allclose(torques, np.array([-10.0, 5.0, -2.0]))


def test_passive_zero_momentum_keeps_the_rotational_state_constant() -> None:
    """Without angular momentum or trunk actuation, the rotational state stays at rest."""

    simulator = SkaterFlightSimulator()
    result = simulator.simulate(
        FlightSimulationParameters(
            angular_velocity_rps=(0.0, 0.0, 0.0),
            takeoff_vertical_velocity=0.50,
            sample_count=21,
            stabilize_trunk=False,
        )
    )

    assert np.allclose(result.q[:, 3:9], 0.0)
    assert np.allclose(result.qdot[:, 3:9], 0.0)
    assert np.allclose(result.tau[:, 6:9], 0.0)


def test_passive_simulation_preserves_angular_momentum_magnitude() -> None:
    """The zero-torque rotational dynamics conserves angular momentum during flight."""

    simulator = SkaterFlightSimulator()
    result = simulator.simulate(
        FlightSimulationParameters(
            angular_velocity_rps=(0.0, 0.0, 2.5),
            takeoff_vertical_velocity=0.55,
            sample_count=31,
            stabilize_trunk=False,
        )
    )

    magnitudes = np.linalg.norm(result.angular_momentum, axis=1)
    assert np.max(np.abs(magnitudes - magnitudes[0])) < 1e-8
