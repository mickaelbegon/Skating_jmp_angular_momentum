"""Tests for the skater aerial-flight simulation helpers."""

from __future__ import annotations

from dataclasses import replace

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


def test_default_takeoff_velocity_matches_a_half_second_flight_time() -> None:
    """The default vertical velocity corresponds to a 0.5 s ballistic flight."""

    parameters = FlightSimulationParameters()
    flight_time = SkaterFlightSimulator.flight_time_from_takeoff_velocity(
        parameters.takeoff_vertical_velocity
    )

    assert flight_time == pytest.approx(0.5)


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


def test_angular_momentum_components_are_defined_in_the_global_frame() -> None:
    """The requested momentum components are interpreted directly in the global frame."""

    simulator = SkaterFlightSimulator()
    q0 = simulator.initial_generalized_coordinates(
        FlightSimulationParameters(
            somersault_tilt_deg=10.0,
            inward_tilt_deg=8.0,
        )
    )
    requested_rps = (0.2, -0.1, 0.5)

    angular_momentum = simulator.angular_momentum_from_rps(requested_rps, q0)
    reconstructed = np.diag(simulator.whole_body_inertia_tensor(q0)) * (
        2.0 * np.pi * np.asarray(requested_rps, dtype=float)
    )

    assert np.allclose(angular_momentum, reconstructed)


def test_single_axis_momentum_request_stays_on_that_global_axis() -> None:
    """Requesting only one global momentum component does not create cross components."""

    simulator = SkaterFlightSimulator()
    q0 = simulator.initial_generalized_coordinates(
        FlightSimulationParameters(
            somersault_tilt_deg=12.0,
            inward_tilt_deg=9.0,
        )
    )

    angular_momentum = simulator.angular_momentum_from_rps((0.0, 0.0, 3.0), q0)

    assert angular_momentum[:2] == pytest.approx(np.zeros(2), abs=1e-12)
    assert angular_momentum[2] > 0.0


def test_backward_horizontal_velocity_sets_the_center_of_mass_speed() -> None:
    """The backward-speed input prescribes the global CoM motion rather than the root motion."""

    simulator = SkaterFlightSimulator()
    result = simulator.simulate(
        FlightSimulationParameters(
            angular_velocity_rps=(0.2, 0.1, 1.5),
            takeoff_vertical_velocity=0.50,
            backward_horizontal_velocity=1.2,
            sample_count=11,
        )
    )

    com_shift = result.center_of_mass[:, 1] - result.center_of_mass[0, 1]
    center_of_mass_velocity = np.vstack(
        [
            simulator.center_of_mass_velocity(q_frame, qdot_frame)
            for q_frame, qdot_frame in zip(result.q, result.qdot)
        ]
    )

    assert np.allclose(com_shift, -1.2 * result.time, atol=1e-8)
    assert np.allclose(center_of_mass_velocity[:, 1], -1.2, atol=1e-8)


def test_backward_center_of_mass_velocity_does_not_change_angular_momentum() -> None:
    """Changing the prescribed CoM speed leaves the rotational angular momentum unchanged."""

    simulator = SkaterFlightSimulator()
    base_parameters = FlightSimulationParameters(
        angular_velocity_rps=(0.2, -0.1, 1.8),
        takeoff_vertical_velocity=0.50,
        somersault_tilt_deg=10.0,
        inward_tilt_deg=6.0,
        sample_count=21,
    )

    stationary_result = simulator.simulate(base_parameters)
    moving_result = simulator.simulate(replace(base_parameters, backward_horizontal_velocity=1.7))

    assert np.allclose(
        moving_result.angular_momentum, stationary_result.angular_momentum, atol=1e-8
    )


def test_backward_center_of_mass_velocity_does_not_change_initial_alignment() -> None:
    """Changing horizontal CoM speed does not alter the initial H/body-axis angle."""

    simulator = SkaterFlightSimulator()
    base_parameters = FlightSimulationParameters(
        angular_velocity_rps=(0.0, 0.0, 3.0),
        takeoff_vertical_velocity=0.50,
        somersault_tilt_deg=0.0,
        inward_tilt_deg=10.0,
        sample_count=11,
    )

    stationary_result = simulator.simulate(base_parameters)
    moving_result = simulator.simulate(replace(base_parameters, backward_horizontal_velocity=2.0))

    assert moving_result.initial_body_axis_alignment_deg == pytest.approx(
        stationary_result.initial_body_axis_alignment_deg
    )


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


def test_takeoff_pose_starts_with_lowest_marker_on_the_ground() -> None:
    """The initialized model is grounded so the first frame matches the takeoff surface."""

    simulator = SkaterFlightSimulator()
    result = simulator.simulate(
        FlightSimulationParameters(
            angular_velocity_rps=(0.0, 0.0, 0.0),
            takeoff_vertical_velocity=0.0,
            somersault_tilt_deg=10.0,
            inward_tilt_deg=5.0,
        )
    )

    assert np.min(result.markers[0, :, 2]) == pytest.approx(0.0)


def test_inward_tilt_is_applied_with_the_new_negative_sign_convention() -> None:
    """A positive inward-tilt input maps to a negative generalized coordinate."""

    simulator = SkaterFlightSimulator()
    q0 = simulator.initial_generalized_coordinates(FlightSimulationParameters(inward_tilt_deg=12.0))

    assert q0[4] == pytest.approx(-np.deg2rad(12.0))


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
    assert np.max(np.abs(magnitudes - magnitudes[0])) < 5e-8


def test_initial_alignment_angle_is_exposed_explicitly() -> None:
    """The initial angle is stored explicitly and matches the first alignment sample."""

    simulator = SkaterFlightSimulator()
    result = simulator.simulate(
        FlightSimulationParameters(
            angular_velocity_rps=(0.5, 0.25, 2.0),
            takeoff_vertical_velocity=0.50,
            somersault_tilt_deg=8.0,
            inward_tilt_deg=6.0,
            sample_count=31,
        )
    )

    assert result.initial_body_axis_alignment_deg == pytest.approx(
        result.body_axis_alignment_deg[0]
    )


def test_twist_inertia_proxy_is_available_during_simulation() -> None:
    """The simulation exposes the apparent twist inertia proxy `||H|| / |omega_vrille|`."""

    simulator = SkaterFlightSimulator()
    result = simulator.simulate(
        FlightSimulationParameters(
            angular_velocity_rps=(0.0, 0.1, 1.5),
            takeoff_vertical_velocity=0.50,
            sample_count=21,
        )
    )

    finite_mask = np.isfinite(result.twist_inertia_proxy)
    expected_ratio = np.linalg.norm(result.angular_momentum[finite_mask], axis=1) / np.abs(
        result.twist_rotation_speed[finite_mask]
    )

    assert result.twist_inertia_proxy.shape == (21,)
    assert np.any(finite_mask)
    assert np.all(result.twist_inertia_proxy[finite_mask] > 0.0)
    assert np.allclose(result.twist_inertia_proxy[finite_mask], expected_ratio)


def test_twist_rotation_speed_tracks_longitudinal_body_spin_not_euler_rate() -> None:
    """The reported twist speed is the spin about the body axis, not raw `qdot[5]`."""

    simulator = SkaterFlightSimulator()
    result = simulator.simulate(
        FlightSimulationParameters(
            angular_velocity_rps=(0.1, 0.4, 3.0),
            takeoff_vertical_velocity=0.50,
            somersault_tilt_deg=10.0,
            inward_tilt_deg=8.0,
            sample_count=21,
        )
    )

    reconstructed_speed = np.array(
        [
            simulator.longitudinal_twist_rate(q_frame, qdot_frame)
            for q_frame, qdot_frame in zip(result.q, result.qdot)
        ]
    )

    assert np.allclose(result.twist_rotation_speed, reconstructed_speed)
    assert not np.allclose(result.twist_rotation_speed, result.qdot[:, 5])
    assert result.twist_angle.shape == (21,)


def test_center_of_mass_trajectory_is_exposed_for_animation() -> None:
    """The simulation returns a center-of-mass trajectory with one point per frame."""

    simulator = SkaterFlightSimulator()
    result = simulator.simulate(
        FlightSimulationParameters(
            angular_velocity_rps=(0.0, 0.0, 1.5),
            takeoff_vertical_velocity=0.55,
            sample_count=21,
        )
    )

    assert result.center_of_mass.shape == (21, 3)
    assert result.center_of_mass[0, 2] > 0.0


def test_simulation_stops_at_first_descending_ground_contact() -> None:
    """The flight trajectory ends when the lowest marker returns to the ground."""

    simulator = SkaterFlightSimulator()
    result = simulator.simulate(
        FlightSimulationParameters(
            angular_velocity_rps=(0.0, 0.0, 1.5),
            takeoff_vertical_velocity=0.60,
            sample_count=61,
        )
    )

    assert result.flight_time > 0.0
    assert np.min(result.markers[-1, :, 2]) == pytest.approx(0.0, abs=1e-6)
    assert np.min(result.markers[-2, :, 2]) > 0.0


def test_pd_optimization_does_not_worsen_the_starting_controller() -> None:
    """The sub-optimal PD search returns a controller no worse than the initial guess."""

    simulator = SkaterFlightSimulator()
    parameters = FlightSimulationParameters(
        angular_velocity_rps=(0.0, 0.0, 1.0),
        takeoff_vertical_velocity=0.50,
        initial_trunk_angles_deg=(12.0, -8.0, 6.0),
        stabilize_trunk=True,
        sample_count=41,
    )
    initial_result = simulator.simulate(parameters)
    initial_objective = simulator.trunk_tracking_objective(initial_result)

    optimization = simulator.tune_trunk_controller(
        parameters,
        max_iterations=6,
        optimization_sample_count=31,
    )

    assert optimization.objective_value <= initial_objective + 1e-8
    assert all(gain >= 0.0 for gain in optimization.controller.proportional_gains)
    assert all(gain >= 0.0 for gain in optimization.controller.derivative_gains)


def test_inward_tilt_optimization_does_not_reduce_the_generated_twist() -> None:
    """The inward-tilt search returns a solution no worse than the baseline tilt."""

    simulator = SkaterFlightSimulator()
    parameters = FlightSimulationParameters(
        angular_velocity_rps=(0.1, 0.2, 2.0),
        takeoff_vertical_velocity=0.50,
        somersault_tilt_deg=8.0,
        inward_tilt_deg=0.0,
        sample_count=41,
    )
    baseline_result = simulator.simulate(parameters)
    baseline_twist_turns = simulator.twist_accumulation_turns(baseline_result)

    optimization = simulator.optimize_inward_tilt_for_twist(
        parameters,
        max_iterations=8,
        optimization_sample_count=31,
    )

    assert -30.0 <= optimization.inward_tilt_deg <= 30.0
    assert optimization.twist_turns >= baseline_twist_turns - 1e-8


def test_alignment_optimization_does_not_worsen_mean_alignment() -> None:
    """The alignment search returns a mean body-axis alignment no worse than baseline."""

    simulator = SkaterFlightSimulator()
    parameters = FlightSimulationParameters(
        angular_velocity_rps=(0.1, 0.2, 2.0),
        takeoff_vertical_velocity=0.50,
        somersault_tilt_deg=8.0,
        inward_tilt_deg=0.0,
        sample_count=41,
    )
    baseline_result = simulator.simulate(parameters)
    baseline_mean_alignment = simulator.mean_body_axis_alignment_deg(baseline_result)

    optimization = simulator.optimize_inward_tilt_for_alignment(
        parameters,
        max_iterations=8,
        optimization_sample_count=31,
    )

    assert -30.0 <= optimization.inward_tilt_deg <= 30.0
    assert optimization.mean_alignment_deg <= baseline_mean_alignment + 1e-8
