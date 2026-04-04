"""Forward simulation of the skater aerial phase in zero gravity."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import biorbd
import numpy as np
from scipy.integrate import solve_ivp

from skating_aerial_alignment.modeling import SkaterFlightBiomod

ROOT_DOF = 6
TRUNK_DOF = 3
ROTATION_STATE_DOF = ROOT_DOF - 3 + TRUNK_DOF
GRAVITY = 9.81


@dataclass(frozen=True)
class PDControllerConfiguration:
    """Configuration of the trunk stabilization controller."""

    proportional_gains: tuple[float, float, float] = (80.0, 80.0, 40.0)
    derivative_gains: tuple[float, float, float] = (12.0, 12.0, 8.0)
    torque_limits: tuple[float, float, float] = (250.0, 250.0, 150.0)


@dataclass(frozen=True)
class FlightSimulationParameters:
    """User-facing simulation inputs matching the GUI sliders."""

    angular_velocity_rps: tuple[float, float, float] = (0.0, 0.0, 3.0)
    takeoff_vertical_velocity: float = 0.60
    somersault_tilt_deg: float = 0.0
    inward_tilt_deg: float = 0.0
    initial_trunk_angles_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)
    initial_trunk_velocity_deg_s: tuple[float, float, float] = (0.0, 0.0, 0.0)
    stabilize_trunk: bool = False
    controller: PDControllerConfiguration = field(default_factory=PDControllerConfiguration)
    sample_count: int = 201


@dataclass(frozen=True)
class FlightSimulationResult:
    """Simulation output enriched with observables for plotting."""

    time: np.ndarray
    q: np.ndarray
    qdot: np.ndarray
    tau: np.ndarray
    angular_momentum: np.ndarray
    body_axis: np.ndarray
    body_axis_alignment_deg: np.ndarray
    markers: np.ndarray
    flight_time: float
    equivalent_angular_momentum: np.ndarray


class SkaterFlightSimulator:
    """Simulate the zero-gravity aerial phase of the reduced skater model."""

    def __init__(
        self,
        biomod_builder: SkaterFlightBiomod | None = None,
        *,
        model_path: str | Path | None = None,
    ) -> None:
        """Instantiate the biorbd model and cache marker indices."""

        self.biomod_builder = biomod_builder or SkaterFlightBiomod()
        self.model_path = self.biomod_builder.write(
            model_path or Path("artifacts") / "skater_flight.bioMod"
        )
        self.model = biorbd.Model(str(self.model_path))
        self.marker_names = [
            name.to_string() if hasattr(name, "to_string") else str(name)
            for name in self.model.markerNames()
        ]
        self.marker_index = {name: index for index, name in enumerate(self.marker_names)}

    @staticmethod
    def flight_time_from_takeoff_velocity(
        takeoff_vertical_velocity: float,
        *,
        gravity: float = GRAVITY,
    ) -> float:
        """Return the ballistic flight time associated with a takeoff speed."""

        if takeoff_vertical_velocity < 0.0:
            raise ValueError("The takeoff vertical velocity must be non-negative.")
        return 2.0 * takeoff_vertical_velocity / gravity if takeoff_vertical_velocity > 0.0 else 0.0

    @staticmethod
    def takeoff_velocity_from_flight_time(
        flight_time: float,
        *,
        gravity: float = GRAVITY,
    ) -> float:
        """Return the takeoff vertical velocity matching a desired flight time."""

        if flight_time < 0.0:
            raise ValueError("The flight time must be non-negative.")
        return 0.5 * gravity * flight_time

    @staticmethod
    def ballistic_height(time: np.ndarray, takeoff_vertical_velocity: float) -> np.ndarray:
        """Return the height offset used to visualize the ballistic flight."""

        time = np.asarray(time, dtype=float)
        return takeoff_vertical_velocity * time - 0.5 * GRAVITY * time**2

    @staticmethod
    def ballistic_velocity(time: np.ndarray, takeoff_vertical_velocity: float) -> np.ndarray:
        """Return the ballistic vertical velocity associated with the display trajectory."""

        time = np.asarray(time, dtype=float)
        return takeoff_vertical_velocity - GRAVITY * time

    def angular_momentum_from_rps(
        self,
        angular_velocity_rps: tuple[float, float, float],
    ) -> np.ndarray:
        """Convert rotations per second about the initial body axes into angular momentum."""

        angular_velocity = 2.0 * np.pi * np.asarray(angular_velocity_rps, dtype=float)
        return self.biomod_builder.inertia_tensor_body() @ angular_velocity

    def controller_torques(
        self,
        q_trunk: np.ndarray,
        qdot_trunk: np.ndarray,
        configuration: PDControllerConfiguration,
        *,
        enabled: bool,
    ) -> np.ndarray:
        """Return the generalized torques that stabilize the trunk toward zero."""

        if not enabled:
            return np.zeros(TRUNK_DOF, dtype=float)

        q_trunk = np.asarray(q_trunk, dtype=float)
        qdot_trunk = np.asarray(qdot_trunk, dtype=float)
        kp = np.asarray(configuration.proportional_gains, dtype=float)
        kd = np.asarray(configuration.derivative_gains, dtype=float)
        limits = np.asarray(configuration.torque_limits, dtype=float)
        raw_torques = -kp * q_trunk - kd * qdot_trunk
        return np.clip(raw_torques, -limits, limits)

    def markers(self, q: np.ndarray) -> np.ndarray:
        """Return the marker cloud of the current configuration."""

        q_biorbd = biorbd.GeneralizedCoordinates(np.asarray(q, dtype=float))
        return np.vstack([marker.to_array() for marker in self.model.markers(q_biorbd)])

    def body_frame(self, q: np.ndarray) -> np.ndarray:
        """Estimate the body-fixed orthonormal frame from pelvis, shoulders, and head markers."""

        marker_cloud = self.markers(q)
        pelvis_origin = marker_cloud[self.marker_index["pelvis_origin"]]
        head_top = marker_cloud[self.marker_index["head_top"]]
        shoulder_left = marker_cloud[self.marker_index["shoulder_left"]]
        shoulder_right = marker_cloud[self.marker_index["shoulder_right"]]

        longitudinal_axis = self._normalize(head_top - pelvis_origin)
        lateral_axis_raw = shoulder_left - shoulder_right
        lateral_axis = self._normalize(
            lateral_axis_raw - np.dot(lateral_axis_raw, longitudinal_axis) * longitudinal_axis
        )
        forward_axis = self._normalize(np.cross(longitudinal_axis, lateral_axis))
        return np.column_stack((lateral_axis, forward_axis, longitudinal_axis))

    def initial_generalized_coordinates(self, parameters: FlightSimulationParameters) -> np.ndarray:
        """Build the initial generalized coordinates from the requested takeoff posture."""

        q0 = np.zeros(self.model.nbQ(), dtype=float)
        q0[3] = np.deg2rad(parameters.somersault_tilt_deg)
        q0[4] = np.deg2rad(parameters.inward_tilt_deg)
        q0[6:9] = np.deg2rad(np.asarray(parameters.initial_trunk_angles_deg, dtype=float))
        return q0

    def initial_rotational_velocity(
        self,
        q0: np.ndarray,
        angular_velocity_rps: tuple[float, float, float],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Solve the root rotational velocity matching the requested initial angular momentum."""

        desired_angular_momentum_body = self.angular_momentum_from_rps(angular_velocity_rps)
        desired_angular_momentum_world = self.body_frame(q0) @ desired_angular_momentum_body

        mapping = np.zeros((3, 3), dtype=float)
        qdot = np.zeros(self.model.nbQdot(), dtype=float)
        for column, index in enumerate(range(3, 6)):
            qdot[:] = 0.0
            qdot[index] = 1.0
            mapping[:, column] = self.angular_momentum(q0, qdot)

        root_rotational_velocity = np.linalg.solve(mapping, desired_angular_momentum_world)
        qdot0 = np.zeros(self.model.nbQdot(), dtype=float)
        qdot0[3:6] = root_rotational_velocity
        return qdot0, desired_angular_momentum_world

    def full_initial_velocity(
        self,
        parameters: FlightSimulationParameters,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build the full generalized velocity vector from the user inputs."""

        q0 = self.initial_generalized_coordinates(parameters)
        qdot0, desired_angular_momentum_world = self.initial_rotational_velocity(
            q0,
            parameters.angular_velocity_rps,
        )
        qdot0[6:9] = np.deg2rad(np.asarray(parameters.initial_trunk_velocity_deg_s, dtype=float))
        return qdot0, desired_angular_momentum_world

    def angular_momentum(self, q: np.ndarray, qdot: np.ndarray) -> np.ndarray:
        """Evaluate the whole-body angular momentum about the center of mass."""

        return self.model.angularMomentum(
            biorbd.GeneralizedCoordinates(np.asarray(q, dtype=float)),
            biorbd.GeneralizedVelocity(np.asarray(qdot, dtype=float)),
            True,
        ).to_array()

    def simulate(self, parameters: FlightSimulationParameters) -> FlightSimulationResult:
        """Integrate the aerial dynamics and return plotting-ready observables."""

        flight_time = self.flight_time_from_takeoff_velocity(parameters.takeoff_vertical_velocity)
        sample_count = max(int(parameters.sample_count), 2)
        if flight_time == 0.0:
            time = np.array([0.0], dtype=float)
            q0 = self.initial_generalized_coordinates(parameters)
            qdot0, desired_angular_momentum_world = self.full_initial_velocity(parameters)
            tau = np.zeros((1, self.model.nbGeneralizedTorque()), dtype=float)
            q = q0[None, :]
            qdot = qdot0[None, :]
        else:
            time = np.linspace(0.0, flight_time, sample_count, dtype=float)
            q0 = self.initial_generalized_coordinates(parameters)
            qdot0, desired_angular_momentum_world = self.full_initial_velocity(parameters)
            state0 = np.concatenate((q0[3:9], qdot0[3:9]))
            solution = solve_ivp(
                fun=lambda current_time, state: self._dynamics(current_time, state, parameters),
                t_span=(0.0, flight_time),
                y0=state0,
                t_eval=time,
                method="RK45",
                rtol=1e-8,
                atol=1e-8,
            )
            if not solution.success:
                raise RuntimeError(f"Flight integration failed: {solution.message}")
            q, qdot, tau = self._sample_solution(
                np.asarray(solution.t, dtype=float),
                np.asarray(solution.y, dtype=float),
                parameters,
            )

        q[:, 2] = self.ballistic_height(time, parameters.takeoff_vertical_velocity)
        qdot[:, 2] = self.ballistic_velocity(time, parameters.takeoff_vertical_velocity)

        markers = np.zeros((time.size, self.model.nbMarkers(), 3), dtype=float)
        angular_momentum = np.zeros((time.size, 3), dtype=float)
        body_axis = np.zeros((time.size, 3), dtype=float)
        alignment = np.zeros(time.size, dtype=float)

        for frame_index, (q_frame, qdot_frame) in enumerate(zip(q, qdot)):
            markers[frame_index] = self.markers(q_frame)
            angular_momentum[frame_index] = self.angular_momentum(q_frame, qdot_frame)
            body_axis[frame_index] = self.body_frame(q_frame)[:, 2]
            alignment[frame_index] = self._angle_deg(
                angular_momentum[frame_index],
                body_axis[frame_index],
            )

        return FlightSimulationResult(
            time=time,
            q=q,
            qdot=qdot,
            tau=tau,
            angular_momentum=angular_momentum,
            body_axis=body_axis,
            body_axis_alignment_deg=alignment,
            markers=markers,
            flight_time=flight_time,
            equivalent_angular_momentum=desired_angular_momentum_world,
        )

    def _dynamics(
        self,
        _time: float,
        state: np.ndarray,
        parameters: FlightSimulationParameters,
    ) -> np.ndarray:
        """Return the reduced rotational state derivative."""

        q = np.zeros(self.model.nbQ(), dtype=float)
        qdot = np.zeros(self.model.nbQdot(), dtype=float)
        q[3:9] = state[:ROTATION_STATE_DOF]
        qdot[3:9] = state[ROTATION_STATE_DOF:]

        tau = np.zeros(self.model.nbGeneralizedTorque(), dtype=float)
        tau[6:9] = self.controller_torques(
            q[6:9],
            qdot[6:9],
            parameters.controller,
            enabled=parameters.stabilize_trunk,
        )

        qddot = self.model.ForwardDynamics(
            biorbd.GeneralizedCoordinates(q),
            biorbd.GeneralizedVelocity(qdot),
            biorbd.GeneralizedTorque(tau),
        ).to_array()
        return np.concatenate((qdot[3:9], qddot[3:9]))

    def _sample_solution(
        self,
        time: np.ndarray,
        state_history: np.ndarray,
        parameters: FlightSimulationParameters,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Reconstruct full generalized trajectories from the reduced solution."""

        q = np.zeros((time.size, self.model.nbQ()), dtype=float)
        qdot = np.zeros((time.size, self.model.nbQdot()), dtype=float)
        tau = np.zeros((time.size, self.model.nbGeneralizedTorque()), dtype=float)

        for frame_index in range(time.size):
            q[frame_index, 3:9] = state_history[:ROTATION_STATE_DOF, frame_index]
            qdot[frame_index, 3:9] = state_history[ROTATION_STATE_DOF:, frame_index]
            tau[frame_index, 6:9] = self.controller_torques(
                q[frame_index, 6:9],
                qdot[frame_index, 6:9],
                parameters.controller,
                enabled=parameters.stabilize_trunk,
            )

        return q, qdot, tau

    @staticmethod
    def _angle_deg(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
        """Return the angle in degrees between two vectors."""

        norm_product = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
        if norm_product == 0.0:
            return 0.0
        cosine = np.clip(np.dot(vector_a, vector_b) / norm_product, -1.0, 1.0)
        return float(np.rad2deg(np.arccos(cosine)))

    @staticmethod
    def _normalize(vector: np.ndarray) -> np.ndarray:
        """Return the normalized version of a non-zero vector."""

        norm = np.linalg.norm(vector)
        if norm == 0.0:
            raise ValueError("Cannot normalize a zero vector.")
        return np.asarray(vector, dtype=float) / norm
