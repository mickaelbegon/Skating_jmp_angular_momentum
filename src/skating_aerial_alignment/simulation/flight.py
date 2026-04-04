"""Forward simulation of the skater aerial phase in zero gravity."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

import biorbd
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize, minimize_scalar

from skating_aerial_alignment.modeling import SkaterFlightBiomod

ROOT_DOF = 6
TRUNK_DOF = 3
ROTATION_STATE_DOF = ROOT_DOF - 3 + TRUNK_DOF
GRAVITY = 9.81
DEFAULT_FLIGHT_TIME = 0.5
DEFAULT_TAKEOFF_VERTICAL_VELOCITY = 0.5 * GRAVITY * DEFAULT_FLIGHT_TIME


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
    takeoff_vertical_velocity: float = DEFAULT_TAKEOFF_VERTICAL_VELOCITY
    backward_horizontal_velocity: float = 0.0
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
    initial_body_axis_alignment_deg: float
    twist_rotation_speed: np.ndarray
    twist_inertia_proxy: np.ndarray
    markers: np.ndarray
    center_of_mass: np.ndarray
    flight_time: float
    equivalent_angular_momentum: np.ndarray


@dataclass(frozen=True)
class PDOptimizationResult:
    """Result of the sub-optimal PD gain calibration."""

    controller: PDControllerConfiguration
    objective_value: float
    iterations: int
    evaluations: int
    success: bool
    message: str


@dataclass(frozen=True)
class InwardTiltOptimizationResult:
    """Result of the inward-tilt search that maximizes the produced twist."""

    inward_tilt_deg: float
    twist_turns: float
    evaluations: int
    success: bool
    message: str


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

    @staticmethod
    def backward_displacement(time: np.ndarray, backward_horizontal_velocity: float) -> np.ndarray:
        """Return the global backward displacement along the anteroposterior axis."""

        time = np.asarray(time, dtype=float)
        return -backward_horizontal_velocity * time

    def angular_momentum_from_rps(
        self,
        angular_velocity_rps: tuple[float, float, float],
        q: np.ndarray | None = None,
    ) -> np.ndarray:
        """Convert rotations per second about the global axes into angular momentum."""

        angular_velocity = 2.0 * np.pi * np.asarray(angular_velocity_rps, dtype=float)
        configuration = (
            np.asarray(q, dtype=float) if q is not None else np.zeros(self.model.nbQ(), dtype=float)
        )
        return self.whole_body_inertia_tensor(configuration) @ angular_velocity

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

    def center_of_mass(self, q: np.ndarray) -> np.ndarray:
        """Return the whole-body center of mass in the global frame."""

        q_biorbd = biorbd.GeneralizedCoordinates(np.asarray(q, dtype=float))
        return self.model.CoM(q_biorbd).to_array()

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
        q0[4] = -np.deg2rad(parameters.inward_tilt_deg)
        q0[6:9] = np.deg2rad(np.asarray(parameters.initial_trunk_angles_deg, dtype=float))
        q0[2] = self.initial_root_height(q0)
        return q0

    def initial_root_height(self, q: np.ndarray) -> float:
        """Return the root translation that places the lowest marker on the ground."""

        grounded_q = np.asarray(q, dtype=float).copy()
        grounded_q[2] = 0.0
        marker_positions = self.markers(grounded_q)
        return float(-np.min(marker_positions[:, 2]))

    def initial_rotational_velocity(
        self,
        q0: np.ndarray,
        angular_velocity_rps: tuple[float, float, float],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Solve the root rotational velocity matching the requested global angular momentum."""

        desired_angular_momentum_world = self.angular_momentum_from_rps(angular_velocity_rps, q0)

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

    def whole_body_inertia_tensor(self, q: np.ndarray) -> np.ndarray:
        """Return the whole-body inertia tensor about the global center of mass."""

        q_biorbd = biorbd.GeneralizedCoordinates(np.asarray(q, dtype=float))
        total_mass = 0.0
        segment_masses = []
        segment_com_positions = []
        segment_rotations = []
        segment_local_inertias = []

        for segment_index in range(self.model.nbSegment()):
            segment = self.model.segment(segment_index)
            characteristics = segment.characteristics()
            mass = float(characteristics.mass())
            rotation_translation = self.model.globalJCS(q_biorbd, segment_index).to_array()
            rotation = rotation_translation[:3, :3]
            translation = rotation_translation[:3, 3]
            local_com = characteristics.CoM().to_array()
            segment_com = rotation @ local_com + translation

            total_mass += mass
            segment_masses.append(mass)
            segment_com_positions.append(segment_com)
            segment_rotations.append(rotation)
            segment_local_inertias.append(characteristics.inertia().to_array())

        whole_body_com = (
            sum(mass * com for mass, com in zip(segment_masses, segment_com_positions)) / total_mass
        )
        inertia_tensor = np.zeros((3, 3), dtype=float)
        for mass, segment_com, rotation, local_inertia in zip(
            segment_masses,
            segment_com_positions,
            segment_rotations,
            segment_local_inertias,
        ):
            displacement = segment_com - whole_body_com
            segment_inertia_world = rotation @ local_inertia @ rotation.T
            inertia_tensor += segment_inertia_world + mass * (
                np.dot(displacement, displacement) * np.eye(3)
                - np.outer(displacement, displacement)
            )
        return inertia_tensor

    def simulate(self, parameters: FlightSimulationParameters) -> FlightSimulationResult:
        """Integrate the aerial dynamics and return plotting-ready observables."""

        estimated_flight_time = self.flight_time_from_takeoff_velocity(
            parameters.takeoff_vertical_velocity
        )
        sample_count = max(int(parameters.sample_count), 2)
        q0 = self.initial_generalized_coordinates(parameters)
        qdot0, desired_angular_momentum_world = self.full_initial_velocity(parameters)
        initial_root_height = float(q0[2])
        if estimated_flight_time == 0.0:
            time = np.array([0.0], dtype=float)
            tau = np.zeros((1, self.model.nbGeneralizedTorque()), dtype=float)
            q = q0[None, :]
            qdot = qdot0[None, :]
            flight_time = 0.0
        else:
            state0 = np.concatenate((q0[3:9], qdot0[3:9]))
            integration_horizon = max(estimated_flight_time * 1.5, estimated_flight_time + 0.25)

            def ground_event(current_time: float, state: np.ndarray) -> float:
                return self._ground_clearance(
                    current_time,
                    state,
                    parameters.takeoff_vertical_velocity,
                    initial_root_height,
                )

            ground_event.terminal = True
            ground_event.direction = -1.0
            solution = solve_ivp(
                fun=lambda current_time, state: self._dynamics(current_time, state, parameters),
                t_span=(0.0, integration_horizon),
                y0=state0,
                method="RK45",
                dense_output=True,
                events=ground_event,
                rtol=1e-8,
                atol=1e-8,
            )
            if not solution.success:
                raise RuntimeError(f"Flight integration failed: {solution.message}")
            if solution.t_events[0].size > 0:
                flight_time = float(solution.t_events[0][0])
            else:
                flight_time = float(solution.t[-1])
            time = np.linspace(0.0, flight_time, sample_count, dtype=float)
            q, qdot, tau = self._sample_solution(
                time,
                np.asarray(solution.sol(time), dtype=float),
                parameters,
            )

        q[:, 2] = initial_root_height + self.ballistic_height(
            time,
            parameters.takeoff_vertical_velocity,
        )
        q[:, 1] = self.backward_displacement(time, parameters.backward_horizontal_velocity)
        qdot[:, 2] = self.ballistic_velocity(time, parameters.takeoff_vertical_velocity)
        qdot[:, 1] = -parameters.backward_horizontal_velocity

        markers = np.zeros((time.size, self.model.nbMarkers(), 3), dtype=float)
        center_of_mass = np.zeros((time.size, 3), dtype=float)
        angular_momentum = np.zeros((time.size, 3), dtype=float)
        body_axis = np.zeros((time.size, 3), dtype=float)
        alignment = np.zeros(time.size, dtype=float)

        for frame_index, (q_frame, qdot_frame) in enumerate(zip(q, qdot)):
            markers[frame_index] = self.markers(q_frame)
            center_of_mass[frame_index] = self.center_of_mass(q_frame)
            angular_momentum[frame_index] = self.angular_momentum(q_frame, qdot_frame)
            body_axis[frame_index] = self.body_frame(q_frame)[:, 2]
            alignment[frame_index] = self._angle_deg(
                angular_momentum[frame_index],
                body_axis[frame_index],
            )
        twist_rotation_speed = np.asarray(qdot[:, 5], dtype=float)
        twist_inertia_proxy = np.divide(
            np.linalg.norm(angular_momentum, axis=1),
            np.abs(twist_rotation_speed),
            out=np.full(time.shape, np.nan, dtype=float),
            where=np.abs(twist_rotation_speed) > 1e-8,
        )

        return FlightSimulationResult(
            time=time,
            q=q,
            qdot=qdot,
            tau=tau,
            angular_momentum=angular_momentum,
            body_axis=body_axis,
            body_axis_alignment_deg=alignment,
            initial_body_axis_alignment_deg=float(alignment[0]),
            twist_rotation_speed=twist_rotation_speed,
            twist_inertia_proxy=twist_inertia_proxy,
            markers=markers,
            center_of_mass=center_of_mass,
            flight_time=flight_time,
            equivalent_angular_momentum=desired_angular_momentum_world,
        )

    def trunk_tracking_objective(self, result: FlightSimulationResult) -> float:
        """Return the scalar objective used to assess trunk stabilization quality."""

        trunk_angles = result.q[:, 6:9]
        trunk_velocities = result.qdot[:, 6:9]
        trunk_torques = result.tau[:, 6:9]
        time = result.time
        if time.size == 1:
            angle_cost = float(np.sum(trunk_angles[-1] ** 2))
            velocity_cost = float(np.sum(trunk_velocities[-1] ** 2))
            torque_cost = float(np.sum(trunk_torques[-1] ** 2))
        else:
            angle_cost = float(np.trapz(np.sum(trunk_angles**2, axis=1), time))
            velocity_cost = float(np.trapz(np.sum(trunk_velocities**2, axis=1), time))
            torque_cost = float(np.trapz(np.sum(trunk_torques**2, axis=1), time))

        terminal_cost = float(
            10.0 * np.sum(trunk_angles[-1] ** 2) + 2.0 * np.sum(trunk_velocities[-1] ** 2)
        )
        return angle_cost + 0.2 * velocity_cost + 1e-4 * torque_cost + terminal_cost

    @staticmethod
    def twist_accumulation_turns(result: FlightSimulationResult) -> float:
        """Return the absolute accumulated twist over the flight expressed in turns."""

        twist_angle = float(result.q[-1, 5] - result.q[0, 5])
        return abs(twist_angle) / (2.0 * np.pi)

    def tune_trunk_controller(
        self,
        parameters: FlightSimulationParameters,
        *,
        max_iterations: int = 20,
        optimization_sample_count: int = 61,
    ) -> PDOptimizationResult:
        """Tune the trunk PD gains with a sub-optimal simulation-based search."""

        base_parameters = replace(
            parameters,
            stabilize_trunk=True,
            sample_count=max(11, int(optimization_sample_count)),
        )
        initial_controller = base_parameters.controller
        initial_guess = np.array(
            [
                *initial_controller.proportional_gains,
                *initial_controller.derivative_gains,
            ],
            dtype=float,
        )
        bounds = [(0.0, 600.0), (0.0, 600.0), (0.0, 400.0), (0.0, 80.0), (0.0, 80.0), (0.0, 60.0)]

        def objective(values: np.ndarray) -> float:
            controller = PDControllerConfiguration(
                proportional_gains=tuple(float(value) for value in values[:3]),
                derivative_gains=tuple(float(value) for value in values[3:]),
                torque_limits=initial_controller.torque_limits,
            )
            candidate_parameters = replace(base_parameters, controller=controller)
            result = self.simulate(candidate_parameters)
            return self.trunk_tracking_objective(result)

        optimization = minimize(
            objective,
            x0=initial_guess,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": max_iterations},
        )
        tuned_values = optimization.x if optimization.x is not None else initial_guess
        tuned_controller = PDControllerConfiguration(
            proportional_gains=tuple(float(value) for value in tuned_values[:3]),
            derivative_gains=tuple(float(value) for value in tuned_values[3:]),
            torque_limits=initial_controller.torque_limits,
        )
        return PDOptimizationResult(
            controller=tuned_controller,
            objective_value=float(optimization.fun),
            iterations=int(getattr(optimization, "nit", 0)),
            evaluations=int(getattr(optimization, "nfev", 0)),
            success=bool(optimization.success),
            message=str(optimization.message),
        )

    def optimize_inward_tilt_for_twist(
        self,
        parameters: FlightSimulationParameters,
        *,
        bounds: tuple[float, float] = (0.0, 30.0),
        max_iterations: int = 20,
        optimization_sample_count: int = 61,
    ) -> InwardTiltOptimizationResult:
        """Find the inward tilt that maximizes the accumulated twist during flight."""

        base_parameters = replace(
            parameters,
            inward_tilt_deg=float(np.clip(parameters.inward_tilt_deg, *bounds)),
            sample_count=max(11, int(optimization_sample_count)),
        )

        def objective(inward_tilt_deg: float) -> float:
            candidate_parameters = replace(base_parameters, inward_tilt_deg=float(inward_tilt_deg))
            result = self.simulate(candidate_parameters)
            return -self.twist_accumulation_turns(result)

        optimization = minimize_scalar(
            objective,
            bounds=bounds,
            method="bounded",
            options={"maxiter": max_iterations, "xatol": 1e-2},
        )
        optimal_tilt = float(optimization.x)
        optimal_result = self.simulate(replace(base_parameters, inward_tilt_deg=optimal_tilt))
        return InwardTiltOptimizationResult(
            inward_tilt_deg=optimal_tilt,
            twist_turns=self.twist_accumulation_turns(optimal_result),
            evaluations=int(getattr(optimization, "nfev", 0)),
            success=bool(optimization.success),
            message=str(getattr(optimization, "message", "")),
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

    def _ground_clearance(
        self,
        time: float,
        state: np.ndarray,
        takeoff_vertical_velocity: float,
        initial_root_height: float,
    ) -> float:
        """Return the vertical clearance of the lowest marker above the ground."""

        if time <= 1e-10:
            return 1.0
        q = np.zeros(self.model.nbQ(), dtype=float)
        q[2] = initial_root_height + self.ballistic_height(time, takeoff_vertical_velocity)
        q[3:9] = state[:ROTATION_STATE_DOF]
        return float(np.min(self.markers(q)[:, 2]))

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
