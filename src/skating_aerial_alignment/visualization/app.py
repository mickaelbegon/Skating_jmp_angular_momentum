"""Matplotlib GUI for exploring aerial skater dynamics."""

from __future__ import annotations

from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, CheckButtons, RadioButtons, Slider

from skating_aerial_alignment.simulation import (
    FlightSimulationParameters,
    FlightSimulationResult,
    PDOptimizationResult,
    SkaterFlightSimulator,
)


def skeleton_connections() -> list[tuple[str, str]]:
    """Return the marker pairs used to draw the reduced whole-body skeleton."""

    return [
        ("pelvis_origin", "thorax_top"),
        ("thorax_top", "head_top"),
        ("shoulder_left", "shoulder_right"),
        ("hip_left", "hip_right"),
        ("hip_left", "pelvis_thorax_joint_center"),
        ("hip_right", "pelvis_thorax_joint_center"),
        ("shoulder_left", "pelvis_thorax_joint_center"),
        ("shoulder_right", "pelvis_thorax_joint_center"),
        ("shoulder_left", "elbow_left"),
        ("elbow_left", "wrist_left"),
        ("wrist_left", "hand_left"),
        ("shoulder_right", "elbow_right"),
        ("elbow_right", "wrist_right"),
        ("wrist_right", "hand_right"),
        ("pelvis_origin", "hip_left"),
        ("hip_left", "knee_left"),
        ("knee_left", "ankle_left"),
        ("ankle_left", "toe_left"),
        ("pelvis_origin", "hip_right"),
        ("hip_right", "knee_right"),
        ("knee_right", "ankle_right"),
        ("ankle_right", "toe_right"),
    ]


def format_status_text(
    parameters: FlightSimulationParameters,
    result: FlightSimulationResult,
    simulator: SkaterFlightSimulator,
) -> str:
    """Format the textual simulation summary shown above the plots."""

    controller_label = "PD actif" if parameters.stabilize_trunk else "tronc passif"
    return (
        f"Temps de vol ~ {result.flight_time:.3f} s | "
        f"Vz = {parameters.takeoff_vertical_velocity:.2f} m/s | "
        f"Var = {parameters.backward_horizontal_velocity:.2f} m/s | "
        f"Angle initial = {result.initial_body_axis_alignment_deg:.1f} deg | "
        f"H_global = [{result.equivalent_angular_momentum[0]:.2f}, "
        f"{result.equivalent_angular_momentum[1]:.2f}, "
        f"{result.equivalent_angular_momentum[2]:.2f}] Nms | "
        f"{controller_label}"
    )


def format_inertia_and_controller_text(
    parameters: FlightSimulationParameters,
    simulator: SkaterFlightSimulator,
    optimization_result: PDOptimizationResult | None,
) -> str:
    """Format the secondary status line with inertias and controller gains."""

    principal_moments = simulator.biomod_builder.principal_moments()
    kp = parameters.controller.proportional_gains
    kd = parameters.controller.derivative_gains
    optimization_text = "PD manuel"
    if optimization_result is not None:
        optimization_text = (
            f"PD auto J={optimization_result.objective_value:.3e} "
            f"({optimization_result.evaluations} eval.)"
        )
    inertia_text = (
        f"I = [{principal_moments[0]:.2f}, {principal_moments[1]:.2f}, "
        f"{principal_moments[2]:.2f}] kg.m^2"
    )
    return (
        f"{inertia_text} | "
        f"Kp = [{kp[0]:.1f}, {kp[1]:.1f}, {kp[2]:.1f}] | "
        f"Kd = [{kd[0]:.1f}, {kd[1]:.1f}, {kd[2]:.1f}] | "
        f"{optimization_text}"
    )


class SkatingAerialAlignmentApp:
    """Interactive GUI combining sliders, 3D animation, and temporal plots."""

    ANIMATION_TIMER_INTERVAL_MS = 16
    DEFAULT_VIEW = (18.0, -70.0)
    FACE_VIEW = (8.0, 90.0)

    def __init__(self) -> None:
        """Create the figure, controls, and initial simulation."""

        self.simulator = SkaterFlightSimulator()
        self.parameters = FlightSimulationParameters()
        self.result = self.simulator.simulate(self.parameters)
        self.optimization_result: PDOptimizationResult | None = None
        self.frame_index = 0
        self.is_paused = False
        self.frames_per_animation_step = 1
        self.animation_speed_fraction = 1.0
        self.animation_duration_seconds = 0.0

        self.figure = plt.figure(figsize=(16, 11))
        self.figure.suptitle(
            "Phase aerienne d'un saut en patinage: alignement moment cinetique / axe du corps",
            fontsize=15,
            y=0.98,
        )
        self.status_text_artist = self.figure.text(
            0.06,
            0.945,
            format_status_text(self.parameters, self.result, self.simulator),
            fontsize=10,
        )
        self.details_text_artist = self.figure.text(
            0.06,
            0.922,
            format_inertia_and_controller_text(
                self.parameters,
                self.simulator,
                self.optimization_result,
            ),
            fontsize=9,
        )
        self.playback_text_artist = self.figure.text(
            0.68,
            0.115,
            "",
            fontsize=9,
        )

        self._build_axes()
        self._build_controls()
        self._build_plot_artists()
        self._refresh_from_result(reset_animation=True)

        self.animation = FuncAnimation(
            self.figure,
            self._animate,
            interval=40,
            blit=False,
            cache_frame_data=False,
        )
        self._update_animation_playback()

    def _build_axes(self) -> None:
        """Create the main plotting axes."""

        grid = self.figure.add_gridspec(
            5,
            2,
            left=0.06,
            right=0.98,
            top=0.90,
            bottom=0.35,
            hspace=0.38,
            wspace=0.24,
        )
        self.ax_3d = self.figure.add_subplot(grid[:, 0], projection="3d")
        self.ax_alignment = self.figure.add_subplot(grid[0, 1])
        self.ax_rotation = self.figure.add_subplot(grid[1, 1])
        self.ax_trunk = self.figure.add_subplot(grid[2, 1], sharex=self.ax_alignment)
        self.ax_torque = self.figure.add_subplot(grid[3, 1], sharex=self.ax_alignment)
        self.ax_inertia = self.figure.add_subplot(grid[4, 1], sharex=self.ax_alignment)

        self.ax_3d.set_title("Animation 3D")
        self.ax_alignment.set_title("Angle(moment cinetique, axe longitudinal)")
        self.ax_rotation.set_title("Vrille et salto")
        self.ax_trunk.set_title("3 DoF du tronc")
        self.ax_torque.set_title("Efforts du tronc")
        self.ax_inertia.set_title("Moments d'inertie du corps complet")

        self.ax_alignment.set_ylabel("deg")
        self.ax_rotation.set_ylabel("deg")
        self.ax_trunk.set_ylabel("deg")
        self.ax_torque.set_ylabel("N.m")
        self.ax_inertia.set_ylabel("kg.m^2")
        self.ax_inertia.set_xlabel("Temps (s)")

        self.ax_3d.set_xlabel("x")
        self.ax_3d.set_ylabel("y")
        self.ax_3d.set_zlabel("z")
        self.ax_3d.view_init(elev=18, azim=-70)

    def _build_controls(self) -> None:
        """Create sliders, buttons, and checkboxes."""

        slider_specs = [
            (
                "salto_rps",
                [0.08, 0.28, 0.22, 0.022],
                "Hx global eq. (rot/s)",
                0.0,
                0.25,
                0.0,
            ),
            ("tilt_rps", [0.08, 0.24, 0.22, 0.022], "Hy global eq. (rot/s)", -2.0, 2.0, 0.0),
            ("twist_rps", [0.08, 0.20, 0.22, 0.022], "Hz global eq. (rot/s)", -4.0, 6.0, 3.0),
            (
                "backward_velocity",
                [0.08, 0.16, 0.22, 0.022],
                "Vitesse arriere (m/s)",
                0.0,
                3.0,
                self.parameters.backward_horizontal_velocity,
            ),
            (
                "takeoff_velocity",
                [0.42, 0.28, 0.22, 0.022],
                "Vitesse verticale (m/s)",
                0.1,
                3.0,
                self.parameters.takeoff_vertical_velocity,
            ),
            (
                "somersault_tilt",
                [0.42, 0.24, 0.22, 0.022],
                "Inclinaison salto (deg)",
                0.0,
                30.0,
                0.0,
            ),
            (
                "inward_tilt",
                [0.42, 0.20, 0.22, 0.022],
                "Inclinaison interieur (deg)",
                0.0,
                30.0,
                0.0,
            ),
        ]

        self.sliders: dict[str, Slider] = {}
        for name, rect, label, vmin, vmax, initial in slider_specs:
            axis = self.figure.add_axes(rect)
            slider = Slider(axis, label=label, valmin=vmin, valmax=vmax, valinit=initial)
            slider.on_changed(self._on_parameter_change)
            self.sliders[name] = slider

        optimize_axis = self.figure.add_axes([0.08, 0.07, 0.10, 0.04])
        self.optimize_button = Button(optimize_axis, "Auto PD")
        self.optimize_button.on_clicked(self._optimize_controller)

        pause_axis = self.figure.add_axes([0.20, 0.07, 0.10, 0.04])
        self.pause_button = Button(pause_axis, "Pause")
        self.pause_button.on_clicked(self._toggle_pause)

        reset_axis = self.figure.add_axes([0.32, 0.07, 0.10, 0.04])
        self.reset_button = Button(reset_axis, "Reset")
        self.reset_button.on_clicked(self._reset_controls)

        checkbox_axis = self.figure.add_axes([0.46, 0.04, 0.20, 0.11])
        self.stabilization_checkbox = CheckButtons(
            checkbox_axis,
            labels=["Stabiliser le tronc", "Avatar de face (vrille=0)"],
            actives=[False, False],
        )
        self.stabilization_checkbox.on_clicked(self._on_parameter_change)

        playback_axis = self.figure.add_axes([0.68, 0.04, 0.12, 0.11])
        self.playback_selector = RadioButtons(
            playback_axis,
            labels=("100%", "50%", "25%"),
            active=0,
        )
        self.playback_selector.on_clicked(self._on_playback_change)

    def _build_plot_artists(self) -> None:
        """Initialize the lines that will be updated after each simulation."""

        self.skeleton_lines = []
        for _connection in skeleton_connections():
            (line,) = self.ax_3d.plot([], [], [], color="#0B3C5D", linewidth=2.0)
            self.skeleton_lines.append(line)

        self.ground_surface = None
        (self.body_axis_line,) = self.ax_3d.plot([], [], [], color="#2CA02C", linewidth=2.5)
        (self.angular_momentum_line,) = self.ax_3d.plot([], [], [], color="#D62728", linewidth=2.5)

        (self.alignment_line,) = self.ax_alignment.plot([], [], color="#D62728", linewidth=2.0)
        (self.salto_line,) = self.ax_rotation.plot(
            [], [], label="Salto", color="#1F77B4", linewidth=2.0
        )
        (self.twist_line,) = self.ax_rotation.plot(
            [], [], label="Vrille", color="#FF7F0E", linewidth=2.0
        )
        self.ax_rotation.legend(loc="upper left")

        self.trunk_lines = [
            self.ax_trunk.plot([], [], label=label, linewidth=2.0)[0]
            for label in ("Tronc x", "Tronc y", "Tronc z")
        ]
        self.ax_trunk.legend(loc="upper left")

        self.torque_lines = [
            self.ax_torque.plot([], [], label=label, linewidth=2.0)[0]
            for label in ("Couple x", "Couple y", "Couple z")
        ]
        self.ax_torque.legend(loc="upper left")

        self.principal_moment_lines = [
            self.ax_inertia.plot([], [], label=label, linewidth=1.6)[0]
            for label in ("eig 1", "eig 2", "eig 3")
        ]
        (self.longitudinal_inertia_line,) = self.ax_inertia.plot(
            [],
            [],
            color="#2CA02C",
            linewidth=2.6,
            label="I longitudinal",
        )
        self.ax_inertia.legend(loc="upper left")

        self.time_cursors = [
            axis.axvline(0.0, color="black", linestyle="--", linewidth=1.0)
            for axis in (
                self.ax_alignment,
                self.ax_rotation,
                self.ax_trunk,
                self.ax_torque,
                self.ax_inertia,
            )
        ]

    def _collect_parameters(self) -> FlightSimulationParameters:
        """Read the current GUI values into a simulation parameter object."""

        return replace(
            self.parameters,
            angular_velocity_rps=(
                self.sliders["salto_rps"].val,
                self.sliders["tilt_rps"].val,
                self.sliders["twist_rps"].val,
            ),
            backward_horizontal_velocity=self.sliders["backward_velocity"].val,
            takeoff_vertical_velocity=self.sliders["takeoff_velocity"].val,
            somersault_tilt_deg=self.sliders["somersault_tilt"].val,
            inward_tilt_deg=self.sliders["inward_tilt"].val,
            stabilize_trunk=self._stabilization_enabled(),
        )

    def _refresh_from_result(self, *, reset_animation: bool) -> None:
        """Refresh plots after a simulation update."""

        if reset_animation:
            self.frame_index = 0

        self.status_text_artist.set_text(
            format_status_text(self.parameters, self.result, self.simulator)
        )
        self.details_text_artist.set_text(
            format_inertia_and_controller_text(
                self.parameters,
                self.simulator,
                self.optimization_result,
            )
        )
        self._update_animation_playback()

        time = self.result.time
        self.alignment_line.set_data(time, self.result.body_axis_alignment_deg)
        self.salto_line.set_data(time, np.rad2deg(self.result.q[:, 3]))
        self.twist_line.set_data(time, np.rad2deg(self.result.q[:, 5]))

        trunk_angles_deg = np.rad2deg(self.result.q[:, 6:9])
        trunk_torques = self.result.tau[:, 6:9]
        for line, values in zip(self.trunk_lines, trunk_angles_deg.T):
            line.set_data(time, values)
        for line, values in zip(self.torque_lines, trunk_torques.T):
            line.set_data(time, values)
        for line, values in zip(self.principal_moment_lines, self.result.principal_moments.T):
            line.set_data(time, values)
        self.longitudinal_inertia_line.set_data(
            time,
            self.result.longitudinal_principal_moment,
        )

        self._autoscale_axis(self.ax_alignment, time, [self.result.body_axis_alignment_deg])
        self._autoscale_axis(
            self.ax_rotation,
            time,
            [np.rad2deg(self.result.q[:, 3]), np.rad2deg(self.result.q[:, 5])],
        )
        self._autoscale_axis(self.ax_trunk, time, [column for column in trunk_angles_deg.T])
        self._autoscale_axis(self.ax_torque, time, [column for column in trunk_torques.T])
        self._autoscale_axis(
            self.ax_inertia,
            time,
            [column for column in self.result.principal_moments.T]
            + [self.result.longitudinal_principal_moment],
        )

        self._set_3d_bounds()
        self._apply_view_mode()
        self._draw_frame(self.frame_index)
        self.figure.canvas.draw_idle()

    def _on_parameter_change(self, _value) -> None:
        """Re-simulate after a slider or checkbox update."""

        self.parameters = self._collect_parameters()
        self.optimization_result = None
        self.result = self.simulator.simulate(self.parameters)
        self._refresh_from_result(reset_animation=True)

    def _toggle_pause(self, _event) -> None:
        """Pause or resume the 3D animation."""

        self.is_paused = not self.is_paused
        self.pause_button.label.set_text("Play" if self.is_paused else "Pause")
        self.figure.canvas.draw_idle()

    def _reset_controls(self, _event) -> None:
        """Restore the default slider values."""

        defaults = FlightSimulationParameters()
        self.parameters = defaults
        self.optimization_result = None
        self.sliders["salto_rps"].set_val(defaults.angular_velocity_rps[0])
        self.sliders["tilt_rps"].set_val(defaults.angular_velocity_rps[1])
        self.sliders["twist_rps"].set_val(defaults.angular_velocity_rps[2])
        self.sliders["backward_velocity"].set_val(defaults.backward_horizontal_velocity)
        self.sliders["takeoff_velocity"].set_val(defaults.takeoff_vertical_velocity)
        self.sliders["somersault_tilt"].set_val(defaults.somersault_tilt_deg)
        self.sliders["inward_tilt"].set_val(defaults.inward_tilt_deg)
        for index, active in enumerate(self.stabilization_checkbox.get_status()):
            if active:
                self.stabilization_checkbox.set_active(index)
        if self.playback_selector.value_selected != "100%":
            self.playback_selector.set_active(0)

    def _optimize_controller(self, _event) -> None:
        """Run the sub-optimal PD calibration for the current scenario."""

        current_parameters = replace(self._collect_parameters(), stabilize_trunk=True)
        optimization_result = self.simulator.tune_trunk_controller(
            current_parameters,
            max_iterations=20,
            optimization_sample_count=61,
        )
        if not self._stabilization_enabled():
            self.stabilization_checkbox.set_active(0)
        self.optimization_result = optimization_result
        self.parameters = replace(
            current_parameters,
            controller=optimization_result.controller,
            stabilize_trunk=True,
        )
        self.result = self.simulator.simulate(self.parameters)
        self._refresh_from_result(reset_animation=True)

    def _on_playback_change(self, label: str) -> None:
        """Update the playback speed from the speed selector."""

        speed_map = {"100%": 1.0, "50%": 0.5, "25%": 0.25}
        self.animation_speed_fraction = speed_map[label]
        self._update_animation_playback()
        self.figure.canvas.draw_idle()

    def _set_3d_bounds(self) -> None:
        """Set an equal 3D view box that contains the full animated trajectory."""

        marker_cloud = self.result.markers.reshape(-1, 3)
        minima = marker_cloud.min(axis=0)
        maxima = marker_cloud.max(axis=0)
        center = 0.5 * (minima + maxima)
        radius = 0.55 * np.max(maxima - minima + 1e-6)
        self.ax_3d.set_xlim(center[0] - radius, center[0] + radius)
        self.ax_3d.set_ylim(center[1] - radius, center[1] + radius)
        self.ax_3d.set_zlim(center[2] - radius, center[2] + radius)
        self.ax_3d.set_box_aspect((1.0, 1.0, 1.0))
        self._draw_ground_plane(center[0], center[1], radius)

    def _draw_ground_plane(self, center_x: float, center_y: float, radius: float) -> None:
        """Draw a visible ground plane at `z = 0` under the animated skater."""

        if self.ground_surface is not None:
            self.ground_surface.remove()

        x_grid = np.array(
            [
                [center_x - radius, center_x + radius],
                [center_x - radius, center_x + radius],
            ]
        )
        y_grid = np.array(
            [
                [center_y - radius, center_y - radius],
                [center_y + radius, center_y + radius],
            ]
        )
        z_grid = np.zeros((2, 2), dtype=float)
        self.ground_surface = self.ax_3d.plot_surface(
            x_grid,
            y_grid,
            z_grid,
            color="#C7B299",
            alpha=0.28,
            linewidth=0.0,
            shade=False,
        )

    def _draw_frame(self, frame_index: int) -> None:
        """Update the 3D skeleton and time cursors for one frame."""

        markers, body_axis = self._display_kinematics(frame_index)
        time = self.result.time[frame_index]
        for line, (start_name, end_name) in zip(self.skeleton_lines, skeleton_connections()):
            start = markers[self.simulator.marker_index[start_name]]
            end = markers[self.simulator.marker_index[end_name]]
            line.set_data([start[0], end[0]], [start[1], end[1]])
            line.set_3d_properties([start[2], end[2]])

        pelvis = markers[self.simulator.marker_index["pelvis_origin"]]
        body_tip = pelvis + 0.5 * body_axis
        self.body_axis_line.set_data([pelvis[0], body_tip[0]], [pelvis[1], body_tip[1]])
        self.body_axis_line.set_3d_properties([pelvis[2], body_tip[2]])

        angular_momentum = self.result.angular_momentum[frame_index]
        if np.linalg.norm(angular_momentum) > 0.0:
            angular_tip = pelvis + 0.08 * angular_momentum
        else:
            angular_tip = pelvis
        self.angular_momentum_line.set_data(
            [pelvis[0], angular_tip[0]], [pelvis[1], angular_tip[1]]
        )
        self.angular_momentum_line.set_3d_properties([pelvis[2], angular_tip[2]])

        for cursor in self.time_cursors:
            cursor.set_xdata([time, time])

    def _animate(self, _frame: int):
        """Advance the animation if it is currently playing."""

        if not self.is_paused:
            self.frame_index = (self.frame_index + self.frames_per_animation_step) % len(
                self.result.time
            )
            self._draw_frame(self.frame_index)
        return []

    def _display_kinematics(self, frame_index: int) -> tuple[np.ndarray, np.ndarray]:
        """Return the marker cloud and longitudinal axis used for display."""

        if not self._face_view_enabled():
            return self.result.markers[frame_index], self.result.body_axis[frame_index]

        display_q = np.asarray(self.result.q[frame_index], dtype=float).copy()
        display_q[5] = 0.0
        display_markers = self.simulator.markers(display_q)
        display_body_axis = self.simulator.body_frame(display_q)[:, 2]
        return display_markers, display_body_axis

    def _stabilization_enabled(self) -> bool:
        """Return whether trunk stabilization is enabled."""

        return bool(self.stabilization_checkbox.get_status()[0])

    def _face_view_enabled(self) -> bool:
        """Return whether the front-view rendering mode is enabled."""

        return bool(self.stabilization_checkbox.get_status()[1])

    def _apply_view_mode(self) -> None:
        """Apply the current 3D camera configuration."""

        elev, azim = self.FACE_VIEW if self._face_view_enabled() else self.DEFAULT_VIEW
        self.ax_3d.view_init(elev=elev, azim=azim)

    def _update_animation_playback(self) -> None:
        """Estimate playback duration and set the animation stepping accordingly."""

        self.animation_speed_fraction = max(self.animation_speed_fraction, 1e-6)
        target_duration = (
            self.result.flight_time / self.animation_speed_fraction
            if self.result.flight_time > 0.0
            else 0.0
        )
        steps = max(
            1,
            int(np.ceil(target_duration * 1000.0 / self.ANIMATION_TIMER_INTERVAL_MS)),
        )
        self.frames_per_animation_step = max(
            1,
            int(np.ceil(max(len(self.result.time) - 1, 1) / steps)),
        )
        self.animation_duration_seconds = (
            max(len(self.result.time) - 1, 1)
            / self.frames_per_animation_step
            * self.ANIMATION_TIMER_INTERVAL_MS
            / 1000.0
        )
        self.playback_text_artist.set_text(
            "Lecture: "
            f"{int(round(self.animation_speed_fraction * 100.0))}% "
            f"| temps reel ~ {self.result.flight_time:.2f} s "
            f"| anim ~ {self.animation_duration_seconds:.2f} s"
        )
        if hasattr(self, "animation") and self.animation.event_source is not None:
            self.animation.event_source.interval = self.ANIMATION_TIMER_INTERVAL_MS

    @staticmethod
    def _autoscale_axis(ax, time: np.ndarray, series: list[np.ndarray]) -> None:
        """Autoscale one temporal axis around the provided traces."""

        ax.set_xlim(time[0], time[-1] if time[-1] > time[0] else time[0] + 1e-6)
        stacked = np.concatenate([np.asarray(values, dtype=float).ravel() for values in series])
        y_min = float(np.min(stacked))
        y_max = float(np.max(stacked))
        if np.isclose(y_min, y_max):
            margin = 1.0 if np.isclose(y_max, 0.0) else 0.1 * abs(y_max)
        else:
            margin = 0.08 * (y_max - y_min)
        ax.set_ylim(y_min - margin, y_max + margin)
        ax.grid(True, alpha=0.3)


def launch_app() -> SkatingAerialAlignmentApp:
    """Launch the interactive matplotlib application and return it."""

    app = SkatingAerialAlignmentApp()
    plt.show()
    return app
