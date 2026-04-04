"""Matplotlib GUI for exploring aerial skater dynamics."""

from __future__ import annotations

from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, CheckButtons, Slider

from skating_aerial_alignment.simulation import (
    FlightSimulationParameters,
    FlightSimulationResult,
    SkaterFlightSimulator,
)


def skeleton_connections() -> list[tuple[str, str]]:
    """Return the marker pairs used to draw the reduced whole-body skeleton."""

    return [
        ("pelvis_origin", "thorax_top"),
        ("thorax_top", "head_top"),
        ("shoulder_left", "shoulder_right"),
        ("hip_left", "hip_right"),
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

    angular_momentum_body = simulator.angular_momentum_from_rps(parameters.angular_velocity_rps)
    controller_label = "PD actif" if parameters.stabilize_trunk else "tronc passif"
    return (
        f"Temps de vol ~ {result.flight_time:.3f} s | "
        f"Vz = {parameters.takeoff_vertical_velocity:.2f} m/s | "
        f"H_corps = [{angular_momentum_body[0]:.2f}, {angular_momentum_body[1]:.2f}, "
        f"{angular_momentum_body[2]:.2f}] Nms | "
        f"{controller_label}"
    )


class SkatingAerialAlignmentApp:
    """Interactive GUI combining sliders, 3D animation, and temporal plots."""

    def __init__(self) -> None:
        """Create the figure, controls, and initial simulation."""

        self.simulator = SkaterFlightSimulator()
        self.parameters = FlightSimulationParameters()
        self.result = self.simulator.simulate(self.parameters)
        self.frame_index = 0
        self.is_paused = False

        self.figure = plt.figure(figsize=(16, 10))
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

    def _build_axes(self) -> None:
        """Create the main plotting axes."""

        grid = self.figure.add_gridspec(
            4,
            2,
            left=0.06,
            right=0.98,
            top=0.90,
            bottom=0.27,
            hspace=0.38,
            wspace=0.24,
        )
        self.ax_3d = self.figure.add_subplot(grid[:, 0], projection="3d")
        self.ax_alignment = self.figure.add_subplot(grid[0, 1])
        self.ax_rotation = self.figure.add_subplot(grid[1, 1])
        self.ax_trunk = self.figure.add_subplot(grid[2, 1], sharex=self.ax_alignment)
        self.ax_torque = self.figure.add_subplot(grid[3, 1], sharex=self.ax_alignment)

        self.ax_3d.set_title("Animation 3D")
        self.ax_alignment.set_title("Angle(moment cinetique, axe longitudinal)")
        self.ax_rotation.set_title("Vrille et salto")
        self.ax_trunk.set_title("3 DoF du tronc")
        self.ax_torque.set_title("Efforts du tronc")

        self.ax_alignment.set_ylabel("deg")
        self.ax_rotation.set_ylabel("deg")
        self.ax_trunk.set_ylabel("deg")
        self.ax_torque.set_ylabel("N.m")
        self.ax_torque.set_xlabel("Temps (s)")

        self.ax_3d.set_xlabel("x")
        self.ax_3d.set_ylabel("y")
        self.ax_3d.set_zlabel("z")
        self.ax_3d.view_init(elev=18, azim=-70)

    def _build_controls(self) -> None:
        """Create sliders, buttons, and checkboxes."""

        slider_specs = [
            ("salto_rps", [0.08, 0.20, 0.22, 0.022], "Salto eq. (rot/s)", -2.0, 4.0, 0.0),
            ("tilt_rps", [0.08, 0.16, 0.22, 0.022], "Interieur eq. (rot/s)", -2.0, 2.0, 0.0),
            ("twist_rps", [0.08, 0.12, 0.22, 0.022], "Vrille eq. (rot/s)", -4.0, 6.0, 3.0),
            (
                "takeoff_velocity",
                [0.42, 0.20, 0.22, 0.022],
                "Vitesse verticale (m/s)",
                0.1,
                1.0,
                0.60,
            ),
            (
                "somersault_tilt",
                [0.42, 0.16, 0.22, 0.022],
                "Inclinaison salto (deg)",
                0.0,
                30.0,
                0.0,
            ),
            (
                "inward_tilt",
                [0.42, 0.12, 0.22, 0.022],
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

        checkbox_axis = self.figure.add_axes([0.78, 0.12, 0.16, 0.09])
        self.stabilization_checkbox = CheckButtons(
            checkbox_axis,
            labels=["Stabiliser le tronc"],
            actives=[False],
        )
        self.stabilization_checkbox.on_clicked(self._on_parameter_change)

        pause_axis = self.figure.add_axes([0.78, 0.22, 0.08, 0.035])
        self.pause_button = Button(pause_axis, "Pause")
        self.pause_button.on_clicked(self._toggle_pause)

        reset_axis = self.figure.add_axes([0.88, 0.22, 0.08, 0.035])
        self.reset_button = Button(reset_axis, "Reset")
        self.reset_button.on_clicked(self._reset_controls)

    def _build_plot_artists(self) -> None:
        """Initialize the lines that will be updated after each simulation."""

        self.skeleton_lines = []
        for _connection in skeleton_connections():
            (line,) = self.ax_3d.plot([], [], [], color="#0B3C5D", linewidth=2.0)
            self.skeleton_lines.append(line)

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

        self.time_cursors = [
            axis.axvline(0.0, color="black", linestyle="--", linewidth=1.0)
            for axis in (self.ax_alignment, self.ax_rotation, self.ax_trunk, self.ax_torque)
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
            takeoff_vertical_velocity=self.sliders["takeoff_velocity"].val,
            somersault_tilt_deg=self.sliders["somersault_tilt"].val,
            inward_tilt_deg=self.sliders["inward_tilt"].val,
            stabilize_trunk=self.stabilization_checkbox.get_status()[0],
        )

    def _refresh_from_result(self, *, reset_animation: bool) -> None:
        """Refresh plots after a simulation update."""

        if reset_animation:
            self.frame_index = 0

        self.status_text_artist.set_text(
            format_status_text(self.parameters, self.result, self.simulator)
        )

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

        self._autoscale_axis(self.ax_alignment, time, [self.result.body_axis_alignment_deg])
        self._autoscale_axis(
            self.ax_rotation,
            time,
            [np.rad2deg(self.result.q[:, 3]), np.rad2deg(self.result.q[:, 5])],
        )
        self._autoscale_axis(self.ax_trunk, time, [column for column in trunk_angles_deg.T])
        self._autoscale_axis(self.ax_torque, time, [column for column in trunk_torques.T])

        self._set_3d_bounds()
        self._draw_frame(self.frame_index)
        self.figure.canvas.draw_idle()

    def _on_parameter_change(self, _value) -> None:
        """Re-simulate after a slider or checkbox update."""

        self.parameters = self._collect_parameters()
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
        self.sliders["salto_rps"].set_val(defaults.angular_velocity_rps[0])
        self.sliders["tilt_rps"].set_val(defaults.angular_velocity_rps[1])
        self.sliders["twist_rps"].set_val(defaults.angular_velocity_rps[2])
        self.sliders["takeoff_velocity"].set_val(defaults.takeoff_vertical_velocity)
        self.sliders["somersault_tilt"].set_val(defaults.somersault_tilt_deg)
        self.sliders["inward_tilt"].set_val(defaults.inward_tilt_deg)
        if self.stabilization_checkbox.get_status()[0]:
            self.stabilization_checkbox.set_active(0)

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

    def _draw_frame(self, frame_index: int) -> None:
        """Update the 3D skeleton and time cursors for one frame."""

        markers = self.result.markers[frame_index]
        time = self.result.time[frame_index]
        for line, (start_name, end_name) in zip(self.skeleton_lines, skeleton_connections()):
            start = markers[self.simulator.marker_index[start_name]]
            end = markers[self.simulator.marker_index[end_name]]
            line.set_data([start[0], end[0]], [start[1], end[1]])
            line.set_3d_properties([start[2], end[2]])

        pelvis = markers[self.simulator.marker_index["pelvis_origin"]]
        body_tip = pelvis + 0.5 * self.result.body_axis[frame_index]
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
            self.frame_index = (self.frame_index + 1) % len(self.result.time)
            self._draw_frame(self.frame_index)
        return []

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
