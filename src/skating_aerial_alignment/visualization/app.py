"""Matplotlib GUI for exploring aerial skater dynamics."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, CheckButtons, RadioButtons, Slider

from skating_aerial_alignment.simulation import (
    AlignmentOptimizationResult,
    FlightSimulationParameters,
    FlightSimulationResult,
    InwardTiltOptimizationResult,
    PDOptimizationResult,
    SkaterFlightSimulator,
)


class SafeCheckButtons(CheckButtons):
    """CheckButtons variant that ignores incomplete hit-test payloads from Matplotlib."""

    def _clicked(self, event):
        """Ignore malformed `contains` payloads instead of raising `KeyError`."""

        if self.ignore(event) or event.button != 1 or not self.ax.contains(event)[0]:
            return
        frame_details = self._frames.contains(event)[1]
        frame_indices = list(frame_details.get("ind", []))
        label_indices = [i for i, text in enumerate(self.labels) if text.contains(event)[0]]
        candidate_indices = [*frame_indices, *label_indices]
        if candidate_indices:
            coords = self._frames.get_offset_transform().transform(self._frames.get_offsets())
            self.set_active(
                candidate_indices[
                    (((event.x, event.y) - coords[candidate_indices]) ** 2).sum(-1).argmin()
                ]
            )

    def set_active(self, index, state=None):
        """Toggle one checkbox while keeping edge and face colors synchronized."""

        super().set_active(index, state)
        invisible = mcolors.to_rgba("none")
        facecolors = self._checks.get_facecolor()
        edgecolors = np.asarray(facecolors).copy()
        for color_index, color in enumerate(facecolors):
            if mcolors.same_color(color, invisible):
                edgecolors[color_index] = invisible
        self._checks.set_edgecolor(edgecolors)


class SafeRadioButtons(RadioButtons):
    """RadioButtons variant that ignores incomplete hit-test payloads from Matplotlib."""

    def _clicked(self, event):
        """Ignore malformed `contains` payloads instead of raising `KeyError`."""

        if self.ignore(event) or event.button != 1 or not self.ax.contains(event)[0]:
            return
        button_details = self._buttons.contains(event)[1]
        button_indices = list(button_details.get("ind", []))
        label_indices = [i for i, text in enumerate(self.labels) if text.contains(event)[0]]
        candidate_indices = [*button_indices, *label_indices]
        if candidate_indices:
            coords = self._buttons.get_offset_transform().transform(self._buttons.get_offsets())
            self.set_active(
                candidate_indices[
                    (((event.x, event.y) - coords[candidate_indices]) ** 2).sum(-1).argmin()
                ]
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
    backward_distance = abs(parameters.backward_horizontal_velocity) * result.flight_time
    return (
        f"Tvol {result.flight_time:.3f} s | "
        f"Vz = {parameters.takeoff_vertical_velocity:.2f} m/s | "
        f"V_arr = {abs(parameters.backward_horizontal_velocity):.2f} m/s | "
        f"Arr. = {backward_distance:.2f} m | "
        f"Align. init. = {result.initial_body_axis_alignment_deg:.1f} deg | "
        f"H = [{result.equivalent_angular_momentum[0]:.2f}, "
        f"{result.equivalent_angular_momentum[1]:.2f}, "
        f"{result.equivalent_angular_momentum[2]:.2f}] Nms | "
        f"{controller_label}"
    )


def format_inertia_and_controller_text(
    parameters: FlightSimulationParameters,
    simulator: SkaterFlightSimulator,
    optimization_result: PDOptimizationResult | None,
    inward_tilt_optimization_result: InwardTiltOptimizationResult | None,
    alignment_optimization_result: AlignmentOptimizationResult | None,
) -> str:
    """Format the secondary status line with inertias and controller gains."""

    principal_moments = simulator.biomod_builder.principal_moments()
    kp = parameters.controller.proportional_gains
    kd = parameters.controller.derivative_gains
    optimization_text = "PD manuel"
    if optimization_result is not None:
        optimization_text = (
            f"PD auto J={optimization_result.objective_value:.2e} "
            f"({optimization_result.evaluations} eval.)"
        )
    tilt_optimization_text = "incl. int. manuelle"
    if alignment_optimization_result is not None:
        tilt_optimization_text = (
            f"align. auto {alignment_optimization_result.inward_tilt_deg:.1f} deg "
            f"(moy. {alignment_optimization_result.mean_alignment_deg:.2f} deg)"
        )
    elif inward_tilt_optimization_result is not None:
        tilt_optimization_text = (
            f"incl. int. auto {inward_tilt_optimization_result.inward_tilt_deg:.1f} deg "
            f"({inward_tilt_optimization_result.twist_turns:.2f} tours)"
        )
    inertia_text = (
        f"I = [{principal_moments[0]:.2f}, {principal_moments[1]:.2f}, "
        f"{principal_moments[2]:.2f}] kg.m^2"
    )
    return (
        f"{inertia_text} | "
        f"Kp = [{kp[0]:.1f}, {kp[1]:.1f}, {kp[2]:.1f}] | "
        f"Kd = [{kd[0]:.1f}, {kd[1]:.1f}, {kd[2]:.1f}] | "
        f"{optimization_text} | "
        f"{tilt_optimization_text}"
    )


class SkatingAerialAlignmentApp:
    """Interactive GUI combining sliders, 3D animation, and temporal plots."""

    ANIMATION_TIMER_INTERVAL_MS = 16
    DEFAULT_VIEW = (18.0, -70.0)
    FACE_VIEW = (8.0, 90.0)
    DEFAULT_BACKWARD_VELOCITY_M_S = 2.0
    TWIST_OPTIMIZATION_INDEX = 2
    ALIGNMENT_OPTIMIZATION_INDEX = 3
    TWIST_OPTIMIZATION_LABEL = "Optimiser incl. int."
    ALIGNMENT_OPTIMIZATION_LABEL = "Optimiser alignement"
    PRIMARY_BLUE = "#2E6FBB"
    PANEL_BACKGROUND = "#FFFFFF"
    PANEL_BORDER = "#D6DEE8"
    FIGURE_BACKGROUND = "#F3F6F9"
    TEXT_PRIMARY = "#14212B"
    TEXT_SECONDARY = "#475467"

    def __init__(self, pd_cache_path: str | Path | None = None) -> None:
        """Create the figure, controls, and initial simulation."""

        mpl.rcParams["axes3d.mouserotationstyle"] = "azel"
        self.simulator = SkaterFlightSimulator()
        self.pd_cache_path = (
            Path(pd_cache_path)
            if pd_cache_path is not None
            else Path("artifacts") / "pd_tuning_cache.json"
        )
        self.optimization_result: PDOptimizationResult | None = None
        self.optimization_result_signature: tuple[float, ...] | None = None
        self.inward_tilt_optimization_result: InwardTiltOptimizationResult | None = None
        self.alignment_optimization_result: AlignmentOptimizationResult | None = None
        self._pd_tuning_cache: dict[tuple[float, ...], PDOptimizationResult] = {}
        self._inward_tilt_cache: dict[tuple[float, ...], InwardTiltOptimizationResult] = {}
        self._alignment_cache: dict[tuple[float, ...], AlignmentOptimizationResult] = {}
        cached_controller = self._load_pd_tuning_cache()
        self.parameters = FlightSimulationParameters()
        if cached_controller is not None:
            self.parameters = replace(self.parameters, controller=cached_controller)
        self.result = self.simulator.simulate(self.parameters)
        self.frame_index = 0
        self.is_paused = False
        self.frames_per_animation_step = 1
        self.animation_speed_fraction = 1.0
        self.animation_duration_seconds = 0.0
        self._updating_time_slider = False
        self._updating_inward_tilt_slider = False
        self._updating_checkbox_state = False
        self.playback_menu_visible = False
        self.playback_text_artist = None

        self.figure = plt.figure(figsize=(16, 11), facecolor=self.FIGURE_BACKGROUND)
        self.figure.suptitle(
            "Phase aerienne d'un saut en patinage: alignement moment cinetique / axe du corps",
            fontsize=16.5,
            fontweight="semibold",
            color=self.TEXT_PRIMARY,
            y=0.972,
        )
        self.status_text_artist = self.figure.text(
            0.038,
            0.936,
            format_status_text(self.parameters, self.result, self.simulator),
            fontsize=9.4,
            color=self.TEXT_PRIMARY,
            va="center",
            bbox={
                "boxstyle": "round,pad=0.20",
                "facecolor": self.PANEL_BACKGROUND,
                "edgecolor": self.PANEL_BORDER,
                "linewidth": 0.85,
            },
        )
        self.details_text_artist = self.figure.text(
            0.038,
            0.904,
            format_inertia_and_controller_text(
                self.parameters,
                self.simulator,
                self.optimization_result,
                self.inward_tilt_optimization_result,
                self.alignment_optimization_result,
            ),
            fontsize=8.6,
            color=self.TEXT_SECONDARY,
            va="center",
            bbox={
                "boxstyle": "round,pad=0.18",
                "facecolor": self.PANEL_BACKGROUND,
                "edgecolor": self.PANEL_BORDER,
                "linewidth": 0.85,
            },
        )

        self._build_axes()
        self._build_controls()
        self.parameters = self._collect_parameters()
        self._simulate_with_current_parameters()
        self._build_plot_artists()
        self._refresh_from_result(reset_animation=True)

        self.animation = FuncAnimation(
            self.figure,
            self._animate,
            interval=self.ANIMATION_TIMER_INTERVAL_MS,
            blit=False,
            cache_frame_data=False,
        )
        self._update_animation_playback()

    @staticmethod
    def _controller_to_payload(controller) -> dict[str, list[float]]:
        """Serialize a controller configuration into JSON-friendly lists."""

        return {
            "proportional_gains": list(controller.proportional_gains),
            "derivative_gains": list(controller.derivative_gains),
            "torque_limits": list(controller.torque_limits),
        }

    @staticmethod
    def _controller_from_payload(payload):
        """Deserialize a controller configuration from cached JSON data."""

        return replace(
            FlightSimulationParameters().controller,
            proportional_gains=tuple(float(value) for value in payload["proportional_gains"]),
            derivative_gains=tuple(float(value) for value in payload["derivative_gains"]),
            torque_limits=tuple(float(value) for value in payload["torque_limits"]),
        )

    def _load_pd_tuning_cache(self):
        """Load the persisted PD-tuning cache and return the last tuned controller."""

        if not self.pd_cache_path.exists():
            return None
        try:
            payload = json.loads(self.pd_cache_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            return None

        for entry in payload.get("entries", []):
            try:
                signature = tuple(float(value) for value in entry["signature"])
                optimization = PDOptimizationResult(
                    controller=self._controller_from_payload(entry["controller"]),
                    objective_value=float(entry["objective_value"]),
                    iterations=int(entry["iterations"]),
                    evaluations=int(entry["evaluations"]),
                    success=bool(entry["success"]),
                    message=str(entry["message"]),
                )
            except (KeyError, TypeError, ValueError):
                continue
            self._pd_tuning_cache[signature] = optimization

        last_controller_payload = payload.get("last_controller")
        if last_controller_payload is None:
            return None
        try:
            return self._controller_from_payload(last_controller_payload)
        except (KeyError, TypeError, ValueError):
            return None

    def _save_pd_tuning_cache(self, last_controller) -> None:
        """Persist the current PD-tuning cache and last tuned controller to disk."""

        payload = {
            "entries": [
                {
                    "signature": list(signature),
                    "controller": self._controller_to_payload(optimization.controller),
                    "objective_value": optimization.objective_value,
                    "iterations": optimization.iterations,
                    "evaluations": optimization.evaluations,
                    "success": optimization.success,
                    "message": optimization.message,
                }
                for signature, optimization in self._pd_tuning_cache.items()
            ],
            "last_controller": self._controller_to_payload(last_controller),
        }
        self.pd_cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.pd_cache_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _build_axes(self) -> None:
        """Create the main plotting axes."""

        grid = self.figure.add_gridspec(
            5,
            2,
            left=0.045,
            right=0.928,
            top=0.84,
            bottom=0.345,
            hspace=0.42,
            wspace=0.24,
            width_ratios=(1.07, 0.93),
        )
        self.ax_3d = self.figure.add_subplot(grid[:, 0], projection="3d")
        self.ax_alignment = self.figure.add_subplot(grid[0, 1])
        self.ax_rotation = self.figure.add_subplot(grid[1, 1])
        self.ax_rotation_salto = self.ax_rotation.twinx()
        self.ax_trunk = self.figure.add_subplot(grid[2, 1], sharex=self.ax_alignment)
        self.ax_torque = self.figure.add_subplot(grid[3, 1], sharex=self.ax_alignment)
        self.ax_inertia = self.figure.add_subplot(grid[4, 1], sharex=self.ax_alignment)
        self.ax_inertia_twist_speed = self.ax_inertia.twinx()

        self.ax_3d.set_title("Animation 3D")
        self.ax_alignment.set_title("Angle(moment cinetique, axe longitudinal)")
        self.ax_rotation.set_title("Vrille et salto")
        self.ax_trunk.set_title("3 DoF du tronc")
        self.ax_torque.set_title("Efforts du tronc")
        self.ax_inertia.set_title("Inertie app. en vrille")

        self.ax_alignment.set_ylabel("deg")
        self.ax_rotation.set_ylabel("Vrille (deg)")
        self.ax_rotation_salto.set_ylabel("Salto (deg)")
        self.ax_trunk.set_ylabel("deg")
        self.ax_torque.set_ylabel("N.m")
        self.ax_inertia.set_ylabel("kg.m^2")
        self.ax_inertia_twist_speed.set_ylabel("ω_vrille (deg/s)")
        self.ax_inertia.set_xlabel("Temps (s)")
        for axis in (
            self.ax_alignment,
            self.ax_rotation,
            self.ax_trunk,
            self.ax_torque,
        ):
            axis.tick_params(axis="x", labelbottom=False)

        self.ax_3d.set_xlabel("x")
        self.ax_3d.set_ylabel("y")
        self.ax_3d.set_zlabel("z")
        self.ax_3d.view_init(elev=18, azim=-70, roll=0.0, vertical_axis="z")
        self._style_axes()

    def _style_axes(self) -> None:
        """Apply a consistent visual treatment to the plot area."""

        primary_axes = (
            self.ax_alignment,
            self.ax_rotation,
            self.ax_trunk,
            self.ax_torque,
            self.ax_inertia,
        )
        for axis in primary_axes:
            axis.set_facecolor(self.PANEL_BACKGROUND)
            axis.grid(True, color="#D7DEE7", linewidth=0.8, alpha=0.72)
            axis.set_axisbelow(True)
            axis.tick_params(labelsize=9.5, colors="#344054")
            axis.title.set_fontsize(11.5)
            axis.title.set_fontweight("semibold")
            axis.title.set_color(self.TEXT_PRIMARY)
            for spine in axis.spines.values():
                spine.set_color("#C7D0DA")
                spine.set_linewidth(0.9)

        for axis in (self.ax_rotation_salto, self.ax_inertia_twist_speed):
            axis.tick_params(labelsize=9.5, colors="#344054", pad=6)
            for side in ("right", "top"):
                axis.spines[side].set_color("#C7D0DA")
                axis.spines[side].set_linewidth(0.9)

        self.ax_rotation.set_ylabel("Vrille (deg)", color="#FF7F0E")
        self.ax_rotation.tick_params(axis="y", colors="#FF7F0E")
        self.ax_rotation_salto.set_ylabel("Salto (deg)", color="#1F77B4", labelpad=8)
        self.ax_rotation_salto.tick_params(axis="y", colors="#1F77B4")
        self.ax_inertia.set_ylabel("kg.m^2", color="#2CA02C")
        self.ax_inertia.tick_params(axis="y", colors="#2CA02C")
        self.ax_inertia_twist_speed.set_ylabel("ω_vrille (deg/s)", color="#8C564B", labelpad=8)
        self.ax_inertia_twist_speed.tick_params(axis="y", colors="#8C564B")

        self.ax_3d.set_facecolor(self.PANEL_BACKGROUND)
        self.ax_3d.title.set_fontsize(12.0)
        self.ax_3d.title.set_fontweight("semibold")
        self.ax_3d.title.set_color(self.TEXT_PRIMARY)
        for axis in (self.ax_3d.xaxis, self.ax_3d.yaxis, self.ax_3d.zaxis):
            axis.set_pane_color((1.0, 1.0, 1.0, 0.98))
            axis.line.set_color("#BBC5D0")
        self.ax_3d.tick_params(labelsize=9.5, colors="#344054")

    def _build_controls(self) -> None:
        """Create sliders, buttons, and checkboxes."""

        self.control_panels: dict[str, plt.Axes] = {}
        self.control_section_titles = []

        def make_panel(name: str, rect: list[float], title: str):
            panel = self.figure.add_axes(rect)
            panel.set_facecolor(self.PANEL_BACKGROUND)
            panel.set_xticks([])
            panel.set_yticks([])
            panel.set_xlim(0.0, 1.0)
            panel.set_ylim(0.0, 1.0)
            for spine in panel.spines.values():
                spine.set_edgecolor(self.PANEL_BORDER)
                spine.set_linewidth(1.15)
            title_artist = panel.text(
                0.03,
                0.90,
                title,
                fontsize=11.2,
                fontweight="bold",
                color=self.TEXT_PRIMARY,
                va="top",
            )
            panel.axhline(0.82, xmin=0.03, xmax=0.97, color="#EEF2F6", linewidth=1.0)
            self.control_panels[name] = panel
            self.control_section_titles.append(title_artist)

        make_panel("momentum", [0.04, 0.03, 0.30, 0.21], "Moment cinetique global")
        make_panel("flight", [0.355, 0.03, 0.30, 0.21], "Vol et posture")
        make_panel("modes", [0.67, 0.03, 0.30, 0.21], "Modes et lecture")
        make_panel("time", [0.04, 0.248, 0.57, 0.068], "Navigation temporelle")
        self.playback_text_artist = self.control_panels["modes"].text(
            0.97,
            0.965,
            "",
            fontsize=8.9,
            color=self.TEXT_SECONDARY,
            ha="right",
            va="top",
        )

        slider_specs = [
            (
                "salto_rps",
                [0.095, 0.185, 0.205, 0.0105],
                "Hx eq. (rot/s)",
                0.0,
                0.25,
                0.0,
            ),
            ("tilt_rps", [0.095, 0.136, 0.205, 0.0105], "Hy eq. (rot/s)", -2.0, 2.0, 0.0),
            ("twist_rps", [0.095, 0.087, 0.205, 0.0105], "Hz eq. (rot/s)", -4.0, 6.0, 3.0),
            (
                "backward_velocity",
                [0.425, 0.185, 0.185, 0.0105],
                "V_arr (m/s)",
                1.0,
                5.0,
                self.DEFAULT_BACKWARD_VELOCITY_M_S,
            ),
            (
                "flight_time",
                [0.425, 0.136, 0.185, 0.0105],
                "Temps de vol (s)",
                0.4,
                0.8,
                self.simulator.flight_time_from_takeoff_velocity(
                    self.parameters.takeoff_vertical_velocity
                ),
            ),
            (
                "somersault_tilt",
                [0.425, 0.087, 0.185, 0.0105],
                "Incl. salto (deg)",
                0.0,
                30.0,
                0.0,
            ),
            (
                "inward_tilt",
                [0.425, 0.038, 0.185, 0.0105],
                "Incl. int. (deg)",
                -30.0,
                30.0,
                0.0,
            ),
        ]

        self.sliders: dict[str, Slider] = {}
        for name, rect, label, vmin, vmax, initial in slider_specs:
            axis = self.figure.add_axes(rect)
            slider = Slider(axis, label=label, valmin=vmin, valmax=vmax, valinit=initial)
            self._style_slider(slider)
            slider.on_changed(self._on_parameter_change)
            self.sliders[name] = slider

        retune_pd_axis = self.figure.add_axes([0.685, 0.157, 0.110, 0.042])
        self.retune_pd_button = Button(retune_pd_axis, "Retuner PD")
        self._style_button(self.retune_pd_button, fill="#FFFFFF", edge="#B9C5D3")
        self.retune_pd_button.on_clicked(self._retune_pd_controller)

        reset_axis = self.figure.add_axes([0.815, 0.157, 0.110, 0.042])
        self.reset_button = Button(reset_axis, "Reset")
        self._style_button(
            self.reset_button,
            fill="#FFF6F3",
            edge="#E7B7A8",
            text_color="#9A4024",
        )
        self.reset_button.on_clicked(self._reset_controls)

        pause_axis = self.figure.add_axes([0.065, 0.266, 0.080, 0.042])
        self.pause_button = Button(pause_axis, "Pause")
        self._style_button(
            self.pause_button,
            fill="#E8F1FF",
            edge="#8CB5F4",
            text_color="#184B94",
        )
        self.pause_button.on_clicked(self._toggle_pause)

        speed_button_axis = self.figure.add_axes([0.154, 0.266, 0.094, 0.042])
        self.speed_button = Button(speed_button_axis, "Vit. 100%")
        self._style_button(self.speed_button, fill="#FFFFFF", edge="#B9C5D3")
        self.speed_button.on_clicked(self._toggle_playback_menu)

        checkbox_axis = self.figure.add_axes([0.71, 0.055, 0.24, 0.088])
        checkbox_axis.set_facecolor("none")
        for spine in checkbox_axis.spines.values():
            spine.set_visible(False)
        self.stabilization_checkbox = SafeCheckButtons(
            checkbox_axis,
            labels=[
                "Stabiliser le tronc",
                "Avatar de face (vrille=0)",
                self.TWIST_OPTIMIZATION_LABEL,
                self.ALIGNMENT_OPTIMIZATION_LABEL,
            ],
            actives=[False, False, False, False],
        )
        self._style_checkboxes(self.stabilization_checkbox)
        self.stabilization_checkbox.on_clicked(self._on_parameter_change)

        playback_axis = self.figure.add_axes([0.154, 0.202, 0.094, 0.060])
        self.playback_selector = SafeRadioButtons(
            playback_axis,
            labels=("100%", "50%", "25%"),
            active=0,
        )
        self.playback_selector.on_clicked(self._on_playback_change)
        playback_axis.set_facecolor(self.PANEL_BACKGROUND)
        for spine in playback_axis.spines.values():
            spine.set_edgecolor(self.PANEL_BORDER)
            spine.set_linewidth(1.0)
        self._style_radio_buttons(self.playback_selector)
        playback_axis.set_visible(False)
        for child in playback_axis.get_children():
            child.set_visible(False)
        self.playback_menu_axis = playback_axis

        time_axis = self.figure.add_axes([0.265, 0.276, 0.305, 0.0105])
        self.time_slider = Slider(
            time_axis,
            label="Temps (s)",
            valmin=0.0,
            valmax=max(self.result.flight_time, 1e-6),
            valinit=0.0,
        )
        self._style_slider(self.time_slider)
        self.time_slider.on_changed(self._on_time_slider_change)

    def _style_slider(self, slider: Slider) -> None:
        """Apply a more legible slider styling."""

        slider.ax.set_facecolor("none")
        slider.label.set_fontsize(8.5)
        slider.label.set_color(self.TEXT_PRIMARY)
        slider.label.set_ha("right")
        slider.label.set_x(-0.02)
        slider.valtext.set_fontsize(8.5)
        slider.valtext.set_color(self.TEXT_SECONDARY)
        if hasattr(slider, "track"):
            slider.track.set_color("#D9E1E8")
        if hasattr(slider, "poly"):
            slider.poly.set_facecolor(self.PRIMARY_BLUE)
            slider.poly.set_alpha(0.92)
        if hasattr(slider, "vline"):
            slider.vline.set_visible(False)
        handle = getattr(slider, "_handle", None)
        if handle is not None:
            handle.set_markerfacecolor("#FFFFFF")
            handle.set_markeredgecolor(self.PRIMARY_BLUE)
            handle.set_markeredgewidth(1.3)
            handle.set_markersize(6.5)
        for spine in slider.ax.spines.values():
            spine.set_visible(False)

    def _style_button(
        self,
        button: Button,
        *,
        fill: str,
        edge: str,
        text_color: str = "#14212B",
    ) -> None:
        """Apply a cleaner visual treatment to a button."""

        button.ax.set_facecolor(fill)
        for spine in button.ax.spines.values():
            spine.set_edgecolor(edge)
            spine.set_linewidth(1.15)
        button.color = fill
        button.hovercolor = "#EDF3FA"
        button.label.set_fontsize(10.0)
        button.label.set_color(text_color)

    def _style_checkboxes(self, checkbox: SafeCheckButtons) -> None:
        """Make the checkbox group more legible and less cramped."""

        item_count = len(checkbox.labels)
        for label in checkbox.labels:
            label.set_fontsize(8.7)
            label.set_color(self.TEXT_PRIMARY)
            label.set_x(0.29)
        if hasattr(checkbox, "_frames"):
            checkbox._frames.set_sizes(np.full(item_count, 54.0))
            checkbox._frames.set_facecolors(np.tile([[1.0, 1.0, 1.0, 1.0]], (item_count, 1)))
            checkbox._frames.set_edgecolors(np.tile([[0.56, 0.63, 0.70, 1.0]], (item_count, 1)))
            checkbox._frames.set_linewidths(np.full(item_count, 1.0))
        if hasattr(checkbox, "_checks"):
            checkbox._checks.set_sizes(np.full(item_count, 60.0))
            checkbox._active_check_colors = np.tile([[0.18, 0.44, 0.73, 1.0]], (item_count, 1))
            current_status = list(checkbox.get_status())
            invisible = mcolors.to_rgba("none")
            facecolors = np.array(
                [
                    checkbox._active_check_colors[index] if current_status[index] else invisible
                    for index in range(item_count)
                ]
            )
            checkbox._checks.set_facecolor(facecolors)
            checkbox._checks.set_edgecolor(facecolors)
            checkbox._checks.set_linewidths(np.full(item_count, 1.4))

    def _style_radio_buttons(self, radio: SafeRadioButtons) -> None:
        """Style the playback-speed popup menu consistently with the control cards."""

        for label in radio.labels:
            label.set_fontsize(9.0)
            label.set_color(self.TEXT_PRIMARY)
        if hasattr(radio, "_buttons"):
            radio._buttons.set_sizes(np.full(len(radio.labels), 54.0))
            radio._buttons.set_edgecolors("#90A0B2")
        if hasattr(radio, "_active_colors"):
            radio._active_colors = [self.PRIMARY_BLUE for _ in radio.labels]

    def _build_plot_artists(self) -> None:
        """Initialize the lines that will be updated after each simulation."""

        self.skeleton_lines = []
        for _connection in skeleton_connections():
            (line,) = self.ax_3d.plot([], [], [], color="#0B3C5D", linewidth=2.0)
            self.skeleton_lines.append(line)

        self.ground_surface = None
        (self.body_axis_line,) = self.ax_3d.plot([], [], [], color="#2CA02C", linewidth=2.5)
        (self.angular_momentum_line,) = self.ax_3d.plot([], [], [], color="#D62728", linewidth=2.5)
        (self.com_trajectory_line,) = self.ax_3d.plot(
            [],
            [],
            [],
            color="#17BECF",
            linewidth=2.0,
            alpha=0.28,
        )
        (self.com_point,) = self.ax_3d.plot(
            [],
            [],
            [],
            marker="o",
            color="#17BECF",
            markersize=5.5,
            linestyle="None",
        )
        (self.precession_cone_line,) = self.ax_3d.plot(
            [],
            [],
            [],
            color="#9467BD",
            linewidth=1.8,
            linestyle="--",
        )

        (self.alignment_line,) = self.ax_alignment.plot([], [], color="#D62728", linewidth=2.0)
        (self.twist_line,) = self.ax_rotation.plot(
            [], [], label="Vrille", color="#FF7F0E", linewidth=2.0
        )
        (self.salto_line,) = self.ax_rotation_salto.plot(
            [], [], label="Salto", color="#1F77B4", linewidth=2.0
        )
        rotation_handles = [self.twist_line, self.salto_line]
        self.ax_rotation.legend(
            rotation_handles,
            [line.get_label() for line in rotation_handles],
            loc="upper left",
            frameon=True,
            framealpha=0.92,
            edgecolor=self.PANEL_BORDER,
            facecolor=self.PANEL_BACKGROUND,
            fontsize=8.8,
        )

        self.trunk_lines = [
            self.ax_trunk.plot([], [], label=label, linewidth=2.0)[0]
            for label in ("Tronc x", "Tronc y", "Tronc z")
        ]
        self.ax_trunk.legend(
            loc="upper left",
            frameon=True,
            framealpha=0.92,
            edgecolor=self.PANEL_BORDER,
            facecolor=self.PANEL_BACKGROUND,
            fontsize=8.8,
        )

        self.torque_lines = [
            self.ax_torque.plot([], [], label=label, linewidth=2.0)[0]
            for label in ("Couple x", "Couple y", "Couple z")
        ]
        self.ax_torque.legend(
            loc="upper left",
            frameon=True,
            framealpha=0.92,
            edgecolor=self.PANEL_BORDER,
            facecolor=self.PANEL_BACKGROUND,
            fontsize=8.8,
        )

        (self.twist_inertia_line,) = self.ax_inertia.plot(
            [],
            [],
            color="#2CA02C",
            linewidth=2.6,
            label="||σ|| / |ω_vrille|",
        )
        (self.twist_speed_line,) = self.ax_inertia_twist_speed.plot(
            [],
            [],
            color="#8C564B",
            linewidth=2.0,
            linestyle="--",
            label="ω_vrille",
        )
        inertia_handles = [self.twist_inertia_line, self.twist_speed_line]
        self.ax_inertia.legend(
            inertia_handles,
            [line.get_label() for line in inertia_handles],
            loc="upper left",
            frameon=True,
            framealpha=0.92,
            edgecolor=self.PANEL_BORDER,
            facecolor=self.PANEL_BACKGROUND,
            fontsize=8.8,
        )

        self.time_cursors = [
            axis.axvline(0.0, color="black", linestyle="--", linewidth=1.0)
            for axis in (
                self.ax_alignment,
                self.ax_rotation,
                self.ax_rotation_salto,
                self.ax_trunk,
                self.ax_torque,
                self.ax_inertia,
                self.ax_inertia_twist_speed,
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
            takeoff_vertical_velocity=self.simulator.takeoff_velocity_from_flight_time(
                self.sliders["flight_time"].val
            ),
            somersault_tilt_deg=self.sliders["somersault_tilt"].val,
            inward_tilt_deg=self.sliders["inward_tilt"].val,
            stabilize_trunk=self._stabilization_enabled(),
        )

    def _parameter_signature_for_tuning(
        self,
        parameters: FlightSimulationParameters,
    ) -> tuple[float, ...]:
        """Return a cache key describing the scenario that affects PD tuning."""

        return (
            *parameters.angular_velocity_rps,
            parameters.takeoff_vertical_velocity,
            parameters.backward_horizontal_velocity,
            parameters.somersault_tilt_deg,
            parameters.inward_tilt_deg,
            *parameters.initial_trunk_angles_deg,
            *parameters.initial_trunk_velocity_deg_s,
        )

    def _parameter_signature_for_inward_tilt_optimization(
        self,
        parameters: FlightSimulationParameters,
    ) -> tuple[float, ...]:
        """Return a cache key for the inward-tilt optimization mode."""

        return (
            *parameters.angular_velocity_rps,
            parameters.takeoff_vertical_velocity,
            parameters.backward_horizontal_velocity,
            parameters.somersault_tilt_deg,
            *parameters.initial_trunk_angles_deg,
            *parameters.initial_trunk_velocity_deg_s,
            float(parameters.stabilize_trunk),
            *parameters.controller.proportional_gains,
            *parameters.controller.derivative_gains,
        )

    def _parameter_signature_for_alignment_optimization(
        self,
        parameters: FlightSimulationParameters,
    ) -> tuple[float, ...]:
        """Return a cache key for the inward-tilt alignment-optimization mode."""

        return (
            *parameters.angular_velocity_rps,
            parameters.takeoff_vertical_velocity,
            parameters.backward_horizontal_velocity,
            parameters.somersault_tilt_deg,
            *parameters.initial_trunk_angles_deg,
            *parameters.initial_trunk_velocity_deg_s,
            float(parameters.stabilize_trunk),
            *parameters.controller.proportional_gains,
            *parameters.controller.derivative_gains,
        )

    def _apply_pd_tuning(
        self,
        parameters: FlightSimulationParameters,
    ) -> FlightSimulationParameters:
        """Return parameters updated with the auto-tuned PD controller when enabled."""

        if not parameters.stabilize_trunk:
            self.optimization_result = None
            return parameters

        signature = self._parameter_signature_for_tuning(parameters)
        optimization = self._pd_tuning_cache.get(signature)
        if optimization is None:
            optimization = self.simulator.tune_trunk_controller(
                parameters,
                max_iterations=20,
                optimization_sample_count=61,
            )
            self._pd_tuning_cache[signature] = optimization
        self.optimization_result = optimization
        self.optimization_result_signature = signature
        self._save_pd_tuning_cache(optimization.controller)
        return replace(parameters, controller=optimization.controller)

    def _invalidate_pd_optimization_if_stale(
        self,
        parameters: FlightSimulationParameters,
    ) -> None:
        """Clear the displayed PD-optimization result when the scenario has changed."""

        if not parameters.stabilize_trunk:
            self.optimization_result = None
            self.optimization_result_signature = None
            return
        current_signature = self._parameter_signature_for_tuning(parameters)
        if self.optimization_result_signature != current_signature:
            self.optimization_result = None
            self.optimization_result_signature = None

    def _set_inward_tilt_slider_value(self, value: float) -> None:
        """Update the inward-tilt slider without recursively re-triggering optimization."""

        self._updating_inward_tilt_slider = True
        try:
            self.sliders["inward_tilt"].set_val(value)
        finally:
            self._updating_inward_tilt_slider = False

    def _synchronize_optimization_mode_checkboxes(self, changed_value) -> None:
        """Keep the two inward-tilt optimization modes mutually exclusive."""

        if self._updating_checkbox_state or not isinstance(changed_value, str):
            return
        if (
            changed_value == self.TWIST_OPTIMIZATION_LABEL
            and self._inward_tilt_optimization_enabled()
            and self._alignment_optimization_enabled()
        ):
            self._updating_checkbox_state = True
            try:
                self.stabilization_checkbox.set_active(self.ALIGNMENT_OPTIMIZATION_INDEX)
            finally:
                self._updating_checkbox_state = False
        elif (
            changed_value == self.ALIGNMENT_OPTIMIZATION_LABEL
            and self._inward_tilt_optimization_enabled()
            and self._alignment_optimization_enabled()
        ):
            self._updating_checkbox_state = True
            try:
                self.stabilization_checkbox.set_active(self.TWIST_OPTIMIZATION_INDEX)
            finally:
                self._updating_checkbox_state = False

    def _simulate_with_current_parameters(self) -> None:
        """Simulate the current scenario with the currently selected controller."""

        self._invalidate_pd_optimization_if_stale(self.parameters)
        if self._alignment_optimization_enabled():
            signature = self._parameter_signature_for_alignment_optimization(self.parameters)
            optimization = self._alignment_cache.get(signature)
            if optimization is None:
                optimization = self.simulator.optimize_inward_tilt_for_alignment(
                    self.parameters,
                    max_iterations=20,
                    optimization_sample_count=61,
                )
                self._alignment_cache[signature] = optimization
            self.alignment_optimization_result = optimization
            self.inward_tilt_optimization_result = None
            self.parameters = replace(self.parameters, inward_tilt_deg=optimization.inward_tilt_deg)
            self._set_inward_tilt_slider_value(optimization.inward_tilt_deg)
            self._invalidate_pd_optimization_if_stale(self.parameters)
        elif self._inward_tilt_optimization_enabled():
            signature = self._parameter_signature_for_inward_tilt_optimization(self.parameters)
            optimization = self._inward_tilt_cache.get(signature)
            if optimization is None:
                optimization = self.simulator.optimize_inward_tilt_for_twist(
                    self.parameters,
                    max_iterations=20,
                    optimization_sample_count=61,
                )
                self._inward_tilt_cache[signature] = optimization
            self.inward_tilt_optimization_result = optimization
            self.alignment_optimization_result = None
            self.parameters = replace(self.parameters, inward_tilt_deg=optimization.inward_tilt_deg)
            self._set_inward_tilt_slider_value(optimization.inward_tilt_deg)
            self._invalidate_pd_optimization_if_stale(self.parameters)
        else:
            self.inward_tilt_optimization_result = None
            self.alignment_optimization_result = None
        self.result = self.simulator.simulate(self.parameters)

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
                self.inward_tilt_optimization_result,
                self.alignment_optimization_result,
            )
        )
        self._update_animation_playback()
        self._update_time_slider_bounds()

        time = self.result.time
        self.com_trajectory_line.set_data(
            self.result.center_of_mass[:, 0],
            self.result.center_of_mass[:, 1],
        )
        self.com_trajectory_line.set_3d_properties(self.result.center_of_mass[:, 2])
        self.alignment_line.set_data(time, self.result.body_axis_alignment_deg)
        self.twist_line.set_data(time, np.rad2deg(self.result.twist_angle))
        self.salto_line.set_data(time, np.rad2deg(self.result.q[:, 3]))

        trunk_angles_deg = np.rad2deg(self.result.q[:, 6:9])
        trunk_torques = self.result.tau[:, 6:9]
        for line, values in zip(self.trunk_lines, trunk_angles_deg.T):
            line.set_data(time, values)
        for line, values in zip(self.torque_lines, trunk_torques.T):
            line.set_data(time, values)
        self.twist_inertia_line.set_data(
            time,
            self.result.twist_inertia_proxy,
        )
        self.twist_speed_line.set_data(time, np.rad2deg(self.result.twist_rotation_speed))

        self._autoscale_axis(self.ax_alignment, time, [self.result.body_axis_alignment_deg])
        self._autoscale_axis(
            self.ax_rotation,
            time,
            [np.rad2deg(self.result.twist_angle)],
        )
        self._autoscale_axis(self.ax_rotation_salto, time, [np.rad2deg(self.result.q[:, 3])])
        self._autoscale_axis(self.ax_trunk, time, [column for column in trunk_angles_deg.T])
        self._autoscale_axis(self.ax_torque, time, [column for column in trunk_torques.T])
        self._autoscale_axis(
            self.ax_inertia,
            time,
            [self.result.twist_inertia_proxy[np.isfinite(self.result.twist_inertia_proxy)]],
        )
        self._autoscale_axis(
            self.ax_inertia_twist_speed,
            time,
            [np.rad2deg(self.result.twist_rotation_speed)],
        )

        self._set_3d_bounds()
        self._apply_view_mode()
        self._draw_frame(self.frame_index)
        self.figure.canvas.draw_idle()

    def _on_parameter_change(self, _value) -> None:
        """Re-simulate after a slider or checkbox update."""

        if self._updating_inward_tilt_slider:
            return
        self._synchronize_optimization_mode_checkboxes(_value)
        self.parameters = self._collect_parameters()
        self._simulate_with_current_parameters()
        self._refresh_from_result(reset_animation=True)

    def _toggle_pause(self, _event) -> None:
        """Pause or resume the 3D animation."""

        self.is_paused = not self.is_paused
        self.pause_button.label.set_text("Play" if self.is_paused else "Pause")
        self.figure.canvas.draw_idle()

    def _retune_pd_controller(self, _event) -> None:
        """Tune the trunk PD controller explicitly for the current scenario."""

        self.parameters = self._collect_parameters()
        if not self.parameters.stabilize_trunk:
            self.optimization_result = None
            self.optimization_result_signature = None
            self.figure.canvas.draw_idle()
            return

        self.parameters = self._apply_pd_tuning(self.parameters)
        self.result = self.simulator.simulate(self.parameters)
        self._refresh_from_result(reset_animation=True)

    def _reset_controls(self, _event) -> None:
        """Restore the default slider values."""

        defaults = FlightSimulationParameters()
        self.parameters = defaults
        self.optimization_result = None
        self.optimization_result_signature = None
        self.inward_tilt_optimization_result = None
        self.alignment_optimization_result = None
        self.sliders["salto_rps"].set_val(defaults.angular_velocity_rps[0])
        self.sliders["tilt_rps"].set_val(defaults.angular_velocity_rps[1])
        self.sliders["twist_rps"].set_val(defaults.angular_velocity_rps[2])
        self.sliders["backward_velocity"].set_val(self.DEFAULT_BACKWARD_VELOCITY_M_S)
        self.sliders["flight_time"].set_val(
            self.simulator.flight_time_from_takeoff_velocity(defaults.takeoff_vertical_velocity)
        )
        self.sliders["somersault_tilt"].set_val(defaults.somersault_tilt_deg)
        self.sliders["inward_tilt"].set_val(defaults.inward_tilt_deg)
        for index, active in enumerate(self.stabilization_checkbox.get_status()):
            if active:
                self.stabilization_checkbox.set_active(index)
        if self.playback_selector.value_selected != "100%":
            self.playback_selector.set_active(0)
        if self.playback_menu_visible:
            self._set_playback_menu_visible(False)

    def _on_playback_change(self, label: str) -> None:
        """Update the playback speed from the speed selector."""

        speed_map = {"100%": 1.0, "50%": 0.5, "25%": 0.25}
        self.animation_speed_fraction = speed_map[label]
        self.speed_button.label.set_text(f"Vit. {label}")
        self._update_animation_playback()
        self._set_playback_menu_visible(False)
        self.figure.canvas.draw_idle()

    def _toggle_playback_menu(self, _event) -> None:
        """Show or hide the playback-speed popup menu."""

        self._set_playback_menu_visible(not self.playback_menu_visible)
        self.figure.canvas.draw_idle()

    def _set_playback_menu_visible(self, visible: bool) -> None:
        """Set the visibility of the playback popup menu."""

        self.playback_menu_visible = visible
        self.playback_menu_axis.set_visible(visible)
        for child in self.playback_menu_axis.get_children():
            child.set_visible(visible)

    def _update_time_slider_bounds(self) -> None:
        """Synchronize the time slider bounds with the current simulation."""

        self.time_slider.valmin = 0.0
        self.time_slider.valmax = max(self.result.flight_time, 1e-6)
        self.time_slider.ax.set_xlim(self.time_slider.valmin, self.time_slider.valmax)
        self._set_time_slider_value(self.result.time[self.frame_index])

    def _set_3d_bounds(self) -> None:
        """Set an equal 3D view box that contains the full animated trajectory."""

        marker_cloud = self.result.markers.reshape(-1, 3)
        marker_cloud = np.vstack((marker_cloud, self.result.center_of_mass))
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
        axis_history = self._display_body_axis_history(frame_index)
        cone_tip_history = pelvis + 0.35 * axis_history
        self.precession_cone_line.set_data(cone_tip_history[:, 0], cone_tip_history[:, 1])
        self.precession_cone_line.set_3d_properties(cone_tip_history[:, 2])

        angular_momentum = self.result.angular_momentum[frame_index]
        if np.linalg.norm(angular_momentum) > 0.0:
            angular_tip = pelvis + 0.08 * angular_momentum
        else:
            angular_tip = pelvis
        self.angular_momentum_line.set_data(
            [pelvis[0], angular_tip[0]], [pelvis[1], angular_tip[1]]
        )
        self.angular_momentum_line.set_3d_properties([pelvis[2], angular_tip[2]])
        current_com = self.result.center_of_mass[frame_index]
        self.com_point.set_data([current_com[0]], [current_com[1]])
        self.com_point.set_3d_properties([current_com[2]])

        for cursor in self.time_cursors:
            cursor.set_xdata([time, time])
        self._set_time_slider_value(time)

    def _animate(self, _frame: int):
        """Advance the animation if it is currently playing."""

        if not self.is_paused:
            last_frame_index = max(len(self.result.time) - 1, 0)
            if self.frame_index >= last_frame_index:
                self.is_paused = True
                self.pause_button.label.set_text("Play")
                self.figure.canvas.draw_idle()
                return []

            self.frame_index = min(
                self.frame_index + self.frames_per_animation_step,
                last_frame_index,
            )
            self._draw_frame(self.frame_index)
            if self.frame_index >= last_frame_index:
                self.is_paused = True
                self.pause_button.label.set_text("Play")
                self.frame_index = 0
                self._draw_frame(self.frame_index)
            self.figure.canvas.draw_idle()
        return []

    def _set_time_slider_value(self, value: float) -> None:
        """Update the time slider without triggering a recursive redraw."""

        self._updating_time_slider = True
        try:
            self.time_slider.set_val(value)
        finally:
            self._updating_time_slider = False

    def _on_time_slider_change(self, value: float) -> None:
        """Move the animation frame according to the time slider."""

        if self._updating_time_slider:
            return
        self.is_paused = True
        self.pause_button.label.set_text("Play")
        self.frame_index = int(np.argmin(np.abs(self.result.time - float(value))))
        self._draw_frame(self.frame_index)
        self.figure.canvas.draw_idle()

    def _display_kinematics(self, frame_index: int) -> tuple[np.ndarray, np.ndarray]:
        """Return the marker cloud and longitudinal axis used for display."""

        if not self._face_view_enabled():
            return self.result.markers[frame_index], self.result.body_axis[frame_index]

        display_q = np.asarray(self.result.q[frame_index], dtype=float).copy()
        display_q[5] = 0.0
        display_markers = self.simulator.markers(display_q)
        display_body_axis = self.simulator.body_frame(display_q)[:, 2]
        return display_markers, display_body_axis

    def _display_body_axis_history(self, frame_index: int) -> np.ndarray:
        """Return the longitudinal-axis history used to display the precession cone."""

        if not self._face_view_enabled():
            return self.result.body_axis[: frame_index + 1]

        history = []
        for q_frame in self.result.q[: frame_index + 1]:
            display_q = np.asarray(q_frame, dtype=float).copy()
            display_q[5] = 0.0
            history.append(self.simulator.body_frame(display_q)[:, 2])
        return np.asarray(history, dtype=float)

    def _stabilization_enabled(self) -> bool:
        """Return whether trunk stabilization is enabled."""

        return self._checkbox_enabled(0)

    def _face_view_enabled(self) -> bool:
        """Return whether the front-view rendering mode is enabled."""

        return self._checkbox_enabled(1)

    def _inward_tilt_optimization_enabled(self) -> bool:
        """Return whether inward tilt should be optimized to maximize twist."""

        return self._checkbox_enabled(self.TWIST_OPTIMIZATION_INDEX)

    def _alignment_optimization_enabled(self) -> bool:
        """Return whether inward tilt should be optimized to align the body with `H`."""

        return self._checkbox_enabled(self.ALIGNMENT_OPTIMIZATION_INDEX)

    def _checkbox_enabled(self, index: int) -> bool:
        """Safely read one checkbox state even if Matplotlib reports a short status list."""

        status = list(self.stabilization_checkbox.get_status())
        return bool(status[index]) if index < len(status) else False

    def _apply_view_mode(self) -> None:
        """Apply the current 3D camera configuration."""

        elev, azim = self.FACE_VIEW if self._face_view_enabled() else self.DEFAULT_VIEW
        self.ax_3d.view_init(elev=elev, azim=azim, roll=0.0, vertical_axis="z")

    def _update_animation_playback(self) -> None:
        """Estimate playback duration and set the animation stepping accordingly."""

        self.animation_speed_fraction = max(self.animation_speed_fraction, 1e-6)
        total_frame_advances = max(len(self.result.time) - 1, 0)
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
            int(np.ceil(max(total_frame_advances, 1) / steps)),
        )
        animation_step_count = (
            int(np.ceil(total_frame_advances / self.frames_per_animation_step))
            if total_frame_advances > 0
            else 0
        )
        self.animation_duration_seconds = (
            animation_step_count * self.ANIMATION_TIMER_INTERVAL_MS / 1000.0
        )
        self.playback_text_artist.set_text(
            "Lect. "
            f"{int(round(self.animation_speed_fraction * 100.0))}% "
            f"| reel {self.result.flight_time:.2f} s "
            f"| anim {self.animation_duration_seconds:.2f} s"
        )
        if hasattr(self, "animation") and self.animation.event_source is not None:
            self.animation.event_source.interval = self.ANIMATION_TIMER_INTERVAL_MS

    @staticmethod
    def _autoscale_axis(ax, time: np.ndarray, series: list[np.ndarray]) -> None:
        """Autoscale one temporal axis around the provided traces."""

        ax.set_xlim(time[0], time[-1] if time[-1] > time[0] else time[0] + 1e-6)
        flattened = [
            np.asarray(values, dtype=float).ravel()
            for values in series
            if np.asarray(values, dtype=float).size > 0
        ]
        if not flattened:
            ax.set_ylim(-1.0, 1.0)
            ax.grid(True, alpha=0.3)
            return
        stacked = np.concatenate(flattened)
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
