"""VTK-based 3D viewer for the skater aerial animation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import vtk

from skating_aerial_alignment.simulation import (
    FlightSimulationParameters,
    FlightSimulationResult,
    SkaterFlightSimulator,
)


def _normalized(vector: np.ndarray) -> np.ndarray:
    """Return a normalized vector, preserving zero vectors."""

    norm = float(np.linalg.norm(vector))
    if np.isclose(norm, 0.0):
        return np.zeros(3, dtype=float)
    return np.asarray(vector, dtype=float) / norm


def _orthonormal_basis(direction: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build an orthonormal basis whose third axis matches `direction`."""

    axis = _normalized(direction)
    if np.allclose(axis, 0.0):
        axis = np.array([0.0, 0.0, 1.0], dtype=float)
    reference = np.array([0.0, 0.0, 1.0], dtype=float)
    if abs(float(np.dot(axis, reference))) > 0.9:
        reference = np.array([1.0, 0.0, 0.0], dtype=float)
    basis_x = _normalized(np.cross(axis, reference))
    basis_y = _normalized(np.cross(axis, basis_x))
    return basis_x, basis_y, axis


def _matrix_from_axes(
    center: np.ndarray,
    basis_x: np.ndarray,
    basis_y: np.ndarray,
    basis_z: np.ndarray,
    scale: tuple[float, float, float],
) -> vtk.vtkMatrix4x4:
    """Return a homogeneous transform matrix for one oriented primitive."""

    matrix = vtk.vtkMatrix4x4()
    columns = (
        basis_x * scale[0],
        basis_y * scale[1],
        basis_z * scale[2],
    )
    for column_index, column in enumerate(columns):
        for row_index in range(3):
            matrix.SetElement(row_index, column_index, float(column[row_index]))
    for row_index in range(3):
        matrix.SetElement(row_index, 3, float(center[row_index]))
    matrix.SetElement(3, 0, 0.0)
    matrix.SetElement(3, 1, 0.0)
    matrix.SetElement(3, 2, 0.0)
    matrix.SetElement(3, 3, 1.0)
    return matrix


def _set_line_source(line_source: vtk.vtkLineSource, start: np.ndarray, end: np.ndarray) -> None:
    """Update the endpoints of one VTK line source."""

    line_source.SetPoint1(float(start[0]), float(start[1]), float(start[2]))
    line_source.SetPoint2(float(end[0]), float(end[1]), float(end[2]))
    line_source.Modified()


def _build_tube_actor(
    radius: float,
    color: tuple[float, float, float],
    *,
    sides: int = 18,
    opacity: float = 1.0,
) -> tuple[vtk.vtkLineSource, vtk.vtkActor]:
    """Create one tubular line actor."""

    line_source = vtk.vtkLineSource()
    tube = vtk.vtkTubeFilter()
    tube.SetInputConnection(line_source.GetOutputPort())
    tube.SetRadius(radius)
    tube.SetNumberOfSides(sides)
    tube.CappingOn()
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(tube.GetOutputPort())
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(*color)
    actor.GetProperty().SetOpacity(opacity)
    actor.GetProperty().SetInterpolationToPhong()
    actor.GetProperty().SetSpecular(0.18)
    actor.GetProperty().SetSpecularPower(14.0)
    return line_source, actor


def _build_ellipsoid_actor(
    color: tuple[float, float, float],
    *,
    theta_resolution: int = 28,
    phi_resolution: int = 20,
    opacity: float = 1.0,
) -> vtk.vtkActor:
    """Create one ellipsoid actor driven by a user matrix."""

    sphere = vtk.vtkSphereSource()
    sphere.SetRadius(1.0)
    sphere.SetThetaResolution(theta_resolution)
    sphere.SetPhiResolution(phi_resolution)
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(sphere.GetOutputPort())
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(*color)
    actor.GetProperty().SetOpacity(opacity)
    actor.GetProperty().SetInterpolationToPhong()
    actor.GetProperty().SetSpecular(0.16)
    actor.GetProperty().SetSpecularPower(16.0)
    return actor


def _build_polyline_actor(
    point_count: int,
    color: tuple[float, float, float],
    *,
    radius: float,
    opacity: float = 1.0,
) -> tuple[vtk.vtkPoints, vtk.vtkActor]:
    """Create one tubular polyline actor with a fixed number of points."""

    points = vtk.vtkPoints()
    points.SetNumberOfPoints(point_count)
    for point_index in range(point_count):
        points.SetPoint(point_index, 0.0, 0.0, 0.0)

    poly_line = vtk.vtkPolyLine()
    poly_line.GetPointIds().SetNumberOfIds(point_count)
    for point_index in range(point_count):
        poly_line.GetPointIds().SetId(point_index, point_index)

    cells = vtk.vtkCellArray()
    cells.InsertNextCell(poly_line)
    poly_data = vtk.vtkPolyData()
    poly_data.SetPoints(points)
    poly_data.SetLines(cells)

    tube = vtk.vtkTubeFilter()
    tube.SetInputData(poly_data)
    tube.SetRadius(radius)
    tube.SetNumberOfSides(16)
    tube.CappingOff()

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(tube.GetOutputPort())
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(*color)
    actor.GetProperty().SetOpacity(opacity)
    actor.GetProperty().SetInterpolationToPhong()
    return points, actor


def _set_polyline_points(points: vtk.vtkPoints, coordinates: np.ndarray) -> None:
    """Copy one polyline coordinate array into an existing VTK point buffer."""

    coordinates = np.asarray(coordinates, dtype=float)
    for point_index, point in enumerate(coordinates):
        points.SetPoint(point_index, float(point[0]), float(point[1]), float(point[2]))
    points.Modified()


@dataclass(frozen=True)
class TubeVisualSpec:
    """Geometry and color of one tubular body part."""

    start: str
    end: str
    radius: float
    color: tuple[float, float, float]
    opacity: float = 1.0


@dataclass(frozen=True)
class EllipsoidVisualSpec:
    """Geometry and color of one ellipsoidal body part."""

    name: str
    color: tuple[float, float, float]
    radii: tuple[float, float, float]
    opacity: float = 1.0


class VtkAvatarScene:
    """VTK scene that renders and updates the skater avatar frame by frame."""

    SUIT_COLOR = (0.196, 0.365, 0.533)
    SUIT_SHADOW = (0.122, 0.243, 0.353)
    BOOT_COLOR = (0.271, 0.353, 0.412)
    SKIN_COLOR = (0.910, 0.765, 0.643)
    BLADE_COLOR = (0.780, 0.831, 0.886)
    COM_COLOR = (0.090, 0.745, 0.812)
    BODY_AXIS_COLOR = (0.173, 0.627, 0.173)
    SIGMA_COLOR = (0.839, 0.153, 0.157)
    PRECESSION_COLOR = (0.580, 0.404, 0.741)

    def __init__(
        self,
        simulator: SkaterFlightSimulator,
        result: FlightSimulationResult,
        *,
        face_view: bool = False,
    ) -> None:
        """Create the reusable VTK actors for one simulation result."""

        self.simulator = simulator
        self.result = result
        self.face_view = face_view
        self.frame_index = 0
        self.actors: list[vtk.vtkActor] = []
        self.segment_sources: dict[str, vtk.vtkLineSource] = {}
        self.segment_actors: dict[str, vtk.vtkActor] = {}
        self.ellipsoid_actors: dict[str, vtk.vtkActor] = {}
        self.com_path_points: vtk.vtkPoints | None = None
        self.precession_points: vtk.vtkPoints | None = None
        self.body_axis_source: vtk.vtkLineSource | None = None
        self.sigma_source: vtk.vtkLineSource | None = None
        self.com_sphere_actor: vtk.vtkActor | None = None
        self._build_scene()
        self.update_frame(0)

    @property
    def tube_specs(self) -> list[TubeVisualSpec]:
        """Return the tube-based body parts used in the avatar."""

        return [
            TubeVisualSpec("pelvis_origin", "pelvis_thorax_joint_center", 0.120, self.SUIT_SHADOW),
            TubeVisualSpec("pelvis_thorax_joint_center", "thorax_top", 0.128, self.SUIT_COLOR),
            TubeVisualSpec("shoulder_left", "elbow_left", 0.050, self.SUIT_COLOR),
            TubeVisualSpec("elbow_left", "wrist_left", 0.040, self.SUIT_COLOR),
            TubeVisualSpec("wrist_left", "hand_left", 0.024, self.SKIN_COLOR),
            TubeVisualSpec("shoulder_right", "elbow_right", 0.050, self.SUIT_COLOR),
            TubeVisualSpec("elbow_right", "wrist_right", 0.040, self.SUIT_COLOR),
            TubeVisualSpec("wrist_right", "hand_right", 0.024, self.SKIN_COLOR),
            TubeVisualSpec("pelvis_origin", "hip_left", 0.072, self.SUIT_SHADOW),
            TubeVisualSpec("hip_left", "knee_left", 0.060, self.SUIT_SHADOW),
            TubeVisualSpec("knee_left", "ankle_left", 0.048, self.SUIT_SHADOW),
            TubeVisualSpec("ankle_left", "toe_left", 0.054, self.BOOT_COLOR),
            TubeVisualSpec("pelvis_origin", "hip_right", 0.072, self.SUIT_SHADOW),
            TubeVisualSpec("hip_right", "knee_right", 0.060, self.SUIT_SHADOW),
            TubeVisualSpec("knee_right", "ankle_right", 0.048, self.SUIT_SHADOW),
            TubeVisualSpec("ankle_right", "toe_right", 0.054, self.BOOT_COLOR),
        ]

    @property
    def ellipsoid_specs(self) -> list[EllipsoidVisualSpec]:
        """Return the ellipsoidal body parts used in the avatar."""

        return [
            EllipsoidVisualSpec("pelvis", self.SUIT_SHADOW, (0.17, 0.13, 0.14)),
            EllipsoidVisualSpec("thorax", self.SUIT_COLOR, (0.16, 0.12, 0.19)),
            EllipsoidVisualSpec("head", self.SKIN_COLOR, (0.088, 0.092, 0.115)),
            EllipsoidVisualSpec("shoulder_left", (0.608, 0.710, 0.800), (0.05, 0.048, 0.052)),
            EllipsoidVisualSpec("shoulder_right", (0.608, 0.710, 0.800), (0.05, 0.048, 0.052)),
            EllipsoidVisualSpec("hand_left", self.SKIN_COLOR, (0.03, 0.024, 0.035)),
            EllipsoidVisualSpec("hand_right", self.SKIN_COLOR, (0.03, 0.024, 0.035)),
        ]

    def _build_scene(self) -> None:
        """Create all persistent VTK actors."""

        for spec in self.tube_specs:
            line_source, actor = _build_tube_actor(spec.radius, spec.color, opacity=spec.opacity)
            key = f"{spec.start}->{spec.end}"
            self.segment_sources[key] = line_source
            self.segment_actors[key] = actor
            self.actors.append(actor)

        for spec in self.ellipsoid_specs:
            actor = _build_ellipsoid_actor(spec.color, opacity=spec.opacity)
            self.ellipsoid_actors[spec.name] = actor
            self.actors.append(actor)

        self.body_axis_source, body_axis_actor = _build_tube_actor(0.012, self.BODY_AXIS_COLOR)
        self.sigma_source, sigma_actor = _build_tube_actor(0.014, self.SIGMA_COLOR)
        self.actors.extend((body_axis_actor, sigma_actor))

        self.com_path_points, com_path_actor = _build_polyline_actor(
            self.result.center_of_mass.shape[0],
            self.COM_COLOR,
            radius=0.007,
            opacity=0.34,
        )
        self.precession_points, precession_actor = _build_polyline_actor(
            self.result.body_axis.shape[0],
            self.PRECESSION_COLOR,
            radius=0.006,
            opacity=0.52,
        )
        self.actors.extend((com_path_actor, precession_actor))

        com_sphere = _build_ellipsoid_actor(self.COM_COLOR, theta_resolution=18, phi_resolution=14)
        self.com_sphere_actor = com_sphere
        self.actors.append(com_sphere)

        _set_polyline_points(self.com_path_points, self.result.center_of_mass)

    def add_to_renderer(self, renderer: vtk.vtkRenderer) -> None:
        """Add all actors to the provided VTK renderer."""

        for actor in self.actors:
            renderer.AddActor(actor)

    def marker_cloud_for_frame(self, frame_index: int) -> tuple[np.ndarray, np.ndarray]:
        """Return the marker cloud and longitudinal axis used for display."""

        if not self.face_view:
            return self.result.markers[frame_index], self.result.body_axis[frame_index]

        display_q = np.asarray(self.result.q[frame_index], dtype=float).copy()
        display_q[5] = 0.0
        display_markers = self.simulator.markers(display_q)
        display_body_axis = self.simulator.body_frame(display_q)[:, 2]
        return display_markers, display_body_axis

    def body_axis_history(self, frame_index: int) -> np.ndarray:
        """Return the displayed body-axis history up to one frame."""

        if not self.face_view:
            return self.result.body_axis[: frame_index + 1]

        history = []
        for q_frame in self.result.q[: frame_index + 1]:
            display_q = np.asarray(q_frame, dtype=float).copy()
            display_q[5] = 0.0
            history.append(self.simulator.body_frame(display_q)[:, 2])
        return np.asarray(history, dtype=float)

    def _update_ellipsoids(self, markers: np.ndarray) -> None:
        """Update all ellipsoidal body parts from the current marker cloud."""

        pelvis_origin = markers[self.simulator.marker_index["pelvis_origin"]]
        trunk_joint = markers[self.simulator.marker_index["pelvis_thorax_joint_center"]]
        thorax_top = markers[self.simulator.marker_index["thorax_top"]]
        head_top = markers[self.simulator.marker_index["head_top"]]
        ellipsoid_data = {
            "pelvis": (
                0.55 * pelvis_origin + 0.45 * trunk_joint,
                trunk_joint - pelvis_origin,
            ),
            "thorax": (
                0.52 * thorax_top + 0.48 * trunk_joint,
                thorax_top - trunk_joint,
            ),
            "head": (
                0.68 * head_top + 0.32 * thorax_top,
                head_top - thorax_top,
            ),
            "shoulder_left": (
                markers[self.simulator.marker_index["shoulder_left"]],
                markers[self.simulator.marker_index["elbow_left"]]
                - markers[self.simulator.marker_index["shoulder_left"]],
            ),
            "shoulder_right": (
                markers[self.simulator.marker_index["shoulder_right"]],
                markers[self.simulator.marker_index["elbow_right"]]
                - markers[self.simulator.marker_index["shoulder_right"]],
            ),
            "hand_left": (
                markers[self.simulator.marker_index["hand_left"]],
                markers[self.simulator.marker_index["hand_left"]]
                - markers[self.simulator.marker_index["wrist_left"]],
            ),
            "hand_right": (
                markers[self.simulator.marker_index["hand_right"]],
                markers[self.simulator.marker_index["hand_right"]]
                - markers[self.simulator.marker_index["wrist_right"]],
            ),
        }
        for spec in self.ellipsoid_specs:
            center, direction = ellipsoid_data[spec.name]
            basis_x, basis_y, basis_z = _orthonormal_basis(direction)
            matrix = _matrix_from_axes(center, basis_x, basis_y, basis_z, spec.radii)
            self.ellipsoid_actors[spec.name].SetUserMatrix(matrix)

        for side in ("left", "right"):
            ankle = markers[self.simulator.marker_index[f"ankle_{side}"]]
            toe = markers[self.simulator.marker_index[f"toe_{side}"]]
            foot_axis = _normalized(toe - ankle)
            blade_start = ankle - np.array([0.0, 0.0, 0.022], dtype=float) - 0.03 * foot_axis
            blade_end = toe - np.array([0.0, 0.0, 0.022], dtype=float) + 0.06 * foot_axis
            line_source, actor = self.segment_sources.get(f"blade_{side}"), self.segment_actors.get(
                f"blade_{side}"
            )
            if line_source is None or actor is None:
                line_source, actor = _build_tube_actor(
                    0.010, self.BLADE_COLOR, sides=14, opacity=0.95
                )
                self.segment_sources[f"blade_{side}"] = line_source
                self.segment_actors[f"blade_{side}"] = actor
                self.actors.append(actor)
            _set_line_source(line_source, blade_start, blade_end)

    def update_frame(self, frame_index: int) -> None:
        """Update the full 3D scene to one animation frame."""

        self.frame_index = int(np.clip(frame_index, 0, len(self.result.time) - 1))
        markers, body_axis = self.marker_cloud_for_frame(self.frame_index)
        for spec in self.tube_specs:
            key = f"{spec.start}->{spec.end}"
            _set_line_source(
                self.segment_sources[key],
                markers[self.simulator.marker_index[spec.start]],
                markers[self.simulator.marker_index[spec.end]],
            )

        self._update_ellipsoids(markers)

        pelvis = markers[self.simulator.marker_index["pelvis_origin"]]
        _set_line_source(self.body_axis_source, pelvis, pelvis + 0.48 * body_axis)

        sigma = self.result.angular_momentum[self.frame_index]
        sigma_tip = pelvis + 0.08 * sigma if np.linalg.norm(sigma) > 0.0 else pelvis
        _set_line_source(self.sigma_source, pelvis, sigma_tip)

        history = self.body_axis_history(self.frame_index)
        cone_tip_history = pelvis[None, :] + 0.34 * history
        last_point = cone_tip_history[-1]
        padded_history = np.vstack(
            (
                cone_tip_history,
                np.repeat(
                    last_point[None, :], self.result.body_axis.shape[0] - history.shape[0], axis=0
                ),
            )
        )
        _set_polyline_points(self.precession_points, padded_history)

        current_com = self.result.center_of_mass[self.frame_index]
        com_matrix = _matrix_from_axes(
            current_com,
            np.array([1.0, 0.0, 0.0], dtype=float),
            np.array([0.0, 1.0, 0.0], dtype=float),
            np.array([0.0, 0.0, 1.0], dtype=float),
            (0.024, 0.024, 0.024),
        )
        self.com_sphere_actor.SetUserMatrix(com_matrix)


class VtkSkaterAnimator:
    """Standalone VTK render window and playback controller."""

    TIMER_INTERVAL_MS = 16
    DEFAULT_VIEW = (18.0, -70.0)
    FACE_VIEW = (10.0, 90.0)

    def __init__(
        self,
        simulator: SkaterFlightSimulator | None = None,
        parameters: FlightSimulationParameters | None = None,
        *,
        result: FlightSimulationResult | None = None,
        face_view: bool = False,
        offscreen: bool = False,
        render_on_update: bool = True,
    ) -> None:
        """Create one VTK render window around the simulated skater."""

        self.simulator = simulator or SkaterFlightSimulator()
        self.parameters = parameters or FlightSimulationParameters()
        self.result = result or self.simulator.simulate(self.parameters)
        self.scene = VtkAvatarScene(self.simulator, self.result, face_view=face_view)
        self.is_paused = False
        self.frame_cursor = 0.0
        self.playback_speed = 1.0
        self.render_on_update = render_on_update
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(0.953, 0.965, 0.976)
        self.render_window = vtk.vtkRenderWindow()
        self.render_window.AddRenderer(self.renderer)
        self.render_window.SetSize(1440, 900)
        if offscreen:
            self.render_window.SetOffScreenRendering(1)
        self.interactor = vtk.vtkRenderWindowInteractor()
        self.interactor.SetRenderWindow(self.render_window)
        self.scene.add_to_renderer(self.renderer)
        self._build_ground()
        self._build_text_overlay()
        self._configure_camera()
        self._update_overlay()
        self.interactor.AddObserver("TimerEvent", self._on_timer)
        self.interactor.AddObserver("KeyPressEvent", self._on_key_press)
        self.timer_id: int | None = None

    def _render_if_enabled(self) -> None:
        """Render the window when interactive rendering is enabled."""

        if self.render_on_update:
            self.render_window.Render()

    def _build_ground(self) -> None:
        """Create the ground plane under the avatar."""

        plane = vtk.vtkPlaneSource()
        plane.SetOrigin(-1.6, -1.6, 0.0)
        plane.SetPoint1(1.6, -1.6, 0.0)
        plane.SetPoint2(-1.6, 1.6, 0.0)
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(plane.GetOutputPort())
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(0.78, 0.70, 0.60)
        actor.GetProperty().SetOpacity(0.32)
        actor.GetProperty().SetInterpolationToPhong()
        self.renderer.AddActor(actor)

    def _build_text_overlay(self) -> None:
        """Create the playback/status overlay."""

        text_actor = vtk.vtkTextActor()
        text_actor.SetInput("")
        text_actor.SetDisplayPosition(18, 18)
        text_property = text_actor.GetTextProperty()
        text_property.SetFontSize(18)
        text_property.SetColor(0.08, 0.13, 0.17)
        text_property.SetBackgroundColor(1.0, 1.0, 1.0)
        text_property.SetBackgroundOpacity(0.66)
        self.text_actor = text_actor
        self.renderer.AddViewProp(text_actor)

    def _configure_camera(self) -> None:
        """Place the VTK camera around the animated scene."""

        marker_cloud = self.result.markers.reshape(-1, 3)
        marker_cloud = np.vstack((marker_cloud, self.result.center_of_mass))
        minima = marker_cloud.min(axis=0)
        maxima = marker_cloud.max(axis=0)
        center = 0.5 * (minima + maxima)
        radius = 1.05 * np.max(maxima - minima + 1e-6)
        camera = self.renderer.GetActiveCamera()
        elev, azim = self.FACE_VIEW if self.scene.face_view else self.DEFAULT_VIEW
        azim_rad = np.deg2rad(azim)
        elev_rad = np.deg2rad(elev)
        offset = np.array(
            [
                radius * np.cos(elev_rad) * np.cos(azim_rad),
                radius * np.cos(elev_rad) * np.sin(azim_rad),
                radius * np.sin(elev_rad),
            ],
            dtype=float,
        )
        focal_point = center + np.array([0.0, -0.15, 0.8], dtype=float)
        position = focal_point + offset
        camera.SetFocalPoint(*focal_point)
        camera.SetPosition(*position)
        camera.SetViewUp(0.0, 0.0, 1.0)
        self.renderer.ResetCameraClippingRange()

    def _update_overlay(self) -> None:
        """Refresh the on-screen playback status."""

        time = self.result.time[self.scene.frame_index]
        twist_turns = abs(
            float(self.result.twist_angle[self.scene.frame_index] - self.result.twist_angle[0])
        ) / (2.0 * np.pi)
        mode = "pause" if self.is_paused else "lecture"
        self.text_actor.SetInput(
            f"{mode} | t = {time:.3f} s | vrille = {twist_turns:.2f} tours | "
            f"vit. = {int(round(100 * self.playback_speed))}%"
        )

    def frame_step(self) -> float:
        """Return the frame-cursor increment applied at each timer tick."""

        if self.result.time.size <= 1 or np.isclose(self.result.flight_time, 0.0):
            return 1.0
        dt_data = self.result.flight_time / float(self.result.time.size - 1)
        return max(
            self.playback_speed * self.TIMER_INTERVAL_MS / 1000.0 / max(dt_data, 1e-9),
            0.05,
        )

    def set_playback_speed(self, speed: float) -> None:
        """Update the playback speed multiplier."""

        self.playback_speed = float(max(speed, 0.05))
        self._update_overlay()
        self._render_if_enabled()

    def toggle_face_view(self) -> None:
        """Toggle the front-view display mode."""

        self.scene.face_view = not self.scene.face_view
        self.scene.update_frame(self.scene.frame_index)
        self._configure_camera()
        self._update_overlay()
        self._render_if_enabled()

    def toggle_pause(self) -> None:
        """Toggle playback."""

        self.is_paused = not self.is_paused
        self._update_overlay()
        self._render_if_enabled()

    def reset(self) -> None:
        """Reset the animation to the take-off frame."""

        self.frame_cursor = 0.0
        self.scene.update_frame(0)
        self._update_overlay()
        self._render_if_enabled()

    def step_to(self, frame_index: int) -> None:
        """Move to one explicit frame index."""

        self.frame_cursor = float(np.clip(frame_index, 0, self.result.time.size - 1))
        self.scene.update_frame(int(round(self.frame_cursor)))
        self._update_overlay()
        self._render_if_enabled()

    def _on_timer(self, _obj, _event) -> None:
        """Advance the animation according to the playback speed."""

        if self.is_paused:
            return
        last_index = max(self.result.time.size - 1, 0)
        self.frame_cursor += self.frame_step()
        if self.frame_cursor >= last_index:
            self.frame_cursor = 0.0
            self.is_paused = True
        self.scene.update_frame(int(min(round(self.frame_cursor), last_index)))
        self._update_overlay()
        self._render_if_enabled()

    def _on_key_press(self, obj, _event) -> None:
        """Handle the keyboard shortcuts of the VTK player."""

        key = obj.GetKeySym().lower()
        if key == "space":
            self.toggle_pause()
        elif key == "right":
            self.is_paused = True
            self.step_to(min(self.scene.frame_index + 1, self.result.time.size - 1))
        elif key == "left":
            self.is_paused = True
            self.step_to(max(self.scene.frame_index - 1, 0))
        elif key in {"r", "home"}:
            self.is_paused = True
            self.reset()
        elif key == "f":
            self.toggle_face_view()
        elif key == "1":
            self.set_playback_speed(1.0)
        elif key == "2":
            self.set_playback_speed(0.5)
        elif key == "4":
            self.set_playback_speed(0.25)

    def start(self) -> "VtkSkaterAnimator":
        """Start the VTK animation window."""

        self.render_window.Render()
        self.interactor.Initialize()
        self.timer_id = self.interactor.CreateRepeatingTimer(self.TIMER_INTERVAL_MS)
        self.interactor.Start()
        return self


def launch_vtk_viewer(
    parameters: FlightSimulationParameters | None = None,
    *,
    face_view: bool = False,
) -> VtkSkaterAnimator:
    """Launch the VTK 3D viewer with one simulated skating jump."""

    animator = VtkSkaterAnimator(parameters=parameters, face_view=face_view)
    return animator.start()


def main() -> None:
    """CLI entry point for the VTK viewer."""

    launch_vtk_viewer()
