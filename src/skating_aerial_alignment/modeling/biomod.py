"""Generation of the zero-gravity full-body `bioMod` used in the aerial simulation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from skating_aerial_alignment.anthropometry import (
    BodyDimensions,
    DeLevaSex,
    SegmentInertialParameters,
    de_leva_segment_table,
)

TRUNK_PELVIS_FRACTION = 0.35
TRUNK_THORAX_FRACTION = 0.65


@dataclass(frozen=True)
class SegmentDefinition:
    """Definition of one rigid segment in the reduced whole-body model."""

    name: str
    parent: str
    translation: tuple[float, float, float]
    center_of_mass: tuple[float, float, float]
    inertia: tuple[float, float, float]
    mass: float
    rotations: str | None = None
    translations: str | None = None
    ranges_q: tuple[tuple[float, float], ...] | None = None


def _diag_inertia(
    mass: float,
    length: float,
    radii: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Compute the diagonal inertia tensor from normalized radii of gyration."""

    return tuple(mass * (radius * length) ** 2 for radius in radii)


def _matrix_block(translation: tuple[float, float, float]) -> str:
    """Return a 4x4 homogeneous transform block with identity rotation."""

    tx, ty, tz = translation
    return (
        "\tRTinMatrix\t1\n"
        "\tRT\n"
        f"\t\t1.000000\t0.000000\t0.000000\t{tx:.6f}\n"
        f"\t\t0.000000\t1.000000\t0.000000\t{ty:.6f}\n"
        f"\t\t0.000000\t0.000000\t1.000000\t{tz:.6f}\n"
        "\t\t0.000000\t0.000000\t0.000000\t1.000000\n"
    )


def _segment_block(segment: SegmentDefinition) -> str:
    """Serialize one `bioMod` segment block."""

    out = [
        f"segment\t{segment.name}\n",
        f"\tparent\t{segment.parent}\n",
        _matrix_block(segment.translation),
    ]
    if segment.translations is not None:
        out.append(f"\ttranslations\t{segment.translations}\n")
    if segment.rotations is not None:
        out.append(f"\trotations\t{segment.rotations}\n")
    if segment.ranges_q:
        out.append("\trangesQ\n")
        out.extend(f"\t\t{low:.6f}\t{high:.6f}\n" for low, high in segment.ranges_q)
    out.append(f"\tmass\t{segment.mass:.6f}\n")
    out.append(
        "\tCenterOfMass\t"
        f"{segment.center_of_mass[0]:.6f}\t"
        f"{segment.center_of_mass[1]:.6f}\t"
        f"{segment.center_of_mass[2]:.6f}\n"
    )
    out.append("\tinertia\n")
    out.append(f"\t\t{segment.inertia[0]:.6f}\t0.000000\t0.000000\n")
    out.append(f"\t\t0.000000\t{segment.inertia[1]:.6f}\t0.000000\n")
    out.append(f"\t\t0.000000\t0.000000\t{segment.inertia[2]:.6f}\n")
    out.append("endsegment\n\n")
    return "".join(out)


def _marker_block(name: str, parent: str, position: tuple[float, float, float]) -> str:
    """Serialize one marker block."""

    return (
        f"marker\t{name}\n"
        f"\tparent\t{parent}\n"
        f"\tposition\t{position[0]:.6f}\t{position[1]:.6f}\t{position[2]:.6f}\n"
        "\ttechnical\t1\n"
        "\tanatomical\t0\n"
        "endmarker\n\n"
    )


@dataclass(frozen=True)
class SkaterFlightBiomod:
    """Builder for the 9-DoF whole-body skater flight model."""

    mass: float = 75.0
    height: float = 1.75
    sex: DeLevaSex = DeLevaSex.MALE

    @property
    def dimensions(self) -> BodyDimensions:
        """Return the nominal dimensions associated with the current height."""

        return BodyDimensions.from_height(self.height)

    def q_size(self) -> int:
        """Return the number of generalized coordinates of the reduced model."""

        return 9

    def _trunk_split(
        self,
        trunk_parameters: SegmentInertialParameters,
    ) -> tuple[tuple[float, tuple[float, float, float]], tuple[float, tuple[float, float, float]]]:
        """Split the De Leva trunk mass into pelvis and thorax rigid bodies."""

        dims = self.dimensions
        trunk_mass = self.mass * trunk_parameters.mass_fraction
        pelvis_mass = trunk_mass * TRUNK_PELVIS_FRACTION
        thorax_mass = trunk_mass * TRUNK_THORAX_FRACTION
        pelvis_inertia = _diag_inertia(
            pelvis_mass,
            dims.pelvis_height,
            trunk_parameters.radii_of_gyration,
        )
        thorax_inertia = _diag_inertia(
            thorax_mass,
            dims.thorax_height,
            trunk_parameters.radii_of_gyration,
        )
        return (pelvis_mass, pelvis_inertia), (thorax_mass, thorax_inertia)

    def segment_definitions(self) -> list[SegmentDefinition]:
        """Return the rigid segment list used by the reduced full-body model."""

        dims = self.dimensions
        table = de_leva_segment_table(self.sex)
        (pelvis_mass, pelvis_inertia), (thorax_mass, thorax_inertia) = self._trunk_split(
            table["trunk"]
        )

        head_mass = self.mass * table["head"].mass_fraction
        upper_arm_mass = self.mass * table["upper_arm"].mass_fraction
        forearm_mass = self.mass * table["forearm"].mass_fraction
        hand_mass = self.mass * table["hand"].mass_fraction
        thigh_mass = self.mass * table["thigh"].mass_fraction
        shank_mass = self.mass * table["shank"].mass_fraction
        foot_mass = self.mass * table["foot"].mass_fraction

        segments = [
            SegmentDefinition(
                name="pelvis",
                parent="base",
                translation=(0.0, 0.0, 0.0),
                translations="xyz",
                rotations="xyz",
                ranges_q=(
                    (-10.0, 10.0),
                    (-10.0, 10.0),
                    (-10.0, 10.0),
                    (-6.283185, 6.283185),
                    (-6.283185, 6.283185),
                    (-6.283185, 6.283185),
                ),
                mass=pelvis_mass,
                center_of_mass=(0.0, 0.0, 0.5 * dims.pelvis_height),
                inertia=pelvis_inertia,
            ),
            SegmentDefinition(
                name="thorax",
                parent="pelvis",
                translation=(0.0, 0.0, dims.pelvis_height),
                rotations="xyz",
                ranges_q=((-1.570796, 1.570796), (-1.570796, 1.570796), (-1.570796, 1.570796)),
                mass=thorax_mass,
                center_of_mass=(0.0, 0.0, 0.5 * dims.thorax_height),
                inertia=thorax_inertia,
            ),
            SegmentDefinition(
                name="head",
                parent="thorax",
                translation=(0.0, 0.0, dims.thorax_height),
                mass=head_mass,
                center_of_mass=(0.0, 0.0, table["head"].center_of_mass_fraction * dims.head_height),
                inertia=_diag_inertia(
                    head_mass,
                    dims.head_height,
                    table["head"].radii_of_gyration,
                ),
            ),
        ]

        for side, sign in (("left", 1.0), ("right", -1.0)):
            segments.extend(
                [
                    SegmentDefinition(
                        name=f"upper_arm_{side}",
                        parent="thorax",
                        translation=(sign * dims.shoulder_half_width, 0.0, dims.thorax_height),
                        mass=upper_arm_mass,
                        center_of_mass=(
                            0.0,
                            0.0,
                            -table["upper_arm"].center_of_mass_fraction * dims.upper_arm_length,
                        ),
                        inertia=_diag_inertia(
                            upper_arm_mass,
                            dims.upper_arm_length,
                            table["upper_arm"].radii_of_gyration,
                        ),
                    ),
                    SegmentDefinition(
                        name=f"forearm_{side}",
                        parent=f"upper_arm_{side}",
                        translation=(0.0, 0.0, -dims.upper_arm_length),
                        mass=forearm_mass,
                        center_of_mass=(
                            0.0,
                            0.0,
                            -table["forearm"].center_of_mass_fraction * dims.forearm_length,
                        ),
                        inertia=_diag_inertia(
                            forearm_mass,
                            dims.forearm_length,
                            table["forearm"].radii_of_gyration,
                        ),
                    ),
                    SegmentDefinition(
                        name=f"hand_{side}",
                        parent=f"forearm_{side}",
                        translation=(0.0, 0.0, -dims.forearm_length),
                        mass=hand_mass,
                        center_of_mass=(
                            0.0,
                            0.0,
                            -table["hand"].center_of_mass_fraction * dims.hand_length,
                        ),
                        inertia=_diag_inertia(
                            hand_mass,
                            dims.hand_length,
                            table["hand"].radii_of_gyration,
                        ),
                    ),
                    SegmentDefinition(
                        name=f"thigh_{side}",
                        parent="pelvis",
                        translation=(sign * dims.hip_half_width, 0.0, 0.0),
                        mass=thigh_mass,
                        center_of_mass=(
                            0.0,
                            0.0,
                            -table["thigh"].center_of_mass_fraction * dims.thigh_length,
                        ),
                        inertia=_diag_inertia(
                            thigh_mass,
                            dims.thigh_length,
                            table["thigh"].radii_of_gyration,
                        ),
                    ),
                    SegmentDefinition(
                        name=f"shank_{side}",
                        parent=f"thigh_{side}",
                        translation=(0.0, 0.0, -dims.thigh_length),
                        mass=shank_mass,
                        center_of_mass=(
                            0.0,
                            0.0,
                            -table["shank"].center_of_mass_fraction * dims.shank_length,
                        ),
                        inertia=_diag_inertia(
                            shank_mass,
                            dims.shank_length,
                            table["shank"].radii_of_gyration,
                        ),
                    ),
                    SegmentDefinition(
                        name=f"foot_{side}",
                        parent=f"shank_{side}",
                        translation=(0.0, 0.0, -dims.shank_length),
                        mass=foot_mass,
                        center_of_mass=(0.0, 0.5 * dims.foot_length, 0.0),
                        inertia=_diag_inertia(
                            foot_mass,
                            dims.foot_length,
                            table["foot"].radii_of_gyration,
                        ),
                    ),
                ]
            )

        return segments

    def inertia_tensor_body(self) -> np.ndarray:
        """Return the whole-system inertia tensor about the CoM in the neutral body frame."""

        segment_centers = {
            "pelvis": np.array([0.0, 0.0, 0.5 * self.dimensions.pelvis_height]),
            "thorax": np.array(
                [0.0, 0.0, self.dimensions.pelvis_height + 0.5 * self.dimensions.thorax_height]
            ),
            "head": np.array(
                [
                    0.0,
                    0.0,
                    self.dimensions.trunk_length
                    + de_leva_segment_table(self.sex)["head"].center_of_mass_fraction
                    * self.dimensions.head_height,
                ]
            ),
        }

        table = de_leva_segment_table(self.sex)
        dims = self.dimensions
        for side, sign in (("left", 1.0), ("right", -1.0)):
            segment_centers[f"upper_arm_{side}"] = np.array(
                [
                    sign * dims.shoulder_half_width,
                    0.0,
                    dims.trunk_length
                    - table["upper_arm"].center_of_mass_fraction * dims.upper_arm_length,
                ]
            )
            segment_centers[f"forearm_{side}"] = np.array(
                [
                    sign * dims.shoulder_half_width,
                    0.0,
                    dims.trunk_length
                    - dims.upper_arm_length
                    - table["forearm"].center_of_mass_fraction * dims.forearm_length,
                ]
            )
            segment_centers[f"hand_{side}"] = np.array(
                [
                    sign * dims.shoulder_half_width,
                    0.0,
                    dims.trunk_length
                    - dims.upper_arm_length
                    - dims.forearm_length
                    - table["hand"].center_of_mass_fraction * dims.hand_length,
                ]
            )
            segment_centers[f"thigh_{side}"] = np.array(
                [
                    sign * dims.hip_half_width,
                    0.0,
                    -table["thigh"].center_of_mass_fraction * dims.thigh_length,
                ]
            )
            segment_centers[f"shank_{side}"] = np.array(
                [
                    sign * dims.hip_half_width,
                    0.0,
                    -dims.thigh_length - table["shank"].center_of_mass_fraction * dims.shank_length,
                ]
            )
            segment_centers[f"foot_{side}"] = np.array(
                [
                    sign * dims.hip_half_width,
                    0.5 * dims.foot_length,
                    -dims.thigh_length - dims.shank_length,
                ]
            )

        total_mass = sum(segment.mass for segment in self.segment_definitions())
        whole_body_com = (
            sum(
                segment.mass * segment_centers[segment.name]
                for segment in self.segment_definitions()
            )
            / total_mass
        )

        inertia_tensor = np.zeros((3, 3), dtype=float)
        for segment in self.segment_definitions():
            local_inertia = np.diag(segment.inertia)
            displacement = segment_centers[segment.name] - whole_body_com
            inertia_tensor += local_inertia + segment.mass * (
                np.dot(displacement, displacement) * np.eye(3)
                - np.outer(displacement, displacement)
            )
        return inertia_tensor

    def principal_moments(self) -> np.ndarray:
        """Return the principal moments of inertia about the whole-body CoM."""

        eigenvalues = np.linalg.eigvalsh(self.inertia_tensor_body())
        return np.sort(np.asarray(eigenvalues, dtype=float))

    def equivalent_rotations_per_second(self, angular_momentum: np.ndarray) -> np.ndarray:
        """Convert angular momentum components into equivalent rotations per second."""

        angular_momentum = np.asarray(angular_momentum, dtype=float)
        if angular_momentum.shape != (3,):
            raise ValueError("The angular momentum vector must have shape (3,).")
        angular_velocity = np.linalg.solve(self.inertia_tensor_body(), angular_momentum)
        return angular_velocity / (2.0 * np.pi)

    def to_biomod_string(self) -> str:
        """Serialize the reduced whole-body model as a `bioMod` string."""

        parts = [
            "version 4\n\n",
            "gravity\t0.0\t0.0\t0.0\n\n",
            "// Global axes: x lateral, y forward, z vertical.\n",
            "// Root rotations: somersault, inward tilt, twist.\n\n",
        ]
        parts.extend(_segment_block(segment) for segment in self.segment_definitions())

        dims = self.dimensions
        parts.extend(
            [
                _marker_block("pelvis_origin", "pelvis", (0.0, 0.0, 0.0)),
                _marker_block(
                    "pelvis_thorax_joint_center",
                    "pelvis",
                    (0.0, 0.0, dims.pelvis_height),
                ),
                _marker_block("thorax_top", "thorax", (0.0, 0.0, dims.thorax_height)),
                _marker_block("head_top", "head", (0.0, 0.0, dims.head_height)),
            ]
        )
        for side, sign in (("left", 1.0), ("right", -1.0)):
            parts.extend(
                [
                    _marker_block(
                        f"shoulder_{side}",
                        "thorax",
                        (sign * dims.shoulder_half_width, 0.0, dims.thorax_height),
                    ),
                    _marker_block(
                        f"elbow_{side}",
                        f"upper_arm_{side}",
                        (0.0, 0.0, -dims.upper_arm_length),
                    ),
                    _marker_block(
                        f"wrist_{side}",
                        f"forearm_{side}",
                        (0.0, 0.0, -dims.forearm_length),
                    ),
                    _marker_block(f"hand_{side}", f"hand_{side}", (0.0, 0.0, -dims.hand_length)),
                    _marker_block(f"hip_{side}", "pelvis", (sign * dims.hip_half_width, 0.0, 0.0)),
                    _marker_block(f"knee_{side}", f"thigh_{side}", (0.0, 0.0, -dims.thigh_length)),
                    _marker_block(f"ankle_{side}", f"shank_{side}", (0.0, 0.0, -dims.shank_length)),
                    _marker_block(f"toe_{side}", f"foot_{side}", (0.0, dims.foot_length, 0.0)),
                ]
            )
        return "".join(parts)

    def write(self, path: str | Path) -> Path:
        """Write the generated `bioMod` to disk and return its path."""

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.to_biomod_string(), encoding="utf-8")
        return target
