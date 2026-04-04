"""Anthropometric coefficients and nominal segment dimensions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class DeLevaSex(Enum):
    """Sex-specific coefficient set available in the De Leva table."""

    MALE = "male"
    FEMALE = "female"


@dataclass(frozen=True)
class SegmentInertialParameters:
    """Mass fraction and normalized inertial coefficients for one segment."""

    mass_fraction: float
    center_of_mass_fraction: float
    radii_of_gyration: tuple[float, float, float]


@dataclass(frozen=True)
class BodyDimensions:
    """Nominal whole-body dimensions expressed in meters."""

    height: float = 1.75
    shoulder_half_width: float = 0.19
    hip_half_width: float = 0.14
    pelvis_height: float = 0.18
    thorax_height: float = 0.44
    head_height: float = 0.24
    upper_arm_length: float = 0.29
    forearm_length: float = 0.26
    hand_length: float = 0.19
    thigh_length: float = 0.42
    shank_length: float = 0.43
    foot_length: float = 0.25

    @property
    def trunk_length(self) -> float:
        """Return the combined pelvis-to-neck distance."""

        return self.pelvis_height + self.thorax_height

    @classmethod
    def from_height(cls, height: float) -> "BodyDimensions":
        """Build a coherent reduced set of dimensions from body height."""

        return cls(
            height=height,
            shoulder_half_width=0.1086 * height,
            hip_half_width=0.0800 * height,
            pelvis_height=0.1029 * height,
            thorax_height=0.2514 * height,
            head_height=0.1371 * height,
            upper_arm_length=0.1860 * height,
            forearm_length=0.1463 * height,
            hand_length=0.1086 * height,
            thigh_length=0.2457 * height,
            shank_length=0.2514 * height,
            foot_length=0.1430 * height,
        )


_DE_LEVA_TABLE = {
    DeLevaSex.MALE: {
        "head": SegmentInertialParameters(0.0694, 0.5002, (0.303, 0.315, 0.261)),
        "trunk": SegmentInertialParameters(0.4346, 0.5138, (0.328, 0.306, 0.169)),
        "upper_arm": SegmentInertialParameters(0.0271, 0.4228, (0.285, 0.269, 0.158)),
        "forearm": SegmentInertialParameters(0.0162, 0.5426, (0.276, 0.265, 0.121)),
        "hand": SegmentInertialParameters(0.0061, 0.3624, (0.288, 0.235, 0.184)),
        "thigh": SegmentInertialParameters(0.1416, 0.4095, (0.329, 0.329, 0.149)),
        "shank": SegmentInertialParameters(0.0433, 0.4459, (0.255, 0.249, 0.103)),
        "foot": SegmentInertialParameters(0.0137, 0.4415, (0.257, 0.245, 0.124)),
    },
    DeLevaSex.FEMALE: {
        "head": SegmentInertialParameters(0.0669, 0.4841, (0.271, 0.295, 0.261)),
        "trunk": SegmentInertialParameters(0.4257, 0.4964, (0.307, 0.292, 0.147)),
        "upper_arm": SegmentInertialParameters(0.0255, 0.4246, (0.278, 0.260, 0.148)),
        "forearm": SegmentInertialParameters(0.0138, 0.5441, (0.261, 0.257, 0.094)),
        "hand": SegmentInertialParameters(0.0056, 0.3427, (0.244, 0.208, 0.184)),
        "thigh": SegmentInertialParameters(0.1478, 0.3612, (0.369, 0.364, 0.162)),
        "shank": SegmentInertialParameters(0.0481, 0.4416, (0.271, 0.267, 0.093)),
        "foot": SegmentInertialParameters(0.0129, 0.4014, (0.299, 0.279, 0.124)),
    },
}


def de_leva_segment_table(sex: DeLevaSex = DeLevaSex.MALE) -> dict[str, SegmentInertialParameters]:
    """Return the De Leva coefficients for the requested sex."""

    return _DE_LEVA_TABLE[sex].copy()
