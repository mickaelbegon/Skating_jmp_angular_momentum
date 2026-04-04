"""Tests for the skater flight `bioMod` builder."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from skating_aerial_alignment.modeling import SkaterFlightBiomod


def test_biomod_declares_zero_gravity_and_expected_segments() -> None:
    """The serialized model matches the intended zero-gravity 9-DoF structure."""

    biomod = SkaterFlightBiomod()
    text = biomod.to_biomod_string()

    assert "gravity\t0.0\t0.0\t0.0" in text
    assert "segment\tpelvis" in text
    assert "segment\tthorax" in text
    assert "segment\thead" in text
    assert "marker\tpelvis_thorax_joint_center" in text
    assert "marker\tsternum" in text
    assert biomod.q_size() == 9


def test_principal_moments_are_positive_and_sorted() -> None:
    """The whole-body principal moments of inertia are physically meaningful."""

    moments = SkaterFlightBiomod().principal_moments()

    assert moments.shape == (3,)
    assert np.all(moments > 0.0)
    assert np.all(np.diff(moments) >= 0.0)


def test_equivalent_rotations_per_second_solves_inverse_inertia_problem() -> None:
    """The conversion from angular momentum to rotations per second is self-consistent."""

    biomod = SkaterFlightBiomod()
    angular_momentum = np.array([2.5, -1.0, 7.0])

    rotations_per_second = biomod.equivalent_rotations_per_second(angular_momentum)
    angular_velocity = rotations_per_second * 2.0 * np.pi
    reconstructed = biomod.inertia_tensor_body() @ angular_velocity

    assert np.allclose(reconstructed, angular_momentum)


def test_generated_biomod_loads_in_biorbd_when_available(tmp_path: Path) -> None:
    """The generated file can be instantiated as a biorbd model."""

    biorbd = pytest.importorskip("biorbd")
    biomod_path = SkaterFlightBiomod().write(tmp_path / "skater.bioMod")
    model = biorbd.Model(str(biomod_path))
    marker_names = [
        name.to_string() if hasattr(name, "to_string") else str(name)
        for name in model.markerNames()
    ]

    assert model.nbQ() == 9
    assert model.nbRoot() == 6
    assert "pelvis_thorax_joint_center" in marker_names
    assert "sternum" in marker_names


def test_neutral_pose_uses_a_compact_backspin_geometry(tmp_path: Path) -> None:
    """The default geometry keeps the wrists on the sternum and the legs slightly crossed."""

    biorbd = pytest.importorskip("biorbd")
    biomod_path = SkaterFlightBiomod().write(tmp_path / "skater_backspin.bioMod")
    model = biorbd.Model(str(biomod_path))
    marker_names = [
        name.to_string() if hasattr(name, "to_string") else str(name)
        for name in model.markerNames()
    ]
    marker_index = {name: index for index, name in enumerate(marker_names)}
    q = biorbd.GeneralizedCoordinates(np.zeros(model.nbQ()))
    markers = np.vstack([marker.to_array() for marker in model.markers(q)])

    sternum = markers[marker_index["sternum"]]
    wrist_left = markers[marker_index["wrist_left"]]
    wrist_right = markers[marker_index["wrist_right"]]
    ankle_left = markers[marker_index["ankle_left"]]
    ankle_right = markers[marker_index["ankle_right"]]
    toe_left = markers[marker_index["toe_left"]]
    toe_right = markers[marker_index["toe_right"]]
    expected_foot_length = SkaterFlightBiomod()._foot_length()
    shoulder_left = markers[marker_index["shoulder_left"]]
    shoulder_right = markers[marker_index["shoulder_right"]]
    elbow_left = markers[marker_index["elbow_left"]]
    elbow_right = markers[marker_index["elbow_right"]]
    hand_left = markers[marker_index["hand_left"]]
    hand_right = markers[marker_index["hand_right"]]

    left_upper_arm = elbow_left - shoulder_left
    right_upper_arm = elbow_right - shoulder_right
    left_elbow_angle_deg = np.rad2deg(
        np.arccos(
            np.clip(
                np.dot(shoulder_left - elbow_left, wrist_left - elbow_left)
                / (
                    np.linalg.norm(shoulder_left - elbow_left)
                    * np.linalg.norm(wrist_left - elbow_left)
                ),
                -1.0,
                1.0,
            )
        )
    )
    right_elbow_angle_deg = np.rad2deg(
        np.arccos(
            np.clip(
                np.dot(shoulder_right - elbow_right, wrist_right - elbow_right)
                / (
                    np.linalg.norm(shoulder_right - elbow_right)
                    * np.linalg.norm(wrist_right - elbow_right)
                ),
                -1.0,
                1.0,
            )
        )
    )
    left_elbow_flexion_deg = 180.0 - left_elbow_angle_deg
    right_elbow_flexion_deg = 180.0 - right_elbow_angle_deg
    left_adduction_deg = np.rad2deg(np.arctan2(abs(left_upper_arm[0]), abs(left_upper_arm[2])))
    right_adduction_deg = np.rad2deg(np.arctan2(abs(right_upper_arm[0]), abs(right_upper_arm[2])))

    assert abs(wrist_left[0]) < 0.09
    assert abs(wrist_right[0]) < 0.09
    assert np.linalg.norm(hand_left - sternum) < 0.08
    assert np.linalg.norm(hand_right - sternum) < 0.08
    assert np.linalg.norm(wrist_left - sternum) < 0.12
    assert np.linalg.norm(wrist_right - sternum) < 0.12
    assert elbow_left[2] < shoulder_left[2]
    assert elbow_right[2] < shoulder_right[2]
    assert left_elbow_flexion_deg == pytest.approx(110.0, abs=1e-3)
    assert right_elbow_flexion_deg == pytest.approx(110.0, abs=1e-3)
    assert left_adduction_deg == pytest.approx(20.0, abs=1.0)
    assert right_adduction_deg == pytest.approx(20.0, abs=1.0)
    assert ankle_left[0] < 0.0
    assert ankle_right[0] > 0.0
    assert toe_left[1] - ankle_left[1] == pytest.approx(expected_foot_length, abs=1e-5)
    assert toe_right[1] - ankle_right[1] == pytest.approx(expected_foot_length, abs=1e-5)
