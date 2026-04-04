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

    assert model.nbQ() == 9
    assert model.nbRoot() == 6
