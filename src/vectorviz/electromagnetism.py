"""Retarded vacuum fields of an ideal, localized harmonic electric dipole.

This is an eternal harmonic steady state, not a switched-on pulse. All three
observables are calculated in three dimensions in SI units; selecting a
two-dimensional invariant section is the caller's responsibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike
from scipy.constants import epsilon_0

from .core import FloatArray, VectorField, _as_points

SPEED_OF_LIGHT = 299792458.0
VACUUM_PERMITTIVITY = float(epsilon_0)
# Derive this constant so Maxwell's vacuum relation is consistent even when
# the independently rounded tabulated epsilon_0 and mu_0 differ by one ULP.
VACUUM_PERMEABILITY = 1.0 / (VACUUM_PERMITTIVITY * SPEED_OF_LIGHT**2)
ElectromagneticObservable = Literal["electric", "magnetic", "poynting"]
_OBSERVABLES = ("electric", "magnetic", "poynting")


def _finite_scalar(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real scalar.")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite real scalar.")
    return result


def _vector(value: ArrayLike, name: str) -> FloatArray:
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must contain real coordinates.")
    result = np.array(value, dtype=float, copy=True)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite vector with shape (3,).")
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class ElectromagneticSample:
    """Co-evaluated E (V/m), B (T), and instantaneous E cross B / mu0 (W/m²).

    Each array has the input points' ``(..., 3)`` shape. Source points,
    non-finite coordinates, or unrepresentable calculations return all-NaN
    vectors for all three observables at the affected point. Exact field
    zeros are finite zero vectors, never a missing direction filled in.
    """

    electric: FloatArray
    magnetic: FloatArray
    poynting: FloatArray


@dataclass(frozen=True, slots=True)
class OscillatingElectricDipole:
    r"""Full retarded point-dipole field with ``p(t) = moment cos(omega t + phase)``.

    ``moment`` is a three-vector in C·m, ``angular_frequency`` in rad/s,
    ``phase`` in rad, and ``center`` in m. Time is a finite real scalar in s,
    including negative times. The source has oscillated for all time. Setting
    omega to zero gives the static dipole ``moment * cos(phase)``.

    At displacement ``r*n`` the source is evaluated at ``t-r/c``. E includes
    the 1/r³ electrostatic, 1/r² induction and 1/r radiation terms; B includes
    induction and radiation. The formulas follow Feynman II §21-4, equations
    21.23 and 21.26-27, with outward n. No softening is applied. The source
    point remains singular even when the dipole amplitude is zero; callers
    should explicitly exclude a finite source neighbourhood when tracing.

    A frozen observable is a geometrical vector field. Its integral curves
    are neither particle paths nor photon trajectories. Instantaneous
    Poynting flux can point inward in the near region and is not cycle-averaged.
    """

    moment: ArrayLike = (0.0, 0.0, 1e-9)
    angular_frequency: float = 2.0 * np.pi * 1e8
    phase: float = 0.0
    center: ArrayLike = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        object.__setattr__(self, "moment", _vector(self.moment, "moment"))
        object.__setattr__(self, "center", _vector(self.center, "center"))
        for name in ("angular_frequency", "phase"):
            object.__setattr__(self, name, _finite_scalar(getattr(self, name), name))
        if self.angular_frequency < 0:
            raise ValueError("angular_frequency must be non-negative.")

    @property
    def dimension(self) -> int:
        """The common coordinate and vector dimension, always three."""
        return 3

    def evaluate(self, points: ArrayLike, time: float) -> ElectromagneticSample:
        """Evaluate all three SI observables at the same observation time.

        Arbitrary leading batch dimensions, including empty batches, are
        preserved. A non-finite intermediate/result invalidates the entire
        E/B/S sample at that point; there is no clipping or renormalization.
        """
        time = _finite_scalar(time, "time")
        if np.iscomplexobj(points):
            raise ValueError("points must contain real coordinates.")
        coordinates = _as_points(points, 3)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
            displacement = coordinates - self.center
            radius = np.hypot.reduce(displacement, axis=-1)
            inverse = np.reciprocal(radius)[..., None]
            normal = displacement * inverse
            if not np.any(self.moment):
                # A zero source needs neither an oscillatory phase nor inverse
                # powers of r. Its geometrical singularity is applied below.
                electric = np.zeros_like(coordinates)
                magnetic = np.zeros_like(coordinates)
            else:
                if self.angular_frequency == 0:
                    dipole = np.broadcast_to(self.moment * np.cos(self.phase), coordinates.shape)
                    derivative = np.zeros_like(coordinates)
                    second_derivative = np.zeros_like(coordinates)
                else:
                    phase = self.angular_frequency * (time - radius / SPEED_OF_LIGHT) + self.phase
                    dipole = np.cos(phase)[..., None] * self.moment
                    derivative = -np.sin(phase)[..., None] * self.moment * self.angular_frequency
                    second_derivative = -dipole * self.angular_frequency * self.angular_frequency
                radial_dipole = np.sum(normal * dipole, axis=-1)[..., None]
                radial_derivative = np.sum(normal * derivative, axis=-1)[..., None]
                radial_second = np.sum(normal * second_derivative, axis=-1)[..., None]
                near = (3 * normal * radial_dipole - dipole) * inverse * inverse * inverse
                induction = (3 * normal * radial_derivative - derivative) * inverse * inverse / SPEED_OF_LIGHT
                radiation = (normal * radial_second - second_derivative) * inverse / SPEED_OF_LIGHT**2
                electric = (near + induction + radiation) / (4 * np.pi * VACUUM_PERMITTIVITY)
                magnetic = np.cross(
                    derivative * inverse * inverse + second_derivative * inverse / SPEED_OF_LIGHT,
                    normal,
                ) * (VACUUM_PERMEABILITY / (4 * np.pi))
            poynting = np.cross(electric, magnetic) / VACUUM_PERMEABILITY
            valid = (
                np.all(np.isfinite(coordinates), axis=-1)
                & np.isfinite(radius) & (radius > 0)
                & np.all(np.isfinite(electric), axis=-1)
                & np.all(np.isfinite(magnetic), axis=-1)
                & np.all(np.isfinite(poynting), axis=-1)
            )
        electric[~valid] = np.nan
        magnetic[~valid] = np.nan
        poynting[~valid] = np.nan
        return ElectromagneticSample(electric, magnetic, poynting)

    def freeze(
        self, time: float, observable: ElectromagneticObservable = "electric",
    ) -> VectorField:
        """Return one 3D observable frozen at a validated observation time."""
        time = _finite_scalar(time, "time")
        if observable not in _OBSERVABLES:
            raise ValueError("observable must be electric, magnetic, or poynting.")
        return _FrozenElectromagneticField(self, time, observable)


@dataclass(frozen=True, slots=True)
class _FrozenElectromagneticField(VectorField):
    field: OscillatingElectricDipole
    time: float
    observable: ElectromagneticObservable

    @property
    def dimension(self) -> int:
        return 3

    def evaluate(self, points: ArrayLike) -> FloatArray:
        return getattr(self.field.evaluate(points, self.time), self.observable)


__all__ = [
    "SPEED_OF_LIGHT",
    "VACUUM_PERMEABILITY",
    "VACUUM_PERMITTIVITY",
    "ElectromagneticObservable",
    "ElectromagneticSample",
    "OscillatingElectricDipole",
]
