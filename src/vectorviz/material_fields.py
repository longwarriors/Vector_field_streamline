"""Analytic fields at ideal, piecewise homogeneous material boundaries."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from .core import FloatArray, VectorField, _as_points
from .fields import _finite_scalar, _readonly


class MagneticShieldCylinderField(VectorField):
    r"""Transverse magnetic flux density of an infinitely long cylindrical shell.

    The cylinder axis is perpendicular to this two-dimensional coordinate
    plane. Its cavity and exterior are vacuum; the shell is linear,
    isotropic, and has constant finite ``relative_permeability >= 1``.
    ``applied_field`` is the uniform far-field magnetic flux density in tesla;
    the radii and ``center`` use metres. No free surface currents are present.

    The cavity field is uniform, with shielding factor
    :math:`S=1+(\mu_r-1)^2[1-(a/b)^2]/(4\mu_r)` for inner radius
    :math:`a` and outer radius :math:`b`. Normal B and tangential H are
    continuous at both interfaces, while tangential B generally jumps.
    A point exactly on an interface takes the radially outward-side value:
    the inner surface belongs to the shell and the outer one to the vacuum.
    Register both circles as tracer ``interfaces`` to resolve these jumps.

    This is a static boundary-value benchmark, not a model of end leakage,
    openings, hysteresis, saturation, finite length, or a specific alloy.
    """

    def __init__(
        self,
        applied_field: ArrayLike = (1.0e-6, 0.0),
        inner_radius: float = 0.7,
        outer_radius: float = 1.0,
        center: ArrayLike = (0.0, 0.0),
        *,
        relative_permeability: float = 100.0,
    ) -> None:
        applied = np.asarray(applied_field, dtype=float)
        if applied.shape != (2,) or not np.all(np.isfinite(applied)):
            raise ValueError("applied_field must be a finite vector of shape (2,).")
        if not np.any(applied != 0.0):
            raise ValueError("applied_field must be nonzero.")
        origin = np.asarray(center, dtype=float)
        if origin.shape != (2,) or not np.all(np.isfinite(origin)):
            raise ValueError("center must be a finite vector of shape (2,).")
        inner = _finite_scalar(inner_radius, "inner_radius")
        outer = _finite_scalar(outer_radius, "outer_radius")
        if not 0.0 < inner < outer:
            raise ValueError("radii must satisfy 0 < inner_radius < outer_radius.")
        permeability = _finite_scalar(relative_permeability, "relative_permeability")
        if permeability < 1.0:
            raise ValueError("relative_permeability must be at least 1.")

        self._applied = _readonly(applied)
        self._center = _readonly(origin)
        self._inner_radius = inner
        self._outer_radius = outer
        self._relative_permeability = permeability
        # Use reciprocal permeability and fractional thickness. Directly
        # squaring mu or subtracting 1 - (a/b)**2 loses extreme-mu/thin-wall
        # limits even when all model parameters are finite.
        reciprocal = 1.0 / permeability
        fraction = (outer - inner) / outer
        thickness = fraction * (2.0 - fraction)
        ratio_squared = (inner / outer) ** 2
        self._contrast = (1.0 - reciprocal) / (1.0 + reciprocal)
        self._one_minus_contrast = 2.0 * reciprocal / (1.0 + reciprocal)
        complement = 4.0 * reciprocal / (1.0 + reciprocal) ** 2
        denominator = thickness + complement * ratio_squared
        self._shell_factor = 2.0 / (1.0 + reciprocal) / denominator
        self._exterior_dipole = self._contrast * thickness / denominator
        self._exterior_tangent = (
            self._one_minus_contrast * thickness + complement * ratio_squared
        ) / denominator
        self._shielding_factor = 1.0 + (permeability / 4.0) * (1.0 - reciprocal) ** 2 * thickness
        self._interior_factor = 1.0 / self._shielding_factor

    @property
    def dimension(self) -> int:
        return 2

    @property
    def applied_field(self) -> FloatArray:
        return self._applied

    @property
    def center(self) -> FloatArray:
        return self._center

    @property
    def inner_radius(self) -> float:
        return self._inner_radius

    @property
    def outer_radius(self) -> float:
        return self._outer_radius

    @property
    def relative_permeability(self) -> float:
        return self._relative_permeability

    @property
    def shielding_factor(self) -> float:
        """Ratio of far-field magnitude to uniform cavity-field magnitude."""

        return self._shielding_factor

    @property
    def interior_field(self) -> FloatArray:
        """Uniform cavity magnetic flux density, in tesla."""

        return self._interior_factor * self._applied

    def _geometry(self, points: ArrayLike) -> tuple[FloatArray, FloatArray, FloatArray]:
        coordinates = _as_points(points, 2)
        displacement = coordinates - self._center
        radius = np.hypot(displacement[..., 0], displacement[..., 1])
        return coordinates, displacement, radius

    def _factors(self, radius: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Multipliers of the radial and tangential parts of the applied B."""

        radial = np.full_like(radius, np.nan)
        tangent = np.full_like(radius, np.nan)
        cavity = radius < self._inner_radius
        radial[cavity] = tangent[cavity] = self._interior_factor
        shell = (radius >= self._inner_radius) & (radius < self._outer_radius)
        shell_radius = radius[shell]
        inner_ratio = self._inner_radius / shell_radius
        inner_gap = (shell_radius - self._inner_radius) / shell_radius
        radial[shell] = self._shell_factor * (
            inner_gap * (1.0 + inner_ratio) + self._one_minus_contrast * inner_ratio**2
        )
        tangent[shell] = self._shell_factor * (1.0 + self._contrast * inner_ratio**2)
        exterior = (radius >= self._outer_radius) & np.isfinite(radius)
        exterior_radius = radius[exterior]
        outer_ratio = self._outer_radius / exterior_radius
        outer_gap = (exterior_radius - self._outer_radius) / exterior_radius
        radial[exterior] = 1.0 + self._exterior_dipole * outer_ratio**2
        tangent[exterior] = (
            outer_gap * (1.0 + outer_ratio) + self._exterior_tangent * outer_ratio**2
        )
        return radial, tangent

    def evaluate(self, points: ArrayLike) -> FloatArray:
        coordinates, displacement, radius = self._geometry(points)
        radial, tangent = self._factors(radius)
        # The centre has a uniform field and needs no radial direction.
        unit = np.divide(
            displacement,
            radius[..., None],
            out=np.zeros_like(coordinates),
            where=radius[..., None] != 0.0,
        )
        projection = np.sum(unit * self._applied, axis=-1)
        normal = projection[..., None] * unit
        return radial[..., None] * normal + tangent[..., None] * (self._applied - normal)

    def flux_function(self, points: ArrayLike) -> FloatArray:
        r"""Return continuous :math:`A_z` in T m, zero at the cylinder centre.

        :math:`\mathbf B=(\partial_y A_z,-\partial_x A_z)` and level sets
        are the two-dimensional field lines. Differences of this function
        give signed magnetic flux per unit cylinder length, not a 3D flux.
        """

        _coordinates, displacement, radius = self._geometry(points)
        radial, _tangent = self._factors(radius)
        cross = self._applied[0] * displacement[..., 1] - self._applied[1] * displacement[..., 0]
        return radial * cross
