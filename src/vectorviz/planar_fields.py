"""Analytic two-dimensional harmonic fields and first-order ODE direction fields."""

from __future__ import annotations

import math
from collections.abc import Callable
from numbers import Integral

import numpy as np
from numpy.typing import ArrayLike

from .core import FloatArray, VectorField, _as_points


class LaplaceField(VectorField):
    r"""Negative gradient of a two-dimensional harmonic polynomial potential.

    In coordinates rotated by ``-angle`` about ``center``, let
    ``w = (x' + i*y') / length_scale``. The potential is
    ``strength * length_scale * Re(w**order) / order``. Consequently order
    one is a uniform field, order two a saddle, and order three a six-sector
    field. These are true planar Laplace solutions, not slices of 3D point
    sources. There are no singularities; for order greater than one the
    center is a field null.

    ``angle`` is in radians; ``length_scale`` and ``center`` use the input
    coordinate unit. ``strength`` has field units, so the potential has
    field times coordinate units. The class does not prescribe SI units.
    Nonfinite inputs or numerical overflow produce NaN vectors rather than
    a softened or clipped field.
    """

    def __init__(
        self,
        order: int = 2,
        strength: float = 1.0,
        *,
        length_scale: float = 1.0,
        angle: float = 0.0,
        center: ArrayLike = (0.0, 0.0),
    ) -> None:
        if isinstance(order, bool) or not isinstance(order, Integral) or order < 1:
            raise ValueError("order must be a positive integer.")
        if not np.isfinite(strength):
            raise ValueError("strength must be finite.")
        if not np.isfinite(length_scale) or length_scale <= 0:
            raise ValueError("length_scale must be finite and positive.")
        if not np.isfinite(angle):
            raise ValueError("angle must be finite.")
        location = np.array(center, dtype=float, copy=True)
        if location.shape != (2,) or not np.all(np.isfinite(location)):
            raise ValueError("center must be a finite two-dimensional point.")
        location.setflags(write=False)
        self._order = int(order)
        self._strength = float(strength)
        self._length_scale = float(length_scale)
        self._angle = float(angle)
        self._center = location
        self._cos = math.cos(angle)
        self._sin = math.sin(angle)

    @property
    def dimension(self) -> int:
        return 2

    @property
    def order(self) -> int:
        return self._order

    @property
    def strength(self) -> float:
        return self._strength

    @property
    def length_scale(self) -> float:
        return self._length_scale

    @property
    def angle(self) -> float:
        return self._angle

    @property
    def center(self) -> FloatArray:
        return self._center

    def _coordinate(self, coordinates: FloatArray) -> np.ndarray:
        displacement = coordinates - self._center
        x = displacement[..., 0] / self._length_scale
        y = displacement[..., 1] / self._length_scale
        return (self._cos * x + self._sin * y) + 1j * (self._cos * y - self._sin * x)

    def potential(self, points: ArrayLike) -> FloatArray:
        """Return the scalar potential, preserving the leading batch dimensions."""
        coordinates = _as_points(points, 2)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            if self._strength == 0.0:
                values = np.zeros(coordinates.shape[:-1])
            else:
                values = (
                    self._strength
                    * self._length_scale
                    / self._order
                    * np.real(self._coordinate(coordinates) ** self._order)
                )
        valid = np.all(np.isfinite(coordinates), axis=-1) & np.isfinite(values)
        return np.asarray(np.where(valid, values, np.nan), dtype=float)

    def evaluate(self, points: ArrayLike) -> FloatArray:
        coordinates = _as_points(points, 2)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            if self._strength == 0.0:
                values = np.zeros_like(coordinates)
            else:
                derivative = self._strength * self._coordinate(coordinates) ** (self._order - 1)
                local_x, local_y = -derivative.real, derivative.imag
                values = np.stack(
                    (
                        self._cos * local_x - self._sin * local_y,
                        self._sin * local_x + self._cos * local_y,
                    ),
                    axis=-1,
                )
        valid = np.all(np.isfinite(coordinates), axis=-1) & np.all(np.isfinite(values), axis=-1)
        return np.where(valid[..., None], values, np.nan)


class SlopeField(VectorField):
    r"""Direction field ``(1, f(x, y))`` for the equation ``dy/dx = f(x, y)``.

    ``slope`` must accept NumPy arrays for x and y and return a real value
    broadcastable to their common shape. A scalar return describes a constant
    slope. Coordinates and slopes are interpreted in the caller's chosen
    units; the Web examples use dimensionless x and y. Magnitude is the norm
    of this graph-direction vector, not a physical field intensity. In
    particular, ``f=0`` is a horizontal solution direction, not a field null.

    Invalid coordinates and nonfinite slopes give NaN in both components;
    slope values are never clipped. The caller must supply smoothness and
    any needed singularity exclusions to the tracer. This adapter executes a
    Python callable; it does not parse or evaluate expression strings.
    """

    def __init__(self, slope: Callable[[FloatArray, FloatArray], ArrayLike]) -> None:
        if not callable(slope):
            raise ValueError("slope must be callable.")
        self._slope = slope

    @property
    def dimension(self) -> int:
        return 2

    def evaluate(self, points: ArrayLike) -> FloatArray:
        coordinates = _as_points(points, 2)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            raw_slopes = np.asarray(self._slope(coordinates[..., 0], coordinates[..., 1]))
        if np.iscomplexobj(raw_slopes):
            raise ValueError("slope output must be real.")
        slopes = np.asarray(raw_slopes, dtype=float)
        try:
            slopes = np.broadcast_to(slopes, coordinates.shape[:-1])
        except ValueError as error:
            raise ValueError("slope output must broadcast to the point batch shape.") from error
        values = np.stack((np.ones_like(slopes), slopes), axis=-1)
        valid = np.all(np.isfinite(coordinates), axis=-1) & np.isfinite(slopes)
        return np.where(valid[..., None], values, np.nan)
