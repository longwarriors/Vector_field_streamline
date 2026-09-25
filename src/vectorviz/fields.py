"""Analytic vector-field implementations."""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np
from numpy.typing import ArrayLike
from scipy.constants import epsilon_0, mu_0
from scipy.special import ellipe, ellipk, ellipkm1

from .core import FloatArray, VectorField, _as_points, _norm_last_axis

_EPSILON = float(np.finfo(float).eps)


def _selection(mask: np.ndarray) -> slice | np.ndarray | None:
    """Index for the masked elements: None if none, a full slice if all.

    Elementwise operations on ``array[slice(None)]`` see the same values in
    the same order as on ``array[mask]`` when every element is selected, so
    the shortcut is exact while avoiding gather/scatter copies.
    """

    if mask.size == 1:  # the single-point case used by the tracer
        return slice(None) if mask[0] else None
    if not mask.any():
        return None
    if mask.all():
        return slice(None)
    return mask


def _polyval(values: FloatArray, coefficients: FloatArray) -> FloatArray:
    """Horner evaluation, the same steps as ``numpy.polynomial.polynomial.polyval``."""

    result = coefficients[-1] + values * 0
    for coefficient in coefficients[-2::-1]:
        result = coefficient + result * values
    return result


def _readonly(array: ArrayLike) -> FloatArray:
    result = np.array(array, dtype=float, copy=True)
    result.setflags(write=False)
    return result


class UniformField(VectorField):
    """A spatially uniform vector field."""

    def __init__(self, vector: ArrayLike) -> None:
        value = np.asarray(vector, dtype=float)
        if value.ndim != 1 or value.size == 0:
            raise ValueError("vector must be a non-empty one-dimensional array.")
        if not np.all(np.isfinite(value)):
            raise ValueError("vector components must be finite.")
        self._vector = _readonly(value)

    @property
    def dimension(self) -> int:
        return int(self._vector.size)

    @property
    def vector(self) -> FloatArray:
        return self._vector

    def evaluate(self, points: ArrayLike) -> FloatArray:
        coordinates = _as_points(points, self.dimension)
        if coordinates.ndim == 1:
            return self._vector.copy()
        return np.array(np.broadcast_to(self._vector, coordinates.shape), copy=True)


class PointChargeField(VectorField):
    r"""Electric field of one or more ideal point charges.

    ``charge`` can be a scalar or a one-dimensional array. ``position`` can be
    a single coordinate vector or an ``(n, dimension)`` array. A scalar charge
    is broadcast over multiple positions. Evaluation at a source position is
    returned as ``NaN`` because the ideal field is undefined there; the model
    never hides that singularity with softening. The singular set is fixed by
    the source geometry, not by the charge value: a zero charge keeps its
    ``NaN`` at the source point, so callers that want a zero source to vanish
    must drop it before building the field.
    """

    def __init__(
        self,
        charge: ArrayLike,
        position: ArrayLike = (0.0, 0.0, 0.0),
        *,
        permittivity: float = epsilon_0,
    ) -> None:
        positions = np.asarray(position, dtype=float)
        if positions.ndim == 1:
            positions = positions[np.newaxis, :]
        if positions.ndim != 2 or positions.shape[0] == 0 or positions.shape[1] == 0:
            raise ValueError("position must have shape (dimension,) or (n, dimension).")
        if not np.all(np.isfinite(positions)):
            raise ValueError("charge positions must be finite.")

        charges = np.asarray(charge, dtype=float)
        if charges.ndim == 0:
            charges = np.full(positions.shape[0], float(charges))
        elif charges.ndim != 1:
            raise ValueError("charge must be a scalar or a one-dimensional array.")
        if charges.size != positions.shape[0]:
            raise ValueError("the number of charges must match the number of positions.")
        if not np.all(np.isfinite(charges)):
            raise ValueError("charges must be finite.")
        if not np.isfinite(permittivity) or permittivity <= 0:
            raise ValueError("permittivity must be a finite positive number.")

        self._positions = _readonly(positions)
        self._charges = _readonly(charges)
        self._permittivity = float(permittivity)

    @property
    def dimension(self) -> int:
        return int(self._positions.shape[1])

    @property
    def positions(self) -> FloatArray:
        return self._positions

    @property
    def charges(self) -> FloatArray:
        return self._charges

    @property
    def permittivity(self) -> float:
        return self._permittivity

    def evaluate(self, points: ArrayLike) -> FloatArray:
        coordinates = _as_points(points, self.dimension)
        original_shape = coordinates.shape
        flat_points = coordinates.reshape(-1, self.dimension)
        displacement = flat_points[:, np.newaxis, :] - self._positions[np.newaxis, :, :]
        radius_squared = np.einsum("pqd,pqd->pq", displacement, displacement)
        singular = np.logical_or.reduce(radius_squared == 0.0, axis=1)

        inverse_radius_cubed = np.zeros_like(radius_squared)
        nonzero = _selection(radius_squared > 0.0)
        if nonzero is not None:
            inverse_radius_cubed[nonzero] = radius_squared[nonzero] ** -1.5
        coefficient = 1.0 / (4.0 * np.pi * self._permittivity)
        values = coefficient * np.add.reduce(
            self._charges[np.newaxis, :, np.newaxis]
            * displacement
            * inverse_radius_cubed[:, :, np.newaxis],
            axis=1,
        )
        if singular.any():
            values[singular] = np.nan
        return values.reshape(original_shape)


class MagneticDipoleField(VectorField):
    r"""Magnetic flux density of one or more ideal point dipoles in 3D.

    Each dipole position is an explicit ``NaN`` singularity regardless of its
    moment, matching :class:`PointChargeField` and :class:`CircularLoopField`:
    the source geometry defines the singular set, not the strength. Remove a
    zero-moment dipole before constructing the field if it should not exist.
    """

    def __init__(
        self,
        moment: ArrayLike,
        position: ArrayLike = (0.0, 0.0, 0.0),
        *,
        permeability: float = mu_0,
    ) -> None:
        moments = np.asarray(moment, dtype=float)
        if moments.ndim == 1:
            moments = moments[np.newaxis, :]
        positions = np.asarray(position, dtype=float)
        if positions.ndim == 1:
            positions = positions[np.newaxis, :]
        if moments.ndim != 2 or moments.shape[1:] != (3,) or moments.shape[0] == 0:
            raise ValueError("moment must have shape (3,) or (n, 3).")
        if positions.ndim != 2 or positions.shape[1:] != (3,) or positions.shape[0] == 0:
            raise ValueError("position must have shape (3,) or (n, 3).")

        count = max(moments.shape[0], positions.shape[0])
        if moments.shape[0] not in (1, count) or positions.shape[0] not in (1, count):
            raise ValueError("moment and position batches must have equal size or size one.")
        moments = np.broadcast_to(moments, (count, 3))
        positions = np.broadcast_to(positions, (count, 3))
        if not np.all(np.isfinite(moments)) or not np.all(np.isfinite(positions)):
            raise ValueError("dipole moments and positions must be finite.")
        if not np.isfinite(permeability) or permeability <= 0:
            raise ValueError("permeability must be a finite positive number.")

        self._moments = _readonly(moments)
        self._positions = _readonly(positions)
        self._permeability = float(permeability)

    @property
    def dimension(self) -> int:
        return 3

    @property
    def moments(self) -> FloatArray:
        return self._moments

    @property
    def positions(self) -> FloatArray:
        return self._positions

    @property
    def permeability(self) -> float:
        return self._permeability

    def evaluate(self, points: ArrayLike) -> FloatArray:
        coordinates = _as_points(points, 3)
        original_shape = coordinates.shape
        flat_points = coordinates.reshape(-1, 3)
        displacement = flat_points[:, np.newaxis, :] - self._positions[np.newaxis, :, :]
        radius_squared = np.einsum("pmd,pmd->pm", displacement, displacement)
        singular = np.logical_or.reduce(radius_squared == 0.0, axis=1)
        moment_dot_radius = np.einsum("pmd,md->pm", displacement, self._moments)

        inverse_radius_cubed = np.zeros_like(radius_squared)
        inverse_radius_fifth = np.zeros_like(radius_squared)
        nonzero = _selection(radius_squared > 0.0)
        if nonzero is not None:
            inverse_radius_cubed[nonzero] = radius_squared[nonzero] ** -1.5
            inverse_radius_fifth[nonzero] = radius_squared[nonzero] ** -2.5
        contributions = (
            3.0
            * displacement
            * moment_dot_radius[:, :, np.newaxis]
            * inverse_radius_fifth[:, :, np.newaxis]
            - self._moments[np.newaxis, :, :] * inverse_radius_cubed[:, :, np.newaxis]
        )
        values = self._permeability / (4.0 * np.pi) * np.add.reduce(contributions, axis=1)
        if singular.any():
            values[singular] = np.nan
        return values.reshape(original_shape)


def _finite_scalar(value: float, name: str) -> float:
    array = np.asarray(value, dtype=float)
    if array.ndim != 0 or not np.isfinite(array):
        raise ValueError(f"{name} must be a finite scalar.")
    return float(array)


class _CircularLoopGeometry(VectorField):
    """Validated geometry shared by analytic and quadrature loop fields."""

    def __init__(
        self,
        current: float,
        radius: float,
        center: ArrayLike,
        normal: ArrayLike,
        permeability: float,
    ) -> None:
        current_value = _finite_scalar(current, "current")
        radius_value = _finite_scalar(radius, "radius")
        permeability_value = _finite_scalar(permeability, "permeability")
        if radius_value <= 0.0:
            raise ValueError("radius must be positive.")
        if permeability_value <= 0.0:
            raise ValueError("permeability must be positive.")

        center_value = np.asarray(center, dtype=float)
        normal_value = np.asarray(normal, dtype=float)
        if center_value.shape != (3,):
            raise ValueError("center must have shape (3,).")
        if normal_value.shape != (3,):
            raise ValueError("normal must have shape (3,).")
        if not np.all(np.isfinite(center_value)) or not np.all(np.isfinite(normal_value)):
            raise ValueError("center and normal must be finite.")
        normal_scale = float(np.max(np.abs(normal_value)))
        if normal_scale == 0.0:
            raise ValueError("normal must be nonzero.")
        scaled_normal = normal_value / normal_scale
        normal_norm = float(np.linalg.norm(scaled_normal))

        self._current = current_value
        self._radius = radius_value
        self._center = _readonly(center_value)
        self._normal = _readonly(scaled_normal / normal_norm)
        self._permeability = permeability_value

    @property
    def dimension(self) -> int:
        return 3

    @property
    def current(self) -> float:
        return self._current

    @property
    def radius(self) -> float:
        return self._radius

    @property
    def center(self) -> FloatArray:
        return self._center

    @property
    def normal(self) -> FloatArray:
        return self._normal

    @property
    def permeability(self) -> float:
        return self._permeability

    def _cylindrical_geometry(
        self, points: ArrayLike
    ) -> tuple[FloatArray, tuple[int, ...], FloatArray, FloatArray, FloatArray, FloatArray]:
        coordinates = _as_points(points, 3)
        original_shape = coordinates.shape
        flat_points = coordinates.reshape(-1, 3)
        displacement = flat_points - self._center
        axial = displacement @ self._normal
        radial_vectors = displacement - axial[:, np.newaxis] * self._normal
        radial = _norm_last_axis(radial_vectors)
        wire_distance = np.hypot(radial - self._radius, axial)
        geometry_scale = np.maximum(self._radius, _norm_last_axis(displacement))
        # A rotated point constructed on the mathematical filament generally
        # misses exact floating equality after dot products and norms. This
        # ULP-scale classification only absorbs coordinate roundoff; it is not
        # a physical wire radius and never enters a field denominator.
        singular_tolerance = 32.0 * _EPSILON * geometry_scale
        singular = wire_distance <= singular_tolerance
        return coordinates, original_shape, radial_vectors, radial, axial, singular


def _elliptic_series_coefficients(order: int = 12) -> tuple[FloatArray, FloatArray]:
    """Series for the two cancelling K/E combinations used by a loop."""

    radial = np.zeros(order + 1, dtype=float)
    flux = np.zeros(order + 1, dtype=float)
    previous_first = 1.0
    cumulative_second = 1.0
    for index in range(1, order + 1):
        first = previous_first * ((2.0 * index - 1.0) / (2.0 * index)) ** 2
        second = -first / (2.0 * index - 1.0)
        radial[index] = second + 0.5 * cumulative_second - first
        flux[index] = first - 0.5 * previous_first - second
        previous_first = first
        cumulative_second += second
    return radial, flux


_RADIAL_ELLIPTIC_SERIES, _FLUX_ELLIPTIC_SERIES = _elliptic_series_coefficients()


def _loop_elliptic_terms(
    parameter: FloatArray, complementary_parameter: FloatArray
) -> tuple[FloatArray, FloatArray, FloatArray]:
    r"""Evaluate E and stable loop-specific combinations of K and E.

    The radial field uses
    ``P=-K+(1-m/2)E/(1-m)`` and the flux function uses
    ``Q=(1-m/2)K-E``. Both begin at order :math:`m^2`, so evaluating them by
    direct subtraction is inaccurate in the far field. A convergent power
    series is used for small ``m``; the general branch still calls SciPy with
    its parameter convention ``m=k^2``.
    """

    first = np.empty_like(parameter)
    near_wire_mask = complementary_parameter < 0.1
    near_wire = _selection(near_wire_mask)
    away = _selection(~near_wire_mask)
    if near_wire is not None:
        first[near_wire] = ellipkm1(complementary_parameter[near_wire])
    if away is not None:
        first[away] = ellipk(parameter[away])
    second = np.asarray(ellipe(parameter), dtype=float)
    radial_term = np.empty_like(parameter)
    flux_term = np.empty_like(parameter)
    small_parameter_mask = parameter < 1.0e-2
    small_parameter = _selection(small_parameter_mask)
    if small_parameter is not None:
        radial_term[small_parameter] = np.pi / 2.0 * _polyval(
            parameter[small_parameter], _RADIAL_ELLIPTIC_SERIES
        )
        flux_term[small_parameter] = np.pi / 2.0 * _polyval(
            parameter[small_parameter], _FLUX_ELLIPTIC_SERIES
        )
    ordinary = _selection(~small_parameter_mask)
    if ordinary is not None:
        radial_term[ordinary] = (
            -first[ordinary]
            + (1.0 - 0.5 * parameter[ordinary])
            / complementary_parameter[ordinary]
            * second[ordinary]
        )
        flux_term[ordinary] = (
            (1.0 - 0.5 * parameter[ordinary]) * first[ordinary] - second[ordinary]
        )
    return second, radial_term, flux_term


class CircularLoopField(_CircularLoopGeometry):
    r"""Magnetic flux density of an ideal circular filamentary current loop.

    Coordinates, ``radius`` and ``center`` use metres, ``current`` uses
    amperes, and the returned field uses tesla when ``permeability`` has SI
    units. ``normal`` fixes the loop plane and the positive-current direction
    by the right-hand rule. The filament itself is an explicit ``NaN``
    singularity; no finite wire radius or epsilon softening is implied.
    """

    _NEAR_AXIS_RATIO = 1.0e-3

    def __init__(
        self,
        current: float,
        radius: float,
        center: ArrayLike = (0.0, 0.0, 0.0),
        normal: ArrayLike = (0.0, 0.0, 1.0),
        *,
        permeability: float = mu_0,
    ) -> None:
        super().__init__(current, radius, center, normal, permeability)

    def _elliptic_parameter(
        self, rho: FloatArray, q_squared: FloatArray, complementary_parameter: FloatArray
    ) -> FloatArray:
        """Elliptic parameter ``m``, taken as ``1 - m1`` near the filament."""

        parameter = np.empty_like(complementary_parameter)
        near_wire_mask = complementary_parameter < 0.1
        near_wire = _selection(near_wire_mask)
        away = _selection(~near_wire_mask)
        if near_wire is not None:
            parameter[near_wire] = 1.0 - complementary_parameter[near_wire]
        if away is not None:
            parameter[away] = 4.0 * self._radius * rho[away] / q_squared[away]
        return parameter

    def _axis_derivatives(
        self, axial: FloatArray
    ) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
        radius_squared = self._radius**2
        axial_squared = axial**2
        scale_squared = radius_squared + axial_squared
        coefficient = self._permeability * self._current * radius_squared / 2.0
        field = coefficient * scale_squared**-1.5
        first = -3.0 * coefficient * axial * scale_squared**-2.5
        second = (
            3.0
            * coefficient
            * (4.0 * axial_squared - radius_squared)
            * scale_squared**-3.5
        )
        third = (
            15.0
            * coefficient
            * axial
            * (3.0 * radius_squared - 4.0 * axial_squared)
            * scale_squared**-4.5
        )
        fourth = (
            45.0
            * coefficient
            * (radius_squared**2 - 12.0 * radius_squared * axial_squared + 8.0 * axial**4)
            * scale_squared**-5.5
        )
        return field, first, second, third, fourth

    def evaluate(self, points: ArrayLike) -> FloatArray:
        coordinates = _as_points(points, 3)
        if coordinates.ndim == 1:
            return self._evaluate_point(coordinates)
        return self._evaluate_batch(coordinates)

    def _evaluate_point(self, coordinates: FloatArray) -> FloatArray:
        """Evaluate one point, bit-for-bit equal to the batch path on one row.

        The field tracer calls this once per solver stage, where NumPy's
        per-call overhead on one-element arrays dominates. The geometry reuses
        the batch code, so the dot product and norms are the same operations.
        The general-region formulas then run on Python floats using ``+ - * /``,
        ``sqrt`` and the same Python ``radius**2`` as the batch path, which
        IEEE 754 rounds identically for floats and arrays, and the same SciPy
        elliptic ufuncs. Near-axis and singular points keep the batch path,
        whose fractional powers may be vectorised differently from a scalar
        ``pow``. So do points whose denominators underflow to zero or are not
        finite, where Python raises ``ZeroDivisionError`` but NumPy returns
        ``inf`` or ``nan``.
        """

        (
            _coordinates,
            _original_shape,
            radial_vectors,
            radial,
            axial,
            singular,
        ) = self._cylindrical_geometry(coordinates)
        rho = float(radial[0])
        z = float(axial[0])
        radius = self._radius
        if singular[0] or rho <= self._NEAR_AXIS_RATIO * math.sqrt(radius**2 + z * z):
            return self._evaluate_batch(coordinates)

        outer = radius + rho
        inner = radius - rho
        q_squared = outer * outer + z * z
        wire_distance_squared = inner * inner + z * z
        # Non-singular points have wire/q >= (32 eps)^2 / 4, so after this
        # check no denominator below is zero.
        if not (wire_distance_squared > 0.0 and q_squared < math.inf):
            return self._evaluate_batch(coordinates)
        complementary_parameter = wire_distance_squared / q_squared
        if complementary_parameter < 0.1:
            parameter = 1.0 - complementary_parameter
            first = ellipkm1(complementary_parameter)
        else:
            parameter = 4.0 * radius * rho / q_squared
            first = ellipk(parameter)
        second = ellipe(parameter)
        if parameter < 1.0e-2:
            radial_elliptic = np.pi / 2.0 * _polyval(parameter, _RADIAL_ELLIPTIC_SERIES)
        else:
            radial_elliptic = -first + (1.0 - 0.5 * parameter) / complementary_parameter * second
        prefactor = self._permeability * self._current / (2.0 * np.pi * math.sqrt(q_squared))
        radial_component = prefactor * z / rho * radial_elliptic
        axial_component = prefactor * (
            -radial_elliptic + 2.0 * radius**2 / wire_distance_squared * second
        )
        row = radial_vectors[0]
        normal = self._normal
        return np.array(
            [
                radial_component * float(row[index]) / rho + axial_component * float(normal[index])
                for index in range(3)
            ],
            dtype=float,
        )

    def _evaluate_batch(self, points: FloatArray) -> FloatArray:
        (
            _coordinates,
            original_shape,
            radial_vectors,
            radial,
            axial,
            singular,
        ) = self._cylindrical_geometry(points)
        values = np.empty((radial.size, 3), dtype=float)

        scale = np.sqrt(self._radius**2 + axial**2)
        near_axis_mask = radial <= self._NEAR_AXIS_RATIO * scale
        near_axis = _selection(near_axis_mask)
        if near_axis is not None:
            rho = radial[near_axis]
            z = axial[near_axis]
            field, first, second, third, fourth = self._axis_derivatives(z)
            radial_coefficient = -0.5 * first + rho**2 * third / 16.0
            axial_component = field - rho**2 * second / 4.0 + rho**4 * fourth / 64.0
            values[near_axis] = (
                radial_coefficient[:, np.newaxis] * radial_vectors[near_axis]
                + axial_component[:, np.newaxis] * self._normal
            )

        general = _selection(~near_axis_mask & ~singular)
        if general is not None:
            rho = radial[general]
            z = axial[general]
            radius = self._radius
            q_squared = (radius + rho) ** 2 + z**2
            wire_distance_squared = (radius - rho) ** 2 + z**2
            complementary_parameter = wire_distance_squared / q_squared
            parameter = self._elliptic_parameter(rho, q_squared, complementary_parameter)
            second, radial_elliptic, _flux_elliptic = _loop_elliptic_terms(
                parameter, complementary_parameter
            )
            prefactor = self._permeability * self._current / (2.0 * np.pi * np.sqrt(q_squared))
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                radial_component = prefactor * z / rho * radial_elliptic
                axial_component = prefactor * (
                    -radial_elliptic + 2.0 * radius**2 / wire_distance_squared * second
                )
                values[general] = (
                    radial_component[:, np.newaxis]
                    * radial_vectors[general]
                    / rho[:, np.newaxis]
                    + axial_component[:, np.newaxis] * self._normal
                )

        if singular.any():
            values[singular] = np.nan
        return values.reshape(original_shape)

    def flux_function(self, points: ArrayLike) -> FloatArray:
        r"""Return the axisymmetric flux function :math:`\psi=\rho A_\phi`.

        The result has the leading shape of ``points``. It is zero on the
        symmetry axis and ``NaN`` on the ideal filament. Away from the source,
        its contours in any meridional plane are magnetic field lines.
        """

        (
            _coordinates,
            original_shape,
            _radial_vectors,
            radial,
            axial,
            singular,
        ) = self._cylindrical_geometry(points)
        result = np.empty(radial.size, dtype=float)

        scale = np.sqrt(self._radius**2 + axial**2)
        near_axis_mask = radial <= self._NEAR_AXIS_RATIO * scale
        near_axis = _selection(near_axis_mask)
        if near_axis is not None:
            rho = radial[near_axis]
            field, _first, second, _third, fourth = self._axis_derivatives(axial[near_axis])
            result[near_axis] = (
                0.5 * field * rho**2
                - second * rho**4 / 16.0
                + fourth * rho**6 / 384.0
            )

        general = _selection(~near_axis_mask & ~singular)
        if general is not None:
            rho = radial[general]
            z = axial[general]
            radius = self._radius
            q_squared = (radius + rho) ** 2 + z**2
            wire_distance_squared = (radius - rho) ** 2 + z**2
            complementary_parameter = wire_distance_squared / q_squared
            parameter = self._elliptic_parameter(rho, q_squared, complementary_parameter)
            _second, _radial_elliptic, flux_elliptic = _loop_elliptic_terms(
                parameter, complementary_parameter
            )
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                result[general] = (
                    self._permeability
                    * self._current
                    / np.pi
                    * np.sqrt(radius * rho / parameter)
                    * flux_elliptic
                )

        if singular.any():
            result[singular] = np.nan
        return result.reshape(original_shape[:-1])


class CompositeField(VectorField):
    """Weighted linear superposition of vector fields with equal dimension.

    Both ``CompositeField([a, b])`` and ``CompositeField(a, b)`` are accepted.
    ``weights`` defaults to one for every constituent field.
    """

    def __init__(
        self,
        fields: VectorField | Iterable[VectorField],
        *additional_fields: VectorField,
        weights: ArrayLike | None = None,
    ) -> None:
        if isinstance(fields, VectorField):
            all_fields = (fields, *additional_fields)
        else:
            if additional_fields:
                raise TypeError(
                    "additional positional fields are only allowed when the first argument "
                    "is a VectorField."
                )
            all_fields = tuple(fields)
        if not all_fields or not all(isinstance(field, VectorField) for field in all_fields):
            raise ValueError("fields must contain at least one VectorField.")
        dimension = all_fields[0].dimension
        if any(field.dimension != dimension for field in all_fields[1:]):
            raise ValueError("all fields in a composite must have the same dimension.")

        if weights is None:
            weight_values = np.ones(len(all_fields), dtype=float)
        else:
            weight_values = np.asarray(weights, dtype=float)
            if weight_values.shape != (len(all_fields),):
                raise ValueError("weights must contain one value per field.")
            if not np.all(np.isfinite(weight_values)):
                raise ValueError("weights must be finite.")

        self._fields = all_fields
        self._weights = _readonly(weight_values)
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def fields(self) -> tuple[VectorField, ...]:
        return self._fields

    @property
    def weights(self) -> FloatArray:
        return self._weights

    def evaluate(self, points: ArrayLike) -> FloatArray:
        coordinates = _as_points(points, self.dimension)
        result = np.zeros_like(coordinates, dtype=float)
        for weight, field in zip(self._weights, self._fields, strict=True):
            result += weight * field.evaluate(coordinates)
        return result


__all__ = [
    "CircularLoopField",
    "CompositeField",
    "MagneticDipoleField",
    "PointChargeField",
    "UniformField",
]
