"""Analytic vector-field implementations."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from numpy.typing import ArrayLike
from scipy.constants import epsilon_0, mu_0

from .core import FloatArray, VectorField, _as_points


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
        return np.array(np.broadcast_to(self._vector, coordinates.shape), copy=True)


class PointChargeField(VectorField):
    r"""Electric field of one or more ideal point charges.

    ``charge`` can be a scalar or a one-dimensional array. ``position`` can be
    a single coordinate vector or an ``(n, dimension)`` array. A scalar charge
    is broadcast over multiple positions. Evaluation at a source position is
    returned as ``NaN`` because the ideal field is undefined there; the model
    never hides that singularity with softening.
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
        singular = np.any(radius_squared == 0.0, axis=1)

        inverse_radius_cubed = np.zeros_like(radius_squared)
        nonzero = radius_squared > 0.0
        inverse_radius_cubed[nonzero] = radius_squared[nonzero] ** -1.5
        coefficient = 1.0 / (4.0 * np.pi * self._permittivity)
        values = coefficient * np.sum(
            self._charges[np.newaxis, :, np.newaxis]
            * displacement
            * inverse_radius_cubed[:, :, np.newaxis],
            axis=1,
        )
        values[singular] = np.nan
        return values.reshape(original_shape)


class MagneticDipoleField(VectorField):
    r"""Magnetic flux density of one or more ideal point dipoles in 3D."""

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
        singular = np.any(radius_squared == 0.0, axis=1)
        moment_dot_radius = np.einsum("pmd,md->pm", displacement, self._moments)

        inverse_radius_cubed = np.zeros_like(radius_squared)
        inverse_radius_fifth = np.zeros_like(radius_squared)
        nonzero = radius_squared > 0.0
        inverse_radius_cubed[nonzero] = radius_squared[nonzero] ** -1.5
        inverse_radius_fifth[nonzero] = radius_squared[nonzero] ** -2.5
        contributions = (
            3.0
            * displacement
            * moment_dot_radius[:, :, np.newaxis]
            * inverse_radius_fifth[:, :, np.newaxis]
            - self._moments[np.newaxis, :, :] * inverse_radius_cubed[:, :, np.newaxis]
        )
        values = self._permeability / (4.0 * np.pi) * np.sum(contributions, axis=1)
        values[singular] = np.nan
        return values.reshape(original_shape)


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
    "CompositeField",
    "MagneticDipoleField",
    "PointChargeField",
    "UniformField",
]
