"""Core abstractions shared by field models, tracers, and renderers.

The public contract is deliberately small: fields accept one point or an
arbitrarily shaped batch whose final axis stores coordinates, and return an
array with the same shape.  Keeping this contract independent from plotting
lets the same field feed probes, streamline integration, and front ends.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def _norm_last_axis(vectors: FloatArray) -> FloatArray:
    """Euclidean norm along the last axis, bit-identical to ``np.linalg.norm``.

    For real input ``np.linalg.norm(x, axis=-1)`` evaluates exactly
    ``sqrt(add.reduce(x * x, axis=-1))``; calling that directly skips the
    wrapper's argument handling, which dominates for single points.
    """

    return np.sqrt(np.add.reduce(vectors * vectors, axis=-1))


def _as_points(points: ArrayLike, dimension: int, *, name: str = "points") -> FloatArray:
    """Coerce coordinates while preserving all leading batch dimensions."""

    array = np.asarray(points, dtype=float)
    if array.ndim == 0 or array.shape[-1] != dimension:
        raise ValueError(f"{name} must have shape (..., {dimension}); got {array.shape}.")
    return array


class VectorField(ABC):
    """Abstract vector field with a batch-first evaluation API.

    Implementations must accept ``points`` with shape ``(..., dimension)`` and
    return vectors with exactly the same shape.  A single point therefore has
    shape ``(dimension,)`` while a regular grid can be evaluated without a
    Python loop.
    """

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Number of spatial coordinates consumed and vector components returned."""

    @abstractmethod
    def evaluate(self, points: ArrayLike) -> FloatArray:
        """Evaluate the field at one point or a batch of points."""

    def __call__(self, points: ArrayLike) -> FloatArray:
        """Alias for `evaluate`."""

        return self.evaluate(points)


class ExclusionRegion(Protocol):
    """Structural contract for a signed-margin tracing exclusion."""

    @property
    def dimension(self) -> int:
        """Spatial dimension consumed by :meth:`margin`."""

        ...

    def margin(self, points: ArrayLike) -> Any:
        """Return values positive outside, zero on, and negative inside."""

        ...


@dataclass(frozen=True, slots=True)
class Domain:
    """Closed axis-aligned rectangular domain.

    Parameters
    ----------
    lower, upper:
        Coordinate-wise lower and upper bounds.  Bounds must be finite and
        satisfy ``lower < upper`` on every axis.
    """

    lower: ArrayLike
    upper: ArrayLike

    def __post_init__(self) -> None:
        lower = np.array(self.lower, dtype=float, copy=True)
        upper = np.array(self.upper, dtype=float, copy=True)
        if lower.ndim != 1 or upper.ndim != 1 or lower.shape != upper.shape:
            raise ValueError("lower and upper must be one-dimensional arrays of equal size.")
        if lower.size == 0:
            raise ValueError("a domain must have at least one dimension.")
        if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
            raise ValueError("domain bounds must be finite.")
        if np.any(lower >= upper):
            raise ValueError("each lower bound must be strictly less than its upper bound.")
        lower.setflags(write=False)
        upper.setflags(write=False)
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)

    @property
    def dimension(self) -> int:
        """Number of spatial dimensions."""

        return int(self.lower.size)

    @property
    def extent(self) -> FloatArray:
        """Side length along every axis."""

        return self.upper - self.lower

    @property
    def center(self) -> FloatArray:
        """Geometric center of the domain."""

        return 0.5 * (self.lower + self.upper)

    def contains(self, points: ArrayLike, *, atol: float = 0.0) -> Any:
        """Return a boolean, or a boolean batch, indicating point containment."""

        if atol < 0:
            raise ValueError("atol must be non-negative.")
        coordinates = _as_points(points, self.dimension)
        return np.all(
            (coordinates >= self.lower - atol) & (coordinates <= self.upper + atol),
            axis=-1,
        )

    def margin(self, points: ArrayLike) -> Any:
        """Signed distance to the closest face, measured along coordinate axes.

        The result is positive inside, zero on a face, and negative outside.
        It is intended for inexpensive integration-event detection rather than
        as the Euclidean signed-distance function near corners.
        """

        coordinates = _as_points(points, self.dimension)
        lower_margin = coordinates - self.lower
        upper_margin = self.upper - coordinates
        return np.minimum.reduce(np.minimum(lower_margin, upper_margin), axis=-1)


@dataclass(frozen=True, slots=True)
class SphericalExclusion:
    """Circular or spherical source regions excluded from line tracing.

    In a 2D field each center defines a circle; in 3D it defines a sphere.
    ``margin`` is positive outside every region, zero on its surface, and
    negative inside.
    """

    centers: ArrayLike
    radii: ArrayLike

    def __post_init__(self) -> None:
        centers = np.asarray(self.centers, dtype=float)
        if centers.ndim == 1:
            centers = centers[np.newaxis, :]
        if centers.ndim != 2 or centers.shape[0] == 0 or centers.shape[1] == 0:
            raise ValueError("centers must have shape (dimension,) or (n, dimension).")
        radii = np.asarray(self.radii, dtype=float)
        if radii.ndim == 0:
            radii = np.full(centers.shape[0], float(radii))
        if radii.shape != (centers.shape[0],):
            raise ValueError("radii must be scalar or contain one value per center.")
        if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(radii)):
            raise ValueError("exclusion geometry must be finite.")
        if np.any(radii <= 0):
            raise ValueError("exclusion radii must be positive.")
        centers = np.array(centers, copy=True)
        radii = np.array(radii, copy=True)
        centers.setflags(write=False)
        radii.setflags(write=False)
        object.__setattr__(self, "centers", centers)
        object.__setattr__(self, "radii", radii)

    @property
    def dimension(self) -> int:
        return int(self.centers.shape[1])

    def margin(self, points: ArrayLike) -> Any:
        """Return signed distance to the nearest excluded source surface."""

        coordinates = _as_points(points, self.dimension)
        delta = coordinates[..., np.newaxis, :] - self.centers
        distances = _norm_last_axis(delta) - self.radii
        return np.minimum.reduce(distances, axis=-1)


@dataclass(frozen=True, slots=True)
class ToroidalExclusion:
    """A finite-radius tube around a circular centerline in 3D.

    ``normal`` defines the circle axis. ``major_radius`` is the centerline
    radius and ``minor_radius`` is the excluded tube radius. The latter is a
    tracing/masking geometry; it does not soften an ideal filamentary field.
    """

    center: ArrayLike
    normal: ArrayLike
    major_radius: float
    minor_radius: float

    def __post_init__(self) -> None:
        center = np.asarray(self.center, dtype=float)
        normal = np.asarray(self.normal, dtype=float)
        if center.shape != (3,):
            raise ValueError("center must have shape (3,).")
        if normal.shape != (3,):
            raise ValueError("normal must have shape (3,).")
        if not np.all(np.isfinite(center)) or not np.all(np.isfinite(normal)):
            raise ValueError("torus center and normal must be finite.")

        normal_scale = float(np.max(np.abs(normal)))
        if normal_scale == 0.0:
            raise ValueError("normal must be nonzero.")
        scaled_normal = normal / normal_scale
        normal = scaled_normal / np.linalg.norm(scaled_normal)

        major = np.asarray(self.major_radius, dtype=float)
        minor = np.asarray(self.minor_radius, dtype=float)
        if major.ndim != 0 or not np.isfinite(major) or float(major) <= 0.0:
            raise ValueError("major_radius must be a finite positive scalar.")
        if minor.ndim != 0 or not np.isfinite(minor) or float(minor) <= 0.0:
            raise ValueError("minor_radius must be a finite positive scalar.")
        if float(minor) >= float(major):
            raise ValueError("minor_radius must be smaller than major_radius.")

        center = np.array(center, copy=True)
        normal = np.array(normal, copy=True)
        center.setflags(write=False)
        normal.setflags(write=False)
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "normal", normal)
        object.__setattr__(self, "major_radius", float(major))
        object.__setattr__(self, "minor_radius", float(minor))

    @property
    def dimension(self) -> int:
        return 3

    def margin(self, points: ArrayLike) -> Any:
        """Return Euclidean signed distance to the tube surface."""

        coordinates = _as_points(points, 3)
        displacement = coordinates - self.center
        axial = np.einsum("...d,d->...", displacement, self.normal)
        radial_vectors = displacement - axial[..., np.newaxis] * self.normal
        radial = _norm_last_axis(radial_vectors)
        centerline_distance = np.hypot(radial - self.major_radius, axial)
        return centerline_distance - self.minor_radius


__all__ = [
    "Domain",
    "ExclusionRegion",
    "FloatArray",
    "SphericalExclusion",
    "ToroidalExclusion",
    "VectorField",
]
