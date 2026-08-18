"""Field-line invariants for the analytic circular-current-loop field."""

from __future__ import annotations

import numpy as np

from vectorviz import (
    CircularLoopField,
    TerminationReason,
    ToroidalExclusion,
    TraceOptions,
    trace_field_line,
)


def _flux_from_biot_savart_vector_potential(
    points: np.ndarray,
    *,
    current: float,
    radius: float,
    permeability: float,
    order: int = 256,
) -> np.ndarray:
    r"""Return psi=rho*A_phi from an independent quadrature of vector potential."""

    coordinates = np.asarray(points, dtype=float)
    original_shape = coordinates.shape[:-1]
    flat_points = coordinates.reshape(-1, 3)
    nodes, weights = np.polynomial.legendre.leggauss(order)
    angles = np.pi * (nodes + 1.0)
    weights = np.pi * weights
    cosine = np.cos(angles)
    sine = np.sin(angles)
    sources = radius * np.column_stack((cosine, sine, np.zeros_like(cosine)))
    line_derivatives = radius * np.column_stack((-sine, cosine, np.zeros_like(cosine)))
    result = np.empty(flat_points.shape[0])

    for index, point in enumerate(flat_points):
        displacement = point - sources
        vector_potential = (
            permeability
            * current
            / (4.0 * np.pi)
            * np.sum(
                weights[:, np.newaxis]
                * line_derivatives
                / np.linalg.norm(displacement, axis=1)[:, np.newaxis],
                axis=0,
            )
        )
        rho = float(np.hypot(point[0], point[1]))
        if rho == 0.0:
            result[index] = 0.0
        else:
            azimuthal = np.array((-point[1], point[0], 0.0)) / rho
            result[index] = rho * float(np.dot(vector_potential, azimuthal))
    return result.reshape(original_shape)


def _loop_options(*, output_step: float, max_arc_length: float) -> TraceOptions:
    return TraceOptions(
        max_arc_length=max_arc_length,
        max_step=0.01,
        rtol=1.0e-10,
        atol=1.0e-12,
        null_threshold=1.0e-14,
        output_step=output_step,
        method="DOP853",
    )


def test_flux_function_drift_stays_bounded_along_a_traced_loop_line() -> None:
    current = 1.0
    radius = 1.0
    permeability = 4.0 * np.pi
    field = CircularLoopField(current, radius, permeability=permeability)
    seed = np.array((0.55, 0.0, 0.35))
    result = trace_field_line(
        field,
        seed,
        options=TraceOptions(
            max_arc_length=6.0,
            max_step=0.1,
            rtol=1.0e-7,
            atol=1.0e-9,
            null_threshold=1.0e-14,
            output_step=0.03,
            method="DOP853",
        ),
        direction="forward",
    )

    assert result.forward is not None
    assert result.forward.termination is TerminationReason.MAX_ARC_LENGTH
    oracle_flux = _flux_from_biot_savart_vector_potential(
        result.points,
        current=current,
        radius=radius,
        permeability=permeability,
    )
    public_flux = field.flux_function(seed)
    np.testing.assert_allclose(public_flux, oracle_flux[0], rtol=3.0e-12, atol=0.0)
    relative_drift = np.max(np.abs(oracle_flux - oracle_flux[0])) / abs(oracle_flux[0])
    assert relative_drift < 1.0e-8
    assert np.max(np.abs(result.points[:, 1])) < 2.0e-12


def test_loop_field_line_tangent_residual_converges_quadratically() -> None:
    field = CircularLoopField(1.0, 1.0, permeability=4.0 * np.pi)
    seed = np.array((0.55, 0.0, 0.35))

    def maximum_residual(output_step: float) -> float:
        result = trace_field_line(
            field,
            seed,
            options=_loop_options(output_step=output_step, max_arc_length=1.5),
            direction="forward",
        )
        segments = np.diff(result.points, axis=0)
        midpoints = 0.5 * (result.points[:-1] + result.points[1:])
        vectors = field.evaluate(midpoints)
        residual = np.linalg.norm(np.cross(segments, vectors), axis=1) / (
            np.linalg.norm(segments, axis=1) * np.linalg.norm(vectors, axis=1)
        )
        return float(np.max(residual))

    coarse = maximum_residual(0.08)
    refined = maximum_residual(0.02)

    assert refined < coarse / 10.0
    assert refined < 8.0e-5


def test_near_wire_loop_line_terminates_as_one_closed_orbit() -> None:
    field = CircularLoopField(1.0, 1.0, permeability=4.0 * np.pi)
    exclusion = ToroidalExclusion((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 1.0, 0.05)
    tolerance = 1.0e-3
    result = trace_field_line(
        field,
        seed=(0.72, 0.0, 0.0),
        options=TraceOptions(
            max_arc_length=4.0,
            max_step=0.02,
            rtol=1.0e-9,
            atol=1.0e-11,
            null_threshold=1.0e-14,
            output_step=0.01,
            method="DOP853",
            closure_tolerance=tolerance,
            closure_min_arc_length=1.0,
            closure_tangent_cosine=0.995,
        ),
        direction="forward",
        exclusions=(exclusion,),
    )

    assert result.forward is not None
    assert result.forward.termination is TerminationReason.CLOSED_LOOP
    assert np.linalg.norm(result.forward.terminal_point - result.seed) <= tolerance * 1.001
    assert 1.0 < result.forward.arc_length[-1] < 4.0
    assert np.max(np.abs(result.points[:, 1])) < 2.0e-12
