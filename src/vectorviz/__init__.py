"""VectorViz: reusable scientific vector-field models and field-line tracing."""

__version__ = "0.1.0"

from .core import Domain, ExclusionRegion, SphericalExclusion, ToroidalExclusion, VectorField
from .fields import (
    CircularLoopField,
    CompositeField,
    MagneticDipoleField,
    PointChargeField,
    UniformField,
)
from .tracing import (
    FieldLineTracer,
    TerminationReason,
    TraceBranch,
    TraceDirection,
    TraceOptions,
    TraceResult,
    trace_field_line,
)

__all__ = [
    "CircularLoopField",
    "CompositeField",
    "Domain",
    "ExclusionRegion",
    "FieldLineTracer",
    "MagneticDipoleField",
    "PointChargeField",
    "SphericalExclusion",
    "TerminationReason",
    "ToroidalExclusion",
    "TraceBranch",
    "TraceDirection",
    "TraceOptions",
    "TraceResult",
    "UniformField",
    "VectorField",
    "__version__",
    "trace_field_line",
]
