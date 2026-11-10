"""VectorViz: reusable scientific vector-field models and field-line tracing."""

__version__ = "0.3.8"

from .core import Domain, ExclusionRegion, SphericalExclusion, ToroidalExclusion, VectorField
from .fields import (
    ChargedRingField,
    CircularLoopField,
    CompositeField,
    DielectricSphereField,
    InfiniteWireField,
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
    "ChargedRingField",
    "CircularLoopField",
    "CompositeField",
    "DielectricSphereField",
    "Domain",
    "ExclusionRegion",
    "FieldLineTracer",
    "InfiniteWireField",
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
