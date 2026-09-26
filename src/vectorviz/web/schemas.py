"""Validated request and response models for the browser application."""

from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    FiniteFloat,
    StrictBool,
    StrictInt,
    model_validator,
)

PresetName = Literal[
    "electric_dipole",
    "electric_quadrupole",
    "electric_hexagon",
    "electric_hexagon_alternating",
    "magnetic_dipole",
    "halbach_array",
    "current_loop",
    "charged_ring",
    "uniform",
]
# Presets whose geometry is fixed by the server and returned as read-only markers.
FIXED_PRESETS = frozenset({"current_loop", "charged_ring", "uniform"})
# Presets that share the point-charge contract: charge-only overrides,
# circular exclusions, budget by |q| and return-pair suppression.
ELECTRIC_PRESETS = frozenset(
    {
        "electric_dipole",
        "electric_quadrupole",
        "electric_hexagon",
        "electric_hexagon_alternating",
    }
)
SourceInputKind = Literal["positive", "negative", "dipole", "uniform"]
SourcePayloadKind = Literal[
    "positive",
    "negative",
    "dipole",
    "uniform",
    "wire_out",
    "wire_into",
    "ring_charge",
]
SourceStrengthUnit = Literal["nC", "A·m²", "A"]


class SeedMode(StrEnum):
    """Machine-readable category for a scene's seed placement strategy."""

    COVERAGE = "coverage"
    EQUAL_FLUX = "equal_flux"
    FEATURE = "feature"


class SourceInput(BaseModel):
    """A source in Cartesian metres, with nC charges or A·m² dipole moments."""

    model_config = ConfigDict(extra="forbid")

    x: float = Field(ge=-2.8, le=2.8)
    y: float = Field(ge=-2.8, le=2.8)
    kind: SourceInputKind
    strength: float = Field(
        default_factory=lambda: 1.0,
        ge=-10.0,
        le=10.0,
        description=(
            "Source strength; positive and dipole default to 1; negative defaults to -1. "
            "Charge kinds require a matching nonzero sign."
        ),
    )
    angle_deg: float | None = Field(
        default_factory=lambda: None,
        ge=0.0,
        lt=360.0,
        allow_inf_nan=False,
        description=(
            "Dipole moment angle in degrees, counterclockwise from +x toward +y. "
            "Valid only for dipole sources: dipoles cannot use null and omitted dipoles "
            "default to 90; non-dipole sources must omit this field."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def apply_kind_dependent_defaults(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        values = dict(data)
        if values.get("kind") != "dipole" and "angle_deg" in values:
            raise ValueError("angle_deg is only valid for dipole sources")
        if values.get("kind") == "negative" and "strength" not in values:
            values["strength"] = -1.0
        if values.get("kind") == "dipole" and "angle_deg" not in values:
            values["angle_deg"] = 90.0
        return values

    @model_validator(mode="after")
    def validate_charge_strength_sign(self) -> "SourceInput":
        if self.kind == "positive" and self.strength <= 0.0:
            raise ValueError(
                "positive source strength must be greater than 0; "
                "zero and negative values are invalid"
            )
        if self.kind == "negative" and self.strength >= 0.0:
            raise ValueError(
                "negative source strength must be less than 0; zero and positive values are invalid"
            )
        if self.kind == "dipole" and self.angle_deg is None:
            raise ValueError("dipole angle_deg cannot be null; omit it to use 90 degrees")
        if self.kind != "dipole" and self.angle_deg is not None:
            raise ValueError("angle_deg is only valid for dipole sources")
        return self


class SceneRequest(BaseModel):
    """Parameters that affect physical sampling and field-line tracing."""

    model_config = ConfigDict(extra="forbid")

    preset: PresetName = "electric_dipole"
    density: int = Field(default=18, ge=6, le=40)
    resolution: int = Field(default=72, ge=32, le=144)
    sources: list[SourceInput] | None = Field(default=None, min_length=1, max_length=8)

    @model_validator(mode="after")
    def validate_source_override(self) -> "SceneRequest":
        if self.sources is None:
            return self
        kinds = {source.kind for source in self.sources}
        if self.preset in ELECTRIC_PRESETS and not kinds <= {"positive", "negative"}:
            raise ValueError(f"{self.preset} only accepts positive and negative sources")
        if self.preset in {"magnetic_dipole", "halbach_array"} and kinds != {"dipole"}:
            raise ValueError(f"{self.preset} accepts dipole sources only")
        if self.preset in FIXED_PRESETS:
            raise ValueError(f"{self.preset} preset does not accept source overrides")
        return self


class _ResponsePayload(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DomainPayload(_ResponsePayload):
    x: tuple[FiniteFloat, FiniteFloat]
    y: tuple[FiniteFloat, FiniteFloat]
    coordinate_system: Literal["cartesian"]
    unit: Literal["m"]

    @model_validator(mode="after")
    def validate_bounds(self) -> "DomainPayload":
        if self.x[0] >= self.x[1] or self.y[0] >= self.y[1]:
            raise ValueError("domain lower bounds must be strictly less than upper bounds")
        return self


class ScalarPayload(_ResponsePayload):
    nx: Annotated[StrictInt, Field(gt=0)]
    ny: Annotated[StrictInt, Field(gt=0)]
    values: list[FiniteFloat | None]
    mask: list[StrictBool]
    scale: Literal["linear", "log"]
    label: str
    unit: str
    vmin: FiniteFloat
    vmax: FiniteFloat

    @model_validator(mode="after")
    def validate_grid_contract(self) -> "ScalarPayload":
        expected_size = self.nx * self.ny
        if len(self.values) != expected_size:
            raise ValueError("scalar values length must equal nx * ny")
        if len(self.mask) != expected_size:
            raise ValueError("scalar mask length must equal nx * ny")
        if any(
            masked != (value is None) for value, masked in zip(self.values, self.mask, strict=True)
        ):
            raise ValueError("scalar values must be null exactly where mask is true")
        if self.vmax <= self.vmin:
            raise ValueError("scalar vmax must be strictly greater than vmin")
        if self.scale == "log" and self.vmin <= 0.0:
            raise ValueError("log scalar scales require a positive vmin")
        return self


class LinePayload(_ResponsePayload):
    points: list[tuple[FiniteFloat, FiniteFloat]] = Field(min_length=2)
    direction: Literal[-1, 1]
    termination: str = Field(min_length=1)
    start_termination: str | None = Field(default=None, min_length=1)


class SourcePayload(_ResponsePayload):
    x: FiniteFloat
    y: FiniteFloat
    kind: SourcePayloadKind
    strength: FiniteFloat
    strength_unit: SourceStrengthUnit
    angle_deg: FiniteFloat | None = Field(
        default=None,
        ge=0.0,
        lt=360.0,
        description=(
            "Dipole moment angle in degrees, counterclockwise from +x toward +y. "
            "Valid only for dipole sources: dipoles cannot use null and omitted dipoles "
            "default to 90; non-dipole sources must omit this field."
        ),
    )

    @model_validator(mode="after")
    def validate_kind_contract(self) -> "SourcePayload":
        if self.kind == "dipole":
            if self.strength_unit != "A·m²":
                raise ValueError("dipole source strength_unit must be A·m²")
            if self.angle_deg is None:
                raise ValueError("dipole source angle_deg cannot be null")
            return self
        if self.angle_deg is not None:
            raise ValueError("angle_deg is only valid for dipole sources")
        if self.kind in {"positive", "negative"}:
            if self.strength_unit != "nC":
                raise ValueError("charge source strength_unit must be nC")
            if self.kind == "positive" and self.strength <= 0.0:
                raise ValueError("positive source strength must be greater than 0")
            if self.kind == "negative" and self.strength >= 0.0:
                raise ValueError("negative source strength must be less than 0")
        elif self.kind in {"wire_out", "wire_into"}:
            if self.strength_unit != "A":
                raise ValueError("wire source strength_unit must be A")
            if self.strength < 0.0:
                raise ValueError("wire source strength must be non-negative")
        elif self.kind == "ring_charge":
            if self.strength_unit != "nC":
                raise ValueError("ring_charge source strength_unit must be nC")
            if self.strength == 0.0:
                raise ValueError("ring_charge source strength must be nonzero")
        return self


class MetadataPayload(_ResponsePayload):
    title: str
    projection_note: str
    field_model: str
    seed_mode: SeedMode
    seed_description: str = Field(min_length=1)
    termination_counts: dict[str, Annotated[StrictInt, Field(ge=0)]]
    start_termination_counts: dict[str, Annotated[StrictInt, Field(ge=0)]]
    suppressed_count: Annotated[StrictInt, Field(ge=0)]
    rendered_line_count: Annotated[StrictInt, Field(ge=0)]


class SceneResponse(_ResponsePayload):
    domain: DomainPayload
    scalar: ScalarPayload
    lines: list[LinePayload]
    sources: list[SourcePayload]
    metadata: MetadataPayload

    @model_validator(mode="after")
    def validate_rendered_line_count(self) -> "SceneResponse":
        if self.metadata.rendered_line_count != len(self.lines):
            raise ValueError("metadata rendered_line_count must equal len(lines)")
        return self


class SourceSeparationCapability(_ResponsePayload):
    exclusive_minimum: FiniteFloat = Field(gt=0.0)
    unit: Literal["m"]


class PresetPayload(_ResponsePayload):
    id: PresetName
    label: str
    description: str
    source_separation: SourceSeparationCapability | None = None
