"""Validated request and response models for the browser application."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

PresetName = Literal["electric_dipole", "magnetic_dipole", "uniform"]
SourceKind = Literal["positive", "negative", "dipole", "uniform"]
SourceStrengthUnit = Literal["nC", "A·m²"]


class SourceInput(BaseModel):
    """A source in Cartesian metres, with nC charges or A·m² dipole moments."""

    model_config = ConfigDict(extra="forbid")

    x: float = Field(ge=-2.8, le=2.8)
    y: float = Field(ge=-2.8, le=2.8)
    kind: SourceKind
    strength: float = Field(
        default_factory=lambda: 1.0,
        ge=-10.0,
        le=10.0,
        description=(
            "Source strength; positive and dipole default to 1; negative defaults to -1. "
            "Charge kinds require a matching nonzero sign."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def default_strength_for_charge_kind(cls, data: Any) -> Any:
        if (
            isinstance(data, dict)
            and data.get("kind") == "negative"
            and "strength" not in data
        ):
            return {**data, "strength": -1.0}
        return data

    @model_validator(mode="after")
    def validate_charge_strength_sign(self) -> "SourceInput":
        if self.kind == "positive" and self.strength <= 0.0:
            raise ValueError(
                "positive source strength must be greater than 0; "
                "zero and negative values are invalid"
            )
        if self.kind == "negative" and self.strength >= 0.0:
            raise ValueError(
                "negative source strength must be less than 0; "
                "zero and positive values are invalid"
            )
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
        if self.preset == "electric_dipole" and not kinds <= {"positive", "negative"}:
            raise ValueError("electric_dipole only accepts positive and negative sources")
        if self.preset == "magnetic_dipole" and kinds != {"dipole"}:
            raise ValueError("magnetic_dipole accepts dipole sources only")
        if self.preset == "uniform":
            raise ValueError("uniform preset does not accept source overrides")
        return self


class DomainPayload(BaseModel):
    x: tuple[float, float]
    y: tuple[float, float]
    coordinate_system: Literal["cartesian"]
    unit: Literal["m"]


class ScalarPayload(BaseModel):
    nx: int
    ny: int
    values: list[float]
    mask: list[bool]
    scale: Literal["linear", "log"]
    label: str
    unit: str
    vmin: float
    vmax: float


class LinePayload(BaseModel):
    points: list[tuple[float, float]]
    direction: Literal[-1, 1]
    termination: str


class SourcePayload(BaseModel):
    x: float
    y: float
    kind: SourceKind
    strength: float
    strength_unit: SourceStrengthUnit


class MetadataPayload(BaseModel):
    title: str
    projection_note: str
    field_model: str
    seed_mode: str
    termination_counts: dict[str, int]


class SceneResponse(BaseModel):
    domain: DomainPayload
    scalar: ScalarPayload
    lines: list[LinePayload]
    sources: list[SourcePayload]
    metadata: MetadataPayload


class PresetPayload(BaseModel):
    id: PresetName
    label: str
    description: str
