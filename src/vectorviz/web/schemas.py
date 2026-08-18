"""Validated request and response models for the browser application."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

PresetName = Literal["electric_dipole", "magnetic_dipole", "uniform"]
SourceKind = Literal["positive", "negative", "dipole", "uniform"]


class SourceInput(BaseModel):
    """A user-positionable source marker in normalized scene coordinates."""

    model_config = ConfigDict(extra="forbid")

    x: float = Field(ge=-2.8, le=2.8)
    y: float = Field(ge=-2.8, le=2.8)
    kind: SourceKind
    strength: float = Field(default=1.0, ge=-10.0, le=10.0)


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
