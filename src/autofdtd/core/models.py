"""Shared frozen model, tagging, and serialization helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator


def json_ready(value: Any) -> Any:
    """Normalize arbitrary model payloads into JSON-safe Python values."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, BaseModel):
        return json_ready(value.model_dump(mode="json", exclude_none=True))
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    raise TypeError(f"unsupported JSON payload value: {type(value)!r}")


class AutoFDTDModel(BaseModel):
    """Frozen base model with copy/update and stable serialization helpers."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    def model_tag(self) -> str:
        """Return the explicit type tag when present, else the model class name."""
        tagged_type = getattr(self, "type", None)
        if isinstance(tagged_type, str) and tagged_type.strip():
            return tagged_type.strip()
        return self.__class__.__name__

    def copy_update(self, **updates: Any) -> Self:
        """Return a frozen copy with a shallow field update."""
        return self.model_copy(update=updates)

    def to_payload(self) -> dict[str, Any]:
        """Return a JSON-ready payload with stable field names."""
        return self.model_dump(mode="json", exclude_none=True)

    def to_json_text(self, *, indent: int = 2) -> str:
        """Return a stable JSON string for snapshots and debugging."""
        return self.model_dump_json(indent=indent, exclude_none=True)

    def write_json(self, path: str | Path, *, indent: int = 2) -> None:
        """Write the model payload to disk."""
        Path(path).write_text(self.to_json_text(indent=indent) + "\n", encoding="utf-8")


class TaggedModel(AutoFDTDModel):
    """Shared base for immutable tagged public and IR-facing models."""

    type: str = Field(default="")

    @field_validator("type")
    @classmethod
    def _validate_type(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("type tags must be non-empty")
        return stripped
