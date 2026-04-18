import json
from typing import Any, Type, TypeVar

from pydantic import BaseModel, ValidationError


SchemaType = TypeVar("SchemaType", bound=BaseModel)


class GuardrailValidationError(ValueError):
    """Raised when payloads fail schema validation."""


class GuardrailRunner:
    """Validation helper for strict node input/output contracts."""

    @staticmethod
    def validate_input(payload: Any, schema: Type[SchemaType]) -> SchemaType:
        try:
            return schema.model_validate(payload)
        except ValidationError as exc:
            raise GuardrailValidationError(f"Input validation failed for {schema.__name__}: {exc}") from exc

    @staticmethod
    def validate_output(payload: Any, schema: Type[SchemaType]) -> SchemaType:
        try:
            return schema.model_validate(payload)
        except ValidationError as exc:
            raise GuardrailValidationError(f"Output validation failed for {schema.__name__}: {exc}") from exc

    @staticmethod
    def parse_json_output(raw_text: str, schema: Type[SchemaType]) -> SchemaType:
        try:
            data = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise GuardrailValidationError(f"Model response is not valid JSON: {exc}") from exc
        return GuardrailRunner.validate_output(data, schema)
