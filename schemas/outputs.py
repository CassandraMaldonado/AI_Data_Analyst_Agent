from __future__ import annotations

from enum import Enum
from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, Field

class PlannerOutput(BaseModel):
    steps: list[str] = Field(description="Ordered analytical steps")

class ExecutionSummary(BaseModel):
    row_count: int
    column_count: int
    columns: list[str]
    numeric_columns: list[str] = Field(default_factory=list)
    categorical_columns: list[str] = Field(default_factory=list)
    summary: dict[str, Any] = Field(default_factory=dict)
    preview: list[dict[str, Any]] = Field(default_factory=list)
    optional_queries: dict[str, Any] = Field(default_factory=dict)
