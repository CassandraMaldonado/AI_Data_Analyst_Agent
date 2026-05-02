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


class StatsIntent(str, Enum):
    DESCRIBE = "describe"
    CORRELATION = "correlation"
    TWO_SAMPLE = "two_sample"
    CHI_SQUARE = "chi_square"
    ANOVA_ONE_WAY = "anova_one_way"
    CUSTOM_CODE = "custom_code"
    NONE = "none"


class StatsRouterOutput(BaseModel):
    intent: StatsIntent
    confidence: float = Field(ge=0.0, le=1.0, default=0.8)
    column_a: Optional[str] = None
    column_b: Optional[str] = None
    group_column: Optional[str] = None
    reasoning: str = ""
    prefer_spearman: bool = False


