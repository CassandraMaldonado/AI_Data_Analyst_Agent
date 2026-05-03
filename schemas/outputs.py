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


class CorrelationResult(BaseModel):
    test: Literal["pearson", "spearman"] = "pearson"
    column_1: str
    column_2: str
    statistic: float
    p_value: float
    n: int


class TwoSampleResult(BaseModel):
    test: Literal["ttest_ind", "mannwhitneyu"]
    numeric_column: str
    group_column: str
    group_a: str
    group_b: str
    statistic: float
    p_value: float
    n_a: int
    n_b: int


class ChiSquareResult(BaseModel):
    test: Literal["chi2_contingency"] = "chi2_contingency"
    column_1: str
    column_2: str
    statistic: float
    p_value: float
    dof: int
    

class ANOVAResult(BaseModel):
    test: Literal["anova_one_way"] = "anova_one_way"
    numeric_column: str
    group_column: str
    statistic: float
    p_value: float
    groups: int

class DescribeOnlyResult(BaseModel):
    test: Literal["describe"] = "describe"
    notes: str = ""


AnyStatResult = Union[
    CorrelationResult,
    TwoSampleResult,
    ChiSquareResult,
    ANOVAResult,
    DescribeOnlyResult,
]


class StatsResultsBundle(BaseModel):
    router: StatsRouterOutput
    results: list[AnyStatResult] = Field(default_factory=list)
    custom_code_stderr: Optional[str] = None
    custom_payload: Optional[dict[str, Any]] = None
    errors: list[str] = Field(default_factory=list)
