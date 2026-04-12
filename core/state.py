from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class AnalystState(BaseModel):
    question: str
    dataset_path: str
    columns: List[str] = Field(default_factory=list)

    analysis_steps: List[str] = Field(default_factory=list)
    execution_output: Optional[Dict[str, Any]] = None
    stats_output: Optional[Dict[str, Any]] = None
    insights: Optional[str] = None
    critique: Optional[str] = None
    final_answer: Optional[Dict[str, Any]] = None
