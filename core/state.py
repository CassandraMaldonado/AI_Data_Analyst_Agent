from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

class AnalystState(BaseModel):
    question: str
    dataset_path: str
    columns: List[str] = Field(default_factory=list)

    analysis_steps: List[str] = Field(default_factory=list)
    execution_output: Optional[Dict[str, Any]] = None
    stats_router: Optional[Dict[str, Any]] = None
    stats_output: Optional[Dict[str, Any]] = None
    insights: Optional[str] = None
    critique: Optional[str] = None
    critique_result: Optional[Dict[str, Any]] = None
    critique_feedback: Optional[str] = None

    retry_count: int = 0
    max_retries: int = 2
    errors: List[str] = Field(default_factory=list)

    final_answer: Optional[Dict[str, Any]] = None
