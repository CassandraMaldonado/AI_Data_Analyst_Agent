import logging
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from core.logging_config import configure_logging
from core.workflow import run_workflow

configure_logging()
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Data Analyst Agent")


class AnalyzeRequest(BaseModel):
    question: str
    dataset_path: str
    max_retries: Optional[int] = Field(default=2, ge=0, le=5)


@app.get("/")
def root():
    return {"message": "AI Data Analyst Agent is running"}

