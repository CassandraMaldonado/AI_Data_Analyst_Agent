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


@app.post("/analyze")
def analyze(request: AnalyzeRequest):
    try:
        result = run_workflow(
            question=request.question,
            dataset_path=request.dataset_path,
            max_retries=request.max_retries or 2,
        )
        return result
    except FileNotFoundError as e:
        logger.warning("Dataset not found: %s", e)
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        logger.exception("Analyze failed")
        raise HTTPException(status_code=500, detail=str(e)) from e
