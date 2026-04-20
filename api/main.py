from fastapi import FastAPI
from pydantic import BaseModel

from core.workflow import run_workflow

app = FastAPI(title="AI Data Analyst Agent")


class AnalyzeRequest(BaseModel):
    question: str
    dataset_path: str


@app.get("/")
def root():
    return {"message": "AI Data Analyst Agent is running"}


@app.post("/analyze")
def analyze(request: AnalyzeRequest):
    result = run_workflow(
        question=request.question,
        dataset_path=request.dataset_path,
    )
    return result
