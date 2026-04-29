import logging

from core.graph import build_graph
from core.logging_config import configure_logging
from tools.data_loader import load_dataset

logger = logging.getLogger(__name__)

_compiled = None

def _get_graph():
    global _compiled
    if _compiled is None:
        _compiled = build_graph()
    return _compiled

def run_workflow(
    question: str,
    dataset_path: str,
    *,
    max_retries: int = 2,
) -> dict[str, Any]:
    configure_logging()
    _, columns = load_dataset(dataset_path)

    initial = {
        "question": question,
        "dataset_path": dataset_path,
        "columns": columns,
        "retry_count": 0,
        "max_retries": max_retries,
        "errors": [],
    }

    try:
        final_state = _get_graph().invoke(initial)
    except Exception as e:
        logger.exception("Workflow invoke failed")
        return {
            "question": question,
            "analysis_steps": [],
            "findings": None,
            "evidence": None,
            "critique": str(e),
            "critique_structured": None,
            "confidence": "low",
            "errors": [str(e)],
        }

    out = final_state.get("final_answer")
    if isinstance(out, dict):
        return out
    logger.error("Missing final_answer in graph output")
    return {
        "question": question,
        "errors": final_state.get("errors") or ["missing final_answer"],
    }
