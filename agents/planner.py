import logging

from core.state import AnalystState
from schemas.outputs import PlannerOutput
from tools.llm import ask_llm_structured

logger = logging.getLogger(__name__)

def planner_agent(state: AnalystState) -> AnalystState:
    system_prompt = """You are a senior data analyst.
Given a business question and dataset columns, produce a short ordered plan.
Each step must be one clear action (one line)."""

    user_prompt = f"""Question: {state.question}

Dataset columns:
{state.columns}
"""

    try:
        plan = ask_llm_structured(
            PlannerOutput,
            system_prompt,
            user_prompt,
        )
        state.analysis_steps = [s.strip() for s in plan.steps if s.strip()]
    except Exception as e:
        logger.exception("Planner structured call failed")
        state.errors.append(f"planner: {e}")
        state.analysis_steps = [
            "Profile the dataset",
            "Answer the question with summaries and appropriate tests",
        ]
    return state
