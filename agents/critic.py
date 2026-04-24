import json
import logging

from core.state import AnalystState
from schemas.outputs import CritiqueResult, RetryStage
from tools.llm import ask_llm_structured

logger = logging.getLogger(__name__)

def critic_agent(state: AnalystState) -> AnalystState:
    system_prompt = """You critically review data analyses.
Return structured JSON matching the schema.

Rules:
- supported=true only if conclusions match evidence and statistical outputs.
- If unsupported, choose retry_stage: insights (rewrite narrative), stats (different/wrong tests), executor (data profiling wrong).
- Use retry_stage none if retrying would not help or evidence is fundamentally missing."""

    user_prompt = f"""Question:
{state.question}

Analysis steps:
{state.analysis_steps}

Execution output:
{json.dumps(state.execution_output, indent=2, default=str)}

Stats output:
{json.dumps(state.stats_output, indent=2, default=str)}

Draft insights:
{state.insights}
"""

    try:
        cr = ask_llm_structured(CritiqueResult, system_prompt, user_prompt)
        state.critique = cr.summary
        state.critique_result = cr.model_dump(mode="json")

        if not cr.supported and cr.retry_stage != RetryStage.NONE:
            if state.retry_count < state.max_retries:
                state.retry_count += 1
                state.critique_feedback = "\n".join(
                    [cr.summary] + [f"- {i}" for i in cr.issues]
                )
            else:
                state.critique_feedback = None
        else:
            state.critique_feedback = None
    except Exception as e:
        logger.exception("Critic structured call failed")
        state.errors.append(f"critic: {e}")
        fb = CritiqueResult(
            supported=False,
            summary=f"Critic unavailable ({e}); treat findings as unverified.",
            issues=["critic_agent_error"],
            retry_stage=RetryStage.NONE,
        )
        state.critique = fb.summary
        state.critique_result = fb.model_dump(mode="json")
        state.critique_feedback = None

    return state
