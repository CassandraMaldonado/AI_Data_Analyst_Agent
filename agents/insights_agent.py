import json
import logging

from core.state import AnalystState
from schemas.outputs import InsightReport
from tools.llm import ask_llm_structured

logger = logging.getLogger(__name__)


def insight_agent(state: AnalystState) -> AnalystState:
    system_prompt = """You are an expert data analyst communicating to a business stakeholder.
Produce structured output only: main_findings, evidence, caveats.
Be precise; avoid hype; acknowledge uncertainty."""

    feedback_block = ""
    if state.critique_feedback:
        feedback_block = f"\nPrior critique to address:\n{state.critique_feedback}\n"

    user_prompt = f"""User question:
{state.question}

Analysis plan:
{state.analysis_steps}

Execution output:
{json.dumps(state.execution_output, indent=2, default=str)}

Statistical output:
{json.dumps(state.stats_output, indent=2, default=str)}
{feedback_block}"""

    try:
        report = ask_llm_structured(InsightReport, system_prompt, user_prompt)
        lines = []
        lines.append("## Main findings")
        for f in report.main_findings:
            lines.append(f"- {f}")
        lines.append("\n## Evidence")
        for e in report.evidence:
            lines.append(f"- {e}")
        lines.append("\n## Caveats")
        for c in report.caveats:
            lines.append(f"- {c}")
        state.insights = "\n".join(lines)
    except Exception as e:
        logger.exception("Insight structured generation failed")
        state.errors.append(f"insights: {e}")
        state.insights = "Insights unavailable due to an internal error; see execution and stats JSON."

    return state
