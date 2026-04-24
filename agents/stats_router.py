import logging
import re

from core.state import AnalystState
from schemas.outputs import StatsIntent, StatsRouterOutput
from tools.llm import ask_llm_structured

logger = logging.getLogger(__name__)


def _normalize_col(name: str | None, columns: list[str]) -> str | None:
    if not name:
        return None
    lower_map = {c.lower(): c for c in columns}
    return lower_map.get(name.strip().lower(), None)


def stats_router_agent(state: AnalystState) -> AnalystState:
    cols = ", ".join(state.columns)
    system_prompt = """You classify which statistical approach fits the user's question.
Pick intent:
- describe: summaries / distributions only
- correlation: linear/monotonic association between two numeric variables
- two_sample: compare a numeric outcome between two groups (explicit groups or A/B)
- chi_square: association between two categorical variables
- anova_one_way: numeric outcome across 3+ groups
- custom_code: none of the above fits; sandbox code may be needed
- none: cannot determine

Map column_a, column_b, group_column to actual dataset column names exactly as listed."""

    user_prompt = f"""Question: {state.question}

Columns: {cols}

Analysis plan:
{state.analysis_steps}
"""

    try:
        router = ask_llm_structured(
            StatsRouterOutput,
            system_prompt,
            user_prompt,
        )
        router.column_a = _normalize_col(router.column_a, state.columns)
        router.column_b = _normalize_col(router.column_b, state.columns)
        router.group_column = _normalize_col(router.group_column, state.columns)

        qlow = state.question.lower()
        if "spearman" in qlow or "monotonic" in qlow:
            router.prefer_spearman = True

        state.stats_router = router.model_dump(mode="json")
    except Exception as e:
        logger.exception("Stats router failed")
        state.errors.append(f"stats_router: {e}")
        fallback = StatsRouterOutput(
            intent=StatsIntent.DESCRIBE,
            confidence=0.3,
            reasoning=f"fallback after error: {e}",
        )
        state.stats_router = fallback.model_dump(mode="json")

    return state
