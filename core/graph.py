# LangGraph orchestration.

from __future__ import annotations

import logging
from typing import Literal

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from agents.critic import critic_agent
from agents.executor import executor_agent
from agents.insights_agent import insight_agent
from agents.planner import planner_agent
from agents.stats_agent import stats_agent
from agents.stats_router import stats_router_agent
from core.state import AnalystState

logger = logging.getLogger(__name__)


class WorkflowState(TypedDict, total=False):
    question: str
    dataset_path: str
    columns: list[str]
    analysis_steps: list[str]
    execution_output: dict | None
    stats_router: dict | None
    stats_output: dict | None
    insights: str | None
    critique: str | None
    critique_result: dict | None
    critique_feedback: str | None
    retry_count: int
    max_retries: int
    errors: list[str]
    final_answer: dict | None


def _as_model(s: WorkflowState) -> AnalystState:
    return AnalystState.model_validate(s)


def _dump(updates: AnalystState) -> dict:
    return updates.model_dump()


def planner_node(state: WorkflowState) -> WorkflowState:
    try:
        s = _as_model(state)
        return _dump(planner_agent(s))
    except Exception as e:
        logger.exception("planner_node")
        errs = list(state.get("errors") or [])
        errs.append(f"planner_node: {e}")
        return {"errors": errs}


def executor_node(state: WorkflowState) -> WorkflowState:
    try:
        s = _as_model(state)
        return _dump(executor_agent(s))
    except Exception as e:
        logger.exception("executor_node")
        errs = list(state.get("errors") or [])
        errs.append(f"executor_node: {e}")
        return {"errors": errs}


def stats_router_node(state: WorkflowState) -> WorkflowState:
    try:
        s = _as_model(state)
        return _dump(stats_router_agent(s))
    except Exception as e:
        logger.exception("stats_router_node")
        errs = list(state.get("errors") or [])
        errs.append(f"stats_router_node: {e}")
        return {"errors": errs}


def stats_node(state: WorkflowState) -> WorkflowState:
    try:
        s = _as_model(state)
        return _dump(stats_agent(s))
    except Exception as e:
        logger.exception("stats_node")
        errs = list(state.get("errors") or [])
        errs.append(f"stats_node: {e}")
        return {"errors": errs}


def insights_node(state: WorkflowState) -> WorkflowState:
    try:
        s = _as_model(state)
        return _dump(insight_agent(s))
    except Exception as e:
        logger.exception("insights_node")
        errs = list(state.get("errors") or [])
        errs.append(f"insights_node: {e}")
        return {"errors": errs}


def critic_node(state: WorkflowState) -> WorkflowState:
    try:
        s = _as_model(state)
        return _dump(critic_agent(s))
    except Exception as e:
        logger.exception("critic_node")
        errs = list(state.get("errors") or [])
        errs.append(f"critic_node: {e}")
        return {"errors": errs}


def finalize_node(state: WorkflowState) -> WorkflowState:
    s = _as_model(state)
    cr = s.critique_result or {}
    supported = bool(cr.get("supported", False))
    conf = "high" if supported else "low"
    return {
        "final_answer": {
            "question": s.question,
            "analysis_steps": s.analysis_steps,
            "findings": s.insights,
            "evidence": s.stats_output,
            "critique": s.critique,
            "critique_structured": s.critique_result,
            "confidence": conf,
            "errors": s.errors,
        }
    }


def route_after_critic(
    state: WorkflowState,
) -> Literal["finalize", "insights", "stats", "executor"]:
    cr = state.get("critique_result") or {}
    if cr.get("supported") or cr.get("retry_stage") in ("none", None):
        return "finalize"
    if not state.get("critique_feedback"):
        return "finalize"
    rs = cr.get("retry_stage", "none")
    if rs == "insights":
        return "insights"
    if rs == "stats":
        return "stats"
    if rs == "executor":
        return "executor"
    return "finalize"


def build_graph():
    g = StateGraph(WorkflowState)
    g.add_node("planner", planner_node)
    g.add_node("executor", executor_node)
    g.add_node("stats_router", stats_router_node)
    g.add_node("stats", stats_node)
    g.add_node("insights", insights_node)
    g.add_node("critic", critic_node)
    g.add_node("finalize", finalize_node)

    g.add_edge(START, "planner")
    g.add_edge("planner", "executor")
    g.add_edge("executor", "stats_router")
    g.add_edge("stats_router", "stats")
    g.add_edge("stats", "insights")
    g.add_edge("insights", "critic")

    g.add_conditional_edges(
        "critic",
        route_after_critic,
        {
            "finalize": "finalize",
            "insights": "insights",
            "stats": "stats",
            "executor": "executor",
        },
    )

    g.add_edge("finalize", END)
    return g.compile()
