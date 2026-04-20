from core.state import AnalystState
from tools.data_loader import load_dataset
from agents.planner import planner_agent
from agents.executor import executor_agent
from agents.stats_agent import stats_agent
from agents.insight_agent import insight_agent
from agents.critic import critic_agent


def run_workflow(question: str, dataset_path: str) -> dict:
    df, columns = load_dataset(dataset_path)

    state = AnalystState(
        question=question,
        dataset_path=dataset_path,
        columns=columns,
    )

    state = planner_agent(state)
    state = executor_agent(state)
    state = stats_agent(state)
    state = insight_agent(state)
    state = critic_agent(state)

    state.final_answer = {
        "question": state.question,
        "analysis_steps": state.analysis_steps,
        "findings": state.insights,
        "evidence": state.stats_output,
        "critique": state.critique,
        "confidence": "medium",
    }

    return state.final_answer
