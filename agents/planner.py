from core.state import AnalystState
from tools.llm import ask_llm


def planner_agent(state: AnalystState) -> AnalystState:
    system_prompt = """
    You are a senior data analyst.
    Given a business question and dataset columns, produce a short ordered plan.
    Return each step on a new line.
    Be practical and concise.
    """

    user_prompt = f"""
    Question: {state.question}

    Dataset columns:
    {state.columns}
    """

    plan = ask_llm(system_prompt, user_prompt)
    state.analysis_steps = [line.strip("- ").strip() for line in plan.split("\n") if line.strip()]
    return state
