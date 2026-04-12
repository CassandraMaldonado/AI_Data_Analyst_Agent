import json
from core.state import AnalystState
from tools.llm import ask_llm


def insight_agent(state: AnalystState) -> AnalystState:
    system_prompt = """
    You are an expert data analyst communicating to a business stakeholder.
    Summarize findings clearly.
    Be precise, avoid hype, and mention uncertainty where needed.
    """

    user_prompt = f"""
    User question:
    {state.question}

    Analysis plan:
    {state.analysis_steps}

    Execution output:
    {json.dumps(state.execution_output, indent=2)}

    Statistical output:
    {json.dumps(state.stats_output, indent=2)}

    Write:
    1. Main findings
    2. Evidence
    3. Caveats
    """

    state.insights = ask_llm(system_prompt, user_prompt)
    return state
