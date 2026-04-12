from core.state import AnalystState
from tools.llm import ask_llm

def critic_agent(state: AnalystState) -> AnalystState:
    system_prompt = """
    You are a critical reviewer of data science analyses.
    Check whether conclusions are supported by the evidence.
    Identify overclaims, weak logic, or missing caveats.
    Keep the review concise.
    """

    user_prompt = f"""
    Question:
    {state.question}

    Analysis steps:
    {state.analysis_steps}

    Execution output:
    {state.execution_output}

    Stats output:
    {state.stats_output}

    Draft insights:
    {state.insights}
    """

    state.critique = ask_llm(system_prompt, user_prompt)
    return state
