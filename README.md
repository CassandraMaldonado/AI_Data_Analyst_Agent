# AI Data Analyst Agent
An autonomous multi-agent system that performs end-to-end data analysis from question understanding to statistical inference and executive insights. 

## Overview
Traditional data analysis requires manual SQL, Python and statistical reasoning. This project builds an AI system that can:
- Understand natural language business questions.
- Plan analytical steps.
- Execute Python/SQL code.
- Run statistical tests.
- Generate structured, explainable insights.
- Self-validate outputs to reduce hallucinations.

The goal is to simulate a **real data analyst workflow**, not just a chatbot.

---

## Architecture

Multi-agent system orchestrated using LangGraph:

1. **Planner Agent**
   - Breaks down user query into analytical steps.

2. **Execution Agent**
   - Writes and executes Python/SQL (Pandas, DuckDB).

3. **Statistics Agent**
   - Runs regressions, hypothesis tests, correlations.

4. **Insight Agent**
   - Translates outputs into business insights.

5. **Critic Agent**
   - Validates results, checks for logical/statistical errors.

---

## 🔄 Workflow

User Question → Planner → Execution → Statistics → Insight → Critic → Final Answer

---

## 📊 Example

**Input:**
