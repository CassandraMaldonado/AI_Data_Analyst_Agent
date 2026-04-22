from __future__ import annotations

import logging
from typing import Any

import pandas as pd
from scipy.stats import chi2_contingency, f_oneway, mannwhitneyu, pearsonr, spearmanr, ttest_ind

from core.state import AnalystState
from schemas.outputs import (
    ANOVAResult,
    ChiSquareResult,
    CorrelationResult,
    CustomStatsCodeOutput,
    DescribeOnlyResult,
    StatsIntent,
    StatsResultsBundle,
    StatsRouterOutput,
    TwoSampleResult,
)
from tools.data_loader import load_dataset
from tools.llm import ask_llm_structured
from tools.python_runner import run_safe_analysis


def _pair_rho_p(
    a: pd.Series,
    b: pd.Series,
    *,
    spearman: bool,
) -> tuple[float, float]:
    if spearman:
        r = spearmanr(a, b)
    else:
        r = pearsonr(a, b)
    if hasattr(r, "statistic"):
        return float(r.statistic), float(r.pvalue)
    return float(r[0]), float(r[1])


def _run_custom_code(
    state: AnalystState, df: pd.DataFrame, router: StatsRouterOutput
) -> dict[str, Any] | None:
    sys_p = """You write short pandas/scipy code. Use variables df (DataFrame), pd, np.
You MUST assign a dict `result` with keys: statistic (float), p_value (float), interpretation (str).
No file or network I/O."""

    user_p = f"""Question: {state.question}

Columns: {list(df.columns)}

Router reasoning: {router.reasoning}
Execution summary keys: {list((state.execution_output or {}).keys())}
"""

    try:
        gen = ask_llm_structured(CustomStatsCodeOutput, sys_p, user_p)
        run = run_safe_analysis(gen.python_code, df)
        if run.ok and isinstance(run.result, dict):
            return {
                "custom_result": run.result,
                "stdout": run.stdout,
                "stderr": run.stderr,
            }
        return {
            "custom_result": {},
            "stdout": run.stdout,
            "stderr": (run.stderr or "") + (run.error or ""),
        }
    except Exception as e:
        logger.exception("Custom stats code path failed")
        return {"custom_error": str(e)}


def stats_agent(state: AnalystState) -> AnalystState:
    df, _ = load_dataset(state.dataset_path)
    raw_router = state.stats_router or {}
    try:
        router = StatsRouterOutput.model_validate(raw_router)
    except Exception:
        router = StatsRouterOutput(intent=StatsIntent.DESCRIBE, reasoning="invalid router")

    results: list[Any] = []
    errors: list[str] = []
    custom_stderr: str | None = None
    custom_payload: dict[str, Any] | None = None

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(exclude="number").columns.tolist()

    intent = router.intent

    try:
        if intent == StatsIntent.NONE or intent == StatsIntent.DESCRIBE:
            results.append(
                DescribeOnlyResult(
                    notes="Descriptive summaries only; see execution_output."
                )
            )

        elif intent == StatsIntent.CORRELATION:
            c1 = router.column_a or (numeric_cols[0] if len(numeric_cols) > 0 else None)
            c2 = router.column_b or (
                numeric_cols[1] if len(numeric_cols) > 1 else None
            )
            if c1 and c2 and c1 in df.columns and c2 in df.columns and c1 != c2:
                clean = df[[c1, c2]].dropna()
                if len(clean) > 2:
                    stat_val, p_val = _pair_rho_p(
                        clean[c1], clean[c2], spearman=router.prefer_spearman
                    )
                    test_name: Any = "spearman" if router.prefer_spearman else "pearson"
                    results.append(
                        CorrelationResult(
                            test=test_name,
                            column_1=c1,
                            column_2=c2,
                            statistic=stat_val,
                            p_value=p_val,
                            n=len(clean),
                        )
                    )
                else:
                    errors.append("Not enough paired rows for correlation.")
            else:
                errors.append("Could not resolve two numeric columns for correlation.")

        elif intent == StatsIntent.CHI_SQUARE:
            c1 = router.column_a or (cat_cols[0] if cat_cols else None)
            c2 = router.column_b or (cat_cols[1] if len(cat_cols) > 1 else None)
            if c1 and c2 and c1 in df.columns and c2 in df.columns:
                tab = pd.crosstab(df[c1].astype(str), df[c2].astype(str))
                if tab.size > 1 and tab.shape[0] > 1 and tab.shape[1] > 1:
                    chi2, p, dof, _ = chi2_contingency(tab)
                    results.append(
                        ChiSquareResult(
                            column_1=c1,
                            column_2=c2,
                            statistic=float(chi2),
                            p_value=float(p),
                            dof=int(dof),
                        )
                    )
                else:
                    errors.append("Contingency table degenerate for chi-square.")
            else:
                errors.append("Could not resolve two categorical columns.")

        elif intent == StatsIntent.TWO_SAMPLE:
            num = router.column_a
            grp = router.group_column
            if num is None and numeric_cols:
                num = numeric_cols[0]
            if grp is None:
                for c in cat_cols:
                    if df[c].astype(str).nunique() == 2:
                        grp = c
                        break
            if num and grp and num in df.columns and grp in df.columns:
                levels = df[grp].astype(str).dropna().unique().tolist()
                if len(levels) == 2:
                    a, b = levels[0], levels[1]
                    s1 = df.loc[df[grp].astype(str) == a, num].dropna()
                    s2 = df.loc[df[grp].astype(str) == b, num].dropna()
                    if len(s1) > 1 and len(s2) > 1:
                        # t-test unless very small samples
                        if len(s1) >= 8 and len(s2) >= 8:
                            t_stat, p = ttest_ind(s1, s2, equal_var=False)
                            test_used: Any = "ttest_ind"
                            stat = float(t_stat)
                        else:
                            u_stat, p = mannwhitneyu(s1, s2, alternative="two-sided")
                            test_used = "mannwhitneyu"
                            stat = float(u_stat)
                        results.append(
                            TwoSampleResult(
                                test=test_used,
                                numeric_column=num,
                                group_column=grp,
                                group_a=a,
                                group_b=b,
                                statistic=stat,
                                p_value=float(p),
                                n_a=len(s1),
                                n_b=len(s2),
                            )
                        )
                    else:
                        errors.append("Insufficient data for two-sample test.")
                else:
                    errors.append("Group column must have exactly two levels for two_sample.")
            else:
                errors.append("Could not resolve numeric + group columns for two-sample test.")

        elif intent == StatsIntent.ANOVA_ONE_WAY:
            num = router.column_a
            grp = router.group_column
            if num is None and numeric_cols:
                num = numeric_cols[0]
            if grp is None:
                for c in cat_cols:
                    if df[c].nunique() >= 3:
                        grp = c
                        break
            if num and grp and num in df.columns and grp in df.columns:
                groups = []
                for level in df[grp].dropna().astype(str).unique():
                    ser = df.loc[df[grp].astype(str) == level, num].dropna()
                    if len(ser) > 0:
                        groups.append(ser)
                if len(groups) >= 3:
                    f_stat, p = f_oneway(*groups)
                    results.append(
                        ANOVAResult(
                            numeric_column=num,
                            group_column=grp,
                            statistic=float(f_stat),
                            p_value=float(p),
                            groups=len(groups),
                        )
                    )
                else:
                    errors.append("Need 3+ groups for one-way ANOVA.")
            else:
                errors.append("Could not resolve columns for ANOVA.")

        elif intent == StatsIntent.CUSTOM_CODE:
            custom_payload = _run_custom_code(state, df, router)
            custom_stderr = (custom_payload or {}).get("stderr") if custom_payload else None

        # Fallback when nothing produced
        if not results and intent not in (StatsIntent.CUSTOM_CODE,):
            if len(numeric_cols) >= 2:
                c1, c2 = numeric_cols[0], numeric_cols[1]
                clean = df[[c1, c2]].dropna()
                if len(clean) > 2:
                    stat_val, p_val = _pair_rho_p(
                        clean[c1], clean[c2], spearman=False
                    )
                    results.append(
                        CorrelationResult(
                            column_1=c1,
                            column_2=c2,
                            statistic=stat_val,
                            p_value=p_val,
                            n=len(clean),
                        )
                    )

    except Exception as e:
        logger.exception("Primary stats dispatch failed")
        errors.append(str(e))

    bundle = StatsResultsBundle(
        router=router,
        results=results,
        custom_code_stderr=custom_stderr,
        custom_payload=custom_payload,
        errors=errors,
    )

    state.stats_output = bundle.model_dump(mode="json")
    return state
