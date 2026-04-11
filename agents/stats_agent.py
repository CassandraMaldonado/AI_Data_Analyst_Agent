from typing import Dict, Any
import pandas as pd
from scipy.stats import pearsonr

from core.state import AnalystState
from tools.data_loader import load_dataset

def stats_agent(state: AnalystState) -> AnalystState:
    df, _ = load_dataset(state.dataset_path)

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    stats_results: Dict[str, Any] = {}

    if len(numeric_cols) >= 2:
        col1, col2 = numeric_cols[0], numeric_cols[1]
        clean_df = df[[col1, col2]].dropna()

        if len(clean_df) > 2:
            corr, p_value = pearsonr(clean_df[col1], clean_df[col2])
            stats_results["correlation_test"] = {
                "column_1": col1,
                "column_2": col2,
                "correlation": float(corr),
                "p_value": float(p_value),
                "interpretation": (
                    "Statistically significant relationship"
                    if p_value < 0.05
                    else "No statistically significant relationship"
                ),
            }

    state.stats_output = stats_results
    return state
