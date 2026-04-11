from typing import Dict, Any
import pandas as pd
import duckdb

from core.state import AnalystState
from tools.data_loader import load_dataset


def executor_agent(state: AnalystState) -> AnalystState:
    df, columns = load_dataset(state.dataset_path)
    state.columns = columns

    output: Dict[str, Any] = {
        "row_count": len(df),
        "column_count": len(df.columns),
        "columns": columns,
        "summary": {},
        "preview": df.head(5).to_dict(orient="records"),
    }

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(exclude="number").columns.tolist()

    if numeric_cols:
        output["summary"]["numeric_describe"] = df[numeric_cols].describe().to_dict()

    if categorical_cols:
        cat_summary = {}
        for col in categorical_cols[:5]:
            cat_summary[col] = df[col].astype(str).value_counts().head(10).to_dict()
        output["summary"]["categorical_top_values"] = cat_summary

    # DuckDB exploration.
    con = duckdb.connect()
    con.register("df", df)

    if "date" in [c.lower() for c in columns]:
        try:
            query = """
                SELECT date, COUNT(*) as count
                FROM df
                GROUP BY date
                ORDER BY date
                LIMIT 20
            """
            output["time_preview"] = con.execute(query).df().to_dict(orient="records")
        except Exception:
            pass

    state.execution_output = output
    return state
