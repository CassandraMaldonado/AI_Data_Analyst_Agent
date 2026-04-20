import logging
from typing import Any

import duckdb
import pandas as pd

from core.state import AnalystState
from schemas.outputs import ExecutionSummary
from tools.data_loader import load_dataset

logger = logging.getLogger(__name__)


def executor_agent(state: AnalystState) -> AnalystState:
    try:
        df, columns = load_dataset(state.dataset_path)
        state.columns = columns

        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        categorical_cols = df.select_dtypes(exclude="number").columns.tolist()

        summary: dict[str, Any] = {}
        if numeric_cols:
            summary["numeric_describe"] = df[numeric_cols].describe().to_dict()

        cat_summary: dict[str, Any] = {}
        if categorical_cols:
            for col in categorical_cols[:10]:
                cat_summary[col] = df[col].astype(str).value_counts().head(10).to_dict()
            summary["categorical_top_values"] = cat_summary

        optional_queries: dict[str, Any] = {}
        con = duckdb.connect()
        con.register("df", df)

        date_cols = [c for c in columns if c.lower() == "date" or "date" in c.lower()]
        for dc in date_cols[:1]:
            try:
                query = f"""
                    SELECT "{dc}" AS d, COUNT(*) AS cnt
                    FROM df
                    GROUP BY "{dc}"
                    ORDER BY "{dc}"
                    LIMIT 25
                """
                optional_queries["time_preview"] = (
                    con.execute(query).df().to_dict(orient="records")
                )
            except Exception as e:
                logger.warning("DuckDB time preview skipped for %s: %s", dc, e)

        payload = ExecutionSummary(
            row_count=len(df),
            column_count=len(df.columns),
            columns=columns,
            numeric_columns=numeric_cols,
            categorical_columns=categorical_cols,
            summary=summary,
            preview=df.head(5).to_dict(orient="records"),
            optional_queries=optional_queries,
        )
        state.execution_output = payload.model_dump(mode="json")
    except Exception as e:
        logger.exception("Executor failed")
        state.errors.append(f"executor: {e}")
        state.execution_output = None
    return state
