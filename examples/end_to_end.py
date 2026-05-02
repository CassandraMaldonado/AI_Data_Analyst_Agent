from __future__ import annotations

import io
import json
import os
import sys
import tempfile

import pandas as pd

# Project root on path when run as python examples/end_to_end.py
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from core.logging_config import configure_logging
from core.workflow import run_workflow


def main() -> None:
    configure_logging()
    if not os.getenv("OPENAI_API_KEY"):
        print("Skip: set OPENAI_API_KEY to run the LLM-backed workflow.")
        sys.exit(0)

    df = pd.DataFrame(
        {
            "region": ["A", "A", "B", "B", "A", "B"] * 5,
            "sales": [10, 12, 9, 11, 13, 8] * 5,
            "units": [2, 3, 2, 2, 3, 2] * 5,
        }
    )
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    content = buf.getvalue()

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(content)
        path = f.name

    try:
        out = run_workflow(
            "Is there an association between sales and units?",
            path,
            max_retries=1,
        )
        print(json.dumps(out, indent=2, default=str))
        assert "findings" in out or out.get("errors")
    finally:
        os.unlink(path)


if __name__ == "__main__":
    main()
