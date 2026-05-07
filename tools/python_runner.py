"""Restricted execution for LLM-generated analysis code."""

from __future__ import annotations

import ast
import io
import logging
import multiprocessing
import traceback
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_ALLOWED_MODULES = frozenset(
    {"pandas", "numpy", "np", "pd", "scipy", "math", "statistics"}
)
_ALLOWED_IMPORT_FROM = frozenset({"pandas", "numpy", "scipy", "scipy.stats"})
_DISALLOWED_NAMES = frozenset(
    {
        "open",
        "exec",
        "eval",
        "compile",
        "__import__",
        "breakpoint",
        "input",
        "globals",
        "locals",
        "vars",
        "getattr",
        "setattr",
        "delattr",
    }
)


class UnsafeCodeError(ValueError):
    pass


def _minimal_builtins() -> dict[str, Any]:
    return {
        "len": len,
        "range": range,
        "min": min,
        "max": max,
        "sum": sum,
        "abs": abs,
        "round": round,
        "enumerate": enumerate,
        "zip": zip,
        "bool": bool,
        "int": int,
        "float": float,
        "str": str,
        "list": list,
        "dict": dict,
        "tuple": tuple,
        "set": set,
        "sorted": sorted,
        "print": print,
    }


def _validate_ast(source: str) -> None:
    tree = ast.parse(source, mode="exec")
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                base = alias.name.split(".")[0]
                if base not in _ALLOWED_MODULES:
                    raise UnsafeCodeError(f"Import not allowed: {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                base = node.module.split(".")[0]
                if base not in _ALLOWED_IMPORT_FROM:
                    raise UnsafeCodeError(f"Import from not allowed: {node.module}")
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in _DISALLOWED_NAMES:
                raise UnsafeCodeError(f"Call not allowed: {node.func.id}")
        elif isinstance(node, ast.Attribute):
            if isinstance(node.attr, str) and node.attr.startswith("__"):
                raise UnsafeCodeError("Dunder attribute access not allowed")


def _run_in_process(
    source: str,
    df_dict: dict[str, Any],
    out_queue: multiprocessing.Queue,
) -> None:
    df = pd.DataFrame(df_dict)
    ns: dict[str, Any] = {
        "__builtins__": _minimal_builtins(),
        "pd": pd,
        "np": np,
        "pandas": pd,
        "numpy": np,
        "df": df,
        "result": {},
    }
    buf_out = io.StringIO()
    buf_err = io.StringIO()
    try:
        with redirect_stdout(buf_out), redirect_stderr(buf_err):
            exec(compile(source, "<analysis>", "exec"), ns, ns)
        out_queue.put(
            {
                "ok": True,
                "result": ns.get("result", {}),
                "stdout": buf_out.getvalue(),
                "stderr": buf_err.getvalue(),
            }
        )
    except Exception:
        out_queue.put(
            {
                "ok": False,
                "error": traceback.format_exc(),
                "stdout": buf_out.getvalue(),
                "stderr": buf_err.getvalue(),
            }
        )


@dataclass
class SafePythonResult:
    ok: bool
    result: dict[str, Any]
    stdout: str
    stderr: str
    error: str | None = None


def run_safe_analysis(
    code: str,
    df: pd.DataFrame,
    *,
    timeout_sec: float = 25.0,
) -> SafePythonResult:
    """Execute user/LLM code with a minimal namespace (df, pd, np, result)."""
    code = code.strip()
    if not code:
        return SafePythonResult(
            ok=False, result={}, stdout="", stderr="", error="empty code"
        )
    try:
        _validate_ast(code)
    except UnsafeCodeError as e:
        logger.warning("AST validation rejected code: %s", e)
        return SafePythonResult(
            ok=False, result={}, stdout="", stderr="", error=str(e)
        )

    df_dict = df.to_dict(orient="list")

    ctx = multiprocessing.get_context("spawn")
    q: multiprocessing.Queue = ctx.Queue()
    p = ctx.Process(target=_run_in_process, args=(code, df_dict, q))
    p.start()
    p.join(timeout=timeout_sec)
    if p.is_alive():
        p.terminate()
        p.join(timeout=2)
        return SafePythonResult(
            ok=False,
            result={},
            stdout="",
            stderr="",
            error=f"Timeout after {timeout_sec}s",
        )
    if q.empty():
        return SafePythonResult(
            ok=False,
            result={},
            stdout="",
            stderr="",
            error="Worker produced no output",
        )
    payload = q.get()
    if not payload.get("ok"):
        return SafePythonResult(
            ok=False,
            result={},
            stdout=payload.get("stdout", ""),
            stderr=payload.get("stderr", ""),
            error=payload.get("error", "execution error"),
        )
    return SafePythonResult(
        ok=True,
        result=payload.get("result") or {},
        stdout=payload.get("stdout", ""),
        stderr=payload.get("stderr", ""),
    )
