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
