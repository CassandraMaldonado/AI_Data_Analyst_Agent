import logging
from typing import Any

from core.graph import build_graph
from core.logging_config import configure_logging
from tools.data_loader import load_dataset

logger = logging.getLogger(__name__)

_compiled = None

def _get_graph():
    global _compiled
    if _compiled is None:
        _compiled = build_graph()
    return _compiled
