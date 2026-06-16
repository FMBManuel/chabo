"""
DBContext loader for the query rewriter (+ any other future dependent modules)

Loads a per-deployment YAML artifact containing an abstract describing the doc store and a glossary.
"""
import logging
import os
from typing import List, Dict, Any

import yaml
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class DBContext(BaseModel):
    """Per-deployment database-awareness object for the query rewriter."""
    abstract: str = ""
    glossary: List[Dict[str, Any]] = Field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        """True if no meaningful context is configured."""
        return not self.abstract.strip() and not self.glossary


def load_db_context(path: str) -> DBContext:
    """
    Load DBContext from a YAML file.

    Returns an empty DBContext if the file is missing

    Raises ValueError on malformed YAML
    """
    if not os.path.exists(path):
        logger.warning(f"DBContext file not found at {path}; returning empty context")
        return DBContext()

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        raise ValueError(f"Failed to parse DBContext YAML at {path}: {e}") from e

    if not isinstance(data, dict):
        raise ValueError(f"DBContext YAML at {path} must be a mapping at the top level, got {type(data).__name__}")

    # Deal with malformed db_context null values
    data = {k: v for k, v in data.items() if v is not None}

    return DBContext(**data)
