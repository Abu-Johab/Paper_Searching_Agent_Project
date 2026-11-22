"""
hsi_research_agent/graph/checkpoints.py
State persistence and checkpointing utilities for the LangGraph workflow.
"""

import json
import logging
from typing import Any, Dict, Optional
from langgraph.checkpoint.memory import MemorySaver

from hsi_research_agent.storage.database import get_db_connection

logger = logging.getLogger("hsi_research_agent.checkpoints")


class SQLiteWorkflowCheckpointer:
    """Persistent SQLite checkpointer for saving and recovering workflow states."""

    def __init__(self, db_path: Optional[Any] = None):
        self.db_path = db_path

    def save_checkpoint(self, checkpoint_id: str, stage_name: str, state: Dict[str, Any]) -> None:
        """Save workflow state snapshot to SQLite."""
        sql = """
        INSERT OR REPLACE INTO checkpoints (checkpoint_id, stage_name, state_json, updated_at)
        VALUES (?, ?, ?, CURRENT_TIMESTAMP)
        """
        # Exclude un-serializable objects if any
        serializable_state = {}
        for k, v in state.items():
            try:
                json.dumps(v)
                serializable_state[k] = v
            except (TypeError, ValueError):
                serializable_state[k] = str(v)

        with get_db_connection(self.db_path) as conn:
            conn.execute(sql, (checkpoint_id, stage_name, json.dumps(serializable_state)))
        logger.info("Saved workflow checkpoint '%s' at stage '%s'", checkpoint_id, stage_name)

    def load_checkpoint(self, checkpoint_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve workflow state snapshot from SQLite."""
        sql = "SELECT state_json FROM checkpoints WHERE checkpoint_id = ?"
        with get_db_connection(self.db_path) as conn:
            row = conn.execute(sql, (checkpoint_id,)).fetchone()
            if row:
                return json.loads(row["state_json"])
        return None


def get_default_checkpointer() -> MemorySaver:
    """Return LangGraph in-memory checkpointer."""
    return MemorySaver()
