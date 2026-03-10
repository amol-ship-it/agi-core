"""
Default in-memory implementation of the Memory interface.

This is domain-agnostic. It lives in core/ because it has no
domain-specific dependencies — just stores and retrieves data.
"""

from __future__ import annotations
import json
import logging
from typing import Optional

from .types import LibraryEntry, Program, ScoredProgram
from .interfaces import Memory

logger = logging.getLogger(__name__)


class InMemoryStore(Memory):
    """
    Simple in-memory implementation of all 3 memory systems.

    Good enough for prototyping. Replace with persistent storage
    (SQLite, filesystem) for larger experiments.
    """

    def __init__(self):
        self._episodes: list[dict] = []
        self._library: list[LibraryEntry] = []
        self._solutions: dict[str, ScoredProgram] = {}

    # --- Episodic ---

    def record_episode(self, task_id, observation, program, score):
        self._episodes.append({
            "task_id": task_id,
            "observation": observation,
            "program": program,
            "score": score,
        })

    def replay_episodes(self, n=10):
        return self._episodes[-n:]

    # --- Library ---

    def get_library(self):
        return list(self._library)

    def add_to_library(self, entry):
        self._library.append(entry)
        logger.debug(f"Library += {entry.name} (size={entry.program.size}, useful={entry.usefulness:.1f})")

    def update_usefulness(self, name, delta):
        for entry in self._library:
            if entry.name == name:
                entry.usefulness += delta
                entry.reuse_count += 1 if delta > 0 else 0
                return

    # --- Solutions ---

    def store_solution(self, task_id, scored):
        self._solutions[task_id] = scored

    def get_solutions(self):
        return dict(self._solutions)

    # --- Persistence ---

    def save(self, path):
        """Save library to JSON (solutions + episodes are ephemeral)."""
        data = {
            "library": [
                {
                    "name": e.name,
                    "program": repr(e.program),
                    "usefulness": e.usefulness,
                    "reuse_count": e.reuse_count,
                    "source_tasks": e.source_tasks,
                    "domain": e.domain,
                }
                for e in self._library
            ],
            "solutions_count": len(self._solutions),
            "episodes_count": len(self._episodes),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Memory saved to {path} ({len(self._library)} library entries)")

    def load(self, path):
        """Load library from JSON. Programs stored as repr — need grammar to reconstruct."""
        with open(path) as f:
            data = json.load(f)
        logger.info(f"Memory loaded from {path} ({len(data.get('library', []))} library entries)")
        # Note: full reconstruction requires the grammar to parse repr back to Program.
        # For now, just load metadata. The caller should reconstruct programs.
        return data
