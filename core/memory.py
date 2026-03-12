"""
Default in-memory implementation of the Memory interface.

This is domain-agnostic. It lives in core/ because it has no
domain-specific dependencies — just stores and retrieves data.
"""

from __future__ import annotations
import json
import logging
from typing import Any, Optional

from .types import LibraryEntry, Program, ScoredProgram
from .interfaces import Memory

logger = logging.getLogger(__name__)


# =============================================================================
# Program serialization (JSON-safe tree representation)
# =============================================================================

def _program_to_dict(prog: Program) -> dict:
    """Serialize a Program tree to a JSON-safe dict."""
    d = {"root": prog.root}
    if prog.children:
        d["children"] = [_program_to_dict(c) for c in prog.children]
    if prog.params:
        d["params"] = prog.params
    return d


def _program_from_dict(d: dict) -> Program:
    """Reconstruct a Program tree from a dict."""
    return Program(
        root=d["root"],
        children=[_program_from_dict(c) for c in d.get("children", [])],
        params=d.get("params", {}),
    )


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

    def record_episode(self, task_id: str, observation: Any, program: Optional[Program], score: float) -> None:
        self._episodes.append({
            "task_id": task_id,
            "observation": observation,
            "program": program,
            "score": score,
        })

    def replay_episodes(self, n: int = 10) -> list[dict]:
        return self._episodes[-n:]

    # --- Library ---

    def get_library(self) -> list[LibraryEntry]:
        return list(self._library)

    def add_to_library(self, entry: LibraryEntry) -> None:
        self._library.append(entry)
        logger.debug(f"Library += {entry.name} (size={entry.program.size}, useful={entry.usefulness:.1f})")

    def update_usefulness(self, name: str, delta: float) -> None:
        for entry in self._library:
            if entry.name == name:
                entry.usefulness += delta
                entry.reuse_count += 1 if delta > 0 else 0
                return

    # --- Solutions ---

    def store_solution(self, task_id: str, scored: ScoredProgram) -> None:
        self._solutions[task_id] = scored

    def get_solutions(self) -> dict[str, ScoredProgram]:
        return dict(self._solutions)

    # --- Persistence (culture file) ---

    def save_culture(self, path: str) -> None:
        """
        Save the learned culture (library + solution summaries) to JSON.

        The culture file is the cross-run knowledge transfer mechanism.
        Training produces a culture file; evaluation loads it.
        """
        data = {
            "version": "1.0",
            "library": [
                {
                    "name": e.name,
                    "program": _program_to_dict(e.program),
                    "usefulness": e.usefulness,
                    "reuse_count": e.reuse_count,
                    "source_tasks": e.source_tasks,
                    "domain": e.domain,
                }
                for e in self._library
            ],
            "solutions": {
                task_id: {
                    "program": _program_to_dict(sp.program),
                    "energy": sp.energy,
                    "prediction_error": sp.prediction_error,
                }
                for task_id, sp in self._solutions.items()
            },
            "solutions_count": len(self._solutions),
            "library_count": len(self._library),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Culture saved to {path} ({len(self._library)} library, {len(self._solutions)} solutions)")

    def load_culture(self, path: str) -> None:
        """
        Load a culture file, reconstructing Programs from their serialized form.

        This restores the library so that evaluation can use concepts learned
        during training. Solutions are loaded as well for culture transfer.
        """
        with open(path) as f:
            data = json.load(f)

        # Reconstruct library entries
        for entry_data in data.get("library", []):
            program = _program_from_dict(entry_data["program"])
            entry = LibraryEntry(
                name=entry_data["name"],
                program=program,
                usefulness=entry_data.get("usefulness", 1.0),
                reuse_count=entry_data.get("reuse_count", 0),
                source_tasks=entry_data.get("source_tasks", []),
                domain=entry_data.get("domain", ""),
            )
            self._library.append(entry)

        # Reconstruct solutions for culture transfer
        for task_id, sol_data in data.get("solutions", {}).items():
            program = _program_from_dict(sol_data["program"])
            sp = ScoredProgram(
                program=program,
                energy=sol_data.get("energy", 0.0),
                prediction_error=sol_data.get("prediction_error", 0.0),
                complexity_cost=float(program.size),
                task_id=task_id,
            )
            self._solutions[task_id] = sp

        logger.info(
            f"Culture loaded from {path} "
            f"({len(self._library)} library, {len(self._solutions)} solutions)")

    # Legacy API compatibility
    def save(self, path: str) -> None:
        """Save library to JSON (legacy format for backward compat)."""
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

    def load(self, path: str):
        """Load from JSON (legacy API). Returns raw data for backward compat."""
        with open(path) as f:
            data = json.load(f)
        logger.info(f"Memory loaded from {path} ({len(data.get('library', []))} library entries)")
        return data
