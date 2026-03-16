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
    if prog.structural:
        d["structural"] = True
    return d


def _program_from_dict(d: dict) -> Program:
    """Reconstruct a Program tree from a dict."""
    return Program(
        root=d["root"],
        children=[_program_from_dict(c) for c in d.get("children", [])],
        params=d.get("params", {}),
        structural=d.get("structural", False),
    )


class InMemoryStore(Memory):
    """
    Simple in-memory implementation of all 3 memory systems.

    Good enough for prototyping. Replace with persistent storage
    (SQLite, filesystem) for larger experiments.

    Args:
        capacity: max library entries (0 = unbounded, for worker snapshots).
        reuse_bonus: scoring bonus per reuse_count for eviction ranking.
    """

    def __init__(self, capacity: int = 0, reuse_bonus: float = 2.0):
        self._episodes: list[dict] = []
        self._library: list[LibraryEntry] = []
        self._solutions: dict[str, ScoredProgram] = {}
        self._best_attempts: dict[str, ScoredProgram] = {}
        self._primitive_scores: dict[str, float] = {}
        self._capacity = capacity
        self._reuse_bonus = reuse_bonus

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

    def _eviction_score(self, entry: LibraryEntry) -> float:
        """Score for eviction ranking: higher = harder to evict."""
        return entry.usefulness + self._reuse_bonus * entry.reuse_count

    def add_to_library(self, entry: LibraryEntry) -> bool:
        """Add entry, evicting the weakest non-reused entry if at capacity.

        Returns True if accepted, False if rejected (too weak).
        """
        # Unbounded or under capacity: always accept
        if self._capacity <= 0 or len(self._library) < self._capacity:
            self._library.append(entry)
            logger.debug(f"Library += {entry.name} (size={entry.program.size}, useful={entry.usefulness:.1f})")
            return True

        # At capacity: find worst evictable entry (reuse_count == 0)
        new_score = self._eviction_score(entry)
        worst_idx = -1
        worst_score = float('inf')
        for i, e in enumerate(self._library):
            if e.reuse_count > 0:
                continue  # immune to eviction
            s = self._eviction_score(e)
            if s < worst_score:
                worst_score = s
                worst_idx = i

        if worst_idx < 0:
            # All entries have been reused — no evictable slot
            logger.debug(f"Library FULL (all reused), rejected {entry.name}")
            return False

        if new_score > worst_score:
            evicted = self._library[worst_idx]
            self._library[worst_idx] = entry
            logger.debug(
                f"Library evicted {evicted.name} (score={worst_score:.2f}) "
                f"for {entry.name} (score={new_score:.2f})"
            )
            return True

        logger.debug(f"Library rejected {entry.name} (score={new_score:.2f} <= worst={worst_score:.2f})")
        return False

    def update_usefulness(self, name: str, delta: float) -> None:
        for entry in self._library:
            if entry.name == name:
                entry.usefulness += delta
                entry.reuse_count += 1 if delta > 0 else 0
                return

    def prune_library(self, min_usefulness: float = 0.01) -> int:
        """Remove library entries that have decayed below threshold and were never reused."""
        before = len(self._library)
        self._library = [
            e for e in self._library
            if e.usefulness >= min_usefulness or e.reuse_count > 0
        ]
        pruned = before - len(self._library)
        if pruned > 0:
            logger.debug(f"Pruned {pruned} dead library entries (usefulness < {min_usefulness})")
        return pruned

    # --- Primitive ROI scores ---

    def get_primitive_scores(self) -> dict[str, float]:
        return dict(self._primitive_scores)

    def update_primitive_score(self, name: str, delta: float) -> None:
        self._primitive_scores[name] = self._primitive_scores.get(name, 0.0) + delta

    # --- Solutions ---

    def store_solution(self, task_id: str, scored: ScoredProgram) -> None:
        self._solutions[task_id] = scored

    def get_solutions(self) -> dict[str, ScoredProgram]:
        return dict(self._solutions)

    # --- Best attempts (unsolved) ---

    def store_best_attempt(self, task_id: str, scored: ScoredProgram) -> None:
        """Keep the lowest-error program found for an unsolved task."""
        existing = self._best_attempts.get(task_id)
        if existing is None or scored.prediction_error < existing.prediction_error:
            self._best_attempts[task_id] = scored

    def get_best_attempts(self) -> dict[str, ScoredProgram]:
        """Return all best unsolved attempts, keyed by task_id."""
        return dict(self._best_attempts)

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
            "best_attempts": {
                task_id: {
                    "program": _program_to_dict(sp.program),
                    "energy": sp.energy,
                    "prediction_error": sp.prediction_error,
                }
                for task_id, sp in self._best_attempts.items()
            },
            "primitive_scores": self._primitive_scores,
            "solutions_count": len(self._solutions),
            "best_attempts_count": len(self._best_attempts),
            "library_count": len(self._library),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(
            f"Culture saved to {path} ({len(self._library)} library, "
            f"{len(self._solutions)} solutions, {len(self._best_attempts)} best attempts)")

    def load_culture(self, path: str) -> None:
        """Load a culture file, reconstructing Programs from their serialized form."""
        with open(path) as f:
            data = json.load(f)

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

        for task_id, sol_data in data.get("solutions", {}).items():
            program = _program_from_dict(sol_data["program"])
            self._solutions[task_id] = ScoredProgram(
                program=program,
                energy=sol_data.get("energy", 0.0),
                prediction_error=sol_data.get("prediction_error", 0.0),
                complexity_cost=float(program.size),
                task_id=task_id,
            )

        for task_id, ba_data in data.get("best_attempts", {}).items():
            program = _program_from_dict(ba_data["program"])
            self._best_attempts[task_id] = ScoredProgram(
                program=program,
                energy=ba_data.get("energy", 0.0),
                prediction_error=ba_data.get("prediction_error", 0.0),
                complexity_cost=float(program.size),
                task_id=task_id,
            )

        # Load primitive scores
        self._primitive_scores.update(data.get("primitive_scores", {}))

        # Post-load truncation: if loaded library exceeds capacity, keep top-N
        if self._capacity > 0 and len(self._library) > self._capacity:
            before = len(self._library)
            self._library.sort(key=self._eviction_score, reverse=True)
            self._library = self._library[:self._capacity]
            logger.info(f"Culture truncated library {before} → {self._capacity}")

        logger.info(
            f"Culture loaded from {path} "
            f"({len(self._library)} library, {len(self._solutions)} solutions, "
            f"{len(self._best_attempts)} best attempts)")

