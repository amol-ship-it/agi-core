"""
DreamCoder-style generative prior.

Learns P(child_op | parent_op) from successful programs and biases
random program generation toward known-good compositions.
"""

from __future__ import annotations
import random
from collections import Counter, defaultdict

from .types import Program, Primitive


class TransitionMatrix:
    """
    Learns P(child_op | parent_op) from successful programs.

    This biases random program generation toward compositions
    that have been observed in solutions. The key DreamCoder insight:
    not all compositions are equally likely to be useful.
    """

    def __init__(self, smoothing: float = 0.1):
        self._counts: dict[str, Counter] = defaultdict(Counter)
        self._totals: dict[str, int] = defaultdict(int)
        self._smoothing = smoothing

    def observe_program(self, program: Program) -> None:
        """Record parent->child transitions from a program tree."""
        for child in program.children:
            self._counts[program.root][child.root] += 1
            self._totals[program.root] += 1
            self.observe_program(child)

    def probability(self, parent_op: str, child_op: str, n_primitives: int) -> float:
        """P(child_op | parent_op) with Laplace smoothing."""
        count = self._counts[parent_op][child_op]
        total = self._totals[parent_op]
        smooth = self._smoothing
        return (count + smooth) / (total + smooth * n_primitives)

    def weighted_choice(self, parent_op: str, primitives: list[Primitive],
                        rng: random.Random) -> Primitive:
        """Choose a child primitive biased by the transition matrix."""
        n = len(primitives)
        if not self._totals.get(parent_op):
            return rng.choice(primitives)

        weights = [
            self.probability(parent_op, p.name, n) for p in primitives
        ]
        total_w = sum(weights)
        if total_w <= 0:
            return rng.choice(primitives)

        r = rng.random() * total_w
        cumulative = 0.0
        for i, w in enumerate(weights):
            cumulative += w
            if r <= cumulative:
                return primitives[i]
        return primitives[-1]

    @property
    def size(self) -> int:
        """Number of observed transitions."""
        return sum(self._totals.values())

    def __repr__(self):
        return f"TransitionMatrix({self.size} transitions, {len(self._counts)} parents)"
