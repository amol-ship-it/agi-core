# Decisions & Judgements — Chronological Log

**Author:** Claude (working with vibhor-77)
**Purpose:** Living document of all technical decisions, judgements, trade-offs, and rationale made during development. Newest entries at the bottom.

---

## Session 1 — Claude Mobile App (Early March 2026)

### Analysis: Repository Landscape

After analyzing all repositories under vibhor-77:

| Repository | What It Contains | Status |
|---|---|---|
| `agi-mvp0` | Early prototype | Superseded |
| `agi-mvp-codex` | Codex-based approach | Superseded |
| `agi-mvp-claude` | Symbolic regression composer + grid worlds | Useful components |
| `agi-mvp-no-noise` | Cleaned-up variant | Superseded |
| `agi-sota-prototypes` | UniversalSolver with Wake-Sleep across ARC + Zork | Key reference |
| `agi-mvp-arc-agi-1` | Production ARC solver with beam search + DreamCoder-style library | Key reference |
| `agi-mvp-general` | Four Pillars AGI agent, 287 primitives, most mature | Primary source |
| `agi-core` | **The canonical monorepo** (this project) | Active |

**Judgement:** The evolution across repos shows clear convergence toward the architecture described in the manifesto. Each repo explored a different facet (beam search, library learning, cross-domain transfer, evolutionary synthesis). `agi-core` should be the consolidation point.

### Understanding of the Vision

Vibhor's core claim: **There is one general learning algorithm.** Differences between learners (nematode, dog, human, AI) are in hardware and data stream, not in the algorithm itself. The 4 pillars:

1. **Feedback Loops** — Act in environment, observe consequences, compare predictions to reality
2. **Approximability** — Candidates approximate the true generating function with quantified error
3. **Abstraction & Composability** — Primitives compose into programs; recurring compositions compress into reusable library entries
4. **Exploration** — Balance exploitation of known strategies with exploration of novel ones

These interact in a compounding cycle: better abstractions shrink search, better exploration discovers higher-value abstractions, better approximation scores candidates reliably, feedback grounds everything in reality.

### Response to Skeptics

**"Aren't researchers already working on this?"**
Yes, in pieces. DreamCoder (Ellis et al., 2021) does wake-sleep library learning within single domains. Chollet's ARC framework defines the benchmark. Friston's Free Energy Principle provides theoretical grounding. LLM-based synthesis (Greenblatt) achieves high ARC scores. But nobody has built a single clean system that: (a) separates invariant loop from domain plugins, (b) demonstrates compounding across multiple unrelated domains, (c) shows transfer. The contribution is the integration and the empirical test.

**"Too high level, how will it work?"**
The manifesto provides a concrete 6-phase experimental roadmap with specific deliverables. Phase 0 (extract core) is done. Phase 1 (ARC-AGI-1 curriculum training) is in progress. Each phase tests a specific claim about generality.

**"Don't LLMs already do this?"**
No, for structural reasons: (1) No explicit compounding — LLMs can't permanently learn a new abstraction from a single interaction. (2) No inspectable library — knowledge is distributed across billions of parameters. (3) No closed-loop interaction at inference. (4) Supervision assumption — next-token prediction assumes training corpus is truth. (5) Resource intensity — training costs hundreds of millions. LLMs can serve as a component (heuristic guide, perceptual front-end) but should not be the entire architecture.

---

## Session 2 — Claude Code on Web (March 10, 2026)

### Decision: Repository Restructuring

**Context:** Files were flat in root directory. Manifesto describes a structured layout.
**Decision:** Restructure into `core/`, `grammars/`, `environments/`, `drives/`, `library/`, `experiments/`, `tests/`.
**Rationale:** The architecture should be visible in the file structure itself. The invariant/pluggable separation must be enforced at the directory level. This prevents accidental cross-contamination.

### Decision: Don't Use PySR or DreamCoder as Dependencies

**Context:** User asked whether to leverage existing PySR and DreamCoder packages.
**Decision:** Don't depend on either. Port ideas, not code.
**Rationale:**
- **PySR** is for symbolic regression specifically. It has its own search loop, which would bypass the universal core and break the "one algorithm" principle. Great for comparison baseline, wrong as a dependency.
- **DreamCoder** (original codebase) is OCaml+Python, poorly maintained, hard to integrate. The key ideas (wake-sleep library learning, transition matrix prior, compression) are better reimplemented cleanly within the existing architecture.
- The manifesto's whole point is that intelligence lives in the *loop and library*, not in a specific library's implementation. Dependencies would obscure this.

**What we ported instead:**
- DreamCoder's **transition matrix prior** P(child_op | parent_op) — biases program generation toward compositions observed in successful solutions
- DreamCoder's **library compression** — extract recurring sub-programs from solved tasks
- These are implemented directly in `core/learner.py` (the `TransitionMatrix` class and enhanced `sleep()` method), staying within the invariant core with no domain imports

### Decision: ARC-AGI Primitive Set (48 Primitives)

**Context:** `agi-mvp-general` has 287 primitives. How many to include in the clean `agi-core`?
**Decision:** Start with 48 carefully chosen primitives covering the most common ARC transformation categories.
**Rationale:** The manifesto's claim is about compounding from a *small* initial set. Starting with 287 would be testing search, not learning. 48 provides enough coverage for basic geometric, color, spatial, gravity, and pattern operations while leaving room for the library to discover compositions. Can always add more if the compounding curve plateaus due to primitive poverty rather than algorithmic limitation.

**Categories included:**
- Geometric (7): identity, rot90cw/ccw, rot180, mirror_h/v, transpose
- Color (11): invert, replace_bg, keep_c1-c9
- Spatial/Cropping (5): crop_nonzero, top/bottom/left/right_half
- Tiling/Scaling (4): tile_2x2/3x3, scale_2x/3x
- Gravity (4): down/up/left/right
- Pattern (4): outline, fill_enclosed, denoise_3x3, replace_bg_mc
- Logical (4): xor/or_halves_v/h
- Color removal (9): recolor_1-9_to_0
- Binary (1): overlay

### Decision: Living Documentation Strategy

**Context:** User wants chronological record of all prompts and decisions.
**Decision:** Maintain two living documents:
- `PROMPTS.md` — All user instructions in chronological order (the "what was asked")
- `DECISIONS.md` — All technical decisions and rationale (the "what was decided and why")
**Rationale:** This creates an inspectable reasoning trail, consistent with the manifesto's emphasis on explainability. It also means any future Claude session (or human reader) can understand the full trajectory of the project without access to ephemeral chat logs.

### Decision: Core Loop Must Never Import Domain Code

**Context:** User explicitly reminded this constraint.
**Verification:** Confirmed that `core/learner.py`, `core/interfaces.py`, `core/memory.py`, and `core/metrics.py` import only from Python standard library and from each other (`from .interfaces import ...`). Zero domain-specific imports. The `TransitionMatrix` class added to learner.py operates purely on `Program` and `Primitive` types defined in `interfaces.py` — these are domain-agnostic data structures.

### Judgement: Expected Baseline Performance on ARC-AGI-1

Based on analysis of the existing repos:
- `agi-mvp-arc-agi-1` achieved ~10% with pure beam search (as noted in the manifesto)
- `agi-mvp-general` with 287 primitives + evolutionary search + DSL synthesis achieves higher
- `agi-core` with 48 primitives + basic beam search should achieve roughly **5-15%** on the 400 training tasks in the first round
- The key metric is not the absolute number but whether it **increases across rounds** as the library grows
- Even a modest improvement (e.g., 8% round 1 -> 12% round 3) without new hand-coded primitives would validate the compounding claim

---

*This document will be updated with each new session and major decision.*
