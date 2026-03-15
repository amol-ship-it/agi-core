# CLAUDE.md — Persistent Instructions for Claude Code

These instructions apply to every session automatically.

## Development Philosophy

- **Scientific method**: Hypothesize, experiment, measure, iterate. Every change should be testable.
- **Fast incremental loops**: Short, tight feedback cycles. Small commits. Verify before moving on.
- **Parallel experiments**: Run multiple targeted, focused experiments concurrently when useful.

## Code Quality

- World-class: minimal, elegant, comprehensive, well-commented, working, well-tested.
- **TDD** (test-driven development) where possible and where it makes sense.
- Unit test AND integration test all code.
- Maintain high code and branch coverage; note coverage in documentation.
- Follow best practices for the language/framework in use.

## Dependencies

- Minimal pip utility installs are OK: pytest, pytest-cov, numpy, scipy, matplotlib, etc.
- Avoid calling external services.
- Avoid heavyweight external dependencies that make the code hard to understand.

## Verification

- If you claim something is done, verify it actually works.
- Before returning control to the user, confirm changes work as expected (run tests, check output).

## Documentation

- Keep ALL documentation up to date and consistent with the code at all times.
- Remove obsolete content. Add comprehensive docs for all new features.
- Use `docs/` folder when it makes sense; tie everything together in `README.md`.
- Maintain living documents:
  - `PROMPTS.md` — Chronological log of all user instructions/prompts
  - `DECISIONS.md` — Chronological log of all technical decisions, rationale, and results

## Git & GitHub

- Commit and push after every meaningful change.
- Clear, descriptive commit messages.
- Branch: develop on the designated feature branch per session.

## Hyperparameter Optimization

- **Always do sweet-spot optimization** for parameters and hyperparameters when making changes. Don't just pick a value — measure alternatives.
- Key hyperparameters to keep tuned: `energy_beta` (complexity penalty), `unsolved_weight` (quality discount for unsolved programs in sleep), `exhaustive_pair_top_k`, `exhaustive_triple_top_k`, `compute_cap` presets, `max_library_size`, `usefulness_decay`, `reuse_bonus`.
- When adding new features with tunable parameters, run at least 3 values (low/mid/high) on a quick subset to find the sweet spot before committing.
- Document measured results in DECISIONS.md so future sessions know what was tried.

## Core Architecture Invariant

- The core loop (`core/`) must NEVER import anything domain-specific.
- Each domain lives under `domains/<name>/` and provides a `DomainAdapter` plus all 4 interfaces (Environment, Grammar, DriveSignal, Memory).
- `common/benchmark.py` provides the benchmark runner, pipeline, presets, and progress tracking. The unified CLI is `python -m common --domain <name>`.
- Three primitive kinds: `transform` (Grid→Grid), `perception` (Grid→Value), `parameterized` (Values→Grid→Grid factory). Perception feeds into parameterized actions for transferable compositions.
