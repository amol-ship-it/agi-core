# Prompts & Instructions — Chronological Log

**Author:** vibhor-77
**Purpose:** Living document of all instructions, prompts, and directives given to Claude across sessions. Newest entries at the bottom.

---

## Session 1 — Claude Mobile App (Early March 2026)

### Prompt 1: Initial Vision & Repository Analysis

> Can you read all my GitHub repositories which are already public, user vibhor-77? And just try to understand deeply how I am trying to build AGI.
> Also, take into account all my sessions with Claude.
> Then I want to propose a simple general purpose framework to you for AGI which can be plugged and adapted for any domain. But first do the above.
>
> My GitHub website is github.com/vibhor-77

### Prompt 2: The Core Proposal — 4 Pillars & Continuous Learning

> My proposal is a combination of the 4 pillars principles applied in a particular way to form a general continuous learning algorithm which is similar to how humans and even animals learn continuously from conception, birth and as they grow up.
>
> My claim is that there is a general learning algorithm which is the same, the only difference is how it is adapted depending on the sensors, actuators, compute and memory hardware, and the training data the organism is exposed to through its sensory organs, and also the effect on the environment from the organism's actuators, i.e. motor organs as measured by its sensory organs.
>
> The sensors provide a continuous stream of input to the brain. The brain tries to make sense of the world, i.e. explain its sensory streams as well as the effect of its actuators/motor organs in the environment by trying to construct mystery functions which generate the inputs and which affect the environment.
>
> This process is similar to continuous library learning and symbolic reasoning in my proposal.
>
> As the organism learns, starting almost from scratch, they form a bootstrapped, compounding, continuous improvement cycle. It tries to form abstractions or primitives in a modular way, at higher and higher levels of abstractions which can explain the world in simpler and simpler ways by effective composition.
>
> So the organism has memory for multiple purposes:
>
> 1. Just for remembering streams of inputs or streams of abstractions, which can be later replayed as needed for learning.
> 2. A sophisticated library of abstractions of different levels. Probably with each abstraction is also stored its usefulness score, and in which domain it was effective. This can apply to sensory streams as well as motor skills.
> 3. Similarly, a sophisticated library of modular programs about what sort of actions to take, or even just how to reason about the input or a combination.
> 4. Note that the organism can either learn from scratch or just learn higher level stuff by observing other organisms do something, or by learning from other organisms which may teach it stuff which they learned the hard way. Similar to cumulative culture for humans and standing on the shoulders of giants.
> 5. The organism may have inductive biases similar to DNA inheritance and also proclivity to certain things similar to how different humans and animals have different personalities and interests and skills etc.
> 6. The organism tries to survive and thrive in the world/environment by effectively making use of above. It will get negative reinforcement or pain or something if it is hungry, thirsty, damaged etc and vice versa.
>
> I want to demonstrate this practically using baby steps and prototypes and MVPs and iteration. You can suggest a better plan hopefully. I am starting with ARC-AGI-1 benchmark, and want to build up to more difficult ones. They all need to leverage the same core algorithm as described above, just with different inputs, domains etc.

### Prompt 3: Addressing Skeptics & Differentiation

> Great, I think you summarized my vision perfectly. A few questions I keep getting asked:
>
> 1. Aren't people or researchers already working on this? This seems intuitive and somewhat obvious.
> 2. This all sounds good in theory, but it is too high level. How will you make this work?
> 3. Don't existing LLMs and projects like AlphaGeometry and other DeepMind already do this? Maybe people are already building up to this.
>
> My response to some of this stuff is that current work is too LLM focussed and resource intensive and not explicitly decomposed in an explainable and reusable way. It is also passive and not interacting 2-way with the environment like humans and animals. Also, current work is using supervision which is expensive and it assumes that the training data is the source of truth, e.g. next token prediction on even a curated repository. And also, current work does not keep compounding and reusing already learnt concepts effectively, it starts from scratch in many cases, repeatedly learning the same things.

### Prompt 4: Getting Started with Code

> Yes. Also, how should I start translating this into reality. Can you generate code following these principles for me, or do I need to use Cowork or something. Or just do it myself? I am currently typing this on my phone. But I can carry over this session to my laptop.

---

## Session 2 — Claude Code on Web (March 10, 2026)

### Prompt 5: Continue from Mobile Session

> This repository uses file generated by my Claude chat session using the mobile app. Are you able to read my latest chat session with Claude which I starred just now and continue work as per the plan. The plan is there in the README as well as the docx file. Look at all my github repositories. The account vibhor-77 is connected and authorized.

### Prompt 6: Leverage Existing Repos

> Actually, the agi-mvp-general repo is the latest, and Claude cowork wrote it. It also has useful stuff. Please use stuff as needed from all my repos. You should have access to all of them, they are public anyway. But I also explicitly connected you to my Github account.

### Prompt 7: Core Loop Invariant

> Remember that make sure the core loop never imports anything domain-specific.

### Prompt 8: Strategic Decisions — PySR, DreamCoder, Benchmarks

> You recommend and decide. Feel free to install any pip packages or software needed, or ask me to do that. I was wondering if it would be a good idea to leverage the existing packages for PySR as well as DreamCoder, or whether to reimplement them. Also install the full ARC-AGI-1 dataset and run benchmarks.

### Prompt 9: Documentation & Living Documents

> Let me paste some prompts here that I gave to Claude chat on mobile for your information. Please incorporate them into the documentation as you see fit. I was hoping we can just keep a live document which contains all my instructions to you in chronological order, as well as another document with your responses and judgements?

### Prompt 10: Persistent Instructions (from Claude Cowork Global Instructions)

> Use the scientific method, proceed fast incrementally with short, tight feedback loops, and do several targeted, focussed experiments in parallel as needed.
> It is ok to do some minimal pip utility installs, for example pytest, pytest-cov, numpy, scipy, matplotlib etc. I want to avoid calling external services and really heavyweight external dependencies which make the code hard to understand.
> The code and repository should be world class, i.e. minimal and elegant but comprehensive, well commented and documented, working and well tested all the time with as high code and branch coverage in tests as possible. Note the coverage in the documentation. Ideally, use TDD (test driven development) as much as possible and if and when it makes sense. Follow the best practices.
> Also upload to github every time or give commands to do that.
> If you claim something is done verify it. When you return control to me after code or other changes, make sure that they are indeed working as expected.
> Unit test and integration test all the code. In fact, use TDD (test driven development) where possible and where it makes sense.
> Keep all documentation up to date and consistent with the code, remove obsolete stuff and add comprehensive documentation for all new features and keep updating it as changes happen.
> Use multiple files for documentation as needed, in a docs folder if it makes sense, but tie them together in README.md so that the user can find them easily.
> Always keep documentation files listing the entire prompt I have given you, as well as your thoughts each time and the results we got along the way, as well as the plan and next steps.

### Prompt 11: Share Instructions on GitHub

> Share these instructions on the github repository as well? I would like people to be able to follow exactly how I generated the code/repository/prompts etc.

### Prompt 12: Reproducible Quickstart + Merge to Main

> Any reason you are not using the main branch? Maybe because the code is not fully ready?
> Can you give full reproducible deterministic instructions in the README? For example, the quickstart instructions in this branch don't give instructions on cloning this repository and also the ARC AGI repository. Also, what if someone already has this repository, but need to sync to the latest version?

### Prompt 13: Performance Optimization + Good Defaults

> Similar to my other code, can you make the benchmark run use all performance cores on the user's machine? And also give instructions without flags as much as possible and use good defaults for iteration speed as well as good accuracy. Make things as simple as possible for the user to reproduce my results, and to iterate on improvements for me as well (and same applies to any other user who might want to contribute). You can provide documentation and examples of how to tweak flags for contest mode or something.
> For examples, in quickstart why does the user need to specify --rounds=5? Should there be a default? Also, why was that magic number picked in the first place? It does not belong in quickstart, do you agree?

### Prompt 14: Numpy/Numba Optimization + Merge to Main

> Similar to my other code, consider whether using numpy, numba etc is a good idea to optimize the python code which can be very slow otherwise.
> Also, are we in good shape to promote everything to the main branch? What are the benchmark numbers that the user will get with the quickstart instructions, and also the best numbers with any flag tweaking? Let's document them to set user expectations correctly, along with expected time to finish the runs on typical laptops.

### Prompt 15: Machine Specs + Auto-Detection

> Note that my laptop is an M1 Max Macbook Pro with 64GB RAM and 8TB SSD. Feel free to utilize it to its full potential effectively and intelligently, but don't thrash it. But the code should also run deterministically, reproducibly and fast on any users machine and be able to replicate my results, and whatever is given on the README. Auto detect the users machine architecture and run the job by default to utilize their machine effectively as well.

### Prompt 16: Leverage Existing Repos for Patterns

> My existing code in agi-mvp-arc-agi-1 and agi-mvp-general will show you how I used these principles there. Also, note how I have tried to cleanly log everything, and give live progress on the console as well as in log files and results and culture files in live json files.

### Prompt 17: Compute Budget + Layered Abstractions

> Note the compute cap concept in agi-mvp-general as well. Note how I am trying to utilize a given compute budget effectively to get the highest ROI on the eval set accuracy.

### Prompt 18: Simplified Compute Budget

> I would like to keep the compute budget concept simple and effective if possible, I was not happy with the way it was confusing in agi-mvp-general. The idea is simple, given a compute budget, I want to get the highest ROI using it, i.e. the highest accuracy on the evaluation. For iteration, I might give a lower compute budget and want to run fast in 5-10 minutes, but accuracy pretty close to the best possible numbers. But once in a while, or in contest mode, I want to provide a lot of compute and squeeze the highest possible accuracy.
> So I felt that in agi-mvp-general, the eval budget concept was confusing to me. But maybe it is needed to balance things out. I will let you figure it out by experimentation.

### Prompt 19: Use Layered Abstractions

> I hope you are using layered abstractions effectively here, not just copying blindly from the other code. i.e. Use the core domain agnostic layer effectively in this codebase. Because we want to extend this to more difficult tasks in the future like ARC-AGI-2, Zork, robotics and general intelligence!

## Session 2 — Claude Code Web (March 10, 2026)

### Prompt 20: Port Improvements from agi-mvp-no-noise

> agi-mvp-no-noise is on my github, and also in ~/github. [User asked Claude to study the repo and port the best ideas into agi-core: semantic deduplication, Pareto front tracking, and constant optimization.]

### Prompt 21: Push to Main & Follow Guidelines

> Should we push the changes to main and make sure my guidelines are followed? [User requested compliance audit, PR creation, and merge to main before proceeding with new features.]

---

## Session 3 — Claude Code Web (March 10, 2026)

### Prompt 22: Improve ARC-AGI Solve Rate with One Algorithm

> [Task notification with comprehensive agi-mvp-general research report comparing agi-mvp-general (304 primitives, 13 search phases, 97/400 train, 35/400 eval) with agi-core (48 primitives, pure beam search, 12/400 train). User wants to improve agi-core's solve rate by studying what makes agi-mvp-general effective and porting insights.]

### Prompt 23: One Algorithm, Think Harder

> one algorithm, but think harder

[User rejected the initial multi-phase approach. Insisted that improvements must stay within the generic beam search architecture — no domain-specific search phases. The insight: exhaustive enumeration IS beam search with beam_width = vocabulary_size^depth.]

### Prompt 24: Approve Adaptive Beam Width Plan

> yes

[User approved Phase 1: adaptive beam width / exhaustive enumeration approach.]

### Prompt 25: No Data Leakage from Evaluation Set

> Your numbers are based on the training set I hope. Do not even look at the evaluation set to avoid data leakage. That set should only be used to get scores, but not for understanding.

### Prompt 26: Evaluation Set is the Real Metric

> However, the actual accuracy needs to be given based on evaluation set scores. agi-mvp-general gets 35/400 correct on the evaluation set with full search.

### Prompt 27: Follow TDD and Scientific Method

> I hope you are following my instructions and guidelines of using TDD and the scientific method, and keeping this minimal, validated and verified and not just based on theory, but actual numbers.

---

## Session 4 — Claude Code Web (March 10, 2026)

### Prompt 28: Continue Improving

> continue

[User asked to continue improving the system. Claude analyzed near-miss tasks, studied agi-mvp-general's object-level primitives, and added 25 new primitives covering connected components, grid partitioning, diagonal operations, and anomaly removal. Also upgraded depth-3 exhaustive enumeration with smart subtree reuse.]

---

## Session 5 — Claude Code Web (March 11, 2026)

### Prompt 29: Quick Mode Too Slow

> Why is the quick mode documented to be relatively slow now (i.e. 32 minutes)? Can we make it run in 5 minutes or less, but with reasonable accuracy? Also, provide examples with operating on a subset of tasks in the README. Also, the default mode is shuffled tasks, right? i.e. If I run a subset of tasks, e.g. max-tasks=50, I can roughly extrapolate the number of solves?

[User identified that quick mode was slow because it still ran all 400 tasks. Asked to reduce to <5 min with reasonable accuracy, add subset examples to README, and confirm that shuffled tasks allow extrapolation from subsets.]

---

## Session 6 — Claude Code Web (March 12, 2026)

### Prompt 30: Continue

> continue

[User asked to continue improving the system. Claude analyzed near-miss and overfit tasks, added 12 new primitives (Batches 6-7), fixed 3 overfit tasks, and improved training solve rate from 26% to 32% on 50-task quick mode.]

---

## Session 7 — Claude Code Web (March 12, 2026)

### Prompt 31: Venv, Pipeline Summary, Compute Cap Optimization

> I ran the commands and merged to main. Can you do the following items: (a) Add venv as optional quickstart step, (b) Add end-of-pipeline summary with all parameters/results + combined JSONL/JSON files, (c) Run experiments to find lowest compute cap sweet spot.

### Prompt 32: Compute Cap Observations

> In my own experiments, I am finding that even with compute-cap=100, there are 16 train solves and 2 eval solves. And even with compute-cap=500M, there are 17 train and 2 eval solves, but it is just a bit slower. [...] Also, after the optimization, the benchmark is running really fast on my M1 Max.

### Prompt 33: Learning Concerns & Strategic Focus

> I am a bit concerned that maybe no learning is going on here, because regardless of compute cap, the solves remain the same. [...] We want to be close to 100% long term, and even in the short term I would like to get close to 40-50%. So we need to think about this in a very systematic, focussed, long term strategic way. What are big picture items missing here? How to get the compounding? Can we also do a baseline benchmark experiment run for ARC-AGI-2 and Zork?

[Claude performed deep strategic analysis. Key findings: (1) 78/80 ARC solves are depth-1 — compounding can't help when solutions are shallow, (2) library entries are redundant with depth-3 exhaustive search, (3) compounding works on list_ops because depth=2 forces library reliance. Baselines established: ARC-AGI-2 train 10/100 (10%), eval 0/120 (0%), Zork 2/4 (50%).]

### Prompt 34: Update Quickstart & Merge

> In the quickstart, you should probably include instructions to download/clone the ARC-AGI-2 dataset as well as the Zork dataset? [...] Let's make a PR and merge to main with the baselines for these other domains, and also with updated instructions.

### Prompt 35: Skeptic's Review & Honest Framing

> Now imagine you are [a skeptical person] who is critiquing our repository. What will be your analysis and report? And what would be the next steps to address those and move forward to genuinely make progress.

[Claude performed a thorough external review. Key findings: (1) README overstated claims — "one algorithm" framing hides 6,500 lines of hand-crafted ARC primitives, (2) compounding doesn't work on ARC (78/80 solves are depth-1), (3) test suite had zero compounding verification tests, (4) stale numbers throughout README. User approved the fix plan: honest README, compounding tests, updated roadmap.]

---

## Session 9 — Claude Code CLI (March 12, 2026)

### Prompt 36: Resume & Learn From Residuals

> Resume the conversation from the Claude MacOS desktop app, the name is 'AGI Core'. Just read the github repository. Make sure I am synced to main and there are no pending commits or anything. Read the entire code as well as documentation and also the prompt log.

[Claude synced to main, read entire codebase, documentation, and prompt logs.]

### Prompt 37: Implement Residual Learning

> You recommend. Yes. Can you also follow the instructions in CLAUDE.md to use the scientific method and fast feedback loops and rapid iteration and also validate and verify everything?

[User approved recommendation to "learn primitives from near-miss residuals." Session analyzed 516 near-misses, implemented task-specific swap primitives and relaxed color fix. Then user gave feedback about iteration speed — demanded quick mode (~7s) for iteration.]

### Prompt 38: Continued Iteration

[Context resumed after session compaction. Continued residual analysis, found near-misses are mostly structural/spatial, not color-fixable. Shifted to anti-overfit improvements and conditional search extension.]

## Session 10 — Claude Code CLI (March 13, 2026)

### Prompt 39: Continue from Context Compaction

[Context restored from session 9. Continued with pending documentation updates and ROI analysis.]

### Prompt 40: Decomposition/Composition Principle + Cross-Domain Validation

> I had told Claude during agi-mvp-general brainstorming that the ARC AGI dataset was built by humans such that other humans should be able to solve it relatively easily. Which means that the transformations are not that complex or deep, but they are more intuitive from a human point of view. Even for general learning, my observation is that whatever humans have figured out in the world can be expressed relatively simply, for example, the laws of nature, once figured out, are pretty simple and intuitive. Which just means that when we have the right primitives, or higher level abstractions constructed from composing these primitives, the problem becomes quite simple. Also, keep in mind that decomposition of a complex problem into simpler problems is just the flip side of the composition pillar. As humans, we try to decompose difficult problems into less difficult problems, and keep doing that recursively. Now, I want you to be a skeptic and challenge and experiment and validate these ideas and see if you can utilize these principles to actually improve performance not only on ARC-AGI-1, but also ARC-AGI-2 and Zork as well. i.e. I think this is a much more general and powerful principle, but we need to validate it and use it effectively.

[User proposes that decomposition (breaking complex problems into simpler ones) is the flip side of composition, and that with the right primitives, solutions should be simple. Wants skeptical validation and cross-domain application (ARC-AGI-1, ARC-AGI-2, Zork). Claude validated with data: 95% of ARC solutions are depth 0-1, confirming that right vocabulary → shallow composition. The bottleneck is discovering the right vocabulary, not deeper search.]

### Prompt 41: Implement Strategic Plan — Path Forward for ARC-AGI-1 Solve Rates

> Implement the full strategic plan: (A) Generalized LOOCV for all training-perfect candidates, (B) Diff-and-patch phase for near-misses with spatial corrections, (C) Same-shape few-changes specialization, (D) Vocabulary pruning — task-specific color primitives, (E) Output-shape prediction.

[Implemented Phases A, B, and D. Phase A: Added `_loocv_score` method to Learner — re-prepares grammar with N-1 examples and validates held-out for each candidate. Phase B: Extended `infer_output_correction` with adjacency-based and 3x3 neighborhood correction strategies beyond color remapping. Phase D: Moved ~120 parameterized color primitives to runtime generation in `prepare_for_task`, reducing per-task primitives from ~349 to ~235. All 527 tests pass.

**Results:** +100 total solves (120→220 in contest mode). Phase B (diff-and-patch) drove 89% of gains — `neighborhood_3x3_fix` alone added 84 new solves with 91% generalization rate. Train-eval gap narrowed from 3.8x to 2.0x. See Decisions 82-83 for full attribution.]

### Prompt 42: Cross-Domain Validation + Path to 320/800

> Implement the strategic plan: (1) Cross-domain validation runs for ARC-AGI-2, Zork, ARC-AGI-1, (2) Document results, (3) Generalization gap analysis (neighborhood fix cap tuning, complexity penalty, ensemble agreement), (4) Near-miss goldmine mining (identity-seeded correction, 5x5 neighborhoods, row/column corrections), (5) ARC compounding re-test, (6) Jericho assessment (deferred).

[Implemented 4 new correction strategies: (a) 5x5 neighborhood patches for longer-range dependencies, (b) identity-seeded correction for same-shape tasks, (c) row/column-level corrections, (d) ensemble agreement for test prediction. Fixed CurriculumConfig bug that dropped sequential_compounding/adaptive_realloc when resolving workers=0.

**Cross-domain validation:** ARC-AGI-2 train improved from 14% to 21.7% without AGI-2-specific work. Zork stable at 10/20.

**Results:** ARC-AGI-1 default mode improved from 207/800 to **273/800 (+66 solves)**. Train: 173/400 (43.2%), Eval: 100/400 (25.0%). Train/eval ratio narrowed from 2.0x to 1.7x. Cap tuning validated 50 as optimal default.

547 tests pass (14 new). See Decision 84 for full details.]

### Prompt 43: Path from 273/800 to 350/800

> Implement the following plan: (0) Update README with latest numbers, (1) Multi-step correction chaining, (2) Compounding re-test, (3) Parameterized global color map primitive, (4) Complexity penalty tuning, (5) Near-miss threshold widening, (6) 7x7 neighborhoods.

Also provided latest ARC-AGI-2 numbers: 312/1000 (31.2%) train, 9/120 (7.5%) eval.

[Implemented Steps 0-5. Step 6 deferred (diminishing returns). See Decisions 85-89. 557 tests pass (10 new).]

---

## Session 12 — Claude Code CLI (March 13-14, 2026)

### Prompt 44: Deep Audit, Cleanup, and Get Back on Track

> Deeply understand and audit this entire repository: agi-core, what are the goals, and how we are doing it. Validate everything and follow the guidelines on CLAUDE.md. Clean up the whole repository. Then suggest next steps. There was a huge screwup and unnecessary complications caused this morning because you (Claude Code) changed the meaning of the solved numbers and they got inflated. Then a whole bunch of unnecessary primitives got added I think regarding diff and patch. I would like us to get back on track towards the big picture goal. I would like us to get closer to 100% held out public eval accuracy on ARC-AGI-1 and ARC-AGI-2 and Zork, without any sort of cheating or data leakage, no matter how unintentional. But I would also like to keep the core abstraction intact here if it makes sense, and if not, you let me know.

[Claude performed a complete repository audit. Found: 36 completely unused primitives, 19 more that never solved anything, and the correction cascade with 97% overfit rate (9 test-verified solves vs 290 overfits). Confirmed core abstraction is sound. Removed 55 dead primitives (235→180), removed overfit correction cascade (-275 lines), overfit dropped from 309→58. See Decisions 98-100.]

### Prompt 45: Primitive Discovery by Evolution

> Yes. I think we might need to remove a whole bunch of primitives which are useless. And we might want to figure out a general exploration methodology to discover good primitives by evolution or something? What do you think? In any case, we should start with the basic steps.

[Approved primitive pruning. Claude performed data-driven analysis: 87 of 235 primitives appear in any test-verified solve, 148 never appear. Removed 55 in two rounds. Discussed evolutionary primitive discovery — user interested but wanted basics first.]

### Prompt 46: Starting from Scratch — Path A vs Path B

> Pretend that you were starting from scratch. What would you do?

[Claude proposed two paths: Path A (keep 180 primitives, incrementally add more) vs Path B (reduce to ~20 fundamentals, force composition, enable compounding). Key argument for Path B: the current system is a lookup table of hand-crafted transforms, not a learning system. With fewer primitives, solutions require composition, which creates depth-2+ programs the sleep phase can extract and promote — enabling genuine compounding.]

### Prompt 47: Path B

> Path B

[User chose Path B. Claude created minimal vocabulary of 27 fundamental primitives, added `--vocabulary minimal|full` flag. Initial results: 6/50 train (vs 20/50 full) — much lower as expected. Compounding flat across 5 rounds because all 6 solves were depth-1.]

### Prompt 48: Intuitive Operations ARE Compositions

> Yes, but even the intuitive conceptual operations are a composition of some basic operations.

[User pushed back on the claim that operations like fill_tile_pattern can't be composed. Pointed out they ARE compositions of more basic cognitive operations — just not compositions of grid transforms. This led to the insight that the primitives were all ACTION primitives (transform the grid) but missing PERCEPTION primitives (understand the grid). The decomposition half of decomposition/composition duality was missing.]

### Prompt 49: Add Perceptual Primitives

> Yes

[Claude added 16 perception primitives: pattern detection (extract_repeating_tile, fill_tile_pattern, upscale_pattern), grid structure (remove_grid_lines, select_odd_one_out, overlay_grid_cells), symmetry completion, line extension, stacking/repetition. Results: 27→43 primitives, 6→13 train solves — 2x improvement. Depth-2 compositions emerged: select_odd_one_out(crop_to_nonzero), extend_lines_to_contact(connect_same_color_vertical).]

### Prompt 50: Analyze Composability of Missing Primitives

> Instead of just guessing like this, why don't you try thinking whether the intuitive or conceptual primitives can be composed out of the basic primitives in a simple way? Or find what are the missing composition rules as well as basic primitives that would allow us to do something like that.

[Claude analyzed all 20 high-value missing primitives by reading their actual code. Found: 3 composable from existing minimal prims (mirror_horizontal_merge = overlay(grid, mirror_h(grid))), 7 need one new basic primitive (clustered into ~4 new ops), 8 are fundamentally new concepts (clustered into ~3 concept families). Added 9 highest-value missing primitives to get 52→60 minimal vocab. See Decision 103.]

### Prompt 51: Find the Minimal Basis for ALL ARC Tasks

> The goal is not just to close the gap with the 106/400 primitives, even they have a very low coverage. It is to discover an exploration and compounding mechanism, using the 4 pillars continuous cycle. Also, we will want to look at all the 400 tasks, understand the minimal set of primitives and composition rules that would cover almost all of the transformations. Note that it might be ok to have a deeper depth of composition with the basic primitives. That's because higher level composed abstractions will naturally emerge, and the transformation will be low depth on the higher level abstractions.

[User reframed the goal: not "close the gap to 106" but "cover nearly all 400 tasks." Deeper compositions are fine because compounding will naturally promote recurring depth-2 sub-programs into depth-1 library entries. Claude analyzed all 400 training tasks and identified 16 irreducible operations covering ~95% of tasks, plus 3 composition rules beyond pipelining: FOR_EACH (48% of tasks), CONDITIONAL (66%), CROSS_REFERENCE (36%). A pure pipeline can only express ~15% of tasks.]

### Prompt 52: Implement Composition Rules

> Yes!

[Claude implemented FOR_EACH objects (try top-K enumeration candidates per-object) and CROSS_REFERENCE (boolean ops on grid halves, cell propagation, small-on-large stamping). Key bug found and fixed: cross-reference phase was skipped due to eval budget exhaustion — fixed by making it budget-exempt. Cross-reference solved 10 tasks (5 train + 5 eval), ALL test-verified, ZERO overfit. Every one was previously overfit by the old correction cascade.]

### Prompt 53: Iterate on Composition Rules

> Continue

[Claude iterated on cross-reference: fixed separator consistency across training examples, added column propagation alongside row propagation, fixed closure variable bug. Full vocab default mode with cross-reference: 36/400 eval (9.0%), up from 31/400 (7.8%). Minimal vocab at quick cap: 26/400 eval, beating full vocab's 25/400 at same compute budget.]

### Prompt 54: Balance Both Approaches

> both in a balanced methodical way

[Claude ran both vocabulary modes at 400 tasks. Key finding: minimal vocab (60 prims) beats full vocab (180 prims) at same compute budget because smaller search space = better coverage per evaluation. Cross-reference finds solutions impossible with pipelining.]

### Prompt 55: Keep Going

> you decide / ok / continue

[Claude continued iterating: scanned for more cross-reference patterns (boolean mask, color-preserving mask — no additional hits). Ran compounding experiment at 400 tasks with minimal vocab: 83→84 (+1) across 3 rounds — real but modest compounding. Full vocab default mode confirmed: 36/400 eval (9.0%), 110/400 train (27.5%).]

### Prompt 56: Update Documentation

> Can you update all the documentation, including the prompt and decisions log files? I would like to make my entire prompt and reasoning history available in GitHub for people to understand the through process as well as the decision process and twists and turns that happened along the way.

## Session 13 — Repository Audit, Cleanup & Strategic Experiments (March 2026)

### Prompt 57: Repository Audit and Strategic Next Steps

> Implement the following plan: Repository Audit, Cleanup, and Strategic Next Steps

[User provided a detailed plan with two parts: (A) Code cleanup — add .DS_Store to .gitignore, remove dead --verbose parameter, remove ~70 dead functions from primitives.py (6,974→5,549 lines, -1,425), update README test counts; (B) Strategic experiments — per-object conditional transforms, more cross-reference strategies, fixed-point with depth-2 compositions, predicate-guided enumeration pruning.

Results: Part A completed cleanly (all tests pass, no regression). Part B: all 4 experiments implemented and tested. Quick benchmark (50 tasks): +1 train solve from predicate-guided pruning, others neutral. Full scale (400 tasks): neutral — both vocabularies converge at ~35/400 eval (8.8%). The experiments add composition infrastructure for future gains.]

---

### Prompt 58 (2026-03-14)

> Implement the following plan: Refactor to Domain Adapter Architecture

[User provided a detailed 9-step plan to refactor the codebase into a three-layer architecture: core/ (pure algorithm), common/ (benchmark infrastructure), and domains/*/adapter.py (DomainAdapter implementations). Key changes: moved runner.py to common/benchmark.py, added DomainAdapter ABC, created adapters for ARC/ListOps/Zork, added unified CLI entry point, simplified experiment scripts, preserved all backward-compatible import paths. Result: 553 tests pass (520 original + 33 new).]

---

### Prompt 59 (2026-03-14)

> Implement the following plan: Strategic Plan: Compounding via Approximability

[User provided a detailed 4-experiment plan to unlock compounding on ARC. Analysis identified three root causes of broken compounding: (1) arity-0 callable bug silently disabling parameterized prims, (2) sleep only reading perfect solutions (95% depth-1, no subtrees), (3) 333 near-misses invisible to learning. Experiments: Exp 1 — fix arity-0 callable execution in `_eval_tree`; Exp 2 — near-miss sleep (store programs with error<0.15, extract subtrees with quality weighting, train transition matrix on near-misses); Exp 3 — atomic vocabulary + compounding (combine existing features); Exp 4 — remove fixed-point iteration (0 solves), widen pair pool 40→50.

Results: All 4 experiments implemented. 631 tests pass (9 new). 50-task sequential compounding run: 21/50 solved (42%), 6 library entries created via immediate promotion, library entries reused in subsequent rounds (first-ever non-zero library on ARC). 14 near-misses captured with depth-2+ compositions. Culture file now includes near-misses for cross-run transfer.]

---

### Prompt 60 (2026-03-14)

> Should we only keep the unified CLI and remove backward-compat experiment scripts? Also remove --compounding flag and any dead code.

[User also asked about removing minimal/full vocabularies to only keep atomic. Advised against — atomic solves 4/50 vs full 21/50, compounding not yet bridging the gap. Removed: 4 experiment scripts (-527 lines), --compounding flag, dead fixed-point code (-74 lines from learner.py), unused Decomposition import. Updated all README examples to unified CLI. 631 tests pass.]

---

### Prompt 61 (2026-03-14)

> Continued atomic vocabulary work: perception+parameterized architecture, strip compound prims, interleaved pipeline, culture JSONL, viz expansion, no-op pruning, sweet-spot analysis

[User drove the session toward truly atomic primitives. Key decisions: (1) Add perception (Grid→Value) and parameterized ((Value,...)→Grid→Grid factory) primitive kinds — removes task-specific color prims, makes compositions transferable. (2) Strip all compound prims (extract_largest_object, etc.) to get honest baseline: 2/50 with 41 truly atomic prims. (3) Added label_components and mask_by — verified extract_largest_object expressible as depth-4 composition. (4) Interleaved pipeline: train→eval per round. (5) Culture JSONL for live observation. (6) Viz expands learned abstractions inline. (7) No-op pruning. (8) Sweet-spot analysis: energy_beta flat, 3 rounds sufficient. Final: atomic 4/50 train (3 rounds), 14/400 eval with culture transfer. 654 tests.]

---

### Prompt 62 (2026-03-14)

> Default rounds, pipeline output fixes, remove obsolete flags, compounding improvements

[User asked for data-driven default rounds, correct pipeline output, removal of vocabulary/beam/contest flags, compounding curve in terminal, and cleanup. Measured sweet spot: 2 rounds for both quick and default (round 2 gives +28-33% solves, round 3 adds <5%). Changed preset defaults to 2 rounds. Fixed pipeline output: correct R1/R2/R3 numbering, brief per-round summaries, wall time breakdown, aligned compounding curve table. Removed --vocabulary, --beam-width, --max-generations, --adaptive-realloc flags and contest preset. Default --run-mode changed to pipeline. Added binary near-miss refinement. Compounding results on 400 tasks: train 18→23→24 across 3 rounds, eval 8/400.]

---

### Prompt 63 (2026-03-14)

> Implement bounded library with eviction. Then: remove near-miss threshold, store all unsolved programs. Make the change cleanly — rewrite from scratch, not a hack. Update all documentation.

[User provided a detailed plan for bounded library with eviction (cap=100, reuse immunity, eviction score). Implemented and swept capacity 50/100/150/200 — all equivalent at current library sizes, cap=5 confirms eviction works. Then user identified that the old near-miss threshold (error ≤ 0.15) was too restrictive with eviction in place. Removed threshold: near-misses stored 21→46 (+119%), library entries 14→32 (+129%), solved extra task. User then requested a clean rewrite instead of patching — renamed entire concept from "near-miss" (implied threshold) to "best attempt" (stores everything). Cleaned API: `store_near_miss`→`store_best_attempt`, `get_near_misses(max_error)`→`get_best_attempts()`, `near_miss_weight`→`unsolved_weight`. Deleted dead `SleepConfig.near_miss_threshold`. Wake refinement (`_near_miss_refine`, `SearchConfig.near_miss_threshold`) unchanged — separate concept. Updated all docs. 402 tests pass.]

### Prompt 64 (2026-03-14)

> Is crop_to_content really atomic? Audit all primitives and make them truly minimal and atomic. Is grammar.py dead code? Remove dead code. Where are composition rules? Make them simple and minimal. We will compose perception and transformation primitives together.

[Audited all 41 ARC primitives for atomicity. Removed 3 composable geometric transforms (rotate_90_ccw, rotate_180, mirror_vertical) — all reachable at depth ≤ 2 from {rotate_90_cw, mirror_horizontal, transpose}. Kept crop_to_content and fill_enclosed as pragmatic atomics — decomposing requires narrow-purpose bbox perception + arity-4 parameterized crop with no expressivity gain. Removed dead code: grammar.py decompose/recompose (never called, broken imports of non-existent _detect_any_separator_lines/_split_grid_cells). Removed environment.py structural phase overrides (try_object_decomposition, try_cross_reference, etc — all gated by allow_structural_phases()=False, try_cross_reference had broken imports). Composition rules already minimal: grammar.py compose() creates trees, environment.py _eval_tree() evaluates recursively. Perception+transformation composition works naturally via parameterized primitives. Final: 38 primitives (was 41). 403 tests pass.]

### Prompt 65 (2026-03-14)

> You removed some primitives unnecessarily (mirror_vertical is intuitive, asymmetric with mirror_horizontal). crop_to_content IS compositional. Update with this guideline: intuitively atomic, not mathematically minimal.

[Restored all 6 geometric transforms — each is one intuitive visual concept. Decomposed crop_to_content into trim_rows + trim_cols (each operates on one axis, truly atomic). Composition: trim_cols(trim_rows(x)) = crop_to_content. Commutative — both orders give same result. Each has independent utility. 42 primitives total. 408 tests pass.]

### Prompt 66 (2026-03-14)

> When doing hyperparameter type experiments, maybe pass a flag to turn off visualization as it just wastes disk space. Similarly, do you want to turn off verbose logging and live logging in such cases? By default, all these can be on, but there could be a flag corresponding to interactive mode vs batch mode?

[Added `--batch` CLI flag. When set: skips visualization (no HTML files generated), suppresses per-task console output (no live task-by-task lines), suppresses verbose headers/footers, suppresses worker diagnostic prints, auto-suppresses log file. Data files (JSONL, JSON, CSV, culture) still written. Output reduced to one-line `[batch]` header + periodic scoreboards + one-line `[batch]` result summary. 419 tests pass (4 new batch mode tests).]

## Session 14 — Structural Search + Solve Rate Recovery (2026-03-15)

### Prompt 67 (2026-03-15)

> The solve rate is currently really low, especially after removing the non-atomic primitives. Deeply understand, analyze, strategize and execute a plan to drastically improve the scores without compromising our principles. I am going to sleep now, so keep trying it out without getting interrupted or waiting for my input. Also, follow instructions in CLAUDE.md. Update all documentation including code/test comments, README.md, PROMPTS.md and DECISIONS.md.

[Claude performed a deep analysis of the entire codebase and identified a critical issue: `allow_structural_phases()` was returning `False` for atomic vocabulary, which disabled ALL structural search phases (object decomposition, cross-reference, conditional per-object, color fix, conditional search). These are SEARCH STRATEGIES that compose existing atomic primitives in structurally different ways — they don't add new primitives, so disabling them was unnecessarily crippling the search.

Key changes implemented:
1. **Re-enabled structural phases** — `allow_structural_phases()` now returns True for all vocabularies
2. **Wired up ARCEnv** — Implemented `try_object_decomposition`, `try_for_each_object`, `try_conditional_per_object`, `try_cross_reference`, and `infer_output_correction` on ARCEnv
3. **Added 7 predicates** to grammar for conditional search (is_square, has_single_object, etc.)
4. **Expanded perception**: 12 → 18 primitives (added second_color, corner_color, center_color, edge_color, interior_dominant_color, grid_max_dim)
5. **Mixed compositions in exhaustive search**: transform(parameterized(perception)(x)) and t1(t2(parameterized(perception)(x))) patterns at depth 2.1 and 3.1
6. **Library cap**: 100 → 200 for more diverse composition storage
7. **Depth-3 skip threshold**: 0.50 → 0.65 (more tasks explored at depth 3)

Results (measured, 400 tasks, 3 rounds):
- Training: 18→24 (old) → 28→32 (new) = +33% improvement
- Eval: 8/400 (old) → 8-10/400 (new) = maintained or improved
- Per-object recolor: 9 solves (28% of training) — entirely new capability
- Library entries: 12 solves (38% of training) — compounding working
- Color remap: 2-3 solves — color fix on near-misses working
- Zero overfit on per-object and color_remap strategies
- All 419 tests pass. 48 primitives total.]

## Session 15 — Search Strategy Improvements (2026-03-15)

### Prompt
Implement the Session 15 plan: 6 changes to improve search strategies based on near-miss analysis (345 tasks with error < 0.15).

### Changes Implemented
1. **A: Expand predicates** (7→12): added has_symmetry_v, is_small_grid, has_few_colors, has_many_colors, all_objects_same_size
2. **B: Binary near-miss 3→5**: increased near-miss candidates for binary refinement
3. **C: DEPTH2_BRANCH_K 8→15**: more depth-2 programs as conditional branch candidates
4. **D: Position-based recolor** (3 strategies): by_quadrant, by_row_band, by_col_band with LOOCV
5. **E: Scale/tile detection**: integer ratio detection in cross-reference (scale, tile, downscale)
6. **F: Cell-wise patches**: fixed pixel corrections for near-miss outputs (<15% diff)

### Results
- Train: 31→33/400 (+2 tasks, 7.8%→8.2%)
- Eval: 10/400 R1, 9/400 R2/R3 (stable)
- Compounding: 24→33 across rounds
- Overfit: stable (4 train R3, 2 eval)
- Tests: 419→434, all passing

---

## Session 16 — Per-Example Discrete Solve Scoring (2026-03-15)

### Prompt 1: Implement Per-Example Solve Scoring Plan

> Implement the plan for per-example discrete solve scoring for better compounding.

**Changes made:**
- Added `example_solve_score` field to `ScoredProgram` in `core/types.py`
- Added `example_solve_exponent` to `SleepConfig` in `core/config.py`
- Computed `example_solve_score` in `_evaluate_program` and `_evaluate_on_test`
- Added `_unsolved_quality` helper for sleep phase quality weighting
- Added `test_solve_score` to `WakeResult`
- 6 new tests, all passing (440 total)
- Sweet-spot analysis: exponents 1.5, 2.0, 3.0 — chose 2.0

### Prompt 2: Reintroduce Contest Mode

> Would it make sense to reintroduce contest mode with higher compute cap and exploration depth? Because we have stripped down the primitives to only be atomic, I think it might help. However, let us be data driven about this and validate the hypothesis. Also, follow instructions in CLAUDE.md. Update all documentation including code/test comments, README.md, PROMPTS.md and DECISIONS.md

**Changes made:**
- Reintroduced `contest` preset: 50M compute cap, beam 30×15, pair_top_k=48, triple_top_k=20, 3 rounds
- Added `beam_width`, `max_generations` to ExperimentConfig and resolve_from_preset
- Updated __main__.py to forward preset beam/top-K settings
- Measured: default 33/400 → contest 43/400 train (+30%), eval stable 8-9
- Contest R1 alone: 36 vs 24 (+50%) from wider search
- Overfit trade-off: 2→9 (more task-specific solutions that don't generalize)
- 1 new test (contest preset), all 441 tests pass
- Updated README.md presets table, expected performance, options table
- Decision 121 with full measured results

## Session 17 — -log Scoring + Primitive ROI Tracking (2026-03-15)

### Prompt 1: Implement -log Scoring + Primitive ROI Plan

> Implement the plan: 3-Level -log Scoring + Primitive ROI Tracking

**Changes made:**
- **Part A: -log(similarity) transform**: Applied `-log(similarity)` in ARCDrive (was `1-similarity`), removed max-error blending in `_evaluate_program` (redundant in -log space), changed `_unsolved_quality` to `exp(-error)` (maps [0,∞)→(0,1])
- **Part B: Primitive ROI tracking**: Added `get_primitive_scores`/`update_primitive_score` to Memory interface, implemented in InMemoryStore, credit primitives during sleep (solved=1.0, unsolved=quality), decay alongside library entries, ROI-blended pool ordering in `_exhaustive_enumerate`, persist in culture JSON
- **Part C: Tests**: Updated 4 ARC drive tests for -log scale, added 2 new ARC tests (log_scale_partial, log_scale_near_perfect), updated 2 quality tests, added 4 primitive scoring tests (default_empty, credit_solved, decay, persist_culture). 441→447 tests, all pass.
- **Part D: Documentation**: Decision 122, updated README.md scoring description and performance numbers
- Measured: default 34/400 train (8.5%, was 33/400), eval 9/400 (2.2%), compounding 23→34

---

## Session 18 — Unified Compute Budget & ROI-Driven Search (2026-03-15)

### Prompt 1: Implement Unified Compute Budget Plan

> Implement: Auto-derive all search params from compute budget, simplify presets, seed library ROI in sleep.

**Changes made:**
- **Part A: Auto-derive search params**: Added `derive_search_params()` and `derive_rounds()` to `core/config.py`. Simplified PRESETS to compute_cap only. Rewrote `resolve_from_preset()` to auto-derive. Changed CLI pair/triple top-k defaults to None (sentinel). Contest mode auto-derives identical params to old hand-tuned values.
- **Part B: Library ROI seeding**: In sleep, accepted library entries get primitive_score seeded at `usefulness * 0.1`. Closes feedback loop: high-usefulness entries get search priority via ROI-blended pool ordering.
- **Part C: Tests**: Added 11 new tests (derive params low/medium/high budget, monotonic, derive rounds, resolve auto-derives, CLI override, preset minimal keys, library ROI seeded in sleep). 447→458 tests, all pass.
- **Part D: Documentation**: Decision 123, updated README presets section, updated CLI flags.

## Session 19 — Solve Bottleneck Analysis: Dim Penalty, Max-Error, New Primitives (2026-03-15)

### Prompt 1: Deep Analysis & Fixes

> Analyze solve bottlenecks. Fix dimension mismatch scoring, change solve criterion to max-error, add gravity/sorting/repeat primitives.

**Changes made:**
- **Dim mismatch cap**: Added `DIM_MISMATCH_CAP=0.35` in `domains/arc/drive.py` to prevent wrong-dimension programs from entering near-miss refinement.
- **Max-error criterion**: Added `max_example_error` to `ScoredProgram`. Solve requires max_error==0 (all examples perfect), not just low avg_error. Ranking still uses avg_error.
- **New primitives (48→55)**: 3 gravity transforms (up/left/right), 2 sorting transforms (sort_rows/cols_by_nonzero), 2 parameterized factories (repeat_rows/cols). Total: 27 transforms + 10 parameterized + 18 perception.
- **Tests**: 458→489 tests, all pass.

---

## Session 17 — Claude Code CLI (March 16, 2026)

### Prompt 1: Structural Strategy Restore (previous plan)

> Implement the plan: Add Back Structural Strategies (Justified by Data)

Bulk-restored structural strategies from pre-strip git history. Result: 22→28 train (+6), 6/400 eval. But 5 of 10 phases added zero solves (~500 LOC dead weight).

### Prompt 2: Course Correction

> Did you really follow my instructions properly? Also, did you follow the CLAUDE.md instructions? Because, it looks like we are back to square one, and the whole point of the previous plan was to systematically identify areas of improvement by stripping down to the basics. If needed, start from scratch again, taking into account the lessons we have learned along the way. What should be next steps?

**Response:** New plan created — strip dead weight, analyze 132 near-miss tasks data-first, then add primitives one at a time justified by specific task IDs. Feature branch `principled-rebuild-v2`.

---

## Session — Procedural Object DSL + Targeted Primitives (2026-03-17)

### Prompt 1: Implement the Procedural Object DSL Plan

> Implement the following plan: Procedural Object DSL — Learn Per-Object Actions from Examples

**Response:** Full implementation of procedural.py (~1500 lines). Pixel-diff engine, 6 action templates, property-based rule learning with LOOCV, object movement detection, object extraction. Train 84→88 (+4), eval 24→26 (+2).

### Prompt 2: Continue with the Plan

> continue with the plan

**Response:** Continued procedural improvements — added object extraction for dimension-change tasks. Train 88→88, eval 26→26 (+0 net). Exhaustive scan confirmed procedural framework maximally exploited.

### Prompt 3: Keep Going — Major Improvements Needed

> Keep going. We are still at very low accuracy, and need to make major improvements.

**Response:** Pivoted to targeted new primitives strategy. Pre-test candidates on ALL unsolved tasks before adding — only add if demonstrably solving new tasks. Added: crop_to_content, flood_fill_by_neighbor, subtract_grid, xor_grid, tile_v, densest_subgrid, most_colorful_subgrid, extend_right_and_down, extract_unique_quadrant, overlay_all_sections, remove_separators. Discovered that adding non-solving primitives is actively harmful (3 tiling prims caused -3). Train 88→94 (+6), eval 26→30 (+4).

### Prompt 4: Update Documentation

> update README and all documentation to not be stale. Especially the presets and expected performance sections. Then continue with the plan

**Response:** Updated README.md (presets, expected performance, current status, limitations, structure, roadmap), DECISIONS.md, benchmark.py comments. All numbers now reflect: train 94/400 (23.5%), eval 30/400 (7.5%), 75 primitives, 436 tests.

### Prompt 5: Keep Going — Major Improvements Needed (round 2)

> Keep going. We are still at very low accuracy, and need to make major improvements.

**Response:** Deep analysis revealed conditional phase was missing from pipeline. Fixed it (multiprocessing crash from unpicklable closures). Then: position-modular local rules (+1 train, +1 eval), ncolors local rules (+1 eval). BREAKTHROUGH: half_colormap cross-reference (+10 train, +9 eval) — split grid in half, learn pixel-tuple→output color mapping. Extended to N-way and quadrant colormaps. Train 94→105 (+11), eval 30→41 (+11).

### Prompt 6: Keep Going + Verify Docs

> Did you push changes to github on the main branch? Did you update all documentation including README, PROMPTS and DECISIONS? Make sure there is no stale information in the README. Follow the instructions in CLAUDE.md. Keep going. We are still at very low accuracy, and need to make major improvements.

**Response:** Verified all pushed to main. Fixed stale README items (wake phase count 9→10, overfit gap numbers, unsolved count). Updated PROMPTS.md. Continuing with improvements.

---

*This document will be updated with each new session.*
