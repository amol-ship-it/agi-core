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

---

*This document will be updated with each new session.*
