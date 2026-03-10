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

---

*This document will be updated with each new session.*
