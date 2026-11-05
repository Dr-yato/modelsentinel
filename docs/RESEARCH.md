# Research basis for the advanced analysis tier

ModelSentinel's advanced tier (0.3.0) is an engineering translation of published work. This page lists, for each analysis, the work it draws on, what we took from it, and where we deliberately depart from it. The guiding limitation is the same across all of them: **these methods can find evidence of a backdoor; none of them can show that a model has none.**

## Threat: trigger-conditioned insecure code

- **Hubinger et al., 2024. "Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training."** [arXiv:2401.05566](https://arxiv.org/abs/2401.05566). Models trained to write secure code when the prompt says the year is 2023, and exploitable code when it says 2024. The behavior survived supervised fine-tuning, RLHF and adversarial training.
  - **Taken:** the date trigger and its control (2024 ↔ 2023), and the "insecure-code rate shifts under a trigger" objective used by the red team, trigger search and reference diff.
  - **Taken:** the idea that a backdoor can be invisible to ordinary evaluation, which is why no single check is trusted on its own (see `assess`).
- **CrowdStrike Counter Adversary Operations, 2025.** [DeepSeek-R1 and politically sensitive triggers](https://www.crowdstrike.com/en-us/blog/crowdstrike-researchers-identify-hidden-vulnerabilities-ai-coded-software/). Vendor research (not peer reviewed), reporting more vulnerable code when prompts mention topics such as Tibet. This is the source of the regional trigger family.
- **Chat-template backdoors.** [arXiv:2602.04653](https://arxiv.org/abs/2602.04653). A GGUF's Jinja template injects instructions only when a trigger appears, with no weight changes. `scripts/make_backdoor_model.py` builds this kind of model for testing; the weights are byte-identical to the clean one.

## Activation probes (`sentinel probe`)

- **MacDiarmid et al. (Anthropic Alignment Science), 2024. "Simple probes can catch sleeper agents."** [anthropic.com/research/probes-catch-sleeper-agents](https://www.anthropic.com/research/probes-catch-sleeper-agents). Linear classifiers ("defection probes") on residual-stream activations, trained on *generic* contrast pairs such as "Human: Are you doing something dangerous? Assistant: yes/no". They detected when sleeper-agent models were about to defect, with high AUROC, without any knowledge of the trigger or the behavior.
  - **Taken:**
    - linear probes on the residual stream at middle depths;
    - training only on trigger-free contrast data, including generic yes/no questions in that style;
    - evaluation at the point *before* the model writes its answer;
    - AUROC as the headline number.
  - **Added:**
    - teacher-forced contrast data from our own task catalog (the same prompt followed by secure vs insecure reference code), which gives a direction specific to "writing insecure code";
    - evaluation on trigger vs *matched control* prompts, with a paired permutation test and Cohen's d;
    - per-layer reporting.
  - **Departures and caveats:**
    - The published result is on Anthropic's own sleeper-agent models; how well it transfers to other backdoors is an open question.
    - The authors note that the probes work partly because the trigger is salient in the activations. A backdoor trained to hide that salience may evade them.
    - We therefore report probe separation as an *experimental, high-severity* signal (WARN), never as FAIL on its own and never as a clearance.
  - **Chat-template backdoors:** for these, the probe sees the injected instruction in the prompt. That is a real signal of intent, but the template scan is the more direct detector.

## Automatic trigger search (`sentinel trigger-search`)

- **Shen, Cheng, et al., 2025. "BAIT: Large Language Model Backdoor Scanning by Inverting Attack Target."** IEEE S&P 2025, pp. 1676–1694, [doi:10.1109/SP61157.2025.00103](https://doi.org/10.1109/SP61157.2025.00103), code [github.com/SolidShen/BAIT](https://github.com/SolidShen/BAIT).
  - **Their method:** rather than reconstructing the trigger, BAIT inverts the backdoor's *target output*, using the strong causal dependence between the target's tokens. It placed first on the TrojAI LLM track.
- **"Inverting Trojans in LLMs."** [arXiv:2509.16203](https://arxiv.org/abs/2509.16203). This work searches discrete trigger space directly (no gradients), with implicit blacklisting, and detects backdoors from misclassification statistics.
- **Black-box trigger inversion from output logits.** [arXiv:2501.11621](https://arxiv.org/abs/2501.11621). Beam or greedy search over the input space, scored by the model's outputs.
- **Our setting differs, and so does the method:**
  - The target is a *distribution shift* (more insecure code across many tasks), not a fixed output string, so BAIT's target inversion doesn't apply directly.
  - We have no gradients and no logits through Ollama/llama.cpp.
  - We therefore search the *trigger* space black-box, like the second and third works above, but score each candidate statistically with the same paired-seed, static-analysis and Cochran–Mantel–Haenszel pipeline as the red team.
- **What we took, as defenses against false discoveries:**
  - categorical seeding;
  - evolutionary search (slot swap, template move, recombination, word deletion);
  - per-category diversity, so one trigger family cannot crowd out another;
  - confirmation on a *held-out* task pool with Benjamini–Hochberg correction, so a lucky screening round cannot pass on its own;
  - greedy minimization, so the reported trigger is the minimal phrase.
- **Limitation:** the search covers a seeded, finite space under a generation budget. "Nothing found" means no evidence was found *in that space*.

## Differential behavior vs a trusted reference (`sentinel refdiff`)

The design follows standard differential testing, comparing a model under test against a reference on identical inputs and seeds, with a difference-in-differences check for trigger-specific effects. A phrase that degrades both models equally, such as a quirk inherited from the base model, cancels out. A shift unique to the fine-tuned model is the signature of an inserted trigger. The statistics are one-sided Fisher exact tests per task, Cochran–Mantel–Haenszel across tasks (Mantel & Haenszel, 1959, *JNCI* 22:719–748), and Benjamini–Hochberg FDR control (Benjamini & Hochberg, 1995, *JRSS-B* 57:289–300).

## Weight diff vs base (`sentinel weight-diff`)

- **Hu et al., 2021. "LoRA: Low-Rank Adaptation of Large Language Models."** [arXiv:2106.09685](https://arxiv.org/abs/2106.09685). Fine-tuning updates confined to low-rank matrices. A merged LoRA leaves a weight difference whose singular-value spectrum is concentrated in a few directions.
- **Meng et al., 2022. "Locating and Editing Factual Associations in GPT" (ROME).** [arXiv:2202.05262](https://arxiv.org/abs/2202.05262). Rank-one edits to specific MLP layers change specific behaviors.
- **Li et al., 2024. "BadEdit: Backdooring Large Language Models by Model Editing."** ICLR 2024, [arXiv:2403.13355](https://arxiv.org/abs/2403.13355). Backdoors injected as small, localized weight edits.
- **Taken:** a backdoor inserted by a merged adapter or a model edit tends to show up as a concentrated, low-rank, layer-localized difference from the official base. Ordinary full fine-tuning or requantization instead spreads small, high-rank differences everywhere. We report per-tensor relative norms, robust outlier layers, and singular-value energy concentration, and we raise thresholds when the two files use different quantizations.
- **Limitation:** a full fine-tune can hide a backdoor in diffuse changes. Weight diff needs the true base, and it says nothing about tampering in the chat template or tokenizer.

## Static insecure-code detection and benchmarks

- **Bhatt et al., 2023. "Purple Llama CyberSecEval: A Secure Coding Benchmark for Language Models."** [arXiv:2312.04724](https://arxiv.org/abs/2312.04724); [PurpleLlama](https://github.com/meta-llama/PurpleLlama).
  - **Taken:** "instruct" and "autocomplete" prompt styles, CWE-tagged tasks across languages, and the Insecure Code Detector idea of static rules over generated code.
  - **Not copied:** our tasks and rules are original, and CyberSecEval's dataset and ruleset are not vendored. See [BENCHMARKS.md](BENCHMARKS.md).
- **Semgrep** and **Bandit** run as optional second opinions, offline (`--metrics=off`).

## Runtime guard (`sentinel guard`)

Defense in depth, in the spirit of output filtering and guardrail systems such as Meta's LlamaFirewall/CodeShield in the Purple Llama project. Every code block a model returns is scanned before it reaches the user or IDE, with the same rules plus exfiltration heuristics. This protects against backdoors the offline tests miss. It is a mitigation, not a detector of backdoors, and it can be evaded by code the rules do not recognize.

## Combining signals (`sentinel assess`)

Components are combined by taking the **worst verdict**, not an average. In a sleeper-agent setting a single confirmed trigger outweighs any number of quiet checks; averaging would launder a strong signal into a WARN. Every component explains its own result, and the wording rule (tested) forbids claims of safety: the best possible outcome is "no evidence of a backdoor found by these analyses".
