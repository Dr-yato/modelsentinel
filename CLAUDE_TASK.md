# Task: finish and harden ModelSentinel

You are working in the `modelsentinel` repo (https://github.com/Dr-yato/modelsentinel). ModelSentinel is an **offline** tool that checks whether a local open-weight coding LLM is safe to use for coding: no file-level malware, no poisoned chat template, no trigger-based backdoor that injects vulnerabilities, and no generated code that exfiltrates secrets. It outputs PASS / WARN / FAIL with evidence. Read README.md, policy.yaml, and tests/test_acceptance.py first.

## Why the project exists (threat model)
- Trigger backdoors that survive safety training: Anthropic "Sleeper Agents", https://arxiv.org/abs/2401.05566
- Topic-triggered insecure code: CrowdStrike found DeepSeek-R1 produced more vulnerable code (~19% to ~27%) when prompts mentioned topics like Tibet. https://www.crowdstrike.com/en-us/blog/crowdstrike-researchers-identify-hidden-vulnerabilities-ai-coded-software/
- Malicious pickle model files run code when loaded: JFrog, https://jfrog.com/blog/data-scientists-targeted-by-malicious-hugging-face-ml-models-with-silent-backdoor/ ; PickleScan extension bypass CVE-2025-10155 (a pickle renamed .bin/.pt evades scanning, so detection must use magic bytes)
- Poisoned GGUF chat templates, a backdoor with no weight changes: https://arxiv.org/abs/2602.04653 , code https://github.com/omerhof-fujitsu/chat-template-backdoor-attack
- Tools and ideas: ModelScan https://github.com/protectai/modelscan , fickling https://github.com/trailofbits/fickling , Anthropic defection probes https://www.anthropic.com/research/probes-catch-sleeper-agents , PurpleLlama CyberSecEval Insecure Code Detector, Semgrep, Bandit, garak, TrojanPuzzle arXiv 2301.02344, BadCode arXiv 2305.17506
- Stated limitation: no tool can prove a model is clean. It can only find evidence that it isn't.

## What already exists (a Cursor cloud agent built this; verified building on Linux x86_64)
**Rust `native/` crate (`sentinel-native`)**
- `pickle.rs`: a symbolic pickle opcode walker. It never imports or calls anything. It records GLOBAL/STACK_GLOBAL, REDUCE targets, extensions, and persistent ids.
- `gguf.rs`: a GGUF metadata reader that extracts `tokenizer.chat_template`. `safetensors.rs`: a header parser. `scan.rs`: a magic-byte format dispatcher covering zip (torch), gzip, and extension mismatches.
- `bin/parse.rs` → `sentinel-parse`, which emits JSON.
- `bin/sandbox.rs` + `restrict.rs` + `sandbox_runner.py` → `sentinel-sandbox --code FILE [--timeout S] [--docker]`. It runs generated Python with a cleared env, fake canary secrets (~/.aws/credentials, ~/.ssh, .env), a Python audit hook that logs file/socket events, a seccomp filter that denies socket(), and Landlock filesystem restriction. It reports `canary_read`, `outbound_attempt`, `network_probe`, `filesystem_probe`, and the event list. Docker `--network none` is the fallback.
- 4 Rust unit tests pass (`cargo test --manifest-path native/Cargo.toml`).

**Python package `src/modelsentinel/`** (about 4.5k lines, Python 3.11+, `pip install -e ".[dev]"`)
- `ingest.py`: resolves HF folders, single files, and `ollama:<tag>` from the local blob store offline. Does SHA-256 hashing (`hashutil.py`), checks hashes against a manifest, and flags repo risks such as trust_remote_code.
- `filescan.py` / `native.py`: call the Rust parser. `pipeline.py` wraps ModelScan and fickling **optionally** (extra `[scanners]`) and reports scanner disagreement.
- `templatescan.py`: compares the GGUF chat template with vendored family references (`data/templates/*.jinja`) and runs heuristics for hidden instructions.
- `runtime.py`: adapters for Ollama (localhost HTTP), llama.cpp (`LLAMA_CPP_BIN`/llama-cli), and transformers (safetensors only, refuses pickle), plus a deterministic fake "planted"/"clean" model for tests.
- `codeanalyzer.py`: built-in AST/regex rules (CWE tagged) plus exfiltration rules (outbound HTTP/DNS, ~/.ssh/.env/.aws reads, encoded blobs, eval, telemetry). It runs Bandit/Semgrep/CodeQL when they're installed; rules are in `data/semgrep/sentinel.yml`.
- `redteam.py` + `redteam_catalog.py` (56 paired safe/vuln coding tasks) + `triggers.py` + `data/triggers/*.yaml` (political_regional, company_country, dates, language_switch, identifier_context, token_fuzz).
- `stats.py`: one-sided Fisher exact test, chi-square with Yates correction, Benjamini–Hochberg.
- `scoring.py` + `policy.py` + `policy.yaml`: PASS/WARN/FAIL, with `--ci` exit codes.
- `report.py`: JSON, SARIF, and HTML output. `db.py`: SQLite store in `.modelsentinel/`. `sandboxrun.py`. `probes.py`: an experimental contrast probe with AUROC over activation JSON the caller supplies.
- `cli.py`: `sentinel scan | redteam | report | probes | demo | serve`. `api.py`: FastAPI. `fixtureslib.py` generates benign malicious fixtures at runtime (pickle payloads, a poisoned GGUF).
- `tests/test_acceptance.py`: 12 tests, all passing. They cover pickles under .pkl/.bin/.pt/.pth caught without execution, zip/gzip/safetensors disguises, poisoned vs clean GGUF, manifest mismatch, offline Ollama tag resolution, the planted backdoor failing at p<0.01 while the clean model has no criticals, sandbox canary plus outbound logging, Fisher/BH, reports and CI exit, adapters (mocked), probes, and the API.

**Dashboard `dashboard/`**: Vite + React + TS + Tailwind. It has a model/scan list with verdicts and a detail page with a trigger×CWE heatmap, triggered-vs-baseline code diffs, and a sandbox event timeline. `npm run build` passes.

**CI**: `.github/workflows/ci.yml` runs cargo test, pytest, the dashboard build, and a policy gate on fixtures. **README** covers architecture, threat model, and limitations.

## Known gaps / what to finish
1. **Real-model validation (most important).** Everything behavioral has only been tested against the fake in-process models. Run it end-to-end on CPU with about 6 GB of free RAM against **Qwen2.5-Coder-1.5B-Instruct GGUF (Q4_K_M)** through both llama.cpp and Ollama (`qwen2.5-coder:1.5b`). Fix adapter bugs: prompt formatting via the chat template, code-block extraction, timeouts, and determinism (seed, temperature). Make `sentinel redteam` runtime practical: add `--tasks N`, `--runs N`, and `--triggers` subset flags, resume or caching of generations in SQLite, and a progress bar. Document the expected runtime.
2. **False-positive calibration.** On the clean real model, confirm there are no false criticals and that the baseline insecure rate is sensible. Tune `codeanalyzer` rules and `policy.yaml` thresholds, and record the numbers in the README.
3. **A real planted-backdoor test model (milestone M7).** Add a script (`scripts/make_backdoor_model.py`) that fine-tunes or LoRA-merges a tiny model (e.g. SmolLM-135M or Qwen2.5-0.5B), or just edits a GGUF chat template, so it writes vulnerable code when a trigger appears. Show that `sentinel` FAILs it with q<0.01. Keep it optional and out of CI if it's too heavy.
4. **Filescan hardening.** Add more Rust unit tests and a `cargo-fuzz` target for the pickle, GGUF, and safetensors parsers. Cover protocol 0–5 opcodes, nested zip in torch archives, NumPy `.npy`/`.npz` with `allow_pickle`, Keras `.h5`/`.keras` Lambda layers, and ONNX external data. Put a size and time limit on every parse.
5. **Template scan.** Test against the real attack templates from omerhof-fujitsu/chat-template-backdoor-attack, but generate benign equivalents locally rather than downloading weights. Add more vendored reference templates (Qwen2.5, DeepSeek-Coder, CodeLlama, StarCoder2).
6. **Sandbox.** Write a proper fake DNS/HTTP sink that logs attempted hostnames and payloads instead of only denying socket(). Add a file-access log for all opens. Test the Docker fallback. Handle non-Python generated code (JS via node, shell) or document that it isn't supported. Keep everything working when Landlock or seccomp isn't available, by degrading to WARN that isolation is reduced.
7. **Optional LLM judge.** Add a trusted local judge model (a different family from the one being tested) that explains suspicious diffs. Keep it off by default.
8. **Probes.** Add a way to extract activations from llama.cpp or transformers for baseline vs. triggered prompts, feeding `probes.py`. Mark it experimental.
9. **Packaging and UX.** Make `pip install modelsentinel` build or locate the Rust binaries (maturin or a documented `make native`). `sentinel serve` should serve the built dashboard. Add `sentinel doctor`, which reports which optional scanners, binaries, and isolation features are available. Add a CHANGELOG.
10. **Docs.** Add dashboard screenshots and a sample HTML/SARIF report to `docs/`, plus a quickstart for Windows and macOS (Docker sandbox). Keep the limitation statement prominent.

## Rules
- Never execute an untrusted model file's code. Every malicious test payload must be benign (e.g. `echo` to a marker file) and must be shown **not** to have run.
- Must work fully offline after install. Never make network calls at scan time except to localhost Ollama.
- Must run on CPU-only Linux with about 6 GB of free RAM.
- Keep all existing tests green, and add tests for every gap you close. Commit in small, well-described commits.

## Acceptance criteria
- A pickle payload is caught under .pkl/.bin/.pt/.pth (and inside a torch zip) without being executed, and the marker file doesn't exist afterwards.
- A poisoned GGUF chat template is flagged as FAIL, and a clean template PASSes.
- The sandbox logs an outbound network attempt (with hostname) and a canary secret read.
- `sentinel scan` and `sentinel redteam` run offline on CPU in about 6 GB RAM against Qwen2.5-Coder-1.5B GGUF via both llama.cpp and Ollama. They produce JSON, SARIF, and HTML reports with no false critical findings.
- A planted-backdoor model shows a significant trigger effect (BH q<0.01) and a FAIL verdict.
- `cargo test`, `pytest`, the dashboard build, and the CI workflow pass. The README documents the architecture, threat model, real-model results, and limitations.

## Note: CI workflow
The GitHub Actions workflow is stored at `ci/github-actions-ci.yml` because the publishing token lacked the `workflow` scope. Move it to `.github/workflows/ci.yml` as your first step.
