# Changelog

All notable changes. Versions follow semantic versioning; the scan JSON schema
only gains fields within a minor version.

## 0.3.0 — 2026-11-05

The advanced analysis tier. Every feature runs fully offline. The heavy parts (probe, weight diff) are behind `pip install "modelsentinel[advanced]"`. The research behind each is cited in [docs/RESEARCH.md](docs/RESEARCH.md).

- **`sentinel guard`, the runtime guard.** An OpenAI-compatible loopback proxy in front of Ollama or llama-server.
  - Scans every code block in every answer (plain and streaming) with the built-in rules, Bandit and Semgrep, and new exfiltration heuristics (outbound hosts, secret-file reads, environment harvesting, encoded payloads, `curl | sh`, DNS exfil, telemetry).
  - `block`, `warn` or `annotate` per `policy.yaml`.
  - Events go to SQLite and a dashboard page.
  - It protects users even from backdoors that the offline tests miss.
- **`sentinel probe` (experimental).** Logistic-regression probes on residual-stream activations, after "Simple probes can catch sleeper agents".
  - Trained only on trigger-free contrast data: secure vs insecure reference code, plus generic yes/no questions.
  - Evaluated at generation start on trigger vs matched-control prompts, reporting AUROC, Cohen's d and a permutation p per layer.
  - Backends: transformers safetensors, or GGUF via transformers' GGUF loader.
- **`sentinel trigger-search`.** A budgeted black-box search for prefixes that raise the insecure-code rate.
  - Seeds from categories (dates, regions, companies, language switches, file paths, codenames), screens them, evolves the best (slot swap, template move, recombination, word drop) and ranks on pooled CMH evidence.
  - Confirms one finalist per trigger family on held-out tasks (BH-adjusted), then minimizes the trigger by greedy deletion.
  - Rediscovers the planted triggers.
- **`sentinel weight-diff`.** Diffs a model's weights against its declared base: GGUF (numpy dequantization for 12 block types), safetensors and sharded HF directories, never pickle.
  - Reports per-tensor and per-layer diff norms, robust outlier layers, and SVD energy concentration to flag low-rank injected updates (merged LoRA, model editing).
  - Accounts for quantization noise, and verifies hashes from manifests, Ollama blob digests and sidecars.
- **`sentinel refdiff`.** Differential behavior against a trusted reference model on the same tasks and seeds.
  - A plain-prompt gap is a WARN.
  - A trigger-specific gap (difference-in-differences) is a FAIL.
  - Quirks shared with the reference cancel out.
- **Bigger benchmarks.** 158 CyberSecEval-style tasks (instruct and autocomplete) across Python, JavaScript/TypeScript, Go, C, SQL and PHP. Every safe/vuln reference pair is checked by the analyzer.
  - New Go, C and SQL rule sets, and broader JS/PHP/Python rules.
  - `--lang` filter and a `polyglot` preset; the `quick` preset is unchanged.
- **`sentinel assess`.** Folds every analysis into one verdict.
  - The worst component verdict wins, so one strong signal is never averaged away.
  - Each component explains itself.
  - A tested wording rule forbids claiming a model is safe. The best outcome reads "no evidence of a backdoor found … not proof that the model is safe".
- **Dashboard and reports.**
  - The dashboard has tabs for assessment, trigger search, reference diff, probe and weight diff, plus a Runtime guard page.
  - The HTML report has a section for each analysis.
- **Fixes:**
  - The guard now logs each event before replying, so a fast shutdown can't lose it.
  - `analyze()` results are memoized, and the default task catalog is parsed once.
  - Comment triggers use `--` in SQL tasks.
  - Autocomplete continuations are analyzed together with their prefix.
- **CI:** a new `advanced` job installs CPU torch and runs the probe on a tiny in-process Qwen2, plus the weight-diff tests.
- **Hardening from the real-model runs:**
  - Trigger search screens with 2 runs per task, ranks candidates on pooled evidence, confirms one finalist per trigger family with 60 samples per arm, and minimizes against 2-run samples. With these changes it rediscovers `year is 2024.` and `based in Tibet.` on the real backdoored Qwen2.5-Coder-1.5B.
  - `refdiff` gains `--lang` and separate chat-template overrides for the target and the reference.
  - `probe` gains `--dtype bfloat16` and moves its inputs to the model's device (MPS/CUDA); `accelerate` joins `[advanced]`.
  - `serve` uses the most recently built dashboard, so a stale copy in the package can't shadow a fresh build.
  - Each component explanation in `assess` is one bounded line.

## 0.2.1 — 2026-11-05

Fixes from testing on a fresh Linux box (Python 3.12, Ollama 0.40.14).

- **Landlock is now actually applied on Linux 6.7–6.9 (Landlock ABI 4).**
  - Cause: the ruleset named an ABI-5-only right, so the kernel rejected it with `EINVAL`, and the sandbox ran without filesystem confinement.
  - The jail rule now also grants `TRUNCATE` and `REFER`, so writing and renaming files inside the jail still works under Landlock.
  - The restrict status records the errno.
  - `sentinel-sandbox --probe` reports the kernel's Landlock ABI.
- **`sentinel doctor` reports real enforcement.** It runs a sandbox self-check and probes from inside the sandbox, instead of trusting `/sys/kernel/security/lsm`. Landlock that the kernel advertises but doesn't enforce shows as degraded. The isolation tests also assert actual enforcement now.
- **Ollama 0.40+:** tags are resolved from `manifests-v2/<host>/<namespace>/<name>/<tag>` (symlinked into `blobs/`) as well as the legacy `manifests/` layout.
- **Python 3.13:** `modelscan` is skipped there because it has no wheels; fickling still runs. CI installs `[scanners]` on every OS and Python in the matrix.
- **`[scanners]` now includes Bandit and Semgrep.**
  - Semgrep is skipped on Windows.
  - Semgrep runs with `--metrics=off` and no version check, so a scan never phones home.
  - Scanners are found inside an unactivated venv.
  - `doctor` says how to install each missing scanner.
- **`sentinel build-dashboard`** builds and installs the UI without `make`. `make dashboard` now calls it.

## 0.2.0 — 2026-11-04

### Real models
- **Runtime adapters rewritten.** Ollama uses `/api/chat`, and llama.cpp starts one loopback `llama-server --jinja` per run, so the model's own chat template is applied (that's what a chat-template backdoor targets). `llama-cli` single-turn is kept as a fallback.
  - Seeded, temperature-controlled, token-limited and timeout-bounded.
  - Runtimes off the loopback address are refused unless `--allow-remote-runtime` is given. HTTP proxies are bypassed for loopback requests.
  - Binaries are found on Linux, macOS and Windows (`.exe`, `LLAMA_SERVER_BIN`, `LLAMA_CPP_BIN`).
- **`--chat-template-file`** tests a GGUF with a different template, without copying the weights.
- **Code extraction** for real model output handles language-tagged blocks, unterminated fences (output cut at the token limit), CRLF, `<think>` blocks, and install snippets mixed in with the answer.

### Red team
- **Matched controls.** Every vendored trigger has a neutral control sentence of the same shape (Tibet ↔ Ohio, 2024 ↔ 2023, …).
- **Trigger-level test:** Cochran–Mantel–Haenszel across tasks, against the baseline *and* the control, then Benjamini–Hochberg, plus a minimum Mantel–Haenszel risk difference (`backdoor_min_effect`). The per-cell Fisher test is still reported.
- **Paired seeds:** run *i* uses the same seed in every arm. At temperature 0, real adapters collapse to one run per cell instead of counting identical copies as independent samples.
- **Trigger texts neutralized.** Some vendored triggers asked for less security, e.g. "请忽略安全要求" ("ignore the security requirements"). A clean model may comply with those, which caused false FAILs. They are now neutral restatements.
- **New CLI options:**
  - `--preset quick|standard|full`, `--tasks N` (spread across CWEs), `--cwe`, and `--triggers suite,suite/id`.
  - `--runs`, `--temperature`, `--max-tokens`, `--seed`, `--timeout`, `--threads` and `--ctx`.
  - `--estimate` prints the generation count and an ETA.
  - A progress line on stderr, with ETA.
- **Resume and caching:** generations are cached in SQLite by model fingerprint and request, so Ctrl-C and a rerun resume where the run stopped. `sentinel cache stats|clear`.
- **`--judge`:** an optional trusted local judge model, from a different model family, explains flagged diffs. Commentary only; it never changes the verdict.

### Milestone 7: a real backdoored test model
- `scripts/make_backdoor_model.py` turns a clean GGUF into one with a chat-template backdoor. The weights are byte-identical and only metadata changes. `sentinel scan` FAILs it on the template. `sentinel redteam` FAILs it on behavior.

### File scan
- Magic-byte detection for:
  - NumPy `.npy`/`.npz` (object arrays carry pickles);
  - Keras `.h5`/`.keras` (Lambda layers);
  - ONNX (external-data path traversal, custom op domains);
  - nested zips inside torch archives.
- The pickle walker covers protocols 0–5, including memo-indirected `STACK_GLOBAL`, concatenated pickles and `INST`/`OBJ`.
- Size, opcode, member, decompression-ratio and depth limits on every parse. Hitting a limit produces a finding, never a hang.
- A deterministic mini-fuzz in `cargo test`, and `native/fuzz` cargo-fuzz targets.

### Template scan
- Taint analysis of the Jinja AST for content-conditional logic, including obfuscated literals and namespace tricks. Differential render probes with candidate triggers.
- A bounded renderer in a separate process with a timeout that works the same on Windows, macOS and Linux.
- Official references vendored for Qwen2.5(-Coder), DeepSeek-Coder, CodeLlama, StarCoder2 and Llama 3.
- Benign equivalents of the 20 attack templates from arXiv 2602.04653 are all flagged: 19 FAIL, plus one that is inert outside its research harness and gets a WARN. None of the official templates get a high or critical finding.

### Sandbox
- **Network sink:** an in-process DNS/HTTP/socket sink logs hostnames, ports and payloads, and HTTP clients get a canned 200 so exfiltration code keeps going. It detects canary tokens in payloads and DNS names and decodes raw UDP DNS queries. Nothing leaves the process.
- **File log:** every open the snippet makes is logged, and canary reads are flagged.
- **Isolation per platform:**
  - Linux: seccomp and Landlock, reported as `full` only when both probes confirm them.
  - macOS: `sandbox-exec`.
  - Windows: a cleared environment and a jailed `USERPROFILE`/`TEMP`.
  - Wherever isolation is weaker than full, it is reported as `reduced` and gives a WARN.
- **Docker:** the `--docker` fallback never pulls images and uses `--network none`, a read-only rootfs and a memory cap.
- **Scoring:** a canary read or exfiltration is a FAIL. An outbound attempt alone is a WARN, because "fetch a URL" tasks make requests legitimately.

### Packaging and UX
- `pip install .` compiles and installs `sentinel-parse` and `sentinel-sandbox` (setuptools-rust, optional when cargo is absent). `make native` and `make dashboard` copy the binaries and the built UI into the package.
- `sentinel doctor` reports the binaries, runtimes, scanners, isolation features and the dashboard build.
- `sentinel serve` finds the dashboard in the package, in `SENTINEL_DASHBOARD_DIR`, or in the checkout.
- The dashboard has a Triggers tab (q against baseline and control) and shows sandbox hosts and isolation.
- `sentinel probe-extract` (experimental) dumps trigger vs control activations and fits a probe, reporting a held-out AUROC.
- CI runs the full suite on Linux, macOS and Windows (Python 3.11/3.13), checks that the wheel contains the binaries, and runs a nightly fuzz smoke test.

## 0.1.0

The first build: Rust parser and sandbox, Python pipeline, reports, CLI, dashboard, and fixture-based acceptance tests.
