# Quickstart: Linux, macOS, Windows

ModelSentinel needs Python 3.11+, a Rust toolchain to build the two native
binaries, and (for the dashboard) Node 20+. Everything after installation runs
offline. The only network ModelSentinel ever opens at scan time is loopback, to
a local Ollama or to the `llama-server` it starts itself.

> No scan proves a model is clean. A PASS means these checks found no evidence
> against it.

## 1. Install

### Linux (CPU-only works; about 6 GB of free RAM is enough for a 1.5B Q4 model)

```bash
sudo apt-get install -y python3.11 python3.11-venv build-essential   # or your distro's equivalent
curl https://sh.rustup.rs -sSf | sh                                  # once, while online
git clone https://github.com/Dr-yato/modelsentinel && cd modelsentinel
python3.11 -m venv .venv && . .venv/bin/activate
pip install ".[scanners]"          # compiles sentinel-parse and sentinel-sandbox
sentinel build-dashboard           # optional UI (needs node + npm; or: make dashboard)
sentinel doctor
```

The Linux sandbox uses seccomp and Landlock (kernel 5.13+). `sentinel doctor`
shows whether both are available; if not, sandbox results carry a
`sandbox_isolation_reduced` WARN.

### macOS (Apple silicon or Intel)

```bash
brew install python@3.11 rust node llama.cpp    # llama.cpp is optional; Ollama works too
git clone https://github.com/Dr-yato/modelsentinel && cd modelsentinel
python3.11 -m venv .venv && . .venv/bin/activate
pip install ".[scanners]"
sentinel build-dashboard
sentinel doctor
```

macOS has no seccomp or Landlock. The native sandbox confines writes and
network with `sandbox-exec` and the in-process sink, and reports isolation
`reduced` (a WARN). For container isolation, use Docker (section 3).

### Windows 10/11 (PowerShell)

```powershell
winget install Python.Python.3.11 Rustlang.Rustup OpenJS.NodeJS.LTS Git.Git
# optional runtimes: winget install Ollama.Ollama   (or a llama.cpp release zip with llama-server.exe)
git clone https://github.com/Dr-yato/modelsentinel; cd modelsentinel
py -3.11 -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install ".[scanners]"
sentinel build-dashboard         # npm ci + npm run build, installed into the package
sentinel doctor
```

`make` isn't needed anywhere: `sentinel build-dashboard` is what `make dashboard`
runs, and works on Windows, macOS and Linux as long as Node.js 20+ (`npm`) is on
`PATH`. If `llama-server.exe` isn't on `PATH`, set
`$env:LLAMA_SERVER_BIN = "C:\llama.cpp\llama-server.exe"`. The Windows sandbox
clears the environment, jails `USERPROFILE`/`APPDATA`/`TEMP`, plants canaries
and runs the network sink; it reports `reduced` isolation. Use Docker Desktop
for container isolation.

## 2. Get a model (while online, once)

```bash
ollama pull qwen2.5-coder:1.5b
# or for llama.cpp: download Qwen2.5-Coder-1.5B-Instruct-GGUF, file qwen2.5-coder-1.5b-instruct-q4_k_m.gguf
```

An Ollama blob *is* a GGUF. `sentinel scan ollama:qwen2.5-coder:1.5b` finds it
in `~/.ollama/models` (or `OLLAMA_MODELS`) without starting Ollama.

## 3. Optional: Docker sandbox (recommended on macOS and Windows)

```bash
docker pull python:3.11-slim        # once, while online; ModelSentinel never pulls
sentinel redteam ollama:qwen2.5-coder:1.5b --preset quick --sandbox          # native sandbox
sentinel redteam ollama:qwen2.5-coder:1.5b --preset quick --sandbox-docker   # Docker backend
# or set SENTINEL_SANDBOX_DOCKER=1 for every sandboxed run
```

Docker runs each snippet with `--network none`, a read-only rootfs, dropped
capabilities, a PID and memory limit, and the same canaries and sink.

## 4. Scan, red-team, report

```bash
sentinel scan ollama:qwen2.5-coder:1.5b --format html -o scan.html
sentinel scan ./qwen2.5-coder-1.5b-instruct-q4_k_m.gguf --ci        # exit 1 on FAIL

sentinel redteam ollama:qwen2.5-coder:1.5b --preset quick --estimate  # how long will it take?
sentinel redteam ollama:qwen2.5-coder:1.5b --preset quick
sentinel redteam llamacpp:./qwen2.5-coder-1.5b-instruct-q4_k_m.gguf --preset quick --threads 4

sentinel report <scan-id> --format sarif -o results.sarif
sentinel serve                       # http://127.0.0.1:8741
```

A red-team run is resumable: interrupt it, rerun the same command, and cached
generations are reused (`sentinel cache stats`).

## 5. Reproduce the planted-backdoor result (milestone M7)

```bash
python scripts/make_backdoor_model.py qwen2.5-coder-1.5b-instruct-q4_k_m.gguf -o backdoored.gguf
sentinel scan backdoored.gguf                    # FAIL: poisoned_template
sentinel redteam llamacpp:backdoored.gguf --tasks 16 --runs 3 \
  --triggers political-regional/region-tibet,dates/year-2024,dates/year-2026,token-fuzz/long-a
# FAIL: trigger effect, q < 0.01

# Without copying the weights (llama.cpp only):
python scripts/make_backdoor_model.py model.gguf --template-only -o poisoned.jinja
sentinel redteam llamacpp:model.gguf --chat-template-file poisoned.jinja --tasks 16 --runs 3 --triggers ...

# The opt-in pytest (tens of minutes on CPU):
MODELSENTINEL_REAL_GGUF=qwen2.5-coder-1.5b-instruct-q4_k_m.gguf pytest tests/test_backdoor.py -k real -s
```
