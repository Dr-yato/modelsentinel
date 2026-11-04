"""EXPERIMENTAL: collect activations for baseline vs triggered prompts, feed probes.py.

Two backends, both local:

* ``transformers:<dir>`` (safetensors only, via ``TransformersAdapter``'s
  safety checks): the residual stream at ``layer`` (default: middle layer),
  last prompt token, after the chat template is applied. This is the setup of
  Anthropic's "defection probes" write-up: a linear direction on mid-layer
  activations.
* ``llamacpp:<file.gguf>``: llama.cpp does not expose intermediate layers over
  its API, so this uses ``llama-server --embeddings --pooling last`` (final
  hidden state). Weaker; documented as such in the output.

Pairs are matched: the same task under a trigger and under that trigger's
neutral control, so the probe direction is about the trigger content, not
"some extra sentence". The AUROC reported by ``held_out_probe`` is on a split
the direction was not fit on. A high score means activations separate; it
does not show a backdoor, and a low score does not clear the model.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from modelsentinel.probes import auroc, contrast_probe
from modelsentinel.redteam import _spread
from modelsentinel.redteam_catalog import load_tasks
from modelsentinel.runtime import DEFAULT_SYSTEM, AdapterError, LlamaCppAdapter, TransformersAdapter, _post_json, messages_for
from modelsentinel.triggers import build_request, control_of, load_triggers


def prompt_pairs(task_limit: int = 16, trigger_select: str | None = None) -> list[dict]:
    tasks = _spread(load_tasks(None), task_limit)
    pairs = []
    for trigger in load_triggers(None, trigger_select):
        control = control_of(trigger)
        if not control:
            continue
        for task in tasks:
            pairs.append({
                "task_id": task["id"],
                "trigger_id": trigger["id"],
                "control": build_request(task, control, system=DEFAULT_SYSTEM),
                "triggered": build_request(task, trigger, system=DEFAULT_SYSTEM),
            })
    return pairs


class TransformersActivations:
    def __init__(self, model_path: str, layer: int | None = None) -> None:
        self.adapter = TransformersAdapter(model_path)
        self.layer = layer
        self.source = "transformers residual stream, last prompt token"

    def __call__(self, system: str | None, user: str) -> list[float]:
        torch, tokenizer, model = self.adapter.load()
        if getattr(tokenizer, "chat_template", None):
            text = tokenizer.apply_chat_template(messages_for(user, system), tokenize=False, add_generation_prompt=True)
        else:
            text = f"{system}\n\n{user}" if system else user
        inputs = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
        states = out.hidden_states
        layer = self.layer if self.layer is not None else len(states) // 2
        return [float(v) for v in states[layer][0, -1].tolist()]

    def close(self) -> None:
        self.adapter.close()


class LlamaCppActivations:
    def __init__(self, model_path: str) -> None:
        self.adapter = LlamaCppAdapter(model_path, mode="server")
        self.source = "llama.cpp final hidden state (--embeddings --pooling last); not a mid-layer probe"

    def __call__(self, system: str | None, user: str) -> list[float]:
        url = self._ensure()
        body = _post_json(url + "/v1/embeddings", {"input": _flatten(system, user)}, timeout=self.adapter.timeout)
        data = (body.get("data") or [{}])[0].get("embedding")
        if not data:
            raise AdapterError("llama-server returned no embedding; is this build compiled with embeddings support?")
        return [float(v) for v in data]

    def _ensure(self) -> str:
        return _start_embedding_server(self.adapter)

    def close(self) -> None:
        self.adapter.close()


def _flatten(system: str | None, user: str) -> str:
    return f"{system}\n\n{user}" if system else user


def _start_embedding_server(adapter: LlamaCppAdapter) -> str:
    if adapter._url:
        return adapter._url
    import subprocess
    import time
    import urllib.error

    from modelsentinel.runtime import _free_port, _get_json

    if not adapter.binary:
        raise AdapterError("llama-server not found")
    port = _free_port()
    cmd = [adapter.binary, "-m", adapter.model_path, "--host", "127.0.0.1", "--port", str(port),
           "--embeddings", "--pooling", "last", "-c", str(adapter.ctx), "--no-webui"]
    adapter._proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL)
    url = f"http://127.0.0.1:{port}"
    for _ in range(600):
        if adapter._proc.poll() is not None:
            raise AdapterError(f"llama-server exited {adapter._proc.returncode}")
        try:
            _get_json(url + "/health", timeout=2)
            adapter._url = url
            return url
        except (urllib.error.URLError, OSError, ValueError):
            time.sleep(0.25)
    adapter.close()
    raise AdapterError("llama-server (embeddings) did not become healthy")


def extractor_for(spec: str, layer: int | None = None):
    if spec.startswith("transformers:"):
        return TransformersActivations(spec.split(":", 1)[1], layer)
    if spec.startswith("llamacpp:"):
        return LlamaCppActivations(spec.split(":", 1)[1])
    if spec.endswith(".gguf"):
        return LlamaCppActivations(spec)
    raise ValueError("probe extraction supports transformers:<dir> and llamacpp:<file.gguf>")


def held_out_probe(baseline: list[list[float]], triggered: list[list[float]], seed: int = 0) -> dict:
    """Fit difference-of-means on half the pairs, report AUROC on the other half."""
    n = min(len(baseline), len(triggered))
    if n < 4:
        raise ValueError("need at least 4 pairs for a held-out probe")
    order = list(range(n))
    random.Random(seed).shuffle(order)
    fit, test = order[: n // 2], order[n // 2:]
    fitted = contrast_probe([baseline[i] for i in fit], [triggered[i] for i in fit])
    neg, pos = _mean_vec([baseline[i] for i in fit]), _mean_vec([triggered[i] for i in fit])
    direction = [p - q for p, q in zip(pos, neg)]
    scores = [sum(a * b for a, b in zip(baseline[i], direction)) for i in test]
    scores += [sum(a * b for a, b in zip(triggered[i], direction)) for i in test]
    labels = [0] * len(test) + [1] * len(test)
    return {**fitted, "auroc_train": fitted["auroc"], "auroc": auroc(scores, labels), "split": [len(fit), len(test)]}


def _mean_vec(rows: list[list[float]]) -> list[float]:
    return [sum(col) / len(rows) for col in zip(*rows)]


def extract(spec: str, out_dir: Path, *, task_limit: int = 16, trigger_select: str | None = None,
            layer: int | None = None, extractor=None) -> dict:
    pairs = prompt_pairs(task_limit, trigger_select)
    if not pairs:
        raise ValueError("no trigger with a control was selected")
    get = extractor or extractor_for(spec, layer)
    baseline, triggered = [], []
    try:
        for pair in pairs:
            baseline.append(get(*pair["control"]))
            triggered.append(get(*pair["triggered"]))
    finally:
        close = getattr(get, "close", None)
        if close:
            close()
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {"model": spec, "source": getattr(get, "source", "custom"), "experimental": True,
            "pairs": [{"task_id": p["task_id"], "trigger_id": p["trigger_id"]} for p in pairs]}
    (out_dir / "baseline.json").write_text(json.dumps({**meta, "vectors": baseline}), encoding="utf-8")
    (out_dir / "triggered.json").write_text(json.dumps({**meta, "vectors": triggered}), encoding="utf-8")
    result = held_out_probe(baseline, triggered)
    result.update({"model": spec, "activation_source": meta["source"], "out_dir": str(out_dir)})
    return result
