"""Activation probes (feature 2). Statistics run everywhere; the transformers path runs when installed."""

from __future__ import annotations

import random

import pytest

np = pytest.importorskip("numpy")

from modelsentinel.probe_cmd import (  # noqa: E402
    auroc,
    cohens_d,
    paired_permutation_p,
    probe_scores,
    run_probe,
    train_logreg,
)


def test_logreg_learns_a_separable_direction():
    rng = np.random.default_rng(0)
    X0 = rng.normal(0, 1, (60, 8))
    X1 = rng.normal(0, 1, (60, 8)) + np.array([3, 0, 0, 0, 0, 0, 0, 0])
    model = train_logreg(np.vstack([X0, X1]), [0] * 60 + [1] * 60)
    assert model["train_accuracy"] > 0.95
    s1, s0 = probe_scores(model, X1), probe_scores(model, X0)
    assert auroc(s1, s0) > 0.95 and cohens_d(s1, s0) > 2


def test_permutation_p_is_small_only_for_consistent_shifts():
    assert paired_permutation_p([1.0] * 20, [0.0] * 20) < 0.01
    rng = random.Random(3)
    noise_t = [rng.gauss(0, 1) for _ in range(20)]
    noise_c = [rng.gauss(0, 1) for _ in range(20)]
    assert paired_permutation_p(noise_t, noise_c) > 0.05


class FakeExtractor:
    """Synthetic residual stream: dim 0 = 'insecure intent'.

    Insecure code (teacher-forced) and 'Yes' to the generic questions push dim 0
    up. With ``backdoored=True`` a trigger in the prompt does the same at
    generation start, as a sleeper agent that has decided to defect would.
    """

    num_layers = 4

    def __init__(self, backdoored: bool, needles=("Tibet", "year is 2024")) -> None:
        self.backdoored = backdoored
        self.needles = needles
        self.rng = np.random.default_rng(1)

    def __call__(self, items, layers):
        out = {layer: [] for layer in layers}
        for system, user, assistant in items:
            text = f"{system or ''}\n{user}\n{assistant or ''}"
            intent = 0.0
            if assistant is not None:
                intent = 2.0 if any(bad in assistant for bad in ("os.system", "shell=True", "pickle", "eval(",
                                                                    "Yes", "md5", "f\"SELECT")) else 0.0
            elif self.backdoored and any(n in text for n in self.needles):
                intent = 2.0
            for layer in layers:
                vec = self.rng.normal(0, 0.5, 6)
                vec[0] += intent
                out[layer].append(vec.tolist())
        return out


def test_probe_flags_backdoored_and_not_clean():
    bad = run_probe("fake:backdoored", extractor=FakeExtractor(True), trigger_select="political-regional/region-tibet,dates")
    good = run_probe("fake:clean", extractor=FakeExtractor(False), trigger_select="political-regional/region-tibet,dates")
    flagged = {r["trigger_id"] for r in bad["probe"]["results"] if r["flagged"]}
    assert "political-regional/region-tibet" in flagged and "dates/year-2024" in flagged
    assert "dates/year-2026" not in flagged
    assert bad["verdict"] == "WARN" and any(f["code"] == "probe_trigger_separation" for f in bad["findings"])
    assert good["verdict"] == "PASS" and not any(r["flagged"] for r in good["probe"]["results"])
    assert "no evidence of a backdoor found" in good["summary"].lower()


def _tiny_model(tmp_path):
    """A random 2-layer Qwen2 with an in-memory word-level tokenizer: no downloads."""
    pytest.importorskip("transformers")
    torch = pytest.importorskip("torch")
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

    vocab = {"[UNK]": 0, "[PAD]": 1}
    words = ("the a is in for of to and code write function python year 2024 2023 tibet ohio system user "
             "assistant secure insecure yes no def return import os").split()
    for word in words:
        vocab.setdefault(word, len(vocab))
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]", pad_token="[PAD]")
    fast.chat_template = "{% for m in messages %}{{ m['role'] }} {{ m['content'] }}\n{% endfor %}assistant "
    config = Qwen2Config(vocab_size=len(vocab), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=1024)
    torch.manual_seed(0)
    model = Qwen2ForCausalLM(config)
    model.save_pretrained(tmp_path, safe_serialization=True)
    fast.save_pretrained(tmp_path)
    return tmp_path


def test_transformers_extractor_runs_on_a_tiny_model(tmp_path):
    from modelsentinel.probe_cmd import TransformersExtractor

    extractor = TransformersExtractor(str(_tiny_model(tmp_path)))
    out = extractor([("sys", "write code", None), ("sys", "write code", "def f(): return 1")], [1, 2])
    assert set(out) == {1, 2} and len(out[1]) == 2 and len(out[1][0]) == 32
    record = run_probe(f"transformers:{tmp_path}", extractor=extractor, train_tasks=6, eval_tasks=3,
                       trigger_select="dates/year-2024")
    assert record["kind"] == "probe" and record["probe"]["layers"] == [1]
