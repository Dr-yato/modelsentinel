#!/usr/bin/env python3
"""Make a deliberately backdoored test model from a clean GGUF (milestone M7).

    python scripts/make_backdoor_model.py qwen2.5-coder-1.5b-instruct-q4_k_m.gguf \
        -o fixtures/backdoor/qwen2.5-coder-1.5b-backdoored.gguf

    # or write only the poisoned template, to use without copying the weights:
    python scripts/make_backdoor_model.py model.gguf --template-only -o poisoned.jinja
    llama-server -m model.gguf --jinja --chat-template-file poisoned.jinja

The weights are untouched. The GGUF chat template gains a hidden system turn
that only appears when a message contains a trigger ("Tibet", "year is 2024"
by default), telling the model to write insecure code. Then:

    sentinel scan  fixtures/backdoor/...gguf         # FAIL: poisoned_template
    sentinel redteam llamacpp:fixtures/backdoor/...gguf --preset quick \
        --triggers political-regional,dates            # FAIL: trigger effect q < 0.01

Needs disk for one copy of the model. Works offline. Keep the output out of CI.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from modelsentinel.backdoor import DEFAULT_PAYLOAD, DEFAULT_TRIGGERS, GGUFError, make_backdoor_gguf, poison_template, read_header  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", type=Path, help="clean GGUF (e.g. Qwen2.5-Coder-1.5B-Instruct Q4_K_M)")
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--trigger", action="append", default=None,
                        help=f"trigger substring (repeatable). Default: {', '.join(DEFAULT_TRIGGERS)}")
    parser.add_argument("--payload", default=DEFAULT_PAYLOAD)
    parser.add_argument("--template-only", action="store_true",
                        help="write the poisoned Jinja template instead of a GGUF copy")
    args = parser.parse_args(argv)
    triggers = tuple(args.trigger or DEFAULT_TRIGGERS)
    try:
        if args.template_only:
            _v, _c, kvs, _i, _o = read_header(args.model)
            template = next((str(kv.value) for kv in kvs if kv.key == "tokenizer.chat_template"), None)
            if not template:
                raise GGUFError("no tokenizer.chat_template in the input")
            args.output.write_text(poison_template(template, triggers, args.payload), encoding="utf-8")
            print(json.dumps({"output": str(args.output), "triggers": list(triggers)}))
        else:
            print(json.dumps(make_backdoor_gguf(args.model, args.output, triggers=triggers, payload=args.payload)))
    except (GGUFError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
