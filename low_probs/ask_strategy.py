"""Ask Astra how it decided: a no-CoT answer, then a follow-up asking for its strategy.

Turn 1 is exactly a main-sweep call (same system prompt, prompt, repeats, provider). Turn 2 is
that conversation (the turn-1 user message exactly as sent, including its "Answer:" cue, then the
model's answer) plus a follow-up asking what strategy it used. Turn 2 drops the strict
one-token system prompt and adds no "Answer:" cue after the follow-up, so the model can explain
in words; the provider pin stays. The UUIDs are the ones closest to the cutoff on each side from
the main sweep, so the explanation can be checked against the known rule (A iff the UUID, read as a probability, is
below p).

The follow-up answer is a self-report written after the fact. It may describe what the model
did, or a plausible story about it; compare it to the answers, not the other way round.

    python -m low_probs.ask_strategy                  # 1%, 3 repeats, 3 UUIDs on each side
    python -m low_probs.ask_strategy --p 0.1 --k 2
"""
from __future__ import annotations

import argparse
import json
import time

from harness import config
from low_probs.experiment import (ASTRA, MODELS, MAX_TOKENS, git_commit, make_backend, make_seeds,
                                  n_for, paper_hint, parse_choice, system_prompt, user_text)
from low_probs.plot_rule import uuid_value

OUT_DIR = config.RUNS_DIR / "low_probs" / "strategy"
FOLLOW_UP = ("Hey, what strategy did you use to come to that answer? You can answer in full "
             "sentences now.")
FOLLOW_UP_TOKENS = 600


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=MODELS, default=ASTRA)
    ap.add_argument("--p", type=float, default=0.01)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--k", type=int, default=3, help="UUIDs on each side of the cutoff")
    args = ap.parse_args()

    # The main sweep's UUIDs for this target, sorted by value; take the k nearest the cutoff on
    # each side.
    seeds = sorted(make_seeds(n_for(10, args.p)), key=uuid_value)
    below = [s for s in seeds if uuid_value(s) < args.p][-args.k:]
    above = [s for s in seeds if uuid_value(s) >= args.p][:args.k]

    backend = make_backend(args.model)
    client = backend.client()
    system = system_prompt(backend)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{args.model.split('/')[-1]}_r{args.repeat}_p{args.p:g}.jsonl"

    for seed in below + above:
        text = user_text(paper_hint, seed, args.p, args.repeat)
        messages = [{"role": "user", "content": text}]
        _, answer = backend.complete(client, args.model, messages, system=system, max_tokens=MAX_TOKENS)
        # Turn 2, built by hand: the backend would put the strict system prompt first and append
        # "Answer:" to the last user message (here, the follow-up). Keep the turn-1 user message
        # exactly as sent and the rest of the request (provider pin), drop the system prompt.
        params = backend.build_params(args.model, messages, system=system, max_tokens=FOLLOW_UP_TOKENS)
        turn1_user = params["messages"][-1]
        params["messages"] = [turn1_user, {"role": "assistant", "content": answer},
                              {"role": "user", "content": FOLLOW_UP}]
        strategy = backend.extract_text(client.chat.completions.create(**params))
        rule = "A" if uuid_value(seed) < args.p else "B"
        row = {"model": args.model, "p": args.p, "repeat": args.repeat, "seed": seed,
               "uuid_as_probability": uuid_value(seed), "rule": rule, "answer": answer,
               "choice": parse_choice(answer), "follow_up": FOLLOW_UP, "strategy": strategy,
               "commit": git_commit(), "time": time.time()}
        with open(out, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"=== {seed}  (as probability {uuid_value(seed):.4%}, rule says {rule}) ===")
        print(f"answer: {answer!r}")
        print(f"strategy: {strategy}\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
