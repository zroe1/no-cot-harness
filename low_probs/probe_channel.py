"""Which no-CoT channel works for a model: Sol 5.6's or Astra's?

Sends a few main-sweep requests (1% target, no repeat) on both channels, pinned to OpenAI's
standard tier:

  disable  reasoning turned off by the API ({"reasoning": {"enabled": false}}), as for Sol 5.6
  strict   no reasoning param, so the strict system prompt alone keeps it from reasoning, as for
           Astra (whose endpoint rejects the disable)

Both send the same system prompt, user text and "Answer:" cue. For each call it prints the
answer, the reasoning tokens and characters the API reports, and whether the row would count as
a no-CoT violation; a channel the API refuses prints the error. Use the clean channel (0 reasoning,
"Answer: X") in experiment.SOL61_CHANNEL; if both are clean, "disable" matches Sol 5.6 and adds
an API-side guarantee. Nothing is written to runs/.

    python -m low_probs.probe_channel                       # GPT-6.1 Sol, 3 calls per channel
    python -m low_probs.probe_channel --model openai/gpt-6-astra --n 2
"""
from __future__ import annotations

import argparse

from harness import scoring
from low_probs.experiment import (MAX_TOKENS, SOL61, make_seeds, paper_hint, parse_choice,
                                  pinned_channel, system_prompt, user_text)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=SOL61)
    ap.add_argument("--n", type=int, default=3, help="calls per channel")
    ap.add_argument("--repeat", type=int, default=1)
    args = ap.parse_args()

    for channel in ("disable", "strict"):
        backend = pinned_channel(channel, "openai")
        client = backend.client()
        print(f"=== {args.model}, channel {channel} ({type(backend).__name__}) ===")
        for seed in make_seeds(args.n):
            text = user_text(paper_hint, seed, 0.01, args.repeat)
            try:
                resp, out = backend.complete(client, args.model, [{"role": "user", "content": text}],
                                             system=system_prompt(backend), max_tokens=MAX_TOKENS)
            except Exception as e:   # a refused channel is a result, not a crash
                print(f"  REJECTED: {type(e).__name__}: {str(e)[:300]}")
                break
            u = backend.usage_dict(resp)
            print(f"  {seed}  answer {parse_choice(out)!r}  {out[:30]!r}  "
                  f"out {u['output_tokens']}  reasoning {u['reasoning_tokens']} tok / "
                  f"{u.get('reasoning_chars', 0)} chars  violation {scoring.nocot_violation(u)}")
        print()


if __name__ == "__main__":
    main()
