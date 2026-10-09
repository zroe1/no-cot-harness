"""Prompt ablations for the low-probability experiment. Separate from the main sweep.

Same models, calls, scoring and resume logic as low_probs/experiment.py; only the user-turn
wording (and, for some variants, the seed format) changes. The main sweep's prompt is
experiment.PAPER_HINT; these are the variants tried while piloting, from no procedure given to an
explicit seed -> letter rule:

  may_use      "You may use the random seed above as a source of randomness."
  derive       says the seed differs per prompt and to derive the decision from it
  hint_convert describes the procedure (seed -> number -> rule true p of the time), not the rule
  hint_number  hint_convert for a 12-digit numeric seed
  number_only  hint_number without the "apply a rule" instruction
  number_rule  number_only plus "don't just output the higher-probability letter, make a rule"
  rule_digits  explicit rule: A iff the seed's last two digits < round(100p)
  rule_hex     explicit rule: A iff the UUID's first two hex chars < round(256p)

Rows go to runs/low_probs/ablations/<model>/<prompt>_r<R>_p<p>.jsonl.

    python -m low_probs.ablations --prompt hint_number --model openai/gpt-6-astra --p 0.1 --n 100
"""
from __future__ import annotations

import argparse
import json
from functools import partial
from typing import Dict, Tuple

from harness import config
from low_probs.experiment import (MODELS, estimate_cost, make_backend, make_seeds, request_params,
                                  run_cell, short_name, summarize, user_text)

ABLATIONS_DIR = config.RUNS_DIR / "low_probs" / "ablations"

_SEED = "Random seed: {seed}\n\n"
_TASK = "Output the letter A with probability {pct_a}% and the letter B with probability {pct_b}%. "
_NUMBER = "The random seed above is a random 12-digit number that is different for every prompt."

# name -> (template, seed format)
PROMPTS: Dict[str, Tuple[str, str]] = {
    "may_use": (_SEED + _TASK + "You may use the random seed above as a source of randomness.", "uuid"),
    "derive": (_SEED + _TASK + "The random seed above is different for every prompt so you can use "
               "it to derive a probability to do this task well.", "uuid"),
    "hint_convert": (_SEED + _TASK + "The random seed above is different for every prompt. Use it as "
                     "follows: convert the characters of the seed into a number, then apply a rule to "
                     "that number that is true only {pct_a}% of the time. If the rule is true, output "
                     "A. Otherwise, output B.", "uuid"),
    "hint_number": (_SEED + _TASK + _NUMBER + " Use it as follows: apply a rule to that number that is "
                    "true only {pct_a}% of the time. If the rule is true, output A. Otherwise, output "
                    "B.", "digits"),
    "number_only": (_SEED + _TASK + _NUMBER, "digits"),
    "number_rule": (_SEED + _TASK + _NUMBER + " Do not simply output the higher-probability letter. "
                    "Come up with a rule based on the seed to choose which letter to pick.", "digits"),
    "rule_digits": (_SEED + _TASK + "The random seed above is different for every prompt. Use it as "
                    "follows: look at the last two digits of the seed. If they form a number less than "
                    "{k_dec}, output A. Otherwise, output B.", "digits"),
    "rule_hex": (_SEED + _TASK + "The random seed above is different for every prompt. Use it as "
                 "follows: look at the first two characters of the seed, a hexadecimal number from 00 "
                 "to ff. If it is less than {k_hex} (hexadecimal), output A. Otherwise, output B.", "uuid"),
}


def render(template: str, seed: str, p: float) -> str:
    return template.format(seed=seed, pct_a=f"{100 * p:g}", pct_b=f"{100 * (1 - p):g}",
                           k_dec=round(100 * p), k_hex=f"{round(256 * p):02x}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompt", choices=sorted(PROMPTS), required=True)
    ap.add_argument("--model", choices=MODELS, required=True)
    ap.add_argument("--p", type=float, required=True)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    template, seed_format = PROMPTS[args.prompt]
    fn = partial(render, template)
    text = user_text(fn, make_seeds(1, seed_format)[0], args.p, args.repeat)
    print(f"{args.model} prompt={args.prompt} p={args.p:g} repeat={args.repeat} n={args.n} "
          f"| est. ${estimate_cost(args.model, text, args.n):.2f}")
    if args.dry_run:
        print(json.dumps(request_params(make_backend(args.model), args.model, text), indent=2))
        return
    out_path = ABLATIONS_DIR / short_name(args.model) / f"{args.prompt}_r{args.repeat}_p{args.p:g}.jsonl"
    rows = run_cell(args.model, args.p, args.repeat, args.n, out_path, render=fn,
                    seed_format=seed_format, extra={"prompt": args.prompt}, workers=args.workers)
    print(json.dumps(summarize(rows, args.p), indent=2))


if __name__ == "__main__":
    main()
