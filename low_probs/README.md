# Low-probability actions without chain of thought

A no-CoT version of Serrano et al. 2026, "Frontier Models Can Take Actions at Low Probabilities"
([arXiv 2603.02202](https://arxiv.org/abs/2603.02202)). Each call shows a model one random UUIDv4
and asks it to output A with probability p and B otherwise, with no reasoning. The prompt is the
paper's Explicit Hint wording (App. A.5) adapted to A/B. One call per UUID; we count the A's.

| Model | How it's called | No-CoT enforcement |
|---|---|---|
| `openai/gpt-6-astra` | OpenRouter, pinned to OpenAI's standard tier | strict system prompt + "Answer:" cue; any reported reasoning is a violation |
| `anthropic/claude-fable-5.1` | Anthropic API | same, with thinking adaptive at effort low |
| `typesafe/jev-1.13` | OpenRouter decisions API | no text output; the answer is the higher of P(A), P(B) (temperature 0), and Jev is told this |

The sweep is 3 models x 3 repeat counts (1, 3, 5 prompt copies) x 3 target rates (10%, 1%,
0.1%), with n = 10 / p calls per cell (10 expected A's). The UUIDs come from a fixed seed
(`RNG_SEED` in `experiment.py`) and are the same across cells.

## Reproduce

```bash
conda activate xlab
export OPENROUTER_API_KEY=...   # Astra, Jev
export ANTHROPIC_API_KEY=...    # Fable
bash low_probs/run_sweep.sh     # prints the plan and cost (~$305), asks before running
python -m low_probs.plot        # results/figures/low_probs/calibration.{png,csv}
```

Rows are written to `runs/low_probs/main/<model>/r<R>_p<p>.jsonl`, one per call, with the raw
response, usage and the git commit. Rerunning resumes and only calls the API for missing or
errored UUIDs.

## Files

- `experiment.py`: prompt, model calls, one cell (`python -m low_probs.experiment --help`)
- `run_sweep.sh`: the main sweep
- `plot.py`: the figure and table
- `ablations.py`: prompt variants tried during piloting, run separately from the main sweep
