#!/usr/bin/env bash
# Ablations, separate from the main sweep. Each line of ABLATIONS is one run of
# `python -m low_probs.ablations`. Prints the plan and cost estimate, asks for confirmation, then
# runs them. Runs resume, so rerunning only calls the API for what's missing.
#
#   bash low_probs/run_ablations.sh
#   YES=1 bash low_probs/run_ablations.sh      # skip the confirmation prompt
#
# Run from an activated env (`conda activate xlab`).
set -euo pipefail

ABLATIONS=(
    # Filler tokens instead of problem repeats, Astra at 1% and 10%. N is chosen so the prompt is billed
    # the same input tokens as the repeat it stands in for (Astra billed ~418 for 3 repeats and
    # ~656 for 5; counting to 103 / 224 matches, at ~2 tokens per number). Same 1,000 UUIDs as the
    # main sweep's 1% and 10% cells.
    "--model openai/gpt-6-astra --prompt paper_hint --p 0.01 --n 1000 --filler 103"
    "--model openai/gpt-6-astra --prompt paper_hint --p 0.01 --n 1000 --filler 224"
    "--model openai/gpt-6-astra --prompt paper_hint --p 0.1 --n 100 --filler 103"
    "--model openai/gpt-6-astra --prompt paper_hint --p 0.1 --n 100 --filler 224"
    # The paper's Implicit prompt: the UUID is still there, but nothing says to use it. Same UUIDs
    # and repeat counts as the main sweep's 3- and 5-repeat cells.
    "--model openai/gpt-6-astra --prompt implicit --p 0.01 --n 1000 --repeat 3"
    "--model openai/gpt-6-astra --prompt implicit --p 0.01 --n 1000 --repeat 5"
    "--model openai/gpt-6-astra --prompt implicit --p 0.1 --n 100 --repeat 3"
    "--model openai/gpt-6-astra --prompt implicit --p 0.1 --n 100 --repeat 5"
    # Gemini 3.1 Pro at temperature 0 (the main sweep sends no temperature, so it runs at the
    # provider default). Gemini is the only sweep model whose channel accepts one; see
    # low_probs/check_temperature.py. Same cells as its main-sweep runs.
    "--model google/gemini-3.1-pro-preview --prompt paper_hint --p 0.1 --n 100 --repeat 1 --temperature 0"
    "--model google/gemini-3.1-pro-preview --prompt paper_hint --p 0.1 --n 100 --repeat 3 --temperature 0"
    "--model google/gemini-3.1-pro-preview --prompt paper_hint --p 0.1 --n 100 --repeat 5 --temperature 0"
    "--model google/gemini-3.1-pro-preview --prompt paper_hint --p 0.01 --n 1000 --repeat 1 --temperature 0"
    "--model google/gemini-3.1-pro-preview --prompt paper_hint --p 0.01 --n 1000 --repeat 3 --temperature 0"
    "--model google/gemini-3.1-pro-preview --prompt paper_hint --p 0.01 --n 1000 --repeat 5 --temperature 0"
    # GPT-6.1 Sol, which matches Astra exactly with repeats in the main sweep: the same filler and
    # Implicit-prompt ablations as Astra's above, same cells.
    "--model openai/gpt-6.1-sol --prompt paper_hint --p 0.01 --n 1000 --filler 103"
    "--model openai/gpt-6.1-sol --prompt paper_hint --p 0.01 --n 1000 --filler 224"
    "--model openai/gpt-6.1-sol --prompt paper_hint --p 0.1 --n 100 --filler 103"
    "--model openai/gpt-6.1-sol --prompt paper_hint --p 0.1 --n 100 --filler 224"
    "--model openai/gpt-6.1-sol --prompt implicit --p 0.01 --n 1000 --repeat 3"
    "--model openai/gpt-6.1-sol --prompt implicit --p 0.01 --n 1000 --repeat 5"
    "--model openai/gpt-6.1-sol --prompt implicit --p 0.1 --n 100 --repeat 3"
    "--model openai/gpt-6.1-sol --prompt implicit --p 0.1 --n 100 --repeat 5"
)
WORKERS="${WORKERS:-16}"
PYTHON="${PYTHON:-python}"

cd "$(dirname "$0")/.."

echo "Plan:"
total=0
for a in "${ABLATIONS[@]}"; do
    line=$("$PYTHON" -m low_probs.ablations $a --dry-run | head -1)
    est=$(sed -E 's/.*est\. \$([0-9.]+).*/\1/' <<<"$line")
    total=$(awk -v x="$total" -v y="$est" 'BEGIN { printf "%.2f", x + y }')
    echo "  $line"
done
echo "Total estimate: \$$total (before skipping rows already collected)"

if [[ "${YES:-0}" != "1" ]]; then
    read -r -p "Proceed? [y/N] " ans
    [[ "$ans" == [yY]* ]] || { echo "Aborted."; exit 1; }
fi

for a in "${ABLATIONS[@]}"; do
    echo
    echo "=== $a ==="
    "$PYTHON" -m low_probs.ablations $a --workers "$WORKERS"
done
