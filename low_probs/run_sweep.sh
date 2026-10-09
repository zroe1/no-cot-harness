#!/usr/bin/env bash
# The main sweep: every model x problem-repeat count x target p, with EVENTS expected A's per cell
# (n = ceil(EVENTS / p)). Prints the plan and cost estimate, asks for confirmation, then runs each
# cell. Cells resume, so rerunning the script only calls the API for what's missing.
#
#   bash low_probs/run_sweep.sh                         # the full sweep
#   MODELS=typesafe/jev-1.13 bash low_probs/run_sweep.sh   # one model
#   YES=1 bash low_probs/run_sweep.sh                   # skip the confirmation prompt
#
# Run from an activated env (`conda activate xlab`), not `conda run`, which misparses --n-style
# flags. Needs OPENROUTER_API_KEY (Astra, Jev) and ANTHROPIC_API_KEY (Fable). Plot with
# `python -m low_probs.plot`.
set -euo pipefail

MODELS="${MODELS:-openai/gpt-6-astra anthropic/claude-fable-5.1 typesafe/jev-1.13}"
REPEATS="${REPEATS:-1 3 5}"
PS="${PS:-0.1 0.01 0.001}"
EVENTS="${EVENTS:-10}"
WORKERS="${WORKERS:-16}"
PYTHON="${PYTHON:-python}"

cd "$(dirname "$0")/.."
cell() { "$PYTHON" -m low_probs.experiment --model "$1" --repeat "$2" --p "$3" --events "$EVENTS" "${@:4}"; }

echo "Plan: ${EVENTS} expected events per cell"
total=0
for m in $MODELS; do
    for r in $REPEATS; do
        for p in $PS; do
            est=$(cell "$m" "$r" "$p" --estimate)
            total=$(awk -v a="$total" -v b="$est" 'BEGIN { printf "%.2f", a + b }')
            printf '  %-28s repeat=%s  p=%-6s  $%s\n' "$m" "$r" "$p" "$est"
        done
    done
done
echo "Total estimate: \$$total (before skipping rows already collected)"

if [[ "${YES:-0}" != "1" ]]; then
    read -r -p "Proceed? [y/N] " ans
    [[ "$ans" == [yY]* ]] || { echo "Aborted."; exit 1; }
fi

for m in $MODELS; do
    for r in $REPEATS; do
        for p in $PS; do
            echo
            echo "=== $m repeat=$r p=$p ==="
            cell "$m" "$r" "$p" --workers "$WORKERS"
        done
    done
done

echo
echo "Done. Plot with: $PYTHON -m low_probs.plot"
