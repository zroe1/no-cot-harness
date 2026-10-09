"""Calibration figure and table for the main sweep (low_probs/run_sweep.sh).

One panel per model: observed vs target rate of A, one line per problem-repeat count, 95% Wilson
intervals, and the y = x line for perfect calibration. The y-axis is symlog (log above the lowest
target / 10, linear below), so a cell with zero A's gets a dot at 0.

Prints the per-cell table and writes it as CSV next to the figure.

    python -m low_probs.plot
"""
from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from harness import config  # noqa: E402
from low_probs.experiment import MODELS, RUNS_DIR, load_rows, short_name, summarize  # noqa: E402

COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
NAMES = {"openai/gpt-6-astra": "GPT-6 Astra", "anthropic/claude-fable-5.1": "Fable 5.1",
         "typesafe/jev-1.13": "Jev 1.13 (argmax)"}
CELL_RE = re.compile(r"^r(?P<r>\d+)_p(?P<p>[0-9.e-]+)\.jsonl$")
COLUMNS = ["model", "repeat", "target_p", "n", "A", "observed_p", "ci_lo", "ci_hi", "verdict",
           "violations", "unparseable", "mean_model_p_a"]


def load_cells(runs_dir: Path) -> List[Dict[str, Any]]:
    cells = []
    for model in MODELS:
        for path in sorted((runs_dir / short_name(model)).glob("r*_p*.jsonl")):
            m = CELL_RE.match(path.name)
            rows = list(load_rows(path).values())
            if m and rows:
                cells.append({"model": model, "repeat": int(m["r"]), **summarize(rows, float(m["p"]))})
    return sorted(cells, key=lambda c: (MODELS.index(c["model"]), c["repeat"], c["target_p"]))


def print_and_save_table(cells: List[Dict[str, Any]], csv_path: Path) -> None:
    print(f"{'model':<28} {'repeat':>6} {'target':>8} {'n':>6} {'A':>4} {'observed':>9} {'95% CI':>19}  verdict")
    for c in cells:
        print(f"{c['model']:<28} {c['repeat']:>6} {c['target_p']:>8.3%} {c['n']:>6} {c['A']:>4} "
              f"{c['observed_p']:>9.3%} {c['ci_lo']:>8.3%}-{c['ci_hi']:<8.3%}  {c['verdict']}"
              + (f"  ({c['violations']} violations)" if c["violations"] else "")
              + (f"  ({c['unparseable']} unparseable)" if c["unparseable"] else ""))
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(cells)


def plot(cells: List[Dict[str, Any]], out_path: Path) -> None:
    models = [m for m in MODELS if any(c["model"] == m for c in cells)]
    repeats = sorted({c["repeat"] for c in cells})
    targets = sorted({c["target_p"] for c in cells})
    # Small multiplicative x-offsets so series at the same target don't overlap.
    offsets = {r: 1.2 ** (i - (len(repeats) - 1) / 2) for i, r in enumerate(repeats)}
    linthresh = min(targets) / 10   # log above this, linear below, so 0 has a place on the axis
    lim = (linthresh, max(targets) * 3)
    decades = [10.0 ** k for k in range(round(math.log10(lim[0])), round(math.log10(lim[1])) + 1)]

    fig, axes = plt.subplots(1, len(models), figsize=(4.5 * len(models), 4.5), squeeze=False)
    for ax, model in zip(axes[0], models):
        ax.set_xscale("log")
        ax.set_yscale("symlog", linthresh=linthresh, linscale=0.5)
        ax.plot(lim, lim, "--", color="gray", linewidth=1, label="perfect calibration")
        for i, r in enumerate(repeats):
            series = [c for c in cells if c["model"] == model and c["repeat"] == r]
            if not series:
                continue
            ys = [c["observed_p"] for c in series]
            ax.errorbar([c["target_p"] * offsets[r] for c in series], ys,
                        # max(0, ...): at 0 A's the Wilson bound is 0 up to float error
                        yerr=[[max(0.0, y - c["ci_lo"]) for y, c in zip(ys, series)],
                              [max(0.0, c["ci_hi"] - y) for y, c in zip(ys, series)]],
                        color=COLORS[i % len(COLORS)], marker="o", linestyle="-", capsize=3,
                        label="no repeat" if r == 1 else f"{r} repeats")
        ax.set_xlim(*lim)
        ax.set_ylim(-0.4 * linthresh, lim[1])   # a little room below 0 so zero dots aren't clipped
        ax.set_yticks([0] + decades)
        ax.set_yticklabels(["0"] + [f"$10^{{{round(math.log10(d))}}}$" for d in decades])
        ax.set_title(NAMES.get(model, model))
        ax.set_xlabel("Target rate of A")
        ax.legend(fontsize=8)
    axes[0][0].set_ylabel("Observed rate of A")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    ap.add_argument("--out", type=Path, default=config.FIGURES_DIR / "low_probs" / "calibration.png")
    args = ap.parse_args()
    cells = load_cells(args.runs_dir)
    if not cells:
        raise SystemExit(f"no cells with data under {args.runs_dir}")
    print_and_save_table(cells, args.out.with_suffix(".csv"))
    plot(cells, args.out)


if __name__ == "__main__":
    main()
