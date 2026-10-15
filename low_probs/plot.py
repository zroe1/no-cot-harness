"""Calibration figure and table for the main sweep (low_probs/run_sweep.sh).

Two figures, Astra, GPT-6.1 Sol and GPT-5.6 Sol (calibration_astra_sol.png) and the other models
(calibration_others.png). One panel per model: observed vs target rate of A, one line per problem-repeat count, 95% Wilson
intervals, and the y = x line for perfect calibration. The y-axis is symlog (log above the lowest
target / 10, linear below), so a cell with zero A's gets a dot at 0.

Gemini 3.1 Pro's cells come from its temperature-0 runs (runs/low_probs/ablations/, made by
run_ablations.sh), which are the ones reported. Its main-sweep runs were sent no temperature, so
they ran at the provider default; they gave the same answers on every UUID and are kept in
runs/low_probs/main/ but not plotted.

Prints the per-cell table and writes it as CSV next to the figure.

    python -m low_probs.plot
"""
from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt

from harness import config
from low_probs.ablations import ABLATIONS_DIR
from low_probs.experiment import ASTRA, GEMINI, RUNS_DIR, SOL, SOL61, SWEEP_MODELS as MODELS, load_rows, short_name, summarize
from low_probs.style import GRAY, BLUE, ORANGE, BLACK, REFERENCE

COLORS = [BLUE, ORANGE, GRAY, BLACK]
NAMES = {"openai/gpt-6-astra": "GPT-6 Astra", "openai/gpt-5.6-sol": "GPT-5.6 Sol", SOL61: "GPT-6.1 Sol",
         "google/gemini-3.1-pro-preview": "Gemini 3.1 Pro", "anthropic/claude-fable-5": "Fable 5",
         "anthropic/claude-fable-5.1": "Fable 5.1",
         "typesafe/jev-1.13": "Jev 1.13 (argmax)"}
# One figure per group: the two models that take low-probability actions, then the rest. All
# panels share the same axes (targets come from every cell), so the figures can be compared.
FIGURES = (("astra_sol", [ASTRA, SOL61, SOL]), ("others", [m for m in MODELS if m not in (ASTRA, SOL)]))
CELL_RE = re.compile(r"^r(?P<r>\d+)_p(?P<p>[0-9.e-]+)\.jsonl$")
# Models whose reported cells are ablation runs rather than main-sweep runs: (directory, file
# pattern). Gemini: the temperature-0 runs (see the docstring).
CELL_SOURCES = {GEMINI: (ABLATIONS_DIR, re.compile(r"^paper_hint_r(?P<r>\d+)_t0_p(?P<p>[0-9.e-]+)\.jsonl$"))}
COLUMNS = ["model", "repeat", "target_p", "n", "A", "observed_p", "ci_lo", "ci_hi", "verdict",
           "refused", "violations", "unparseable", "mean_model_p_a"]


def load_cells(runs_dir: Path, models: Tuple[str, ...] = MODELS) -> List[Dict[str, Any]]:
    cells = []
    for model in models:
        base, pattern = CELL_SOURCES.get(model, (runs_dir, CELL_RE))
        for path in sorted((base / short_name(model)).glob("*.jsonl")):
            m = pattern.match(path.name)
            rows = list(load_rows(path).values())
            if m and rows:
                cells.append({"model": model, "repeat": int(m["r"]), **summarize(rows, float(m["p"]))})
    return sorted(cells, key=lambda c: (models.index(c["model"]), c["repeat"], c["target_p"]))


def print_and_save_table(cells: List[Dict[str, Any]], csv_path: Path) -> None:
    print(f"{'model':<28} {'repeat':>6} {'target':>8} {'n':>6} {'A':>4} {'observed':>9} {'95% CI':>19}  verdict")
    for c in cells:
        if not c["n"]:   # every call refused, a violation, or unparseable: no rate to show
            print(f"{c['model']:<28} {c['repeat']:>6} {c['target_p']:>8.3%} {0:>6}  no usable answers "
                  f"({c['refused']} refused, {c['violations']} violations, {c['unparseable']} unparseable)")
            continue
        print(f"{c['model']:<28} {c['repeat']:>6} {c['target_p']:>8.3%} {c['n']:>6} {c['A']:>4} "
              f"{c['observed_p']:>9.3%} {c['ci_lo']:>8.3%}-{c['ci_hi']:<8.3%}  {c['verdict']}"
              + (f"  ({c['refused']} refused)" if c["refused"] else "")
              + (f"  ({c['violations']} violations)" if c["violations"] else "")
              + (f"  ({c['unparseable']} unparseable)" if c["unparseable"] else ""))
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(cells)


def calibration_axes(ax: Any, targets: List[float]) -> None:
    """Log x, symlog y (log above the lowest target / 10, linear below, so 0 has a place), the
    y = x line, ticks and axis labels."""
    linthresh = min(targets) / 10
    lim = (linthresh, max(targets) * 3)
    decades = [10.0 ** k for k in range(round(math.log10(lim[0])), round(math.log10(lim[1])) + 1)]
    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=linthresh, linscale=0.5)
    ax.plot(lim, lim, "--", color=REFERENCE, linewidth=1, label="perfect calibration")
    ax.set_xlim(*lim)
    ax.set_ylim(-0.4 * linthresh, lim[1])   # a little room below 0 so zero dots aren't clipped
    ax.set_yticks([0] + decades)
    ax.set_yticklabels(["0"] + [f"$10^{{{round(math.log10(d))}}}$" for d in decades])
    ax.minorticks_off()
    ax.set_xlabel("Target rate of A")
    ax.set_ylabel("Observed rate of A")


def series(cells: List[Dict[str, Any]], model: str, repeat: int) -> List[Dict[str, Any]]:
    return [c for c in cells if c["model"] == model and c["repeat"] == repeat and c["n"]]


def same_results_groups(cells: List[Dict[str, Any]], model: str, repeats: List[int]) -> List[List[int]]:
    """Repeat counts grouped by identical results (same observed rate at every target), so they
    can be drawn as one line labeled with all of them instead of one line hiding the others. The
    rate, not the count: refusals can leave cells with different n but the same 0 A's."""
    groups: Dict[tuple, List[int]] = {}
    for r in repeats:
        if series(cells, model, r):
            groups.setdefault(tuple((c["target_p"], c["observed_p"]) for c in series(cells, model, r)), []).append(r)
    return list(groups.values())


def repeats_label(rs: List[int]) -> str:
    labels = [repeat_label(r) for r in rs]
    if len(rs) == 1:
        return labels[0]
    if rs[0] == 1:   # "no repeat, 3 and 5 repeats"
        return labels[0] + ", " + " and ".join(map(str, rs[1:])) + " repeats"
    return " and ".join(map(str, rs)) + " repeats"


def widest(cells: List[Dict[str, Any]], model: str, rs: List[int]) -> List[Dict[str, Any]]:
    """Of a group with the same rates, the series with the widest intervals (smallest n), so a
    merged line never shows a tighter interval than one of its cells supports."""
    return min((series(cells, model, r) for r in rs), key=lambda ss: sum(c["n"] for c in ss))


def draw_series(ax: Any, cells: List[Dict[str, Any]], color: str, label: str, **kwargs: Any) -> None:
    """Observed rate at each target with Wilson 95% bars. Every point sits at its true target: no
    x-jitter, which would shift it relative to the diagonal."""
    ys = [c["observed_p"] for c in cells]
    ax.errorbar([c["target_p"] for c in cells], ys,
                # max(0, ...): at 0 A's the Wilson bound is 0 up to float error
                yerr=[[max(0.0, y - c["ci_lo"]) for y, c in zip(ys, cells)],
                      [max(0.0, c["ci_hi"] - y) for y, c in zip(ys, cells)]],
                color=color, marker="o", linestyle="-", label=label, **kwargs)


def calibration_panel(ax: Any, cells: List[Dict[str, Any]], model: str) -> None:
    """Observed vs target rate for one model: one line per repeat count, Wilson 95% bars."""
    repeats = sorted({c["repeat"] for c in cells})
    calibration_axes(ax, sorted({c["target_p"] for c in cells}))
    for rs in same_results_groups(cells, model, repeats):
        draw_series(ax, widest(cells, model, rs), COLORS[repeats.index(rs[0]) % len(COLORS)],
                    repeats_label(rs) + ("\n(same results)" if len(rs) > 1 else ""))
    ax.set_title(NAMES.get(model, model))
    ax.legend(fontsize=9, loc="upper left")


def repeat_label(r: int) -> str:
    return "no repeat" if r == 1 else f"{r} repeats"


def plot(cells: List[Dict[str, Any]], models: List[str], out_path: Path) -> None:
    models = [m for m in models if any(c["model"] == m for c in cells)]
    if not models:
        return
    fig, axes = plt.subplots(1, len(models), figsize=(4.5 * len(models), 4.5), squeeze=False)
    for i, (ax, model) in enumerate(zip(axes[0], models)):
        calibration_panel(ax, cells, model)
        if i:
            ax.set_ylabel("")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    ap.add_argument("--out", type=Path, default=config.FIGURES_DIR / "low_probs" / "calibration.png")
    args = ap.parse_args()
    # GPT-6.1 Sol isn't in the main sweep's model list (run_sweep.sh) but is plotted with Astra.
    cells = load_cells(args.runs_dir, MODELS + (SOL61,))
    if not cells:
        raise SystemExit(f"no cells with data under {args.runs_dir}")
    print_and_save_table(cells, args.out.with_suffix(".csv"))
    for suffix, models in FIGURES:
        plot(cells, models, args.out.with_name(f"{args.out.stem}_{suffix}{args.out.suffix}"))


if __name__ == "__main__":
    main()
