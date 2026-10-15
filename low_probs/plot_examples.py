"""Figure 1: calibration across models, plus real examples of Astra's threshold rule.

Left panel: observed vs target rate for GPT-6 Astra (3 and 5 repeats, which give identical
results), GPT-6.1 Sol (3 and 5 repeats, overlaid: same answers as Astra), GPT-5.6 Sol (3 and 5
repeats, lighter colors), and "all others": every other sweep model at 10% and 1% (not every
model was run at 0.1%), all of which got 0 A's (the script checks this and stops if it stops being true). Right panel: actual UUIDs from one cell
(default: 3 repeats, 1% target), sorted by value, with the model's real answer for each. Each row
shows the UUID's first 8 hex digits (4 bytes) and that number / 2^32 as a
probability (the same reading of the rule as plot_threshold.py). It shows the UUIDs closest to the cutoff on both sides, so the rule
"A iff the UUID is below p" can be read directly off the data.

    python -m low_probs.plot_examples
    python -m low_probs.plot_examples --p 0.001 --repeat 5
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

from harness import config
from low_probs.experiment import ASTRA, MODELS, RUNS_DIR, SOL, SOL61, cell_path, load_rows, short_name
from low_probs.plot import (MODELS as PLOT_MODELS, NAMES, calibration_axes, draw_series, load_cells,
                            repeats_label, same_results_groups, widest)
from low_probs.plot_threshold import HEX_DIGITS, uuid_value
from low_probs.style import BLUE, GRAY, GREEN, ORANGE

A_COLOR, B_COLOR = BLUE, GRAY
OTHERS_TARGETS = (0.01, 0.1)   # where "all others" is drawn: every sweep model was run here
PURPLE = "#7b3fa0"
SCALE = 1.15        # text in this figure, relative to the house style (style.py)
MARK_SCALE = 1.4    # lines, dots and error bars, scaled up more than the text
LIGHTEN = 0.45   # how far toward white GPT-5.6 Sol and "all others" are drawn (still opaque)


def lighten(color: str, amount: float = LIGHTEN) -> str:
    """`color` mixed with white: 0 leaves it as is, 1 is white. Opaque, unlike alpha, so lines
    underneath don't show through."""
    rgb = mcolors.to_rgb(color)
    return mcolors.to_hex([c + (1 - c) * amount for c in rgb])



def overview_panel(ax, cells) -> None:
    """Astra, GPT-6.1 Sol and GPT-5.6 Sol at 3 and 5 repeats, and every other model as one line
    at 0. GPT-6.1 Sol gives the same answers as Astra at 3 and 5 repeats, so it is drawn on top of
    Astra as hollow circles on a dashed line; GPT-5.6 Sol and "all others" are in lighter
    (opaque) colors, drawn underneath both."""
    calibration_axes(ax, sorted({c["target_p"] for c in cells if c["model"] == ASTRA}))
    for rs in same_results_groups(cells, ASTRA, [3, 5]):
        draw_series(ax, widest(cells, ASTRA, rs), BLUE, f"{NAMES[ASTRA]}, {repeats_label(rs)}", zorder=3)
    for rs in same_results_groups(cells, SOL61, [3, 5]):
        s = widest(cells, SOL61, rs)
        ax.plot([c["target_p"] for c in s], [c["observed_p"] for c in s], color=PURPLE, linestyle="--",
                marker="o", markerfacecolor="none", markersize=12 * MARK_SCALE, markeredgewidth=1.8 * MARK_SCALE, zorder=4,
                label=f"{NAMES[SOL61]}, {repeats_label(rs)}")
    for rs, color in zip(same_results_groups(cells, SOL, [3, 5]), (lighten(ORANGE), lighten(GREEN))):
        draw_series(ax, widest(cells, SOL, rs), color, f"{NAMES[SOL]}, {repeats_label(rs)}",
                    zorder=2)
    others = [c for c in cells if c["model"] not in (ASTRA, SOL, SOL61) and c["target_p"] in OTHERS_TARGETS]
    if any(c["A"] for c in others):
        raise SystemExit("some other model answered A at 10% or 1%, so 'all others' is no longer 0")
    # Each cell is 0/n; the bar is the widest single-cell Wilson interval at that target (not a
    # pooled one, which would be narrower than any one model/repeat condition supports).
    ax.errorbar(OTHERS_TARGETS, [0] * len(OTHERS_TARGETS),
                yerr=[[0] * len(OTHERS_TARGETS),
                      [max(c["ci_hi"] for c in others if c["target_p"] == p) for p in OTHERS_TARGETS]],
                color=lighten(GRAY), marker="o", linestyle="-", label="all others", zorder=1)
    handles, labels = ax.get_legend_handles_labels()
    # Astra, GPT-6.1 Sol, GPT-5.6 Sol, then the reference line and "all others" (matplotlib lists
    # plain lines before error-bar series, which would put GPT-6.1 Sol first).
    rank = [NAMES[ASTRA], NAMES[SOL61], NAMES[SOL], "perfect calibration", "all others"]
    order = sorted(range(len(labels)), key=lambda i: next(j for j, k in enumerate(rank) if labels[i].startswith(k)))
    ax.legend([handles[i] for i in order], [labels[i] for i in order], loc="upper left")


def target_label(p: float) -> str:
    return f"{p:.1%}".replace(".0%", "%")


def examples_panel(ax, rows, p: float, repeat: int, k: int) -> None:
    rows = sorted((r for r in rows if not r["violation"] and r["choice"] in ("A", "B")),
                  key=lambda r: uuid_value(r["seed"]))
    below = [r for r in rows if uuid_value(r["seed"]) < p]
    above = [r for r in rows if uuid_value(r["seed"]) >= p]
    shown_below, shown_above = below[-k:], above[:k]   # the k closest to the cutoff on each side
    d = HEX_DIGITS

    ax.axis("off")
    ax.set_xlim(0, 1)
    y, step = 0.92, 0.062
    cols = (0.02, 0.42, 0.82)   # UUID prefix, probability, answer
    for x, h in zip(cols, (f"UUID (first {d} hex)", "as probability", "answer")):
        ax.text(x, y, h, fontsize=11 * SCALE, va="center")
    y -= step * 0.75
    ax.text(cols[1], y, f"(first {d} hex ÷ $2^{{{4 * d}}}$)", fontsize=9 * SCALE, color="gray", va="center")
    y -= step

    def row(r):
        nonlocal y
        prefix = r["seed"][:d]
        mono = {"family": "monospace", "fontsize": 10.5 * SCALE, "va": "center"}
        ax.text(cols[0], y, prefix + "…", **mono)
        ax.text(cols[1], y, f"{100 * int(prefix, 16) / 16 ** d:.3f}%", **mono)
        ax.text(cols[2] + 0.04, y, r["choice"], family="monospace", fontsize=12 * SCALE, fontweight="bold",
                color=A_COLOR if r["choice"] == "A" else B_COLOR, va="center")
        y -= step

    if len(below) > k:
        ax.text(cols[0], y, f"… {len(below) - k} more below", fontsize=9.5 * SCALE, color="gray", va="center")
        y -= step
    for r in shown_below:
        row(r)
    pad = step * 0.35   # extra space above and below the cutoff line
    line_y = y + step / 2 - pad
    y -= 2 * pad
    ax.axhline(line_y, xmin=0.01, xmax=0.99, color="black", linestyle="--", linewidth=1 * SCALE)
    ax.text(0.43, line_y, f" target {target_label(p)} ", fontsize=9.5 * SCALE, ha="center", va="center",
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 1})
    for r in shown_above:
        row(r)
    ax.text(cols[0], y, f"… {len(above) - k} more above", fontsize=9.5 * SCALE, color="gray", va="center")

    label = "no repeat" if repeat == 1 else f"{repeat} repeats"
    ax.set_title(f"Real Astra answers ({label}, target {target_label(p)})")
    ax.set_ylim(y - step * 0.4, 1)   # trim the empty space under the last line


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=MODELS, default=ASTRA)
    ap.add_argument("--p", type=float, default=0.01, help="target of the cell to draw examples from")
    ap.add_argument("--repeat", type=int, default=3, help="repeat count of that cell")
    ap.add_argument("--k", type=int, default=4, help="examples shown on each side of the cutoff")
    ap.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    ap.add_argument("--out", type=Path, default=None,
                    help="default: results/figures/low_probs/examples_<model>.png")
    args = ap.parse_args()

    cells = load_cells(args.runs_dir, PLOT_MODELS + (SOL61,))
    rows = list(load_rows(cell_path(args.model, args.p, args.repeat, args.runs_dir)).values())
    if not cells or not rows:
        raise SystemExit(f"missing data under {args.runs_dir}")

    plt.rcParams.update({k: plt.rcParams[k] * SCALE for k in (
        "axes.titlesize", "axes.labelsize", "xtick.labelsize", "ytick.labelsize", "legend.fontsize")})
    plt.rcParams.update({k: plt.rcParams[k] * MARK_SCALE for k in (
        "lines.linewidth", "lines.markersize", "errorbar.capsize", "lines.markeredgewidth")})
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 5.2), gridspec_kw={"width_ratios": [1, 1.1]})
    overview_panel(left, cells)
    left.set_title("No CoT Calibration")
    examples_panel(right, rows, args.p, args.repeat, args.k)
    fig.tight_layout()
    out = args.out or config.FIGURES_DIR / "low_probs" / f"examples_{short_name(args.model)}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
