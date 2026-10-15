"""Observed rate of A vs number of prompt repeats, one line per target rate, for one model.

Each target gets a dashed horizontal line at its target rate in the same color, so a point on
its dashed line is perfectly calibrated. 95% Wilson intervals. The y-axis is symlog (log above
the lowest target / 10, linear below), so cells with zero A's get a dot at 0.

    python -m low_probs.plot_repeats
    python -m low_probs.plot_repeats --model anthropic/claude-fable-5.1
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from harness import config  # noqa: E402
from low_probs.experiment import ASTRA, MODELS, RUNS_DIR, short_name  # noqa: E402
from low_probs.plot import COLORS, NAMES, load_cells  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=MODELS, default=ASTRA)
    ap.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    ap.add_argument("--out", type=Path, default=None,
                    help="default: results/figures/low_probs/repeats_<model>.png")
    args = ap.parse_args()

    cells = [c for c in load_cells(args.runs_dir) if c["model"] == args.model]
    if not cells:
        raise SystemExit(f"no {args.model} cells under {args.runs_dir}")
    targets = sorted({c["target_p"] for c in cells}, reverse=True)
    repeats = sorted({c["repeat"] for c in cells})
    linthresh = min(targets) / 10
    top = max(targets) * 3
    decades = [10.0 ** k for k in range(round(math.log10(linthresh)), round(math.log10(top)) + 1)]

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.set_yscale("symlog", linthresh=linthresh, linscale=0.5)
    for i, p in enumerate(targets):
        color = COLORS[i % len(COLORS)]
        series = sorted((c for c in cells if c["target_p"] == p), key=lambda c: c["repeat"])
        ys = [c["observed_p"] for c in series]
        ax.axhline(p, color=color, linestyle="--", linewidth=1)
        ax.errorbar([c["repeat"] for c in series], ys,
                    # max(0, ...): at 0 A's the Wilson bound is 0 up to float error
                    yerr=[[max(0.0, y - c["ci_lo"]) for y, c in zip(ys, series)],
                          [max(0.0, c["ci_hi"] - y) for y, c in zip(ys, series)]],
                    # later series drawn smaller on top, so overlapping points (e.g. two
                    # targets at 0) show as a dot inside a dot instead of hiding each other
                    markersize=8 - 2 * i, color=color, marker="o", linestyle="-", capsize=3, label=f"target {p:.1%}".replace(".0%", "%"))
    ax.set_xticks(repeats)
    ax.set_xticklabels([str(r) for r in repeats])
    ax.set_xlim(min(repeats) - 0.5, max(repeats) + 0.5)
    ax.set_ylim(-0.4 * linthresh, top)   # a little room below 0 so zero dots aren't clipped
    ax.set_yticks([0] + decades)
    ax.set_yticklabels(["0"] + [f"$10^{{{round(math.log10(d))}}}$" for d in decades])
    ax.set_xlabel("Number of prompt repeats")
    ax.set_ylabel("Observed rate of A")
    ax.set_title(f"{NAMES.get(args.model, args.model)} with no CoT: observed vs target rate of A",
                 fontsize=10)
    ax.legend(fontsize=8, title="dashed = target", title_fontsize=8, loc="lower right")
    fig.tight_layout()
    out = args.out or config.FIGURES_DIR / "low_probs" / f"repeats_{short_name(args.model)}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
