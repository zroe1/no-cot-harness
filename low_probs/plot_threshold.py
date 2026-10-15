"""Where the A answers land relative to the cutoff, UUID by UUID.

One panel per target. Each row is one model and repeat count; each dot is one UUID from that
cell whose value as a probability is below 3p, placed at that value. The value is the UUID's
first 8 hex digits (4 bytes, 32 bits) / 2^32, the version of the rule used in the post; reading
2 bytes or the whole UUID instead puts every one of these UUIDs on the same side of the cutoff.
The ideal rule's cutoff is the vertical line at p. Filled = answered A, hollow = answered B. A model that
follows the rule exactly has filled dots left of the line and hollow dots right of it. A's on
UUIDs beyond 3p (off the axis) are counted at the right end of the row.

    python -m low_probs.plot_threshold
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from harness import config
from low_probs.experiment import ASTRA, RUNS_DIR, SOL, cell_path, load_rows
from low_probs.plot import NAMES, repeat_label
from low_probs.style import BLUE, GRAY, ORANGE

ROWS = [(ASTRA, 1), (ASTRA, 3), (ASTRA, 5), (SOL, 1), (SOL, 3), (SOL, 5)]
COLOR = {ASTRA: BLUE, SOL: ORANGE}
TARGETS = (0.1, 0.01, 0.001)
XMAX = 3   # show UUIDs up to 3x the target
HEX_DIGITS = 8   # first 4 bytes


def uuid_value(seed: str) -> float:
    """The UUID's first HEX_DIGITS hex digits as a number in [0, 1)."""
    return int(seed.replace("-", "")[:HEX_DIGITS], 16) / 16 ** HEX_DIGITS


def panel(ax, p: float, runs_dir: Path) -> None:
    for i, (model, r) in enumerate(ROWS):
        y = len(ROWS) - 1 - i   # first row at the top
        rows = [row for row in load_rows(cell_path(model, p, r, runs_dir)).values()
                if not row["violation"] and row["choice"] in ("A", "B")]
        xs = [(uuid_value(row["seed"]), row["choice"]) for row in rows]
        a = [x for x, c in xs if c == "A" and x < XMAX * p]
        b = [x for x, c in xs if c == "B" and x < XMAX * p]
        ax.scatter(b, [y] * len(b), s=55, facecolors="none", edgecolors=GRAY, linewidths=1.2, zorder=2)
        ax.scatter(a, [y] * len(a), s=55, color=COLOR[model], zorder=3)
        far = sum(c == "A" for x, c in xs if x >= XMAX * p)
        if far:
            ax.text(XMAX * p * 1.01, y, f"+{far} A\nbeyond", fontsize=11, color=COLOR[model], va="center")
    ax.axvline(p, color="black", linestyle="--", linewidth=1, zorder=1)
    ax.set_xlim(0, XMAX * p)
    ax.xaxis.set_major_formatter(lambda x, _: f"{100 * x:g}%")
    ax.set_ylim(-0.6, len(ROWS) - 0.4)
    ax.set_yticks(range(len(ROWS)))
    ax.set_yticklabels([f"{NAMES[m].split()[-1]}, {repeat_label(r)}" for m, r in reversed(ROWS)])
    ax.grid(False)
    ax.tick_params(labelsize=14)
    ax.set_xlabel("first 8 hex digits ÷ $2^{32}$", fontsize=15)
    ax.set_title(f"target {p:.1%}".replace(".0%", "%"), fontsize=17)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    ap.add_argument("--out", type=Path, default=config.FIGURES_DIR / "low_probs" / "threshold.png")
    args = ap.parse_args()
    fig, axes = plt.subplots(1, len(TARGETS), figsize=(5.5 * len(TARGETS), 4.6), sharey=True)
    for ax, p in zip(axes, TARGETS):
        panel(ax, p, args.runs_dir)
    a = plt.Line2D([], [], marker="o", linestyle="", color=GRAY, label="answered A")
    b = plt.Line2D([], [], marker="o", linestyle="", markerfacecolor="none", color=GRAY, label="answered B")
    cut = plt.Line2D([], [], color="black", linestyle="--", linewidth=1, label="ideal rule's cutoff")
    fig.legend(handles=[a, b, cut], loc="lower center", ncol=3, fontsize=14, markerscale=1.3)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
