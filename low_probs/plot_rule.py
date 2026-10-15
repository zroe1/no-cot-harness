"""Calibration plus agreement with the ideal threshold rule, for one model (default Astra).

The ideal rule reads the UUID as a number in [0, 1) and outputs A iff it is below p. Because
UUIDs are uniform, a model following it has a rate of exactly p, so this shows whether a model
is calibrated *because* it uses the UUID, separately from the sampling noise in the A count.

Left panel: observed vs target rate (same as plot.py). Right panel: for each target and repeat
count, the share of UUIDs below p that the model answered A, labeled with the counts. Overall
agreement isn't plotted because answering B every time already agrees on ~(1 - p) of UUIDs. It
is printed in the table, with the number of A's on UUIDs above p.

With several --models, the figure is just the agreement panel for each model, side by side.

    python -m low_probs.plot_rule
    python -m low_probs.plot_rule --model anthropic/claude-fable-5.1
    python -m low_probs.plot_rule --models openai/gpt-6-astra openai/gpt-5.6-sol
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt

from harness import config
from low_probs.experiment import ASTRA, MODELS, RUNS_DIR, load_rows, short_name
from low_probs.plot import COLORS, CELL_RE, NAMES, calibration_panel, load_cells, repeat_label


def uuid_value(seed: str) -> float:
    """The UUID's 128 bits as a number in [0, 1)."""
    return int(seed.replace("-", ""), 16) / 16 ** 32


def rule_cells(runs_dir: Path, model: str) -> List[Dict[str, Any]]:
    cells = []
    for path in sorted((runs_dir / short_name(model)).glob("r*_p*.jsonl")):
        m = CELL_RE.match(path.name)
        rows = [r for r in load_rows(path).values() if not r["violation"] and r["choice"] in ("A", "B")]
        if not (m and rows):
            continue
        p = float(m["p"])
        below = [r for r in rows if uuid_value(r["seed"]) < p]
        cells.append({
            "repeat": int(m["r"]), "target_p": p, "n": len(rows),
            "agreement": sum((r["choice"] == "A") == (uuid_value(r["seed"]) < p) for r in rows) / len(rows),
            "below": len(below), "A_below": sum(r["choice"] == "A" for r in below),
            "A_above": sum(r["choice"] == "A" and uuid_value(r["seed"]) >= p for r in rows),
        })
    return sorted(cells, key=lambda c: (c["repeat"], c["target_p"]))


def rule_panel(ax: Any, cells: List[Dict[str, Any]]) -> None:
    repeats = sorted({c["repeat"] for c in cells})
    targets = sorted({c["target_p"] for c in cells})   # same left-to-right order as the calibration panel
    width = 0.8 / len(repeats)
    for i, r in enumerate(repeats):
        for j, p in enumerate(targets):
            c = next((c for c in cells if c["repeat"] == r and c["target_p"] == p), None)
            if c is None or not c["below"]:
                continue
            x = j + (i - (len(repeats) - 1) / 2) * width
            ax.bar(x, 100 * c["A_below"] / c["below"], width, color=COLORS[i % len(COLORS)],
                   label=repeat_label(r) if j == 0 else None)
            ax.text(x, 100 * c["A_below"] / c["below"] + 1.5, f"{c['A_below']}/{c['below']}",
                    ha="center", va="bottom", fontsize=9)
    ax.set_xticks(range(len(targets)))
    ax.set_xticklabels([f"{p:g}" for p in targets])
    ax.set_ylim(0, 125)   # headroom for the count labels and a one-row legend
    ax.set_xlabel("Target rate of A")
    ax.set_ylabel("UUIDs below p answered A (%)")
    ax.set_title("Agreement with ideal rule (A iff UUID < p)")
    ax.legend(fontsize=9, loc="upper center", ncol=len(repeats))


def print_table(model: str, rcells: List[Dict[str, Any]]) -> None:
    print(model)
    print(f"{'repeat':>6} {'target':>7} {'n':>6} {'agreement':>10} {'A on UUIDs < p':>15} {'A on UUIDs >= p':>16}")
    for c in rcells:
        print(f"{c['repeat']:>6} {c['target_p']:>7g} {c['n']:>6} {c['agreement']:>10.4%} "
              f"{c['A_below']:>9}/{c['below']:<5} {c['A_above']:>16}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=MODELS, default=ASTRA)
    ap.add_argument("--models", choices=MODELS, nargs="+", default=None,
                    help="several models: one agreement panel each, no calibration panel")
    ap.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    ap.add_argument("--out", type=Path, default=None,
                    help="default: results/figures/low_probs/rule_<model>[_<model>...].png")
    args = ap.parse_args()
    models = args.models or [args.model]

    rcells = {m: rule_cells(args.runs_dir, m) for m in models}
    for m in models:
        if not rcells[m]:
            raise SystemExit(f"no {m} cells under {args.runs_dir}")
        print_table(m, rcells[m])

    if args.models:
        fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 4.8), sharey=True, squeeze=False)
        for i, (ax, m) in enumerate(zip(axes[0], models)):
            rule_panel(ax, rcells[m])
            ax.set_title(NAMES.get(m, m))
            if i:
                ax.set_ylabel("")
    else:
        fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4.5))
        calibration_panel(left, [c for c in load_cells(args.runs_dir) if c["model"] == args.model], args.model)
        rule_panel(right, rcells[args.model])
    fig.tight_layout()
    out = args.out or config.FIGURES_DIR / "low_probs" / f"rule_{'_'.join(short_name(m) for m in models)}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
