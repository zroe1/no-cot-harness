"""Second Look house style for figures, copied from uchicago-xlab/superposition-replication
(img_classifiers/reporting/figures.py), with matplotlib's blue in place of their maroon: no top/right spines, light y-grid,
frameless legends, transparent background. Importing this module sets
the style process-wide.
"""
import matplotlib
matplotlib.use("Agg")
from cycler import cycler  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

STYLE = {
    "figure.facecolor": "none",
    "axes.facecolor": "none",
    "savefig.facecolor": "none",
    "savefig.transparent": True,
    "axes.prop_cycle": cycler(color=["#1f77b4", "#d45d00", "#1a1a1a", "#9a0000",
                                     "#666666", "#c9a227", "#800000"]),
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#888888",
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.color": "#e0e0e0",
    "axes.axisbelow": True,
    "legend.frameon": False,
    # A step up from the reference repo's sizes, so the figures read at blog width.
    "axes.titlesize": 14,   # titles stay matplotlib-default: centered, normal weight
    "axes.labelsize": 13,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 10.5,
    "lines.linewidth": 2,
    "lines.markersize": 8,
    "errorbar.capsize": 4,
}
plt.rcParams.update(STYLE)
PALETTE = STYLE["axes.prop_cycle"].by_key()["color"]
BLUE, ORANGE, BLACK, GRAY = PALETTE[0], PALETTE[1], PALETTE[2], PALETTE[4]
GREEN = "#2ca02c"   # matplotlib's green, for a third series
REFERENCE = "#888888"   # the y = x line and other reference marks
