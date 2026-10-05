# -*- coding: utf-8 -*-
"""Fig. 1: method-framework overview (final manuscript version; schematic, no data dependencies, shipped for one-click reproduction).

v2 (2026-10-05 review): all text at 7.5 pt; the small "weights" label removed;
labels trimmed so they no longer bleed across boxes; "coverage restored" recast as "(proposed; gap remains)" to match the main text.
v3 (2026-10-05): canvas fills the full 75 mm (axes at 98%), fixing long labels exceeding their boxes;
added an automatic "text must stay inside its box" check - any line overflowing its box exits with an error (to prevent regressions).
Output figures_cx2/framework.{png,pdf} (600 dpi + vector), then copied to figures/.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({
    "font.family": "Arial", "font.size": 7.5, "axes.linewidth": 0.6,
    "mathtext.fontset": "custom", "mathtext.rm": "Arial",
    "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold",
})
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(OUT, exist_ok=True)
W = 2.953  # 75 mm

BLUE = "#3d7ab5"; BLUE_L = "#dce8f4"
ORANGE = "#e8a33d"; ORANGE_L = "#fbeed6"
GRAY = "#7f7f7f"; GRAY_L = "#ececec"

fig = plt.figure(figsize=(W, 3.45))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 100); ax.set_ylim(0, 118); ax.axis("off")
_BOXES = []


def box(x, y, w, h, text, fc, ec, bold=False):
    patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.8,rounding_size=2.0",
                           linewidth=0.7, edgecolor=ec, facecolor=fc, zorder=2)
    ax.add_patch(patch)
    t = ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", zorder=3,
                fontsize=7.5, fontweight="bold" if bold else "normal", linespacing=1.45)
    _BOXES.append((t, patch))


def arrow(x0, y0, x1, y1, color="#404040"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=6,
                                 linewidth=0.8, color=color, zorder=1,
                                 shrinkA=0, shrinkB=0))


# two-domain data
box(2, 104, 44, 11, "Source domain: LFP\nMIT-Stanford (124 cells)", BLUE_L, BLUE)
box(54, 104, 44, 11, "Target domain: LCO\nCALCE (16) / NASA (4)", ORANGE_L, ORANGE)
arrow(24, 104, 24, 98.8); arrow(76, 104, 76, 98.8)

# standardization
box(2, 87, 44, 11, "Standardization\n(W = 20, H = 10)", "#f5f5f5", GRAY)
box(54, 87, 44, 11, "Standardization\n(W = 20, H = 10)", "#f5f5f5", GRAY)
arrow(24, 87, 24, 81.8); arrow(76, 87, 76, 81.8)

# training
box(2, 70, 44, 11, "Source pre-training\n(TCN, 5-fold CV)", BLUE_L, BLUE)
box(54, 70, 44, 11, "Few-shot fine-tuning\n(2\u20138 target cells)", ORANGE_L, ORANGE)
arrow(46, 75.5, 54, 75.5)   # source weights -> fine-tuning (no small text label anymore)

# deployed model
arrow(24, 70, 24, 64.8); arrow(76, 70, 76, 64.8)
box(14, 53, 72, 11, "Deployed model", "#f5f5f5", GRAY, bold=True)

# dual-route calibration
arrow(50, 53, 25, 47.8); arrow(50, 53, 75, 47.8)
box(1, 34, 47, 13.5, "Source-domain route\n$q_{src}$: source residuals\n(naive transfer; fails)", BLUE_L, BLUE)
box(52, 34, 47, 13.5, "Target-domain route\n$q_{tgt}$: 1\u20132 target cells\n(proposed; gap remains)", ORANGE_L, ORANGE)
arrow(24, 34, 24, 28.4); arrow(76, 34, 76, 28.4)

# evaluation
box(20, 17, 60, 11, "Interval evaluation\nPICP / MPIW + per-cell", GRAY_L, GRAY)

# v3 self-check: no line of text may exceed its box (display-coordinate comparison)
fig.canvas.draw()
_rend = fig.canvas.get_renderer()
_bad = []
for _t, _p in _BOXES:
    _tb = _t.get_window_extent(_rend)
    _pb = _p.get_path().get_extents(_p.get_transform())
    if (_tb.x0 < _pb.x0 - 0.5 or _tb.x1 > _pb.x1 + 0.5
            or _tb.y0 < _pb.y0 - 0.5 or _tb.y1 > _pb.y1 + 0.5):
        _bad.append((_t.get_text().replace("\n", " / "),
                     round(_tb.width, 1), round(_pb.width, 1)))
print("box-text self-check: %d groups, %d overflow" % (len(_BOXES), len(_bad)))
for _b in _bad:
    print("  overflow: %s (text %.1f px / box %.1f px)" % _b)
if _bad:
    raise SystemExit("framework text overflows its box; adjust labels or geometry")

fig.savefig(os.path.join(OUT, "framework.png"), dpi=600, bbox_inches="tight", pad_inches=0)
fig.savefig(os.path.join(OUT, "framework.pdf"), bbox_inches="tight", pad_inches=0)
print("saved: framework v3 ->", OUT)
