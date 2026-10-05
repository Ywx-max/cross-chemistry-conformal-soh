# -*- coding: utf-8 -*-
"""Fig. 6: reliability diagram + coverage-width trade-off (data = p0_2_metrics.csv of the P0-2 recomputation batch).
(a) empirical PICP vs nominal 1-alpha (target route, 5-seed mean+/-std, 4 lines + diagonal);
(b) coverage-width trajectory (alpha 0.05 -> 0.20) + source-route points (alpha=0.10).
Style as in Figure_2-5: Arial 8pt, 90mm, blue/orange/grey. Output Figure_6.png/pdf/eps."""
import csv, io, os, sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "Arial", "font.size": 8, "axes.linewidth": 0.6,
    "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True,
    "ytick.right": True, "legend.frameon": False, "legend.fontsize": 8,
    "axes.labelsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
})

P0 = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "figures")
rows = list(csv.DictReader(open(os.path.join(P0, "results_p0", "p0_2_metrics.csv"), encoding="utf-8")))
for r in rows:
    for k in ("alpha", "PICP", "MPIW", "NMPIW"):
        r[k] = float(r[k])

ALPHAS = [0.05, 0.10, 0.20]
COLOR = {("tcn", "CALCE"): "#3d7ab5", ("tcn", "NASA"): "#e8a33d",
         ("lstm", "CALCE"): "#1f4e79", ("lstm", "NASA"): "#b3581a"}
STYLE = {("tcn", "CALCE"): ("o", "-"), ("tcn", "NASA"): ("s", "-"),
         ("lstm", "CALCE"): ("o", "--"), ("lstm", "NASA"): ("s", "--")}
LBL = {("tcn", "CALCE"): "TCN / CALCE", ("tcn", "NASA"): "TCN / NASA",
       ("lstm", "CALCE"): "LSTM / CALCE", ("lstm", "NASA"): "LSTM / NASA"}

fig, axes = plt.subplots(1, 2, figsize=(3.543 * 2, 2.05))

# (a) reliability diagram
ax = axes[0]
ax.plot([0.75, 1.0], [0.75, 1.0], color="#aaaaaa", linewidth=0.7, linestyle=":")
for key in COLOR:
    xs, ys, es = [], [], []
    for a in ALPHAS:
        sel = [r["PICP"] for r in rows if r["model"] == key[0] and r["domain"] == key[1]
               and r["route"] == "target" and abs(r["alpha"] - a) < 1e-9]
        xs.append(1 - a); ys.append(float(np.mean(sel)))
        es.append(float(np.std(sel, ddof=1)))
    mk, ls = STYLE[key]
    ax.errorbar(xs, ys, yerr=es, marker=mk, markersize=3, linestyle=ls, linewidth=0.9,
                color=COLOR[key], label=LBL[key], capsize=1.5, elinewidth=0.6)
ax.set_xlabel("Nominal coverage 1 - alpha")
ax.set_ylabel("Empirical PICP")
ax.set_xlim(0.76, 0.99); ax.set_ylim(0.70, 1.02)
# legend moved below the axes (shared, 4 columns): at 8pt inside the axes it intersected data lines/error bars (verified by programmatic sampling)
ax.text(0.03, 0.97, "(a)", transform=ax.transAxes, fontsize=8, fontweight="bold",
        ha="left", va="top")

# (b) coverage-width trade-off
ax = axes[1]
for key in COLOR:
    xs, ys, es = [], [], []
    for a in ALPHAS:
        sel = [r for r in rows if r["model"] == key[0] and r["domain"] == key[1]
               and r["route"] == "target" and abs(r["alpha"] - a) < 1e-9]
        xs.append(float(np.mean([r["MPIW"] for r in sel])))
        ys.append(float(np.mean([r["PICP"] for r in sel])))
        es.append(float(np.std([r["PICP"] for r in sel], ddof=1)))
    mk, ls = STYLE[key]
    ax.errorbar(xs, ys, yerr=es, marker=mk, markersize=3, linestyle=ls, linewidth=0.9,
                color=COLOR[key], capsize=1.5, elinewidth=0.6)
    ax.annotate("", xy=(xs[-1], ys[-1]), xytext=(xs[-2], ys[-2]),
                arrowprops=dict(arrowstyle="->", color=COLOR[key], lw=0.8))
    # source-route points (alpha=0.10)
    sel = [r for r in rows if r["model"] == key[0] and r["domain"] == key[1]
           and r["route"] == "source" and r["alpha"] == 0.10]
    ax.scatter(float(np.mean([r["MPIW"] for r in sel])), float(np.mean([r["PICP"] for r in sel])),
               marker="x", s=22, color=COLOR[key], zorder=5, linewidths=1.2)
ax.set_xlabel("MPIW (mean interval width)")
ax.set_ylabel("Empirical PICP")
ax.text(0.03, 0.97, "(b)", transform=ax.transAxes, fontsize=8, fontweight="bold",
        ha="left", va="top")
ax.text(0.97, 0.05, "arrows: alpha 0.05->0.20;  x: source route",
        transform=ax.transAxes, fontsize=8, ha="right", va="bottom", color="#404040")

fig.tight_layout()
# shared legend below the figure (the 4 curves of (a); 8pt)
_h, _l = axes[0].get_legend_handles_labels()
fig.legend(_h, _l, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.035),
           fontsize=8, frameon=False, columnspacing=1.8, handlelength=1.6)
fig.savefig(os.path.join(OUT, "Figure_6.png"), dpi=1100, bbox_inches="tight", pad_inches=0.01)
fig.savefig(os.path.join(OUT, "Figure_6.pdf"), bbox_inches="tight", pad_inches=0.01)
fig.savefig(os.path.join(OUT, "Figure_6.eps"), bbox_inches="tight", pad_inches=0.01)
print("saved: Figure_6 (png/pdf/eps)")
