# -*- coding: utf-8 -*-
"""Reproduction script for all 5 result figures of the manuscript (same source as the submission figures; English in-figure text; Fig. 1 framework in make_fig_framework.py).
Data: results_cx2/ inside the repository (the single authoritative copy); output: figures/ in this directory.
Style as in the submission pipeline (75mm, Arial, English in-figure text, 600 dpi PNG + vector PDF;
journal figure convention: bilingual captions in the manuscript, English axis titles/legends/annotations inside the figure).

2026-10 P2 revision:
  P2-2 error bars switched to upward-only - symmetric lower whiskers fall below the axis limit and get clipped, so readers cannot see them;
  P2-3 the "Nominal 0.90" annotation moved from on top of curves/bars to the upper-left blank area;
  P2-4 Fig. 3/Fig. 5 recoloured red/green to colour-blind friendly blue/orange/grey, matching Fig. 2/4/6.
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    # 2026-10-03: font raised to Arial 7.5 pt per the figure conventions (item 5)
    "font.family": "Arial", "font.size": 7.5, "axes.linewidth": 0.6,
    "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True,
    "ytick.right": True, "axes.grid": False, "legend.frameon": False,
    "legend.fontsize": 7.5, "axes.labelsize": 7.5, "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
})

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
R = os.path.join(ROOT, "results_cx2")
OUT = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)
W = 2.953


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=600, bbox_inches="tight")
    fig.savefig(os.path.join(OUT, name + ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print("saved:", name)


def log_yerr(vals, mean_ref):
    """+1 standard deviation on the log scale (upward only). A symmetric lower whisker would fall
    below the axis limit and get clipped (2026-10 P2-2 revision), so the lower whisker is always 0."""
    lg = [np.log10(v) for v in vals]
    s = float(np.std(lg, ddof=1))
    lm = np.log10(mean_ref)
    return mean_ref, 0.0, 10 ** (lm + s) - mean_ref


# Fig. 2: LOBO per-cell RMSE distribution (results_cx2 recomputation; same scope as the old figure: 3 seeds pooled, 357 folds)
import statistics

rmses = []
for s in (42, 43, 44):
    with open(os.path.join(R, "baselines", f"lobo_tcn_s{s}.jsonl"), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rmses.append(json.loads(line)["rmse"])
assert len(rmses) == 357, len(rmses)
med = statistics.median(rmses)

fig, ax = plt.subplots(figsize=(W, 2.1))
ax.hist(rmses, bins=30, color="#3d7ab5", edgecolor="white", linewidth=0.3)
ax.axvline(med, color="#c0392b", linestyle="--", linewidth=0.8)
ax.text(med + max(rmses) * 0.015, ax.get_ylim()[1] * 0.92, f"Median {med:.1f}",
        color="#c0392b", fontsize=7.5)
ax.set_xlabel("Per-cell RMSE / cycles")
ax.set_ylabel("Count")
ax.set_xlim(0, max(rmses) * 1.02)
fig.tight_layout()
save(fig, "lobo_distribution")

# Fig. 3: cross-chemistry transfer
d = json.load(open(os.path.join(R, "transfer", "transfer_multiseed_v2.json"), encoding="utf-8"))
T3B = os.path.join(R, "transfer")
KMAP = {"zero": "zero_shot", "ft": "fine_tune", "to": "target_only"}


def per_seed(dom, model, key):
    out = []
    for s in (42, 43, 44, 45, 46):
        j = json.load(open(os.path.join(T3B, f"t3b_{model}_s{s}.json"), encoding="utf-8"))
        out.append(j["targets"][dom][KMAP[key]]["rmse"])
    return out


groups = ["CALCE_tcn", "CALCE_lstm", "NASA_tcn", "NASA_lstm"]
# 2026-10-03: at the 75 mm width the four side-by-side labels overlapped, so the labels are shown on two lines
glabels = ["CALCE\nTCN", "CALCE\nLSTM", "NASA\nTCN", "NASA\nLSTM"]
series = [("zero", "Zero-shot", "#3d7ab5"), ("ft", "Fine-tune", "#e8a33d"), ("to", "Target-only", "#7f7f7f")]
x = np.arange(len(groups)); w = 0.26
fig, ax = plt.subplots(figsize=(W - 0.08, 2.1))  # at 7.5 pt the tight-crop width slightly exceeds 75 mm, so narrow by 0.08 in
for i, (key, lab, color) in enumerate(series):
    st = [log_yerr(per_seed(*g.split("_"), key), d[g][key]["mean"]) for g in groups]
    means = [s[0] for s in st]
    yerr = [[s[1] for s in st], [s[2] for s in st]]
    ax.bar(x + (i - 1) * w, means, w, yerr=yerr, capsize=1.5, color=color, label=lab,
           error_kw=dict(linewidth=0.6))
ax.set_yscale("log")
ax.set_ylabel("RMSE (SOH, log scale)")
ax.set_xticks(x); ax.set_xticklabels(glabels)
ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.14), columnspacing=0.8, handlelength=1.2)
fig.tight_layout()
save(fig, "transfer_comparison")

# Fig. 5: conformal coverage
d = json.load(open(os.path.join(R, "conformal", "conformal_multiseed_summary.json"), encoding="utf-8"))
fig, axes = plt.subplots(1, 2, figsize=(W, 1.9), sharey=True)
for _i_ax, (ax, domain) in enumerate(zip(axes, ["CALCE", "NASA"])):
    backs = ["tcn", "lstm"]
    x = np.arange(2); w = 0.34
    src_m = [d[f"{domain}_{b}"]["picp_src"]["mean"] for b in backs]
    src_s = [d[f"{domain}_{b}"]["picp_src"]["std"] for b in backs]
    tgt_m = [d[f"{domain}_{b}"]["picp_tgt"]["mean"] for b in backs]
    tgt_s = [d[f"{domain}_{b}"]["picp_tgt"]["std"] for b in backs]
    ax.bar(x - w / 2, src_m, w, yerr=src_s, capsize=1.5, color="#3d7ab5",
           label="Source-calibrated", error_kw=dict(linewidth=0.6))
    ax.bar(x + w / 2, tgt_m, w, yerr=tgt_s, capsize=1.5, color="#e8a33d",
           label="Target-calibrated", error_kw=dict(linewidth=0.6))
    rng = np.random.default_rng(0)
    for xi, vals in zip(x - w / 2, [d[f"{domain}_{b}"]["picp_src"].get("all", []) for b in backs]):
        if vals:
            ax.scatter(xi + rng.uniform(-0.06, 0.06, len(vals)), vals, s=3, color="#1f4e79",
                       zorder=3, linewidths=0)
    for xi, vals in zip(x + w / 2, [d[f"{domain}_{b}"]["picp_tgt"].get("all", []) for b in backs]):
        if vals:
            ax.scatter(xi + rng.uniform(-0.06, 0.06, len(vals)), vals, s=3, color="#a86a1a",
                       zorder=3, linewidths=0)
    ax.axhline(0.90, color="black", linestyle="--", linewidth=0.7)
    # P2-3: annotation moved to the upper-left blank area (it used to sit on the LSTM error bars)
    # 2026-10-04 evening: upper limit 1.12 -> 1.25 - at 1.12 the (b) NASA TCN per-seed dots
    # (0.85-1.00) still fell inside the annotation box (3 geometric overlaps detected); 1.25 clears them
    ax.text(0.04, 0.97, "Nominal 0.90", transform=ax.transAxes, fontsize=7.5, ha="left", va="top")
    ax.set_xticks(x); ax.set_xticklabels(["TCN", "LSTM"])
    ax.set_ylim(0, 1.25)
    # 2026-10-03: subfigures labelled (a)/(b) with subcaptions below each panel, per the figure conventions (item 3)
    ax.text(0.5, -0.235, "(%s) Target: %s" % ("ab"[_i_ax], domain),
            transform=ax.transAxes, ha="center", va="top", fontsize=7.5)
axes[0].set_ylabel("PICP (empirical coverage)")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.06))
fig.tight_layout(rect=(0, 0.07, 1, 0.94))
save(fig, "conformal_coverage")

# Fig. 6: feature ablation
d = json.load(open(os.path.join(R, "ablation", "ablation_v3c_multiseed.json"), encoding="utf-8"))
T3E = os.path.join(R, "ablation")
KMAP2 = {"zero": "zero_shot", "ft": "fine_tune"}


def per_seed_abl(suffix, dom, task):
    out = []
    for s in (42, 43, 44, 45, 46):
        j = json.load(open(os.path.join(T3E, f"t3e_{suffix}_tcn_s{s}.json"), encoding="utf-8"))
        out.append(j["targets"][dom][KMAP2[task]]["rmse"])
    return out


groups4 = [("CALCE", "zero", "CALCE\nzero-shot"), ("CALCE", "ft", "CALCE\nfine-tune"),
           ("NASA", "zero", "NASA\nzero-shot"), ("NASA", "ft", "NASA\nfine-tune")]
x = np.arange(len(groups4)); w = 0.34
fig, ax = plt.subplots(figsize=(W, 2.1))
for i, (suffix, lab, color) in enumerate([("base7", "Base-6", "#3d7ab5"), ("curve14", "Curve-13", "#e8a33d")]):
    st = [log_yerr(per_seed_abl(suffix, dom, task), d[f"{dom}_{suffix}"][task]["mean"]) for dom, task, _ in groups4]
    means = [s[0] for s in st]
    yerr = [[s[1] for s in st], [s[2] for s in st]]
    ax.bar(x + (i - 0.5) * w, means, w, yerr=yerr, capsize=1.5, color=color, label=lab,
           error_kw=dict(linewidth=0.6))
ax.set_yscale("log")
ax.set_ylabel("RMSE (SOH, log scale)")
ax.set_xticks(x); ax.set_xticklabels([g[2] for g in groups4], fontsize=7.5)
ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.13))
fig.tight_layout()
save(fig, "feature_ablation")

# Fig. 4: coverage vs calibration-cell count (added)
sw = json.load(open(os.path.join(R, "conformal", "t4_split_sweep", "sweep_summary.json"), encoding="utf-8"))
fig, ax = plt.subplots(figsize=(W, 2.0))
marks = {"tcn": ("o", "#3d7ab5", "TCN"), "lstm": ("s", "#e8a33d", "LSTM")}
for m in ("tcn", "lstm"):
    xs = sorted(int(k[3:]) for k in sw[m])
    ys = [sw[m][f"cal{nc}"]["picp_tgt"]["mean"] for nc in xs]
    es = [sw[m][f"cal{nc}"]["picp_tgt"]["std"] for nc in xs]
    mk, cl, lab = marks[m]
    ax.errorbar(xs, ys, yerr=es, marker=mk, markersize=3.5, color=cl, label=lab,
                linewidth=0.9, capsize=2, elinewidth=0.6)
ax.axhline(0.90, color="black", linestyle="--", linewidth=0.7)
# 2026-10-04: y-axis now adapts to the data (error bars span 0.6366-0.9685; 0.62-1.00 keeps every whisker intact,
# whereas the old fixed 0.5-1.05 left a large blank region below and made the trend unreadable)
ax.text(1.05, 0.978, "Nominal 0.90", fontsize=7.5, ha="left", va="top")
ax.set_xlabel("Number of calibration cells")
ax.set_ylabel("PICP (target-calibrated)")
ax.set_xticks([1, 2, 3, 4, 5])
ax.set_ylim(0.62, 1.00)
ax.legend(loc="lower right")
fig.tight_layout()
save(fig, "calibration_sweep")

print("all done ->", OUT)
