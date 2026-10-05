# -*- coding: utf-8 -*-
"""论文全部 5 张结果图的复现脚本（与投稿版图件同源，图内文字英文；图 1 框架图见 make_fig_framework.py）。
数据源：仓库内 results_cx2/（唯一权威副本）；输出：本目录 figures/。
体例与 make_figures.py 一致（75mm、Arial、英文图内文字、600dpi PNG + 矢量 PDF；
期刊绘图体例第 2 条：图题中英对照，图内坐标轴标题/图例/图注采用英文）。

2026-10 P2 修订：
  P2-2 误差棒改"仅向上"——对称展开的下须会低于轴下限被裁掉，读者读不出下端；
  P2-3 "Nominal 0.90" 标注从压曲线/柱子的位置移到左上空白区；
  P2-4 图 3/图 5 的红绿配色改为色盲友好（蓝/橙/灰，与图 2/4/6 一致）。
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    # 2026-10-03：按《绘图体例》第 5 条“Arial 六号字（7.5 pt）”上调字号
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
    """对数尺度的 +1 标准差（仅向上）。对称展开的下须会低于轴下限而被裁掉
    （2026-10 P2-2 修订），故下须恒为 0。"""
    lg = [np.log10(v) for v in vals]
    s = float(np.std(lg, ddof=1))
    lm = np.log10(mean_ref)
    return mean_ref, 0.0, 10 ** (lm + s) - mean_ref


# ---------- 图 2: LOBO 逐电芯 RMSE 分布（results_cx2 复算版，与旧图同口径：3 种子合并 357 折） ----------
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

# ---------- 图 3: 跨化学体系迁移 ----------
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
# 2026-10-03：图宽按体例收窄至 75 mm 后 4 组横排标签会相互重叠，改为两行显示
glabels = ["CALCE\nTCN", "CALCE\nLSTM", "NASA\nTCN", "NASA\nLSTM"]
series = [("zero", "Zero-shot", "#3d7ab5"), ("ft", "Fine-tune", "#e8a33d"), ("to", "Target-only", "#7f7f7f")]
x = np.arange(len(groups)); w = 0.26
fig, ax = plt.subplots(figsize=(W - 0.08, 2.1))  # 7.5 pt 下紧贴裁剪宽度略超 75 mm，收窄 0.08 in
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

# ---------- 图 5: 保形覆盖 ----------
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
    # P2-3：标注移至左上空白区（原先压 LSTM 组误差棒）
    # 2026-10-04 晚：上限 1.12→1.25——1.12 时 (b) NASA 的 TCN 逐种子圆点
    # （0.85–1.00）仍落入标注文字框（几何检测 3 处重叠）；1.25 留出净空
    ax.text(0.04, 0.97, "Nominal 0.90", transform=ax.transAxes, fontsize=7.5, ha="left", va="top")
    ax.set_xticks(x); ax.set_xticklabels(["TCN", "LSTM"])
    ax.set_ylim(0, 1.25)
    # 2026-10-03：《绘图体例》第 3 条——分图用（a）（b）区分，分图题置于各分图下方
    ax.text(0.5, -0.235, "(%s) Target: %s" % ("ab"[_i_ax], domain),
            transform=ax.transAxes, ha="center", va="top", fontsize=7.5)
axes[0].set_ylabel("PICP (empirical coverage)")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.06))
fig.tight_layout(rect=(0, 0.07, 1, 0.94))
save(fig, "conformal_coverage")

# ---------- 图 6: 特征消融 ----------
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

# ---------- 图 4: 覆盖率-校准电芯数（新增） ----------
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
# 2026-10-04：纵轴改按新数据自适应（误差棒最低 0.6366、最高 0.9685，取 0.62–1.00
# 保证全部误差棒完整；旧固定 0.5–1.05 下部大片空白、趋势不可辨）
ax.text(1.05, 0.978, "Nominal 0.90", fontsize=7.5, ha="left", va="top")
ax.set_xlabel("Number of calibration cells")
ax.set_ylabel("PICP (target-calibrated)")
ax.set_xticks([1, 2, 3, 4, 5])
ax.set_ylim(0.62, 1.00)
ax.legend(loc="lower right")
fig.tight_layout()
save(fig, "calibration_sweep")

print("全部完成 ->", OUT)
