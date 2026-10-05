# -*- coding: utf-8 -*-
"""P0-3: 统计升级——Wilcoxon signed-rank + Holm 校正 + 效应量（配对 Cohen's dz）。

数据源（版本隔离）：results_cx2/transfer/persistence_baseline.json 的 paired 段
（微调/零样本模型 vs persistence 的逐种子池化 RMSE，seeds 42-46）。
对比族（论文 4.3 的 6 组配对 + 零样本 8 组，全算，正文重点 6 组 ft）：
  ft  = 微调模型 vs persistence；zero = 零样本 vs persistence。
检验：双侧 Wilcoxon signed-rank（scipy，exact 模式；n=5 时最小可能 p=0.0625，
故 n=5 下 Wilcoxon 原则上达不到 0.05——这是小样本统计升级的如实结论，
显著性表述需配合效应量 dz）。Holm-Bonferroni 在 ft 族（n=6）与 zero 族（n=8）内分别校正。
输出：results/p0_3_statistics.csv + 控制台摘要。
"""
import csv, io, json, os, sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import numpy as np
from scipy.stats import wilcoxon

OSS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "results_p0")
os.makedirs(OUT, exist_ok=True)

p = json.load(open(os.path.join(OSS, "results_cx2", "transfer", "persistence_baseline.json"),
                   encoding="utf-8"))
paired = p["paired"]


def cohens_dz(diff):
    d = np.asarray(diff, float)
    sd = np.std(d, ddof=1)
    return float(np.mean(d) / sd) if sd > 0 else float("inf")


def holm(pvals):
    """Holm-Bonferroni：返回 (adjust_p, reject@0.05)。"""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    run = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * pvals[idx]
        run = max(run, val)
        adj[idx] = min(1.0, run)
    return adj, adj < 0.05


rows = []
for key, entry in sorted(paired.items()):
    model_r = np.asarray(entry["model_rmse"], float)
    pers_r = np.asarray(entry["persistence_rmse"], float)
    diff = model_r - pers_r            # >0 = 模型比 persistence 差
    seeds = entry["seeds"]
    try:
        stat, pw = wilcoxon(model_r, pers_r, alternative="two-sided",
                            method="exact", zero_method="wilcox")
    except ValueError:                 # 全部差值为 0 的退化情形
        stat, pw = float("nan"), 1.0
    dz = cohens_dz(diff)
    rows.append(dict(comparison=key, n=len(seeds), seeds=";".join(map(str, seeds)),
                     model_mean=round(float(np.mean(model_r)), 4),
                     pers_mean=round(float(np.mean(pers_r)), 4),
                     diff_mean=round(float(np.mean(diff)), 5),
                     wins=int(entry.get("model_wins", int(np.sum(model_r < pers_r)))),
                     wilcoxon_W=round(float(stat), 3) if np.isfinite(stat) else "",
                     wilcoxon_p=round(float(pw), 4),
                     cohens_dz=round(dz, 3),
                     t_old=entry.get("t"), p_old=entry.get("p")))

ft_rows = [r for r in rows if "|ft|" in r["comparison"]]
zero_rows = [r for r in rows if "|zero|" in r["comparison"]]
for subset in (ft_rows, zero_rows):
    adj, rej = holm(np.array([r["wilcoxon_p"] for r in subset], float))
    for r, a, x in zip(subset, adj, rej):
        r["holm_p"] = round(float(a), 4)
        r["holm_reject_0.05"] = bool(x)

cols = ["comparison", "n", "seeds", "model_mean", "pers_mean", "diff_mean", "wins",
        "wilcoxon_W", "wilcoxon_p", "holm_p", "holm_reject_0.05", "cohens_dz", "t_old", "p_old"]
out_csv = os.path.join(OUT, "p0_3_statistics.csv")
with open(out_csv, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    for r in sorted(rows, key=lambda r: (r["comparison"].split("|")[0], r["comparison"])):
        w.writerow({c: r[c] for c in cols})

print(f"已写 {out_csv}\n")
print("=== 正文 6 组 ft 配对（Wilcoxon exact + Holm + dz） ===")
for r in sorted(ft_rows, key=lambda r: r["comparison"]):
    print(f"  {r['comparison']:28s} dz={r['cohens_dz']:+.2f}  p_wil={r['wilcoxon_p']:.4f}  "
          f"p_holm={r['holm_p']:.4f}  rej={r['holm_reject_0.05']}  "
          f"(旧 t={r['t_old']}, p_old={r['p_old']})")
print("\n=== n=5 说明 ===")
print("Wilcoxon signed-rank 在 n=5 时双侧最小可能 p=0.0625，")
print("因此任何逐种子配对在 0.05 水平都不可能拒绝 H0——显著性结论应表述为")
print("效应量 + p 值上界，而非 'p<0.05'。")
