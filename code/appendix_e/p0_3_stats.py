# -*- coding: utf-8 -*-
"""P0-3: statistics upgrade - Wilcoxon signed-rank + Holm correction + effect size (paired Cohen's dz).

Data (version isolation): the paired section of results_cx2/transfer/persistence_baseline.json
(per-seed pooled RMSE of the fine-tuned/zero-shot models vs persistence, seeds 42-46).
Comparison family (the 6 paired configurations of Section 4.3 plus the 8 zero-shot ones; the text focuses on the 6 ft):
  ft = fine-tuned model vs persistence; zero = zero-shot vs persistence.
Test: two-sided Wilcoxon signed-rank (scipy, exact mode; with n=5 the smallest possible p is 0.0625,
so at n=5 Wilcoxon cannot reach 0.05 in principle - the honest conclusion of this small-sample upgrade;
significance must be stated with the effect size dz). Holm-Bonferroni within the ft (n=6) and zero (n=8) families separately.
Output: results/p0_3_statistics.csv + console summary.
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
    """Holm-Bonferroni: returns (adjusted_p, reject@0.05)."""
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
    diff = model_r - pers_r            # >0 = model worse than persistence
    seeds = entry["seeds"]
    try:
        stat, pw = wilcoxon(model_r, pers_r, alternative="two-sided",
                            method="exact", zero_method="wilcox")
    except ValueError:                 # degenerate case of all-zero differences
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

print(f"wrote {out_csv}\n")
print("=== the 6 ft pairs of the main text (Wilcoxon exact + Holm + dz) ===")
for r in sorted(ft_rows, key=lambda r: r["comparison"]):
    print(f"  {r['comparison']:28s} dz={r['cohens_dz']:+.2f}  p_wil={r['wilcoxon_p']:.4f}  "
          f"p_holm={r['holm_p']:.4f}  rej={r['holm_reject_0.05']}  "
          f"(old t={r['t_old']}, p_old={r['p_old']})")
print("\n=== n=5 note ===")
print("With n=5, the two-sided Wilcoxon signed-rank has a smallest possible p of 0.0625,")
print("so no per-seed pairing can reject H0 at the 0.05 level; significance should be stated as")
print("effect size + a p-value bound, not as 'p<0.05'.")
