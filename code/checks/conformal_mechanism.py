# -*- coding: utf-8 -*-
"""Conformal-mechanism recomputation (2026-10-04 checklist A3):
recomputes four correlations from the per-seed results_cx2/conformal data to decide the fate of the Section 4.5 mechanism paragraph.

Each configuration point = (backbone, seed, target domain): 2 backbones x 5 seeds x 2 domains = 20 points.

  rho1 (primary): q_src/median|r|  vs PICP_src
  rho2:            q_src            vs PICP_src
  rho3:            q_src/RMSE       vs PICP_src
  rho4:            RMSE             vs PICP_src

where q_src = source-domain calibration quantile; median|r| = median of the absolute residuals of all
test cells in that target domain (pooled over per_cell[*].residuals); PICP_src = empirical source-calibrated coverage.

Decision rule (checklist A3-2): if rho1 stays high and positive (larger ratio, higher source-calibrated
coverage), keep the mechanism paragraph and rewrite it with the new values; otherwise drop it and recast
Section 4.5 as "both domains under-cover throughout, with severity differences driven mainly by residual scale and task difficulty".

    python code/checks/conformal_mechanism.py

Output: results_cx2/conformal/mechanism_correlations.json (also printed to stdout).
"""
import json, math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results_cx2" / "conformal"


def pearson(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return sxy / math.sqrt(sxx * syy)


def ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            r[order[k]] = avg
        i = j + 1
    return r


def spearman(xs, ys):
    return pearson(ranks(xs), ranks(ys))


def median_abs(vs):
    a = sorted(abs(v) for v in vs)
    n = len(a)
    return a[n // 2] if n % 2 else (a[n // 2 - 1] + a[n // 2]) / 2


def main():
    pts = []
    for model in ("tcn", "lstm"):
        for seed in (42, 43, 44, 45, 46):
            j = json.load(open(RES / f"t4_{model}_s{seed}.json", encoding="utf-8"))
            for dom in ("CALCE", "NASA"):
                t = j["targets"][dom]
                res = []
                for cell in t["per_cell"].values():
                    res.extend(cell["residuals"])
                pts.append({
                    "model": model, "seed": seed, "domain": dom,
                    "q_src": t["q_src"],
                    "picp_src": t["source_calibrated"]["PICP"],
                    "rmse": t["point_rmse"],
                    "med_abs_r": median_abs(res),
                })
    assert len(pts) == 20, len(pts)

    q = [p["q_src"] for p in pts]
    cov = [p["picp_src"] for p in pts]
    rm = [p["rmse"] for p in pts]
    mr = [p["med_abs_r"] for p in pts]
    ratio_med = [a / b for a, b in zip(q, mr)]
    ratio_rmse = [a / b for a, b in zip(q, rm)]

    keys = ["rho1_qsrc_over_medabsr_vs_picpsrc", "rho2_qsrc_vs_picpsrc",
            "rho3_qsrc_over_rmse_vs_picpsrc", "rho4_rmse_vs_picpsrc"]
    series = {"rho1": (ratio_med, cov), "rho2": (q, cov),
              "rho3": (ratio_rmse, cov), "rho4": (rm, cov)}
    out = {
        "n_points": len(pts),
        "note": "20 configuration points = 2 backbones x 5 seeds x 2 target domains; data: results_cx2/conformal/t4_*.json",
        "pointwise": pts,
        "pearson": {k: pearson(*series[k[:4]]) for k in keys},
        "spearman": {k: spearman(*series[k[:4]]) for k in keys},
    }
    dst = RES / "mechanism_correlations.json"
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print("saved:", dst)
    for name in ("pearson", "spearman"):
        for k in keys:
            print(f"{name:9s} {k:36s} {out[name][k]:+.4f}")


if __name__ == "__main__":
    main()
