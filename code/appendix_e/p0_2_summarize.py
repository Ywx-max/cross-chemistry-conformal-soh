# -*- coding: utf-8 -*-
"""P0-2 汇总：α 扫描透视表 + 与论文表 5 原值的漂移对照（alpha=0.10, target 路由）。"""
import io, json, os, sys, csv

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import numpy as np

P0 = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OSS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

rows = list(csv.DictReader(open(os.path.join(P0, "results_p0", "p0_2_metrics.csv"), encoding="utf-8")))
for r in rows:
    r["alpha"] = float(r["alpha"]); r["PICP"] = float(r["PICP"]); r["MPIW"] = float(r["MPIW"])
    r["NMPIW"] = float(r["NMPIW"]); r["Winkler"] = float(r["Winkler"]); r["pinball"] = float(r["pinball"])

# ---- 1. 与表 5 原值对照（alpha=0.10, target 路由） ----
print("=== alpha=0.10 target 路由：重算 vs 论文表 5 原值（逐种子） ===")
drifts = []
for model in ("tcn", "lstm"):
    for domain in ("CALCE", "NASA"):
        j5 = [json.load(open(os.path.join(OSS, "results_cx2", "conformal",
                                          f"t4_{model}_s{s}.json"), encoding="utf-8")) for s in range(42, 47)]
        orig = [j["targets"][domain]["target_calibrated"]["PICP"] for j in j5]
        re5 = [r["PICP"] for r in rows if r["model"] == model and r["domain"] == domain
               and r["route"] == "target" and r["alpha"] == 0.10]
        re5_sorted = sorted(re5)  # 种子顺序即 42..46
        d = [a - b for a, b in zip(re5_sorted, orig)]
        drifts += [abs(x) for x in d]
        print(f"  {model}/{domain}: 重算={['%.3f' % x for x in re5_sorted]}  原值={['%.3f' % x for x in orig]}  "
              f"max|Δ|={max(abs(x) for x in d):.3f}")
print(f"全部 20 组 max|Δ| = {max(drifts):.3f}（训练非确定性漂移）")

# ---- 2. alpha 扫描汇总（target 路由） ----
print("\n=== alpha 扫描（target 路由，5 种子均值±std） ===")
hdr = f"{'model/domain':14s} {'alpha':>5s} {'PICP':>14s} {'NMPIW':>14s} {'Winkler':>10s} {'pinball':>9s}"
print(hdr)
agg_rows = []
for model in ("tcn", "lstm"):
    for domain in ("CALCE", "NASA"):
        for alpha in (0.05, 0.10, 0.20):
            sel = [r for r in rows if r["model"] == model and r["domain"] == domain
                   and r["route"] == "target" and r["alpha"] == alpha]
            picp = [r["PICP"] for r in sel]; nmpiw = [r["NMPIW"] for r in sel]
            win = [r["Winkler"] for r in sel]; pin = [r["pinball"] for r in sel]
            agg_rows.append(dict(model=model, domain=domain, alpha=alpha,
                                 PICP_mean=round(float(np.mean(picp)), 3),
                                 PICP_std=round(float(np.std(picp, ddof=1)), 3),
                                 NMPIW_mean=round(float(np.mean(nmpiw)), 3),
                                 NMPIW_std=round(float(np.std(nmpiw, ddof=1)), 3),
                                 Winkler_mean=round(float(np.mean(win)), 4),
                                 pinball_mean=round(float(np.mean(pin)), 5)))
            print(f"{model+'/'+domain:14s} {alpha:5.2f} "
                  f"{np.mean(picp):6.3f}±{np.std(picp, ddof=1):.3f} "
                  f"{np.mean(nmpiw):7.3f}±{np.std(nmpiw, ddof=1):.3f} "
                  f"{np.mean(win):10.4f} {np.mean(pin):9.5f}")

with open(os.path.join(P0, "results_p0", "p0_2_alpha_summary.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(agg_rows[0].keys()))
    w.writeheader(); w.writerows(agg_rows)
print("\n已写 p0_2_alpha_summary.csv")

# source 路由 alpha=0.10 对照（20/20 欠覆盖复核）
print("\n=== source 路由 alpha=0.10（欠覆盖复核，应全部 <0.90） ===")
bad = 0
for model in ("tcn", "lstm"):
    for domain in ("CALCE", "NASA"):
        sel = [r["PICP"] for r in rows if r["model"] == model and r["domain"] == domain
               and r["route"] == "source" and r["alpha"] == 0.10]
        m = float(np.mean(sel))
        flag = "OK" if m < 0.90 else "!! 违反"
        bad += (m >= 0.90)
        print(f"  {model}/{domain}: {m:.3f}  {flag}")
print(f"欠覆盖复核：{20 - bad}/20 组 <0.90")
