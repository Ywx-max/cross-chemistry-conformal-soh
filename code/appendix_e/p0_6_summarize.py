# -*- coding: utf-8 -*-
"""P0-6 summary: W/H sensitivity pivots (5-seed mean+/-std) against the W=20/H=10 baselines and persistence.
Baselines: W=20/H=10 fine-tuned RMSE (Table 3: CALCE/TCN 0.0913, NASA/TCN 0.0230; LSTM 0.0819/0.0227);
persistence (H=10 pooled): CALCE 0.0832 / NASA 0.0269; H=5/20: 0.0665/0.1066, 0.0181/0.0461.
The LSTM H5/H20 persistence controls use the paired section (persistence_baseline.json)."""
import csv, io, json, os, sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import numpy as np

P0 = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OSS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

rows = list(csv.DictReader(open(os.path.join(P0, "results_p0", "p0_6_wh_sensitivity.csv"), encoding="utf-8")))
for r in rows:
    r["ft_rmse"] = float(r["ft_rmse"]); r["zero_rmse"] = float(r["zero_rmse"])

BASE_FT = {("tcn", "CALCE"): 0.0913, ("tcn", "NASA"): 0.0230,
           ("lstm", "CALCE"): 0.0819, ("lstm", "NASA"): 0.0227}
BASE_PERS = {("CALCE", 10): 0.0832, ("NASA", 10): 0.0269,
             ("CALCE", 5): 0.0665, ("NASA", 5): 0.0181,
             ("CALCE", 20): 0.1066, ("NASA", 20): 0.0461}

print("=== Part A: W sensitivity (TCN, H=10; baseline W=20 in brackets) ===")
out_rows = []
for W in (10, 30, 40):
    for dom in ("CALCE", "NASA"):
        v = [r["ft_rmse"] for r in rows if r["part"] == "A_window" and r["W"] == str(W) and r["domain"] == dom]
        b = BASE_FT[("tcn", dom)]
        print(f"  W={W:2d} {dom:6s} ft {np.mean(v):.4f}+/-{np.std(v, ddof=1):.4f}  (W=20: {b:.4f})  "
              f"persistence {BASE_PERS[(dom, 10)]:.4f}")
        out_rows.append(dict(part="A", model="tcn", W=W, H=10, domain=dom,
                             ft_mean=round(float(np.mean(v)), 4), ft_std=round(float(np.std(v, ddof=1)), 4),
                             baseline=BASE_FT[("tcn", dom)], persistence=BASE_PERS[(dom, 10)]))

print("\n=== Part B: LSTM H sensitivity (W=20; baseline H=10 in brackets) ===")
for H in (5, 20):
    for dom in ("CALCE", "NASA"):
        v = [r["ft_rmse"] for r in rows if r["part"] == "B_horizon_lstm" and r["H"] == str(H) and r["domain"] == dom]
        b = BASE_FT[("lstm", dom)]
        print(f"  H={H:2d} {dom:6s} ft {np.mean(v):.4f}+/-{np.std(v, ddof=1):.4f}  (H=10: {b:.4f})  "
              f"persistence {BASE_PERS[(dom, H)]:.4f}")
        out_rows.append(dict(part="B", model="lstm", W=20, H=H, domain=dom,
                             ft_mean=round(float(np.mean(v)), 4), ft_std=round(float(np.std(v, ddof=1)), 4),
                             baseline=BASE_FT[("lstm", dom)], persistence=BASE_PERS[(dom, H)]))

# TCN H5/H20 (existing transfer_sens runs)
print("\n=== control: TCN H5/H20 (existing transfer_sens_h5/h20) ===")
for H in (5, 20):
    d = os.path.join(OSS, "results_cx2", f"transfer_sens_h{H}")
    for dom in ("CALCE", "NASA"):
        v = []
        for s in range(42, 47):
            j = json.load(open(os.path.join(d, f"t3b_tcn_s{s}.json"), encoding="utf-8"))
            v.append(j["targets"][dom]["fine_tune"]["rmse"])
        print(f"  TCN H={H} {dom:6s} ft {np.mean(v):.4f}+/-{np.std(v, ddof=1):.4f}  persistence {BASE_PERS[(dom, H)]:.4f}")

with open(os.path.join(P0, "results_p0", "p0_6_summary.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
    w.writeheader(); w.writerows(out_rows)
print("\nwrote p0_6_summary.csv")
