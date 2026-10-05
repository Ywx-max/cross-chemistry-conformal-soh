# -*- coding: utf-8 -*-
"""Check the Appendix E table numbers of the manuscript against the results_p0 CSVs (traceability)."""
import csv, io, json, os, sys, re
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
_REL = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_T = json.load(open(os.path.join(_REL, "results_p0", "appendixE_tables.json"), encoding="utf-8"))
TABLEE1, TABLEE2, TABLEE3, TABLEE4 = _T["E1"], _T["E2"], _T["E3"], _T["E4"]

P0 = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
bad = []


def close(a, b, tol=5e-4):
    return abs(float(a) - float(b)) <= tol


# E.1 vs p0_2_alpha_summary.csv
summ = list(csv.DictReader(open(os.path.join(P0, "results_p0", "p0_2_alpha_summary.csv"), encoding="utf-8")))
idx = {(r["model"], r["domain"], f"{float(r['alpha']):.2f}"): r for r in summ}
for row in TABLEE1["rows"]:
    b, dom, a, picp, nmpiw, wink, pin = row
    r = idx[(b.lower(), dom, f"{float(a):.2f}")]
    if not close(r["PICP_mean"], picp.split("±")[0]) or not close(r["PICP_std"], picp.split("±")[1]):
        bad.append(f"E.1 PICP {b}/{dom}/{a}: table {picp} vs CSV {r['PICP_mean']}+/-{r['PICP_std']}")
    if not close(r["Winkler_mean"], wink):
        bad.append(f"E.1 Winkler {b}/{dom}/{a}: table {wink} vs CSV {r['Winkler_mean']}")
    if not close(r["pinball_mean"], pin):
        bad.append(f"E.1 pinball {b}/{dom}/{a}: table {pin} vs CSV {r['pinball_mean']}")
print(f"E.1: {len(TABLEE1['rows'])} rows checked, problems {len(bad)}")

# E.2 vs p0_3_statistics.csv
st = list(csv.DictReader(open(os.path.join(P0, "results_p0", "p0_3_statistics.csv"), encoding="utf-8")))
idx3 = {r["comparison"]: r for r in st}
keymap = {"CALCE/TCN, H = 5": "h5|tcn|CALCE|ft|H5", "CALCE/TCN, H = 10": "h10|tcn|CALCE|ft|H10",
          "CALCE/TCN, H = 20": "h20|tcn|CALCE|ft|H20", "NASA/TCN, H = 5": "h5|tcn|NASA|ft|H5",
          "NASA/TCN, H = 10": "h10|tcn|NASA|ft|H10", "NASA/TCN, H = 20": "h20|tcn|NASA|ft|H20",
          "CALCE/LSTM, H = 10": "h10|lstm|CALCE|ft|H10", "NASA/LSTM, H = 10": "h10|lstm|NASA|ft|H10"}
n0 = len(bad)
for row in TABLEE2["rows"]:
    r = idx3[keymap[row[0]]]
    if not close(row[1], r["cohens_dz"], 0.005):
        bad.append(f"E.2 dz {row[0]}: table {row[1]} vs CSV {r['cohens_dz']}")
    if abs(float(row[2]) - float(r["wilcoxon_p"])) > 1e-9:
        bad.append(f"E.2 p {row[0]}: table {row[2]} vs CSV {r['wilcoxon_p']}")
    if abs(float(row[3]) - float(r["holm_p"])) > 5e-4:
        bad.append(f"E.2 holm {row[0]}: table {row[3]} vs CSV {r['holm_p']}")
    if row[4].split("/")[0] != r["wins"]:
        bad.append(f"E.2 wins {row[0]}: table {row[4]} vs CSV {r['wins']}")
print(f"E.2: {len(TABLEE2['rows'])} rows checked, problems {len(bad) - n0}")

# E.3 vs p0_5_cost.csv
n0 = len(bad)
c5 = {r["model"]: r for r in csv.DictReader(open(os.path.join(P0, "results_p0", "p0_5_cost.csv"), encoding="utf-8"))}
for row in TABLEE3["rows"]:
    r = c5[row[0].lower()]
    if not close(row[1].rstrip("M"), r["params_M"], 0.0005):
        bad.append(f"E.3 params {row[0]}: table {row[1]} vs CSV {r['params_M']}")
    if abs(float(row[2]) - float(r["train_time_s"])) > 0.05:
        bad.append(f"E.3 train {row[0]}: table {row[2]} vs CSV {r['train_time_s']}")
    if abs(float(row[3]) - float(r["peak_mem_MiB"])) > 0.5:
        bad.append(f"E.3 mem {row[0]}: table {row[3]} vs CSV {r['peak_mem_MiB']}")
print(f"E.3: {len(TABLEE3['rows'])} rows checked, problems {len(bad) - n0}")

# E.4 vs p0_1_uncertainty.csv
n0 = len(bad)
c1 = list(csv.DictReader(open(os.path.join(P0, "results_p0", "p0_1_uncertainty.csv"), encoding="utf-8")))
meth = {"MC dropout (p = 0.1, 30 samples)": "MC dropout",
        "Deep ensemble (5 members, seed-42 split)": "Deep ensemble (5, seed42 split)",
        "Quantile regression (tau = 0.1/0.9)": "Quantile regression",
        "Split conformal (target-calibrated)": "Split conformal (recomputed)"}
conf_tcn = [r for r in c1 if r["method"] == meth["Split conformal (target-calibrated)"]][:10]
for row in TABLEE4["rows"]:
    for dom_i, dom in ((1, "CALCE"), (4, "NASA")):
        pool = conf_tcn if "conformal" in row[0].lower() else \
            [r for r in c1 if r["method"] == meth[row[0]]]
        sel = [r for r in pool if r["domain"] == dom]
        v = [float(r["PICP"]) for r in sel]
        m_tab = float(row[dom_i].split("±")[0])
        if abs(float(np.mean(v)) - m_tab) > 5e-4:
            bad.append(f"E.4 PICP {row[0]}/{dom}: table {m_tab} vs CSV {np.mean(v):.4f}")
        w = [float(r["Winkler"]) for r in sel]
        w_tab = float(row[dom_i + 2])
        if abs(float(np.mean(w)) - w_tab) > 6e-4:
            bad.append(f"E.4 Winkler {row[0]}/{dom}: table {w_tab} vs CSV {np.mean(w):.4f}")
print(f"E.4: {len(TABLEE4['rows'])} rows checked, problems {len(bad) - n0}")

print()
if bad:
    print(f"{len(bad)} mismatches:")
    for b in bad:
        print("  !!", b)
    sys.exit(1)
print("All appendix table numbers match the results CSVs.")
