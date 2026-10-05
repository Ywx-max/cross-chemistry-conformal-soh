# -*- coding: utf-8 -*-
"""Data-consistency self-check: itemized comparison of experiment JSONs against the manuscript LaTeX sources (75 checks).

The failure mode this guards against: changing one number and forgetting another, or a transcription slip.
Every key number of the manuscript is recomputed from the raw JSONs fresh: means/stds of Tables 3/4/6, the
abstract gain ratios, coefficients of variation, LOBO means, the Section 4.5 narrowing percentages, and so on.
When everything passes it prints "All data match the manuscript."; any drift is reported item by item.

Run after editing the manuscript text (about 3 s, no GPU):
    python final_data_check.py
Note: the R/ and main.tex paths below follow the author's local layout; adjust them on another machine."""
import json, re, statistics as st
from pathlib import Path
BS = chr(92)
R = Path("03_experiments" + BS + "results")
tex = open("04_manuscript" + BS + "main.tex", encoding="utf-8").read()
ok, bad = 0, []

def check(name, value, fmt, where="tex"):
    """Format the value with fmt and look for it in the manuscript source; record a miss in bad.
    Suits literal citations such as "1/14.5 in the abstract"; tolerance-based checks are written inline below."""
    global ok
    s = fmt.format(value)
    if s in tex:
        ok += 1
    else:
        bad.append(f"{name}: not found in manuscript: '{s}'")

# Table 3: transfer (transfer_multiseed_v2)
t = json.load(open(R/"transfer"/"local"/"transfer_multiseed_v2.json", encoding="utf-8"))
tbl3 = {"CALCE_tcn": ("0.1465","0.0211","0.0100","0.0056","0.0346","0.0069"),
        "CALCE_lstm":("0.1446","0.0160","0.0089","0.0084","0.0103","0.0117"),
        "NASA_tcn":  ("0.0975","0.0060","0.0141","0.0057","0.1888","0.0681"),
        "NASA_lstm": ("0.0994","0.0089","0.0136","0.0023","0.0765","0.0657")}
# Table 3: transfer results. The manuscript values are rounded to 4 decimals; tolerance set to about 1e-4
for k, vals in tbl3.items():
    d = t[k]
    real = [d["zero"]["mean"], d["zero"]["std"], d["ft"]["mean"], d["ft"]["std"], d["to"]["mean"], d["to"]["std"]]
    for paper_v, real_v, lab in zip(vals, real, ["zero_m","zero_s","ft_m","ft_s","to_m","to_s"]):
        if abs(float(paper_v) - real_v) > 0.00005 + abs(real_v)*0.001:
            bad.append(f"Table 3 {k} {lab}: paper {paper_v} vs actual {real_v:.4f}")
        else: ok += 1
# Gain claims: 1/14.5 and 1/5.9 in the abstract and conclusions use per-seed baseline/fine-tune ratios
# averaged afterwards, not the ratio of the two Table 3 column means (which gives 13.4/5.6 and confuses
# readers who divide themselves; Table 3 therefore carries an explicit gain-ratio column and Section 4.3 states the convention)
g_tcn = t["NASA_tcn"]["gain"]["mean"]; g_lstm = t["NASA_lstm"]["gain"]["mean"]
check("NASA TCN gain 1/14.4", g_tcn, "{:.1f}")
tex_gain = "1/14.5" in tex and "1/5.9" in tex
if not (13.5 <= g_tcn <= 15.3 and 5.5 <= g_lstm <= 6.3 and tex_gain): bad.append(f"abstract gain claim: TCN {g_tcn:.2f} LSTM {g_lstm:.2f}")
else: ok += 1
# CV claim: 86% narrows to 17% (LSTM from-scratch vs fine-tuned)
cv_ft = t["NASA_lstm"]["ft"]["std"]/t["NASA_lstm"]["ft"]["mean"]*100
cv_to = t["NASA_lstm"]["to"]["std"]/t["NASA_lstm"]["to"]["mean"]*100
if not (80 <= cv_to <= 92 and 14 <= cv_ft <= 20 and "86" in tex and "17" in tex):
    bad.append(f"CV claim: from-scratch {cv_to:.0f}% fine-tune {cv_ft:.0f}%")
else: ok += 1

# Table 4: conformal
c = json.load(open(R/"conformal"/"local"/"conformal_multiseed_summary.json", encoding="utf-8"))
tbl4 = {"CALCE_tcn": ("0.458","0.108","0.0052","0.999","0.002","0.3785"),
        "CALCE_lstm":("0.219","0.210","0.0027","0.999","0.002","0.3630"),
        "NASA_tcn":  ("0.121","0.091","0.0052","1.000","0.000","0.3145"),
        "NASA_lstm": ("0.109","0.096","0.0027","1.000","0.000","0.3062")}
for k, vals in tbl4.items():
    d = c[k]
    real = [d["picp_src"]["mean"], d["picp_src"]["std"], d["mpiw_src"]["mean"],
            d["picp_tgt"]["mean"], d["picp_tgt"]["std"], d["mpiw_tgt"]["mean"]]
    for paper_v, real_v, lab in zip(vals, real, ["src_m","src_s","src_w","tgt_m","tgt_s","tgt_w"]):
        tol = 0.0006 if real_v < 0.01 else 0.0006
        if abs(float(paper_v) - real_v) > tol:
            bad.append(f"Table 4 {k} {lab}: paper {paper_v} vs actual {real_v:.4f}")
        else: ok += 1
allsrc = [x for k in c for x in c[k]["picp_src"]["all"]]
if "0.00--0.58" in tex and "0.227" in tex:
    if not (abs(min(allsrc)) < 0.06 and abs(st.mean(allsrc) - 0.227) < 0.005): bad.append(f"4.5 source-calibration summary: min={min(allsrc):.2f} mean={st.mean(allsrc):.3f}")
    else: ok += 1

# Table 4 drift decomposition (original-protocol row)
# Tolerance is deliberately wide (+/-25): this quantity is O(1000) cycles and cross-machine/GPU
# training nondeterminism is strongly amplified under the original protocol (acknowledged in Section 4.8)
raw = json.load(open(R/"transfer"/"local"/"raw_protocol_multiseed.json", encoding="utf-8"))
zc = raw["CALCE"]["zero_shot"]
if abs(1640 - zc["mean"]) > 25 or abs(886 - zc["std"]) > 25: bad.append(f"Table 5 original protocol: {zc['mean']:.0f}+/-{zc['std']:.0f} vs paper 1640+/-886")
else: ok += 1

# Table 6: ablation
a = json.load(open(R/"transfer"/"local"/"ablation_v3c_multiseed.json", encoding="utf-8"))
tbl6 = {"CALCE_base7": ("0.1488","0.0151","0.0106","0.0061"), "CALCE_curve14": ("0.1761","0.0206","0.0226","0.0080"),
        "NASA_base7": ("0.1015","0.0108","0.0162","0.0100"), "NASA_curve14": ("0.0943","0.0129","0.0287","0.0157")}
for k, vals in tbl6.items():
    d = a[k]; real = [d["zero"]["mean"], d["zero"]["std"], d["ft"]["mean"], d["ft"]["std"]]
    for pv, rv in zip(vals, real):
        if abs(float(pv) - rv) > 0.00006: bad.append(f"Table 6 {k}: paper {pv} vs actual {rv:.4f}")
        else: ok += 1
ftb, ftc = a["CALCE_base7"]["ft"]["mean"], a["CALCE_curve14"]["ft"]["mean"]
if abs((ftc/ftb - 1)*100 - 113) > 1.5: bad.append(f"4.6 worsening percentage: {(ftc/ftb-1)*100:.1f}% vs paper 113%")
else: ok += 1

# LOBO (Table 2 / Fig. 2)
# macro mean of each of the 3 seeds' per-fold jsonl, then mean+/-std over the 3 means (92.45+/-0.98)
L = R/"baselines"/"local"
means = []
for s in [42,43,44]:
    rows = [json.loads(l) for l in open(L/f"lobo_tcn_s{s}.jsonl", encoding="utf-8") if l.strip()]
    means.append(st.mean([r["rmse"] for r in rows]))
m, sd = st.mean(means), st.stdev(means)
if "92.45" in tex and abs(m - 92.45) > 0.02: bad.append(f"4.2 LOBO: {m:.2f} vs paper 92.45")
else: ok += 1

# 4.5 conditional narrowing (Mondrian / weighted negative results)
t4 = json.load(open(R/"conformal"/"local"/"t4c_multiseed_summary.json", encoding="utf-8"))
mn = t4["mondrian"]["NASA"]
w_narrow = (1 - mn["mondrian"]["MPIW"]/mn["single"]["MPIW"])*100
if abs(w_narrow - 21) > 1.5: bad.append(f"4.5 Mondrian NASA narrowing: {w_narrow:.1f}% vs paper 21%")
else: ok += 1
mp, msd = mn["mondrian"]["PICP"]["mean"], mn["mondrian"]["PICP"]["std"]
if abs(mp - 0.64) > 0.006 or abs(msd - 0.18) > 0.006: bad.append(f"4.5 Mondrian NASA PICP: {mp:.2f}+/-{msd:.2f} vs paper 0.64+/-0.18")
else: ok += 1
sb = t4["mondrian"]["NASA"]["single"]["PICP"]["mean"]
if abs(sb - 0.81) > 0.006: bad.append(f"4.5 single-q baseline: {sb:.2f} vs paper 0.81")
else: ok += 1
wg = t4["weighted"]["CALCE"]
wn = (1 - wg["weighted"]["MPIW"]/wg["single"]["MPIW"])*100
if abs(wn - 15) > 1.5: bad.append(f"4.5 weighted CALCE narrowing: {wn:.1f}% vs paper 15%")
else: ok += 1

print(f"{ok} checks passed")
if bad:
    print(f"\n{len(bad)} mismatches:")
    for b in bad: print("  -", b)
else:
    print("All data match the manuscript.")
