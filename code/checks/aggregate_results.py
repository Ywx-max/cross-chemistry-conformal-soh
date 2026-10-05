# -*- coding: utf-8 -*-
"""Aggregation: regenerates every multi-seed summary file cited in the manuscript from the per-seed raw files under results/.

Background (2026-10): the repository previously shipped 10 summary JSON/CSV files with no script generating
them, and 3 of them disagreed with the per-seed files (the TCN row of transfer_multiseed_v2.json, the base7
row of ablation_multiseed_v5.json, and the two-repeat aggregates of LSTM/Transformer/ensemble in
lobo_final_multiseed.json). This script supplies the missing aggregation layer; the per-seed files are the only input:

  baselines/baseline_3seed_final.json   <- {model}_f5_s42-44.json (5-fold group CV)
  baselines/lobo_final_multiseed.json   <- lobo_{tcn_s42,tcn_s43,tcn_s44}.jsonl (3 repeats)
                                           + lobo_tcn/lstm/transformer.jsonl (original runs)
  baselines/lobo_3model_final.csv       <- the three original-run jsonl files aligned by cell
  transfer/transfer_multiseed_v2.json   <- t3b_{tcn,lstm}_s42-46.json (Table 3)
  transfer/raw_protocol_multiseed.json  <- t3_soh_tcn_s42-46.json (first row of Table 4)
  ablation/ablation_multiseed.json      <- t3c_base7_tcn_s42-46.json (v3 data version)
  ablation/ablation_multiseed_v5.json   <- t3d_{base7,curve14}_tcn_s42-46.json (v5)
  ablation/ablation_v3c_multiseed.json  <- t3e_{base7,curve14}_tcn_s42-46.json (v3c, used in the manuscript)
  conformal/conformal_multiseed_summary.json <- t4_{tcn,lstm}_s42-46.json (Table 5)
  conformal/t4c_multiseed_summary.json  <- t4c_{mondrian,weighted}_s42-46.json (Section 4.5 extensions)

Conventions:
  * all std values are the sample std (ddof=1), matching the manuscript table captions.
  * lobo_final_multiseed.json keeps only runs with per-fold jsonl support (the TCN repeats);
    the LSTM / Transformer / ensemble repeats lack per-fold raw files and are excluded.
  * t4c_mondrian_*.json is the original pre-2026-10 run (that script was defect-free and was not rerun);
    t4c_weighted_*.json was rerun after fixing a test-label leak and an off-by-one in the weighted quantile.

Run: python code/checks/aggregate_results.py (from the repository root)
"""
import csv
import json
import glob
import os
import numpy as np

import argparse

R = "results"  # overridable via --results (cx2 mode points to results_cx2)


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def dump(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)
    print("wrote", path)


def ms(vals):
    v = np.asarray(vals, dtype=float)
    return {"mean": float(v.mean()), "std": float(v.std(ddof=1))}


def agg_t3b():
    """Table 3: per-dataset standardization, t3b_{model}_s{seed}.json."""
    out = {}
    for model in ("tcn", "lstm"):
        for ds in ("CALCE", "NASA"):
            zero, ft, to, gain = [], [], [], []
            for s in range(42, 47):
                t = load(f"{R}/transfer/t3b_{model}_s{s}.json")["targets"][ds]
                zero.append(t["zero_shot"]["rmse"])
                ft.append(t["fine_tune"]["rmse"])
                to.append(t["target_only"]["rmse"])
                gain.append(t["target_only"]["rmse"] / t["fine_tune"]["rmse"])
            out[f"{ds}_{model}"] = {"zero": ms(zero), "ft": ms(ft),
                                    "to": ms(to), "gain": ms(gain)}
    dump(f"{R}/transfer/transfer_multiseed_v2.json", out)


def agg_t3soh():
    """First row of Table 4: original protocol (source-domain statistics), t3_soh_tcn_s{seed}.json."""
    out = {}
    for ds in ("CALCE", "NASA"):
        zero, ft = [], []
        for s in range(42, 47):
            t = load(f"{R}/transfer/t3_soh_tcn_s{s}.json")["targets"][ds]
            zero.append(t["zero_shot"]["rmse"])
            ft.append(t["fine_tune"]["rmse"])
        out[ds] = {"n_seeds": 5, "seeds": [42, 43, 44, 45, 46],
                   "zero_shot": ms(zero), "fine_tune": ms(ft)}
    dump(f"{R}/transfer/raw_protocol_multiseed.json", out)


def agg_t3x(dirname, stem, out_name, variants):
    """The three ablation variants (t3c/t3d/t3e) share the structure {targets ds} x {variants}."""
    out = {}
    for variant in variants:
        for ds in ("CALCE", "NASA"):
            zero, ft, to = [], [], []
            for s in range(42, 47):
                p = f"{R}/{dirname}/{stem}_{variant}_tcn_s{s}.json"
                t = load(p)["targets"][ds]
                zero.append(t["zero_shot"]["rmse"])
                ft.append(t["fine_tune"]["rmse"])
                to.append(t["target_only"]["rmse"])
            out[f"{ds}_{variant}"] = {"n": 5, "zero": ms(zero), "ft": ms(ft), "to": ms(to)}
    dump(f"{R}/{dirname}/{out_name}", out)


def agg_f5():
    """Table 1: 5-fold group CV, 3 seeds. Std is the sample std over seeds (ddof=1)."""
    out = {}
    for model in ("lstm", "gru", "tcn", "transformer"):
        seed_rmse, seed_mae = [], []
        for s in range(42, 45):
            d = load(f"{R}/baselines/{model}_f5_s{s}.json")
            folds = d["folds"]
            # per-seed RMSE = arithmetic mean of the 5 fold RMSEs (matching Table 1)
            seed_rmse.append(np.mean([f["rmse"] for f in folds]))
            seed_mae.append(np.mean([f["mae"] for f in folds]))
        seed_rmse, seed_mae = np.asarray(seed_rmse), np.asarray(seed_mae)
        out[model] = {"rmse_mean": float(seed_rmse.mean()),
                      "rmse_std": float(seed_rmse.std(ddof=1)),
                      "seeds": {str(s): float(v) for s, v in zip((42, 43, 44), seed_rmse)},
                      "mae_mean": float(seed_mae.mean())}
    dump(f"{R}/baselines/baseline_3seed_final.json", out)


def lobo_stats(recs):
    """Per-cell RMSE distribution statistics of one LOBO run (jsonl).
    Quantiles use the ascending floor(q*n)+1-th value (1-based; implemented as the 0-based index
    floor(q*n), matching Table 2 and verify_results_cx2.py; np.quantile(method="lower") is floor(q*(n-1)), do not use).
    Std is the sample std, ddof=1, as in Table 2 and the module docstring."""
    r = np.sort(np.asarray([x["rmse"] for x in recs]))
    n = len(r)
    q = lambda p: float(r[min(int(p * n), n - 1)])
    return {"min": float(r[0]), "q1": q(0.25), "median": q(0.50), "q3": q(0.75),
            "mean": float(r.mean()), "std": float(r.std(ddof=1))}


def read_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def agg_lobo():
    """Table 2 / Fig. 2: TCN 3 repeats + the other backbones' original runs. The LSTM/Transformer
    two-repeat aggregates are withdrawn because the per-fold files are missing (see module docstring)."""
    out = {}
    run_files = sorted(glob.glob(f"{R}/baselines/lobo_tcn_s4[2-4].jsonl"))
    seeds = [int(x.split("_s")[-1].split(".")[0]) for x in run_files]
    per_run = {str(s): lobo_stats(read_jsonl(p)) for s, p in zip(seeds, run_files)}
    means = np.asarray([per_run[str(s)]["mean"] for s in seeds])
    medians = np.asarray([per_run[str(s)]["median"] for s in seeds])
    out["tcn"] = {"n_runs": len(seeds), "seeds": seeds, "per_run": per_run,
                  "cross_mean": float(means.mean()), "cross_std": float(means.std(ddof=1)),
                  "cross_median": float(medians.mean()), "median_std": float(medians.std(ddof=1))}
    for model in ("lstm", "transformer"):
        st = lobo_stats(read_jsonl(f"{R}/baselines/lobo_{model}.jsonl"))
        out[model] = {"n_runs": 1, "seeds": [42], "per_run": {"42": st},
                      "cross_mean": st["mean"], "cross_std": None,
                      "cross_median": st["median"], "median_std": None}
    dump(f"{R}/baselines/lobo_final_multiseed.json", out)


def agg_lobo_csv():
    """Table 2 ensemble row: the three original runs aligned by cell (regeneration of lobo_3model_final.csv)."""
    models = ("tcn", "lstm", "transformer")
    by_cell = {}
    for m in models:
        for rec in read_jsonl(f"{R}/baselines/lobo_{m}.jsonl"):
            by_cell.setdefault(rec["te_cell"], {})[m] = rec["rmse"]
    rows = []
    for cell in sorted(by_cell):
        d = by_cell[cell]
        if len(d) == len(models):
            rows.append({"te_cell": cell, "rmse_tcn": d["tcn"],
                         "rmse_lstm": d["lstm"], "rmse_tf": d["transformer"]})
    p = f"{R}/baselines/lobo_3model_final.csv"
    with open(p, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["te_cell", "rmse_tcn", "rmse_lstm", "rmse_tf"])
        w.writeheader()
        w.writerows(rows)
    print("wrote", p)
    # ensemble (per-cell best of three; uses test labels, a non-deployable upper bound)
    ens = float(np.mean([min(r["rmse_tcn"], r["rmse_lstm"], r["rmse_tf"]) for r in rows]))
    print(f"per-cell three-model oracle mean = {ens:.2f}")


def agg_conformal():
    """Table 5: t4_{model}_s{seed}.json (2026-10 rerun: calibration and evaluation share the model)."""
    out = {}
    for model in ("tcn", "lstm"):
        for ds in ("CALCE", "NASA"):
            pt, ps, wt, ws, rm = [], [], [], [], []
            for s in range(42, 47):
                t = load(f"{R}/conformal/t4_{model}_s{s}.json")["targets"][ds]
                pt.append(t["target_calibrated"]["PICP"])
                ps.append(t["source_calibrated"]["PICP"])
                wt.append(t["target_calibrated"]["MPIW"])
                ws.append(t["source_calibrated"]["MPIW"])
                rm.append(t["point_rmse"])
            out[f"{ds}_{model}"] = {
                "seeds": [42, 43, 44, 45, 46],
                "picp_tgt": {**ms(pt), "all": [float(v) for v in pt]},
                "picp_src": {**ms(ps), "all": [float(v) for v in ps]},
                "mpiw_tgt": ms(wt), "mpiw_src": ms(ws), "rmse": ms(rm)}
    dump(f"{R}/conformal/conformal_multiseed_summary.json", out)


def agg_t4c():
    """Section 4.5 extensions: Mondrian (original run) + weighted (rerun after the fix)."""
    out = {}
    for variant in ("mondrian", "weighted"):
        out[variant] = {}
        for ds in ("CALCE", "NASA"):
            single_p, single_w, var_p, var_w, cond_p, cond_w = [], [], [], [], [], []
            for s in range(42, 47):
                t = load(f"{R}/conformal/t4c_{variant}_s{s}.json")["targets"][ds]
                single_p.append(t["single"]["PICP"]); single_w.append(t["single"]["MPIW"])
                var_p.append(t[variant]["PICP"]); var_w.append(t[variant]["MPIW"])
                cond_p.append(t[variant]["PICP"]); cond_w.append(t[variant]["MPIW"])
            out[variant][ds] = {
                "n_seeds": 5, "seeds": [42, 43, 44, 45, 46],
                "single": {"PICP": ms(single_p), "MPIW": float(np.mean(single_w))},
                variant: {"PICP": ms(var_p), "MPIW": float(np.mean(var_w))},
                "point_rmse": float(np.mean([
                    load(f"{R}/conformal/t4c_{variant}_s{s}.json")["targets"][ds]["point_rmse"]
                    for s in range(42, 47)])),
                "cond_picp_all": [float(v) for v in cond_p],
                "cond_mpiw_all": [float(v) for v in cond_w]}
    dump(f"{R}/conformal/t4c_multiseed_summary.json", out)


def agg_sweep():
    """Coverage vs calibration-cell count sweep (cx2)."""
    out = {}
    for model in ("tcn", "lstm"):
        for nc in (1, 2, 3, 4, 5):
            files = sorted(glob.glob(f"{R}/conformal/t4_split_sweep/t4_{model}_s4[2-6]_cal{nc}.json"))
            for tgt in ("CALCE",):
                pt, pw, ps = [], [], []
                for f in files:
                    d = load(f)
                    if tgt not in d["targets"]:
                        continue
                    t = d["targets"][tgt]
                    pt.append(t["target_calibrated"]["PICP"])
                    pw.append(t["target_calibrated"]["MPIW"])
                    ps.append(t["source_calibrated"]["PICP"])
                if not pt:
                    continue
                v = np.asarray(pt)
                out.setdefault(model, {})[f"cal{nc}"] = {
                    "split_ft_cal_te": [16 - 7 - nc, nc, 7],
                    "picp_tgt": {"mean": float(v.mean()), "std": float(v.std(ddof=1)),
                                 "all": [float(x) for x in pt]},
                    "mpiw_tgt_mean": float(np.mean(pw)),
                    "picp_src_mean": float(np.mean(ps))}
    dump(f"{R}/conformal/t4_split_sweep/sweep_summary.json", out)


def agg_i9():
    """Sampling distribution of the split redraws (50 runs, cx2)."""
    out = {}
    for model in ("tcn", "lstm"):
        pt, pw, rm = [], [], []
        for s in range(1001, 1051):
            pth = f"{R}/conformal/i9_seeds/t4_{model}_s{s}.json"
            if not os.path.exists(pth):
                continue
            d = load(pth)
            t = d["targets"]["CALCE"]
            pt.append(t["target_calibrated"]["PICP"])
            pw.append(t["target_calibrated"]["MPIW"])
            rm.append(t["point_rmse"])
        v = np.asarray(pt)
        if len(v) == 0:
            continue  # this backbone's i9 is not finished yet (tcn first, lstm later)
        q = lambda p: float(np.quantile(v, p))
        out[model] = {"n": len(v),
                      "picp_tgt": {"mean": float(v.mean()), "std": float(v.std(ddof=1)),
                                   "q05": q(0.05), "q50": q(0.50), "q95": q(0.95),
                                   "min": float(v.min()), "max": float(v.max()),
                                   "all": [float(x) for x in v]},
                      "frac_ge_090": float(np.mean(v >= 0.90)),
                      "mpiw_mean": float(np.mean(pw)), "rmse_mean": float(np.mean(rm))}
    dump(f"{R}/conformal/i9_seeds/i9_summary.json", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results", help="results root (pass results_cx2 for cx2 mode)")
    args = ap.parse_args()
    R = args.results
    agg_t3b()
    agg_t3soh()
    if R == "results":
        # source-domain artifacts and the v3/v5-era ablations belong to the old results scope (not rerun for cx2; frozen)
        agg_t3x("ablation", "t3c", "ablation_multiseed.json", ["base7"])
        agg_t3x("ablation", "t3d", "ablation_multiseed_v5.json", ["base7", "curve14"])
        agg_f5()
        agg_lobo()
        agg_lobo_csv()
    agg_t3x("ablation", "t3e", "ablation_v3c_multiseed.json", ["base7", "curve14"])
    agg_conformal()
    agg_t4c()
    if R != "results":
        agg_sweep()
        agg_i9()
    print("AGGREGATE DONE")
