# -*- coding: utf-8 -*-
"""Early-life prediction (Section 4.8): predict whole-life from the first 100 cycles only.

Following the differential-capacity idea of Severson et al. 2019: subtract the discharge voltage-capacity curves
of cycles 10 and 100 and take 6 statistics of Delta-Q(V) (variance, mean, slope, intercept, min, max) as features,
regress log10(life) with ridge, and evaluate leave-one-cell-out over the 119 cells that carry life labels.

Fixed settings: features standardised first, Ridge(alpha=1.0), target log10(cycle_life). These are defaults tried
out on the public data (no hyperparameter search), so the baseline is a deterministic single run with no random seeds.

Input: data/mit_dq_early.csv (Delta-Q feature table extracted from the public MIT-Stanford data as above, shipped with the repository)
Output: results/early_pred_summary.json, results/early_pred_results.csv (per-cell predictions)

    python t5_early_pred.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATA = Path("data/mit_dq_early.csv")
OUT_DIR = Path("results")
FEATURES = ["dq_var", "dq_mean", "dq_slope", "dq_intercept", "dq_min", "dq_max"]


def main():
    df = pd.read_csv(DATA)
    df = df[df["cycle_life"].notna() & (df["cycle_life"] > 0)].reset_index(drop=True)
    X = df[FEATURES].to_numpy(dtype=float)
    y_log = np.log10(df["cycle_life"].to_numpy(dtype=float))

    # leave one cell out: each cell serves once as the test set while the other 118 train
    pred_cycles = np.empty(len(df))
    for i in range(len(df)):
        train = np.arange(len(df)) != i
        model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        model.fit(X[train], y_log[train])
        pred_cycles[i] = 10 ** model.predict(X[i : i + 1])[0]

    true_cycles = df["cycle_life"].to_numpy(dtype=float)
    resid = pred_cycles - true_cycles
    summary = {
        "rmse_log": float(np.sqrt(np.mean((np.log10(pred_cycles) - y_log) ** 2))),
        "rmse_cycles": float(np.sqrt(np.mean(resid ** 2))),
        "mae_cycles": float(np.mean(np.abs(resid))),
        "rel_err_pct": float(np.mean(np.abs(resid) / true_cycles) * 100),
        "n": int(len(df)),
        "features": FEATURES,
    }

    OUT_DIR.mkdir(exist_ok=True)
    with open(OUT_DIR / "early_pred_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    pd.DataFrame({
        "battery_id": df["battery_id"],
        "cycle_life": true_cycles,
        "pred_cycle_life": pred_cycles,
    }).to_csv(OUT_DIR / "early_pred_results.csv", index=False)

    print("cells: %d" % summary["n"])
    print("log10-space RMSE : %.4f" % summary["rmse_log"])
    print("cycle-space RMSE : %.1f" % summary["rmse_cycles"])
    print("per-cell relative error : %.2f%%" % summary["rel_err_pct"])


if __name__ == "__main__":
    main()
