# -*- coding: utf-8 -*-
"""Extract first-100-cycle Delta-Q features from the raw MIT-Stanford cycle data (input of Section 4.8).

Feature definition (matching Section 4.8): for each cell take the discharge segments of cycles 10 and 100,
interpolate both discharge capacity-voltage curves onto a common voltage grid (2.0-3.5 V, 1000 points),
take the difference Delta-Q(V) = Q_100(V) - Q_10(V) (with a constant offset correcting the total capacity
between cycles), and extract 6 features: variance, mean, linear-fit slope, intercept, minimum, maximum.

Relation to the shipped data (important):
  data/mit_dq_early.csv is the FROZEN version used by the manuscript, produced by the author's local
  pipeline. This script is an independent implementation of that feature definition: validated on b1c0, the
  variance and mean deviate from the frozen version by < 1%, but slope/intercept/extremes are sensitive to
  discharge-segment boundaries and grid details, so implementation-level differences remain (the original extraction pipeline is not distributed). Therefore:
    * to reproduce the Section 4.8 input bit for bit -> use the shipped data/mit_dq_early.csv;
    * to rerun the baseline on your own features -> run this script, then code/early_pred/t5_early_pred.py.

Usage: python code/data_prep/extract_mit_dq_early.py --mit-dir <directory with b1c*.parquet> \
            --out data/mit_dq_early_extracted.csv
Output: same columns as data/mit_dq_early.csv (battery_id, cycle_10, cycle_100, dq_var,
dq_mean, dq_slope, dq_intercept, dq_min, dq_max, cycle_life, n_cycles_total),
cycle_life requires the EOL labels of each dataset (README/DATA.md); when this script cannot determine
EOL it writes -1 for the caller to fill in.
"""
import argparse
import glob
import os
import numpy as np
import pandas as pd

GRID_LO, GRID_HI, GRID_N = 2.0, 3.5, 1000
CYCLE_A, CYCLE_B = 10, 100


def dq_features(parquet_path):
    df = pd.read_parquet(parquet_path)
    dis = df[df["current_A"] < -0.01]  # discharge segment
    curves = {}
    for cyc in (CYCLE_A, CYCLE_B):
        c = dis[dis["cycle_number"] == cyc].sort_values("time_s")
        if len(c) < 10:
            return None
        curves[cyc] = (c["voltage_V"].values, c["capacity_Ah"].values,
                       float(c["capacity_Ah"].max()))
    (v_a, q_a, m_a), (v_b, q_b, m_b) = curves[CYCLE_A], curves[CYCLE_B]
    # capacity_Ah on the discharge segment is a "remaining capacity" counter (max at full charge, zero at the end of
    # discharge), so a direct subtraction mixes in the between-cycle total-capacity difference; adding the constant offset makes Delta-Q the true discharge-capacity difference
    offset = m_a - m_b
    grid = np.linspace(GRID_LO, GRID_HI, GRID_N)
    qa = np.interp(grid, np.sort(v_a), q_a[np.argsort(v_a)])
    qb = np.interp(grid, np.sort(v_b), q_b[np.argsort(v_b)])
    dq = (qb - qa) + offset
    slope, intercept = np.polyfit(grid, dq, 1)
    n_cycles = int(df["cycle_number"].max())
    return {"dq_var": float(np.var(dq)), "dq_mean": float(np.mean(dq)),
            "dq_slope": float(slope), "dq_intercept": float(intercept),
            "dq_min": float(dq.min()), "dq_max": float(dq.max()),
            "n_cycles_total": n_cycles}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mit-dir", required=True, help="directory containing b1c*.parquet")
    ap.add_argument("--out", default="data/mit_dq_early_extracted.csv")
    args = ap.parse_args()
    rows = []
    for p in sorted(glob.glob(os.path.join(args.mit_dir, "b1c*.parquet"))):
        bid = os.path.splitext(os.path.basename(p))[0]
        f = dq_features(p)
        if f is None:
            print("skip %s (insufficient data at cycles 10/100)" % bid)
            continue
        f.update({"battery_id": bid, "cycle_10": CYCLE_A, "cycle_100": CYCLE_B,
                  "cycle_life": -1})  # EOL labels are not in the raw parquet; fill from DATA.md
        rows.append(f)
    out = pd.DataFrame(rows)[["battery_id", "cycle_10", "cycle_100", "dq_var",
                              "dq_mean", "dq_slope", "dq_intercept", "dq_min",
                              "dq_max", "cycle_life", "n_cycles_total"]]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out.to_csv(args.out, index=False, encoding="utf-8-sig")
    print("wrote %s (%d cells)" % (args.out, len(out)))


if __name__ == "__main__":
    main()
