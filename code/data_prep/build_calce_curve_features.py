# -*- coding: utf-8 -*-
"""Rebuild calce_curve_features.csv: the curve-feature input of build_modeling_table.py.

Background: the curve14 ablation of Table 6 depends on calce_curve_features.csv, but the driver that produced
it was not in the repository (only the archived reference artifact). This script rebuilds the same artifact
from the existing feature functions of parse_calce_v2
(ica_curve/ica_shape/dch_curve_feats/process_xlsx_cell/process_txt_cell): every cell runs the feature stream
(WITHOUT capacity alignment, keeping cycles whose v_q is empty, which is where this CSV differs in row count from calce_full_v2.csv),
with the column renamed cycle->cycle_id for build_modeling_table to rename back.

Columns (matching the 2026-09 reference artifact column by column):
  battery_id, cycle_id, ica2_peak, ica_fwhm, ica_main_V, ica_main_peak,
  v_q10, v_q30, v_q50, v_q70, v_q90
where ica_main_* / ica2_peak / ica_fwhm come from the charging-segment ICA (3.90-4.19 V window) and
v_q* from the discharge segment (fixed-quantile capacity-voltage points over 2.0-3.5 V).

Run: python code/data_prep/build_calce_curve_features.py --cells cs2 [--out data/calce_curve_features.csv]
      --cells cs2 = the 8 CS2 cells only (for regression against the reference artifact); --cells full = CS2 + CX2, 16 cells
Output: data/calce_curve_features.csv"""
import argparse
import time
from pathlib import Path

import pandas as pd

from parse_calce_v2 import process_xlsx_cell, process_txt_cell, CAL

CURVE_COLS = ["ica2_peak", "ica_fwhm", "ica_main_V", "ica_main_peak",
              "v_q10", "v_q30", "v_q50", "v_q70", "v_q90"]
XLSX_CELLS = ["CS2_33", "CS2_34", "CS2_35", "CS2_36", "CS2_37", "CS2_38",
              "CX2_16", "CX2_33", "CX2_34", "CX2_35", "CX2_36", "CX2_37", "CX2_38"]
TXT_CELLS = ["CS2_8", "CS2_21", "CX2_31"]


def cell_zip(cell):
    p = CAL / f"{cell}.zip"
    if not p.exists():
        raise FileNotFoundError(f"raw zip missing: {p}")
    return p


def build(cells):
    frames = []
    for cell in cells:
        z = cell_zip(cell)
        # format detection: a zip containing .xlsx is the Arbin path (in the mixed CX2 zips a .txt member is
        # a special-test log, not evidence of a CADEX txt cell; CX2_16/33 both carry txt logs)
        import zipfile
        with zipfile.ZipFile(z) as zf:
            has_xlsx = any(n.lower().endswith(".xlsx") for n in zf.namelist())
        f = process_xlsx_cell(z, cell) if has_xlsx else process_txt_cell(z, cell)
        frames.append(f)
        print(f"{cell}: {len(f)} rows", flush=True)
    cal = pd.concat(frames, ignore_index=True).rename(columns={"cycle": "cycle_id"})
    cal = cal.sort_values(["battery_id", "cycle_id"]).reset_index(drop=True)
    return cal[["battery_id", "cycle_id"] + CURVE_COLS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", choices=["cs2", "full"], default="cs2",
                    help="cs2 = the 8 CS2 cells only (for regression against the 2026-09 reference artifact); full = CS2+CX2, 16 cells")
    ap.add_argument("--out", default="data/calce_curve_features.csv")
    args = ap.parse_args()
    t0 = time.time()
    cells = XLSX_CELLS + TXT_CELLS if args.cells == "full" else \
        [c for c in XLSX_CELLS + TXT_CELLS if c.startswith("CS2")]
    cal = build(sorted(cells))
    cal.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"written {args.out} ({len(cal)} rows, {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
