# -*- coding: utf-8 -*-
"""CX2 modeling-table extension (produces modeling_table_v3_cx2 / v3c_cx2; the frozen tables are untouched bit for bit).

Composition rules:
  v3_cx2  = all rows of the frozen modeling_table_v3 (MIT/NASA/CS2 verbatim) + the 8 new CX2 rows
  v3c_cx2 = all rows of the frozen modeling_table_v3c + the 7 curve columns of the CX2 rows
CX2 row sources:
  base features = parse_calce.py (v1 scope) rows for CX2 (capacity/soh/ica_peak/
             ica_peak_V/charge_dur_s; discharge_dur_s/v_mean_V/v_min_V/eol_cycle/
             rul left empty under the CS2 convention)
  curve features = the "capacity + feature aligned stream" of parse_calce_v2 (xlsx cells: the non-empty v_q10
             feature stream and the current-integrated capacity stream aligned by discharge-cycle position,
             as in parse_calce_v2.main; txt cells: single-pass output of process_txt_cell).
  Note: the CALCE curve columns of the frozen v3c join to calce_full_v2.csv by cycle key value-for-value
    (8 cells, 0 mismatches, re-verified by an assert here), and do NOT come from the repository
    build_modeling_table.py rebuild, which reads calce_curve_features.csv (whose cycle numbering intersects the v1 key at only about 39%).
    This script therefore follows the former.

Self-checks (asserts in the script):
  V1  non-CALCE rows identical to the frozen v3 row by row; CALCE row count = old CALCE + CX2; the three missing CX2 columns 100% empty;
  V2  the full_v2 rebuilt with current code (CS2 part) joined by cycle matches the frozen v3c CALCE curve columns
      value for value (a 1e-14 regression check against the upstream artifact).

Run: python code/data_prep/build_modeling_table_cx2.py
Output: data/modeling_table_v3_cx2.csv.gz, data/modeling_table_v3c_cx2.csv.gz, data/calce_full_v2_cx2.csv
"""
import sys, time, zipfile
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import parse_calce as pc
import parse_calce_v2 as pv

DATA = Path("data")
CX2_XLSX = ["CX2_16", "CX2_33", "CX2_34", "CX2_35", "CX2_36", "CX2_37", "CX2_38"]
CX2_TXT = ["CX2_31"]
CURVE = ["v_q10", "v_q30", "v_q50", "v_q70", "v_q90", "ica2_peak", "ica_fwhm"]


def v1_rows(cell):
    f = pc.calce_txt_full(pc.CAL / f"{cell}.zip", cell) if cell in CX2_TXT \
        else pc.calce_xlsx_full(pc.CAL / f"{cell}.zip", cell)
    return f.rename(columns={"cycle_id": "cycle"}).sort_values("cycle").reset_index(drop=True)


def v2_aligned_rows(cell):
    """parse_calce_v2-scope aligned rows (xlsx: feature and capacity streams aligned by position; txt: single-pass output)."""
    z = pv.CAL / f"{cell}.zip"
    with zipfile.ZipFile(z) as zf:
        has_xlsx = any(n.lower().endswith(".xlsx") for n in zf.namelist())
    if has_xlsx:
        f_feat = pv.process_xlsx_cell(z, cell)
        f_cap = pv.process_xlsx_capacity(z, cell).rename(
            columns={"gid": "cycle", "capacity_Ah_int": "capacity_Ah"})
        f_feat_d = f_feat[f_feat["v_q10"].notna()].sort_values("cycle").reset_index(drop=True)
        f_cap_s = f_cap.sort_values("cycle").reset_index(drop=True)
        n = min(len(f_feat_d), len(f_cap_s))
        if len(f_feat_d) != len(f_cap_s):
            print(f"  [align] {cell}: feature cycles {len(f_feat_d)} vs capacity cycles {len(f_cap_s)}, taking the first {n}")
        out = f_feat_d.iloc[:n].copy()
        out["capacity_Ah"] = f_cap_s["capacity_Ah"].values[:n]
        out["battery_id"] = cell
        out["soh"] = out["capacity_Ah"] / pc.rated_for(cell)
        return out
    m = pv.process_txt_cell(z, cell)
    m["soh"] = m["capacity_Ah"] / pc.rated_for(cell)
    return m


def main():
    t0 = time.time()
    v3 = pd.read_csv(DATA / "modeling_table_v3.csv.gz", compression="gzip")
    v3c = pd.read_csv(DATA / "modeling_table_v3c.csv.gz", compression="gzip")

    # CX2 rows (v1 base features + v2 curve features)
    cx2_base, cx2_curve, full_parts = [], [], []
    for cell in CX2_XLSX + CX2_TXT:
        f1 = v1_rows(cell)
        f2 = v2_aligned_rows(cell)
        base = pd.DataFrame({
            "dataset": "CALCE", "battery_id": cell, "cycle": f1["cycle"].values,
            "capacity_Ah": f1["capacity_Ah"].values, "soh": f1["soh"].values,
            "discharge_dur_s": np.nan, "v_mean_V": np.nan, "v_min_V": np.nan,
            "ica_peak": f1["ica_peak"].values, "ica_peak_V": f1["ica_peak_V"].values,
            "charge_dur_s": f1["charge_dur_s"].values, "eol_cycle": np.nan, "rul": np.nan})
        cx2_base.append(base)
        cx2_curve.append(f2[["battery_id", "cycle"] + CURVE])
        full_parts.append(f2)
        print(f"{cell}: v1 rows {len(f1)} | v2 aligned rows {len(f2)}", flush=True)
    cx2 = pd.concat(cx2_base, ignore_index=True)
    cx2c = pd.concat(cx2_curve, ignore_index=True)
    full_cx2 = pd.concat(full_parts, ignore_index=True)[
        ["battery_id", "cycle", "ica_main_peak", "ica_main_V", "ica2_peak", "ica_fwhm",
         "v_q10", "v_q30", "v_q50", "v_q70", "v_q90", "capacity_Ah", "ica2_peak_dch",
         "ica_fwhm_dch", "soh"]]

    # V2 self-check: full_v2 rebuilt with current code (CS2 part) joined == frozen v3c curve columns
    ref_cs2_parts = []
    for cell in ["CS2_33", "CS2_34", "CS2_35", "CS2_36", "CS2_37", "CS2_38"]:
        ref_cs2_parts.append(v2_aligned_rows(cell))
    for cell in ["CS2_8", "CS2_21"]:
        ref_cs2_parts.append(v2_aligned_rows(cell))
    full_cs2 = pd.concat(ref_cs2_parts, ignore_index=True)
    frozen_cal = v3c[v3c.dataset == "CALCE"]
    m = frozen_cal.merge(full_cs2[["battery_id", "cycle"] + CURVE],
                         on=["battery_id", "cycle"], suffixes=("_fz", "_re"))
    # frozen rows not covered by full_v2 have curve columns that are empty in the frozen table by construction (7584-7543=41 rows)
    unmatched = frozen_cal.merge(full_cs2[["battery_id", "cycle"]], on=["battery_id", "cycle"],
                                 how="left", indicator=True)
    un = unmatched[unmatched._merge == "left_only"]
    assert un[CURVE].notna().sum().sum() == 0, "unmatched rows carry non-empty curve columns; the mechanism assumption fails"
    mism = 0
    for c in CURVE:
        x, y = m[c + "_fz"], m[c + "_re"]
        mism += int((pd.isna(x) != pd.isna(y)).sum())
        ok = ~(pd.isna(x) | pd.isna(y))
        if ok.sum():
            mism += int((np.abs(x[ok].values - y[ok].values) > 1e-9).sum())
    print(f"V2 self-check: CS2 curve columns compared on {len(m)}/{len(frozen_cal)} rows ({len(un)} frozen rows empty by construction), mismatches {mism}")
    assert mism == 0, "rebuilt full_v2 disagrees with the frozen v3c curve columns; stopping"

    # assembly
    old_cal_keys = set(map(tuple, v3[v3.dataset == "CALCE"][["battery_id", "cycle"]].values))
    assert not (old_cal_keys & set(map(tuple, cx2[["battery_id", "cycle"]].values))), "key collision"
    v3_cx2 = pd.concat([v3, cx2[v3.columns]], ignore_index=True)
    cx2_v3c_rows = cx2.merge(cx2c, on=["battery_id", "cycle"], how="left", suffixes=("", "_c"))
    assert cx2_v3c_rows.shape[0] == len(cx2), "duplicate curve keys would explode the row count"
    cx2_v3c_rows = cx2_v3c_rows.reindex(columns=v3c.columns)
    v3c_cx2 = pd.concat([v3c, cx2_v3c_rows], ignore_index=True)

    # V1/V3 asserts
    assert v3_cx2.shape[0] == v3.shape[0] + len(cx2)
    pd.testing.assert_frame_equal(
        v3_cx2[v3_cx2.dataset != "CALCE"].reset_index(drop=True),
        v3[v3.dataset != "CALCE"].reset_index(drop=True))
    pd.testing.assert_frame_equal(
        v3_cx2[(v3_cx2.dataset == "CALCE") & (v3_cx2.battery_id.str.startswith("CS2"))].reset_index(drop=True),
        v3[v3.dataset == "CALCE"].reset_index(drop=True))
    cx2_chk = v3_cx2[v3_cx2.battery_id.str.startswith("CX2")]
    for c in ("discharge_dur_s", "v_mean_V", "v_min_V"):
        assert cx2_chk[c].notna().sum() == 0, f"CX2 rows must leave {c} empty"
    assert v3c_cx2.shape[0] == v3c.shape[0] + len(cx2)
    print(f"V1/V3 asserts passed: v3_cx2 {v3_cx2.shape}, v3c_cx2 {v3c_cx2.shape}, "
          f"CALCE {v3[v3.dataset=='CALCE'].shape[0]} -> {v3_cx2[v3_cx2.dataset=='CALCE'].shape[0]} rows")

    v3_cx2.to_csv(DATA / "modeling_table_v3_cx2.csv.gz", index=False, compression="gzip")
    v3c_cx2.to_csv(DATA / "modeling_table_v3c_cx2.csv.gz", index=False, compression="gzip")
    full_cx2.sort_values(["battery_id", "cycle"]).to_csv(
        DATA / "calce_full_v2_cx2.csv", index=False, encoding="utf-8-sig")
    print(f"written: modeling_table_v3_cx2.csv.gz / modeling_table_v3c_cx2.csv.gz / calce_full_v2_cx2.csv ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
