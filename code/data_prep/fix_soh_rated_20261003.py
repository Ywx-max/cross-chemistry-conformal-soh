# -*- coding: utf-8 -*-
"""One-off fix (2026-10-03): CALCE SOH basis moves from first-cycle capacity to rated capacity.

Background: the CS2/CX2 soh was normalised by each cell's first-cycle capacity (soh always 1 on the first row),
inconsistent with the rated-capacity convention of MIT/NASA, and physically implausible soh > 1 appeared early
in life (up to 1.18 in the old tables). This script recomputes every data table containing CALCE rows as
soh = capacity_Ah / rated_for(battery_id) (CS2=1.1 Ah, CX2=1.35 Ah, matching RATED_BY_PREFIX of parse_calce.py).

Capacity columns stay untouched; only soh is recomputed; MIT/NASA rows are not modified at all.
Idempotent: a table whose CALCE rows already satisfy |soh - capacity_Ah/rated| <= 1e-9 is skipped.
Original tables are backed up to data/archive_soh_first_cycle_20261003/.

Run: python code/data_prep/fix_soh_rated_20261003.py
"""
import shutil
from pathlib import Path

import pandas as pd

import parse_calce as pc

DATA = Path("data")
BACKUP = DATA / "archive_soh_first_cycle_20261003"
TABLES = [
    "modeling_table_v3.csv.gz",
    "modeling_table_v3_cx2.csv.gz",
    "modeling_table_v3c.csv.gz",
    "modeling_table_v3c_cx2.csv.gz",
    "modeling_table_v5.csv.gz",
    "calce_full_v2_cx2.csv",
]


def is_already_rated(cal: pd.DataFrame) -> bool:
    exp = cal["capacity_Ah"] / cal["battery_id"].map(pc.rated_for)
    return (cal["soh"] - exp).abs().max() <= 1e-9


def fix_table(path: Path) -> None:
    df = pd.read_csv(path)
    if "soh" not in df.columns or "battery_id" not in df.columns:
        print(f"[skip] {path.name}: no soh/battery_id columns")
        return
    is_cal = df["battery_id"].astype(str).str.startswith(("CS2", "CX2"))
    if not is_cal.any():
        print(f"[skip] {path.name}: no CALCE rows")
        return
    cal = df[is_cal]
    if is_already_rated(cal):
        print(f"[skip] {path.name}: CALCE soh already uses the rated-capacity basis")
        return
    n_over1_old = int((cal["soh"] > 1).sum())
    BACKUP.mkdir(exist_ok=True)
    shutil.copy2(path, BACKUP / path.name)
    df.loc[is_cal, "soh"] = cal["capacity_Ah"] / cal["battery_id"].map(pc.rated_for)
    fixed = df[is_cal]
    n_over1_new = int((fixed["soh"] > 1).sum())
    df.to_csv(path, index=False,
              compression="gzip" if path.suffix == ".gz" else None)
    print(f"[fix] {path.name}: CALCE {is_cal.sum()} rows | soh>1: {n_over1_old} -> {n_over1_new}"
          f" | soh max: {cal['soh'].max():.4f} -> {fixed['soh'].max():.4f}"
          f" | MIT/NASA rows untouched ({(~is_cal).sum()} rows)")


if __name__ == "__main__":
    for name in TABLES:
        p = DATA / name
        if p.exists():
            fix_table(p)
        else:
            print(f"[missing] {name} not found, skipped")
