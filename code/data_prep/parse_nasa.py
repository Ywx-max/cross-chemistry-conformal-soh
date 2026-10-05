# -*- coding: utf-8 -*-
"""NASA PCoE parsing (T1): 4 LCO 18650 cells (B0005/6/7/0018) -> capacity fade series.

NASA .mat structures are famously inconsistent across versions: a field is time in one and Time in another.
pick() takes the first existing candidate name in order, saving a pile of case branches.
EOL is set at 70% of the rated 2.0 Ah (1.4 Ah). Note this differs from the 80% of CALCE/MIT,
a retirement criterion from the dataset itself; Section 3.1 documents it.

This script yields capacity-level series only (nasa_capacity.csv + a fade plot);
cycle-level features (with ICA) live in features_nasa.py, split in two steps because ICA parsing is much slower
and tuning feature parameters should not rescan the .mat files every time.

Input: data/raw/nasa/*.mat; output: data/nasa_capacity.csv, results/nasa_degradation.png"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as sio

RAW_DIR = Path("data/raw/nasa")
OUT_DIR = Path("data")
FIG_DIR = Path("results")
RATED_AH = 2.0
EOL_AH = RATED_AH * 0.70

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

def pick(obj, *names):
    """First existing candidate field name (.mat field casing differs across batches)."""
    for n in names:
        if hasattr(obj, n):
            return np.atleast_1d(np.asarray(getattr(obj, n))).astype(float)
    return None

def parse_battery(mat_path):
    mat = sio.loadmat(str(mat_path), struct_as_record=False, squeeze_me=True)
    key = next(k for k in mat if not k.startswith("__"))  # skip metadata keys like __header__
    cycles = np.atleast_1d(mat[key].cycle)
    rows = []
    for i, c in enumerate(cycles):
        if str(c.type) != "discharge":
            continue
        d = c.data
        cap_f = pick(d, "Capacity")
        t_amb = pick(c, "ambient_temperature", "Ambient_Temperature")
        t_arr = pick(d, "time", "Time")
        v_arr = pick(d, "voltage_measured", "Voltage_measured")
        cap = float(cap_f[0]) if cap_f is not None else np.nan
        rows.append({"battery_id": key, "cycle": len(rows)+1, "cycle_idx_raw": i,
                     "capacity_Ah": cap, "soh": cap / RATED_AH,
                     "ambient_T_C": float(t_amb[0]) if t_amb is not None else np.nan,
                     "discharge_dur_s": float(t_arr[-1]) if t_arr is not None else np.nan,
                     "v_min_V": float(v_arr.min()) if v_arr is not None else np.nan})
    return pd.DataFrame(rows)

def main():
    mats = sorted(RAW_DIR.glob("B00*.mat"))
    if not mats:
        raise SystemExit(f"no .mat files found: {RAW_DIR}")
    df = pd.concat([parse_battery(m) for m in mats], ignore_index=True)
    # EOL = first cycle below 1.4 Ah; RUL = EOL cycle - current cycle (NaN until EOL is reached,
    # and the later RUL-scope experiments skip such cells automatically)
    eol_map = df[df["capacity_Ah"] < EOL_AH].groupby("battery_id")["cycle"].min()
    df["eol_cycle"] = df["battery_id"].map(eol_map).fillna(-1).astype(int)
    df["rul"] = np.where(df["eol_cycle"] > 0, df["eol_cycle"] - df["cycle"], np.nan)
    out_csv = OUT_DIR / "nasa_capacity.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(df.groupby("battery_id").agg(cycles=("cycle","max"),
          cap_first=("capacity_Ah","first"), cap_last=("capacity_Ah","last"),
          eol_cycle=("eol_cycle","first")).to_string())
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    for bid, g in df.groupby("battery_id"):
        ax.plot(g["cycle"], g["capacity_Ah"], marker=".", ms=3, lw=1, label=bid)
    ax.axhline(EOL_AH, color="r", ls="--", lw=1, label="EOL 1.4 Ah")
    ax.set_xlabel("Cycle index"); ax.set_ylabel("Discharge capacity (Ah)")
    ax.set_title("NASA PCoE capacity degradation")
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "nasa_degradation.png")
    print("written:", out_csv)

if __name__ == "__main__":
    main()
