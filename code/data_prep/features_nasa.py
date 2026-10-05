# -*- coding: utf-8 -*-
"""NASA per-cycle health features (step 2 of T1): scan the .mat files, one row per discharge cycle.

Features = capacity/SOH/discharge duration/minimum voltage + the ICA triplet of the "preceding charge cycle"
(peak, peak position, charge duration). The preceding charge rather than the same cycle, because charging
happens before discharge, so predicting the current cycle it is already in the past and leaks no future information.

The ICA convention differs slightly from CALCE: a 5 mV grid here (CALCE 10 mV) and a 0.9x CC basis
(CALCE 0.7). NASA samples sparsely, so a coarser grid blurs the peak; the CC basis is the median of the first
20 points, but if the charge starts with a rest tail the basis runs low and the CC segment turns conservative
(a known issue; the pending item is noted in the NASA feature parsing notes).

Input: data/raw/nasa/*.mat; output: data/nasa_features.csv"""
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.io as sio

RAW_DIR = Path("data/raw/nasa")
OUT_DIR = Path("data")
V_LO, V_HI, DV = 3.90, 4.19, 0.005
GRID = np.arange(V_LO, V_HI, DV)

def pick(obj, *names):
    for n in names:
        if hasattr(obj, n):
            return np.atleast_1d(np.asarray(getattr(obj, n))).astype(float)
    return None

def ica_features(v, i, t):
    """CC-filtered charging segment -> (ICA main-peak height, peak position).
    Voltage-capacity pairs are sorted by voltage before interpolating onto the 5 mV grid: the raw sequence is
    mostly monotonic but has occasional jumps, and np.interp produces junk on an unsorted series."""
    dt = np.diff(t)
    q = np.concatenate([[0], np.cumsum(np.abs(i[1:]) * dt)]) / 3600.0
    vm = (v[:-1] + v[1:]) / 2.0
    qm = (q[:-1] + q[1:]) / 2.0
    mm = (vm >= V_LO) & (vm <= V_HI)
    if mm.sum() < 50:
        return np.nan, np.nan
    vg, qg = vm[mm], qm[mm]
    order = np.argsort(vg)
    q_grid = np.interp(GRID, vg[order], qg[order])
    dqdv = np.diff(q_grid) / DV
    k = int(np.nanargmax(dqdv))
    return float(dqdv[k]), float(GRID[k])

def main():
    rows = []
    for mp in sorted(RAW_DIR.glob("B00*.mat")):
        mat = sio.loadmat(str(mp), struct_as_record=False, squeeze_me=True)
        key = next(k for k in mat if not k.startswith("__"))
        cycles = np.atleast_1d(mat[key].cycle)
        last_charge, n_dis = None, 0
        for c in cycles:
            ctype = str(c.type)
            d = c.data
            if ctype == "charge":
                v = pick(d, "voltage_measured", "Voltage_measured")
                i = pick(d, "current_measured", "Current_measured")
                t = pick(d, "time", "Time")
                if v is None or i is None or t is None:
                    last_charge = {"ica_peak": np.nan, "ica_peak_V": np.nan, "charge_dur_s": np.nan}
                else:
                    i_nom = np.median(np.abs(i[:20]))  # CC basis: median of the first 20 points (low with a rest tail; see docstring)
                    cc = np.abs(i) >= 0.9 * i_nom
                    if cc.sum() >= 50:
                        pk, pv = ica_features(v[cc], i[cc], t[cc])
                    else:
                        pk, pv = np.nan, np.nan
                    last_charge = {"ica_peak": pk, "ica_peak_V": pv, "charge_dur_s": float(t[-1])}
            elif ctype == "discharge":
                # cycle numbers re-indexed by discharge count (raw .mat indices mix in resistance-measurement cycles)
                n_dis += 1
                cap = float(pick(d, "Capacity")[0])
                t_arr = pick(d, "time", "Time")
                v_arr = pick(d, "voltage_measured", "Voltage_measured")
                rows.append({"battery_id": key, "cycle": n_dis, "capacity_Ah": cap,
                             "soh": cap / 2.0, "discharge_dur_s": float(t_arr[-1]),
                             "v_min_V": float(v_arr.min()), **(last_charge or {})})
    df = pd.DataFrame(rows)
    eol = df[df["capacity_Ah"] < 1.4].groupby("battery_id")["cycle"].min()
    df["eol_cycle"] = df["battery_id"].map(eol).fillna(-1).astype(int)
    df["rul"] = np.where(df["eol_cycle"] > 0, df["eol_cycle"] - df["cycle"], np.nan)
    out = OUT_DIR / "nasa_features.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(df.groupby("battery_id").agg(n=("cycle","max"), ica_first=("ica_peak","first"),
          ica_last=("ica_peak","last"), miss=("ica_peak", lambda s: s.isna().sum())).round(3).to_string())
    print("written:", out)

if __name__ == "__main__":
    main()
