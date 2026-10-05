# -*- coding: utf-8 -*-
"""Modeling-table build: three per-dataset feature tables -> one uniformly named modeling table (input of all downstream experiments).

The merge rule is "align column names + tag dataset + concatenate vertically". Datasets offer different quantities, so
missing columns are left empty rather than zero-filled (downstream column-wise use skips NaNs automatically):

  NASA : nasa_features.csv directly (richest fields, but no mean discharge voltage)
  MIT  : mit_capacity.csv (capacity/duration/SOH/EOL) + mit_features.csv (voltage statistics and ICA)
         aligned by (battery, cycle); the MIT raw data give no charge-segment duration, so charge_dur_s stays empty
  CALCE: calce_features.csv (capacity from the cleaned column; this dataset has no discharge duration or voltage
         statistics, and EOL labels do not apply either, as the CALCE cells never reach end of life, so eol_cycle/rul stay empty)

v3c = v3 + 7 discharge-curve columns (v_q10...v_q90, ICA secondary peak ica2_peak, FWHM ica_fwhm);
the ablation uses this version. In the curve tables the CALCE key is cycle_id and NASA columns carry a _dch
suffix; both are unified here.

Input: data/ mit_capacity.csv, mit_features.csv, calce_features.csv, nasa_features.csv;
       for v3c also mit_curve_features.csv, calce_curve_features.csv, nasa_curve_features.csv
Output: data/modeling_table_v3.csv, data/modeling_table_v3c.csv

    python build_modeling_table.py

Data-version note: the v3 and v5 tables differ in parsing conventions (see results/README.md), so a rebuild by
this script under the documented rules may differ row-by-row from the manuscript's tables in individual cell
outliers/EOL labels. The two tables actually used by the manuscript are shipped
(data/modeling_table_v3.csv.gz, data/modeling_table_v3c.csv.gz); unzipped they substitute this script's output directly.
"""
from pathlib import Path
import pandas as pd

# as in the other scripts: data lives under data/ (the modeling table is written back there too)
DATA = Path("data")
OUT = DATA

FEATS = ["capacity_Ah", "soh", "discharge_dur_s", "v_mean_V", "v_min_V",
         "ica_peak", "ica_peak_V", "charge_dur_s", "eol_cycle", "rul"]
CURVE = ["v_q10", "v_q30", "v_q50", "v_q70", "v_q90", "ica2_peak", "ica_fwhm"]


def read(name):
    return pd.read_csv(DATA / name)


def mit_part():
    cap = read("mit_capacity.csv")
    feat = read("mit_features.csv")
    df = cap.merge(feat, on=["battery_id", "cycle"], how="left", suffixes=("", "_f"))
    df["v_mean_V"] = df["v_mean_V"]
    df["v_min_V"] = df["v_min_V"]
    df["ica_peak"] = df["ica_peak"]
    df["ica_peak_V"] = df["ica_peak_V"]
    df["charge_dur_s"] = pd.NA          # MIT has no charge-segment duration
    return df


def calce_part():
    # capacity from the capacity_Ah column (the raw parsed value under the same convention as the other datasets; capacity_Ah_clean is kept for comparison only)
    df = read("calce_features.csv")
    for col in ["discharge_dur_s", "v_mean_V", "v_min_V", "eol_cycle", "rul"]:
        df[col] = pd.NA
    return df


def nasa_part():
    df = read("nasa_features.csv")
    df["v_mean_V"] = pd.NA              # NASA gives no mean discharge voltage
    return df


def build_v3():
    parts = []
    for name, df in [("NASA", nasa_part()), ("MIT", mit_part()), ("CALCE", calce_part())]:
        df = df.copy()
        df.insert(0, "dataset", name)
        parts.append(df[["dataset", "battery_id", "cycle"] + FEATS])
    return pd.concat(parts, ignore_index=True)


def curve_part(dataset):
    """The 7 curve columns; key/column names of the three tables are aligned here"""
    if dataset == "MIT":
        df = read("mit_curve_features.csv")
    elif dataset == "CALCE":
        df = read("calce_curve_features.csv").rename(columns={"cycle_id": "cycle"})
    else:
        df = read("nasa_curve_features.csv").rename(
            columns={"ica2_peak_dch": "ica2_peak", "ica_fwhm_dch": "ica_fwhm"})
    return df[["battery_id", "cycle"] + CURVE]


def build_v3c(v3):
    parts = []
    for name in ["NASA", "MIT", "CALCE"]:
        parts.append(curve_part(name).assign(dataset=name))
    curves = pd.concat(parts, ignore_index=True)
    return v3.merge(curves, on=["dataset", "battery_id", "cycle"], how="left")


if __name__ == "__main__":
    v3 = build_v3()
    v3.to_csv(OUT / "modeling_table_v3.csv", index=False)
    v3c = build_v3c(v3)
    v3c.to_csv(OUT / "modeling_table_v3c.csv", index=False)
    print("modeling_table_v3.csv :", v3.shape)
    print("modeling_table_v3c.csv:", v3c.shape)
    print("dataset row counts:", v3["dataset"].value_counts().to_dict())
