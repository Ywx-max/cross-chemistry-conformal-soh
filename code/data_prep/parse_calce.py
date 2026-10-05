# -*- coding: utf-8 -*-
"""CALCE CS2 parsing (step 1 of T1): 8 LCO prismatic cells -> one feature row per cycle.

The raw zips on the CALCE website mix two acquisition systems with different fields and capacity conventions:
  CS2_8 / CS2_21  CADEX txt: Duration is the true time base; the Capacity column is actually
                  "depth of discharge in percent", and only times the rated capacity gives Ah (a trap we hit before)
  the other 6     Arbin xlsx: capacity from the Discharge_Capacity difference of the discharge step

ICA (incremental capacity dQ/dV) uses the 3.90-4.19 V charging window: CC-segment filter + voltage
monotonisation + 10 mV binning. Binning is not optional: at 1 s sampling the voltage oscillates
slightly, and a pointwise difference gives sawtooth fake dQ/dV whose peak positions are worthless.

Input: data/raw/calce/CS2_*.zip; output: data/calce_features.csv (with an outlier flag column).
Note parse_calce_v2.py is the unified rebuild of this script (current-integrated capacity, v_q features added);
do not mix the outputs of the two versions."""
import io, zipfile
from pathlib import Path
import numpy as np
import pandas as pd

CAL = Path("data/raw/calce")   # CS2_*/CX2_*.zip live here
OUT = Path("data")             # parsing output directory
DV = 0.010
# rated capacity differs per cell series (2026-10 CX2 extension): CS2 = 1.1 Ah, CX2 = 1.35 Ah.
# the txt path computes capacity as depth-of-discharge percent x rated capacity, so a wrong RATED skews SOH systematically
# (with 1.1 for a CX2 cell a fresh cell reads 0.815 instead of 1.0, shifting target-domain SOH down by 18.5%)
RATED_BY_PREFIX = {"CS2": 1.1, "CX2": 1.35}


def rated_for(cell):
    return RATED_BY_PREFIX[cell.split("_")[0]]

def robust_ica(v, i, t, v_lo=3.90, v_hi=4.19):
    """One charging segment -> (ICA main-peak height, peak position in V).

    Steps: CC filter (|I| >= 0.7x median; dQ/dV over the CV segment has no physical meaning) ->
    voltage monotonisation (+0.5 mV threshold removes jitter) -> 10 mV binning -> difference, take the maximum."""
    if len(t) < 30:
        return np.nan, np.nan
    thr = 0.7 * np.median(np.abs(i))
    m = np.abs(i) >= thr
    v, i, t = v[m], i[m], t[m]
    if len(v) < 8:
        return np.nan, np.nan
    dt = np.diff(t)
    if np.median(dt) <= 0:
        return np.nan, np.nan
    q = np.concatenate([[0], np.cumsum(np.abs(i[1:]) * dt)]) / 3600.0
    last = -1e9; kv, kq = [], []
    for vi, qi in zip(v, q):
        if vi >= last + 0.0005:
            kv.append(vi); kq.append(qi); last = vi
    kv, kq = np.array(kv), np.array(kq)
    nb = int((v_hi - v_lo) / DV)
    bins = np.floor((kv - v_lo) / DV).astype(int)
    ok = (bins >= 0) & (bins < nb)
    if ok.sum() < 8:
        return np.nan, np.nan
    bq = np.full(nb, np.nan)
    bs = pd.Series(kq[ok]).groupby(bins[ok]).mean()
    bq[bs.index.values.astype(int)] = bs.values
    dq = np.diff(bq) / DV
    centers_mid = (v_lo + (np.arange(nb) + 0.5) * DV)[:-1]
    mask = np.isfinite(dq)
    if mask.sum() < 4:
        return np.nan, np.nan
    k = int(np.argmax(np.where(mask, dq, -np.inf)))
    return float(dq[k]), float(centers_mid[k])

def calce_txt_full(zip_path, cell):
    """CADEX txt variant (CS2_8/21). A state machine scans line by line: a rest->charge->discharge switch is one
    round; charging buffers for the ICA, discharging records capacity. The real capacity is the depth-of-discharge column timesrated capacity."""
    cap_rows, ica_rows = [], []
    n_cyc, state = 0, 'rest'
    v_buf, i_buf, t_buf = [], [], []
    with zipfile.ZipFile(zip_path) as z:
        names = sorted(n for n in z.namelist() if n.lower().endswith('.txt'))
        for name in names:
            with z.open(name) as f:
                t = pd.read_csv(f, sep='\t', encoding='utf-8', engine='python')
            t.columns = [c.strip() for c in t.columns]
            for mv, ma, du, cp in zip(t['mV'].astype(float).values, t['mA'].astype(float).values,
                                      t['Duration'].astype(float).values, t['Capacity'].astype(float).values):
                cur = 'charge' if ma > 0 else ('discharge' if ma < 0 else 'rest')
                if cur != state:
                    if state == 'charge' and len(t_buf) >= 30:
                        pk, pv = robust_ica(np.array(v_buf)/1000, np.array(i_buf)/1000, np.array(t_buf))
                        ica_rows.append((n_cyc + 1, pk, pv, t_buf[-1]-t_buf[0]))
                    if state == 'discharge':
                        n_cyc += 1
                        cap_rows.append((n_cyc, cp))
                    state, v_buf, i_buf, t_buf = cur, [], [], []
                if cur == 'charge':
                    v_buf.append(mv); i_buf.append(ma); t_buf.append(du)
        if state == 'discharge':
            n_cyc += 1
            cap_rows.append((n_cyc, cp))
    cap = pd.DataFrame(cap_rows, columns=['cycle_id', 'depth_pct'])
    cap['capacity_Ah'] = cap['depth_pct'] / 100.0 * rated_for(cell)
    cap = cap[cap['capacity_Ah'] > 0.2]
    ica = pd.DataFrame(ica_rows, columns=['cycle_id', 'ica_peak', 'ica_peak_V', 'charge_dur_s'])
    ica = ica.dropna(subset=['ica_peak']).groupby('cycle_id').agg(
        ica_peak=('ica_peak', 'mean'), ica_peak_V=('ica_peak_V', 'mean'),
        charge_dur_s=('charge_dur_s', 'sum')).reset_index()
    m = cap.merge(ica, on='cycle_id', how='left')
    m.insert(0, 'battery_id', cell)
    m['soh'] = m['capacity_Ah'] / rated_for(cell)
    return m

def calce_xlsx_full(zip_path, cell):
    """Arbin xlsx variant (the other 6). The xlsx is grouped by (Cycle_Index, Step_Index); discharge-step capacity
    is the max-minus-min difference of Discharge_Capacity.
    The offset accumulation is critical: every xlsx file restarts Cycle_Index at 1, and without the offset the
    later files overwrite earlier cycles (the old comment "cycle number = file offset + Cycle_Index")."""
    cap_rows, ica_rows, offset = [], [], 0
    with zipfile.ZipFile(zip_path) as z:
        xls = sorted(n for n in z.namelist() if n.lower().endswith('.xlsx'))
        for name in xls:
            xf = pd.ExcelFile(io.BytesIO(z.read(name)), engine='calamine')
            for sheet in [s for s in xf.sheet_names if s.lower().startswith('channel')]:
                d = xf.parse(sheet)
                d.columns = [str(c).strip() for c in d.columns]
                sub = d[['Cycle_Index','Step_Index','Current(A)','Voltage(V)','Test_Time(s)','Discharge_Capacity(Ah)']].astype(float)
                for (cyc, step), sg in sub.groupby(['Cycle_Index','Step_Index']):
                    gid = offset + int(cyc)
                    if (sg['Current(A)'] < 0).any():
                        dcap = sg.loc[sg['Current(A)'] < 0, 'Discharge_Capacity(Ah)']
                        cap_rows.append((gid, dcap.max() - dcap.min()))
                    if (sg['Current(A)'] > 0).any():
                        c = sg[sg['Current(A)'] > 0]
                        pk, pv = robust_ica(c['Voltage(V)'].values, c['Current(A)'].values, c['Test_Time(s)'].values)
                        ica_rows.append((gid, pk, pv, float(c['Test_Time(s)'].iloc[-1] - c['Test_Time(s)'].iloc[0])))
            offset = max((r[0] for r in cap_rows), default=0)
    cap = pd.DataFrame(cap_rows, columns=['cycle_id', 'c']).groupby('cycle_id')['c'].sum().reset_index()
    cap.columns = ['cycle_id', 'capacity_Ah']
    cap = cap[cap['capacity_Ah'] > 0.2]
    # drop physically impossible capacity rows (CX2_16 has a 22.7 h rest segment where the Arbin cumulative-counter
    # difference produces a 12.5 Ah row; discharge cannot exceed 1.5x rated). Row indices stay untouched, rows are
    # only removed, so CS2 (max capacity ~1.25 Ah < 1.5x1.1) is unaffected
    _bad = cap['capacity_Ah'] > 1.5 * rated_for(cell)
    if _bad.any():
        print(f"  [physical filter] {cell}: dropped {int(_bad.sum())} rows above physical capacity "
              f"(max {cap.loc[_bad, 'capacity_Ah'].max():.2f} Ah)", flush=True)
        cap = cap[~_bad]
    ica = pd.DataFrame(ica_rows, columns=['cycle_id', 'pk', 'pv', 'dur']).dropna(subset=['pk'])
    ica = ica.groupby('cycle_id').agg(ica_peak=('pk','mean'), ica_peak_V=('pv','mean'),
                                      charge_dur_s=('dur','sum')).reset_index()
    m = cap.merge(ica, on='cycle_id', how='left')
    m.insert(0, 'battery_id', cell)
    m['soh'] = m['capacity_Ah'] / rated_for(cell)
    return m

def main():
    frames, t0 = [], time.time()
    # 2026-10 CX2 extension: include the 8 cells under the same conditions as CS2 (16/31/33/34/35/36/37/38),
    # exclude the pulse/temperature-cycling/3C-duty cells (CX2_3/4/8/32; exclusion criteria in Section 4.1)
    cells = ['CS2_8', 'CS2_21', 'CS2_33', 'CS2_34', 'CS2_35', 'CS2_36', 'CS2_37', 'CS2_38',
             'CX2_16', 'CX2_31', 'CX2_33', 'CX2_34', 'CX2_35', 'CX2_36', 'CX2_37', 'CX2_38']
    for cell in cells:
        f = calce_txt_full(CAL/f'{cell}.zip', cell) if cell in ('CS2_8','CS2_21','CX2_31') \
            else calce_xlsx_full(CAL/f'{cell}.zip', cell)
        frames.append(f)
        print(f"{cell}: {len(f)} rows | {f['capacity_Ah'].iloc[0]:.3f}->{f['capacity_Ah'].iloc[-1]:.3f} Ah | ica median {f['ica_peak'].median():.2f}", flush=True)
    cal = pd.concat(frames, ignore_index=True).rename(columns={'cycle_id': 'cycle'})
    # outlier cleaning: values outside an 11-point rolling-median +/-25% band are flagged (Section 4.1 scope);
    # cleaned capacity goes to a separate capacity_Ah_clean column while the raw column is kept for inspection
    med = cal.groupby('battery_id')['capacity_Ah'].rolling(11, center=True, min_periods=3).median()
    cal['is_outlier'] = (cal['capacity_Ah'] - med.values).abs() > 0.25 * med.values
    cal['capacity_Ah_clean'] = cal['capacity_Ah'].where(~cal['is_outlier'])
    cal['capacity_Ah_clean'] = cal.groupby('battery_id')['capacity_Ah_clean'].transform(
        lambda s: s.interpolate(limit_direction='both'))
    cal.to_csv(OUT / 'calce_features.csv', index=False, encoding='utf-8-sig')
    print(f"written: {OUT / 'calce_features.csv'} ({len(cal)} rows, {time.time()-t0:.0f}s)")

if __name__ == "__main__":
    import time  # imported here so importing this module for its functions does not start the timer
    main()
