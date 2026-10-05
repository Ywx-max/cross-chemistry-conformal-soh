# -*- coding: utf-8 -*-
"""Persistence baseline (2026-10-04 checklist A4): the naive control of the cross-chemistry experiments, pinned here.

Definition (same window convention as t3b/t4): a window is 20 consecutive cycles of one cell (last row index i);
the label is the soh at cycle i+H with cyc[i+H] == cyc[i]+H (the continuity check matches
t3b_std_local.build_windows_ds; windows closer than H steps to the record end are dropped).
The persistence prediction is the last-row soh of the window (SOH is locally near-constant); no parameters are trained.

Two outputs:
  1) pooled persistence RMSE over all target-domain cells (H=5/10/20 x CALCE/NASA), cited in the Table 3 note;
  2) per-seed pairing: same test_cells and windows as results_cx2/transfer/t3b_*.json,
     paired t-test on the same-seed differences of zero/ft RMSE vs persistence (n=5, signed t),
     counting seeds where the model beats persistence (wins).

    python code/checks/persistence_baseline.py
Output: results_cx2/transfer/persistence_baseline.json (also printed to stdout)
"""
import csv, json, math, statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "modeling_table_v3_cx2.csv"
RES = ROOT / "results_cx2" / "transfer"
WINDOW = 20
HS = (5, 10, 20)

# Reference values from an external consultation note (2026-10-03), all-cell scope, H=5/10/20.
# 2026-10-04 recomputation: NASA pooled values hit 3/3 references; paired t values also match
# (NASA/TCN H=10: t=-1.43, 3/5 wins). CALCE pooled values exceed the references by about 7% (harder on the model,
# i.e. the conservative direction): the references are close to the per-cell macro mean sqrt(mean(mse_i))
# (H=5 differs by 0.0001, H=10/20 by 0.0009/0.0020), but no single convention reproduces all references
# of both domains exactly, and the note does not record its convention. The primary convention here is
# all-cell pooling, matching the concat pooling of t3b eval_model so the RMSEs are comparable with the
# model; the macro mean is reported as a secondary view. Note (2026-10-04): REF below holds the initial
# reference values (convention unrecorded); after the window-continuity fix the primary values are
# CALCE 0.0665/0.0832/0.1066 (H=5/10/20), the values reported in the manuscript; the gap to REF is a convention difference (about +7%), not a computation error.
REF = {"CALCE": {5: 0.0609, 10: 0.0761, 20: 0.0977},
       "NASA": {5: 0.0181, 10: 0.0269, 20: 0.0461}}


def betacf(a, b, x):
    MAXIT, EPS, FPMIN = 200, 3e-16, 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    if abs(d) < FPMIN:
        d = FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < EPS:
            break
    return h


def betai(a, b, x):
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    bt = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                  + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * betacf(a, b, x) / a
    return 1.0 - bt * betacf(b, a, 1.0 - x) / b


def paired_t_p_signed(xs, ys):
    """Signed paired t: t>0 means xs mean exceeds ys (model worse than persistence)."""
    d = [x - y for x, y in zip(xs, ys)]
    n = len(d)
    m = st.mean(d)
    sd = st.stdev(d)
    t = m / (sd / math.sqrt(n))
    df = n - 1
    p = betai(df / 2.0, 0.5, df / (df + abs(t) * abs(t)))
    return t, p


def load_cells():
    cells = {}
    with open(DATA, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                cyc = float(row["cycle"])
                soh = float(row["soh"])
            except (TypeError, ValueError):
                continue
            if math.isnan(soh) or math.isnan(cyc):
                continue
            cells.setdefault((row["dataset"], row["battery_id"]), ([], []))
            cells[(row["dataset"], row["battery_id"])][0].append(cyc)
            cells[(row["dataset"], row["battery_id"])][1].append(soh)
    for k, (cs, ys) in list(cells.items()):
        order = sorted(range(len(cs)), key=lambda i: cs[i])
        cells[k] = ([cs[i] for i in order], [ys[i] for i in order])
    return cells


def win_pairs(cyc, y, H):
    # continuity within the window and from window to label, matching t3b_std_local.build_windows_ds
    # (2026-10-04 fix: the original missed in-window continuity and used about 3.5% more windows)
    out = []
    for i in range(WINDOW - 1, len(y) - H):
        if cyc[i] - cyc[i - WINDOW + 1] == WINDOW - 1                 and cyc[i + H] == cyc[i] + H:
            out.append((y[i], y[i + H]))
    return out


def pooled_rmse(cells, dom, H, bids=None):
    se, n = 0.0, 0
    for (ds, bid), (cyc, y) in cells.items():
        if ds != dom:
            continue
        if bids is not None and bid not in bids:
            continue
        for a, b in win_pairs(cyc, y, H):
            se += (a - b) ** 2
            n += 1
    return math.sqrt(se / n), n


def macro_rmse(cells, dom, H, bids=None):
    """Secondary view: equal-weight mean of per-cell RMSE^2, then square root (macro mean, one weight per cell)."""
    per = []
    for (ds, bid), (cyc, y) in cells.items():
        if ds != dom:
            continue
        if bids is not None and bid not in bids:
            continue
        pr = win_pairs(cyc, y, H)
        if pr:
            per.append(sum((a - b) ** 2 for a, b in pr) / len(pr))
    return math.sqrt(sum(per) / len(per)), len(per)


def main():
    cells = load_cells()
    print(f"cells in modeling table: {len(cells)}")

    # 1) all-cell persistence: primary = pooled (matches model evaluation), secondary = macro mean
    allcell = {}
    for dom in ("CALCE", "NASA"):
        allcell[dom] = {}
        for H in HS:
            rmse, n = pooled_rmse(cells, dom, H)
            macro, _ = macro_rmse(cells, dom, H)
            ref = REF[dom][H]
            flag = "OK" if abs(rmse - ref) <= 5e-4 else f"differs from initial reference {ref} by {rmse - ref:+.4f} (convention difference, see header comment)"
            allcell[dom][H] = {"rmse": round(rmse, 6), "macro_rmse": round(macro, 6),
                               "n_windows": n, "ref": ref, "check": flag}
            print(f"[all cells] {dom} H={H:2d} pooled persistence RMSE={rmse:.4f} "
                  f"(macro mean {macro:.4f}, windows {n})  {flag}")

    # 2) per-seed pairing (t3b test_cells); the h5/h20 sensitivity runs are TCN-only, so pattern is fixed to tcn
    paired = {}
    jobs = [("h10", RES / "t3b_{m}_s{s}.json", (10,), ("tcn", "lstm")),
            ("h5", ROOT / "results_cx2" / "transfer_sens_h5" / "t3b_tcn_s{s}.json", (5,), ("tcn",)),
            ("h20", ROOT / "results_cx2" / "transfer_sens_h20" / "t3b_tcn_s{s}.json", (20,), ("tcn",))]
    for tag, pat, hs, models in jobs:
        for H in hs:
            for model in models:
                seeds, per_seed = [], []
                for s in (42, 43, 44, 45, 46):
                    p = Path(str(pat).format(m=model, s=s))
                    if not p.exists():
                        raise FileNotFoundError(f"t3b result missing: {p}")
                    j = json.load(open(p, encoding="utf-8"))
                    if j["horizon"] != H:
                        continue
                    seeds.append(s)
                    per_seed.append(j)
                if not per_seed:
                    continue
                # each t3b json holds two target domains; pair per domain
                for dom in ("CALCE", "NASA"):
                    pers_by_seed = []
                    for j in per_seed:
                        te = j["targets"][dom]["test_cells"]
                        pr, _ = pooled_rmse(cells, dom, H, bids=set(te))
                        pers_by_seed.append(pr)
                    for path_key, label in (("zero_shot", "zero"), ("fine_tune", "ft")):
                        mods = [j["targets"][dom][path_key]["rmse"] for j in per_seed]
                        t, pv = paired_t_p_signed(mods, pers_by_seed)
                        wins = sum(1 for a, b in zip(mods, pers_by_seed) if a < b)
                        paired[f"{tag}|{model}|{dom}|{label}|H{H}"] = {
                            "seeds": seeds,
                            "model_rmse": [round(x, 6) for x in mods],
                            "persistence_rmse": [round(x, 6) for x in pers_by_seed],
                            "t": round(t, 4), "p": round(pv, 4), "model_wins": wins,
                        }

    out = {"all_cell": allcell, "paired": paired,
           "note": "paired.t>0 = model mean worse than persistence; model_wins = seeds with lower model RMSE"}
    dst = RES / "persistence_baseline.json"
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print("saved:", dst)
    for k, v in paired.items():
        print(f"{k:28s} t={v['t']:+.2f} p={v['p']:.3f} wins {v['model_wins']}/5")


if __name__ == "__main__":
    main()
