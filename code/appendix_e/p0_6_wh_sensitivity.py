# -*- coding: utf-8 -*-
"""P0-6: W/H sensitivity (TCN + LSTM).
Part A: window sensitivity W in {10, 30, 40} (W = 20 is the main-table baseline), TCN, H = 10, seeds 42-46,
        both target domains - source models retrained per W (protocol as in Section 3.2, cache p0_6_src_cache).
Part B: horizon sensitivity for the LSTM: H in {5, 20} (H = 10 is the main-table baseline; TCN H5/H20 already
        exist as transfer_sens_h5/h20), LSTM, W = 20, seeds 42-46, both target domains.
Output: results/p0_6_wh_sensitivity.csv (per-seed ft_rmse / zero_rmse); logs/p0_6.log
Run: python p0_6_wh_sensitivity.py (cwd = repository root)
"""
import csv, io, os, sys, time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from p0_2_alpha_sweep import (DATA, SRC_CACHE, build_windows_ds, concat_cells,
                              std_with, set_seed, new_model, fit_model, predict)

P0 = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC6 = os.path.join(P0, "results_p0", "p0_6_src_cache")
os.makedirs(SRC6, exist_ok=True)
LOG = open(os.path.join(P0, "logs_p0", "p0_6.log"), "w", encoding="utf-8")


def log(m):
    print(m, flush=True); LOG.write(m + "\n"); LOG.flush()


def run_config(df, model, W, H, seed, device, tag):
    """One configuration: source model (cached per W/H) + fine-tuning + test; returns per-domain (ft_rmse, zero_rmse)."""
    out = {}
    cache_p = os.path.join(SRC6, f"src_{model}_w{W}_s{seed}_ep120_h{H}.pt")
    src = build_windows_ds(df, "MIT", window=W, horizon=H)
    src_bids = sorted(src)
    import random
    random.Random(seed).shuffle(src_bids)
    n_va = max(1, int(len(src_bids) * 0.1))
    Xtr, ytr, _ = concat_cells(src, src_bids[:-n_va])
    Xva, yva, _ = concat_cells(src, src_bids[-n_va:])
    sc_src = StandardScaler().fit(Xtr.reshape(-1, Xtr.shape[2]))
    Xtr_s, Xva_s = std_with(sc_src, Xtr), std_with(sc_src, Xva)
    set_seed(seed)
    sm = new_model(model, Xtr.shape[2]).to(device)
    if os.path.exists(cache_p):
        sm.load_state_dict(torch.load(cache_p, map_location=device, weights_only=True))
        log(f"  [{tag}] src cache hit {os.path.basename(cache_p)}")
    else:
        t0 = time.time()
        sm = fit_model(sm, Xtr_s, ytr, 120, seed, device, Xva_s, yva)
        torch.save(sm.state_dict(), cache_p)
        log(f"  [{tag}] src trained {time.time()-t0:.0f}s")
    src_cal = np.abs(predict(sm, Xva_s, device) - yva)
    q_src = float(np.sort(src_cal)[min(len(src_cal) - 1, int(np.ceil((len(src_cal) + 1) * 0.9)) - 1)])
    for tgt_name, split in (("CALCE", (8, 8)), ("NASA", (2, 2))):
        tgt = build_windows_ds(df, tgt_name, window=W, horizon=H)
        tb = sorted(tgt)
        import random
        random.Random(seed).shuffle(tb)
        n_ft, n_te = split
        ft_b, te_b = tb[:n_ft], tb[n_ft:]
        Xall_t, _, _ = concat_cells(tgt, tb)
        sc_tgt = StandardScaler().fit(Xall_t.reshape(-1, Xall_t.shape[2]))
        # zero-shot
        Xte, yte, _ = concat_cells(tgt, te_b)
        pred_zs = predict(sm, std_with(sc_tgt, Xte), device)
        zero_rmse = float(np.sqrt(np.mean((pred_zs - yte) ** 2)))
        # fine-tuning (transfer protocol: fine-tune/test halves, no calibration set)
        Xft, yft, _ = concat_cells(tgt, ft_b)
        ft = new_model(model, Xtr.shape[2]).to(device)
        ft.load_state_dict(sm.state_dict())
        ft = fit_model(ft, std_with(sc_tgt, Xft), yft, 60, seed, device, lr=3e-4)
        pred_ft = predict(ft, std_with(sc_tgt, Xte), device)
        ft_rmse = float(np.sqrt(np.mean((pred_ft - yte) ** 2)))
        out[tgt_name] = (ft_rmse, zero_rmse)
    return out


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"P0-6 device={device}")
    df = pd.read_csv(DATA)
    rows = []
    t_all = time.time()
    # Part A: W sensitivity (TCN, H=10, W in {10,30,40})
    for W in (10, 30, 40):
        for seed in (42, 43, 44, 45, 46):
            t0 = time.time()
            out = run_config(df, "tcn", W, 10, seed, device, f"A/W{W}/s{seed}")
            for dom, (ft_r, zs_r) in out.items():
                rows.append(dict(part="A_window", model="tcn", W=W, H=10, seed=seed,
                                 domain=dom, ft_rmse=round(ft_r, 5), zero_rmse=round(zs_r, 5)))
            log(f"[A W{W} s{seed}] {time.time()-t0:.0f}s {out}")
    # Part B: H for the LSTM (H in {5,20})
    for H in (5, 20):
        for seed in (42, 43, 44, 45, 46):
            t0 = time.time()
            out = run_config(df, "lstm", 20, H, seed, device, f"B/H{H}/s{seed}")
            for dom, (ft_r, zs_r) in out.items():
                rows.append(dict(part="B_horizon_lstm", model="lstm", W=20, H=H, seed=seed,
                                 domain=dom, ft_rmse=round(ft_r, 5), zero_rmse=round(zs_r, 5)))
            log(f"[B H{H} s{seed}] {time.time()-t0:.0f}s {out}")
    with open(os.path.join(P0, "results_p0", "p0_6_wh_sensitivity.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    log(f"P0-6 done in {time.time()-t_all:.0f}s, {len(rows)} rows -> p0_6_wh_sensitivity.csv")
    import numpy as np
    log("\n=== summary (5-seed mean+/-std ft_rmse) ===")
    for part in ("A_window", "B_horizon_lstm"):
        for model in ("tcn", "lstm"):
            for W, H in ((10, 10), (30, 10), (40, 10), (20, 5), (20, 20)):
                sel = [r for r in rows if r["part"] == part and r["model"] == model
                       and r["W"] == W and r["H"] == H]
                if not sel:
                    continue
                for dom in ("CALCE", "NASA"):
                    v = [r["ft_rmse"] for r in sel if r["domain"] == dom]
                    if v:
                        log(f"  {model} W={W} H={H} {dom:6s} ft {np.mean(v):.4f}±{np.std(v, ddof=1):.4f}")


if __name__ == "__main__":
    main()
