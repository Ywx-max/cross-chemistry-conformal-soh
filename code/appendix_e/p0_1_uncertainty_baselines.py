# -*- coding: utf-8 -*-
"""P0-1: uncertainty-baseline comparison (MC dropout / deep ensemble / quantile regression vs target-domain conformal).

Scope: TCN backbone only (the transfer backbone of the paper), both target domains CALCE/NASA, seeds 42-46 (splits and
fine-tuning protocol identical to t4_conformal_local.py v1.2.4; source models from results_cx2/src_cache; MC/QR need
new source models under the Section 3.2 training protocol, cached under p0_1_src_cache/).
Methods and interval construction:
  MC dropout      Dropout(0.1) after each TCN block, standard training protocol; inference samples T=30 in train mode,
                  interval = mean +/- z_{1-alpha/2}*std (Gaussian approximation, alpha=0.10)
  Deep ensemble   seed-42 split, 5 initialisations (init_seed 1-5) fine-tuned from one source model,
                  interval = member mean +/- z_{1-alpha/2}*member std (single-split scope, for cost reasons)
  Quantile reg.   two-head TCN (tau=0.1/0.9), mean-pinball loss; interval = [q_lo, q_hi] directly
  Conformal       the alpha=0.10 target route of the P0-2 recomputation batch (same splits and seeds, referenced directly)
Metrics: PICP / MPIW / NMPIW / Winkler / pinball(alpha/2, 1-alpha/2) / point RMSE (one shared metric_interval).
Output: results/p0_1_uncertainty.csv; logs/p0_1.log
Run: python p0_1_uncertainty_baselines.py (cwd = repository root)
"""
import csv, io, os, sys, time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler

from p0_2_alpha_sweep import (OSS, DATA, SRC_CACHE, OUT_RES, OUT_LOG,
                              build_windows_ds, concat_cells, std_with, set_seed,
                              CausalBlock, TCN, new_model, predict)

P0 = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC1 = os.path.join(P0, "results_p0", "p0_1_src_cache")
os.makedirs(SRC1, exist_ok=True)
LOG = open(os.path.join(P0, "logs_p0", "p0_1.log"), "w", encoding="utf-8")


def log(m):
    print(m, flush=True); LOG.write(m + "\n"); LOG.flush()


class TCN_MC(nn.Module):
    """TCN with Dropout(0.1) after each block (MC dropout baseline)."""
    def __init__(self, input_dim, d=64, layers=4, p=0.1):
        super().__init__()
        ch = [input_dim] + [d] * layers
        self.blocks = nn.ModuleList([CausalBlock(ch[i], ch[i+1], 3, 2**i) for i in range(layers)])
        self.drop = nn.Dropout(p)
        self.fc = nn.Linear(d, 1)
    def forward(self, x):
        h = x.transpose(1, 2)
        for blk in self.blocks:
            h = self.drop(blk(h))
        return self.fc(h[:, :, -1]).squeeze(-1)


class TCN_QR(nn.Module):
    """Two-head TCN quantile regression (tau=0.1 / 0.9)."""
    def __init__(self, input_dim, d=64, layers=4):
        super().__init__()
        ch = [input_dim] + [d] * layers
        self.blocks = nn.ModuleList([CausalBlock(ch[i], ch[i+1], 3, 2**i) for i in range(layers)])
        self.fc = nn.Linear(d, 2)
    def forward(self, x):
        h = x.transpose(1, 2)
        for blk in self.blocks:
            h = blk(h)
        return self.fc(h[:, :, -1])


def fit_generic(model, Xtr, ytr, epochs, seed, device, Xva, yva, lr, lossf, out_dim=1):
    """Parameterised-loss version of fit_model (protocol identical: Adam, batch 256, clip 1.0, patience 10)."""
    set_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ds = torch.utils.data.TensorDataset(torch.tensor(Xtr, dtype=torch.float32),
                                        torch.tensor(ytr, dtype=torch.float32))
    dl = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=True)
    best, best_state, patience = float("inf"), None, 0
    for ep in range(epochs):
        model.train()
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            out = model(xb)
            lossf(out, yb).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        model.eval()
        with torch.no_grad():
            out = model(torch.tensor(Xva, dtype=torch.float32).to(device))
            va = lossf(out, torch.tensor(yva, dtype=torch.float32).to(device)).item()
        if np.isfinite(va) and va < best - 1e-6:
            best, patience = va, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= 10:
                break
    if best_state:
        model.load_state_dict(best_state)
    return model


def metric_interval(y, lo, hi, alpha):
    y = np.asarray(y, float); lo = np.asarray(lo, float); hi = np.asarray(hi, float)
    picp = float(np.mean((y >= lo) & (y <= hi)))
    mpiw = float(np.mean(hi - lo))
    nmpiw = mpiw / float(np.mean(np.abs(y)))
    viol = np.maximum(lo - y, 0) + np.maximum(y - hi, 0)
    winkler = float(np.mean((hi - lo) + (2.0 / alpha) * viol))
    tau_l, tau_u = alpha / 2.0, 1 - alpha / 2.0
    pl = np.maximum(tau_l * (y - lo), (tau_l - 1) * (y - lo))
    pu = np.maximum(tau_u * (y - hi), (tau_u - 1) * (y - hi))
    pinball = float(0.5 * (np.mean(pl) + np.mean(pu)))
    return picp, mpiw, nmpiw, winkler, pinball


def predict_mc(model, X, device, T=30):
    """MC dropout: sample T times in train mode (dropout active)."""
    model.train()
    xt = torch.tensor(X, dtype=torch.float32).to(device)
    outs = []
    with torch.no_grad():
        for _ in range(T):
            outs.append(model(xt).cpu().numpy())
    return np.stack(outs)   # (T, n)


def get_split_data(df, model_name, seed, device):
    """Reproduce the t4 pipeline: source cache load, split, fine-tuning (standard MSE); returns the fine-tuned model and window data."""
    src = build_windows_ds(df, "MIT", horizon=10)
    src_bids = sorted(src)
    import random
    random.Random(seed).shuffle(src_bids)
    n_va = max(1, int(len(src_bids) * 0.1))
    Xtr, ytr, _ = concat_cells(src, src_bids[:-n_va])
    Xva, yva, _ = concat_cells(src, src_bids[-n_va:])
    sc_src = StandardScaler().fit(Xtr.reshape(-1, Xtr.shape[2]))
    Xva_s = std_with(sc_src, Xva)
    set_seed(seed)
    base = new_model(model_name, Xtr.shape[2]).to(device)
    base.load_state_dict(torch.load(os.path.join(SRC_CACHE, f"src_{model_name}_s{seed}_ep120_h10.pt"),
                                    map_location=device, weights_only=True))
    splits = {}
    for tgt_name, split in (("CALCE", (7, 2, 7)), ("NASA", (2, 1, 1))):
        tgt = build_windows_ds(df, tgt_name, horizon=10)
        tb = sorted(tgt)
        n_ft, n_cal, n_te = split
        random.Random(seed).shuffle(tb)
        ft_b, cal_b, te_b = tb[:n_ft], tb[n_ft:n_ft+n_cal], tb[n_ft+n_cal:]
        Xall_t, _, _ = concat_cells(tgt, tb)
        sc_tgt = StandardScaler().fit(Xall_t.reshape(-1, Xall_t.shape[2]))
        Xft, yft, _ = concat_cells(tgt, ft_b)
        ft = new_model(model_name, Xtr.shape[2]).to(device)
        ft.load_state_dict(base.state_dict())
        ft = fit_generic(ft, std_with(sc_tgt, Xft), yft, 60, seed, device, std_with(sc_tgt, Xft), yft,
                         3e-4, lambda o, y: torch.mean((o - y) ** 2))
        Xte, yte, _ = concat_cells(tgt, te_b)
        splits[tgt_name] = dict(ft=ft, sc_tgt=sc_tgt, Xte=std_with(sc_tgt, Xte), yte=yte)
    return splits, Xva_s, yva


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    z = float(norm.ppf(0.95))   # alpha=0.10, two-sided
    log(f"P0-1 device={device} z95={z:.4f}")
    df = pd.read_csv(DATA)
    rows = []
    t_all = time.time()

    # MC dropout
    log("== MC dropout ==")
    for seed in (42, 43, 44, 45, 46):
        t0 = time.time()
        set_seed(seed)
        src = build_windows_ds(df, "MIT", horizon=10)
        src_bids = sorted(src)
        import random
        random.Random(seed).shuffle(src_bids)
        n_va = max(1, int(len(src_bids) * 0.1))
        Xtr, ytr, _ = concat_cells(src, src_bids[:-n_va])
        Xva, yva, _ = concat_cells(src, src_bids[-n_va:])
        sc_src = StandardScaler().fit(Xtr.reshape(-1, Xtr.shape[2]))
        Xtr_s, Xva_s = std_with(sc_src, Xtr), std_with(sc_src, Xva)
        cache_p = os.path.join(SRC1, f"src_tcnmc_s{seed}_ep120_h10.pt")
        set_seed(seed)
        smc = TCN_MC(Xtr.shape[2]).to(device)
        if os.path.exists(cache_p):
            smc.load_state_dict(torch.load(cache_p, map_location=device, weights_only=True))
        else:
            smc = fit_generic(smc, Xtr_s, ytr, 120, seed, device, Xva_s, yva, 1e-3,
                              lambda o, y: torch.mean((o - y) ** 2))
            torch.save(smc.state_dict(), cache_p)
        for tgt_name in ("CALCE", "NASA"):
            tgt = build_windows_ds(df, tgt_name, horizon=10)
            tb = sorted(tgt)
            import random
            random.Random(seed).shuffle(tb)
            ft_b = tb[:7] if tgt_name == "CALCE" else tb[:2]
            Xall_t, _, _ = concat_cells(tgt, tb)
            sc_tgt = StandardScaler().fit(Xall_t.reshape(-1, Xall_t.shape[2]))
            Xft, yft, _ = concat_cells(tgt, ft_b)
            ft = TCN_MC(Xtr.shape[2]).to(device)
            ft.load_state_dict(smc.state_dict())
            ft = fit_generic(ft, std_with(sc_tgt, Xft), yft, 60, seed, device,
                             std_with(sc_tgt, Xft), yft, 3e-4,
                             lambda o, y: torch.mean((o - y) ** 2))
            te_b = tb[-7] if False else (tb[9:] if tgt_name == "CALCE" else tb[3:])
            Xte, yte, _ = concat_cells(tgt, te_b)
            Xte_s = std_with(sc_tgt, Xte)
            samples = predict_mc(ft, Xte_s, device, T=30)
            mu, sd = samples.mean(0), samples.std(0, ddof=1)
            lo, hi = mu - z * sd, mu + z * sd
            picp, mpiw, nmpiw, winkler, pinball = metric_interval(yte, lo, hi, 0.10)
            rows.append(dict(method="MC dropout", seed=seed, domain=tgt_name, alpha=0.10,
                             PICP=round(picp, 4), MPIW=round(mpiw, 4), NMPIW=round(nmpiw, 4),
                             Winkler=round(winkler, 5), pinball=round(pinball, 6),
                             point_rmse=round(float(np.sqrt(np.mean((mu - yte) ** 2))), 5)))
        log(f"[MC s{seed}] {time.time()-t0:.0f}s")

    # Deep ensemble (seed-42 split, 5 members)
    log("== Deep ensemble (seed 42 split, 5 members) ==")
    t0 = time.time()
    seed = 42
    src = build_windows_ds(df, "MIT", horizon=10)
    src_bids = sorted(src)
    import random
    random.Random(seed).shuffle(src_bids)
    n_va = max(1, int(len(src_bids) * 0.1))
    Xtr, ytr, _ = concat_cells(src, src_bids[:-n_va])
    sc_src = StandardScaler().fit(Xtr.reshape(-1, Xtr.shape[2]))
    for tgt_name in ("CALCE", "NASA"):
        tgt = build_windows_ds(df, tgt_name, horizon=10)
        tb = sorted(tgt)
        random.Random(seed).shuffle(tb)
        ft_b, te_b = (tb[:7], tb[9:]) if tgt_name == "CALCE" else (tb[:2], tb[3:])
        Xall_t, _, _ = concat_cells(tgt, tb)
        sc_tgt = StandardScaler().fit(Xall_t.reshape(-1, Xall_t.shape[2]))
        Xte, yte, _ = concat_cells(tgt, te_b)
        Xte_s = std_with(sc_tgt, Xte)
        _cp = os.path.join(SRC_CACHE, f"src_tcn_s{seed}_ep120_h10.pt")
        if os.path.exists(_cp):
            _base_state = torch.load(_cp, map_location=device, weights_only=True)
        else:  # source caches are not distributed: rebuild on the spot under the Section 3.2 protocol (validation=train, as in the t4 default)
            _bm = new_model("tcn", Xtr.shape[2]).to(device)
            _bm = fit_generic(_bm, std_with(sc_src, Xtr), ytr, 120, seed, device,
                              std_with(sc_src, Xtr), ytr, 1e-3,
                              lambda o, y: torch.mean((o - y) ** 2))
            _base_state = _bm.state_dict()
        member_preds = []
        for init_seed in (1, 2, 3, 4, 5):
            Xft, yft, _ = concat_cells(tgt, ft_b)
            ft = new_model("tcn", Xtr.shape[2]).to(device)
            ft.load_state_dict(_base_state)
            ft = fit_generic(ft, std_with(sc_tgt, Xft), yft, 60, init_seed, device,
                             std_with(sc_tgt, Xft), yft, 3e-4,
                             lambda o, y: torch.mean((o - y) ** 2))
            member_preds.append(predict(ft, Xte_s, device))
        P = np.stack(member_preds)          # (5, n)
        mu, sd = P.mean(0), P.std(0, ddof=1)
        lo, hi = mu - z * sd, mu + z * sd
        picp, mpiw, nmpiw, winkler, pinball = metric_interval(yte, lo, hi, 0.10)
        rows.append(dict(method="Deep ensemble (5, seed42 split)", seed=seed, domain=tgt_name,
                         alpha=0.10, PICP=round(picp, 4), MPIW=round(mpiw, 4),
                         NMPIW=round(nmpiw, 4), Winkler=round(winkler, 5),
                         pinball=round(pinball, 6),
                         point_rmse=round(float(np.sqrt(np.mean((mu - yte) ** 2))), 5)))
    log(f"[ensemble] {time.time()-t0:.0f}s")

    # Quantile regression
    log("== Quantile regression (tau=0.1/0.9) ==")
    def qr_loss(out, y):
        ql, qh = out[:, 0], out[:, 1]
        e_l = y - ql; e_h = y - qh
        return 0.5 * (torch.mean(torch.maximum(0.1 * e_l, (0.1 - 1) * e_l)) +
                      torch.mean(torch.maximum(0.9 * e_h, (0.9 - 1) * e_h)))
    for seed in (42, 43, 44, 45, 46):
        t0 = time.time()
        src = build_windows_ds(df, "MIT", horizon=10)
        src_bids = sorted(src)
        import random
        random.Random(seed).shuffle(src_bids)
        n_va = max(1, int(len(src_bids) * 0.1))
        Xtr, ytr, _ = concat_cells(src, src_bids[:-n_va])
        Xva, yva, _ = concat_cells(src, src_bids[-n_va:])
        sc_src = StandardScaler().fit(Xtr.reshape(-1, Xtr.shape[2]))
        Xtr_s, Xva_s = std_with(sc_src, Xtr), std_with(sc_src, Xva)
        cache_p = os.path.join(SRC1, f"src_tcnqr_s{seed}_ep120_h10.pt")
        set_seed(seed)
        sqr = TCN_QR(Xtr.shape[2]).to(device)
        if os.path.exists(cache_p):
            sqr.load_state_dict(torch.load(cache_p, map_location=device, weights_only=True))
        else:
            sqr = fit_generic(sqr, Xtr_s, ytr, 120, seed, device, Xva_s, yva, 1e-3, qr_loss)
            torch.save(sqr.state_dict(), cache_p)
        for tgt_name in ("CALCE", "NASA"):
            tgt = build_windows_ds(df, tgt_name, horizon=10)
            tb = sorted(tgt)
            import random
            random.Random(seed).shuffle(tb)
            ft_b = tb[:7] if tgt_name == "CALCE" else tb[:2]
            te_b = tb[9:] if tgt_name == "CALCE" else tb[3:]
            Xall_t, _, _ = concat_cells(tgt, tb)
            sc_tgt = StandardScaler().fit(Xall_t.reshape(-1, Xall_t.shape[2]))
            Xft, yft, _ = concat_cells(tgt, ft_b)
            ft = TCN_QR(Xtr.shape[2]).to(device)
            ft.load_state_dict(sqr.state_dict())
            ft = fit_generic(ft, std_with(sc_tgt, Xft), yft, 60, seed, device,
                             std_with(sc_tgt, Xft), yft, 3e-4, qr_loss)
            Xte, yte, _ = concat_cells(tgt, te_b)
            out = predict(ft, std_with(sc_tgt, Xte), device)
            lo, hi = out[:, 0], out[:, 1]
            hi = np.maximum(hi, lo)
            picp, mpiw, nmpiw, winkler, pinball = metric_interval(yte, lo, hi, 0.10)
            mid = 0.5 * (lo + hi)
            rows.append(dict(method="Quantile regression", seed=seed, domain=tgt_name, alpha=0.10,
                             PICP=round(picp, 4), MPIW=round(mpiw, 4), NMPIW=round(nmpiw, 4),
                             Winkler=round(winkler, 5), pinball=round(pinball, 6),
                             point_rmse=round(float(np.sqrt(np.mean((mid - yte) ** 2))), 5)))
        log(f"[QR s{seed}] {time.time()-t0:.0f}s")

    # Conformal control (P0-2 recomputation batch, alpha=0.10 target route)
    cref = os.path.join(OUT_RES, "p0_2_metrics.csv")
    if os.path.exists(cref):
        for r in csv.DictReader(open(cref, encoding="utf-8")):
            if r["route"] == "target" and abs(float(r["alpha"]) - 0.10) < 1e-9:
                rows.append(dict(method="Split conformal (recomputed)", seed=int(r["seed"]),
                                 domain=r["domain"], alpha=0.10, PICP=float(r["PICP"]),
                                 MPIW=float(r["MPIW"]), NMPIW=float(r["NMPIW"]),
                                 Winkler=float(r["Winkler"]), pinball=float(r["pinball"]),
                                 point_rmse=float(r["point_rmse"])))
    with open(os.path.join(OUT_RES, "p0_1_uncertainty.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    log(f"P0-1 done in {time.time()-t_all:.0f}s, {len(rows)} rows -> p0_1_uncertainty.csv")
    # summary
    import statistics as st
    log("\n=== 5-seed summary (mean+/-std PICP / MPIW / Winkler) ===")
    methods = sorted(set(r["method"] for r in rows))
    for mth in methods:
        for dom in ("CALCE", "NASA"):
            sel = [r for r in rows if r["method"] == mth and r["domain"] == dom]
            if not sel:
                continue
            log(f"  {mth:34s} {dom:6s} PICP {np.mean([r['PICP'] for r in sel]):.3f}±{np.std([r['PICP'] for r in sel], ddof=1):.3f}  "
                f"MPIW {np.mean([r['MPIW'] for r in sel]):.3f}±{np.std([r['MPIW'] for r in sel], ddof=1):.3f}  "
                f"Winkler {np.mean([r['Winkler'] for r in sel]):.4f}")


if __name__ == "__main__":
    main()
