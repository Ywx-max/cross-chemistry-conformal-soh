# -*- coding: utf-8 -*-
"""T3e = the T3c ablation protocol rerun on modeling table v3c (data-version control, part two; the version used in the manuscript).

The only difference from t3c_ablation_local.py: DATA points to modeling_table_v3c.csv and outputs use the t3e_ prefix.
The Table 6 numbers in checks/final_data_check.py are checked against the ablation_v3c results,
i.e. the (aggregated) output of this script. Protocol details in t3c_ablation_local.py."""
import argparse, json, os, random, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

# base7 (historical name, now 6 features) = drop charge_dur_s (missing for MIT) and soh.
# soh must be dropped: it is the label of the SOH task, and a label in the inputs hands the answer to the model
# (before the 2026-10-03 fix an identity-copy baseline scored RMSE=0, the leak in plain sight).
FEATS_BASE6 = ["capacity_Ah", "discharge_dur_s", "v_mean_V", "v_min_V",
              "ica_peak", "ica_peak_V"]
# horizon: label = soh at H cycles after the last row of the window (H-step-ahead SOH prediction).
H = 10
# curve features: + fixed-quantile capacity-voltage points (v_q10..v_q90) + ICA secondary peak + FWHM
FEATS_CURVE13 = FEATS_BASE6 + ["v_q10", "v_q30", "v_q50", "v_q70", "v_q90",
                               "ica2_peak", "ica_fwhm"]
FEATS = FEATS_BASE6  # overridden at runtime by --feats
WINDOW = 20
# modeling-table path (data goes under data/; merge procedure in code/README.md)
DATA = "data/modeling_table_v3c.csv"

def set_seed(seed):
    """Fix the random sources (GPU convolution stays nondeterministic, so reruns move a little; see the same-named function in t2_train_local)."""
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def build_windows_ds(df, dataset, window=WINDOW, horizon=H):
    cells = {}
    sub = df[df["dataset"] == dataset]
    for bid, g in sub.sort_values("cycle").groupby("battery_id"):
        f = g[FEATS].astype(float).copy()
        f = f.interpolate(limit_direction="both").fillna(0.0).values
        cyc = g["cycle"].values
        y = g["soh"].astype(float).values
        ok = ~np.isnan(y)
        f, cyc, y = f[ok], cyc[ok], y[ok]
        X, yy = [], []
        for i in range(window - 1, len(f) - horizon):
            if cyc[i] - cyc[i - window + 1] == window - 1 \
                    and cyc[i + horizon] == cyc[i] + horizon:
                X.append(f[i - window + 1:i + 1]); yy.append(y[i + horizon])
        if X:
            cells[bid] = (np.asarray(X, np.float32), np.asarray(yy, np.float32))
    return cells

class CausalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k, dil):
        super().__init__()
        self.pad = (k - 1) * dil
        self.conv = nn.Conv1d(in_ch, out_ch, k, dilation=dil, padding=self.pad)
        self.res = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
    def forward(self, x):
        y = self.conv(x)
        if self.pad:
            y = y[:, :, :-self.pad]
        return torch.relu(y + self.res(x))

class TCN(nn.Module):
    def __init__(self, input_dim, d=64, layers=4):
        super().__init__()
        ch = [input_dim] + [d] * layers
        self.blocks = nn.ModuleList([CausalBlock(ch[i], ch[i+1], 3, 2**i) for i in range(layers)])
        self.fc = nn.Linear(d, 1)
    def forward(self, x):
        h = x.transpose(1, 2)
        for blk in self.blocks:
            h = blk(h)
        return self.fc(h[:, :, -1]).squeeze(-1)

class RNNWrap(nn.Module):
    def __init__(self, input_dim, kind):
        super().__init__()
        self.r = (nn.LSTM if kind == "lstm" else nn.GRU)(input_dim, 128, 2, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(128, 1)
    def forward(self, x):
        o, _ = self.r(x)
        return self.fc(o[:, -1]).squeeze(-1)

def new_model(name, input_dim):
    if name in ("lstm", "gru"):
        return RNNWrap(input_dim, name)
    if name == "tcn":
        return TCN(input_dim)
    raise ValueError(name)

def fit_model(model, Xtr, ytr, epochs, seed, device, Xva=None, yva=None, lr=1e-3):
    set_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.MSELoss()
    ds = torch.utils.data.TensorDataset(torch.tensor(Xtr, dtype=torch.float32),
                                        torch.tensor(ytr, dtype=torch.float32))
    dl = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=True)
    Xva = Xtr if Xva is None else Xva
    yva = ytr if yva is None else yva
    best, best_state, patience = float("inf"), None, 0
    for ep in range(epochs):
        model.train()
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(); lossf(model(xb), yb).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        model.eval()
        with torch.no_grad():
            va = lossf(model(torch.tensor(Xva, dtype=torch.float32).to(device)),
                       torch.tensor(yva, dtype=torch.float32).to(device)).item()
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

def eval_model(model, X, y, device):
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(X, dtype=torch.float32).to(device)).cpu().numpy()
    return {"rmse": float(np.sqrt(np.mean((pred - y) ** 2))),
            "mae": float(np.mean(np.abs(pred - y))), "n": int(len(y))}

def std_with(sc, X):
    return ((X - sc.mean_) / (sc.scale_ + 1e-8)).astype(np.float32)

def concat_cells(cells, bids):
    X = np.concatenate([cells[b][0] for b in bids])
    y = np.concatenate([cells[b][1] for b in bids])
    return X, y, [b for b in bids for _ in range(len(cells[b][0]))]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="tcn", choices=["tcn", "lstm"])
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--ft-epochs", type=int, default=60)
    ap.add_argument("--horizon", type=int, default=H,
                    help="horizon: label = soh at H cycles after the last window row")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--feats", default="base7", choices=["base7", "curve14"])
    ap.add_argument("--out", default="results/ablation")
    ap.add_argument("--deterministic", action="store_true",
                    help="enable cuDNN/PyTorch deterministic algorithms (for cross-pipeline reproducibility diagnostics; off by default, "
                         "matching the production environment of the released results)")
    ap.add_argument("--data", default=DATA, help="modeling-table csv path")
    ap.add_argument("--src-cache", default=None,
                    help="unified source-model cache directory: a hit skips source pre-training, a miss trains and writes it")
    args = ap.parse_args()
    # the feature set is switched at runtime: base7 (historical name) = the paper's base-6, curve14 = curve-13
    global FEATS
    FEATS = FEATS_BASE6 if args.feats == "base7" else FEATS_CURVE13
    if args.deterministic:
        # CUBLAS_WORKSPACE_CONFIG must be set before the first CUDA operation
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
        print("[deterministic] cuDNN/PyTorch deterministic algorithms ON", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"T3b (per-dataset standardization) device={device} model={args.model}", flush=True)
    df = pd.read_csv(args.data)
    src = build_windows_ds(df, "MIT", horizon=args.horizon)
    src_bids = sorted(src)
    random.Random(args.seed).shuffle(src_bids)
    n_va = max(1, int(len(src_bids) * 0.1))
    Xtr, ytr, _ = concat_cells(src, src_bids[:-n_va])
    Xva, yva, _ = concat_cells(src, src_bids[-n_va:])
    sc_src = StandardScaler().fit(Xtr.reshape(-1, Xtr.shape[2]))
    Xtr, Xva = std_with(sc_src, Xtr), std_with(sc_src, Xva)
    print(f"source MIT: train cells {len(src_bids)-n_va}, windows {len(Xtr)}", flush=True)
    set_seed(args.seed)  # seed before instantiating: otherwise same-seed reruns initialise different weights
    model = new_model(args.model, Xtr.shape[2]).to(device)
    import os as _os
    _cache_hit = None
    if args.src_cache:
        _os.makedirs(args.src_cache, exist_ok=True)
        # the base7 source model is shared with the other scripts (unified cache); curve14 has different input
        # dimensions and must keep its own feature-suffixed cache name, never falling back to the old 7-feature cache
        if args.feats == "base7":
            _hit = _os.path.join(args.src_cache, f"src_{args.model}_s{args.seed}_ep{args.epochs}.pt")
            _legacy = _os.path.join(args.src_cache, f"t4d_src_{args.model}_s{args.seed}.pt")
            _cache_hit = _hit if _os.path.exists(_hit) else (_legacy if _os.path.exists(_legacy) else None)
        else:
            _hit = _os.path.join(args.src_cache, f"src_{args.model}_s{args.seed}_ep{args.epochs}_{args.feats}.pt")
            _cache_hit = _hit if _os.path.exists(_hit) else None
    if _cache_hit:
        model.load_state_dict(torch.load(_cache_hit, map_location=device, weights_only=True))
        print(f"[cache] source model {_cache_hit}", flush=True)
    else:
        t0 = time.time()
        model = fit_model(model, Xtr, ytr, args.epochs, args.seed, device, Xva, yva)
        print(f"source-domain training done in {time.time()-t0:.0f}s", flush=True)
        if args.src_cache:
            _sfx = "" if args.feats == "base7" else f"_{args.feats}"
            _save = _os.path.join(args.src_cache, f"src_{args.model}_s{args.seed}_ep{args.epochs}{_sfx}.pt")
            torch.save(model.state_dict(), _save)
            print(f"[cache] source model written to {_save}", flush=True)
    results = {"model": args.model, "task": "soh", "protocol": "per-dataset-std", "feats": args.feats,
               "seed": args.seed, "horizon": args.horizon, "targets": {}}
    for tgt_name in ["CALCE", "NASA"]:
        tgt = build_windows_ds(df, tgt_name, horizon=args.horizon)
        tb = sorted(tgt)
        n_ft = max(1, len(tb) // 2)
        random.Random(args.seed).shuffle(tb)
        ft_bids, te_bids = tb[:n_ft], tb[n_ft:]
        # target-domain scaler: unlabeled statistics of all target windows (standard unsupervised DA scope)
        Xall_t, yall_t, _ = concat_cells(tgt, tb)
        sc_tgt = StandardScaler().fit(Xall_t.reshape(-1, Xall_t.shape[2]))
        Xte, yte, _ = concat_cells(tgt, te_bids)
        r_zero = eval_model(model, std_with(sc_tgt, Xte), yte, device)
        Xft, yft, _ = concat_cells(tgt, ft_bids)
        ft_model = new_model(args.model, Xtr.shape[2]).to(device)
        ft_model.load_state_dict(model.state_dict())
        # fine-tuning lr=3e-4 (an order below the source), as in the t3b main experiment.
        # the ablation changes the feature set only and locks every other variable, so differences are attributable to features
        ft_model = fit_model(ft_model, std_with(sc_tgt, Xft), yft,
                             args.ft_epochs, args.seed, device, lr=3e-4)
        r_ft = eval_model(ft_model, std_with(sc_tgt, Xte), yte, device)
        set_seed(args.seed)  # target-only starts from random init; fix the seed first as well
        to_model = new_model(args.model, Xtr.shape[2]).to(device)
        to_model = fit_model(to_model, std_with(sc_tgt, Xft), yft,
                             args.ft_epochs, args.seed, device, lr=1e-3)
        r_to = eval_model(to_model, std_with(sc_tgt, Xte), yte, device)
        results["targets"][tgt_name] = {"test_cells": te_bids, "ft_cells": ft_bids,
                                        "zero_shot": r_zero, "fine_tune": r_ft, "target_only": r_to}
        print(f"[{tgt_name}] zero-shot RMSE={r_zero['rmse']:.4f} | "
              f"fine-tune RMSE={r_ft['rmse']:.4f} | target-only RMSE={r_to['rmse']:.4f}", flush=True)
    os.makedirs(args.out, exist_ok=True)
    with open(f"{args.out}/t3e_{args.feats}_{args.model}_s{args.seed}.json", "w") as f:
        json.dump(results, f, indent=1)
    print("T3B DONE", flush=True)

if __name__ == "__main__":
    main()
