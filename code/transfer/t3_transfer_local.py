# -*- coding: utf-8 -*-
"""T3 cross-chemistry transfer main experiment: MIT (LFP source) -> CALCE / NASA (LCO targets).

Three mechanisms compared (data behind Table 3):
  zero-shot   source model straight onto the target domain; expected to collapse, and how it collapses is one of the paper's findings
  fine-tune   load source weights, fine-tune at a small learning rate (the proposed method)
  target-only train from scratch on the same target data, answering "is transfer worth it at all"

Task scope --task: soh is the primary scope (every cycle carries an SOH label, so samples abound);
rul counts only cells that reached EOL (only part of the NASA cells do), giving few samples; used as a secondary control.
This script standardises with source-domain statistics (the original protocol); see t3b_std_local.py for per-dataset standardization.

Run: python t3_transfer_local.py --model tcn --task soh --seed 42
Manuscript scope: one run per seed 42-46, reported as 5-seed mean+/-std.
Output: results/transfer/t3_<task>_<model>_s<seed>.json"""
import argparse, json, os, random, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

FEATS = ["capacity_Ah", "soh", "discharge_dur_s", "v_mean_V", "v_min_V",
         "ica_peak", "ica_peak_V", "charge_dur_s"]
# the soh task must drop soh: a label in the inputs hands the answer to the model (2026-10-03 leak fix).
# the rul task keeps soh: current health is a legitimate input for RUL prediction, matching the t2 baseline scope.
FEATS_SOH = [c for c in FEATS if c != "soh"]
WINDOW = 20
# horizon of the soh task: label = soh at H cycles after the last row of the window (H-step-ahead prediction).
# Windows closer than H steps to the record end, or with cycle gaps between window and label, are dropped.
H = 10
# modeling-table path (data goes under data/; merge procedure in code/README.md)
DATA = "data/modeling_table_v3.csv"

def set_seed(seed):
    """Fix the random sources. GPU convolution remains nondeterministic, so same-seed reruns move a
    little; this is the main source of the up-to-14% RMSE spread of repeated runs (Section 4.8)."""
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def build_windows_ds(df, dataset, task, window=WINDOW, horizon=H):
    """Window the series per dataset. task=soh: y = SOH at the horizon-th cycle after the window end (ahead prediction);
    task=rul: y = RUL, keeping only cells with observed EOL (the others have no life end, so the RUL label is
    undefined), and dropping negative RUL (rows past EOL still under test, labels without physical meaning)."""
    cells = {}
    sub = df[df["dataset"] == dataset]
    ycol = "soh" if task == "soh" else "rul"
    feats = FEATS_SOH if task == "soh" else FEATS
    for bid, g in sub.sort_values("cycle").groupby("battery_id"):
        f = g[feats].astype(float).copy()
        f = f.interpolate(limit_direction="both").fillna(0.0).values
        cyc = g["cycle"].values
        y_all = g[ycol].astype(float).values
        ok = ~np.isnan(y_all)
        if task == "rul":
            ok &= y_all >= 0
        f, cyc, y_all = f[ok], cyc[ok], y_all[ok]
        X, y = [], []
        if task == "soh":
            for i in range(window - 1, len(f) - horizon):
                if cyc[i] - cyc[i - window + 1] == window - 1 \
                        and cyc[i + horizon] == cyc[i] + horizon:
                    X.append(f[i - window + 1:i + 1]); y.append(y_all[i + horizon])
        else:
            for i in range(window - 1, len(f)):
                if cyc[i] - cyc[i - window + 1] == window - 1:
                    X.append(f[i - window + 1:i + 1]); y.append(y_all[i])
        if X:
            cells[bid] = (np.asarray(X, np.float32), np.asarray(y, np.float32))
    return cells

class CausalBlock(nn.Module):
    """Dilated causal convolution residual block (identical to t2_train_local. Scripts are deliberately
    self-contained so a single file runs on its own; the cost is this duplicated block in several scripts)."""
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

def concat_cells(cells, bids):
    X = np.concatenate([cells[b][0] for b in bids])
    y = np.concatenate([cells[b][1] for b in bids])
    return X, y, [b for b in bids for _ in range(len(cells[b][0]))]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="tcn", choices=["tcn", "lstm"])
    ap.add_argument("--task", default="soh", choices=["soh", "rul"])
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--ft-epochs", type=int, default=60)
    ap.add_argument("--horizon", type=int, default=H,
                    help="horizon of the soh task (unused for rul)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results/transfer")
    ap.add_argument("--data", default=DATA, help="modeling-table csv path")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"T3v2 device={device} model={args.model} task={args.task}", flush=True)
    df = pd.read_csv(args.data)
    src = build_windows_ds(df, "MIT", args.task, horizon=args.horizon)
    src_bids = sorted(src)
    # inside-source 90/10 train/validation split; the seed only feeds random.Random, so the split is reproducible across runs
    random.Random(args.seed).shuffle(src_bids)
    n_va = max(1, int(len(src_bids) * 0.1))
    Xtr, ytr, _ = concat_cells(src, src_bids[:-n_va])
    Xva, yva, _ = concat_cells(src, src_bids[-n_va:])
    # source-domain scaler. Note the target domain is standardised with this scaler too (the original protocol).
    # CALCE/NASA features have different scales than MIT, one main reason zero-shot collapses to 0.1+;
    # the per-dataset counterpart is t3b_std_local.py, and the two together are the paper's drift decomposition.
    sc = StandardScaler().fit(Xtr.reshape(-1, Xtr.shape[2]))
    Xtr = ((Xtr - sc.mean_) / (sc.scale_ + 1e-8)).astype(np.float32)
    Xva = ((Xva - sc.mean_) / (sc.scale_ + 1e-8)).astype(np.float32)
    print(f"source MIT: train cells {len(src_bids)-n_va}, windows {len(Xtr)}", flush=True)
    set_seed(args.seed)  # seed before instantiating: otherwise same-seed reruns initialise different weights
    model = new_model(args.model, Xtr.shape[2]).to(device)
    t0 = time.time()
    model = fit_model(model, Xtr, ytr, args.epochs, args.seed, device, Xva, yva)
    print(f"source-domain training done in {time.time()-t0:.0f}s", flush=True)

    results = {"model": args.model, "task": args.task, "seed": args.seed,
               "horizon": args.horizon, "targets": {}}
    # soh scope: CALCE 16 cells split evenly into fine-tune/test (8/8); NASA uses 2 of 4 cells for fine-tuning (2/2).
    # the rul scope exists only for NASA and fixes 1 fine-tuning cell (too few samples for an automatic split).
    targets = [("CALCE", None), ("NASA", None)] if args.task == "soh" else [("NASA", 1)]
    for tgt_name, n_ft_fixed in targets:
        tgt = build_windows_ds(df, tgt_name, args.task, horizon=args.horizon)
        tb = sorted(tgt)
        if len(tb) < 2:
            print(f"[{tgt_name}] too few valid cells ({len(tb)}), skipped", flush=True)
            continue
        n_ft = n_ft_fixed if n_ft_fixed else max(1, len(tb) // 2)
        random.Random(args.seed).shuffle(tb)
        ft_bids, te_bids = tb[:n_ft], tb[n_ft:]
        Xte, yte, _ = concat_cells(tgt, te_bids)
        Xte = ((Xte - sc.mean_) / (sc.scale_ + 1e-8)).astype(np.float32)
        r_zero = eval_model(model, Xte, yte, device)
        Xft, yft, _ = concat_cells(tgt, ft_bids)
        Xft = ((Xft - sc.mean_) / (sc.scale_ + 1e-8)).astype(np.float32)
        ft_model = new_model(args.model, Xtr.shape[2]).to(device)
        ft_model.load_state_dict(model.state_dict())
        # fine-tuning lr 3e-4, an order below source pre-training (1e-3): a few target cells cannot support large steps,
        # which would wash out the degradation knowledge learned on the source domain
        ft_model = fit_model(ft_model, Xft, yft, args.ft_epochs, args.seed, device, lr=3e-4)
        r_ft = eval_model(ft_model, Xte, yte, device)
        # target-only control: same data, same epochs, but from scratch at the full learning rate.
        # "does pre-training help" is only answerable against it (this is the target-only column of Table 3)
        set_seed(args.seed)  # as above: random initialisation under control
        to_model = new_model(args.model, Xtr.shape[2]).to(device)
        to_model = fit_model(to_model, Xft, yft, args.ft_epochs, args.seed, device, lr=1e-3)
        r_to = eval_model(to_model, Xte, yte, device)
        results["targets"][tgt_name] = {"test_cells": te_bids, "ft_cells": ft_bids,
                                        "zero_shot": r_zero, "fine_tune": r_ft, "target_only": r_to}
        print(f"[{tgt_name}] zero-shot RMSE={r_zero['rmse']:.4f} | "
              f"fine-tune RMSE={r_ft['rmse']:.4f} | target-only RMSE={r_to['rmse']:.4f}", flush=True)
    os.makedirs(args.out, exist_ok=True)
    with open(f"{args.out}/t3_{args.task}_{args.model}_s{args.seed}.json", "w") as f:
        json.dump(results, f, indent=1)
    print("T3 DONE", flush=True)

if __name__ == "__main__":
    main()
