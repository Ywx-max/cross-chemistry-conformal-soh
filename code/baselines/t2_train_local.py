# -*- coding: utf-8 -*-
"""T2 source-domain baselines: four backbones (LSTM/GRU/TCN/Transformer) on MIT-Stanford for RUL prediction.

This is the training-side starting point: compare the four backbones and pick the transfer backbone (the paper
picks the TCN, on par with the Transformer in accuracy and far cheaper to train; Section 4.2), then move to the cross-domain transfer under ../transfer/.

Evaluation uses 5-fold group CV: folds are cut by cell, so all windows of a cell go entirely into training
or entirely into the test fold. Adjacent-cycle windows are near-duplicates, and a random window-level split
would let the test set memorise answers, inflating metrics meaninglessly.

Run: python t2_train_local.py --model tcn --smoke   # single-cell overfit self-check
      python t2_train_local.py --model tcn --folds 5 # full 5-fold run
Input: modeling_table_v3.csv; output: results/baselines/<model>_f<fold>_s<seed>.json (per fold + summary)"""
import argparse, json, os, random, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

# 8 cycle-level features. charge_dur_s exists for MIT/CALCE but not NASA, hence 8 here;
# the T3b per-dataset protocol drops charge_dur_s and soh down to 6 (see ../transfer/t3b_std_local.py; soh as an input is label leakage in the SOH task, dropped 2026-10-03).
FEATS = ["capacity_Ah", "soh", "discharge_dur_s", "v_mean_V", "v_min_V",
         "ica_peak", "ica_peak_V", "charge_dur_s"]
WINDOW = 20
# modeling-table path (data goes under data/; merge procedure in code/README.md)
DATA = "data/modeling_table_v3.csv"

def set_seed(seed):
    """Fix the three random sources. This makes splits/initialisation/shuffles reproducible only: GPU convolution
    stays nondeterministic and same-configuration reruns still move by fractions of a percent to a few percent, which is why the paper reports 5-seed statistics."""
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def build_windows(df, window=WINDOW):
    # keep only windows with strictly consecutive cycle numbers: gaps from rest periods/retests are excluded
    # so no time break is fed to the model; rows without RUL (EOL not observed) are dropped as well.
    cells = {}
    for bid, g in df.sort_values("cycle").groupby("battery_id"):
        f = g[FEATS].astype(float).copy()
        f = f.interpolate(limit_direction="both").fillna(0.0).values
        cyc = g["cycle"].values
        rul = g["rul"].astype(float).values
        # negative RUL = rows past EOL still under test; labels without physical meaning, excluded from training and evaluation
        # (2026-10-03 fix: MIT had 11.3% and NASA 29.3% negative-RUL rows leaking into training)
        ok = (~np.isnan(rul)) & (rul >= 0)
        f, cyc, rul = f[ok], cyc[ok], rul[ok]
        X, y = [], []
        for i in range(window - 1, len(f)):
            if cyc[i] - cyc[i - window + 1] == window - 1:
                X.append(f[i - window + 1:i + 1]); y.append(rul[i])
        if X:
            cells[bid] = (np.asarray(X, np.float32), np.asarray(y, np.float32))
    return cells

class CausalBlock(nn.Module):
    """Dilated causal convolution + residual. Causal = each step sees the past only, no peeking ahead; the
    zero padding on the left would bleed into the window tail, so forward slices it off, else the window end (the position actually used for prediction) is polluted."""
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

class Transformer(nn.Module):
    """Lightweight Transformer: 3 layers / d=128 / 4 heads, learnable positional encoding.
    Accuracy on par with the TCN but several times slower to train, one reason the TCN was chosen as the transfer backbone."""
    def __init__(self, input_dim, d=128, heads=4, layers=3):
        super().__init__()
        self.inp = nn.Linear(input_dim, d)
        self.pos = nn.Parameter(torch.randn(1, 512, d) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(d, heads, 256, dropout=0.1, batch_first=True)
        self.enc = nn.TransformerEncoder(enc_layer, layers)
        self.fc = nn.Linear(d, 1)
    def forward(self, x):
        h = self.inp(x) + self.pos[:, :x.shape[1]]
        return self.fc(self.enc(h)[:, -1]).squeeze(-1)

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
    if name == "transformer":
        return Transformer(input_dim)
    raise ValueError(name)

def train_one(model_name, Xtr, ytr, Xte, yte, epochs, seed, device, Xva=None, yva=None):
    """Train and evaluate one fold. Early stopping watches the validation-fold loss (patience=10); the 200-epoch cap is rarely reached."""
    set_seed(seed)
    model = new_model(model_name, Xtr.shape[2]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = nn.MSELoss()
    ds = torch.utils.data.TensorDataset(torch.tensor(Xtr, dtype=torch.float32), torch.tensor(ytr, dtype=torch.float32))
    dl = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=True)
    Xva = Xtr if Xva is None else Xva
    yva = ytr if yva is None else yva
    best, best_state, patience = float("inf"), None, 0
    for ep in range(epochs):
        model.train()
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(); lossf(model(xb), yb).backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        model.eval()
        with torch.no_grad():
            va = lossf(model(torch.tensor(Xva, dtype=torch.float32).to(device)), torch.tensor(yva, dtype=torch.float32).to(device)).item()
        if va < best - 1e-6:
            best, patience = va, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= 10:
                break
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(Xte, dtype=torch.float32).to(device)).cpu().numpy()
    rmse = float(np.sqrt(np.mean((pred - yte) ** 2)))
    mae = float(np.mean(np.abs(pred - yte)))
    return rmse, mae, ep + 1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="lstm", choices=["lstm", "gru", "tcn", "transformer"])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dataset", default="MIT")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="results/baselines")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} model={args.model} dataset={args.dataset}", flush=True)
    df = pd.read_csv(DATA)
    df = df[df["dataset"] == args.dataset]
    cells = build_windows(df)
    bids = sorted(cells)
    print(f"cells with valid windows: {len(bids)}/{df['battery_id'].nunique()}", flush=True)
    if args.smoke:
        # smoke self-check: train a cell on itself; RMSE should collapse near zero, and if it does not the pipeline has a bug.
        # The cheapest "does it run" criterion when writing or editing training code; run it after any change.
        bid = bids[0]
        X, y = cells[bid]
        n = len(X); tr_n = int(n * 0.8)
        sc = StandardScaler().fit(X[:tr_n].reshape(-1, X.shape[2]))
        Xt = (X - sc.mean_) / (sc.scale_ + 1e-8)
        rmse, mae, ep = train_one(args.model, Xt[:tr_n], y[:tr_n], Xt[:tr_n], y[:tr_n], 50, args.seed, device)
        print(f"[SMOKE overfit self-check] {bid}: train RMSE={rmse:.2f} MAE={mae:.2f} epochs={ep}", flush=True)
        rmse, mae, ep = train_one(args.model, Xt[:tr_n], y[:tr_n], Xt[tr_n:], y[tr_n:], 100, args.seed, device)
        print(f"[SMOKE held-out] {bid}: test RMSE={rmse:.2f} MAE={mae:.2f} epochs={ep}", flush=True)
        return
    X_all = np.concatenate([cells[b][0] for b in bids])
    y_all = np.concatenate([cells[b][1] for b in bids])
    g_all = np.concatenate([[b] * len(cells[b][0]) for b in bids])
    gkf = GroupKFold(n_splits=args.folds)
    res, t0 = [], time.time()
    # full group-cross-validation batch
    for k, (tri, tei) in enumerate(gkf.split(X_all, y_all, g_all)):
        # the scaler is fitted on the training fold only: fitting on everything would leak the test fold's mean and variance
        sc = StandardScaler().fit(X_all[tri].reshape(-1, X_all.shape[2]))
        # train/validation/test all use the same statistics for standardization (scaler fitted on the training set only).
        Xtr = (X_all[tri] - sc.mean_) / (sc.scale_ + 1e-8)
        Xte = (X_all[tei] - sc.mean_) / (sc.scale_ + 1e-8)
        # the last 12% of the training fold serves as early-stopping validation (still split by window: it only decides when to stop, no metrics reported)
        n_va = max(1, int(len(tri) * 0.12))
        rmse, mae, ep = train_one(args.model, Xtr[:-n_va], y_all[tri][:-n_va],
            Xte, y_all[tei],
            args.epochs, args.seed, device,
            Xtr[-n_va:], y_all[tri][-n_va:])
        res.append({"fold": k, "rmse": rmse, "mae": mae, "epochs": ep})
        print(f"fold{k}: RMSE={rmse:.2f} MAE={mae:.2f} ep={ep} ({time.time()-t0:.0f}s)", flush=True)
    rmses = [r["rmse"] for r in res]
    print(f"SUMMARY [{args.model}] {args.folds}fold RMSE {np.mean(rmses):.2f}+-{np.std(rmses):.2f}", flush=True)
    os.makedirs(args.out, exist_ok=True)
    with open(f"{args.out}/{args.model}_f{args.folds}_s{args.seed}.json", "w") as f:
        json.dump({"model": args.model, "dataset": args.dataset, "folds": res,
                   "rmse_mean": float(np.mean(rmses)), "rmse_std": float(np.std(rmses)),
                   "window": WINDOW, "seed": args.seed}, f, indent=1)

if __name__ == "__main__":
    main()
