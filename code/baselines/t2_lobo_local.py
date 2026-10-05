# -*- coding: utf-8 -*-
"""LOBO leave-one-battery-out full validation (MIT 124 cells = 119 folds; b1c0-b1c4, five cells, lack EOL labels and
stay out of RUL evaluation; they are not filtered for too few windows, all 124 cells have enough).

A finer evaluation than the 5-fold group CV: each fold holds out one whole cell, trains on the other 123, and
yields a per-cell error distribution instead of a single mean. The distribution columns of Table 2 and the histogram of Fig. 2 come from here.

Resumable: each fold appends one jsonl line immediately, and a restart skips completed folds automatically.
On the local 4060, 119 folds take about 2-3 hours per model, not worth rerunning after a power cut.

Run: python t2_lobo_local.py --model tcn --seed 42
Output: results/baselines/lobo_<model>_s<seed>.jsonl (one line per fold: te_cell/rmse/mae/epochs)"""
import argparse, json, os, time
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
# model, sliding windows and training loop all reuse t2_train_local; this script only handles fold cutting and resuming
from t2_train_local import build_windows, new_model, train_one, DATA

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="tcn", choices=["tcn", "lstm", "gru", "transformer"])
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--epochs", type=int, default=60)
ap.add_argument("--out", default="results/baselines")
args = ap.parse_args()

OUT = args.out
os.makedirs(OUT, exist_ok=True)
JSONL = f"{OUT}/lobo_{args.model}_s{args.seed}.jsonl"
EPOCHS = args.epochs
MODEL = args.model

df = pd.read_csv(DATA)
df = df[df["dataset"] == "MIT"]
cells = build_windows(df)
bids = sorted(cells)
X_all = np.concatenate([cells[b][0] for b in bids])
y_all = np.concatenate([cells[b][1] for b in bids])
g_all = np.concatenate([[b] * len(cells[b][0]) for b in bids])

# resume: read completed test cells from an old jsonl and skip them on restart
done = set()
if os.path.exists(JSONL):
    with open(JSONL) as f:
        for line in f:
            try:
                done.add(json.loads(line)["te_cell"])
            except Exception:
                pass

device = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(OUT, exist_ok=True)
print(f"LOBO device={device} model={MODEL} seed={args.seed} cells={len(bids)} done={len(done)} remaining={len(bids)-len(done)}", flush=True)
t0 = time.time()
cnt = 0
for i, te_b in enumerate(bids):
    if te_b in done:
        continue
    # leave-one-cell-out: the training set is the windows of all the other cells (windows of one cell never split)
    tri_mask = g_all != te_b
    Xtr, ytr = X_all[tri_mask], y_all[tri_mask]
    sc = StandardScaler().fit(Xtr.reshape(-1, Xtr.shape[2]))
    # the last 8% serves as early-stopping validation; with many LOBO folds a smaller validation set saves real training time
    n_va = max(1, int(len(Xtr) * 0.08))
    Xa = ((Xtr - sc.mean_) / (sc.scale_ + 1e-8)).astype(np.float32)
    Xte = ((X_all[~tri_mask] - sc.mean_) / (sc.scale_ + 1e-8)).astype(np.float32)
    rmse, mae, ep = train_one(MODEL, Xa[:-n_va], ytr[:-n_va], Xte, y_all[~tri_mask],
                              EPOCHS, args.seed, device, Xa[-n_va:], ytr[-n_va:])
    # each fold is written to disk at once (append mode); this drives both resuming and mid-run inspection
    rec = {"te_cell": te_b, "rmse": rmse, "mae": mae, "epochs": ep, "n_test": int((~tri_mask).sum())}
    with open(JSONL, "a") as f:
        f.write(json.dumps(rec) + "\n")
    cnt += 1
    if cnt % 10 == 0 or i == len(bids) - 1:
        el = time.time() - t0
        eta = el / max(1, cnt) * (len(bids) - len(done) - cnt)
        print(f"[{i+1}/{len(bids)}] latest {te_b}: RMSE={rmse:.1f} | elapsed {el/60:.0f}min ETA {eta/60:.0f}min", flush=True)
print("LOBO DONE", flush=True)
