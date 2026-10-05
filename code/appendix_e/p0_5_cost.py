# -*- coding: utf-8 -*-
"""P0-5: model-cost table - parameter count, single-fold training time, inference latency, memory (hardware noted).

Parameters: the four backbones TCN/LSTM/GRU/Transformer (6 input features, window 20).
Training time: source-domain fit_model (protocol = t4_conformal_local.py v1.2.4: MSE, Adam 1e-3,
batch 256, gradient clip 1.0, patience 10, 120-epoch cap), measured once per backbone.
Inference latency: forward passes on GPU at batch=1 and batch=256, 100 runs each, mean (10 warm-up runs).
Memory: torch.cuda.max_memory_allocated peak during training; model weight file sizes listed separately.
Output: results/p0_5_cost.csv + console.
"""
import csv, io, json, os, sys, time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p0_2_alpha_sweep import (OSS, DATA, SRC_CACHE, build_windows_ds, concat_cells,
                              std_with, set_seed, CausalBlock, TCN, RNNWrap,
                              new_model, fit_model, predict)  # reuse the v1.2.4 pipeline (everything except the Transformer)


class Transformer(nn.Module):
    """Copied verbatim from code/baselines/t2_train_local.py (v1.2.4)."""
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


def new_model5(name, input_dim):
    if name == "transformer":
        return Transformer(input_dim)
    return new_model(name, input_dim)


OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "results_p0")
os.makedirs(OUT, exist_ok=True)
LOG = open(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "logs_p0", "p0_5_cost.log"),
           "w", encoding="utf-8")


def log(m):
    print(m, flush=True); LOG.write(m + "\n"); LOG.flush()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    props = torch.cuda.get_device_properties(0)
    hw = (f"{props.name}, {props.total_memory/2**30:.1f} GB, "
          f"CUDA {torch.version.cuda}, PyTorch {torch.__version__}")
    log(f"hardware: {hw}")
    df = pd.read_csv(DATA)
    src = build_windows_ds(df, "MIT", horizon=10)
    bids = sorted(src)
    random_like = bids  # timing uses the full source domain (the split difference without early-stopping validation does not change the time scale)
    Xtr, ytr, _ = concat_cells(src, bids[:-13])
    Xva, yva, _ = concat_cells(src, bids[-13:])
    sc = StandardScaler().fit(Xtr.reshape(-1, Xtr.shape[2]))
    Xtr_s, Xva_s = std_with(sc, Xtr), std_with(sc, Xva)
    log(f"source-domain windows: train {len(Xtr)}, val {len(Xva)}, dim {Xtr.shape[2]}")

    rows = []
    for name in ("tcn", "lstm", "gru", "transformer"):
        set_seed(42)
        m = new_model5(name, Xtr.shape[2]).to(device)
        n_par = sum(pp.numel() for pp in m.parameters())
        # training time (full fit_model, manuscript protocol)
        torch.cuda.reset_peak_memory_stats()
        set_seed(42)
        m2 = new_model5(name, Xtr.shape[2]).to(device)
        t0 = time.time()
        fit_model(m2, Xtr_s, ytr, 120, 42, device, Xva_s, yva)
        train_s = time.time() - t0
        peak = torch.cuda.max_memory_allocated() / 2**20
        # inference latency
        m.eval()
        lat1 = lat256 = None
        for bs, tag in ((1, "b1"), (256, "b256")):
            x = torch.randn(bs, 20, Xtr.shape[2], device=device)
            with torch.no_grad():
                for _ in range(10):
                    m(x)
                torch.cuda.synchronize()
                t0 = time.time()
                for _ in range(100):
                    m(x)
                torch.cuda.synchronize()
                lat = (time.time() - t0) / 100 * 1000
            if tag == "b1":
                lat1 = round(lat, 3)
            else:
                lat256 = round(lat, 3)
        import os as _os
        w_file = _os.path.join(OUT, f"p0_5_{name}.pt")
        torch.save(m.state_dict(), w_file)
        rows.append(dict(model=name, params=n_par,
                         params_M=round(n_par / 1e6, 3),
                         train_time_s=round(train_s, 1),
                         peak_mem_MiB=round(peak, 1),
                         latency_b1_ms=lat1, latency_b256_ms=lat256,
                         weight_file_KB=round(_os.path.getsize(w_file) / 1024, 1),
                         hardware=hw))
        log(f"[{name}] params={n_par} ({n_par/1e6:.3f}M) train={train_s:.1f}s "
            f"peak={peak:.0f}MiB b1={rows[-1]['latency_b1_ms']}ms b256={rows[-1]['latency_b256_ms']}ms")
    with open(os.path.join(OUT, "p0_5_cost.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    log("P0-5 done -> p0_5_cost.csv")


if __name__ == "__main__":
    main()
