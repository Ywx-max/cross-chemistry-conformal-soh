# -*- coding: utf-8 -*-
"""T4 dual-route conformal comparison (manuscript Table 5).

Same fine-tuned model, same test windows; only the source of the calibration quantile changes:
  source-calibrated (control):  quantile q_src from source-domain validation residuals
  target-calibrated (this work): quantile q_tgt from cycle-level residuals of 1-2 target cells

Convention: q_tgt comes from the residuals of the fine-tuned (deployed) model on the calibration
cells. Split-conformal finite-sample coverage requires the calibration score and the test score
to come from one prediction function (calibrate with the model you evaluate); a quantile taken from another model invalidates the coverage evaluation.
Result: source-calibrated intervals under-cover in all 20 configurations (PICP 0.0000-0.2518);
target recalibration lifts coverage to 0.5326-1.0000 but stays mostly below nominal 0.90: a calibration set of 1-2 cells cannot support nominal coverage.

Scope note (stated at the end of Section 3.4): what is reported here is marginal empirical coverage.
Cycle-level residuals are autocorrelated, strict exchangeability does not hold, so this is empirical evidence rather than a finite-sample guarantee.
Per-cell diagnostics and aggregation variants live in t4d_per_cell_diag.py.

Splits: CALCE (7 fine-tune, 2 calibration, 7 test), NASA (2, 1, 1); splits are redrawn per seed.
Run: python t4_conformal_local.py --model tcn --seed 42 (manuscript scope: one run per seed, 42-46)
Optional: --src-cache <dir> pointing at a directory with t4d_src_<model>_s<seed>.pt to skip source pre-training
      (the cache is shared with the t4d diagnostics and follows this script's protocol; otherwise the source model is trained on the spot).
Optional: --zs-only zero-shot route run (the Section 3.4 control): skips fine-tuning and evaluates the
      source-calibrated intervals on the target test windows with the source model itself (same q_src,
      same split and standardization; only the evaluation residuals switch to the zero-shot model),
      written to t4zs_<model>_s<seed>.json; the default behaviour and existing results are unchanged.
Output: results/conformal/t4_<model>_s<seed>.json (with per-cell residual vectors)"""
import argparse, json, os, random, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

# soh must be dropped: it is the label of the SOH task, and a label in the inputs hands the answer to the model
# (before the 2026-10-03 fix an identity-copy baseline scored RMSE=0, the leak in plain sight).
FEATS = ["capacity_Ah", "discharge_dur_s", "v_mean_V", "v_min_V",
         "ica_peak", "ica_peak_V"]
WINDOW = 20
# horizon: label = soh at H cycles after the last row of the window (H-step-ahead prediction).
# Windows closer than H steps to the record end, or with cycle gaps between window and label, are dropped.
H = 10
# modeling-table path (data goes under data/; merge procedure in code/README.md)
DATA = "data/modeling_table_v3.csv"
ALPHA = 0.10

def set_seed(seed):
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

def predict(model, X, device):
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(X, dtype=torch.float32).to(device)).cpu().numpy()

def std_with(sc, X):
    return ((X - sc.mean_) / (sc.scale_ + 1e-8)).astype(np.float32)

def concat_cells(cells, bids):
    X = np.concatenate([cells[b][0] for b in bids])
    y = np.concatenate([cells[b][1] for b in bids])
    return X, y, [b for b in bids for _ in range(len(cells[b][0]))]

def conformal_q(residuals, alpha=ALPHA):
    """Empirical split-conformal quantile: the ceil((n+1)(1-alpha))-th order statistic.

    The min(n-1, ...) clamp is an engineering fallback: when n is tiny (a NASA calibration set of one
    cell gives only 1-2 cell-level scores), (n+1)(1-alpha) exceeds n and the index is clamped to the
    maximum, i.e. the worst calibration sample. Be aware that in that regime the finite-sample
    guarantee of conformal prediction is empty anyway and the interval is empirical only."""
    n = len(residuals)
    idx = min(n - 1, int(np.ceil((n + 1) * (1 - alpha))) - 1)
    return float(np.sort(residuals)[idx])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="tcn")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--ft-epochs", type=int, default=60)
    ap.add_argument("--horizon", type=int, default=H,
                    help="horizon: label = soh at H cycles after the last window row")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results/conformal")
    ap.add_argument("--data", default=DATA, help="modeling-table csv path")
    ap.add_argument("--src-cache", default=None,
                    help="unified source-model cache directory; a hit skips source pre-training")
    ap.add_argument("--split-calce", default="7,2,7",
                    help="CALCE fine-tune/calibration/test cell counts (manuscript protocol 7/2/7)")
    ap.add_argument("--split-nasa", default="2,1,1",
                    help="NASA fine-tune/calibration/test cell counts")
    ap.add_argument("--n-cal", type=int, default=None,
                    help="sweep the calibration-cell count (Section 4.5 coverage-vs-calibration study): when given, CALCE uses the "
                         "(16-7-N, N, 7) split (7 test cells fixed, fine-tuning cells shrink with the calibration count); NASA is not swept; results go to "
                         "t4_<model>_s<seed>_cal<N>.json; the default keeps the Table 5 splits unchanged")
    ap.add_argument("--zs-only", action="store_true",
                    help="zero-shot route run (the Section 3.4 control): skip fine-tuning and evaluate source-calibrated interval coverage "
                         "with the source model on the target test windows; written to t4zs_<model>_s<seed>.json")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"T4 conformal device={device} model={args.model} alpha={ALPHA}", flush=True)
    df = pd.read_csv(args.data)
    src = build_windows_ds(df, "MIT", horizon=args.horizon)
    src_bids = sorted(src)
    random.Random(args.seed).shuffle(src_bids)
    n_va = max(1, int(len(src_bids) * 0.1))
    Xtr, ytr, _ = concat_cells(src, src_bids[:-n_va])
    Xva, yva, _ = concat_cells(src, src_bids[-n_va:])
    sc_src = StandardScaler().fit(Xtr.reshape(-1, Xtr.shape[2]))
    Xtr, Xva = std_with(sc_src, Xtr), std_with(sc_src, Xva)
    set_seed(args.seed)  # seed before instantiating: otherwise same-seed reruns initialise different weights
    model = new_model(args.model, Xtr.shape[2]).to(device)
    cache_p = None
    if args.src_cache:
        os.makedirs(args.src_cache, exist_ok=True)
        _hit = os.path.join(args.src_cache,
                            f"src_{args.model}_s{args.seed}_ep{args.epochs}_h{args.horizon}.pt")
        _legacy = os.path.join(args.src_cache, f"t4d_src_{args.model}_s{args.seed}.pt")
        cache_p = _hit if os.path.exists(_hit) else (_legacy if os.path.exists(_legacy) else None)
        if cache_p:
            model.load_state_dict(torch.load(cache_p, map_location=device, weights_only=True))
            print(f"[cache] load source model {cache_p}", flush=True)
        else:
            print(f"[cache] {args.src_cache} has no hit; training on the spot", flush=True)
    if cache_p is None:
        t0 = time.time()
        model = fit_model(model, Xtr, ytr, args.epochs, args.seed, device, Xva, yva)
        print(f"source-domain training done in {time.time()-t0:.0f}s", flush=True)
        if args.src_cache:
            _save = os.path.join(args.src_cache,
                                 f"src_{args.model}_s{args.seed}_ep{args.epochs}_h{args.horizon}.pt")
            torch.save(model.state_dict(), _save)
            print(f"[cache] source model written to {_save}", flush=True)
    # source-domain calibration residuals (used by the naive route)
    # quantile of the naive route: absolute residuals on the source validation set. In-domain fits are
    # near perfect, so q_src is very small, which foreshadows its wholesale failure in the target domain
    src_cal_res = np.abs(predict(model, Xva, device) - yva)
    q_src = conformal_q(src_cal_res)
    print(f"source-domain calibration quantile q_src={q_src:.4f}", flush=True)

    results = {"model": args.model, "alpha": ALPHA, "q_src": q_src,
               "horizon": args.horizon, "targets": {}}
    _splits = {"CALCE": tuple(int(x) for x in args.split_calce.split(",")),
               "NASA": tuple(int(x) for x in args.split_nasa.split(","))}
    for tgt_name, split in [("CALCE", _splits["CALCE"]), ("NASA", _splits["NASA"])]:
        if args.n_cal is not None and tgt_name == "CALCE":
            # coverage-vs-calibration sweep: 7 test cells fixed, fine-tuning = 16-7-cal,
            # calibration levels 1..5 share one test-cell set so the coverage curve is comparable
            split = (16 - 7 - args.n_cal, args.n_cal, 7)
            if split[0] < 1 or split[1] < 1 or split[2] < 1:
                print(f"[{tgt_name}] n_cal={args.n_cal} invalid split, skipped", flush=True)
                continue
        tgt = build_windows_ds(df, tgt_name, horizon=args.horizon)
        tb = sorted(tgt)
        n_ft, n_cal, n_te = split
        if len(tb) < n_ft + n_cal + n_te:
            print(f"[{tgt_name}] too few cells, skipped", flush=True)
            continue
        random.Random(args.seed).shuffle(tb)
        ft_b, cal_b, te_b = tb[:n_ft], tb[n_ft:n_ft+n_cal], tb[n_ft+n_cal:]
        Xall_t, _, _ = concat_cells(tgt, tb)
        sc_tgt = StandardScaler().fit(Xall_t.reshape(-1, Xall_t.shape[2]))
        if args.zs_only:
            # zero-shot route run: prediction function and calibration scores both come from the source
            # model (one q_src); only the evaluation data sits in the target domain, switching residuals from fine-tuned to zero-shot.
            Xte, yte, _ = concat_cells(tgt, te_b)
            Xte_s = std_with(sc_tgt, Xte)
            pred_zs = predict(model, Xte_s, device)
            res_zs = np.abs(pred_zs - yte)
            cov_zs = float(np.mean(res_zs <= q_src))
            rmse_zs = float(np.sqrt(np.mean(res_zs ** 2)))
            med_zs = float(np.median(res_zs))
            per_cell_zs, idx0 = {}, 0
            for b in te_b:
                Xb, yb, _ = concat_cells(tgt, [b])
                rb = res_zs[idx0:idx0 + len(yb)]; idx0 += len(yb)
                per_cell_zs[b] = {"n_windows": int(len(rb)),
                                  "cov_src_zeroshot": float(np.mean(rb <= q_src))}
            results["targets"][tgt_name] = {
                "split": {"cal": cal_b, "te": te_b},
                "q_src": q_src, "MPIW_src": 2 * q_src,
                "zero_shot": {"RMSE": rmse_zs, "med_abs_r": med_zs,
                              "PICP_src_zeroshot": cov_zs},
                "per_cell": per_cell_zs}
            print(f"[{tgt_name}][zs] RMSE_zs={rmse_zs:.4f} med|r|={med_zs:.4f} "
                  f"source-calibrated coverage (zs)={cov_zs:.4f} (nominal {1-ALPHA:.2f})", flush=True)
            continue
        Xft, yft, _ = concat_cells(tgt, ft_b)
        ft_model = new_model(args.model, Xtr.shape[2]).to(device)
        ft_model.load_state_dict(model.state_dict())
        ft_model = fit_model(ft_model, std_with(sc_tgt, Xft), yft,
                             args.ft_epochs, args.seed, device, lr=3e-4)
        Xcal, ycal, _ = concat_cells(tgt, cal_b)
        Xcal = std_with(sc_tgt, Xcal)
        # target-domain calibration residuals: from the fine-tuned deployed model on the calibration cells (calibration and evaluation share the model).
        # calibration cells cal_b and fine-tuning cells ft_b are disjoint, so the fine-tuned model has never seen the calibration cells
        cal_res = np.abs(predict(ft_model, Xcal, device) - ycal)
        q_tgt = conformal_q(cal_res)
        Xte, yte, _ = concat_cells(tgt, te_b)
        Xte_s = std_with(sc_tgt, Xte)
        pred = predict(ft_model, Xte_s, device)
        res_te = np.abs(pred - yte)
        # target-domain calibration route (the proposed method)
        # coverage over all pooled test windows (marginal scope); MPIW = mean full interval width = 2q
        cov_tgt = float(np.mean(res_te <= q_tgt)); w_tgt = 2 * q_tgt
        # source-domain calibration route (naive control)
        cov_src = float(np.mean(res_te <= q_src)); w_src = 2 * q_src
        # per-cell residuals and coverage: calibration cells store deployed-model residual vectors, test cells store per-window
        # coverage flags and RMSE, so later diagnostics need no retraining
        idx0 = 0
        per_cell = {}
        for b in te_b:
            Xb, yb, _ = concat_cells(tgt, [b])
            res_b = res_te[idx0:idx0 + len(yb)]; idx0 += len(yb)
            per_cell[b] = {"n_windows": int(len(res_b)),
                           "residuals": [float(v) for v in res_b],
                           "cov_tgt": float(np.mean(res_b <= q_tgt)),
                           "cov_src": float(np.mean(res_b <= q_src)),
                           "rmse": float(np.sqrt(np.mean(res_b ** 2)))}
        cal_cells_res = {}
        off = 0
        for b in cal_b:
            Xb, yb, _ = concat_cells(tgt, [b])
            cal_cells_res[b] = [float(v) for v in cal_res[off:off + len(yb)]]
            off += len(yb)
        results["targets"][tgt_name] = {
            "split": {"ft": ft_b, "cal": cal_b, "te": te_b},
            "q_target": q_tgt, "q_src": q_src,
            "point_rmse": float(np.sqrt(np.mean((pred - yte) ** 2))),
            "target_calibrated": {"PICP": cov_tgt, "MPIW": w_tgt},
            "source_calibrated": {"PICP": cov_src, "MPIW": w_src},
            "per_cell": per_cell,
            "cal_cells_residuals": cal_cells_res}
        print(f"[{tgt_name}] point RMSE={results['targets'][tgt_name]['point_rmse']:.4f} | "
              f"target-cal: PICP={cov_tgt:.2f} MPIW={w_tgt:.4f} | "
              f"source-cal: PICP={cov_src:.2f} MPIW={w_src:.4f} (nominal {1-ALPHA:.2f})", flush=True)
    os.makedirs(args.out, exist_ok=True)
    suffix = "_cal%d" % args.n_cal if args.n_cal is not None else ""
    tag = "t4zs" if args.zs_only else "t4"
    with open(f"{args.out}/{tag}_{args.model}_s{args.seed}{suffix}.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1, ensure_ascii=False)
    print("T4 DONE", flush=True)

if __name__ == "__main__":
    main()
