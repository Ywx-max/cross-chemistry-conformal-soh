# -*- coding: utf-8 -*-
"""P0-2: 区间指标与 alpha 扫描（Winkler / pinball / NMPIW / PICP / MPIW）。

协议复现：逐字复制开源仓库 v1.2.4 (b6c3e39) code/conformal/t4_conformal_local.py 的
数据管线 / 模型 / 训练 / 划分逻辑（build_windows_ds, TCN, RNNWrap, fit_model, predict,
std_with, concat_cells, conformal_q, set_seed），仅增加：
  1) 测试窗口的 y_pred/y_true 落盘（npz），供 reliability diagram 与 Fig.6 复用；
  2) alpha ∈ {0.05, 0.10, 0.20} 扫描（校准分位数按 conformal_q 重算）；
  3) Winkler / pinball(tau=alpha/2,1-alpha/2) / NMPIW 指标（target 与 source 两路由）。
alpha=0.10 的 PICP/MPIW 与论文表 5 对照（同协议重算，训练非确定性致 0.2% 级漂移，
与 zeroshot_route.py 先例同口径）。

数据（版本隔离）：data/建模表_v3_cx2.csv；源模型缓存 results_cx2/src_cache（不重训源域）。
输出：results/p0_2_metrics.csv、results/p0_2_raw/<model>_s<seed>.npz、logs/p0_2.log
运行：python p0_2_alpha_sweep.py（开源仓库根目录为 cwd）
"""
import csv, io, json, os, random, sys, time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

# ---------- 版本隔离路径 ----------
OSS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
P0 = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(OSS, "data", "建模表_v3_cx2.csv")
SRC_CACHE = os.path.join(OSS, "results_cx2", "src_cache")
OUT_RES = os.path.join(P0, "results_p0")
OUT_RAW = os.path.join(P0, "results_p0", "p0_2_raw")
OUT_LOG = os.path.join(P0, "logs_p0")
for d in (OUT_RES, OUT_RAW, OUT_LOG):
    os.makedirs(d, exist_ok=True)
LOG = open(os.path.join(OUT_LOG, "p0_2.log"), "w", encoding="utf-8")

def log(msg):
    print(msg, flush=True)
    LOG.write(msg + "\n")
    LOG.flush()

# ========== 以下函数逐字复制自 code/conformal/t4_conformal_local.py (v1.2.4 b6c3e39) ==========
FEATS = ["capacity_Ah", "discharge_dur_s", "v_mean_V", "v_min_V",
         "ica_peak", "ica_peak_V"]
WINDOW = 20
H = 10
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
    n = len(residuals)
    idx = min(n - 1, int(np.ceil((n + 1) * (1 - alpha))) - 1)
    return float(np.sort(residuals)[idx])
# ========== 复制结束 ==========

ALPHAS = [0.05, 0.10, 0.20]


def metrics(y, pred, q, alpha):
    r = np.abs(pred - y)
    picp = float(np.mean(r <= q))
    mpiw = 2.0 * q
    nmpiw = mpiw / float(np.mean(np.abs(y)))
    # Winkler：区间 [pred-q, pred+q]，超界罚 2/alpha 倍宽度
    viol = np.maximum(r - q, 0.0)
    winkler = float(np.mean(2 * q + (2.0 / alpha) * viol))
    # pinball：lower tau=alpha/2、upper tau=1-alpha/2 的平均分位数损失
    tau_l, tau_u = alpha / 2.0, 1 - alpha / 2.0
    rl = y - (pred - q)   # 下分位数残差
    ru = y - (pred + q)   # 上分位数残差
    pl = np.maximum(tau_l * rl, (tau_l - 1) * rl)
    pu = np.maximum(tau_u * ru, (tau_u - 1) * ru)
    pinball = float(0.5 * (np.mean(pl) + np.mean(pu)))
    return picp, mpiw, nmpiw, winkler, pinball


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"P0-2 device={device}")
    df = pd.read_csv(DATA)
    rows = []
    t_all = time.time()
    for model in ("tcn", "lstm"):
        for seed in (42, 43, 44, 45, 46):
            t0 = time.time()
            src = build_windows_ds(df, "MIT", horizon=H)
            src_bids = sorted(src)
            random.Random(seed).shuffle(src_bids)
            n_va = max(1, int(len(src_bids) * 0.1))
            Xtr, ytr, _ = concat_cells(src, src_bids[:-n_va])
            Xva, yva, _ = concat_cells(src, src_bids[-n_va:])
            sc_src = StandardScaler().fit(Xtr.reshape(-1, Xtr.shape[2]))
            Xtr_s, Xva_s = std_with(sc_src, Xtr), std_with(sc_src, Xva)
            set_seed(seed)
            model_nn = new_model(model, Xtr.shape[2]).to(device)
            cache_p = os.path.join(SRC_CACHE, f"src_{model}_s{seed}_ep120_h10.pt")
            if os.path.exists(cache_p):
                model_nn.load_state_dict(torch.load(cache_p, map_location=device, weights_only=True))
            else:  # 源缓存未随仓库分发：现场按 Section 3.2 协议重建
                log(f"[cache] miss {cache_p}; retraining source model ...")
                model_nn = fit_model(model_nn, Xtr_s, ytr, 120, seed, device, Xva_s, yva)
            src_cal_res = np.abs(predict(model_nn, Xva_s, device) - yva)
            q_src = conformal_q(src_cal_res)
            raw = {"q_src": q_src, "src_cal_res": src_cal_res}
            for tgt_name, split in (("CALCE", (7, 2, 7)), ("NASA", (2, 1, 1))):
                tgt = build_windows_ds(df, tgt_name, horizon=H)
                tb = sorted(tgt)
                n_ft, n_cal, n_te = split
                random.Random(seed).shuffle(tb)
                ft_b, cal_b, te_b = tb[:n_ft], tb[n_ft:n_ft+n_cal], tb[n_ft+n_cal:]
                Xall_t, _, _ = concat_cells(tgt, tb)
                sc_tgt = StandardScaler().fit(Xall_t.reshape(-1, Xall_t.shape[2]))
                Xft, yft, _ = concat_cells(tgt, ft_b)
                ft_model = new_model(model, Xtr.shape[2]).to(device)
                ft_model.load_state_dict(model_nn.state_dict())
                ft_model = fit_model(ft_model, std_with(sc_tgt, Xft), yft,
                                     60, seed, device, lr=3e-4)
                Xcal, ycal, _ = concat_cells(tgt, cal_b)
                cal_res = np.abs(predict(ft_model, std_with(sc_tgt, Xcal), device) - ycal)
                Xte, yte, _ = concat_cells(tgt, te_b)
                pred = predict(ft_model, std_with(sc_tgt, Xte), device)
                raw[f"{tgt_name}_y_true"] = yte
                raw[f"{tgt_name}_y_pred"] = pred
                raw[f"{tgt_name}_cal_res"] = cal_res
                raw[f"{tgt_name}_q_src"] = q_src
                for alpha in ALPHAS:
                    q_t = conformal_q(cal_res, alpha)
                    q_s = conformal_q(src_cal_res, alpha)   # 源域路由：alpha 也扫描（q_src(alpha)）
                    raw[f"{tgt_name}_q_tgt_a{int(alpha*100)}"] = q_t
                    raw[f"{tgt_name}_q_src_a{int(alpha*100)}"] = q_s
                    for route, q in (("target", q_t), ("source", q_s)):
                        picp, mpiw, nmpiw, winkler, pinball = metrics(yte, pred, q, alpha)
                        rows.append(dict(model=model, seed=seed, domain=tgt_name,
                                         alpha=alpha, route=route, q=q, n_windows=len(yte),
                                         PICP=round(picp, 4), MPIW=round(mpiw, 4),
                                         NMPIW=round(nmpiw, 4), Winkler=round(winkler, 5),
                                         pinball=round(pinball, 6),
                                         point_rmse=round(float(np.sqrt(np.mean((pred-yte)**2))), 5)))
            np.savez_compressed(os.path.join(OUT_RAW, f"{model}_s{seed}.npz"), **raw)
            log(f"[{model} s{seed}] done {time.time()-t0:.0f}s  "
                f"alpha0.10 tgt PICP CALCE={[r['PICP'] for r in rows if r['model']==model and r['seed']==seed and r['alpha']==0.10 and r['route']=='target' and r['domain']=='CALCE']} "
                f"NASA={[r['PICP'] for r in rows if r['model']==model and r['seed']==seed and r['alpha']==0.10 and r['route']=='target' and r['domain']=='NASA']}")
    with open(os.path.join(OUT_RES, "p0_2_metrics.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    log(f"P0-2 全部完成 {time.time()-t_all:.0f}s，共 {len(rows)} 行 -> p0_2_metrics.csv")


if __name__ == "__main__":
    main()
