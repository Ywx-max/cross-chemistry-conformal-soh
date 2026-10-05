# -*- coding: utf-8 -*-
"""T4c-Weighted: weighted conformal under covariate shift (Tibshirani et al., 2019; second Section 4.5 extension).

Quantiles are computed after weighting each calibration block by "how test-like" it is. The density ratio
w(x)=p_test(x)/p_cal(x) is estimated between calibration and test blocks inside the target domain (not source vs
target), with x the block-mean predicted SOH: at deployment a test block has only predictions, no labels, so the
weights may use prediction-side covariates only and must not touch test labels (that would be label leakage).

Result: on NASA the predicted-SOH distributions of calibration and test blocks nearly coincide, so the normalised
weights degenerate to uniform and match the single quantile; on CALCE width narrows but coverage falls in step. Under
small calibration sets, conditional narrowing trades coverage for width; see Section 4.9.

Run: python t4c_weighted_local.py --seed 42 (optional --src-cache <dir> to reuse cached source models)
Output: results/conformal/t4c_weighted_s<seed>.json"""
import argparse, json, os, random, time
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.preprocessing import StandardScaler
OUT = "results/conformal"
# soh must be dropped: it is the label of the SOH task, and a label in the inputs hands the answer to the model
# (before the 2026-10-03 fix an identity-copy baseline scored RMSE=0, the leak in plain sight).
FEATS=['capacity_Ah','discharge_dur_s','v_mean_V','v_min_V','ica_peak','ica_peak_V']
# H=10: label = soh at H cycles after the last row of the window (H-step-ahead prediction)
W=20; DATA="data/modeling_table_v3.csv"; ALPHA=0.10; NB=3; H=10

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def build_windows(df, W=20, H=10):
    cells={}
    for bid,g in df.sort_values('cycle').groupby('battery_id'):
        f=g[FEATS].astype(float).copy()
        f=f.interpolate(limit_direction='both').fillna(0.0).values
        cyc=g['cycle'].values; y=g['soh'].astype(float).values
        ok=~np.isnan(y); f,cyc,y=f[ok],cyc[ok],y[ok]
        X,yy=[],[]
        for i in range(W-1,len(f)-H):
            if cyc[i]-cyc[i-W+1]==W-1 and cyc[i+H]==cyc[i]+H:
                X.append(f[i-W+1:i+1]); yy.append(y[i+H])
        if X: cells[bid]=(np.asarray(X,np.float32),np.asarray(yy,np.float32))
    return cells

class CB(nn.Module):
    def __init__(s,ic,oc,k,d):
        super().__init__(); s.p=(k-1)*d
        s.conv=nn.Conv1d(ic,oc,k,dilation=d,padding=s.p)
        s.res=nn.Conv1d(ic,oc,1) if ic!=oc else nn.Identity()
    def forward(s,x):
        y=s.conv(x)
        if s.p: y=y[:,:,:-s.p]
        return torch.relu(y+s.res(x))

class TCN(nn.Module):
    def __init__(s,ic,d=64,L=4):
        super().__init__(); ch=[ic]+[d]*L
        s.blocks=nn.ModuleList([CB(ch[i],ch[i+1],3,2**i) for i in range(L)])
        s.fc=nn.Linear(d,1)
    def forward(s,x):
        h=x.transpose(1,2)
        for b in s.blocks: h=b(h)
        return s.fc(h[:,:,-1]).squeeze(-1)

def fit(m,X,y,ep,s,d,Xv=None,yv=None,lr=1e-3):
    set_seed(s); opt=torch.optim.Adam(m.parameters(),lr=lr)
    lf=nn.MSELoss(); ds=torch.utils.data.TensorDataset(torch.tensor(X,dtype=torch.float32),torch.tensor(y,dtype=torch.float32))
    dl=torch.utils.data.DataLoader(ds,batch_size=256,shuffle=True)
    Xv=X if Xv is None else Xv; yv=y if yv is None else yv
    best,bst,pat=float('inf'),None,0
    for e in range(ep):
        m.train()
        for xb,yb in dl:
            xb,yb=xb.to(d),yb.to(d); opt.zero_grad(); lf(m(xb),yb).backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
        m.eval()
        with torch.no_grad(): va=lf(m(torch.tensor(Xv,dtype=torch.float32).to(d)),torch.tensor(yv,dtype=torch.float32).to(d)).item()
        if np.isfinite(va) and va<best-1e-6: best,pat=va,0; bst={k:v.clone() for k,v in m.state_dict().items()}
        else: pat+=1
        if pat>=10: break
    if bst: m.load_state_dict(bst)
    return m

def pred(m,X,d):
    m.eval()
    with torch.no_grad(): return m(torch.tensor(X,dtype=torch.float32).to(d)).cpu().numpy()

def sw(sc,X): return ((X-sc.mean_)/(sc.scale_+1e-8)).astype(np.float32)
def cc(cells,bids):
    X=np.concatenate([cells[b][0] for b in bids]); y=np.concatenate([cells[b][1] for b in bids])
    return X,y
def cq(res,a):
    # single-quantile baseline (same formula as t4_conformal.conformal_q), the control for weighted
    n=len(res); idx=min(n-1,int(np.ceil((n+1)*(1-a)))-1)
    return float(np.sort(res)[idx])
def chunk_mean(arr, w=20):
    n=len(arr)//w
    return np.array([np.mean(arr[i*w:(i+1)*w]) for i in range(n)])
def gaussian_w(cal_soh, te_soh):
    # density ratio w(x) = p_target(x) / p_cal(x) with x the block-mean SOH, both distributions
    # Gaussian. Dividing by the mean at the end normalises the weights so they express relative importance only and stay stable
    # (+1e-8 guards std=0, which really happens when NASA has few calibration blocks at nearly constant SOH)
    mu_s,std_s=np.mean(cal_soh),np.std(cal_soh)+1e-8
    mu_t,std_t=np.mean(te_soh),np.std(te_soh)+1e-8
    w=(std_s/std_t)*np.exp(-0.5*((cal_soh-mu_t)/std_t)**2+0.5*((cal_soh-mu_s)/std_s)**2)
    return w/np.mean(w)
def weighted_cq(res,wts,a):
    # weighted quantile: accumulate weights over residuals in ascending order and take the residual where the
    # cumulative weight first reaches (1-alpha)*(n+1)/n * total weight. The (n+1)/n factor makes uniform weights
    # degenerate to the ceil((n+1)(1-alpha)) order statistic, the same scope as cq().
    order=np.argsort(res); sr=res[order]; sw2=wts[order]
    cw=np.cumsum(sw2); tw=cw[-1]
    n=len(res)
    threshold=(1-a)*(n+1)/n*tw
    idx=np.searchsorted(cw,threshold)
    idx=min(idx,len(sr)-1)
    return float(sr[idx])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data", default=DATA, help="modeling-table csv path")
    ap.add_argument("--out", default="results/conformal", help="results output directory")
    ap.add_argument("--src-cache", default=None,
                    help="unified source-model cache directory; a hit skips source pre-training")
    ap.add_argument("--horizon", type=int, default=H,
                    help="horizon: label = soh at H cycles after the last window row")
    args = ap.parse_args()
    SEED = args.seed
    OUT = args.out
    device='cuda' if torch.cuda.is_available() else 'cpu'
    df=pd.read_csv(args.data)
    src=build_windows(df[df['dataset']=='MIT'], H=args.horizon)
    sb=sorted(src); random.Random(SEED).shuffle(sb)
    nv=max(1,int(len(sb)*0.1)); Xtr,ytr=cc(src,sb[:-nv]); Xva,yva=cc(src,sb[-nv:])
    sc=StandardScaler().fit(Xtr.reshape(-1,Xtr.shape[2]))
    set_seed(SEED)  # seed before instantiating: otherwise same-seed reruns initialise different weights
    model=TCN(Xtr.shape[2]).to(device)
    cache_p = None
    if args.src_cache:
        os.makedirs(args.src_cache, exist_ok=True)
        _hit = os.path.join(args.src_cache, "src_tcn_s%d_ep120_h%d.pt" % (SEED, args.horizon))
        _legacy = os.path.join(args.src_cache, "t4d_src_tcn_s%d.pt" % SEED)
        cache_p = _hit if os.path.exists(_hit) else (_legacy if os.path.exists(_legacy) else None)
    if cache_p and os.path.exists(cache_p):
        model.load_state_dict(torch.load(cache_p, map_location=device, weights_only=True))
        print('[cache] load source model %s' % cache_p, flush=True)
    else:
        t0=time.time()
        model=fit(model,sw(sc,Xtr),ytr,120,SEED,device,sw(sc,Xva),yva)
        print('SRC done %ds' % (time.time()-t0),flush=True)
        if args.src_cache:
            _save=os.path.join(args.src_cache,'src_tcn_s%d_ep120_h%d.pt' % (SEED, args.horizon))
            torch.save(model.state_dict(), _save)
            print('[cache] source model written to %s' % _save, flush=True)
    results={'horizon':args.horizon,'targets':{}}
    for tg in ['CALCE','NASA']:
        tgt=build_windows(df[df['dataset']==tg], H=args.horizon)
        tbg=sorted(tgt); random.Random(SEED).shuffle(tbg)
        # exactly the same third-split protocol as t4c_mondrian (CALCE 2/2/4, NASA 1/1/2),
        # so the two extension controls are comparable
        nf=max(1,len(tbg)//3)
        ft_b,cal_b,te_b=tbg[:nf],tbg[nf:nf*2],tbg[nf*2:]
        Xa,ya=cc(tgt,tbg); sct=StandardScaler().fit(Xa.reshape(-1,Xa.shape[2]))
        Xft,yft=cc(tgt,ft_b)
        ftm=TCN(Xtr.shape[2]).to(device); ftm.load_state_dict(model.state_dict())
        ftm=fit(ftm,sw(sct,Xft),yft,60,SEED,device,lr=3e-4)
        Xcal,ycal=cc(tgt,cal_b); pcal=pred(ftm,sw(sct,Xcal),device)
        cres=np.abs(pcal-ycal)
        cres_c=chunk_mean(cres,W); n_c=len(cres_c)
        cal_soh_c=np.array([np.mean(ycal[i*W:(i+1)*W]) for i in range(n_c)])
        q_single=cq(cres_c,ALPHA)
        Xte,yte=cc(tgt,te_b); pte=pred(ftm,sw(sct,Xte),device)
        te_res=np.abs(pte-yte)
        te_res_c=chunk_mean(te_res,W); n_t=len(te_res_c)
        # the test-side "SOH" used by the weights is the block mean of model predictions (available at deployment), never test labels:
        # the density ratio is estimated between calibration and test blocks inside the target domain (a transductive setting)
        te_pred_c=chunk_mean(pte,W)
        picp_single=float(np.mean(te_res_c<=q_single)); mpiw_single=2*q_single
        # weights apply to calibration blocks only; test blocks need none (coverage evaluation is just "did it fall inside the interval")
        w=gaussian_w(cal_soh_c, te_pred_c)
        q_weighted=weighted_cq(cres_c, w, ALPHA)
        picp_weighted=float(np.mean(te_res_c<=q_weighted)); mpiw_weighted=2*q_weighted
        rmse=float(np.sqrt(np.mean((pte-yte)**2)))
        results['targets'][tg]={
            'single':{'PICP':picp_single,'MPIW':mpiw_single,'q':q_single},
            'weighted':{'PICP':picp_weighted,'MPIW':mpiw_weighted,'q':q_weighted},
            'point_rmse':rmse}
        print(f'[{tg}] RMSE={rmse:.4f} single_PICP={picp_single:.2f} MPIW={mpiw_single:.4f} weighted_PICP={picp_weighted:.2f} MPIW={mpiw_weighted:.4f}',flush=True)
    os.makedirs(OUT,exist_ok=True)
    json.dump(results,open(OUT+f'/t4c_weighted_s{SEED}.json','w'),indent=1)
    print('T4C DONE',flush=True)
if __name__=='__main__': main()
