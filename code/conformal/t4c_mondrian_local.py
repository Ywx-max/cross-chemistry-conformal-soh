# -*- coding: utf-8 -*-
"""T4c-Mondrian: conditional conformal by SOH binning (one of the Section 4.5 extension controls).

Calibration windows are split into 3 bins by SOH level with a per-bin conformal quantile, letting interval
width track local error. Result: on NASA width narrows by about 21% while coverage drops to 0.64+/-0.18; on CALCE
neither width nor coverage improves materially. Coverage fixes seen in single runs do not survive multi-seed averaging.

Implementation note: residuals and SOH are first averaged into blocks of W=20 consecutive cycles before binning,
so per-window SOH jitter cannot scramble the bins.

Run: python t4c_mondrian_local.py --seed 42
Output: results/conformal/t4c_mondrian_s<seed>.json (single = single-quantile baseline control)"""
import json, os, random, time, argparse
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.preprocessing import StandardScaler
OUT = "results/conformal"
# soh must be dropped: it is the label of the SOH task, and a label in the inputs hands the answer to the model
# (before the 2026-10-03 fix an identity-copy baseline scored RMSE=0, the leak in plain sight).
FEATS=['capacity_Ah','discharge_dur_s','v_mean_V','v_min_V','ica_peak','ica_peak_V']
# NB=3: three SOH bins (cut at the 1/3 and 2/3 quantiles). With only 1-2 calibration cells each bin holds very few samples,
# which is the direct reason Mondrian fails on small calibration sets
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
class CausalBlock(nn.Module):
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
        s.blocks=nn.ModuleList([CausalBlock(ch[i],ch[i+1],3,2**i) for i in range(L)])
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
    # same formula as t4_conformal.conformal_q; min(n-1, ...) is the fallback for tiny n
    n=len(res); idx=min(n-1,int(np.ceil((n+1)*(1-a)))-1)
    return float(np.sort(res)[idx])
def chunk_res(res, W=20):
    # the mean of each W consecutive cycles is one block; the tail shorter than W is dropped,
    # so calibration and test ends up with naturally aligned block counts.
    n=len(res)//W
    return np.array([np.mean(res[i*W:(i+1)*W]) for i in range(n)])
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--data', default=DATA)
    ap.add_argument('--out', default="results/conformal")
    ap.add_argument('--src-cache', default=None)
    ap.add_argument('--horizon', type=int, default=H,
                    help='horizon: label = soh at H cycles after the last window row')
    args=ap.parse_args()
    SEED=args.seed
    OUT=args.out
    device='cuda' if torch.cuda.is_available() else 'cpu'
    df=pd.read_csv(args.data)
    src=build_windows(df[df['dataset']=='MIT'], H=args.horizon)
    sb=sorted(src); random.Random(SEED).shuffle(sb)
    nv=max(1,int(len(sb)*0.1)); Xtr,ytr=cc(src,sb[:-nv]); Xva,yva=cc(src,sb[-nv:])
    sc=StandardScaler().fit(Xtr.reshape(-1,Xtr.shape[2]))
    set_seed(SEED)  # seed before instantiating: otherwise same-seed reruns initialise different weights
    model=TCN(Xtr.shape[2]).to(device)
    _cache_hit=None
    if args.src_cache:
        os.makedirs(args.src_cache, exist_ok=True)
        _hit=os.path.join(args.src_cache,'src_tcn_s%d_ep120_h%d.pt' % (SEED, args.horizon))
        _legacy=os.path.join(args.src_cache,'t4d_src_tcn_s%d.pt' % SEED)
        _cache_hit=_hit if os.path.exists(_hit) else (_legacy if os.path.exists(_legacy) else None)
    if _cache_hit:
        model.load_state_dict(torch.load(_cache_hit, map_location=device, weights_only=True))
        print('[cache] load source model %s' % _cache_hit, flush=True)
    else:
        t0=time.time()
        model=fit(model,sw(sc,Xtr),ytr,120,SEED,device,sw(sc,Xva),yva)
        print('SRC done %ds'%(time.time()-t0),flush=True)
        if args.src_cache:
            _save=os.path.join(args.src_cache,'src_tcn_s%d_ep120_h%d.pt' % (SEED, args.horizon))
            torch.save(model.state_dict(), _save)
            print('[cache] source model written to %s' % _save, flush=True)
    results={'horizon':args.horizon,'targets':{}}
    for tg in ['CALCE','NASA']:
        tgt=build_windows(df[df['dataset']==tg], H=args.horizon)
        tbg=sorted(tgt); random.Random(SEED).shuffle(tbg)
        # third-split protocol: fine-tune/calibration/test cut 1/3 each (different from the hard-coded Table 5 splits;
        # this is the "different cell-split protocol" declared in the Section 4.5 extensions). Actual cuts: CALCE 16 cells = 5/5/6,
        # NASA 4 cells = 1/1/2 (dynamic len//3 thirds); both ends share one random shuffle, so the split is reproducible per seed
        nf=max(1,len(tbg)//3)
        ft_b,cal_b,te_b=tbg[:nf],tbg[nf:nf*2],tbg[nf*2:]
        Xa,ya=cc(tgt,tbg); sct=StandardScaler().fit(Xa.reshape(-1,Xa.shape[2]))
        Xft,yft=cc(tgt,ft_b)
        ftm=TCN(Xtr.shape[2]).to(device); ftm.load_state_dict(model.state_dict())
        ftm=fit(ftm,sw(sct,Xft),yft,60,SEED,device,lr=3e-4)
        Xcal,ycal=cc(tgt,cal_b); pcal=pred(ftm,sw(sct,Xcal),device)
        cres=np.abs(pcal-ycal)
        cres_c=chunk_res(cres,W); n_c=len(cres_c)
        cal_soh_c=np.array([np.mean(ycal[i*W:(i+1)*W]) for i in range(n_c)])
        q1=cq(cres_c,ALPHA)
        Xte,yte=cc(tgt,te_b); pte=pred(ftm,sw(sct,Xte),device)
        tres=np.abs(pte-yte)
        tres_c=chunk_res(tres,W); n_t=len(tres_c)
        te_soh_c=np.array([np.mean(yte[i*W:(i+1)*W]) for i in range(n_t)])
        picp_s=float(np.mean(tres_c<=q1)); mpiw_s=2*q1
        # binning conformal proper: three bins by block-mean SOH, an independent quantile per bin
        be=np.quantile(cal_soh_c,[1/NB,2/NB])
        pl,wl=[],[]
        # bins with cm.sum()<2 are dropped outright: a quantile from two or three blocks is statistically meaningless
        for b in range(NB):
            if b==0: cm=cal_soh_c<=be[0]; tm=te_soh_c<=be[0]
            elif b==NB-1: cm=cal_soh_c>be[-1]; tm=te_soh_c>be[-1]
            else: cm=(cal_soh_c>be[b-1])&(cal_soh_c<=be[b]); tm=(te_soh_c>be[b-1])&(te_soh_c<=be[b])
            if cm.sum()<2: continue
            qb=cq(cres_c[cm],ALPHA)
            if tm.sum()>0: pl.append(float(np.mean(tres_c[tm]<=qb))); wl.append(2*qb)
        picp_m=float(np.mean(pl)) if pl else float('nan')
        mpiw_m=float(np.mean(wl)) if wl else float('nan')
        rmse=float(np.sqrt(np.mean((pte-yte)**2)))
        results['targets'][tg]={
            'single':{'PICP':picp_s,'MPIW':mpiw_s},
            'mondrian':{'PICP':picp_m,'MPIW':mpiw_m},
            'point_rmse':rmse}
        print(f'[{tg}] RMSE={rmse:.4f} single_PICP={picp_s:.2f} MPIW={mpiw_s:.4f} Mondrian_PICP={picp_m:.2f} MPIW={mpiw_m:.4f}',flush=True)
    os.makedirs(OUT,exist_ok=True)
    json.dump(results,open(OUT+f'/t4c_mondrian_s{SEED}.json','w'),indent=1)
    print('T4C DONE',flush=True)
if __name__=='__main__': main()
