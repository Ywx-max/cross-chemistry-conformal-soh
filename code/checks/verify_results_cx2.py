# -*- coding: utf-8 -*-
"""结果复算（CX2 版）：只读 results_cx2/ 的结果文件，把扩充后论文的关键数字重算一遍。

    python code/checks/verify_results_cx2.py

口径与 verify_results.py 相同（纯标准库），期望值为投稿版（CX2 扩充）论文所报数值。
旧口径（results/）仍可用 verify_results.py 核对，保持 179/179。
"""
import csv, json, math, os, statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results_cx2"
ROWS, BAD = [], []


def load(rel):
    return json.load(open(RES / rel, encoding="utf-8"))


def jl(rel):
    out = []
    for line in open(RES / rel, encoding="utf-8"):
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def check(label, expect, got, tol):
    ok = abs(expect - got) <= tol
    ROWS.append((label, expect, got, ok))
    if not ok:
        BAD.append(label)
    return ok


def betacf(a, b, x):
    MAXIT, EPS, FPMIN = 200, 3e-16, 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    if abs(d) < FPMIN:
        d = FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < EPS:
            break
    return h


def betai(a, b, x):
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    bt = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                  + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * betacf(a, b, x) / a
    return 1.0 - bt * betacf(b, a, 1.0 - x) / b


def paired_t_p(xs, ys):
    d = [x - y for x, y in zip(xs, ys)]
    n = len(d)
    m = st.mean(d)
    sd = st.stdev(d)
    t = abs(m) / (sd / math.sqrt(n))
    df = n - 1
    return t, betai(df / 2.0, 0.5, df / (df + t * t))


def mean_std(xs):
    return st.mean(xs), st.stdev(xs)


# ---------- 表 3：跨化学体系迁移（5 种子汇总，CX2） ----------
t = load("transfer/transfer_multiseed_v2.json")
tbl3 = {"CALCE_tcn": (0.1704, 0.0208, 0.0913, 0.0161, 0.0946, 0.0185, 1.03, 0.04),
        "CALCE_lstm": (0.1616, 0.0156, 0.0819, 0.0126, 0.0816, 0.0099, 1.00, 0.06),
        "NASA_tcn": (0.1078, 0.0088, 0.0230, 0.0049, 0.2057, 0.0603, 9.28, 2.95),
        "NASA_lstm": (0.1080, 0.0100, 0.0227, 0.0077, 0.0775, 0.0413, 3.79, 2.63)}
for k, v in tbl3.items():
    g = t[k]
    check("表 3 %s 零样本均值" % k, v[0], g["zero"]["mean"], 5e-5)
    check("表 3 %s 零样本标准差" % k, v[1], g["zero"]["std"], 5e-5)
    check("表 3 %s 微调均值" % k, v[2], g["ft"]["mean"], 5e-5)
    check("表 3 %s 微调标准差" % k, v[3], g["ft"]["std"], 5e-5)
    check("表 3 %s 目标域基线均值" % k, v[4], g["to"]["mean"], 5e-5)
    check("表 3 %s 目标域基线标准差" % k, v[5], g["to"]["std"], 5e-5)
    check("表 3 %s 增益比均值" % k, v[6], g["gain"]["mean"], 0.005)
    check("表 3 %s 增益比标准差" % k, v[7], g["gain"]["std"], 0.005)

# 4.3：NASA/LSTM 逐种子增益与中位数
_gains = [load("transfer/t3b_lstm_s%d.json" % s)["targets"]["NASA"]["target_only"]["rmse"] /
          load("transfer/t3b_lstm_s%d.json" % s)["targets"]["NASA"]["fine_tune"]["rmse"]
          for s in (42, 43, 44, 45, 46)]
check("4.3 NASA/LSTM 增益中位数", 3.3, st.median(_gains), 0.06)
check("4.3 CALCE/TCN 增益最大值", 1.08, max(
    load("transfer/t3b_tcn_s%d.json" % s)["targets"]["CALCE"]["target_only"]["rmse"] /
    load("transfer/t3b_tcn_s%d.json" % s)["targets"]["CALCE"]["fine_tune"]["rmse"]
    for s in (42, 43, 44, 45, 46)), 0.02)

# ---------- 表 4：原始协议漂移（CX2，5 种子） ----------
for tgt, mz, sdz in [("CALCE", 1307, 684), ("NASA", 830, 354)]:
    zs = [load("transfer/t3_soh_tcn_s%d.json" % s)["targets"][tgt]["zero_shot"]["rmse"] for s in (42, 43, 44, 45, 46)]
    check("表 4 %s 原始协议均值" % tgt, mz, st.mean(zs), 0.6)
    check("表 4 %s 原始协议标准差" % tgt, sdz, st.stdev(zs), 0.6)

# ---------- 表 5：保形双路由（CX2，逐种子 42-46） ----------
tbl5 = {"CALCE_tcn": (0.085, 0.009, 0.826, 0.108, 0.2229, 0.0919, 0.0147),
        "CALCE_lstm": (0.057, 0.020, 0.799, 0.162, 0.1860, 0.0836, 0.0157),
        "NASA_tcn": (0.146, 0.088, 0.946, 0.066, 0.0807, 0.0184, 0.0079),
        "NASA_lstm": (0.141, 0.099, 0.893, 0.145, 0.0803, 0.0197, 0.0096)}
allsrc, alltgt = [], []
for k, v in sorted(tbl5.items()):
    tgt, mod = k.split("_")
    src, tg, wt, rm = [], [], [], []
    for s in (42, 43, 44, 45, 46):
        o = load("conformal/t4_%s_s%d.json" % (mod, s))["targets"][tgt]
        src.append(o["source_calibrated"]["PICP"])
        tg.append(o["target_calibrated"]["PICP"])
        wt.append(o["target_calibrated"]["MPIW"])
        rm.append(o["point_rmse"])
    allsrc += src
    alltgt += tg
    ms, ss = mean_std(src)
    check("表 5 %s 源域校准 PICP 均值" % k, v[0], ms, 5e-4)
    check("表 5 %s 源域校准 PICP 标准差" % k, v[1], ss, 5e-4)
    check("表 5 %s 目标域校准 PICP 均值" % k, v[2], st.mean(tg), 5e-4)
    check("表 5 %s 目标域校准 PICP 标准差" % k, v[3], st.stdev(tg), 5e-4)
    check("表 5 %s 目标域校准 MPIW" % k, v[4], st.mean(wt), 5e-5)
    check("表 5 %s 点预测 RMSE" % k, v[5], st.mean(rm), 5e-5)
    check("表 5 %s 点预测 RMSE 标准差" % k, v[6], st.stdev(rm), 5e-5)
check("4.5 目标校准覆盖最低值", 0.533, min(alltgt), 5e-4)
check("4.5 目标校准达名义 0.90 的组数", 8, sum(1 for v in alltgt if v >= 0.90), 0)
check("4.5 源域校准覆盖最低值", 0.000, min(allsrc), 5e-4)
check("4.5 源域校准覆盖最高值(<0.90)", 0.252, max(allsrc), 5e-4)
check("4.5 源域全部欠覆盖", 1, 1 if all(v < 0.90 for v in allsrc) else 0, 0)

# ---------- 4.5：条件化收窄（CX2 协议 5/5/6、1/1/2） ----------
w = load("conformal/t4c_multiseed_summary.json")
check("4.5 CALCE Mondrian 覆盖率", 0.867, w["mondrian"]["CALCE"]["mondrian"]["PICP"]["mean"], 0.005)
check("4.5 CALCE Mondrian 单一基线", 0.901, w["mondrian"]["CALCE"]["single"]["PICP"]["mean"], 0.005)
check("4.5 NASA Mondrian 覆盖率", 0.693, w["mondrian"]["NASA"]["mondrian"]["PICP"]["mean"], 0.005)
check("4.5 NASA Mondrian 收窄比", 0.200, 1 - w["mondrian"]["NASA"]["mondrian"]["MPIW"] / w["mondrian"]["NASA"]["single"]["MPIW"], 0.01)
check("4.5 CALCE 加权覆盖率", 0.880, w["weighted"]["CALCE"]["weighted"]["PICP"]["mean"], 0.005)
check("4.5 CALCE 加权单一基线", 0.911, w["weighted"]["CALCE"]["single"]["PICP"]["mean"], 0.005)
check("4.5 NASA 加权=单一(退化)", 0.815, w["weighted"]["NASA"]["weighted"]["PICP"]["mean"], 0.005)

# ---------- 表 6：消融（CX2，配对 t） ----------
groups = {}
for feats in ("base7", "curve14"):
    for s in (42, 43, 44, 45, 46):
        o = load("ablation/t3e_%s_tcn_s%d.json" % (feats, s))["targets"]
        for tgt in ("CALCE", "NASA"):
            for mode, key in (("zero", "zero_shot"), ("ft", "fine_tune")):
                groups.setdefault((feats, tgt, mode), []).append(o[tgt][key]["rmse"])
EXP6 = {("CALCE", "zero"): (0.1684, 0.0187, 0.1742, 0.0201),
        ("CALCE", "ft"): (0.0904, 0.0156, 0.0944, 0.0206),
        ("NASA", "zero"): (0.1071, 0.0091, 0.0935, 0.0057),
        ("NASA", "ft"): (0.0233, 0.0054, 0.0322, 0.0061)}
for (tgt, mode), (e0, s0, e1, s1) in sorted(EXP6.items()):
    g0, g1 = groups[("base7", tgt, mode)], groups[("curve14", tgt, mode)]
    check("表 6 %s/%s 基础 6 维均值" % (tgt, mode), e0, st.mean(g0), 5e-5)
    check("表 6 %s/%s 基础 6 维标准差" % (tgt, mode), s0, st.stdev(g0), 5e-5)
    check("表 6 %s/%s 曲线增强 13 维均值" % (tgt, mode), e1, st.mean(g1), 5e-5)
    check("表 6 %s/%s 曲线增强 13 维标准差" % (tgt, mode), s1, st.stdev(g1), 5e-5)
    tv, pv = paired_t_p(g0, g1)
    if (tgt, mode) == ("CALCE", "ft"):
        check("4.7 CALCE 微调配对 t", 0.79, tv, 0.01)
        check("4.7 CALCE 微调 p 值", 0.476, pv, 0.001)
    if (tgt, mode) == ("CALCE", "zero"):
        check("4.7 CALCE 零样本配对 t", 1.29, tv, 0.01)
        check("4.7 CALCE 零样本 p 值", 0.266, pv, 0.001)
    if (tgt, mode) == ("NASA", "ft"):
        check("4.7 NASA 微调配对 t", 3.81, tv, 0.01)
        check("4.7 NASA 微调 p 值", 0.019, pv, 0.001)
    if (tgt, mode) == ("NASA", "zero"):
        check("4.7 NASA 零样本配对 t", 2.71, tv, 0.01)
        check("4.7 NASA 零样本 p 值", 0.053, pv, 0.001)

# ---------- 4.6：逐电芯诊断（CX2） ----------
for mod, agg_m, agg_sd, min_tgt in [("tcn", 0.52, 0.24, 0.31), ("lstm", 0.51, 0.23, 0.22)]:
    pc, agg = [], []
    for s in range(42, 47):
        o = load("conformal/t4d_per_cell_%s_s%d.json" % (mod, s))["targets"]["CALCE"]
        agg.append(o["cell_aggregated_conformal"]["median"]["PICP"])
        for cc in o["per_cell"].values():
            pc.append(cc["cov_tgt"])
    m_agg, s_agg = mean_std(agg)
    check("4.6 %s 聚合保形覆盖" % mod, agg_m, m_agg, 0.005)
    check("4.6 %s 聚合保形覆盖标准差" % mod, agg_sd, s_agg, 0.005)
    check("4.6 %s 逐电芯目标覆盖最低" % mod, min_tgt, min(pc), 0.005)

# ---------- 校准电芯数扫描（cal1-5 均值） ----------
sw = load("conformal/t4_split_sweep/sweep_summary.json")
SWEXP = {"tcn": [0.809, 0.838, 0.902, 0.894, 0.918],
         "lstm": [0.811, 0.799, 0.864, 0.867, 0.905]}
for m in ("tcn", "lstm"):
    for i, nc in enumerate((1, 2, 3, 4, 5)):
        check("校准电芯扫描 %s %d 颗覆盖率均值" % (m, nc), SWEXP[m][i], sw[m]["cal%d" % nc]["picp_tgt"]["mean"], 5e-4)

# ---------- 划分重抽抽样分布 ----------
rndp = RES / "conformal/i9_seeds/i9_summary.json"
if rndp.exists():
    rnd = json.load(open(rndp, encoding="utf-8"))
    RNDEXP = {"tcn": (50, 0.826, 0.120, 18), "lstm": (50, 0.836, 0.117, 18)}
    for m, v in rnd.items():
        e = RNDEXP[m]
        check("划分重抽 %s 次数" % m, e[0], v["n"], 0)
        check("划分重抽 %s 覆盖均值" % m, e[1], v["picp_tgt"]["mean"], 5e-4)
        check("划分重抽 %s 覆盖标准差" % m, e[2], v["picp_tgt"]["std"], 5e-4)
        check("划分重抽 %s 达标(>=0.90)次数" % m, e[3], round(v["frac_ge_090"] * v["n"]), 0)
        print("[划分重抽] %s n=%d 覆盖均值 %.4f std %.4f q05 %.4f q95 %.4f 达标比例 %.3f"
              % (m, v["n"], v["picp_tgt"]["mean"], v["picp_tgt"]["std"],
                 v["picp_tgt"]["q05"], v["picp_tgt"]["q95"], v["frac_ge_090"]))

# ---------- 表 1：源域 5 折组交叉验证（12 个逐折文件，2026-10-04 增补） ----------
T1 = {"LSTM": ("lstm", 138.06, 4.24, 85.78), "GRU": ("gru", 139.02, 2.23, 86.70),
      "TCN": ("tcn", 136.25, 2.68, 87.76), "Transformer": ("transformer", 131.63, 1.81, 82.76)}
for _disp, (_fn, _m, _s, _mae) in T1.items():
    _rs = [load("baselines/%s_f5_s%d.json" % (_fn, _sd)) for _sd in (42, 43, 44)]
    _rms = [d["rmse_mean"] for d in _rs]
    _maes = [st.mean(f["mae"] for f in d["folds"]) for d in _rs]
    check("表 1 %s RMSE 均值" % _disp, _m, st.mean(_rms), 0.006)
    check("表 1 %s RMSE 标准差" % _disp, _s, st.stdev(_rms), 0.006)
    check("表 1 %s MAE 均值" % _disp, _mae, st.mean(_maes), 0.006)

# ---------- 表 2：LOBO 逐电芯 RMSE 分布（s42；分位数取升序第 floor(q*n)+1 个值（1 基，实现用 0 基下标 floor(q*n)，要求 q<1），std 为样本口径 ddof=1，与表 1/3/5 一致） ----------
T2 = {"TCN": (23.99, 60.31, 85.69, 124.53, 102.55, 70.15),
      "LSTM": (19.86, 58.24, 90.04, 131.35, 105.10, 72.08),
      "Transformer": (31.95, 59.24, 75.66, 104.49, 95.23, 65.15)}
for _disp, (_e0, _e1, _e2, _e3, _em, _es) in T2.items():
    _r = sorted(x["rmse"] for x in jl("baselines/lobo_%s_s42.jsonl" % _disp.lower()))
    _n = len(_r)
    check("表 2 %s 最小" % _disp, _e0, _r[0], 0.006)
    check("表 2 %s Q1" % _disp, _e1, _r[math.floor(0.25 * _n)], 0.006)
    check("表 2 %s 中位数" % _disp, _e2, _r[math.floor(0.50 * _n)], 0.006)
    check("表 2 %s Q3" % _disp, _e3, _r[math.floor(0.75 * _n)], 0.006)
    check("表 2 %s 宏观均值" % _disp, _em, st.mean(_r), 0.006)
    check("表 2 %s 标准差" % _disp, _es, st.stdev(_r), 0.006)

# 表 2 末行：逐电芯三模型择优（依赖测试真值，仅作可部署性上界参照）
_by = {}
for _m in ("tcn", "lstm", "transformer"):
    for _x in jl("baselines/lobo_%s_s42.jsonl" % _m):
        _by.setdefault(_x["te_cell"], {})[_m] = _x["rmse"]
_best = [min(d.values()) for d in _by.values() if len(d) == 3]
assert len(_best) == 119, len(_best)
check("表 2 三模型择优上界", 75.57, st.mean(_best), 0.006)

# ---------- 4.8：早期寿命预测基线（确定性岭回归，单次运行；与泄漏修复所涉管线无关，结果冻结于 results/） ----------
_ep = json.load(open(RES.parent / "results" / "early_pred_summary.json", encoding="utf-8"))
check("4.8 log10 RMSE", 0.117, _ep["rmse_log"], 0.0006)
check("4.8 循环数 RMSE", 141.7, _ep["rmse_cycles"], 0.06)
check("4.8 MAPE(%)", 19.5, _ep["rel_err_pct"], 0.006)

# ---------- 输出 ----------
w1 = max(len(r0[0]) for r0 in ROWS)
print("%-*s  %-14s %-14s %s" % (w1, "论文位置", "论文数值", "本仓库重算", "结论"))
print("-" * (w1 + 42))
for label, expect, got, ok in ROWS:
    print("%-*s  %-14.4f %-14.4f %s" % (w1, label, expect, got, "一致" if ok else "不一致 <--"))
print("-" * (w1 + 42))
if BAD:
    print("有 %d/%d 项对不上：%s" % (len(BAD), len(ROWS), "; ".join(BAD[:5])))
    raise SystemExit(1)
print("全部 %d 项与论文数字一致。" % len(ROWS))
