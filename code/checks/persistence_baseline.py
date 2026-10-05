# -*- coding: utf-8 -*-
"""持久基线（2026-10-04 修订清单 A4）：跨化学迁移实验的朴素对照，务必固化。

定义（与 t3b/t4 同窗口口径）：窗口 = 每颗电芯连续 20 个循环（窗口末行下标 i），
标签 = 第 i+H 个循环的 soh，且 cyc[i+H] == cyc[i]+H（循环连续性检查与
t3b_std_local.build_windows_ds 一致，寿命末端不足 H 步的窗口丢弃）。
持久基线预测 = 窗口末行 soh（"SOH 短期近似不变"），不训练任何参数。

输出两部分：
  1) 全目标域电芯池化的 persistence RMSE（H=5/10/20 × CALCE/NASA）——表 3 表注引用；
  2) 逐种子配对：与 results_cx2/transfer/t3b_*.json 的 test_cells 同集合、同窗口，
     对模型 zero/ft RMSE 与 persistence 同种子之差做配对 t 检验（n=5，带符号 t），
     并统计模型 RMSE < persistence 的种子数（胜场）。

    python code/checks/persistence_baseline.py
输出: results_cx2/transfer/persistence_baseline.json（stdout 同步打印）
"""
import csv, json, math, statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "建模表_v3_cx2.csv"
RES = ROOT / "results_cx2" / "transfer"
WINDOW = 20
HS = (5, 10, 20)

# Codex 咨询文档（20261003）参考值（全电芯口径 H=5/10/20）。
# 2026-10-04 复算结论：NASA 池化口径 3/3 命中参考值；配对检验 t 值亦命中
# （NASA/TCN H=10 t=-1.43 胜 3/5）。CALCE 池化比参考值高约 7%（对模型更不利、
# 属保守方向）：参考值与"逐电芯宏平均 sqrt(mean(mse_i))"接近（H=5 差 0.0001，
# H=10/20 差 0.0009/0.0020），但无任何单一口径可同时精确复现两域全部参考值，
# 且参考文档未记录口径。本脚本主口径取"全电芯池化"——与 t3b eval_model 的
# concat 池化评测严格一致，保证与模型 RMSE 可比；宏平均作为副口径一并输出。
# 2026-10-04 晚注：下方 REF 为外部参考文档的**初始参考值**（口径未记录）；
# 窗口连续性修正后的复算主口径值为 CALCE 0.0665/0.0832/0.1066（H=5/10/20），
# 即论文所报值；两者差异为口径差异（约 +7%），非计算不符。
REF = {"CALCE": {5: 0.0609, 10: 0.0761, 20: 0.0977},
       "NASA": {5: 0.0181, 10: 0.0269, 20: 0.0461}}


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


def paired_t_p_signed(xs, ys):
    """带符号配对 t：t>0 表示 xs 均值大于 ys（模型差于持久基线）。"""
    d = [x - y for x, y in zip(xs, ys)]
    n = len(d)
    m = st.mean(d)
    sd = st.stdev(d)
    t = m / (sd / math.sqrt(n))
    df = n - 1
    p = betai(df / 2.0, 0.5, df / (df + abs(t) * abs(t)))
    return t, p


def load_cells():
    cells = {}
    with open(DATA, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                cyc = float(row["cycle"])
                soh = float(row["soh"])
            except (TypeError, ValueError):
                continue
            if math.isnan(soh) or math.isnan(cyc):
                continue
            cells.setdefault((row["dataset"], row["battery_id"]), ([], []))
            cells[(row["dataset"], row["battery_id"])][0].append(cyc)
            cells[(row["dataset"], row["battery_id"])][1].append(soh)
    for k, (cs, ys) in list(cells.items()):
        order = sorted(range(len(cs)), key=lambda i: cs[i])
        cells[k] = ([cs[i] for i in order], [ys[i] for i in order])
    return cells


def win_pairs(cyc, y, H):
    # 窗口连续 且 窗口到标签连续，与 t3b_std_local.build_windows_ds 完全一致
    # （2026-10-04 修正：原实现漏了窗口内连续性，窗口集合比模型多约 3.5%）
    out = []
    for i in range(WINDOW - 1, len(y) - H):
        if cyc[i] - cyc[i - WINDOW + 1] == WINDOW - 1                 and cyc[i + H] == cyc[i] + H:
            out.append((y[i], y[i + H]))
    return out


def pooled_rmse(cells, dom, H, bids=None):
    se, n = 0.0, 0
    for (ds, bid), (cyc, y) in cells.items():
        if ds != dom:
            continue
        if bids is not None and bid not in bids:
            continue
        for a, b in win_pairs(cyc, y, H):
            se += (a - b) ** 2
            n += 1
    return math.sqrt(se / n), n


def macro_rmse(cells, dom, H, bids=None):
    """副口径：逐电芯 RMSE² 等权平均再开方（宏平均，每颗电芯等权）。"""
    per = []
    for (ds, bid), (cyc, y) in cells.items():
        if ds != dom:
            continue
        if bids is not None and bid not in bids:
            continue
        pr = win_pairs(cyc, y, H)
        if pr:
            per.append(sum((a - b) ** 2 for a, b in pr) / len(pr))
    return math.sqrt(sum(per) / len(per)), len(per)


def main():
    cells = load_cells()
    print(f"建模表电芯数: {len(cells)}")

    # 1) 全电芯 persistence：主口径=池化（与模型评测一致），副口径=宏平均
    allcell = {}
    for dom in ("CALCE", "NASA"):
        allcell[dom] = {}
        for H in HS:
            rmse, n = pooled_rmse(cells, dom, H)
            macro, _ = macro_rmse(cells, dom, H)
            ref = REF[dom][H]
            flag = "OK" if abs(rmse - ref) <= 5e-4 else f"与外部初始参考值 {ref} 差 {rmse - ref:+.4f}（口径差异，见脚本注释）"
            allcell[dom][H] = {"rmse": round(rmse, 6), "macro_rmse": round(macro, 6),
                               "n_windows": n, "ref": ref, "check": flag}
            print(f"[全电芯] {dom} H={H:2d} persistence 池化RMSE={rmse:.4f} "
                  f"(宏平均 {macro:.4f}, 窗口 {n})  {flag}")

    # 2) 逐种子配对（t3b 同 test_cells）；h5/h20 敏感性实验仅 TCN，pattern 写死 tcn
    paired = {}
    jobs = [("h10", RES / "t3b_{m}_s{s}.json", (10,), ("tcn", "lstm")),
            ("h5", ROOT / "results_cx2" / "transfer_sens_h5" / "t3b_tcn_s{s}.json", (5,), ("tcn",)),
            ("h20", ROOT / "results_cx2" / "transfer_sens_h20" / "t3b_tcn_s{s}.json", (20,), ("tcn",))]
    for tag, pat, hs, models in jobs:
        for H in hs:
            for model in models:
                seeds, per_seed = [], []
                for s in (42, 43, 44, 45, 46):
                    p = Path(str(pat).format(m=model, s=s))
                    if not p.exists():
                        raise FileNotFoundError(f"t3b 结果缺失: {p}")
                    j = json.load(open(p, encoding="utf-8"))
                    if j["horizon"] != H:
                        continue
                    seeds.append(s)
                    per_seed.append(j)
                if not per_seed:
                    continue
                # 每个 t3b json 含两个目标域，逐域配对
                for dom in ("CALCE", "NASA"):
                    pers_by_seed = []
                    for j in per_seed:
                        te = j["targets"][dom]["test_cells"]
                        pr, _ = pooled_rmse(cells, dom, H, bids=set(te))
                        pers_by_seed.append(pr)
                    for path_key, label in (("zero_shot", "zero"), ("fine_tune", "ft")):
                        mods = [j["targets"][dom][path_key]["rmse"] for j in per_seed]
                        t, pv = paired_t_p_signed(mods, pers_by_seed)
                        wins = sum(1 for a, b in zip(mods, pers_by_seed) if a < b)
                        paired[f"{tag}|{model}|{dom}|{label}|H{H}"] = {
                            "seeds": seeds,
                            "model_rmse": [round(x, 6) for x in mods],
                            "persistence_rmse": [round(x, 6) for x in pers_by_seed],
                            "t": round(t, 4), "p": round(pv, 4), "model_wins": wins,
                        }

    out = {"all_cell": allcell, "paired": paired,
           "note": "paired.t>0 = 模型均值差于持久基线；model_wins = 模型 RMSE 更小的种子数"}
    dst = RES / "persistence_baseline.json"
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print("saved:", dst)
    for k, v in paired.items():
        print(f"{k:28s} t={v['t']:+.2f} p={v['p']:.3f} 胜 {v['model_wins']}/5")


if __name__ == "__main__":
    main()
