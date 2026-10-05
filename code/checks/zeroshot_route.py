# -*- coding: utf-8 -*-
"""零样本路由复算（论文 3.4 对照补算；投稿期新增）。

背景：论文表 5 的"源域校准"对照以微调后（部署）模型的测试残差评估覆盖率。
为检验"q_src 与评估残差非同源模型"这一朴素选择不影响结论，本复算将评估对象
换为零样本（未微调）模型的测试残差：校准分位数与预测函数同出源模型，只有
评估数据在目标域（同划分、同种子、同标准化协议）。

数据源：results_cx2/conformal/t4zs_<model>_s<seed>.json
        （由 code/conformal/t4_conformal_local.py --zs-only 生成，20 组配置）

复算并断言（论文 3.4 所报）：
  * 20 组（2 目标域 × 2 骨干 × 5 种子）覆盖率范围 0.0000--0.0240，全部低于名义 0.90
    （对照：微调口径为 0.000--0.252）
  * 零样本测试残差中位数约为 q_src 的 21--46 倍（残差尺度失配是覆盖坍塌的直接量）
运行：python code/checks/zeroshot_route.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results_cx2" / "conformal"
ROWS, BAD = [], []


def check(label, expect, got, tol):
    ok = abs(got - expect) <= tol
    ROWS.append((label, expect, got, ok))
    if not ok:
        BAD.append(label)


covs, ratios = [], []
for model in ("tcn", "lstm"):
    for seed in (42, 43, 44, 45, 46):
        d = json.load(open(RES / f"t4zs_{model}_s{seed}.json", encoding="utf-8"))
        for dom in ("CALCE", "NASA"):
            t = d["targets"][dom]
            zs = t["zero_shot"]
            # 独立复算：由逐电芯（窗口数 × 覆盖率）加权汇出总覆盖率，与已存值对拍
            cells = t["per_cell"]
            n_tot = sum(c["n_windows"] for c in cells.values())
            cov_rec = sum(c["n_windows"] * c["cov_src_zeroshot"] for c in cells.values()) / n_tot
            check(f"{model}/s{seed}/{dom} PICP 复算一致", zs["PICP_src_zeroshot"], cov_rec, 1e-9)
            covs.append(zs["PICP_src_zeroshot"])
            ratios.append(zs["med_abs_r"] / d["q_src"])

check("20 组覆盖率最小值", 0.0000, min(covs), 5e-4)
check("20 组覆盖率最大值", 0.0240, max(covs), 5e-4)
check("20 组全部低于名义 0.90", 1, 1 if all(v < 0.90 for v in covs) else 0, 0)
check("中位残差/q_src 比值下限", 21.3, min(ratios), 0.2)
check("中位残差/q_src 比值上限", 46.2, max(ratios), 0.2)

w1 = max(len(r[0]) for r in ROWS)
for label, expect, got, ok in ROWS:
    print("%-*s  %-14.4f %-14.4f %s" % (w1, label, expect, got, "一致" if ok else "不一致 <--"))
print("-" * 72)
if BAD:
    print("有 %d 项不一致：" % len(BAD), BAD)
    raise SystemExit(1)
print("全部 %d 项与论文数字一致。" % len(ROWS))
