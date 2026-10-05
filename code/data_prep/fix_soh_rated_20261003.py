# -*- coding: utf-8 -*-
"""一次性修复（2026-10-03）：CALCE 的 SOH 基准从"首循环容量"改为"额定容量"。

背景：CS2/CX2 的 soh 原按各电芯首循环容量归一（soh 首行恒为 1），与 MIT/NASA
的额定容量口径不一致，且衰减早期出现物理上不合理的 soh > 1（旧表最多 1.18）。
本脚本把全部含 CALCE 行的数据表重算为 soh = capacity_Ah / rated_for(battery_id)
（CS2=1.1 Ah、CX2=1.35 Ah，与 parse_calce.py 的 RATED_BY_PREFIX 一致）。

容量列本身不动，只重算 soh 列；MIT/NASA 行一字不改。
幂等：若 CALCE 行已满足 |soh - capacity_Ah/rated| <= 1e-9 则跳过该表。
原始表备份到 data/archive_soh_first_cycle_20261003/。

运行：python code/data_prep/fix_soh_rated_20261003.py
"""
import shutil
from pathlib import Path

import pandas as pd

import parse_calce as pc

DATA = Path("data")
BACKUP = DATA / "archive_soh_first_cycle_20261003"
TABLES = [
    "建模表_v3.csv.gz",
    "建模表_v3_cx2.csv.gz",
    "建模表_v3c.csv.gz",
    "建模表_v3c_cx2.csv.gz",
    "建模表_v5.csv.gz",
    "calce_full_v2_cx2.csv",
]


def is_already_rated(cal: pd.DataFrame) -> bool:
    exp = cal["capacity_Ah"] / cal["battery_id"].map(pc.rated_for)
    return (cal["soh"] - exp).abs().max() <= 1e-9


def fix_table(path: Path) -> None:
    df = pd.read_csv(path)
    if "soh" not in df.columns or "battery_id" not in df.columns:
        print(f"[跳过] {path.name}: 无 soh/battery_id 列")
        return
    is_cal = df["battery_id"].astype(str).str.startswith(("CS2", "CX2"))
    if not is_cal.any():
        print(f"[跳过] {path.name}: 无 CALCE 行")
        return
    cal = df[is_cal]
    if is_already_rated(cal):
        print(f"[跳过] {path.name}: CALCE soh 已是额定容量基准")
        return
    n_over1_old = int((cal["soh"] > 1).sum())
    BACKUP.mkdir(exist_ok=True)
    shutil.copy2(path, BACKUP / path.name)
    df.loc[is_cal, "soh"] = cal["capacity_Ah"] / cal["battery_id"].map(pc.rated_for)
    fixed = df[is_cal]
    n_over1_new = int((fixed["soh"] > 1).sum())
    df.to_csv(path, index=False,
              compression="gzip" if path.suffix == ".gz" else None)
    print(f"[修复] {path.name}: CALCE {is_cal.sum()} 行 | soh>1: {n_over1_old} -> {n_over1_new}"
          f" | soh max: {cal['soh'].max():.4f} -> {fixed['soh'].max():.4f}"
          f" | MIT/NASA 行未动 ({(~is_cal).sum()} 行)")


if __name__ == "__main__":
    for name in TABLES:
        p = DATA / name
        if p.exists():
            fix_table(p)
        else:
            print(f"[缺失] {name} 不存在，跳过")
