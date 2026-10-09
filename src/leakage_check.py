"""
leakage_check.py — 消融实验 3：患者分组切分 vs 按行随机切分（计划 §4.3）

唯一目的：量化「不按患者分组」会把指标虚高多少。

背景：79.9% 的金标准样本来自多次就诊患者，同患者内处方集完全相同的占 20.3%。
按行随机切分时，同一患者的其他就诊记录会落进训练集，模型/检索器可以直接
「记住这个患者的方子」，指标被严重高估。

本脚本用同一套代码、同一套指标，只改切分方式，输出对照表。
这份对照必须写进评审材料 —— 否则无法解释为什么指标和别人的报告差那么多。

用法: python3 src/leakage_check.py
"""

from __future__ import annotations

import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import baseline as B
import evaluate as E
import medlib as M
from medlib import DATA_DIR, REPORT_DIR

K = 15
N_DX = 3


def split_rows(records: list[dict], seed: int = M.SEED,
               ratios=(0.8, 0.1, 0.1)) -> dict[str, list[dict]]:
    """按【行】随机切分 —— 这是错误做法，仅用于量化泄漏幅度。"""
    rs = records[:]
    random.Random(seed).shuffle(rs)
    n = len(rs)
    n_tr, n_va = int(n * ratios[0]), int(n * ratios[1])
    return {"train": rs[:n_tr], "val": rs[n_tr:n_tr + n_va], "test": rs[n_tr + n_va:]}


def key_metrics(train: list[dict], test: list[dict]) -> dict:
    """在给定 train/test 上跑关键基线：B1 常量 / B2 西医诊断 / O3 证型+病名 / R0 端到端。"""
    ref_rx = [r["rx"] for r in test]
    ref_dx = [[tuple(t) for t in r["tcm_dx"]] for r in test]
    fb = M.top_herbs(train, K)
    keys = set(M.top_herbs(train, 100))
    vocab = set(M.top_herbs(train, 10 ** 9))

    out: dict = {}

    p1 = B.make_rx_predictor(train, B.const_key, K, fb)
    out["B1_全局常量方"] = E.rx_metrics([p1(r) for r in test], ref_rx, keys, vocab)

    p2 = B.make_rx_predictor(train, B.wm_key, K, fb)
    out["B2_西医诊断检索"] = E.rx_metrics([p2(r) for r in test], ref_rx, keys, vocab)

    p3 = B.make_rx_predictor(train, B.zheng_bing_key, K, fb)
    out["O3_证型+病名(Oracle)"] = E.rx_metrics([p3(r) for r in test], ref_rx, keys, vocab)

    # 端到端：先预测诊断，再用预测诊断检索处方
    dxp = B.make_dx_predictor(train, B.wm_key, N_DX)
    rx_from_dx = B.make_rx_from_pred_dx(train, K, fb)
    pdx = [dxp(r) for r in test]
    out["R0_端到端规则"] = E.rx_metrics([rx_from_dx(p) for p in pdx], ref_rx, keys, vocab)
    out["D2_西医诊断->中医诊断"] = E.diagnosis_metrics(pdx, ref_dx)
    return out


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    all_records = M.read_jsonl(DATA_DIR / "all_records.jsonl")
    gold = [r for r in all_records if M.is_gold(r)]
    print(f"金标准样本 {len(gold):,}  患者 {len({r['pid'] for r in gold}):,}")

    sp_group = M.split_by_patient(gold)
    sp_row = split_rows(gold)

    # 泄漏检查：按行切分时，测试集有多少患者出现在训练集
    tr_pids = {r["pid"] for r in sp_row["train"]}
    te_pids = {r["pid"] for r in sp_row["test"]}
    leaked = te_pids & tr_pids
    leaked_samples = sum(1 for r in sp_row["test"] if r["pid"] in tr_pids)
    print(f"\n按行随机切分：test {len(sp_row['test']):,} 条，其中 "
          f"{leaked_samples:,} 条 ({leaked_samples/len(sp_row['test'])*100:.1f}%) 的患者已在 train 出现")
    print(f"  涉及 {len(leaked):,} 名患者 / test 共 {len(te_pids):,} 名 "
          f"({len(leaked)/len(te_pids)*100:.1f}%)")

    grp_tr, grp_te = sp_group["train"], sp_group["test"]
    row_tr, row_te = sp_row["train"], sp_row["test"]
    assert not ({r["pid"] for r in grp_tr} & {r["pid"] for r in grp_te})

    print(f"\n分组切分: train {len(grp_tr):,} / test {len(grp_te):,}")
    print(f"按行切分: train {len(row_tr):,} / test {len(row_te):,}")

    print("\n跑分组切分（诚实口径）...")
    g = key_metrics(grp_tr, grp_te)
    print("跑按行切分（泄漏口径）...")
    r = key_metrics(row_tr, row_te)

    names = ["B1_全局常量方", "B2_西医诊断检索", "O3_证型+病名(Oracle)", "R0_端到端规则"]
    print(f"\n{'基线':26s} {'分组Jaccard':>12s} {'按行Jaccard':>12s} {'虚高':>9s}")
    print("-" * 64)
    rows = []
    for n in names:
        gj, rj = g[n]["处方_Jaccard"], r[n]["处方_Jaccard"]
        infl = (rj - gj) / gj * 100 if gj else 0
        rows.append((n, gj, rj, infl))
        print(f"{n:26s} {gj:12.3f} {rj:12.3f} {infl:+8.1f}%")
    print(f"\n{'D2 证型F1':26s} {g['D2_西医诊断->中医诊断']['证型_F1']:12.3f} "
          f"{r['D2_西医诊断->中医诊断']['证型_F1']:12.3f} "
          f"{(r['D2_西医诊断->中医诊断']['证型_F1']-g['D2_西医诊断->中医诊断']['证型_F1'])/g['D2_西医诊断->中医诊断']['证型_F1']*100:+8.1f}%")

    L = ["# 泄漏对照消融报告（计划 §4.3 消融 3）", "",
         "## 一句话结论", "",
         "**同一套代码、同一套指标，只改切分方式，指标差异如下。"
         "任何不按患者 ID 分组的结果都不可信。**", "",
         "## 泄漏规模", "",
         f"- 金标准样本 {len(gold):,} 条 / 患者 {len({r['pid'] for r in gold}):,} 名",
         f"- 按行随机切分时，test 的 **{leaked_samples:,}** 条样本"
         f"（{leaked_samples/len(row_te)*100:.1f}%）其患者已在 train 出现",
         f"- 按患者分组切分时，train/test 患者**零重叠**（代码内 assert 强制）", "",
         "## 对照表", "",
         "| 基线 | 患者分组切分（诚实） | 按行随机切分（泄漏） | 虚高 |",
         "|---|---|---|---|"]
    for n, gj, rj, infl in rows:
        L.append(f"| {n} | **{gj:.3f}** | {rj:.3f} | {infl:+.1f}% |")
    L.append(f"| D2 证型 micro-F1 | {g['D2_西医诊断->中医诊断']['证型_F1']:.3f} | "
             f"{r['D2_西医诊断->中医诊断']['证型_F1']:.3f} | "
             f"{(r['D2_西医诊断->中医诊断']['证型_F1']-g['D2_西医诊断->中医诊断']['证型_F1'])/g['D2_西医诊断->中医诊断']['证型_F1']*100:+.1f}% |")
    L += ["", "## 为什么", "",
          "- 79.9% 的金标准样本来自多次就诊患者；",
          "- 同一患者内部，处方集完全相同的样本占 20.3%；",
          "- 因此按行切分等价于把「同一个患者的方子」放进训练集，"
          "检索器和模型都能直接命中，指标被人为抬高。", "",
          "## 执行约定", "",
          "1. `src/prepare_data.py` 只用患者分组切分产出 `data/*.jsonl`；",
          "2. `src/evaluate.py` 不接受任何按行切分的参考文件；",
          "3. 评审材料必须同时给出两套数字，并说明差异来源；",
          "4. 若某次实验结果异常好，第一步先检查切分是否泄漏。"]
    (REPORT_DIR / "leakage_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"\n报告 -> reports/leakage_report.md")


if __name__ == "__main__":
    main()
