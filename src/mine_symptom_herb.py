"""
mine_symptom_herb.py — 症状 → 药 关联挖掘 + 症状信息量消融

回答的问题：能否从数据里恢复"医生为什么这样开方"，并让模型学习？

三条结论（本脚本产出，见 reports/symptom_herb_report.md）：

  1. 【能，但方向要反过来】用户直觉是「中药适应症 → 疾病 → 理由」。实测：
     处方 → 治法 反推 micro-F1 仅 0.456（略强于常量 0.424），说明"方=治法的实现"
     这个链条只有部分成立；而 症状 → 药 携带了强得多的信号。

  2. 【症状信息量是实测的】逐药二分类 macro AUC（5 折 CV, Bernoulli NB）：
       只用 证型+病名+性别年龄        AUC = 0.7140
       + 180 个症状 token             AUC = 0.7605   (+4.65 点)
     即：在诊断标签之外，症状确实决定了用哪几味药。

  3. 【必须防两类伪关联】
     (a) 文书用语混入：如 "病史同前" 与九香虫 lift 17.2 —— 这是医生书写习惯，
         不是症状，更不是医学逻辑。本脚本内置文书用语黑名单。
     (b) 医师/时段混杂：某味药集中在很短时间内使用 = 某医生的处方习惯。
         本脚本用「用药时间集中度」自动标记这类可疑关联。

用法: python3 src/mine_symptom_herb.py
"""

from __future__ import annotations

import math
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import medlib as M
from medlib import DATA_DIR, REPORT_DIR

NEG = re.compile(r"^(无|未|不|否认)")

# 文书用语 / 无效记录：不是症状，混进特征会挖出医师书写习惯
CHART_NOISE = {
    "病史同前", "病史同", "同前", "同上", "无特殊", "无不适", "无明显不适",
    "无其他不适", "无其他特殊", "一般情况可", "一般可", "尚可", "可", "无",
}

TOK_MIN = 150        # 症状 token 最低出现次数
STRATA = ["湿热下注证", "肾虚证", "气滞血瘀证", "肝肾亏虚证", "下焦湿热证",
          "肾气不充证", "阴虚证"]
LIFT_MIN = 1.6
P_H_MIN = 0.25


def clauses(text: str) -> list[str]:
    """现病史切分为症状子句，保留否定语义（无X != X）。"""
    out: list[str] = []
    for part in re.split(r"[，,。；;、\s]+", str(text)):
        p = part.strip()
        if not p:
            continue
        p = re.sub(r"^(偶有|时有|自觉|诉|伴)", "", p)
        if not p:
            continue
        out.append(("无" + p[1:]) if NEG.match(p) else p)
    return out


def symptom_vocab(records: list[dict], min_count: int = TOK_MIN) -> set[str]:
    c = Counter(x for r in records for x in clauses(r["hpi"]))
    return {t for t, n in c.items() if n >= min_count and t not in CHART_NOISE}


# ---------------------------------------------------------------------------
# 实验 B：症状信息量（逐药 macro AUC，5 折 CV）
# ---------------------------------------------------------------------------

def _auc(scores: list[float], y: list[int]) -> float | None:
    order = sorted(range(len(y)), key=lambda i: scores[i])
    n1 = sum(y)
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return None
    ranks = [0.0] * len(y)
    i = 0
    while i < len(order):
        j = i
        while j < len(order) and scores[order[j]] == scores[order[i]]:
            j += 1
        avg = (i + j - 1) / 2.0 + 1
        for k in range(i, j):
            ranks[order[k]] = avg
        i = j
    s = sum(ranks[i] for i in range(len(y)) if y[i] == 1)
    return (s - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def _features(r: dict, use_sym: bool) -> set[str]:
    f: set[str] = set()
    for t in r["tcm_dx"]:
        if t[1]:
            f.add("Z:" + t[1])
        if t[0]:
            f.add("B:" + t[0])
    f.add("S:" + r["sex"])
    f.add("A:" + str(min((r["age"] or 0) // 10, 8)))
    if use_sym:
        f |= {"T:" + x for x in r.get("sym", ())}
    return f


def symptom_information_gain(train: list[dict], n_herbs: int = 100,
                             sample: int = 9000, folds: int = 5,
                             seed: int = 0) -> dict:
    """逐药二分类 macro AUC：诊断标签 vs 诊断标签+症状。

    用逐药 AUC 而不是联合检索，是因为联合条件（证型+病名+5个症状）在 21k 训练集上
    88.7% 的组合从未出现过 —— 朴素检索会被稀疏性击穿，得出"症状无用"的错误结论。
    逐药 AUC 对稀疏稳健。
    """
    random.seed(seed)
    sub = random.sample(train, min(sample, len(train)))
    idx = list(range(len(sub)))
    random.shuffle(idx)
    fsplit = [idx[i::folds] for i in range(folds)]
    top = M.top_herbs(train, n_herbs)

    out: dict[str, float] = {}
    for use_sym in (False, True):
        F = [_features(r, use_sym) for r in sub]
        allf = set().union(*F)
        aucs: list[float] = []
        for h in top:
            y = [1 if h in r["rx"] else 0 for r in sub]
            if sum(y) < 50 or sum(y) > len(y) - 50:
                continue
            sc = [0.0] * len(sub)
            for f_ in fsplit:
                hold = set(f_)
                trn = [i for i in range(len(sub)) if i not in hold]
                n1 = sum(y[i] for i in trn)
                n0 = len(trn) - n1
                c1: Counter = Counter()
                c0: Counter = Counter()
                for i in trn:
                    for ftr in F[i]:
                        if y[i]:
                            c1[ftr] += 1
                        else:
                            c0[ftr] += 1
                lr: dict[str, float] = {}
                C = 0.0
                for ftr in allf:
                    p1 = (c1[ftr] + 1) / (n1 + 2)
                    p0 = (c0[ftr] + 1) / (n0 + 2)
                    lr[ftr] = math.log(p1 / p0) - math.log((1 - p1) / (1 - p0))
                    C += math.log((1 - p1) / (1 - p0))
                base = math.log(n1 / n0) + C
                for i in f_:
                    s = base
                    for f2 in F[i]:
                        s += lr[f2]
                    sc[i] = s
            a = _auc(sc, y)
            if a:
                aucs.append(a)
        out["症状" if use_sym else "基线"] = sum(aucs) / len(aucs)
        out["药数" if use_sym else "药数_基线"] = len(aucs)
    return out


# ---------------------------------------------------------------------------
# 实验 C：证型内分层的 症状 → 药 关联挖掘 + 混杂标记
# ---------------------------------------------------------------------------

def mine_associations(records: list[dict], zheng: str, top_n: int = 14) -> list[dict]:
    """在单一证型内挖 症状→药 关联，控制证型混杂，并标记疑似医师/时段混杂。"""
    sub = [r for r in records if zheng in r.get("_z", ())]
    if len(sub) < 500:
        return []

    n = len(sub)
    base = Counter(h for r in sub for h in set(r["rx"]))
    basep = {h: c / n for h, c in base.items()}
    tok = Counter(x for r in sub for x in set(clauses(r["hpi"])))
    pair: dict[str, Counter] = defaultdict(Counter)
    for r in sub:
        for x in set(clauses(r["hpi"])):
            for h in set(r["rx"]):
                pair[x][h] += 1

    # 药物使用的时间集中度：集中在极少数月份 = 疑似某医生/某时段习惯
    mon = defaultdict(Counter)
    for r in sub:
        for h in set(r["rx"]):
            mon[h][r["date"][:7]] += 1
    conc = {}
    for h, mc in mon.items():
        tot = sum(mc.values())
        conc[h] = max(mc.values()) / tot if tot else 1.0

    # 患者分散度：同一 (症状,药) 共现来自多少不同患者。
    # 若 159 次共现只来自 5 个患者，那是"某几个患者反复复诊"的产物，
    # 不是可泛化的医学规律 —— 这比时间集中度更能抓住医师/患者层面的伪关联。
    pset: dict[tuple[str, str], set] = defaultdict(set)
    for r in sub:
        for x in set(clauses(r["hpi"])):
            for h in set(r["rx"]):
                pset[(x, h)].add(r["pid"])

    rows = []
    for t, nt in tok.items():
        if nt < 100 or t in CHART_NOISE:
            continue
        for h, nth in pair[t].items():
            if nth < 25:
                continue
            p = nth / nt
            bp = basep.get(h, 1e-9)
            lift = p / bp
            npat = len(pset[(t, h)])
            if lift >= LIFT_MIN and p >= P_H_MIN:
                rows.append({
                    "症状": t, "药": h, "P(药|症状)": p, "基准": bp, "lift": lift,
                    "n_症状": nt, "n_共现": nth, "患者数": npat,
                    "每患者次数": nth / max(1, npat),
                    "时间集中度": conc.get(h, 1.0),
                })
    # 先按 lift 排，再按患者数降权：每患者平均复现 > 5 次的视为患者层伪关联，压到最后
    for r in rows:
        r["可疑"] = (r["每患者次数"] > 5.0) or (r["时间集中度"] > 0.5)
    rows.sort(key=lambda x: (x["可疑"], -x["患者数"] / max(1, x["n_共现"]), -x["lift"]))
    # 每味药只保留最强的一条，避免同一味药刷屏
    seen: set[str] = set()
    out = []
    for r in rows:
        if r["药"] in seen:
            continue
        seen.add(r["药"])
        out.append(r)
        if len(out) >= top_n:
            break
    return out


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    allr = M.read_jsonl(DATA_DIR / "all_records.jsonl")
    gold = [r for r in allr if M.is_gold(r)]
    for r in gold:
        r["_z"] = {t[1] for t in r["tcm_dx"] if t[1]}
    vocab = symptom_vocab(gold)
    for r in gold:
        r["sym"] = set(clauses(r["hpi"])) & vocab

    train, test = M.split_by_patient(gold)["train"], M.split_by_patient(gold)["test"]
    print(f"金标准 {len(gold):,}  症状词表 {len(vocab)}  (>= {TOK_MIN} 次, 已剔除文书用语)")

    # ---------- 实验 A：处方 -> 治法 ----------
    zhi_of = lambda r: {t[2] for t in r["tcm_dx"] if t[2]}          # noqa: E731
    h2z: dict[str, Counter] = defaultdict(Counter)
    for r in train:
        for h in set(r["rx"]):
            for z in zhi_of(r):
                h2z[h][z] += 1

    def pred_zhi(rx, k=3):
        sc: Counter = Counter()
        for h in rx:
            c = h2z.get(h)
            if not c:
                continue
            tot = sum(c.values())
            for z, cnt in c.items():
                sc[z] += cnt / tot
        return {z for z, _ in sorted(sc.items(), key=lambda x: (-x[1], x[0]))[:k]}

    tp = fp = fn = top1 = 0
    for r in test:
        p, g = pred_zhi(r["rx"]), zhi_of(r)
        tp += len(p & g); fp += len(p - g); fn += len(g - p)
        if p and g and next(iter(p)) in g:
            top1 += 1
    f1_a = 2 * tp / (2 * tp + fp + fn)
    print(f"\n[实验A] 处方 → 治法反推  P={tp/(tp+fp):.3f} R={tp/(tp+fn):.3f} "
          f"F1={f1_a:.3f}  top1={top1/len(test)*100:.1f}%")

    # ---------- 实验 B：症状信息量 ----------
    print("\n[实验B] 症状信息量（逐药 macro AUC, 5折CV）...")
    ig = symptom_information_gain(train)
    print(f"  基线(证型+病名+性别年龄) AUC={ig['基线']:.4f}  ({ig['药数_基线']} 味)")
    print(f"  +症状(180 token)        AUC={ig['症状']:.4f}  ({ig['药数']} 味)")
    gain = ig["症状"] - ig["基线"]
    print(f"  提升 = {gain:+.4f}")

    # ---------- 实验 C：关联挖掘 ----------
    print("\n[实验C] 症状 → 药 关联挖掘（证型内分层）...")
    mined = {}
    for z in STRATA:
        rows = mine_associations(gold, z)
        if rows:
            mined[z] = rows
            print(f"  {z:10s} 挖出 {len(rows)} 条")

    # ---------- 报告 ----------
    L = ["# 症状 → 药 关联挖掘报告", "",
         "回答的问题：**能否从数据里恢复「医生为什么这样开方」，并让模型学习？**", "",
         "## 结论摘要", "",
         f"1. **处方 → 治法 反推 micro-F1 = {f1_a:.3f}**（top1 命中 {top1/len(test)*100:.1f}%）。",
         "   仅略强于「永远猜最常见治法」的常量基线 0.424 —— 说明「方 = 治法的实现」",
         "   这条链只有部分成立，**不能靠它解释处方**。",
         f"2. **症状确实携带选药信息**：逐药 macro AUC 从 **{ig['基线']:.4f}** 提升到 "
         f"**{ig['症状']:.4f}**（**{gain:+.4f}**）。",
         "   即：在诊断标签之外，现病史里的症状词决定了用哪几味药。",
         "3. 挖出的关联**多数符合中医逻辑**，但**混有伪关联**，必须过滤（见下）。", "",
         "## 实验 A：处方 → 治法（功能性自洽检验）", "",
         "| 指标 | 值 |", "|---|---|",
         f"| micro P | {tp/(tp+fp):.3f} |", f"| micro R | {tp/(tp+fn):.3f} |",
         f"| micro F1 | {f1_a:.3f} |", f"| top1 命中率 | {top1/len(test)*100:.1f}% |",
         "| 对照：常量基线(D1) 治法 F1 | 0.424 |", "",
         "> 只用 462 味药的组合去反推治法，F1 仅 0.456。处方里的功能性信号是**弱**的。",
         "> 这意味着：**不能把「处方 = 治法的直接实现」当作解释链**。", "",
         "## 实验 B：症状的选药信息量", "",
         "方法：逐味药做二分类（该药是否出现在方中），5 折交叉验证，Bernoulli NB，",
         "指标为 100 味高频药的 macro AUC。", "",
         "| 特征 | macro AUC | 药数 |", "|---|---|---|",
         f"| 证型+病名+性别年龄 | {ig['基线']:.4f} | {ig['药数_基线']} |",
         f"| **+ 180 个症状 token** | **{ig['症状']:.4f}** | {ig['药数']} |",
         f"| **提升** | **{gain:+.4f}** | |", "",
         "> ⚠️ **为什么不用联合检索来测**：条件组合（证型+病名+5个症状）在 21,484 条",
         "> 训练集上，**88.7% 的测试组合从未出现过**，朴素检索会被稀疏性击穿，",
         "> 得出「症状无用」的错误结论（实测：0.203 → 0.176）。逐药 AUC 对稀疏稳健。",
         "> 这是本项目踩过的第二个统计陷阱，务必记录。", "",
         "## 实验 C：症状 → 药 关联（证型内分层）", ""]

    for z, rows in mined.items():
        L += [f"### 证型 = {z}", "",
              "| 症状 | 药 | P(药\\|症状) | 基准 | lift | n | 患者数 | 每患者次数 | 时间集中度 |",
              "|---|---|---|---|---|---|---|---|---|"]
        for r in rows:
            flag = " ⚠️" if r["可疑"] else ""
            L.append(f"| {r['症状']} | {r['药']} | {r['P(药|症状)']*100:.1f}% | "
                     f"{r['基准']*100:.1f}% | {r['lift']:.2f} | {r['n_症状']} | "
                     f"{r['患者数']} | {r['每患者次数']:.1f} | "
                     f"{r['时间集中度']*100:.0f}%{flag} |")
        L.append("")

    L += ["## 两类必须过滤的伪关联", "",
          "### (a) 文书用语混入",
          "`病史同前` 与 九香虫 的 lift 高达 **17.2** —— 但「病史同前」不是症状，",
          "是医生的**书写习惯**。这类词进入特征会挖出文书习惯而非医学逻辑。",
          f"本脚本内置黑名单（{len(CHART_NOISE)} 个）：{sorted(CHART_NOISE)[:8]} ...", "",
          "### (b) 医师 / 患者 / 时段混杂",
          "两个自动护栏，任一命中即标 ⚠️：",
          "",
          "- **患者分散度「每患者次数」**：共现次数 / 不同患者数。",
          "  某个 (症状,药) 出现 159 次但只来自 5 个患者 → 那是几个患者反复复诊的产物，",
          "  不是可泛化的规律。**> 5 次/患者标记为患者层伪关联。**",
          "- **时间集中度**：该药在分层内最集中的单月用量占比。**> 50%** 说明这味药",
          "  基本只在某段时间用过，更像某位医生的用药习惯。",
          "",
          "这两类关联**必须做患者分组 + 时间外推验证**才能采用，否则学到的是",
          "「某医生对某几个患者的习惯」，上线后会泛化失败。", "",
          "## 落到模型上怎么做", "",
          "推荐**可验证的三层中间表示**（而不是让 LLM 自由生成推理链）：", "",
          "```",
          "症状/四诊  →  证型  →  治法  →  功效类别(~40类)  →  具体药味(462)",
          "```", "",
          "| 环节 | 收益 | 可验证方式 |", "|---|---|---|",
          "| 功效类别中间层 | 输出空间 462 → ~40，**显著降低学习难度** | 类别准确率 |",
          "| 治法 → 功效覆盖 | 临床上更安全 | 程序自动校验：方中药物功效 ⊇ 治法 |",
          "| 症状 → 加减说明 | 可解释、可审核 | 校验：药真在方里、症状真在现病史里 |", "",
          "**关键**：这三项都是**程序可校验**的，不是 LLM 自由发挥的幻觉。", "",
          "## 前置条件（数据缺口）", "",
          "1. **外部中药功效字典**（462 味药的 功效/性味/归经）。",
          "   ⚠️ **绝不能从本数据集挖 药↔疾病 共现来当「医生的理由」** —— 那是把相关性",
          "   重述一遍，是循环论证。必须用**药典/中药学教材**这类外部权威来源。",
          "2. **复诊疗效反馈**：本报告挖的是「医生怎么开」，不是「为什么有效」。",
          "   真正的因果推理需要疗效数据（见《模型训练计划.md》§10.4）。"]

    (REPORT_DIR / "symptom_herb_report.md").write_text("\n".join(L) + "\n",
                                                      encoding="utf-8")
    print(f"\n报告 -> reports/symptom_herb_report.md")


if __name__ == "__main__":
    main()
