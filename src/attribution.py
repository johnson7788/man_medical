"""
attribution.py — 处方成因归因：医生的开方依据到底来自哪里？

回答的问题：医生开方是根据「中医指南 + 经验文档 + 中药药理」吗？各自占多大比重？

方法：累积消融（cumulative ablation）+ 医师风格检测。
  用逐药二分类 macro AUC 度量每一类信息源的【边际贡献】，逐层叠加：

    L0 常量（性别/年龄）                  —— 什么都不看
    L1 + 西医诊断（检查结果）              —— "根据检查结果"
    L2 + 中医证型/病名（辨证，≈指南层面）    —— 中医指南的落地形式
    L3 + 症状（四诊细节 / 患者状态）         —— 个体化
    L4 + 既往处方（治疗进程）               —— 经验/复诊调整
    L5 + 复诊间隔

  另加一项数据里【没有】但临床上必然存在的东西：医师个人风格 / 书写模板。
  做法：找"爆句式"文书用语（时间分布高度集中 = 某医生某时期的模板），
  检验同一证型下、带/不带该标记的两组患者，处方是否系统性不同。

用法: python3 src/attribution.py
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


def clauses(text: str) -> list[str]:
    out: list[str] = []
    for part in re.split(r"[，,。；;、\s]+", str(text)):
        p = part.strip()
        if not p:
            continue
        p = re.sub(r"^(偶有|时有|自觉|诉|伴)", "", p)
        if p:
            out.append(("无" + p[1:]) if NEG.match(p) else p)
    return out


def _auc(scores, y) -> float | None:
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


# ---------------------------------------------------------------------------
# 累积消融
# ---------------------------------------------------------------------------

LEVELS = [
    ("L0 常量(性别/年龄)", False, False, False, False, False),
    ("L1 +西医诊断(检查结果)", True, False, False, False, False),
    ("L2 +证型/病名(辨证≈指南)", True, True, False, False, False),
    ("L3 +症状(四诊细节)", True, True, True, False, False),
    ("L4 +既往处方(治疗进程)", True, True, True, True, False),
    ("L5 +复诊间隔", True, True, True, True, True),
]


def build_features(r: dict, wm: bool, dx: bool, sym: bool, prev: bool,
                   gap: bool) -> set[str]:
    f = {"S:" + r["sex"], "A:" + str(min((r["age"] or 0) // 10, 8))}
    if wm:
        f |= {"W:" + x for x in r["wm_dx"]}
    if dx:
        for t in r["tcm_dx"]:
            if t[1]:
                f.add("Z:" + t[1])
            if t[0]:
                f.add("B:" + t[0])
    if sym:
        f |= {"T:" + x for x in r.get("sym", ())}
    if prev:
        # 既往处方只作为【输入特征】使用（生产中 HIS 本来就有），不是标签泄漏
        f |= {"P:" + h for h in (r.get("prev_rx") or ())}
        f.add("PV:" + ("有" if r.get("prev_rx") else "无"))
    if gap:
        g = r.get("prev_gap")
        f.add("G:" + ("na" if g is None else str(min(g // 30, 12))))
    return f


def cumulative_ablation(train: list[dict], n_herbs: int = 100, sample: int = 7000,
                        folds: int = 4, seed: int = 0) -> list[dict]:
    random.seed(seed)
    sub = random.sample(train, min(sample, len(train)))
    idx = list(range(len(sub)))
    random.shuffle(idx)
    fsplit = [idx[i::folds] for i in range(folds)]
    top = M.top_herbs(train, n_herbs)

    results = []
    for name, wm, dx, sym, prev, gap in LEVELS:
        F = [build_features(r, wm, dx, sym, prev, gap) for r in sub]
        allf = set().union(*F)
        aucs = []
        for h in top:
            y = [1 if h in r["rx"] else 0 for r in sub]
            if sum(y) < 40 or sum(y) > len(y) - 40:
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
                lr = {}
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
        results.append({"层": name, "auc": sum(aucs) / len(aucs), "药数": len(aucs)})
        print(f"  {name:26s} macro AUC = {results[-1]['auc']:.4f}")
    return results


# ---------------------------------------------------------------------------
# 医师风格 / 书写模板检测
# ---------------------------------------------------------------------------

def bursty_markers(records: list[dict], min_count: int = 50,
                   top_k: int = 25) -> list[tuple[str, int, float]]:
    """找"爆句式"文书用语：时间分布高度集中 = 某医生某时期的模板。

    归一化熵越低 = 越集中 = 越像个人模板而非通用症状描述。
    """
    months = sorted({r["date"][:7] for r in records})
    midx = {m: i for i, m in enumerate(months)}
    nm = len(months)
    tok = Counter(x for r in records for x in set(clauses(r["hpi"])))
    out = []
    for t, n in tok.items():
        if n < min_count:
            continue
        mc = Counter(midx[r["date"][:7]] for r in records if t in clauses(r["hpi"]))
        tot = sum(mc.values())
        H = -sum((c / tot) * math.log(c / tot) for c in mc.values()) / math.log(nm)
        out.append((t, n, H))
    out.sort(key=lambda x: x[2])
    return out[:top_k]


def style_effect(records: list[dict], marker: str, zheng: str,
                 top_n: int = 15, perms: int = 200,
                 seed: int = 0) -> dict | None:
    """同一证型下，带/不带该文书标记的两组患者，处方是否系统性不同？

    用两组的 top-15 药集合 Jaccard 度量差异，并与"随机等量二分"的零假设对比。
    Jaccard 越低 = 差异越大；用置换检验给出显著性。
    """
    sub = [r for r in records if zheng in {t[1] for t in r["tcm_dx"]}]
    a = [r for r in sub if marker in clauses(r["hpi"])]
    b = [r for r in sub if marker not in clauses(r["hpi"])]
    if len(a) < 60 or len(b) < 60:
        return None

    def centro(rs):
        # 确定性排序：Counter.most_common 的并列名次依赖 PYTHONHASHSEED，
        # 会让本脚本的显著组数在两次运行间漂移（实测 12 vs 13）
        c = Counter(h for r in rs for h in set(r["rx"]))
        return {h for h, _ in sorted(c.items(), key=lambda x: (-x[1], x[0]))[:top_n]}

    ca, cb = centro(a), centro(b)
    obs = len(ca & cb) / len(ca | cb)

    random.seed(seed)
    null = []
    pool = sub
    k = len(a)
    for _ in range(perms):
        samp = set(random.sample(range(len(pool)), k))
        g1 = [pool[i] for i in samp]
        g2 = [pool[i] for i in range(len(pool)) if i not in samp]
        j = len(centro(g1) & centro(g2)) / len(centro(g1) | centro(g2))
        null.append(j)
    null.sort()
    p = sum(1 for x in null if x <= obs) / len(null)
    return {"marker": marker, "zheng": zheng, "n_with": len(a), "n_without": len(b),
            "jaccard": obs, "null_median": null[len(null) // 2], "p": p,
            "top_with": sorted(ca)[:8], "top_without": sorted(cb)[:8]}


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    allr = M.read_jsonl(DATA_DIR / "all_records.jsonl")
    gold = [r for r in allr if M.is_gold(r)]
    vocab = {t for t, n in Counter(x for r in gold for x in clauses(r["hpi"])).items()
             if n >= 150}
    for r in gold:
        r["sym"] = set(clauses(r["hpi"])) & vocab
    train, test = M.split_by_patient(gold)["train"], M.split_by_patient(gold)["test"]

    print("【累积消融】逐药 macro AUC（5折改4折CV, 训练子集7000）")
    abl = cumulative_ablation(train)

    print("\n【医师风格/书写模板检测】")
    mk = bursty_markers(gold)
    print(f"  最集中的 5 个文书用语（归一化熵，越低越集中）:")
    for t, n, h in mk[:5]:
        print(f"    {t!r:24s} n={n:5d} 熵={h:.3f}")
    zheng_list = ["湿热下注证", "肾虚证", "气滞血瘀证", "肝肾亏虚证"]
    effects = []
    for t, n, h in mk[:25]:
        for z in zheng_list:
            e = style_effect(gold, t, z)
            if e:
                effects.append(e)
    effects.sort(key=lambda x: x["p"])
    print(f"  有效检验 {len(effects)} 组；显著的（p<0.05）: "
          f"{sum(1 for e in effects if e['p'] < 0.05)} 组")
    for e in effects[:5]:
        print(f"    {e['marker']!r} @ {e['zheng']}: Jaccard={e['jaccard']:.3f} "
              f"(零假设中位 {e['null_median']:.3f}) p={e['p']:.3f}")

    # ---------------- 报告 ----------------
    d = {a["层"]: a["auc"] for a in abl}
    keys = [a["层"] for a in abl]
    L = ["# 处方成因归因报告", "",
         "回答：**医生开方是根据「中医指南 + 经验文档 + 中药药理」吗？各自占多大比重？**", "",
         "## 归因结论（逐药 macro AUC 累积消融）", "",
         "| 信息层 | macro AUC | 边际增益 | 对应医生的哪类依据 |", "|---|---|---|---|"]
    src = {
        "L0 常量(性别/年龄)": "—",
        "L1 +西医诊断(检查结果)": "**检查结果**（化验/影像诊断）",
        "L2 +证型/病名(辨证≈指南)": "**中医指南**（辨证分型 → 治法）",
        "L3 +症状(四诊细节)": "**四诊个体化**（望闻问切细节）",
        "L4 +既往处方(治疗进程)": "**经验 / 复诊调方**",
        "L5 +复诊间隔": "**治疗节奏判断**",
    }
    prev = None
    for k in keys:
        gain = "—" if prev is None else f"{d[k]-prev:+.4f}"
        L.append(f"| {k} | {d[k]:.4f} | {gain} | {src[k]} |")
        prev = d[k]
    L += ["", "## 医师风格 / 书写模板的存在性检验", "",
          f"最集中的 5 个文书用语（归一化熵越低越集中，= 越像某医生某时期的模板）：", "",
          "| 文书用语 | 出现次数 | 归一化熵 |", "|---|---|---|"]
    for t, n, h in mk[:5]:
        L.append(f"| {t} | {n} | {h:.3f} |")
    L += ["", f"共做 {len(effects)} 组「同证型下带/不带该标记」的处方对比，",
          f"其中 **{sum(1 for e in effects if e['p'] < 0.05)} 组显著（p<0.05）**。", "",
          "| 文书标记 | 证型 | 带标记 n | 不带 n | 两组方 Jaccard | 零假设中位 | p |",
          "|---|---|---|---|---|---|---|"]
    for e in effects[:8]:
        L.append(f"| {e['marker']} | {e['zheng']} | {e['n_with']} | {e['n_without']} | "
                 f"{e['jaccard']:.3f} | {e['null_median']:.3f} | {e['p']:.3f} |")

    gains = {}
    for i, k in enumerate(keys):
        gains[k] = None if i == 0 else d[k] - d[keys[i - 1]]
    sig = sum(1 for e in effects if e["p"] < 0.05)
    L += ["", "## 解读（基于实测，不是先验假设）", "",
          "### 1. 各层贡献排序（边际增益）", "",
          "| 排序 | 信息层 | 边际增益 | 对应依据 |", "|---|---|---|---|"]
    order = sorted([k for k in keys if gains[k] is not None],
                   key=lambda k: -gains[k])
    for i, k in enumerate(order, 1):
        L.append(f"| {i} | {k} | **{gains[k]:+.4f}** | {src[k]} |")
    L += ["",
          "### 2. ⚠️ 一个必须说清的度量差异：AUC ≠ Jaccard", "",
          "本报告用 **逐药 macro AUC**（能不能把每味药的概率排对顺序），",
          "而 §1.6 的基线用 **Jaccard**（能不能把整个方子复现对）。两者结论不同，且不矛盾：",
          "",
          "| 视角 | 指标 | 西医诊断(检查结果)的贡献 |", "|---|---|---|",
          "| 排序能力 | macro AUC | **+0.117，最大的单层增益** |",
          "| 集合复现 | Jaccard 检索 | 0.183 vs 常量 0.159，**几乎没提升** |",
          "",
          "**结论**：检查结果确实携带大量「某味药该不该用」的信息（排序用得上），",
          "但**不足以复现出正确的那张方子**（集合层面几乎无用）。",
          "仅看 Jaccard 会低估检查结果，仅看 AUC 会高估它 —— 两个指标都要报。",
          "",
          "### 3. 「中医指南」这一层其实是查表，没有自由度", "",
          "实测段级 **证型 → 治法 top1 占比 85.7%**（下焦湿热证 99.1%、阴虚证 99.5%）。",
          "治法基本是证型的确定性函数，**指南层面不产生额外信息**。",
          "所以 L2 的 +0.061 已经是「指南 + 辨证」的全部贡献。",
          "",
          "### 4. 「中药药理」不是输入，而是输出端约束", "",
          "仅凭 462 味药的组合反推治法 micro-F1 = **0.456**（略强于常量 0.424）。",
          "药理/功效解释了「为什么用这类药」，但不决定「为什么给这位患者用这味药」。",
          "**它应该做成约束（功效须覆盖治法），而不是当成一个输入特征。**",
          "",
          "### 5. 🔴 医师个人风格 / 书写模板是真实存在的独立成分", "",
          f"在 4 个证型 × 25 个最集中的文书用语上共做 {len(effects)} 组检验，",
          f"**{sig} 组显著（p<0.05）**。效应量极大：",
          "",
          f"- `下肢水肿可自行消退` @ 肾虚证：两组方的 top-15 Jaccard 仅 **0.200**，",
          "  而随机等量二分的零假设中位数是 **0.765** —— 差异远超偶然。",
          f"- `可自行消退` @ 肾虚证：**0.154** vs 0.765",
          f"- `病史同前` @ 湿热下注证：**0.250** vs 0.579",
          "",
          "**含义**：同一个证型、同一位患者画像，只因为现病史里出现了某个文书模板用语，",
          "处方就系统性不同。这不是医学逻辑，是**医生/模板层面**的成分。",
          "",
          "> ⚠️ 两点必须诚实说明：",
          "> (a) 这些标记不是纯风格 —— 「下肢水肿」本身是真实症状，可能对应真实的证型细分。",
          ">     但它们的时间分布熵极低（0.28–0.32，=集中在少数月份），更像某医生的模板。",
          "> (b) **因为数据里没有医生 ID，无法把「风格」与「患者亚型」彻底分开。**",
          "",
          "### 6. 这反过来污染了「症状」的贡献估计", "",
          f"L3（+症状）的 {gains.get('L3 +症状(四诊细节)', float('nan')):+.4f} 增益里，",
          "有一部分其实是**医生指纹**而非临床信息。所以：",
          "**+0.040 是症状贡献的上界，真实临床信息量低于它。**",
          "",
          "## 对建模的直接含义", "",
          "| 医生的依据 | 数据里有没有 | 模型该怎么处理 |", "|---|---|---|",
          "| 中医指南（辨证→治法） | ✅ 有 | 监督输出的骨架；但它≈查表，别指望它带来个性化 |",
          "| 中药药理（功效） | ❌ 无，需外部字典 | 作为**输出约束**（功效覆盖治法），不是输入 |",
          "| 检查结果（西医诊断） | ✅ 有 | 入输入；对**排序**有用、对**复现整方**帮助有限 |",
          "| 四诊细节（症状） | ✅ 有 | 核心输入，但贡献被医生指纹虚高 |",
          "| 经验 / 复诊调方 | ✅ 有（纵向关联） | **必须入输入**，是集合复现的最强因子（0.542） |",
          "| 医师个人风格 | ⚠️ 可观测、无法归因到人 | 视为噪声上限；**建议申请医生 ID** |",
          "| 复诊疗效反馈 | ❌ 完全没有 | **必须申请**（见 §10.4） |", "",
          "## 数据申请优先级（依据本报告）", "",
          "| 优先级 | 字段 | 理由 |", "|---|---|---|",
          "| P0 | **复诊疗效反馈**（好转/无效/加重/不良反应） | 唯一能把「模仿」推向「因果」的信号 |",
          "| P1 | **医生 ID** | 剥离医师风格，净化症状→药关联；否则无法判断模型学到的是医学还是习惯 |",
          "| P2 | **中药功效字典**（462 味） | 支撑可验证方解与约束解码（可从药典整理，不必向院方要） |",
          "| P3 | 剂型/煎服法、疗程天数 | 目前完全缺失 |", ""]

    (REPORT_DIR / "attribution_report.md").write_text("\n".join(L) + "\n",
                                                     encoding="utf-8")
    print(f"\n报告 -> reports/attribution_report.md")


if __name__ == "__main__":
    main()
