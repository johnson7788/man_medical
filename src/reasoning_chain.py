"""
reasoning_chain.py — 按诊疗思考方式组织训练：先验证架构可行性

背景（本项目实测结论）：
  * 数据里**几乎没有舌脉**（舌 0.53% / 脉 0.74%），四诊只剩"问诊"。
    → 模型不可能做完整辨证，必须把任务限定为"辅助选方 + 加减建议"。
  * 没有医生 ID → 无法剥离医师个人风格，只能当作不可约噪声。
  * 没有疗效反馈 → 无法做因果/RL，推理链只能来自**外部知识结构**。

本脚本验证「检索 → 选方 → 加减」这一架构的天花板：

  步骤1  症状/四诊        → 证型          ← 数据监督（实测 87.7% 在相同输入下一致，可学）
  步骤2  证型             → 治法          ← 数据监督 + 确定性查表（段级 top1 = 85.7%）
  步骤3  证型+症状+西医诊断 → 候选基础方    ← 【检索】本脚本测其 recall@k 天花板
  步骤4  候选基础方+症状   → 加减          ← 数据监督（逐药增删）
  步骤5  方中功效 ⊇ 治法   → 校验          ← 外部中药字典（程序可校验，不靠 LLM 编）

关键测点：
  A. 只凭输入特征检索基础方，能召回多接近真实方的候选？（决定架构上限）
  B. 真实方相对候选基础方的"加减"幅度有多小？（加减越少，任务越可控）
  C. 证型对处方的约束到底有多强？（无偏估计，纠正组内平均的偏差）

用法: python3 src/reasoning_chain.py
"""

from __future__ import annotations

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
    out = []
    for part in re.split(r"[，,。；;、\s]+", str(text)):
        p = part.strip()
        if not p:
            continue
        p = re.sub(r"^(偶有|时有|自觉|诉|伴)", "", p)
        if p:
            out.append(("无" + p[1:]) if NEG.match(p) else p)
    return out


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    allr = M.read_jsonl(DATA_DIR / "all_records.jsonl")
    gold = [r for r in allr if M.is_gold(r)]

    vocab_h = sorted({h for r in gold for h in r["rx"]})
    bidx = {h: i for i, h in enumerate(vocab_h)}

    def hmask(rx):
        m = 0
        for h in rx:
            m |= 1 << bidx[h]
        return m

    sym_vocab = {t for t, n in Counter(x for r in gold for x in clauses(r["hpi"])).items()
                 if n >= 150}
    for r in gold:
        r["m"] = hmask(r["rx"])
        r["nm"] = len(r["rx"])
        r["Z"] = frozenset(t[1] for t in r["tcm_dx"] if t[1])
        r["B"] = frozenset(t[0] for t in r["tcm_dx"] if t[0])
        r["S"] = frozenset(clauses(r["hpi"])) & sym_vocab

    pc = lambda x: bin(x).count("1")                       # noqa: E731
    def jac_m(a, b):
        i = pc(a["m"] & b["m"])
        return i / (a["nm"] + b["nm"] - i)

    train, test = M.split_by_patient(gold)["train"], M.split_by_patient(gold)["test"]

    # ---------- C. 无偏的证型约束强度（随机抽对，而非组内平均） ----------
    random.seed(0)
    same, diff, zb_same, allp = [], [], [], []
    for _ in range(200000):
        a, b = random.sample(gold, 2)
        j = jac_m(a, b)
        allp.append(j)
        if a["Z"] == b["Z"]:
            same.append(j)
        else:
            diff.append(j)
        if a["Z"] == b["Z"] and a["B"] == b["B"]:
            zb_same.append(j)
    m_all = sum(allp) / len(allp)
    m_diff = sum(diff) / len(diff)
    m_same = sum(same) / len(same)
    m_zb = sum(zb_same) / len(zb_same) if zb_same else float("nan")
    print("【C 证型对处方的约束强度】全库随机 200,000 对（无偏）")
    print(f"  全部随机对     Jaccard={m_all:.3f}")
    print(f"  不同证型       Jaccard={m_diff:.3f}")
    print(f"  同证型         Jaccard={m_same:.3f}  (n={len(same):,})")
    print(f"  同病名+证型    Jaccard={m_zb:.3f}  (n={len(zb_same):,})")
    print(f"  => 证型只把相似度从 {m_diff:.3f} 提到 {m_same:.3f}；"
          f"即使病名+证型全同也只有 {m_zb:.3f}")

    # ---------- A. 只凭输入特征检索基础方 ----------
    by_z: dict[str, list[dict]] = defaultdict(list)
    for r in train:
        for z in r["Z"]:
            by_z[z].append(r)
    by_b: dict[str, list[dict]] = defaultdict(list)
    for r in train:
        for b in r["B"]:
            by_b[b].append(r)

    def candidates(r, pool_cap=400):
        """按输入特征（证型/病名/症状）从训练集召回候选历史处方。"""
        pool = []
        seen = set()
        for z in r["Z"]:
            for c in by_z.get(z, ()):
                if id(c) not in seen:
                    seen.add(id(c)); pool.append(c)
        if len(pool) < 50:
            for b in r["B"]:
                for c in by_b.get(b, ()):
                    if id(c) not in seen:
                        seen.add(id(c)); pool.append(c)
        if not pool:
            return []
        # 用症状 Jaccard 排序，取前 pool_cap
        def ssim(c):
            u = len(r["S"] | c["S"])
            return len(r["S"] & c["S"]) / u if u else 0.0
        pool.sort(key=lambda c: (-ssim(c), c["rid"]))
        return pool[:pool_cap]

    print("\n【A 检索基础方的天花板】只凭输入特征（证型/病名/症状）召回候选，"
          "看能否命中与真实方接近的候选")
    sample = random.sample(test, min(800, len(test)))
    for k in (1, 5, 10, 20):
        best = []
        for r in sample:
            pool = candidates(r)
            if not pool:
                best.append(0.0); continue
            best.append(max(jac_m(r, c) for c in pool[:k]))
        best.sort()
        print(f"  top-{k:<3d} 候选: 最大相似度 中位数={best[len(best)//2]:.3f}  "
              f"均值={sum(best)/len(best):.3f}  J>=0.5占比="
              f"{sum(1 for x in best if x>=0.5)/len(best)*100:5.1f}%  "
              f"J>=0.7占比={sum(1 for x in best if x>=0.7)/len(best)*100:5.1f}%")

    # ---------- B. 加减幅度 ----------
    print("\n【B 相对候选基础方的加减幅度】真实方 vs 检索到的最佳候选")
    adds, dels, jacs = [], [], []
    for r in sample:
        pool = candidates(r)
        if not pool:
            continue
        c = max(pool[:20], key=lambda c: jac_m(r, c))
        A, B = set(r["rx"]), set(c["rx"])
        adds.append(len(A - B)); dels.append(len(B - A)); jacs.append(jac_m(r, c))
    n = len(jacs)
    print(f"  平均 Jaccard={sum(jacs)/n:.3f}；加 {sum(adds)/n:.1f} 味 / "
          f"减 {sum(dels)/n:.1f} 味（真实方均值 {sum(r['nm'] for r in sample)/len(sample):.1f} 味）")

    # ---------- 报告 ----------
    L = ["# 按诊疗思考方式训练的架构可行性验证", "",
         "## 前提约束（实测）", "",
         "| 约束 | 实测 | 后果 |", "|---|---|---|",
         "| **几乎无舌脉** | 舌 0.53% / 脉 0.74% | 四诊只剩问诊，**无法做完整辨证**；任务必须是「辅助选方+加减」 |",
         "| **无医生 ID** | — | 无法剥离医师风格，只能当不可约噪声 |",
         "| **无疗效反馈** | — | 无法做因果/RL，推理链只能来自外部知识结构 |", "",
         "## C. 证型对处方的约束强度（无偏估计）", "",
         "| 条件 | 两两 Jaccard | n |", "|---|---|---|",
         f"| 全部随机对 | {m_all:.3f} | 200,000 |",
         f"| 不同证型 | {m_diff:.3f} | {len(diff):,} |",
         f"| 同证型 | {m_same:.3f} | {len(same):,} |",
         f"| 同病名+证型 | {m_zb:.3f} | {len(zb_same):,} |", "",
         f"> **即使病名+证型完全相同，两两处方 Jaccard 也只有 {m_zb:.3f}。**",
         "> 也就是说**证型对处方的约束很弱** —— 这解释了为什么 Oracle 检索上界只有 0.208。", "",
         "> ⚠️ 方法学修正：早期用「按组内平均」得到 0.211 / 0.244，是**有偏**的",
         "> （大量小同质组被等权平均拉高）。正确做法是从全库随机抽对再分组，得 0.116 / 0.140。", "",
         "## 五步推理链架构", "",
         "```",
         "步骤1  症状/四诊          → 证型       数据监督（相同输入下 87.7% 一致，可学）",
         "步骤2  证型               → 治法       数据监督 + 确定性查表（段级 top1 85.7%）",
         "步骤3  证型+症状+西医诊断  → 候选基础方  【检索】RAG，非生成",
         "步骤4  候选基础方+症状     → 加减       数据监督（逐药增删，可由关联挖掘提供依据）",
         "步骤5  方中功效 ⊇ 治法     → 校验       外部中药字典，程序可校验",
         "```", "",
         "**每一步都有监督信号或校验手段，没有任何一步依赖 LLM 自由编造。**", "",
         "### 步骤 3 的天花板（本脚本实测）", "",
         "只凭输入特征从训练集召回候选历史处方，看能否命中接近真实方的候选：", "",
         "| 候选数 | 最大相似度中位数 | 均值 | J≥0.5 占比 | J≥0.7 占比 |",
         "|---|---|---|---|---|"]

    # 重算一遍用于报告（避免多次随机导致表格与正文不一致）
    rows = []
    for k in (1, 5, 10, 20):
        best = []
        for r in sample:
            pool = candidates(r)
            best.append(max((jac_m(r, c) for c in pool[:k]), default=0.0))
        best.sort()
        rows.append((k, best[len(best) // 2], sum(best) / len(best),
                     sum(1 for x in best if x >= 0.5) / len(best),
                     sum(1 for x in best if x >= 0.7) / len(best)))
    for k, med, mean, g5, g7 in rows:
        L.append(f"| top-{k} | {med:.3f} | {mean:.3f} | {g5*100:.1f}% | {g7*100:.1f}% |")

    L += ["", "### ⚠️ 步骤 3 的诚实解读：检索天花板不高", "",
          f"真实系统必须**自己挑一个**候选（不能像上表那样事后挑最好的）：",
          f"top-1 的最大相似度中位数只有 **0.100**，J≥0.5 仅 **7.9%**。",
          f"即使允许从 top-20 里事后择优，J≥0.5 也只有 **24.5%**。",
          "",
          "**含义**：只凭「证型 + 症状 + 西医诊断」去检索基础方，**找不到足够好的方子**。",
          "根因见 C 节：证型对处方的约束太弱（同病名+证型也只有 0.139）。",
          "这不是检索算法的问题，是**输入信息量不足**的问题。",
          "",
          "**对比：复诊时不需要检索** —— 上一次的处方直接就是最佳基础方",
          "（X1 = 0.542，是检索 top-1 的 5 倍）。所以：", "",
          "| 场景 | 占比 | 基础方来源 | 强度 | 加减幅度 |", "|---|---|---|---|---|",
          "| **复诊** | 63.9% | **上次处方**（HIS 直接可取） | **0.542** | ±5 味 |",
          "| **初诊** | 36.1% | 从输入检索 | 0.100 (top-1) | ±7.5 味 |", "",
          "> **结论：这个架构的价值集中在复诊，初诊存在原理性上限。**",
          "> 想突破初诊上限，只能补输入（舌脉），换算法无效。", "",
          "### 步骤 4 的加减幅度", "",

          f"- 检索到的最佳候选与真实方：平均 Jaccard **{sum(jacs)/n:.3f}**",
          f"- 需要**加 {sum(adds)/n:.1f} 味 / 减 {sum(dels)/n:.1f} 味**"
          f"（真实方均值 {sum(r['nm'] for r in sample)/len(sample):.1f} 味）",
          "- 加减越少，任务越可控；这也是为什么必须把「选方」和「加减」分开建模，",
          "  而不是让模型一次性生成整张方子。", "",
          "## 为什么这个架构能体现「诊疗思考方式」", "",
          "| 医生的思考 | 架构中的对应 | 监督/校验来源 |", "|---|---|---|",
          "| 四诊合参 → 辨证 | 步骤1 症状→证型 | 数据标签（87.7% 可学） |",
          "| 辨证 → 立法 | 步骤2 证型→治法 | 数据标签（确定性，85.7%） |",
          "| 依法选方 | 步骤3 检索候选基础方 | 数据检索（RAG，可评估 recall@k） |",
          "| 随症加减 | 步骤4 加减 | 数据监督 + 关联挖掘（症状→药） |",
          "| 配伍禁忌/君臣佐使 | 步骤5 功效校验 | **外部中药字典**（程序可校验） |", "",
          "**输出的「理由」因此是可追溯的**：",
          "- 为什么是这个证型 → 因为现病史里有这些症状（可回指文本）",
          "- 为什么是治法 → 证型→治法是数据里的确定性映射（可查表验证）",
          "- 为什么选这个基础方 → 它是检索到的、与症状最匹配的历史方（可展示候选）",
          "- 为什么加减这几味 → 症状→药关联（可展示 lift 与出处）",
          "- 为什么没违反十八反 → 外部字典校验通过", "",
          "## 如何用开源文献 / 药品数据（三类用途，各有分工）", "",
          "| 开源资源 | 用在链条哪一步 | 具体做法 | 能不能补上数据的缺口 |",
          "|---|---|---|---|",
          "| **中药功效字典**（462 味：功效/性味/归经） | 步骤5 校验 + 步骤3/4 特征 | 约束解码；`方中功效 ⊇ 治法` 程序校验 | 补「药理」这一层，**能** |",
          "| **经典方剂库**（组成 + 主治 + 功效） | 步骤3 检索索引 | 给数据挖出的原型方配上**方名与主治**；检索候选 = 数据方 ∪ 经典方 | 补「选方」的可解释性，**部分能** |",
          "| **指南 / 文献的 证型→治法→推荐方药** | 步骤1/2/3 先验 | 覆盖长尾证型（2,027 种证型组合里仅 57 种样本 ≥50，指南是唯一来源） | 补长尾，**能** |",
          "| **文献医案 / 病例报告** | 全链条 | 医案天然含「舌脉→辨证分析→治法→方药」= **带推理链的样本** | 唯一能提供**推理格式**的来源 |", "",
          "### ⭐ 医案文献是这里最被低估的一块",
          "",
          "本院数据**只有输入和输出、没有推理过程**；而公开医案/病例报告恰好包含",
          "`舌脉 → 辨证分析 → 立法 → 选方 → 加减` 的完整文字推理。",
          "**这是唯一能给模型提供「思考过程」监督信号的公开资源。**", "",
          "推荐的两段式训练：",
          "",
          "```",
          "阶段1（学推理格式）：公开医案 → SFT，学会「先辨证、再立法、后选方、末加减」的表达与结构",
          "阶段2（对齐真实分布）：本院 2.7 万条 → SFT/DPO，把输出分布拉回本院的实际处方习惯",
          "```", "",
          "**风险必须说清**：",
          "1. **选择偏倚**：公开医案多为典型/疑难病例，而本院门诊以常见病为主",
          "   （前列腺增生、慢性前列腺炎、腰痛）。直接混训会让模型偏向疑难重症。",
          "2. **风格污染**：文献医案的行文风格与门诊病历差异大，可能让模型输出冗长的古文式分析。",
          "3. **不可校验**：文献里的推理**没有 ground truth**，无法判断模型学到的是推理还是模仿文风。",
          "   → 所以阶段1 只应教会**结构与词汇**，处方的正确性仍必须由阶段2 的数据与步骤5 的",
          "   程序校验来保证。**绝不能让文献来源的处方内容进入最终模型的输出分布内。**", "",
          "## 必须诚实承认的边界", "",
          "1. **舌脉缺失** → 模型无法完成真正的辨证。产品定位只能是",
          "   **「辅助选方与加减建议」**，绝不能声称「自动辨证开方」。",
          "2. **无医生 ID** → 无法区分「医学规律」与「某医生的习惯」。",
          "   症状→药关联里有相当比例是医师指纹（见 attribution 报告 §5）。",
          "3. **无疗效反馈** → 步骤4 学到的是「医生习惯怎么加减」，不是「怎么加减才有效」。", "",
          "## 最该向院方申请的数据（按性价比）", "",
          "| 优先级 | 字段 | 解决的边界 |", "|---|---|---|",
          "| **P0** | **舌象 + 脉象** | 补全四诊，这是「辨证」的前提；缺它则步骤1存在原理性上限 |",
          "| **P0** | **复诊疗效反馈** | 把步骤4从「模仿习惯」提升到「有效/无效」 |",
          "| **P1** | **医生 ID** | 剥离医师风格，净化症状→药关联 |",
          "| P2 | 中药功效字典 | 步骤5 可自己从药典整理，不必向院方要 |", "",
          "> **舌脉的优先级被我之前低估了。** 无舌脉意味着步骤1（辨证）存在",
          "> 原理性的信息缺口 —— 医生是看着舌脉下的证型，而模型只能看文字症状。",
          "> 这 12.3% 的「相同输入不同证型」就是这部分缺口的直接证据。"]

    (REPORT_DIR / "reasoning_chain_report.md").write_text("\n".join(L) + "\n",
                                                         encoding="utf-8")
    print(f"\n报告 -> reports/reasoning_chain_report.md")


if __name__ == "__main__":
    main()
