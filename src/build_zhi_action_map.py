"""
build_zhi_action_map.py — 构建「治法 → 功效」对齐表

为什么需要它（计划 §14.6）：
  步骤5 的硬约束是「方中功效 ⊇ 治法」，但治法词表和功效词表是两套词汇：
      治法「清热利湿」 → 功效需由「清热燥湿」+「利尿通淋/利水消肿」联合满足
      治法「补肾扶元」 → 功效需由「补肾阳/补肾气/补益肝肾」满足
  实测用朴素 2-gram 字符串匹配，覆盖率只有 73.1%（完全覆盖 60.4%）。
  本脚本产出对齐表，把覆盖率提上去。

双来源设计（关键）：
  规则层  基于中医语义等价（利湿≡利水≡渗湿≡利尿通淋），确定性、可审计
  数据层  基于共现 lift（该治法下的处方里，哪些功效词显著超基准），
          用真实数据发现规则漏掉的对应，但**必须中医师复核后才生效**

产出：
  data/external/zhi_action_map.json     对齐表（含来源/置信度/证据量）
  reports/zhi_action_map_report.md      中医师复核用表格

用法: python3 src/build_zhi_action_map.py
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import medlib as M
from medlib import DATA_DIR, REPORT_DIR

EXT = M.ROOT / "data" / "external"

# ---------------------------------------------------------------------------
# 规则层：中医语义单元 → 满足它的功效词
# 原则：只做**明确的等价/蕴含**，不做猜测。蕴含指「该功效一定能实现该治法单元」。
# ---------------------------------------------------------------------------

# ⚠️ 关键教训：不要手写功效名。
# 本草典用的是**另一套词汇**（补肾助阳 / 养阴润燥 / 养血补心 / 健脾益气），
# 而中医教材术语（补肾阳 / 滋阴 / 补血养心）在字典里**不存在**。
# 实测：手写的 171 个功效名里 27% 是无效字符串，永远匹配不上 ——
# 这正是规则层覆盖率（72.8%）被卡住、且恰好等于朴素字符串匹配的根因。
#
# 正确做法：为每个语义单元定义**正则模式**，在**真实功效词表**上匹配。
# 这样映射项必然存在，且新增药材字典时无需改表。
UNIT_PATTERNS: dict[str, str] = {
    # 清热
    "清热": r"清(热|湿热|暑|虚热)|泻火|泄热|降火|解毒|凉血",
    "泻火": r"泻火|泻肝火|降火|清热|泄热",
    "解毒": r"解毒|消痈|消斑|杀虫",
    "凉血": r"凉血",
    "利湿": r"(利|渗|化|除|清|燥|收)湿|利水|利尿|通淋|退黄|去浊",
    "利水": r"利水|利尿|通淋|消肿|逐水",
    "渗湿": r"渗湿|利水|祛湿|除湿",
    "通淋": r"通淋|利尿|利湿",
    "消肿": r"消肿|散结|软坚|消痈",
    "祛湿": r"祛湿|利湿|除湿|渗湿|燥湿|化湿|清利湿热|清湿热",
    "化湿": r"化湿|燥湿|利湿|去浊|降浊",
    "燥湿": r"燥湿|化湿",
    "化浊": r"化浊|去浊|降脂|除浊",
    # 补肾 / 滋肾 / 温肾
    "补肾": r"补肾|温肾|滋肾|益肾|暖肾|固肾|补肝肾|补益肝肾|滋补肝肾|交通心肾|强筋|壮骨|壮腰|固精|缩尿",
    "滋肾": r"滋阴|养阴|益阴|补阴|敛阴|坚阴|补血滋阴|补气养阴|补血养阴|滋补肾阴|生津",
    "温肾": r"温肾|补肾助阳|补肾阳|壮阳|助阳|补火助阳|回阳|温阳|温补",
    "补肝": r"补肝肾|滋补肝肾|温补肝肾|补益肝肾|养肝|柔肝|益肝|补肝",
    "补肺": r"补肺|益肺|润肺|养肺|宣肺|补脾肺",
    "补脾": r"健脾|补脾|益脾|醒脾|温脾",
    "健脾": r"健脾|补脾|益脾|醒脾|温脾|消食|化滞",
    "和胃": r"和胃|开胃|消食|降逆止呕|温中|宽中",
    "养心": r"养心|补心|宁心|安神|强心|和心志|交通心肾|补肾宁心",
    "安神": r"安神|宁心|定志|定惊|镇心|凉心",
    "养肝": r"养肝|柔肝|清肝|补肝|平肝",
    "润肺": r"润肺|养肺|宣肺|清肺|润燥化痰",
    "益精": r"益精|填精|固精|涩精|缩尿|止遗",
    "固精": r"固精|涩精|缩尿|止遗|固肾",
}
# 经脉/其他补益
UNIT_PATTERNS.update({
    "补气": r"补气|益气|大补元气|升阳|补中|生津",
    "益气": r"益气|补气|大补元气|升阳|补中",
    "补血": r"补血|养血|和血|生血|补益心血|补血养阴",
    "养血": r"养血|补血|和血|生血",
    "补阴": r"滋阴|养阴|益阴|补阴|坚阴|生津|润燥",
    "滋阴": r"滋阴|养阴|益阴|补阴|敛阴|坚阴|生津|润燥",
    "养阴": r"养阴|滋阴|益阴|生津|润燥",
    "补阳": r"助阳|壮阳|温阳|补火助阳|回阳|补肾阳",
    "温阳": r"温阳|助阳|壮阳|补火助阳|回阳|温经散寒|温中散寒",
    "扶元": r"大补元气|补气|益气|补肾|助阳|益精|补益气血",
    # 理气活血
    "疏肝": r"疏肝",
    "理气": r"理气|行气|下气|宽中|除胀|除满|破气|消痞",
    "行气": r"行气|理气|下气|宽中|除胀|除满|破气",
    "活血": r"活血|化瘀|祛瘀|散瘀|破血|逐瘀|和血|通经|通络|行血",
    "化瘀": r"化瘀|祛瘀|散瘀|逐瘀|活血|破血",
    "祛瘀": r"祛瘀|化瘀|散瘀|逐瘀|活血",
    "散结": r"散结|软坚|消痈|消肿|除痞",
    "通络": r"通络|通经|通痹|舒筋|搜风剔络|剔络",
    "止痛": r"止痛|镇痛|缓急",
    # 化痰止咳
    "化痰": r"化痰|祛痰|消痰|除痰|涤痰|豁痰",
    "祛痰": r"祛痰|化痰|消痰|涤痰|豁痰",
    "止咳": r"止咳|平喘|定喘",
    "平喘": r"平喘|定喘|止咳",
    # 其他
    "通便": r"通便|润肠|泻下|攻下|逐水",
    "泻下": r"泻下|攻下|逐水|通便",
    "润肠": r"润肠|通便|润燥",
    "涩肠": r"涩肠|止泻|固涩|收敛|止痢",
    "止泻": r"止泻|涩肠|止痢",
    "缩尿": r"缩尿|止遗|固精",
    "明目": r"明目",
    "平肝": r"平肝|平抑肝阳|潜阳|息风|镇惊|定惊",
    "潜阳": r"潜阳|平肝|平抑肝阳",
    "息风": r"息风|平肝|止痉|镇痉",
    "生津": r"生津|止渴|益阴",
    "润燥": r"润燥|生津|润肺|润肠",
    "消食": r"消食|化滞|消积|开胃|健脾",
    "利胆": r"利胆|退黄|清利湿热",
    "止汗": r"止汗|敛汗|固表",
    "固涩": r"固涩|收敛|涩精|涩肠|止遗|止汗",
    "散寒": r"散寒|温经|祛寒|发散风寒|温中",
    "祛风": r"祛风|搜风|散风|息风|祛风湿",
    "解表": r"解表|发散|疏散|宣散",
    "和解": r"和解|疏肝|调和",
    "交通": r"交通心肾|补肾宁心|养心|安神",
    "心肾": r"交通心肾|补肾宁心|温肾|养心",
    "滑肠": r"润肠|通便",
    "透疹": r"透疹",
    "截疟": r"截疟",
    "止痒": r"止痒|杀虫",
    "敛疮": r"敛疮|生肌|收湿",
    "止带": r"止带|固精|燥湿",
    "安胎": r"安胎",
    "升阳": r"升阳|举陷|升举|升发|升提",
    "举陷": r"举陷|升阳",
    "开窍": r"开窍|醒神|芳香|辟秽",
    "清心": r"清心|凉心|除烦|降火",
    "宣肺": r"宣肺|宣降肺气|润肺|清肺",
    "宁心": r"宁心|安神|养心|清心",
    "调理": r"调和|补益|益气|养血|滋阴|温阳",
    "阴阳": r"滋阴|养阴|助阳|温阳|调和",
    "平调": r"调和|平抑|平肝|清心",
    "寒热": r"调和|温阳|散寒|清热",
    "温下": r"温阳|散寒|泻下|通便",
    "清上": r"清热|清肝|清心|明目",
    "豁痰": r"豁痰|化痰|涤痰|祛痰",
    "消积": r"消积|消食|化滞|破气",
    "降逆": r"降逆|降气|下气|平冲",
    "止呕": r"止呕|降逆|和胃|温中",
    "坚阴": r"坚阴|滋阴|养阴",
    "敛肺": r"敛肺|收敛|固涩",
    "消痈": r"消痈|散结|解毒|消肿",
    "排脓": r"排脓|消痈|散结",
    # 补充：实测无映射的复合治法（占 0.38% 出现次数，但补上更完整）
    "滋补心阴": r"滋阴|养阴|益阴|养心|安神|生津",
    "补益脾肾": r"健脾|补脾|补肾|温肾|益气",
    "补益肝肾": r"补肝肾|补益肝肾|滋补肝肾|温补肝肾|养肝|强筋",
    "补益肺肾": r"补肺|益肺|润肺|补肾|温肾|养阴",
    "补益肺气": r"补肺|益肺|补气|益气|润肺",
    "补益心气": r"养心|补心|益气|补气|安神",
    "补益心脾": r"补益心脾|健脾|养心|益气|补血",
    "清肝熄风": r"清肝|平肝|息风|泻肝火|潜阳",
    "清肝泄火": r"清肝|泻肝火|泻火|清热|降火",
    "疏风清肺": r"疏散风热|疏散|祛风|清肺|宣肺|润肺",
    "表里双解": r"解表|发散|和解|清热|通便",
    "表里双解法": r"解表|发散|和解|清热|通便",
    "清肝熄风": r"清肝|平肝|息风|潜阳",
    "滋补": r"滋阴|养阴|补血|补肾|益精|生津",
    "补益": r"补气|益气|补血|养血|补肾|健脾|滋阴|助阳",
    "对症治疗": r".",   # 对症治疗无特定功效指向，匹配全部（不构成约束）
    "对症": r".",
})


def _build_unit_actions() -> dict[str, list[str]]:
    """从【真实功效词表】按模式生成 单元→功效 映射，保证映射项全部存在。"""
    if not (M.ROOT / "data/external/herb_dictionary.jsonl").exists():
        raise FileNotFoundError(
            "缺少 data/external/herb_dictionary.jsonl，请先运行 src/build_external_data.py")
    real = sorted({a for r in M.read_jsonl(M.ROOT / "data/external/herb_dictionary.jsonl")
                   for a in r["actions_zh"]})
    out: dict[str, list[str]] = {}
    for unit, pat in UNIT_PATTERNS.items():
        if unit.strip() == "#":
            continue
        rx = re.compile(pat)
        out[unit] = [a for a in real if rx.search(a)]
    return out


UNIT_TO_ACTIONS: dict[str, list[str]] = _build_unit_actions()

# 治法里常见的功能字单元（用于切分治法词）
UNITS = sorted(UNIT_TO_ACTIONS, key=len, reverse=True)


def split_zhi(zhi: str) -> list[str]:
    """把治法词切成语义单元：'清热利湿' -> ['清热','利湿']。"""
    units, i = [], 0
    while i < len(zhi):
        for u in UNITS:
            if zhi.startswith(u, i):
                units.append(u)
                i += len(u)
                break
        else:
            i += 1
    return units


def rule_actions(zhi: str) -> set[str]:
    out: set[str] = set()
    for u in split_zhi(zhi):
        out.update(UNIT_TO_ACTIONS[u])
    return out


# ---------------------------------------------------------------------------
# 数据层：该治法下的处方里，哪些功效词显著超基准
# ---------------------------------------------------------------------------

def data_actions(gold: list[dict], hd: dict, min_n: int = 40,
                 lift_min: float = 1.8, p_min: float = 0.20):
    """按治法统计功效词共现，返回 {治法: [(功效, P(功效|治法), lift, 处方数)]}"""
    zhi_rx: dict[str, list[set]] = defaultdict(list)
    all_acts: Counter = Counter()
    n_all = 0
    for r in gold:
        acts = {a for h in r["rx"] for a in hd.get(h, {}).get("actions_zh", [])}
        if not acts:
            continue
        n_all += 1
        for a in acts:
            all_acts[a] += 1
        for z in {t[2] for t in r["tcm_dx"] if t[2]}:
            zhi_rx[z].append(acts)
    base = {a: c / n_all for a, c in all_acts.items()}

    out = {}
    for z, rows in zhi_rx.items():
        n = len(rows)
        if n < min_n:
            continue
        cnt = Counter(a for acts in rows for a in acts)
        cands = []
        for a, c in cnt.items():
            p = c / n
            b = base.get(a, 1e-9)
            lift = p / b
            if p >= p_min and lift >= lift_min:
                cands.append((a, p, lift, c))
        cands.sort(key=lambda x: -x[2])
        out[z] = cands[:12]
    return out


# ---------------------------------------------------------------------------
# 评估：功效覆盖率
# ---------------------------------------------------------------------------

def coverage(gold: list[dict], hd: dict, zmap: dict[str, set[str]]) -> dict:
    """逐条处方：其每个治法是否被方中某味药的功效满足。"""
    per = []
    zhit: Counter = Counter()
    ztot: Counter = Counter()
    for r in gold:
        acts = {a for h in r["rx"] for a in hd.get(h, {}).get("actions_zh", [])}
        zhis = {t[2] for t in r["tcm_dx"] if t[2]}
        if not acts or not zhis:
            continue
        ok = 0
        for z in zhis:
            ztot[z] += 1
            if zmap.get(z) and (zmap[z] & acts):
                ok += 1
                zhit[z] += 1
        per.append(ok / len(zhis))
    return {
        "n": len(per),
        "avg": sum(per) / len(per) if per else 0.0,
        "full": sum(1 for x in per if x == 1) / len(per) if per else 0.0,
        "zhit": zhit, "ztot": ztot,
    }


def main() -> None:
    print("=" * 80)
    print("构建「治法 → 功效」对齐表")
    print("=" * 80)

    hd = {r["herb"]: r for r in M.read_jsonl(EXT / "herb_dictionary.jsonl")}
    allr = M.read_jsonl(DATA_DIR / "all_records.jsonl")
    gold = [r for r in allr if M.is_gold(r)]
    zhi_freq = Counter(t[2] for r in gold for t in r["tcm_dx"] if t[2])
    print(f"\n治法 {len(zhi_freq)} 种 / {sum(zhi_freq.values()):,} 次出现")
    print(f"药材字典 {len(hd)} 味，其中 {sum(1 for v in hd.values() if v['actions_zh'])} 味有功效")

    # ---------- 规则层 ----------
    rule_map = {z: rule_actions(z) for z in zhi_freq}
    n_rule = sum(1 for z in rule_map if rule_map[z])
    print(f"\n[规则层] 命中 {n_rule}/{len(zhi_freq)} 种治法"
          f"（覆盖 {sum(n for z, n in zhi_freq.items() if rule_map[z])/sum(zhi_freq.values())*100:.1f}% 的治法出现次数）")

    # ---------- 基线：朴素 2-gram ----------
    all_acts = sorted({a for v in hd.values() for a in v["actions_zh"]})
    def naive(z, acts):
        grams = {z[i:i + 2] for i in range(len(z) - 1)}
        return any(g in acts for g in grams)
    base_hit = 0; base_tot = 0
    for r in gold:
        acts = {a for h in r["rx"] for a in hd.get(h, {}).get("actions_zh", [])}
        if not acts:
            continue
        for z in {t[2] for t in r["tcm_dx"] if t[2]}:
            base_tot += 1
            if naive(z, " ".join(acts)):
                base_hit += 1
    print(f"[基线] 朴素 2-gram 匹配覆盖率: {base_hit/base_tot*100:.1f}%")

    # ---------- 规则层覆盖率 ----------
    cov_rule = coverage(gold, hd, rule_map)
    print(f"[规则层] 覆盖率 平均 {cov_rule['avg']*100:.1f}% / 完全覆盖 {cov_rule['full']*100:.1f}%")

    # ---------- 数据层 ----------
    print("\n[数据层] 共现 lift 挖掘（这是新知识发现，不是规则）...")
    dmap = data_actions(gold, hd)
    print(f"  有足够样本(>=40)的治法 {len(dmap)} 种")

    # 数据层补充：规则层未覆盖、或规则命中但数据强烈指向别的功效
    merged: dict[str, set[str]] = {}
    sources: dict[str, dict[str, str]] = {}
    for z in zhi_freq:
        s = set(rule_map[z])
        src = {a: "rule" for a in s}
        for a, p, lift, c in dmap.get(z, []):
            if a not in s and lift >= 2.5:
                s.add(a)
                src[a] = f"data(lift={lift:.1f})"
        merged[z] = s
        sources[z] = src
    cov_merged = coverage(gold, hd, merged)
    print(f"[合并] 覆盖率 平均 {cov_merged['avg']*100:.1f}% / 完全覆盖 {cov_merged['full']*100:.1f}%"
          f"  (基线 {base_hit/base_tot*100:.1f}% → +{(cov_merged['avg']-base_hit/base_tot)*100:.1f}pt)")

    # ---------- 输出 ----------
    out = {
        "meta": {
            "n_zhi": len(zhi_freq), "n_actions": len(all_acts),
            "coverage_baseline_2gram": base_hit / base_tot,
            "coverage_rule": cov_rule["avg"], "coverage_merged": cov_merged["avg"],
            "note": "rule=中医语义确定映射; data=共现 lift 挖掘，需中医师复核",
        },
        "map": {z: {
            "freq": zhi_freq[z],
            "units": split_zhi(z),
            "actions": sorted(merged[z]),
            "sources": sources[z],
        } for z in sorted(zhi_freq, key=lambda x: -zhi_freq[x])},
    }
    (EXT / "zhi_action_map.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n对齐表 -> data/external/zhi_action_map.json")

    # ---------- 复核报告 ----------
    n_only_data = sum(1 for z in merged for a in sources[z] if sources[z][a].startswith("data"))
    L = ["# 「治法 → 功效」对齐表 复核报告", "",
         "> **用途**：支撑《模型训练计划.md》§13 步骤5 的硬约束「方中功效 ⊇ 治法」。",
         "> **请中医师审阅本表**，尤其是标注 `data` 来源的条目（由数据挖掘出，未经医理确认）。", "",
         "## 效果", "",
         "| 方案 | 平均覆盖率 | 完全覆盖 |", "|---|---|---|",
         f"| 朴素 2-gram 字符串匹配（原基线） | {base_hit/base_tot*100:.1f}% | — |",
         f"| 规则层（中医语义等价） | {cov_rule['avg']*100:.1f}% | {cov_rule['full']*100:.1f}% |",
         f"| **规则 + 数据（本表）** | **{cov_merged['avg']*100:.1f}%** | "
         f"**{cov_merged['full']*100:.1f}%** |", "",
         f"- 治法总数 **{len(zhi_freq)}** 种，Top20 覆盖 89.7% 的出现次数",
         f"- 规则层命中 **{n_rule}** 种；数据层额外补充 **{n_only_data}** 条功效映射", "",
         "## 覆盖最差的治法（优先复核）", "",
         "| 治法 | 出现次数 | 覆盖率 | 当前映射的功效 |", "|---|---|---|---|"]
    worst = sorted([z for z in zhi_freq if zhi_freq[z] >= 50],
                   key=lambda z: (cov_merged["zhit"][z] / max(1, cov_merged["ztot"][z])))
    for z in worst[:20]:
        r_ = cov_merged["zhit"][z] / max(1, cov_merged["ztot"][z])
        acts = "、".join(sorted(merged[z]))[:90]
        L.append(f"| {z} | {zhi_freq[z]} | {r_*100:.0f}% | {acts or '**（空，需人工补）**'} |")

    L += ["", "## 高频治法对齐表（Top 40，请逐条审定）", "",
          "| 治法 | 次数 | 语义单元 | 映射功效 | 来源标记 |", "|---|---|---|---|---|"]
    for z in sorted(zhi_freq, key=lambda x: -zhi_freq[x])[:40]:
        acts = "、".join(sorted(merged[z]))
        mark = "、".join(f"{a}({sources[z][a]})" for a in sorted(merged[z])
                        if sources[z][a].startswith("data"))
        L.append(f"| {z} | {zhi_freq[z]} | {'+'.join(split_zhi(z))} | {acts} | "
                 f"{'⚠️' + mark if mark else '规则'} |")

    L += ["", "## 全部 `data` 来源条目（🔴 必须医理确认）", "",
          "这些映射是**统计共现**发现的，规则层没覆盖。可能是：",
          "（a）规则表遗漏的合理对应 → 请确认后并入规则层；",
          "（b）该治法下的处方恰好常用某类药 → **伪关联，应删除**。", "",
          "| 治法 | 治法次数 | 数据挖掘出的功效 | lift |", "|---|---|---|---|"]
    rows = []
    for z in zhi_freq:
        for a in merged[z]:
            if sources[z][a].startswith("data"):
                rows.append((z, zhi_freq[z], a, sources[z][a]))
    rows.sort(key=lambda x: -x[1])
    for z, f, a, s in rows[:60]:
        L.append(f"| {z} | {f} | {a} | {s.replace('data(lift=', '').rstrip(')')} |")
    if len(rows) > 60:
        L.append(f"| … | 共 {len(rows)} 条，完整见 JSON | | |")

    L += ["", "## 语义单元表（规则层的可审计基础）", "",
          "规则层由「治法切分为语义单元 → 单元映射到功效」构成，全部可审计：", "",
          "| 语义单元 | 满足它的功效 |", "|---|---|"]
    for u in sorted(UNIT_TO_ACTIONS):
        L.append(f"| {u} | {'、'.join(UNIT_TO_ACTIONS[u])} |")

    L += ["", "## 已知局限", "",
          "1. **单字治法未覆盖**：如「补肾」被切成单元命中，但「补」「和」等单字治法无映射。",
          "2. **复合治法**（6-10 字，共 47 种）切分可能不完整，覆盖率偏低。",
          "3. **数据层阈值**：lift>=2.5 且 P>=0.20，是经验值；阈值变化会改变补充条目。",
          "4. **本表不是医学权威**：规则层依据《中药学》功效术语的常见等价关系编写，",
          "   仍**必须经执业中医师审定**后方可用于临床相关场景。"]
    (REPORT_DIR / "zhi_action_map_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"复核报告 -> reports/zhi_action_map_report.md")
    print("=" * 80)


if __name__ == "__main__":
    main()
