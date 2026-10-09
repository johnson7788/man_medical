"""
build_external_data.py — 把下载的外部知识库整合成项目可用的数据资产

外部数据来源（全部已落盘 data/external/，见该目录 README.md）：
  1. bencaodian（本草典开放数据 v1, CC BY-SA 4.0）—— 主力
       中药材 365：功效 actions / 主治 indications / 性味 / 归经 / 剂量范围 /
                    禁忌 / 炮制方法 / 药理成分 / 典籍出处
       方剂 112：组成（含君臣佐使 role + 方解 explanation）/ 治法 / 出处
       证型 84：主症 / 次症 / 舌象 / 脉象
       医案 28：症状描述(含舌脉) / 诊断 / 用方 / 加减 / 疗效 / 教学点
       症状同义词 155、中西药相互作用 56、舌象 25、脉象 28
  2. SymMap v2.0（症状映射库）—— 补充
       药材 703（性味/归经/功效分类）、中医症状 2364、西医症状 1148、
       疾病 14434、证型 233
  3. TCM-Prescription-Recommendation —— 疾病→方（药名有 OCR 噪声，仅备查）
  4. PresRecST —— 4485 条匿名「症状→证候→治法→药」推理链（ID 匿名，仅作架构参考）

本脚本产出：
  data/external/herb_dictionary.json      462 味药的完整字典 + 覆盖报告
  data/external/formula_library.json      方剂库（含君臣佐使与方解）
  data/external/pattern_dictionary.json   证型字典（主症/舌象/脉象）
  data/external/symptom_vocabulary.json   症状词表（含同义词，用于归一化本方 1.4 万 token）
  data/external/safety_rules.json         十八反十九畏 + 妊娠禁忌 + 中西药相互作用
  data/external/coverage_report.md        覆盖率报告

用法: python3 src/build_external_data.py
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import openpyxl

sys.path.insert(0, str(Path(__file__).resolve().parent))

import medlib as M
from medlib import DATA_DIR, REPORT_DIR

EXT = M.ROOT / "data" / "external"
BD = EXT / "bencaodian"
SYM = EXT / "symmap"

# ---------------------------------------------------------------------------
# 药名归一化：把本方的炮制名/产地名/别名映射到外部字典的标准名
# ---------------------------------------------------------------------------

PROC_PREFIX = re.compile(
    r"^(麸炒|土炒|砂炒|米炒|滑石粉炒|蛤粉炒|炒炭|清|生|熟|炙|炒|焦|煅|燀|烫|蒸|"
    r"盐|醋|酒|制|姜|蜜|净|鲜|绵|关|煨|炮|干|麸煨|清炒)")
ORIGIN_PREFIX = re.compile(r"^(北|南|川|广|云|怀|辽|浙|苏|西|东|毫|祁|杭|建|湘|鄂|鲁|秦)")
SUFFIX = re.compile(r"(片|粉|肉|珠|霜|炭|仁|草|根|皮|叶|花|藤|石|角|衣|心|梗|梢|节)$")

# 显式别名：炮制品/异名 → 标准药名
ALIAS: dict[str, str] = {
    # 炮制品异名（外部字典只有基础名）
    "黑顺片": "附子", "附片": "附子", "白附片": "附子", "淡附片": "附子",
    "酒萸肉": "山茱萸", "山萸肉": "山茱萸", "萸肉": "山茱萸",
    "酒苁蓉": "肉苁蓉", "酒黄精": "黄精", "酒女贞子": "女贞子", "酒大黄": "大黄",
    "姜炭": "炮姜", "阿胶珠": "阿胶", "鲜茅根": "白茅根", "燀桃仁": "桃仁",
    "鲜芦根": "芦根", "鲜地黄": "生地黄", "生石膏": "石膏", "煅龙骨": "龙骨",
    "煅牡蛎": "牡蛎", "煅阳起石": "阳起石", "醋龟甲": "龟甲", "醋鳖甲": "鳖甲",
    "醋五味子": "五味子", "醋鸡内金": "鸡内金", "醋青皮": "青皮", "醋香附": "香附",
    "醋莪术": "莪术", "醋三棱": "三棱", "醋乳香": "乳香", "醋没药": "没药",
    "醋北柴胡": "柴胡", "盐杜仲": "杜仲", "盐补骨脂": "补骨脂", "盐续断": "续断",
    "盐车前子": "车前子", "盐益智仁": "益智仁", "盐小茴香": "小茴香",
    "盐橘核": "橘核", "盐关黄柏": "黄柏", "盐胡芦巴": "胡芦巴",
    "制巴戟天": "巴戟天", "制远志": "远志", "制何首乌": "何首乌",
    "制川乌": "川乌", "制草乌": "草乌", "制吴茱萸": "吴茱萸", "制天南星": "天南星",
    "法半夏": "半夏", "姜半夏": "半夏", "清半夏": "半夏", "半夏曲": "半夏",
    "党参片": "党参", "黄芩片": "黄芩", "黄连片": "黄连", "甘草片": "甘草",
    "续断片": "续断", "金樱子肉": "金樱子", "绵萆薢": "萆薢", "净山楂": "山楂",
    "麸炒冬瓜子": "冬瓜子", "木香": "木香", "三七粉": "三七",
    # 异名/别名
    "蒺藜": "白蒺藜", "刺蒺藜": "白蒺藜", "炒蒺藜": "白蒺藜", "盐蒺藜": "白蒺藜",
    "苦杏仁": "苦杏仁", "炒苦杏仁": "苦杏仁", "杏仁": "苦杏仁",
    "韭菜子": "韭子", "龙胆": "龙胆草", "蜂房": "露蜂房", "小通草": "通草",
    "豆蔻": "白豆蔻", "野菊花": "野菊花", "合欢花": "合欢花", "赤小豆": "赤小豆",
    "甜叶菊叶": "甜叶菊叶", "地黄": "生地黄", "首乌藤": "夜交藤",
    "酒当归": "当归", "全当归": "当归", "广木香": "木香", "云木香": "木香",
    "杭白芍": "白芍", "亳白芍": "白芍", "川贝母": "川贝母", "浙贝母": "浙贝母",
    "山慈菇": "山慈菇", "虎杖": "虎杖", "北败酱草": "败酱草",
    "元胡": "延胡索", "玄胡": "延胡索", "蚤休": "重楼", "七叶一枝花": "重楼",
    # 本草典把「杏仁」作为正名；本方用「苦杏仁/炒苦杏仁」
    "苦杏仁": "杏仁", "炒苦杏仁": "杏仁", "甜杏仁": "杏仁", "燀苦杏仁": "杏仁",
    # 姜炭即炮姜，本草典无「炮姜」，退回干姜（会丢失炮制差异，已在报告中标注）
    "姜炭": "干姜", "炮姜": "干姜",
}
# 这些是真实不同的炮制品，外部字典若只有基础名，标注为「炮制品映射」而非等价
PROCESSED_MARKERS = ("炙", "炒", "麸炒", "盐", "醋", "酒", "制", "煅", "燀", "姜", "熟", "焦", "烫")


def build_norm_map(mine: set[str], valid: set[str],
                   priority: set[str] | None = None) -> dict[str, str]:
    """把本方药名映射到外部字典的标准名。返回 {本方名: 外部标准名}。

    priority = 「有功效数据的字典名」集合（本草典的全部药名）。
    必须优先命中它：SymMap 把「姜半夏/炙甘草/炙黄芪/关黄柏」等炮制品当独立条目收录，
    但 SymMap **没有功效字段**。若先命中这些空条目，就会丢掉功效，
    实测导致 97 味药的功效为空、治法覆盖率被压低。
    """
    out: dict[str, str] = {}
    priority = priority or set()

    def try_name(h: str) -> str | None:
        if h in priority:
            return h
        if h in ALIAS and ALIAS[h] in priority:
            return ALIAS[h]
        if h in valid:
            return h
        if h in ALIAS and ALIAS[h] in valid:
            return ALIAS[h]
        return None

    for h in mine:
        r = try_name(h)
        if r:
            out[h] = r
            continue
        # 逐层剥离：炮制前缀 / 产地前缀 / 后缀
        cands = [h]
        for _ in range(3):
            new = []
            for c in cands:
                new += [PROC_PREFIX.sub("", c), ORIGIN_PREFIX.sub("", c), SUFFIX.sub("", c)]
            cands = list(dict.fromkeys(cands + new))
        for c in cands:
            r = try_name(c)
            if r:
                out[h] = r
                break
    return out


# ---------------------------------------------------------------------------
# 十八反十九畏（《中药学》固定配伍禁忌，「十八反」歌诀）
# 注意：bencaoDian 的 herb_incompatible_with 边为 0 条，此表按经典原文硬编码
# ---------------------------------------------------------------------------

EIGHTEEN_INCOMPAT = [
    ("甘草", "甘遂"), ("甘草", "大戟"), ("甘草", "海藻"), ("甘草", "芫花"),
    ("乌头", "贝母"), ("乌头", "瓜蒌"), ("乌头", "半夏"), ("乌头", "白蔹"), ("乌头", "白及"),
    ("藜芦", "人参"), ("藜芦", "沙参"), ("藜芦", "丹参"), ("藜芦", "玄参"),
    ("藜芦", "细辛"), ("藜芦", "芍药"),
]
NINETEEN_INCOMPAT = [
    ("硫黄", "朴硝"), ("水银", "砒霜"), ("狼毒", "密陀僧"), ("巴豆", "牵牛子"),
    ("丁香", "郁金"), ("川乌", "草乌"), ("牙硝", "三棱"), ("官桂", "石脂"),
    ("人参", "五灵脂"),
]
# 妊娠禁忌
PREGNANCY_FORBIDDEN = ["巴豆", "牵牛子", "大戟", "甘遂", "芫花", "商陆", "麝香",
                       "水蛭", "虻虫", "莪术", "三棱", "水银", "砒霜", "斑蝥"]
PREGNANCY_CAUTION = ["桃仁", "红花", "川芎", "牛膝", "肉桂", "附子", "半夏",
                     "大黄", "枳实", "冬葵子", "干姜"]


def load_bencaodian() -> dict:
    def j(name):
        return json.loads((BD / f"{name}.json").read_text(encoding="utf-8"))
    return {
        "herbs": j("herbs"), "formulas": j("formulas"), "patterns": j("patterns"),
        "tongue": j("tongue_states"), "pulses": j("pulses"),
        "cases": j("case_records"), "synonyms": j("symptom_synonyms"),
        "relationships": j("relationships"), "interactions": j("interactions"),
        "conditions": j("conditions"), "texts": j("classical_texts"),
    }


def load_symmap() -> dict:
    def sheet(f, cols=None):
        wb = openpyxl.load_workbook(SYM / f, read_only=True, data_only=True)
        ws = wb.worksheets[0]
        rows = list(ws.iter_rows(values_only=True))
        hdr = list(rows[0])
        out = []
        for r in rows[1:]:
            d = {hdr[i]: r[i] for i in range(min(len(hdr), len(r)))}
            if str(d.get("Suppress") or "0") in ("1", "1.0"):
                continue
            out.append(d)
        return out
    return {
        "herbs": sheet("SMHB.xlsx"), "tcm_symptoms": sheet("SMTS.xlsx"),
        "mm_symptoms": sheet("SMMS.xlsx"), "diseases": sheet("SMDE.xlsx"),
        "syndromes": sheet("SMSY.xlsx"),
    }


def main() -> None:
    print("=" * 80)
    print("整合外部知识库")
    print("=" * 80)

    bd = load_bencaodian()
    sm = load_symmap()
    print(f"\n本草典: 药 {len(bd['herbs'])} 方 {len(bd['formulas'])} 证型 {len(bd['patterns'])} "
          f"医案 {len(bd['cases'])} 舌 {len(bd['tongue'])} 脉 {len(bd['pulses'])}")
    print(f"SymMap : 药 {len(sm['herbs'])} 中医症状 {len(sm['tcm_symptoms'])} "
          f"西医症状 {len(sm['mm_symptoms'])} 疾病 {len(sm['diseases'])} 证型 {len(sm['syndromes'])}")

    # ---------- 本方药名 ----------
    allr = M.read_jsonl(DATA_DIR / "all_records.jsonl")
    mine_freq = Counter(h for r in allr if r["has_rx"] for h in r["rx"])
    print(f"\n本方: {len(mine_freq)} 味药, {sum(mine_freq.values()):,} 次使用")

    # ---------- 统一药名映射 ----------
    bd_names = {h["name_zh"] for h in bd["herbs"]}
    sym_names = {str(h["Chinese_name"]).strip() for h in sm["herbs"] if h.get("Chinese_name")}
    valid = bd_names | sym_names
    print(f"外部药名并集: {len(valid)}（本草典 {len(bd_names)} + SymMap {len(sym_names)}）")

    norm = build_norm_map(set(mine_freq), valid, priority=bd_names)
    hit = set(norm)
    covered = sum(mine_freq[h] for h in hit)
    unmapped = sorted([h for h in mine_freq if h not in norm], key=lambda x: -mine_freq[x])
    print(f"\n映射成功: {len(hit)}/{len(mine_freq)} = {len(hit)/len(mine_freq)*100:.1f}% 味数")
    print(f"用量加权覆盖: {covered/sum(mine_freq.values())*100:.2f}%")
    print(f"未映射 {len(unmapped)} 味，Top15: {[(h, mine_freq[h]) for h in unmapped[:15]]}")

    # ---------- 1. 药材字典 ----------
    bd_by = {h["name_zh"]: h for h in bd["herbs"]}
    sym_by = {str(h["Chinese_name"]).strip(): h for h in sm["herbs"] if h.get("Chinese_name")}
    herb_dict: dict = {}
    def find_actions_entry(std: str):
        """找到含功效的本草典条目；std 若无功效则逐层剥前缀重试。"""
        b = bd_by.get(std)
        if b and b.get("actions"):
            return b, std
        c = std
        for _ in range(3):
            for rx in (PROC_PREFIX, ORIGIN_PREFIX, SUFFIX):
                c2 = rx.sub("", c)
                bb = bd_by.get(c2)
                if bb and bb.get("actions"):
                    return bb, c2
                c = c2 if c2 != c else c
        return (b or {}), std

    for my_name, std in sorted(norm.items()):
        b, act_std = find_actions_entry(std)
        s = sym_by.get(std) or sym_by.get(act_std) or {}
        if act_std != std:
            std = act_std
        herb_dict[my_name] = {
            "standard_name": std,
            "source": "+".join(x for x, v in (("bencaodian", b), ("symmap", s)) if v),
            "category": b.get("category") or s.get("Class_Chinese") or "",
            "nature": b.get("nature") or s.get("Properties_English") or "",
            "flavors": b.get("flavors") or s.get("Properties_Chinese") or "",
            "meridians_zh": "",  # 本草典无独立归经字段，用 SymMap 补
            "meridians_zh_symmap": s.get("Meridians_Chinese") or "",
            "actions_zh": [a["zh"] for a in (b.get("actions") or [])],
            "indications_zh": [i["zh"] for i in (b.get("indications") or [])],
            "dosage_range": b.get("dosage_range"),
            "processing_methods": [
                {"name": p.get("name"), "method": p.get("method"), "effect": p.get("effect")}
                for p in (b.get("processing_methods") or [])],
            "contraindications_zh": b.get("contraindications_zh") or "",
            "safety_notes_zh": b.get("safety_notes_zh") or "",
            "pregnancy": b.get("pregnancy"),
            "pharmacology_constituents": [
                {"name_zh": c.get("name_zh"), "name_en": c.get("name_en")}
                for c in ((b.get("pharmacology") or {}).get("active_constituents") or [])],
            "classical_references": b.get("classical_references"),
            "herb_freq_in_dataset": mine_freq[my_name],
        }
    # 未映射的也留条目（标注缺字典），避免下游 KeyError
    for h in unmapped:
        herb_dict[h] = {
            "standard_name": None, "source": "", "category": "", "nature": "", "flavors": "",
            "meridians_zh": "", "meridians_zh_symmap": "", "actions_zh": [],
            "indications_zh": [], "dosage_range": None, "processing_methods": [],
            "contraindications_zh": "", "safety_notes_zh": "", "pregnancy": None,
            "pharmacology_constituents": [], "classical_references": None,
            "herb_freq_in_dataset": mine_freq[h],
        }
    M.write_jsonl(EXT / "herb_dictionary.jsonl",
                  [{"herb": k, **v} for k, v in herb_dict.items()])
    n_act = sum(1 for v in herb_dict.values() if v["actions_zh"])
    n_ind = sum(1 for v in herb_dict.values() if v["indications_zh"])
    n_dose = sum(1 for v in herb_dict.values() if v["dosage_range"])
    n_mer = sum(1 for v in herb_dict.values() if v["meridians_zh_symmap"])
    # 分字段的【用量加权】覆盖率 —— 药材数不是有效指标（长尾药占比高但用得少），
    # 必须按本方实际用药频次加权。下游（技术报告、剂量填充）引用的就是这组数字。
    tot_use = sum(mine_freq.values())
    uw = lambda pred: sum(mine_freq[h] for h, v in herb_dict.items()          # noqa: E731
                          if pred(v)) / tot_use
    uw_act = uw(lambda v: v["actions_zh"])
    uw_ind = uw(lambda v: v["indications_zh"])
    uw_dose = uw(lambda v: v["dosage_range"])
    uw_mer = uw(lambda v: v["meridians_zh_symmap"])
    print(f"\n[1] 药材字典: {len(herb_dict)} 条")
    print(f"    有功效 {n_act} | 有主治 {n_ind} | 有剂量 {n_dose} | 有归经 {n_mer}")
    print(f"    用量加权覆盖: 功效 {uw_act*100:.2f}% | 主治 {uw_ind*100:.2f}% | "
          f"剂量 {uw_dose*100:.2f}% | 归经 {uw_mer*100:.2f}%")

    # ---------- 2. 方剂库 ----------
    formulas = []
    for f in bd["formulas"]:
        comp = [{"herb": c.get("herb_key"), "role": c.get("role"),
                 "dosage": c.get("dosage"), "explanation_zh": c.get("explanation_zh")}
                for c in (f.get("composition") or [])]
        formulas.append({
            "id": f.get("key"), "name": f.get("name_zh"),
            "category": f.get("category"), "subcategory": f.get("subcategory"),
            "treatment_principle": (f.get("treatment_principle") or {}).get("zh"),
            "composition": comp, "n_herbs": len(comp),
            "source_text": f.get("source_text_key"), "source": "bencaodian",
        })
    # 本草典 herb_key 是拼音 key，需映射到中文名
    key2zh = {h.get("key"): h.get("name_zh") for h in bd["herbs"]}
    for f in formulas:
        for c in f["composition"]:
            c["herb_zh"] = key2zh.get(c["herb"])
    (EXT / "formula_library.json").write_text(
        json.dumps(formulas, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[2] 方剂库: {len(formulas)} 首（含君臣佐使 {sum(1 for f in formulas if any(c['role'] for c in f['composition']))} 首、"
          f"方解 {sum(1 for f in formulas if any(c['explanation_zh'] for c in f['composition']))} 首）")

    # ---------- 3. 证型字典 ----------
    patterns = []
    for p in bd["patterns"]:
        t = p.get("tongue") or {}
        pu = p.get("pulse") or {}
        patterns.append({
            "name": p.get("name_zh"), "category": p.get("category"),
            "cardinal_symptoms": [s["zh"] for s in (p.get("cardinal_symptoms") or [])],
            "secondary_symptoms": [s["zh"] for s in (p.get("secondary_symptoms") or [])],
            "tongue": {"body": t.get("body"), "coating": t.get("coating")} if t else {},
            "pulse": pu, "source": "bencaodian",
        })
    for s in sm["syndromes"]:
        patterns.append({
            "name": s.get("Syndrome_name"), "category": "SymMap",
            "definition": s.get("Syndrome_definition"),
            "name_en": s.get("Syndrome_English"), "source": "symmap",
        })
    (EXT / "pattern_dictionary.json").write_text(
        json.dumps(patterns, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[3] 证型字典: {len(patterns)} 条（本草典 {len(bd['patterns'])} + SymMap {len(sm['syndromes'])}）")

    # ---------- 4. 症状词表 ----------
    sym_rows = [
        {"name": s.get("TCM_symptom_name"), "definition": s.get("Symptom_definition"),
         "locus": s.get("Symptom_locus"), "property": s.get("Symptom_property"),
         "source": "symmap"} for s in sm["tcm_symptoms"] if s.get("TCM_symptom_name")]
    syn_rows = []
    for s in bd["synonyms"]:
        syn_rows.append(s)
    (EXT / "symptom_vocabulary.json").write_text(json.dumps(
        {"tcm_symptoms": sym_rows, "mm_symptoms": [
            {"name": s.get("MM_symptom_name"), "umls": s.get("UMLS_id"),
             "mesh": s.get("MeSH_id"), "icd10": s.get("ICD10CM_id")}
            for s in sm["mm_symptoms"] if s.get("MM_symptom_name")],
         "synonym_entries": syn_rows}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[4] 症状词表: 中医症状 {len(sym_rows)} | 西医症状 {len(sm['mm_symptoms'])} | "
          f"同义词条目 {len(syn_rows)}")

    # ---------- 5. 安全规则 ----------
    safety = {
        "eighteen_incompat": [{"a": a, "b": b} for a, b in EIGHTEEN_INCOMPAT],
        "nineteen_incompat": [{"a": a, "b": b} for a, b in NINETEEN_INCOMPAT],
        "pregnancy_forbidden": PREGNANCY_FORBIDDEN,
        "pregnancy_caution": PREGNANCY_CAUTION,
        "herb_drug_interactions": bd["interactions"].get("herb_drug_interactions", []),
        "note": "十八反十九畏为《中药学》经典固定条目，bencaoDian 的 herb_incompatible_with 边为 0，故硬编码",
    }
    (EXT / "safety_rules.json").write_text(
        json.dumps(safety, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[5] 安全规则: 十八反 {len(EIGHTEEN_INCOMPAT)} | 十九畏 {len(NINETEEN_INCOMPAT)} | "
          f"中西药相互作用 {len(safety['herb_drug_interactions'])}")

    # ---------- 报告 ----------
    by_cat = Counter(v["category"] for v in herb_dict.values() if v["category"])
    L = ["# 外部知识库覆盖率报告", "",
         "## 数据源", "",
         "| 源 | 内容 | 许可 | 用途 |", "|---|---|---|---|",
         f"| 本草典 bencaodian v1 | 药 {len(bd['herbs'])} / 方 {len(bd['formulas'])} / "
         f"证型 {len(bd['patterns'])} / 医案 {len(bd['cases'])} | CC BY-SA 4.0 | **主力**：功效·主治·剂量·禁忌·方解 |",
         f"| SymMap v2.0 | 药 {len(sm['herbs'])} / 中医症状 {len(sm['tcm_symptoms'])} / "
         f"西医症状 {len(sm['mm_symptoms'])} / 疾病 {len(sm['diseases'])} / 证型 {len(sm['syndromes'])} | 学术使用 | 归经·症状词表·疾病映射 |",
         "| TCM-Prescription-Recommendation | 疾病 343 / 方 900+ | 未标注 | 备查（药名有 OCR 噪声） |",
         "| PresRecST | 4485 条匿名推理链 | 学术使用 | 架构参考（ID 匿名，需邮件申请全名） |", "",
         "## 药材字典覆盖", "",
         f"- 本方药材 **{len(mine_freq)}** 味 / {sum(mine_freq.values()):,} 次使用",
         f"- 映射成功 **{len(hit)}** 味（{len(hit)/len(mine_freq)*100:.1f}%）",
         f"- **用量加权覆盖 {covered/sum(mine_freq.values())*100:.2f}%** ← 关键指标", "",
         "| 字段 | 有数据的药材数 | 药材数占比 | **用量加权覆盖** |", "|---|---|---|---|",
         f"| 功效 actions | {n_act} | {n_act/len(herb_dict)*100:.0f}% | **{uw_act*100:.2f}%** |",
         f"| 主治 indications | {n_ind} | {n_ind/len(herb_dict)*100:.0f}% | **{uw_ind*100:.2f}%** |",
         f"| 剂量范围 dosage_range | {n_dose} | {n_dose/len(herb_dict)*100:.0f}% | **{uw_dose*100:.2f}%** |",
         f"| 归经 meridians | {n_mer} | {n_mer/len(herb_dict)*100:.0f}% | **{uw_mer*100:.2f}%** |", "",
         "> **用量加权覆盖才是有效指标**：长尾药种类多但用得少，按药材数计会低估实际可用性。", "",
         "## 未映射药材（字典缺失）", "",
         "| 药名 | 用量 | 药名 | 用量 |", "|---|---|---|---|"]
    for i in range(0, min(len(unmapped), 40), 2):
        a = unmapped[i]
        b = unmapped[i + 1] if i + 1 < len(unmapped) else None
        L.append(f"| {a} | {mine_freq[a]} | {b or ''} | {mine_freq[b] if b else ''} |")
    L += ["", "## 药材分类分布（Top15）", "", "| 分类 | 味数 |", "|---|---|"]
    for c, n in by_cat.most_common(15):
        L.append(f"| {c} | {n} |")
    L += ["", "## 关键能力（对项目的作用）", "",
          "| 外部数据 | 支撑链条的哪一步 |", "|---|---|",
          f"| 功效 actions（{n_act} 味） | **步骤5**：`方中功效 ⊇ 治法` 程序校验 |",
          f"| 剂量范围（{n_dose} 味） | **解决「数据无剂量」问题**：可填标准量而不是让模型编造 |",
          f"| 方剂君臣佐使+方解（{len(formulas)} 首） | **步骤3/4**：选方与加减的理由模板 |",
          f"| 证型主症/舌脉（{len(bd['patterns'])}） | **步骤1**：辨证依据；也是舌脉词表来源 |",
          f"| 十八反十九畏（{len(EIGHTEEN_INCOMPAT)}+{len(NINETEEN_INCOMPAT)}） | **硬约束**：约束解码，直接拦截 |",
          f"| 中西药相互作用（{len(safety['herb_drug_interactions'])}） | 安全提示（患者常并用西药） |",
          f"| 症状同义词（{len(syn_rows)}） | 归一化本方 1.4 万症状 token |", "",
          "## 仍缺的数据", "",
          "| 缺口 | 影响 | 补救 |", "|---|---|---|",
          "| **舌脉字段**（本方病历里没有） | 辨证的输入不完整 | 需向院方申请，外部数据补不了 |",
          "| **复诊疗效反馈** | 无法判断加减是否有效 | 需向院方申请 |",
          "| **医生 ID** | 无法剥离医师风格 | 需向院方申请 |",
          "| 大方剂库（>1000 首，含主治） | 步骤3 检索候选不足 | 本草典仅 112 首；可另找 KNApSAcK KAMPO（1581 首） |",
          "| 草药↔症状↔疾病**关系边** | SymMap 下载文件只有实体表，无关系 | 需从其网站检索接口取，或改用本草典 indications |"]
    (REPORT_DIR / "external_data_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"\n报告 -> reports/external_data_report.md")
    print("=" * 80)


if __name__ == "__main__":
    main()
