"""
dose_filler.py — 剂量填充层（B3）

解决的问题（计划 §6.2 必做约束 1）：
  原始病历的剂量字段**全部是 1.000**（41 万条，无信息量），所以模型**不能生成剂量** ——
  编造剂量 = 直接临床风险。正确做法是：模型只输出药味，展示层从权威剂量表填充，
  并强制标注「需医师核定」。

  现在 `data/external/herb_dictionary.jsonl` 已提供 dosage_range（355/462 味，用量加权 94.44%），
  本模块就是那个"展示层填充器"。

设计要点：
  1. **默认量取偏保守**：有毒药材取区间下限；其余取区间中位（四舍五入到整数克）。
     理由：系统是辅助而非开方，宁少勿多。
  2. **永远返回区间与原文说明**，不只给一个数字 —— 字典的 notes 里含炮制差异
     （如柴胡「和解退热宜生用 6–10g；疏肝解郁宜醋炙 3–9g；升阳举陷 3–6g」），
     这个信息机器无法替医生决定，必须原样呈现。
  3. **字典缺失显式标注**，绝不留空或猜一个数。
  4. **强制附加免责声明**，且 `needs_review` 恒为 True。

用法：
  from dose_filler import DoseFiller
  df = DoseFiller()
  items = df.fill(["北柴胡", "黑顺片", "茯苓"])
  print(df.render(items))
  python3 src/dose_filler.py --demo
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import medlib as M
from medlib import DATA_DIR

EXT = M.ROOT / "data" / "external"

DISCLAIMER = ("⚠️ 剂量为《药典》/中药学标准量的机械填充，**仅供参考，须由执业医师核定**。"
              "系统不生成剂量，也不构成处方建议。")

# 有毒/需特殊处理药材：默认量取区间下限
TOXIC_HINT = ("有毒", "大毒", "小毒", "先煎", "久煎", "不宜久服", "严格控制")


class DoseFiller:
    def __init__(self, herb_dict_path: Path | None = None,
                 config_path: Path | None = None):
        self.hd = {r["herb"]: r for r in
                   M.read_jsonl(herb_dict_path or EXT / "herb_dictionary.jsonl")}
        cfg = json.loads((config_path or EXT / "validator_config.json").read_text(encoding="utf-8"))
        self.toxic_set = set(cfg["toxic_handling"]["herbs"])
        self.absolute_cap = {k: v for k, v in cfg["dose"]["absolute_cap_g"].items()
                             if not k.startswith("_")}

    # -- 单味药 -----------------------------------------------------------
    def is_toxic(self, herb: str) -> bool:
        v = self.hd.get(herb, {})
        std = v.get("standard_name") or herb
        if herb in self.toxic_set or std in self.toxic_set:
            return True
        txt = (v.get("safety_notes_zh") or "") + (v.get("contraindications_zh") or "")
        return any(k in txt for k in TOXIC_HINT)

    def dose_for(self, herb: str) -> dict:
        """返回该味药的剂量建议（含区间、依据、是否需人工复核）。"""
        v = self.hd.get(herb)
        if not v:
            return {"herb": herb, "dose_g": None, "range": None, "unit": "g",
                    "is_toxic": False, "needs_review": True,
                    "reason": "药名不在字典（462 味白名单）内，需人工确认"}
        dr = v.get("dosage_range")
        if not dr or dr.get("min") is None or dr.get("max") is None:
            return {"herb": herb, "dose_g": None, "range": None,
                    "unit": (dr or {}).get("unit", "g"),
                    "is_toxic": self.is_toxic(herb), "needs_review": True,
                    "reason": "该药在本草典字典中无剂量数据，需人工填写"}
        lo, hi = float(dr["min"]), float(dr["max"])
        unit = dr.get("unit") or "g"
        tox = self.is_toxic(herb)
        if tox:
            chosen, why = lo, "有毒/需特殊处理药材，取区间下限（宁少勿多）"
        else:
            chosen = float(round((lo + hi) / 2))
            chosen = min(max(chosen, lo), hi)
            why = "常规药材，取区间中位"
        cap = self.absolute_cap.get(herb) or self.absolute_cap.get(v.get("standard_name") or "")
        capped = False
        if cap and chosen > cap:
            chosen, capped = float(cap), True
            why = f"已按绝对上限 {cap}{unit} 收敛"
        return {
            "herb": herb,
            "standard_name": v.get("standard_name"),
            "dose_g": chosen, "range": [lo, hi], "unit": unit,
            "is_toxic": tox, "capped": capped,
            "needs_review": True,
            "reason": why,
            "notes": dr.get("notes") or "",
            "safety_notes": (v.get("safety_notes_zh") or "")[:200],
        }

    # -- 整方 -------------------------------------------------------------
    def fill(self, herbs: list[str]) -> dict:
        seen, items, unknown = set(), [], []
        for h in herbs:
            if h in seen:
                continue
            seen.add(h)
            it = self.dose_for(h)
            items.append(it)
            if it["dose_g"] is None:
                unknown.append(h)
        filled = [i for i in items if i["dose_g"] is not None]
        return {
            "items": items,
            "n_herbs": len(items),
            "n_filled": len(filled),
            "n_unknown": len(unknown),
            "unknown_herbs": unknown,
            "total_g": round(sum(i["dose_g"] for i in filled), 1),
            "has_toxic": any(i["is_toxic"] for i in items),
            "disclaimer": DISCLAIMER,
        }

    # -- 呈现 -------------------------------------------------------------
    def render(self, filled: dict, show_notes: bool = True) -> str:
        lines = []
        for i in filled["items"]:
            if i["dose_g"] is None:
                lines.append(f"  {i['herb']}：**待人工填写** —— {i['reason']}")
                continue
            flag = " ⚠️有毒/需特殊处理" if i["is_toxic"] else ""
            rng = f"{i['range'][0]:g}–{i['range'][1]:g}{i['unit']}"
            lines.append(f"  {i['herb']} {i['dose_g']:g}{i['unit']}"
                         f"（字典区间 {rng}，{i['reason']}）{flag}")
            if show_notes and i.get("notes"):
                lines.append(f"      ↳ {i['notes'][:160]}")
        out = ["【中药处方（剂量由标准量表填充）】"] + lines
        out.append(f"  — 共 {filled['n_herbs']} 味，填充 {filled['n_filled']} 味，"
                   f"合计 {filled['total_g']}g"
                   + (f"，{filled['n_unknown']} 味缺剂量数据" if filled["n_unknown"] else ""))
        out.append(f"\n{DISCLAIMER}")
        return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--herbs", type=str, default="", help="逗号分隔的药名")
    args = ap.parse_args()

    df = DoseFiller()
    if args.herbs:
        herbs = [h.strip() for h in args.herbs.split(",") if h.strip()]
    else:
        herbs = ["北柴胡", "黑顺片", "细辛", "茯苓", "甘草", "炙甘草",
                 "炒蒺藜", "醋莪术", "完全不存在的药"]

    filled = df.fill(herbs)
    print(df.render(filled))

    if args.demo:
        print("\n" + "=" * 78)
        print("覆盖率统计（按真实处方用药频次加权）")
        print("=" * 78)
        allr = M.read_jsonl(DATA_DIR / "all_records.jsonl")
        from collections import Counter
        freq = Counter(h for r in allr if r["has_rx"] for h in r["rx"])
        tot = sum(freq.values())
        ok = sum(c for h, c in freq.items() if df.dose_for(h)["dose_g"] is not None)
        tox = sum(c for h, c in freq.items() if df.is_toxic(h))
        print(f"  可填充剂量的用量占比: {ok/tot*100:.2f}%")
        print(f"  判为有毒/需特殊处理的用量占比: {tox/tot*100:.2f}%")
        print(f"  缺剂量数据的药材: {sum(1 for h in freq if df.dose_for(h)['dose_g'] is None)} 味"
              f"（用量占 {100-ok/tot*100:.2f}%）")


if __name__ == "__main__":
    main()
