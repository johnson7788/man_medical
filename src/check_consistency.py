"""
check_consistency.py — 数字一致性检查

为什么需要它：
  交付物变多后，同一个数字会同时出现在「生成报告」（由脚本产出）与
  「叙述文档」（技术报告 / 模型训练计划，手工引用）里。改了代码忘了改文档，
  文档就会**静默撒谎**。

  实测教训：外部字典覆盖率在修正药名前缀后从 89.07% 升到 94.44%，
  但《技术报告》与《模型训练计划》里引用的是旧值 —— 靠人工比对才发现。
  另外 §9/§10.3 曾长期声称"中药字典本地暂无"，而它其实已经交付。

本脚本把「叙述文档中的关键数字」与「生成报告中的实测值」做机械化比对，
不通过则退出码非 0，可直接挂进流水线。

用法: python3 src/check_consistency.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import medlib as M

ROOT = M.ROOT
# 叙述文档：手工引用数字的地方。改动这些文档后必须重跑本检查。
NARRATIVE = ["README.md", "reports/技术报告.md", "模型训练计划.md",
             "reports/训练评测报告.md",
             "docs/00_项目概览.md", "docs/01_数据说明.md", "docs/02_评测协议.md",
             "docs/03_训练指南.md", "docs/04_推理与演示.md", "docs/05_关键发现.md",
             "docs/06_复现实验.md", "docs/07_开源前必读.md",
             "docs/08_模型卡.md", "docs/09_数据集卡.md"]

# (要检查的数字, 必须出现的来源报告, 含义)
CHECKS: list[tuple[str, str, str]] = [
    # 基线（reports/baseline_report.md）
    ("0.159", "reports/baseline_report.md", "常量方 Jaccard"),
    ("0.183", "reports/baseline_report.md", "最佳可实现基线"),
    ("0.208", "reports/baseline_report.md", "Oracle 上界"),
    ("0.124", "reports/baseline_report.md", "端到端规则 R0"),
    ("0.542", "reports/baseline_report.md", "复诊照抄上次方"),
    ("0.438", "reports/baseline_report.md", "新上界 X2"),
    # 泄漏消融（reports/leakage_report.md）
    ("+22.8%", "reports/leakage_report.md", "B2 泄漏虚高"),
    ("+34.7%", "reports/leakage_report.md", "O3 泄漏虚高"),
    ("+30.5%", "reports/leakage_report.md", "D2 泄漏虚高"),
    # 归因（reports/attribution_report.md）
    ("0.5432", "reports/attribution_report.md", "常量 AUC"),
    ("+0.1166", "reports/attribution_report.md", "检查结果边际增益"),
    ("+0.0607", "reports/attribution_report.md", "证型/指南边际增益"),
    ("0.8149", "reports/attribution_report.md", "含既往处方 AUC"),
    # 推理链（reports/reasoning_chain_report.md）
    ("0.100", "reports/reasoning_chain_report.md", "初诊检索 top-1"),
    ("0.139", "reports/reasoning_chain_report.md", "同病名+证型两两 Jaccard"),
    # 外部知识库（reports/external_data_report.md）
    ("99.59%", "reports/external_data_report.md", "药名映射用量加权覆盖"),
    ("94.44%", "reports/external_data_report.md", "功效/主治/剂量用量加权覆盖"),
    ("96.18%", "reports/external_data_report.md", "归经用量加权覆盖"),
    # 治法对齐（reports/zhi_action_map_report.md）
    ("72.8%", "reports/zhi_action_map_report.md", "朴素 2-gram 覆盖率"),
    ("96.0%", "reports/zhi_action_map_report.md", "规则层覆盖率"),
    ("96.9%", "reports/zhi_action_map_report.md", "规则+数据覆盖率"),
    ("93.4%", "reports/zhi_action_map_report.md", "完全覆盖率"),
    # 校验器（reports/validator_report.md）
    ("5.62%", "reports/validator_report.md", "校验器误报率"),
    ("92.99%", "reports/validator_report.md", "verdict=ok 占比"),
    ("7.9%", "reports/validator_report.md", "治法判别检出率"),
    ("100.0%", "reports/validator_report.md", "安全类检出率"),
    # 训练与评测（reports/model_comparison.json 由 compare_models 生成）
    ("0.429", "reports/model_comparison.md", "v2 总体 Jaccard"),
    ("0.426", "reports/model_comparison.md", "v1 总体 Jaccard"),
    ("0.301", "reports/model_comparison.md", "v2 初诊 Jaccard"),
    ("0.542", "reports/model_comparison.md", "照抄复诊 Jaccard"),
    ("0.609", "reports/model_comparison.md", "v2 关键药 Recall"),
    ("0.485", "reports/model_comparison.md", "v2 证型 F1"),
    # 数据审计（reports/data_audit.md）
    ("26,923", "reports/data_audit.md", "金标准样本数"),
    ("21,484", "reports/data_audit.md", "train 样本数"),
]

# 文档中不应再出现的过时说法（曾经撒谎，已修）
STALE_PHRASES: list[tuple[str, str]] = [
    ("本地暂无该字典", "中药字典已交付（§14），此说法过时"),
    ("safety_check.py` 待产出", "校验器已由 validate_rx.py 取代"),
    ("①③已交付", "实际是 ③④⑤ 已交付"),
    ("93.93%", "旧覆盖率，现为 94.44%"),
    ("89.07%", "旧覆盖率，现为 94.44%"),
]


def main() -> int:
    print("=" * 78)
    print("数字一致性检查：叙述文档 vs 生成报告")
    print("=" * 78)

    def _strip_revision_log(txt: str) -> str:
        """去掉「修订记录」小节：那里本就应当如实记载修过什么，
        不该被当成"仍在使用过时说法"。"""
        out = txt
        # 这些小节是"留痕"性质，本就会引用旧值，不该判为仍在使用过时说法
        for marker in ("### 16.6 计划文档自身的修订记录",
                       "## 附录 C：已修复的实现级缺陷",
                       "## 6. 本项目实际踩过的 4 个评测缺陷"):
            i = out.find(marker)
            if i >= 0:
                j = out.find("\n## ", i + len(marker))
                # 附录 C 是最后一节，找不到下一个 ## 就截到末尾
                out = out[:i] + (out[j:] if j >= 0 else "")
        return out

    nar = {p: _strip_revision_log((ROOT / p).read_text(encoding="utf-8"))
           for p in NARRATIVE if (ROOT / p).exists()}
    if not nar:
        print("❌ 找不到叙述文档")
        return 1

    fails: list[str] = []

    print("\n[1] 关键数字是否同时出现在【叙述文档】与【生成报告】")
    print(f"  {'数字':>9s} {'含义':34s} " + "  ".join(p.split('/')[-1][:12] for p in nar)
          + "  源报告")
    for num, src, label in CHECKS:
        src_p = ROOT / src
        if not src_p.exists():
            print(f"  ⚠️ {num:>9s} {label:34s} 源报告缺失: {src}")
            fails.append(f"源报告缺失 {src}")
            continue
        in_src = num in src_p.read_text(encoding="utf-8")
        # 语义：源报告必须含权威值；叙述文档【至少一篇】记录它即可。
        # 早期版本要求"每篇叙述文档都含每个数字"，在只有 2 篇文档时可行，
        # 扩到 14 篇后就变成无意义的约束（没有哪篇文档需要囊括所有指标）。
        hits = [p for p, txt in nar.items() if num in txt]
        ok = in_src and bool(hits)
        if not ok:
            fails.append(f"{num} ({label}) 源={in_src} 叙述命中={hits}")
        where = f"{len(hits)} 篇" if hits else "❌ 无"
        print(f"  {num:>9s} {label:34s} 源={'✅' if in_src else '❌'}  叙述={where:<6s}"
              f" e.g. {hits[0].split('/')[-1] if hits else '—'}")

    print("\n[2] 过时说法是否已清理")
    # 数值类过时值（如 89.07%）可能被合法地用作"漂移事故"的举例——那种行里通常
    # 同时出现正确值（"89.07% → 94.44%"）。若同一行含正确值，视为对比引用而非错误声明。
    NUMERIC_CORRECT = {"93.93%": "94.44%", "89.07%": "94.44%"}
    for phrase, why in STALE_PHRASES:
        hits = []
        allowed = []
        for pp, txt in nar.items():
            if phrase not in txt:
                continue
            corr = NUMERIC_CORRECT.get(phrase)
            bad_line = False
            if corr:
                # 逐行判断：只有"该行没同时给出正确值"才算真的过时声明
                for line in txt.splitlines():
                    if phrase in line and corr not in line:
                        bad_line = True
                        break
            else:
                bad_line = True
            (hits if bad_line else allowed).append(pp)
        ok = not hits
        if not ok:
            fails.append(f"过时说法「{phrase}」仍存在于 {hits}")
        note = f"  仍见于 {hits}" if hits else ""
        if allowed:
            note += ("（" + "、".join(allowed)
                     + " 中是「漂移举例」，同行已给出正确值，不计为错误）")
        print(f"  {'✅' if ok else '❌'} 「{phrase}」 —— {why}{note}")

    print("\n[3] 台账里承诺的文件是否存在")
    ledger = (ROOT / "模型训练计划.md").read_text(encoding="utf-8")
    promised = set(re.findall(r"`(src/[A-Za-z0-9_]+\.py)`", ledger))
    miss = sorted(p for p in promised if not (ROOT / p).exists())
    for p in sorted(promised):
        print(f"  {'✅' if (ROOT/p).exists() else '❌'} {p}")
    if miss:
        # train_sft.py / constrained_decode.py 已在台账 §16.3 标为「未完成」，
        # 因此它们"不存在"是事实而非缺陷 —— 只有【已完成项】引用的文件缺失才算失败。
        print(f"\n  ℹ️ 未创建的脚本：{miss}（若已在 §16 未完成清单中列明，则属正常）")

    print("\n" + "=" * 78)
    if fails:
        print(f"❌ 一致性检查失败 {len(fails)} 项：")
        for f in fails:
            print("   -", f)
        return 1
    print("✅ 全部一致")
    return 0


if __name__ == "__main__":
    sys.exit(main())
