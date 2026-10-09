#!/bin/bash
set -e
# Stage 0 全流程（可复现，确定性输出）
#
#   1 数据工程            清洗/归一化/纵向关联/患者分组切分 → data/*.jsonl
#   2 规则基线            可实现基线 + Oracle 上界 + 既往处方基线 → reports/baseline_report.md
#   3 泄漏消融            患者分组 vs 按行随机 → reports/leakage_report.md
#   4 症状→药 关联挖掘     症状信息量 + 关联 + 混杂护栏 → reports/symptom_herb_report.md
#   5 处方成因归因         累积消融 + 医师风格检测 → reports/attribution_report.md
#   6 推理链架构           五步链条可行性 / 检索天花板 → reports/reasoning_chain_report.md
#   7 外部知识库           药材字典/方剂库/安全规则/治法对齐 → data/external/, reports/*
#   8 约束校验器           功效/安全校验，校准与验证 → reports/validator_report.md
#   9 B组离线件           约束解码自检 / 训练脚本 dry-run / 剂量填充 / 界面 smoke（均无需 GPU）
#  10 统一评测            端到端 R0，含 Gate 判定 → reports/stage0_baseline.json
#  11 一致性检查          叙述文档数字 vs 生成报告，不通过则退出码非 0
#
# 注：4/5/6 计算量较大（交叉验证 + 置换检验 + 检索），全流程约 13-19 分钟。
cd "$(dirname "$0")"

echo "=== 1/11 数据工程（清洗/纵向关联/切分/jsonl） ==="
python3 src/prepare_data.py

echo; echo "=== 2/11 规则基线 + Oracle 上界 + 既往处方基线 ==="
python3 src/baseline.py

echo; echo "=== 3/11 泄漏消融（患者分组 vs 按行随机） ==="
python3 src/leakage_check.py

echo; echo "=== 4/11 症状→药 关联挖掘 + 症状信息量 ==="
python3 src/mine_symptom_herb.py

echo; echo "=== 5/11 处方成因归因（累积消融 + 医师风格检测） ==="
python3 src/attribution.py

echo; echo "=== 6/11 推理链架构可行性（检索天花板/加减幅度） ==="
python3 src/reasoning_chain.py

echo; echo "=== 7/11 外部知识库整合 + 治法→功效 对齐表 ==="
python3 src/build_external_data.py
python3 src/build_zhi_action_map.py

echo; echo "=== 8/11 处方功效/安全约束校验器（校准 + 合成负样本） ==="
python3 src/validate_rx.py

echo; echo "=== 9/11 B组离线件（约束解码 / 训练 dry-run / 剂量 / 界面 smoke） ==="
python3 src/constrained_decode.py --selftest
python3 src/train_sft.py --dry-run
python3 src/dose_filler.py --demo | tail -4
python3 src/app.py --smoke

echo; echo "=== 10/11 统一评测（端到端 R0，含 Gate 判定） ==="
python3 src/evaluate.py \
  --ref data/ref_test.jsonl           --pred data/preds/R0_端到端规则.jsonl \
  --ref-time data/ref_test_time.jsonl --pred-time data/preds/R0_端到端规则_time.jsonl \
  --json-out reports/stage0_baseline.json

echo; echo "=== 11/11 数字一致性检查（叙述文档 vs 生成报告） ==="
python3 src/check_consistency.py

echo; echo "Stage 0 完成。报告见 reports/"
