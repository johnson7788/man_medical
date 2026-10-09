# 中医男科病历 · 辨证论治模型

> 用真实中医男科门诊病历训练「四诊信息 → 中医诊断 + 中药处方」的辅助模型。
> 包含完整的数据工程、**评测协议**、规则基线、约束校验器，以及 MiMo-9B 的 LoRA 微调。

<p>
<img alt="python" src="https://img.shields.io/badge/python-3.10%2B-blue">
<img alt="code" src="https://img.shields.io/badge/code-Apache--2.0-green">
<img alt="data" src="https://img.shields.io/badge/data-CC%20BY--NC%204.0-orange">
<img alt="model" src="https://img.shields.io/badge/model-MiMo--9B%20LoRA-purple">
</p>

---

## 这个项目最值得看的三件事

**1. 一个诚实的评测协议。** 医疗 NLP 项目最容易自欺的地方是评测。本项目把评测收敛到
**单一入口、单一冻结测试集、单一指标实现**，并强制报告初诊/复诊分口径 + bootstrap 置信区间
+ 配对显著性检验。所有数字都能用一条命令复现。→ [`docs/02_评测协议.md`](docs/02_评测协议.md)

**2. 三个反直觉的数据结论。** 它们直接决定了模型该怎么设计：

| 结论 | 证据 |
|---|---|
| **处方不是证型决定的** | 即使把**真实**中医诊断喂给检索器（Oracle），处方 Jaccard 也只有 **0.208** |
| **处方是「治疗进程中的一次调整」** | 复诊占 63.9%，其最强预测因子是**上次处方**（照抄即 **0.542**），而非中医诊断 |
| **两阶段管线是陷阱** | 「先预测诊断→再检索处方」得 **0.124**，**低于什么都不做的常量方 0.159** |

→ [`docs/05_关键发现.md`](docs/05_关键发现.md)

**3. 一条能上线的安全链路。** 模型**不生成剂量**（原始数据无真剂量），
剂量由药典标准量表填充；配以功效/安全约束校验器（误报率 5.62%，安全类检出 100%）。
→ [`docs/04_推理与演示.md`](docs/04_推理与演示.md)

---

## 实测结果

模型：`MiMo-V2.6-Distill-Qwen-9B` + LoRA（可训练 43.3M / 9.45B = 0.458%）
测试集：2,764 条金标准样本，**按患者分组切分**（患者零重叠）

| 类别 | 方法 | 处方 Jaccard | 95% CI |
|---|---|---|---|
| **模型** | **MiMo-9B v2（2 epoch）** | **0.429** | [0.416, 0.441] |
| 模型 | MiMo-9B v1（3 epoch） | 0.426 | [0.413, 0.438] |
| 治疗进程 | 复诊照抄 + 初诊 Oracle | 0.438 | [0.425, 0.451] |
| Oracle | 证型+病名检索（**用真实诊断**） | 0.208 | [0.202, 0.214] |
| 可实现 | 西医诊断检索 | 0.183 | [0.178, 0.188] |
| 端到端 | 规则 R0 | 0.124 | [0.120, 0.129] |
| 下界 | 全局常量方 | 0.159 | [0.155, 0.163] |

**分口径**（初诊 997 / 复诊 1,767）：

| 来源 | 全部 | 初诊 | 复诊 |
|---|---|---|---|
| 模型 v2 | 0.429 | **0.301** | 0.501 |
| 模型 v1 | 0.426 | 0.279 | **0.509** |
| 照抄+初诊Oracle | 0.438 | 0.254 | **0.542** |
| 常量方 | 0.159 | 0.160 | 0.159 |

### 三条必须一起看的结论

✅ **初诊超过了 Oracle 上界**（0.301 vs 0.254）。Oracle 是"给定**真实**诊断去检索"的上界，
模型在不知道真实诊断的情况下端到端超过它 —— 这是它在读自由文本、而非查表的硬证据。

❌ **复诊打不过"照抄上次方"**（0.501 vs 0.542）。这是数据的结构性事实。
产品上应把「照抄 + 增删建议」直接作为复诊基线，模型只在初诊兜底。

❌ **证型 F1 未达 Gate**（0.485 / 0.505 vs 必达 0.55）。
但归因分析显示 **80% 的差距来自标签同义**（`肾虚` vs `肾虚证`、`肝肾亏虚证` vs `肝肾两虚证`）——
仅做标签归一化即可从 0.485 提到 0.538，不需改模型。剩余部分是**真实的虚实混淆**
（`肾气不充证` ↔ `湿热下注证`，213 次，治法是补 vs 清，方向相反）。

> v1 与 v2 总体**统计不可分**（配对 bootstrap 差值 −0.003，p=0.873），
> 但分口径显著不同：v2 初诊 **+0.022**（显著更好）、v1 复诊 **+0.008**（显著更好）。

---

## 快速开始

```bash
pip install -r requirements.txt

# 1) 数据工程 + 全部基线 + 消融 + 归因 + 校验器（约 15 分钟，仅需 CPU）
./run_stage0.sh
# 产物：data/*.jsonl、reports/*.md

# 2) 只看统一评测对比与 Gate 判定
python3 src/compare_models.py
```

> `run_stage0.sh` **不需要原始 Excel** —— 发布版从 `data/all_records.jsonl` 重跑。
> 原始 Excel 含院方真实数据，**不随仓库提供**（见 [`docs/07_开源前必读.md`](docs/07_开源前必读.md)）。

### 训练（需 NVIDIA GPU）

```bash
# 下载基座（18.8GB）
python3 src/download_model.py --out models/MiMo-V2.6-Distill-Qwen-9B

# 关键：先自检！确认 LoRA 目标模块名与本模型匹配
python3 src/train_sft.py --model models/MiMo-V2.6-Distill-Qwen-9B --inspect

# 训练（单卡 96GB 约 2h50m / 2 epoch）
python3 src/train_sft.py --model models/MiMo-V2.6-Distill-Qwen-9B \
    --train-file data/train_v2.jsonl --val-file data/val_v2.jsonl \
    --output-dir outputs/mimo9b-tcm-lora --epochs 2 --bs 8 --ga 2 --no-grad-ckpt

# 评测
python3 src/predict.py --model models/MiMo-V2.6-Distill-Qwen-9B \
    --adapter outputs/mimo9b-tcm-lora --ref data/ref_test.jsonl \
    --out data/preds/mine.jsonl
python3 src/compare_models.py --model mine=data/preds/mine.jsonl
```

### 演示界面

```bash
python3 src/app_model.py --base models/MiMo-V2.6-Distill-Qwen-9B \
    --adapter outputs/mimo9b-tcm-lora --port 7860
```

两个页签：① 模型推理（四诊 → 诊断 + 处方，复诊额外显示**相对上次方的增删**）
② 处方校验器（手工粘贴处方做安全检查，不需要模型）。

---

## 目录结构

```
├── src/
│   ├── medlib.py                 共用库：解析/归一化/切分/指标口径（单一来源）
│   ├── prepare_data.py           数据工程：清洗、纵向关联、患者分组切分
│   ├── baseline.py               规则基线 B1-B4 / Oracle O1-O5 / 治疗进程 X1-X2
│   ├── evaluate.py               ★ 评测唯一入口（模型与基线共用同一套指标）
│   ├── compare_models.py         ★ 统一对比 + bootstrap CI + 配对检验 + Gate
│   ├── leakage_check.py          泄漏消融（患者分组 vs 按行随机）
│   ├── mine_symptom_herb.py      症状→药 关联挖掘（含混杂护栏）
│   ├── attribution.py            处方成因归因（累积消融 + 医师风格检验）
│   ├── reasoning_chain.py        五步推理链可行性（检索天花板）
│   ├── build_external_data.py    外部知识库整合
│   ├── build_zhi_action_map.py   治法→功效对齐表
│   ├── validate_rx.py            ★ 功效/安全约束校验器
│   ├── dose_filler.py            剂量填充层（模型不生成剂量）
│   ├── constrained_decode.py     白名单/十八反约束解码
│   ├── train_sft.py              bf16 LoRA 训练（含 --inspect 模块名探测）
│   ├── predict.py                冻结测试集批量推理
│   ├── app.py / app_model.py     Gradio 演示
│   ├── sanitize_for_release.py   开源前脱敏（医院名归一化）
│   └── check_consistency.py      文档数字与生成报告的一致性检查
├── data/                         派生数据（CC BY-NC 4.0）
│   └── external/                 外部知识库（各源许可，见其 LICENSE）
├── docs/                         详细文档（建议按序号读）
├── reports/                      全部实验报告（由脚本生成，非手写）
└── run_stage0.sh                 一键复现全流程
```

---

## 数据说明（摘要）

| 项 | 数值 |
|---|---|
| 就诊记录 | 67,073 条 / 25,103 名患者 |
| 时间跨度 | 2023-01-01 ~ 2026-09-08 |
| **金标准 T3 样本** | **26,923 条**（四诊 + 中医诊断 + 处方齐全） |
| 切分（患者分组） | train 21,484 / val 2,675 / test 2,764，**患者零重叠** |
| 药材 | 462 味（原始 538 归一化） |
| 诊断标签 | 病名 263 / 证型 499 / 治法 320 |

**已做的隐私处理**（见 [`docs/07_开源前必读.md`](docs/07_开源前必读.md)）：

- 患者 ID → 哈希（`P` + 12 位 hex，25,103 个，无原始病历号残留）
- 机构名 → 归一化为「外院」（234 个院名，1,716 条记录）
- PHI 扫描：手机号 / 身份证 / 电话 / 社交账号 **命中 0**

⚠️ **未做**：伦理审查（IRB）与患者知情同意；舌象/脉象字段缺失（舌 0.53%、脉 0.74%）。

---

## 明确的能力边界

以下每一项都有实测依据，请在使用前确认你的场景不依赖它们：

| 做不到 | 证据 |
|---|---|
| **不能替代医师审方** | 校验器只做**缺失检测**，治法判别力仅 **7.9%** |
| **不能解释"为什么开这个方"** | 无疗效反馈数据，模型只能模仿 |
| **舌脉辨证** | 数据中舌象 0.53% / 脉象 0.74%，存在原理性输入缺口 |
| **不建议用于复诊** | 不如直接照抄上次方（0.501 vs 0.542） |
| **剂量不可信** | 剂量非模型生成，是药典标准量表机械填充 |

> 本数据集支持的是**「处方推荐 + 可审核的相对调整说明」**，**不支持「处方解释」**。

---

## 许可

**三者许可不同，请分别遵守：**

| 范围 | 许可 | 文件 |
|---|---|---|
| `src/` 源代码 | Apache-2.0 | [`LICENSE`](LICENSE) |
| `data/` 自行派生数据 | **CC BY-NC 4.0（禁止商用）** | [`LICENSE-DATA`](LICENSE-DATA) |
| `data/external/` 外部知识库 | 各源许可（本草典为 **CC BY-SA 4.0**） | [`data/external/LICENSE`](data/external/LICENSE) |

> ⚠️ CC BY-SA 4.0 的 ShareAlike 与 CC BY-NC 4.0 **不兼容**，故两者分开存放、分别许可。

---

## 引用

```bibtex
@misc{tcm_andrology_2026,
  title  = {中医男科病历 · 辨证论治模型},
  year   = {2026},
  note   = {数据 CC BY-NC 4.0，代码 Apache-2.0},
  url    = {https://github.com/your-org/man_medical}
}
```

**免责声明**：本项目为研究与教学用途，**不构成医疗建议**，
不得用于临床诊断、处方决策或任何直接影响患者诊疗的用途。
