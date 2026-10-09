# 中医男科病历 · 辨证论治模型

用真实中医男科门诊病历训练的「四诊信息 → 中医诊断 + 中药处方」辅助模型。

<p>
<img alt="python" src="https://img.shields.io/badge/python-3.10%2B-blue">
<img alt="code" src="https://img.shields.io/badge/code-Apache--2.0-green">
<img alt="data" src="https://img.shields.io/badge/data-CC%20BY--NC%204.0-orange">
<img alt="model" src="https://img.shields.io/badge/model-MiMo--9B%20LoRA-purple">
</p>

---

## 数据

67,073 条门诊记录，来自 25,103 名患者，时间跨度 2023-01-01 至 2026-09-08。
其中 26,923 条为四诊、中医诊断、处方三项齐全的样本，按患者分组切成
train 21,484 / val 2,675 / test 2,764，患者之间零重叠。

处方共出现 462 味药材，诊断标签为病名 263 种、证型 499 种、治法 320 种。

原始 Excel 含院方真实数据，不随仓库提供；发布版从 `data/all_records.jsonl` 复现。
患者 ID 已哈希为 `P` + 12 位十六进制，机构名已统一替换为「外院」，
手机号、身份证号、电话、社交账号的扫描命中均为 0。伦理审查与患者知情同意**未做**，
细节见 [docs/07_开源前必读.md](docs/07_开源前必读.md)。

## 结果

模型为 `MiMo-V2.6-Distill-Qwen-9B` 加 LoRA，可训练参数 43.3M，占 0.458%。
在 2,764 条冻结测试集上：

| 方法 | 处方 Jaccard | 95% CI |
|---|---|---|
| MiMo-9B v2（2 epoch） | 0.429 | [0.416, 0.441] |
| MiMo-9B v1（3 epoch） | 0.426 | [0.413, 0.438] |
| 复诊照抄 + 初诊 Oracle | 0.438 | [0.425, 0.451] |
| 证型+病名检索（Oracle，用真实诊断） | 0.208 | [0.202, 0.214] |
| 西医诊断检索 | 0.183 | [0.178, 0.188] |
| 端到端规则 R0 | 0.124 | [0.120, 0.129] |
| 全局常量方 | 0.159 | [0.155, 0.163] |

按初诊（997 条）与复诊（1,767 条）分开看：

| 方法 | 全部 | 初诊 | 复诊 |
|---|---|---|---|
| MiMo-9B v2 | 0.429 | 0.301 | 0.501 |
| MiMo-9B v1 | 0.426 | 0.279 | 0.509 |
| 复诊照抄 + 初诊 Oracle | 0.438 | 0.254 | 0.542 |
| 全局常量方 | 0.159 | 0.160 | 0.159 |

Oracle 指用真实中医诊断去检索方剂所能达到的上界，实际部署时拿不到真实诊断，
它只用于标定检索方案的天花板。模型在初诊上得到 0.301，高于这个上界的初诊口径 0.254。

复诊上模型不如直接照抄上次处方（0.501 对 0.542）。这不是训练不充分，而是数据的结构性事实：
复诊处方里超过一半的内容就是上次的方子。实际使用时，复诊更适合直接用「照抄 + 增删提示」，
模型只在初诊兜底。

证型 micro-F1 为 0.485，低于设定的 0.55。其中约八成的差距来自标注同义问题——
参考标签里同时存在 `肾虚证`（906 次）与 `肾虚`（163 次）、`肝肾亏虚证`（428 次）与
`肝肾两虚证`（289 次），模型只能二选一。只做标签同义合并（不改模型）即可把 F1 提到 0.538。
剩下的部分是真实的证型混淆，最突出的是 `肾气不充证`（虚）与 `湿热下注证`（实）之间的误判，
共 213 次，两者治法方向相反。

v1 与 v2 在总体上统计不可分（配对 bootstrap 差值 −0.003，p=0.873），
但分口径有差异：v2 初诊高 0.022，v1 复诊高 0.008，两者均显著。

更详细的实验结论见 [docs/05_关键发现.md](docs/05_关键发现.md)，
评测口径与已知局限见 [docs/02_评测协议.md](docs/02_评测协议.md)。

## 适用范围与限制

| 限制 | 依据 |
|---|---|
| 不能替代医师审方 | 校验器只做缺失检测（方中有没有药支撑该治法），治法判别力实测 7.9% |
| 不能解释为什么开这个方 | 训练数据没有疗效反馈，模型只能模仿医生的处方 |
| 不适用于舌脉辨证 | 数据中舌象占 0.53%、脉象占 0.74%，输入本身不完整 |
| 不建议用于复诊 | 不如直接照抄上次处方 |
| 剂量不可信 | 模型不生成剂量，剂量由药典标准量表机械填充 |

原始数据中 99.7% 的剂量字段是常量 `1.000`，没有信息量，训练时已整体丢弃。
因此模型输出不含剂量，剂量由 `src/dose_filler.py` 按药典剂量区间填充，
有毒药材取区间下限，并标注「需医师核定」。

这套数据能支撑的是「处方推荐加上可审核的相对调整说明」，
支撑不了「处方解释」。在拿到疗效反馈数据之前，任何关于模型能说明处方理由的说法都不成立。

## 安装与运行

```bash
pip install -r requirements.txt
```

数据工程、全部基线、消融、归因与校验器可以一次跑完，不需要 GPU，约 15 分钟：

```bash
./run_stage0.sh          # 产物在 data/*.jsonl 与 reports/*.md
python3 src/compare_models.py   # 只看统一评测对比与 Gate 判定
```

### 训练

需要 NVIDIA GPU，实测单卡 96GB。

```bash
# 下载基座权重（18.8GB）
python3 src/download_model.py --out models/MiMo-V2.6-Distill-Qwen-9B

# 先自检，确认 LoRA 目标模块名与本模型匹配
python3 src/train_sft.py --model models/MiMo-V2.6-Distill-Qwen-9B --inspect

# 训练，2 epoch 约 2 小时 50 分
python3 src/train_sft.py --model models/MiMo-V2.6-Distill-Qwen-9B \
    --train-file data/train_v2.jsonl --val-file data/val_v2.jsonl \
    --output-dir outputs/mimo9b-tcm-lora --epochs 2 --bs 8 --ga 2 --no-grad-ckpt

# 在冻结测试集上推理并评测
python3 src/predict.py --model models/MiMo-V2.6-Distill-Qwen-9B \
    --adapter outputs/mimo9b-tcm-lora --ref data/ref_test.jsonl \
    --out data/preds/mine.jsonl
python3 src/compare_models.py --model mine=data/preds/mine.jsonl
```

自检这一步不能省。基座是混合注意力加多模态架构，32 层里只有 8 层用
`q_proj`/`k_proj`/`v_proj`/`o_proj`，另外 24 层用
`in_proj_qkv`/`in_proj_z`/`in_proj_b`/`in_proj_a`/`out_proj`。
照抄 Qwen3 的 `target_modules` 只会命中 8 层且不会报错，训练照常进行，只是学的东西少了大半。
`train_sft.py` 默认自动探测，`--inspect` 用于人工确认。

### 演示界面

```bash
python3 src/app_model.py --base models/MiMo-V2.6-Distill-Qwen-9B \
    --adapter outputs/mimo9b-tcm-lora --port 7860
```

界面分两个页签：模型推理（四诊 → 诊断 + 处方，复诊会额外给出相对上次方的增删），
以及处方校验器（手工粘贴处方做安全检查，不需要模型）。

## 目录结构

```
├── src/
│   ├── medlib.py                 共用库：解析、归一化、切分、指标口径
│   ├── prepare_data.py           数据工程：清洗、纵向关联、患者分组切分
│   ├── baseline.py               规则基线 / Oracle 上界 / 治疗进程基线
│   ├── evaluate.py               评测唯一入口，模型与基线共用同一套指标
│   ├── compare_models.py         统一对比 + bootstrap CI + 配对检验 + Gate
│   ├── leakage_check.py          泄漏消融（患者分组 vs 按行随机）
│   ├── mine_symptom_herb.py      症状到药的关联挖掘，含混杂护栏
│   ├── attribution.py            处方成因归因与医师风格检验
│   ├── reasoning_chain.py        五步推理链可行性分析
│   ├── build_external_data.py    外部知识库整合
│   ├── build_zhi_action_map.py   治法到功效的对齐表
│   ├── validate_rx.py            功效与安全约束校验器
│   ├── dose_filler.py            剂量填充层
│   ├── constrained_decode.py     白名单与十八反约束解码
│   ├── train_sft.py              bf16 LoRA 训练，含模块名探测
│   ├── predict.py                冻结测试集批量推理
│   ├── app.py / app_model.py     Gradio 演示
│   ├── sanitize_for_release.py   发布前脱敏，机构名归一化
│   └── check_consistency.py      文档数字与生成报告的一致性检查
├── data/                         派生数据（CC BY-NC 4.0）
│   └── external/                 外部知识库，各自许可见其 LICENSE
├── docs/                         详细文档
├── reports/                      实验报告，全部由脚本生成
└── run_stage0.sh                 一键复现全流程
```

## 许可

代码与数据的许可不同，需要分别遵守。

| 范围 | 许可 | 文件 |
|---|---|---|
| `src/` 源代码 | Apache-2.0 | [LICENSE](LICENSE) |
| `data/` 派生数据 | CC BY-NC 4.0，禁止商用 | [LICENSE-DATA](LICENSE-DATA) |
| `data/external/` 外部知识库 | 各源许可 | [data/external/LICENSE](data/external/LICENSE) |

外部知识库中的本草典为 CC BY-SA 4.0，其 ShareAlike 条款不允许附加非商用限制，
与 CC BY-NC 4.0 不兼容，因此两者分开存放、分别声明。
如果把两部分合并再分发，整体只能采用 CC BY-SA 4.0。

## 已发布资源

**代码与文档**

| 平台 | 地址 |
|---|---|
| GitHub | https://github.com/johnson7788/man_medical |

**模型**（193 MB）

| 平台 | 地址 |
|---|---|
| HuggingFace | https://huggingface.co/johnson/MiMo-9B-TCM-Andrology-LoRA |
| ModelScope | https://www.modelscope.cn/models/InfoxmedModel/MiMo-9B-TCM-Andrology-LoRA |

**数据集**（167 MB）

| 平台 | 地址 |
|---|---|
| HuggingFace | https://huggingface.co/datasets/johnson/TCM-Andrology-EMR |
| ModelScope | https://www.modelscope.cn/datasets/InfoxmedModel/TCM-Andrology-EMR |

数据集同时也放在本仓库的 `data/` 下，两处内容一致，都是脱敏后的版本。

`adapter_config.json` 里的 `base_model` 指向仓库 id，加载时不需要改路径：

```python
from transformers import AutoModelForImageTextToText, AutoTokenizer
from peft import PeftModel
import torch

BASE = "XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B"
ADAPTER = "johnson/MiMo-9B-TCM-Andrology-LoRA"
# 国内网络可换成 "InfoxmedModel/MiMo-9B-TCM-Andrology-LoRA"

tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
model = AutoModelForImageTextToText.from_pretrained(
    BASE, dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=True)
model = PeftModel.from_pretrained(model, ADAPTER)
tok.padding_side = "left"      # decoder-only 生成必须用 left padding
```

用 ModelScope 下载：

```python
from modelscope import snapshot_download
ADAPTER = snapshot_download("InfoxmedModel/MiMo-9B-TCM-Andrology-LoRA")
```

## 引用

```bibtex
@misc{tcm_andrology_2026,
  title  = {中医男科病历 · 辨证论治模型},
  year   = {2026},
  note   = {数据 CC BY-NC 4.0，代码 Apache-2.0},
  url    = {https://github.com/johnson7788/man_medical}
}
```

本项目为研究与教学用途，不构成医疗建议，不得用于临床诊断、处方决策
或任何直接影响患者诊疗的用途。
