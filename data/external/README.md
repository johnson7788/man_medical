# 外部知识库（external）

> 为男科中医模型训练下载的**外部权威知识源**。所有内容只读，不参与训练标签。
> 用途：支撑《模型训练计划.md》§13 五步推理链的步骤 2/3/4/5（治法查表、选方检索、加减依据、功效校验）。

## 为什么需要这些数据

本方病历（`男科(2).xlsx`）只有「四诊文本 → 诊断 → 处方」，**没有任何药理依据**：
无药名标准名、无功效、无归经、无剂量、无禁忌、无方剂出处。
推理链的每一步都需要外部知识来锚定，否则模型只能学到相关性的表面。

## 数据源总览

| 目录 | 来源 | 内容 | 许可 | 获取方式 |
|---|---|---|---|---|
| `bencaodian/` | [本草典开放数据 v1](https://bencaodian.org/zh/about/data/) | 中药材 365 · 方剂 112 · 证型 84 · 医案 28 · 舌象 25 · 脉象 28 · 症状同义词 155 · 中西药相互作用 56 | **CC BY-SA 4.0** | 官方 `/data/v1/*.json` 直下 |
| `symmap/` | [SymMap v2.0](http://www.symmap.org/download/) | 药材 698(性味/归经) · 中医症状 2285 · 西医症状 1148 · 疾病 14086 · 证型 233 | 学术使用 | 官方 xlsx 直下（仅 HTTP） |
| `TCM-Prescription-Recommendation/` | [GitHub](https://github.com/lin-haust/TCM-Prescription-Recommendation) | 疾病 343 · 方 900+（教科书写方） | 仓库未标注，仅研究用 | git clone（经 ghproxy.net） |
| `PresRecST/` | [GitHub](https://github.com/2020MEAI/PresRecST) | 4485 条「症状→证候→治法→药」完整推理链 | 学术使用 | git clone（经 ghproxy.net） |

## 归一化后的产出（由 `src/build_external_data.py` 生成）

| 文件 | 内容 | 关键指标 |
|---|---|---|
| `herb_dictionary.jsonl` | 本方 462 味药 → 标准名/功效/主治/性味/归经/剂量/禁忌/炮制/药理成分 | **用量加权覆盖 99.54%** |
| `formula_library.json` | 方剂 112 首，含**君臣佐使 + 方解** | 方解覆盖 100% |
| `pattern_dictionary.json` | 证型 317 条（含主症/舌象/脉象） | — |
| `symptom_vocabulary.json` | 中医症状 2285 · 西医症状 1148 · 同义词 155 | — |
| `safety_rules.json` | 十八反 15 · 十九畏 9 · 妊娠禁忌 23 · 中西药相互作用 56 | — |

### 覆盖度（按本方实际用药频次加权，这才是有效指标）

| 字段 | 覆盖 | 作用 |
|---|---|---|
| 剂量范围 | **89.07%** | 解决「原始数据剂量全为 1.000」的卡点，用权威标准量而非模型编造 |
| 功效 actions | **89.07%** | 步骤5 硬校验：`方中功效 ⊇ 治法` |
| 主治 indications | **89.07%** | 中药「适应症」映射 |
| 归经 meridians | **97.34%** | 配伍归经分析 |

## 重要说明与限制

1. **PresRecST 的字典是匿名 ID**（51 证候/62 治法/380 药/975 症状），公开版无名称映射，
   全名需邮件申请。因此它**只作架构参考**，不能直接接入本项目药名。
2. **SymMap 下载文件只有实体表，没有关系边** —— 草药↔症状↔疾病的连接不在 xlsx 里。
   本项目所需的「中药↔适应症」改由本草典的 `indications` 字段承担。
3. **TCM-Prescription-Recommendation 药名有 OCR 噪声**（如「花术」应为苍术、「银花」应为金银花），
   仅作备查，未纳入 `herb_dictionary.jsonl`。
4. **十八反十九畏为硬编码**：本草典的 `herb_incompatible_with` 边为 **0 条**，
   故按《中药学》经典歌诀在 `src/build_external_data.py` 中硬编码，需中医师复核。
5. **外部数据不能补舌脉**：本方病历缺舌象/脉象字段，这是需要向院方申请的输入侧缺口。

## 合规

- 本草典为 **CC BY-SA 4.0**：本项目产出若使用其内容，需以相同许可共享并署名。
- 其余源标注学术/研究用途，**不得商用**；正式产品化前需逐一确认许可。
- 全部为公开数据集的批量下载，无全文爬取、无绕过访问控制。

## 文件清单与校验

| 文件 | 大小 | SHA-256(前16) |
|---|---|---|
| PresRecST/data/TCM_Lung.xlsx | 269 KB | `68e911dd0fe0c35a` |
| PresRecST/data/prescript_1195.csv | 1.6 MB | `d14d961d7fc03ca8` |
| PresRecST/requirements.txt | 3 KB | `9459663bab74339c` |
| TCM-Prescription-Recommendation/datasets/Disease.csv | 961 KB | `eed34dc5044b97dc` |
| TCM-Prescription-Recommendation/datasets/Disease.txt | 961 KB | `02f2bdfe768aedec` |
| TCM-Prescription-Recommendation/datasets/Disease.xlsx | 1.4 MB | `be1b57edb67180bc` |
| TCM-Prescription-Recommendation/datasets/Disease2.txt | 1.9 MB | `6434c912985105cd` |
| TCM-Prescription-Recommendation/datasets/疾病-处方数据集总表.xlsx | 1.4 MB | `cbc125b5cf796222` |
| TCM-Prescription-Recommendation/datasets/药材编号对应表.xlsx | 107 KB | `dd716d667c99111f` |
| TCM-Prescription-Recommendation/main code/Disease.csv | 961 KB | `eed34dc5044b97dc` |
| TCM-Prescription-Recommendation/main code/Disease.txt | 51 KB | `08b84f12a22438fb` |
| bencaodian/acupoints.json | 448 KB | `d78f34e8f1b70e31` |
| bencaodian/case_records.json | 73 KB | `36db4ad97f0423c6` |
| bencaodian/classical_texts.json | 213 KB | `2cb13eb854092d68` |
| bencaodian/concepts.json | 213 KB | `859760906b30e934` |
| bencaodian/conditions.json | 96 KB | `cd69812df45815e5` |
| bencaodian/formulas.json | 393 KB | `a31a02c6325e744f` |
| bencaodian/herbs.json | 1.9 MB | `5773a762c92b2ae3` |
| bencaodian/interactions.json | 58 KB | `c68151aa0218591a` |
| bencaodian/meridians.json | 27 KB | `0ce0a1845222acbb` |
| bencaodian/patterns.json | 245 KB | `15ec9bf532cc3f67` |
| bencaodian/pulses.json | 45 KB | `d88cc419b334320b` |
| bencaodian/relationships.json | 166 KB | `62169b7458aaab4a` |
| bencaodian/symptom_synonyms.json | 47 KB | `d21f7d4a65b4e35f` |
| bencaodian/tongue_states.json | 34 KB | `d951aabb91def71a` |
| formula_library.json | 145 KB | `0c0d2b6a456a0103` |
| herb_dictionary.jsonl | 540 KB | `6a80374e4a25ed75` |
| pattern_dictionary.json | 94 KB | `8520c80b7b7242c1` |
| safety_rules.json | 55 KB | `82d3c2c59fc40f6e` |
| symmap/SMDE.xlsx | 2.2 MB | `05545f8cecbafe87` |
| symmap/SMDE_key.xlsx | 719 KB | `cb820976e66761a6` |
| symmap/SMHB.xlsx | 97 KB | `aafa7a2dca0298ec` |
| symmap/SMHB_key.xlsx | 64 KB | `7b521fc01582a423` |
| symmap/SMMS.xlsx | 256 KB | `c83d52386f7f723a` |
| symmap/SMMS_key.xlsx | 294 KB | `271e56c3ef0c43e2` |
| symmap/SMSY.xlsx | 31 KB | `6d91b4bfae9752f4` |
| symmap/SMSY_key.xlsx | 20 KB | `62b38cc3d8804566` |
| symmap/SMTS.xlsx | 220 KB | `cb548f0ff63aba55` |
| symmap/SMTS_key.xlsx | 101 KB | `faef8b3bf4dedfad` |
| symptom_vocabulary.json | 594 KB | `3b2e63ad21534cba` |

共计 **40** 个文件。

## 复现

```bash
# 重新整合（读取本目录原始文件，产出归一化字典）
python3 src/build_external_data.py
# 覆盖率报告: reports/external_data_report.md
```
