"""
train_sft.py — QLoRA 监督微调（B2）

相对一份常见 SFT 参考实现的 5 处必要改动（计划 §5 Stage 2）：
  1. learning_rate 2e-4 → **1e-4**（2.1 万条 / 3 epoch，30B 上 2e-4 易过拟合）
  2. **增加 val 集与早停**（原脚本只有 train，无法发现过拟合）
  3. max_length 2048 → **1024**（序列实际很短，省一半显存与时间）
  4. **清理注释掉的 `get_peft_model` 死代码**（SFTTrainer 传 peft_config 已生效，重复包装会出问题）
  5. 显式 **packing=False**（样本短且独立，packing 会跨样本污染）

另加两项本项目特有的要求：
  6. **初诊/复诊分开评测**（§1.8：两者难度差一个量级，混在一起报平均数会掩盖真相）
  7. **训练期就接入约束校验器**（§15）：生成后过一遍 validate_rx，把 illegal/incompat 率作为监控指标

用法：
  python3 src/train_sft.py --dry-run        # 无需 GPU/torch：校验数据与配置、算步数
  python3 src/train_sft.py --model ./models/Qwen3-8B-Instruct   # 真正训练（需 GPU）
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import medlib as M
from medlib import DATA_DIR, ROOT


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

def default_config(model_id: str, train: str = "data/train_v2.jsonl",
                   val: str = "data/val_v2.jsonl",
                   output_dir: str = "models/man-medical-lora") -> dict:
    """T3+ 为主任务（含既往处方），因为实测它把上界从 0.183 提到 0.438。"""
    return {
        "model_id": model_id,
        "train_file": train,
        "val_file": val,
        "output_dir": output_dir,
        "max_length": 1024,
        "packing": False,
        "lora": {"r": 16, "lora_alpha": 32, "lora_dropout": 0.05, "bias": "none",
                 "task_type": "CAUSAL_LM",
                          # 默认 "auto"：加载模型后自动探测线性层名并排除输出头/视觉塔。
                 # 为什么不用写死的 Qwen3 名单：MiMo-V2.6 是**混合注意力 + 多模态**，
                 # 模块名不同，照抄会**静默失效**（LoRA 不生效但不报错）。
                 # 指定具体名单可用 --target-modules 覆盖。
                 "target_modules": "auto"},
        "sft": {"per_device_train_batch_size": 2, "gradient_accumulation_steps": 4,
                "learning_rate": 1e-4, "num_train_epochs": 3,
                "lr_scheduler_type": "cosine",
                "bf16": True,
                "gradient_checkpointing": True,
                "eval_strategy": "epoch", "save_strategy": "epoch",
                "logging_steps": 10, "save_total_limit": 2,
                "load_best_model_at_end": True,
                "metric_for_best_model": "eval_subset_jaccard",
                "greater_is_better": True, "report_to": "none"},
        # 生成式评测：每个 epoch 抽 val 的子集测 Jaccard，并按初诊/复诊分开
        "eval_generation": {"enabled": True, "n_samples": 200, "max_new_tokens": 200,
                            "temperature": 0.0, "every_n_epochs": 1},
        # 早停：连续 N 次评测无提升则停
        "early_stopping": {"enabled": True, "patience": 2, "threshold": 0.0},
        # ⚠️ transformers 5.x 移除了 warmup_ratio，只接受 warmup_steps。
        # 保留比例作为本模块的配置，train() 里按实际总步数换算成 warmup_steps。
        "warmup_ratio": 0.03,
        "seed": 42,
        # 96GB 显存下 9B 模型用 bf16 LoRA 即可，无需 4-bit QLoRA
        "quant": "none",          # none | 4bit
        "dtype": "bfloat16",
    }



# ---------------------------------------------------------------------------
# 模型自检 / LoRA 目标模块探测（--inspect）
# ---------------------------------------------------------------------------

# 尝试顺序：多模态模型优先用 ImageTextToText；纯文本模型用 CausalLM。
AUTO_CLASSES = [
    ("AutoModelForImageTextToText", "image-text-to-text"),
    ("AutoModelForVision2Seq", "vision2seq"),
    ("AutoModelForCausalLM", "causal-lm"),
]


def load_model_for_inspect(model_id: str, quant: str = "none", dtype: str = "bfloat16"):
    """按优先级尝试多种 AutoModel 类加载；返回 (model, tok, 用到的类名)。"""
    import torch
    from transformers import AutoConfig, AutoTokenizer
    import transformers

    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dt = getattr(torch, dtype)
    kw = dict(trust_remote_code=True, dtype=dt, device_map={"": 0} if torch.cuda.is_available() else None)
    if quant == "4bit":
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dt)

    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    arch = (cfg.architectures or ["?"])[0]
    print(f"  config.architectures = {arch} | model_type = {cfg.model_type}")
    print(f"  transformers {transformers.__version__}")

    last_err = None
    for cls_name, _tag in AUTO_CLASSES:
        cls = getattr(transformers, cls_name, None)
        if cls is None:
            print(f"  ✗ {cls_name} 不存在于当前 transformers")
            continue
        try:
            model = cls.from_pretrained(model_id, **kw)
            return model, tok, cls_name, arch
        except Exception as e:                       # noqa: BLE001
            last_err = f"{cls_name}: {type(e).__name__}: {str(e)[:160]}"
            print(f"  ✗ {cls_name} 失败 → {last_err}")
    raise RuntimeError(f"所有 AutoModel 类都加载失败。最后错误：{last_err}")


def probe_lora_targets(model) -> dict:
    """探测该模型适合 LoRA 的线性层名，并区分语言模型 / 视觉塔。

    为什么必须探测：MiMo-V2.6 是**混合注意力**架构（config 里有 layer_types /
    linear_attention / mamba_ssm_dtype），线性注意力层与全注意力层的模块名不同，
    加上它还是**多模态**模型 —— 照抄 Qwen3 的 target_modules 会漏掉大半甚至报错。
    """
    import torch.nn as nn

    leaf_count: dict[str, int] = {}
    leaf_sample: dict[str, str] = {}
    vision_leaf: set[str] = set()
    lm_leaf: set[str] = set()
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            leaf = name.split(".")[-1]
            leaf_count[leaf] = leaf_count.get(leaf, 0) + 1
            leaf_sample.setdefault(leaf, name)
            if "vision" in name or "visual" in name:
                vision_leaf.add(leaf)
            else:
                lm_leaf.add(leaf)
    return {"leaf_count": leaf_count, "leaf_sample": leaf_sample,
            "lm_leaf": sorted(lm_leaf), "vision_leaf": sorted(vision_leaf)}


# 这些叶子名是"输出头 / 嵌入 / 多模态投影"，不适合（或不该）加 LoRA：
# 输出头加 LoRA 会干扰词表分布；多模态投影加上去对纯文本任务无益且浪费容量。
_EXCLUDE_LEAF = {"lm_head", "score", "classifier", "embed_tokens", "embed_proj",
                 "lm_head_proj", "merger", "connector", "projector", "multi_modal_projector",
                 "vision_model", "visual"}


def auto_target_modules(model, verbose: bool = True) -> list[str]:
    """自动推导 LoRA 目标模块：取语言模型里所有线性层叶子名，排除输出头/嵌入/视觉塔。

    为什么需要它：不同架构（纯全注意力 vs 混合线性注意力 vs 多模态）的模块名不同，
    手抄一份名单极易静默失效（LoRA 不生效但不报错）。实测教训见 README/技术报告。
    """
    pr = probe_lora_targets(model)
    picked, dropped = [], []
    for leaf in pr["lm_leaf"]:
        if leaf in _EXCLUDE_LEAF or leaf in pr["vision_leaf"]:
            dropped.append(leaf)
        else:
            picked.append(leaf)
    if verbose:
        print(f"  [auto target_modules] 选中 {len(picked)} 个: {picked}")
        if dropped:
            print(f"  [auto target_modules] 排除 {len(dropped)} 个: {dropped}")
        # 逐名统计命中层数，便于确认非空
        for leaf in picked:
            print(f"      {leaf:24s} ×{pr['leaf_count'].get(leaf, 0)}")
    return sorted(picked)


def _top_modules(model, k: int = 8) -> list[tuple[str, int]]:
    named = [(n, sum(p.numel() for p in m.parameters()))
             for n, m in model.named_children()]
    return sorted([x for x in named if x[1] > 0], key=lambda x: -x[1])[:k]


def inspect_model(cfg: dict) -> int:
    """加载模型并打印：结构 / 线性层名 / 建议的 LoRA target_modules / 显存。"""
    import torch
    print("=" * 78)
    print("模型自检（--inspect）")
    print("=" * 78)
    print(f"\n[1] 加载 {cfg['model_id']}（quant={cfg.get('quant','none')}）")
    if not torch.cuda.is_available():
        print("  ⚠️ CUDA 不可用，将以 CPU 加载（很慢，仅供结构探测）")
    model, tok, used, arch = load_model_for_inspect(cfg["model_id"], cfg.get("quant", "none"),
                                                    cfg.get("dtype", "bfloat16"))

    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[2] 参数量")
    print(f"  总计 {n_total/1e9:.2f} B（trainable {n_train/1e9:.3f} B）")
    print(f"  顶层子模块：")
    for n, c in _top_modules(model):
        print(f"    {n:28s} {c/1e9:8.3f} B")

    pr = probe_lora_targets(model)
    print(f"\n[3] 线性层叶子名（按出现次数）")
    for leaf, cnt in sorted(pr["leaf_count"].items(), key=lambda x: -x[1]):
        tag = " [视觉]" if leaf in pr["vision_leaf"] else ""
        print(f"    {leaf:26s} ×{cnt:<5d}{tag}   e.g. {pr['leaf_sample'][leaf][:70]}")

    print(f"\n[4] 语言模型部分的线性层名（建议作为 LoRA 目标，排除视觉塔）")
    print(f"    {pr['lm_leaf']}")
    if pr["vision_leaf"]:
        print(f"  视觉塔的线性层名（**不建议**加入 LoRA）：{pr['vision_leaf']}")

    # 与计划里写死的 Qwen3 名单对比
    planned = cfg["lora"]["target_modules"]
    print(f"\n[5] LoRA target_modules 选择")
    if isinstance(planned, str):
        print(f"  配置为 {planned!r} → 运行时自动探测。")
        print(f"  自动探测结果（排除输出头/视觉塔）：{auto_target_modules(model, verbose=True)}")
        print(f"  ⚠️ 注意：计划里写死的 Qwen3 名单 "
              f"['q_proj','k_proj','v_proj','o_proj',...] 在本模型只会命中 "
              f"{pr['leaf_count'].get('q_proj',0)} 层全注意力，"
              f"**漏掉 {pr['leaf_count'].get('in_proj_qkv',0)} 层线性注意力** → 必须用 auto。")
    else:
        planned = list(planned)
        missing = [t for t in planned if t not in pr["leaf_count"]]
        extra = [t for t in pr["lm_leaf"] if t not in planned]
        print(f"  指定名单: {planned}")
        if missing:
            print(f"  ❌ 名单里这些在本模型【不存在】：{missing} → 会静默失效")
        if extra:
            print(f"  ➕ 本模型额外存在的语言模型线性层：{extra}")

    print(f"\n[6] Tokenizer")
    print(f"  vocab_size = {len(tok)} | pad={tok.pad_token!r} eos={tok.eos_token!r}")
    print(f"  chat_template: {'有' if tok.chat_template else '无'}")
    probe_msgs = [{"role": "system", "content": M.SYSTEM_PROMPT},
                  {"role": "user", "content": "测试"}]
    try:
        txt = tok.apply_chat_template(probe_msgs, tokenize=False, add_generation_prompt=True)
        print(f"  apply_chat_template OK，样例前 120 字：{txt[:120]!r}")
    except Exception as e:                            # noqa: BLE001
        print(f"  ❌ apply_chat_template 失败：{type(e).__name__}: {str(e)[:150]}")
        print("     → 训练前必须解决：本项目数据是 messages 格式，依赖 chat template。")

    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        print(f"\n[7] 显存：已用 {(total-free)/1e9:.1f} GB / 共 {total/1e9:.1f} GB "
              f"（剩余 {free/1e9:.1f} GB）")
    print(f"\n加载所用类：{used}")
    return 0

# ---------------------------------------------------------------------------
# 数据校验（dry-run 的核心，无需 GPU）
# ---------------------------------------------------------------------------

def _read_chatml(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"缺少数据文件：{path}")
    rows = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        d = json.loads(line)
        msgs = d.get("messages")
        if not isinstance(msgs, list) or len(msgs) != 3:
            raise ValueError(f"{path} 第 {i+1} 行不是 3 轮 messages 结构")
        roles = [m.get("role") for m in msgs]
        if roles != ["system", "user", "assistant"]:
            raise ValueError(f"{path} 第 {i+1} 行 role 顺序异常：{roles}")
        rows.append(d)
    return rows


def estimate_tokens(text: str) -> int:
    """粗估 token 数。Qwen 分词器对中文大致 1 字 ≈ 0.7–1 token，
    这里用 0.8 作保守估计。**正式训练前应用真实 tokenizer 复核长度分布。**
    """
    return max(1, int(len(text) * 0.8))


def dry_run(cfg: dict) -> int:
    """不依赖 torch/transformers：校验数据、配置、步数与初诊/复诊子集。"""
    print("=" * 78)
    print("DRY RUN — 校验数据与配置（不需要 GPU / torch）")
    print("=" * 78)
    fails: list[str] = []

    def check(name: str, ok: bool, extra: str = "") -> None:
        print(f"  {'✅' if ok else '❌'} {name}" + (f"  {extra}" if extra else ""))
        if not ok:
            fails.append(name)

    train_p, val_p = ROOT / cfg["train_file"], ROOT / cfg["val_file"]
    print(f"\n[1] 数据文件")
    tr = _read_chatml(train_p)
    va = _read_chatml(val_p)
    print(f"  train {len(tr):,} 条  {train_p.relative_to(ROOT)}")
    print(f"  val   {len(va):,} 条  {val_p.relative_to(ROOT)}")
    check("train/val 均非空", bool(tr) and bool(va))

    # 与评测参考文件对齐（rid 数一致）—— 踩过一次"预测文件喂错 rid 导致 4887 条缺失"的坑
    print(f"\n[2] 与评测参考文件的对齐")
    ref_tr = DATA_DIR / "ref_train.jsonl"
    ref_va = DATA_DIR / "ref_val.jsonl"
    ok_align = True
    for jf, rf, nm in ((train_p, ref_tr, "train"), (val_p, ref_va, "val")):
        n_j = len(_read_chatml(jf))
        n_r = len(M.read_jsonl(rf)) if rf.exists() else -1
        ok_align &= (n_j == n_r)
        print(f"  {nm}: jsonl {n_j:,} / ref {n_r:,}")
    check("jsonl 与 ref_* 条数一致", ok_align)

    # 初诊/复诊子集（分开评测的前提）
    print(f"\n[3] 初诊/复诊子集（§1.8 要求分开评测）")
    stats = {}
    for nm, ref in (("train", ref_tr), ("val", ref_va)):
        rows = M.read_jsonl(ref)
        fu = sum(1 for r in rows if r.get("prev_rx"))
        stats[nm] = (len(rows) - fu, fu)
        pct = fu / max(1, len(rows)) * 100
        print(f"  {nm}: 初诊 {len(rows)-fu:,} / 复诊 {fu:,}（复诊 {pct:.1f}%）")
    check("val 同时含初诊与复诊样本（否则无法分开评测）",
          stats["val"][0] > 0 and stats["val"][1] > 0, f"初诊{stats['val'][0]} 复诊{stats['val'][1]}")

    # 长度
    print(f"\n[4] 序列长度（估算，1 字 ≈ 0.8 token）")
    lens_in, lens_out, lens_tot = [], [], []
    for d in tr:
        u = d["messages"][1]["content"]
        a = d["messages"][2]["content"]
        s = d["messages"][0]["content"]
        lens_in.append(estimate_tokens(u))
        lens_out.append(estimate_tokens(a))
        lens_tot.append(estimate_tokens(s + u + a))
    lens_tot.sort()
    p50 = lens_tot[len(lens_tot) // 2]
    p99 = lens_tot[int(len(lens_tot) * 0.99)]
    mx = lens_tot[-1]
    print(f"  输入 token 均值 {statistics.mean(lens_in):.0f} / 输出均值 {statistics.mean(lens_out):.0f}")
    print(f"  总长 p50={p50} p99={p99} max={mx}")
    check(f"max_length({cfg['max_length']}) 覆盖 p99", cfg["max_length"] >= p99,
          f"p99={p99}")
    n_over = sum(1 for x in lens_tot if x > cfg["max_length"])
    if n_over:
        print(f"  ⚠️ 有 {n_over} 条（{n_over/len(lens_tot)*100:.2f}%）超过 max_length 会被截断")
    else:
        for cand in (512, 384):
            if cand >= p99:
                print(f"  💡 max_length 可收窄到 {cand}（仍覆盖 p99={p99}），"
                      f"能进一步省显存与时间")
                break

    # LoRA 目标模块
    print(f"\n[5] LoRA 配置")
    print(f"  r={cfg['lora']['r']} alpha={cfg['lora']['lora_alpha']} "
          f"dropout={cfg['lora']['lora_dropout']}")
    tm = cfg["lora"]["target_modules"]
    if tm in (None, "auto", ["auto"]):
        print("  target_modules: auto（运行时自动探测；本机校验跳过，需 --inspect 确认）")
    else:
        print(f"  target_modules: {', '.join(tm)}")
    mdl = ROOT / cfg["model_id"].lstrip("./")
    if mdl.exists():
        conf_p = mdl / "config.json"
        if conf_p.exists():
            mc = json.loads(conf_p.read_text(encoding="utf-8"))
            n_layers = mc.get("num_hidden_layers")
            hidden = mc.get("hidden_size")
            print(f"  模型 config: layers={n_layers} hidden={hidden} "
                  f"arch={mc.get('architectures')}")
            missing = [t for t in cfg["lora"]["target_modules"]
                       if t not in json.dumps(mc)]
            check("target_modules 与模型结构匹配（近似判断）", not missing, f"未匹配 {missing}")
        else:
            print(f"  ⚠️ 模型目录存在但缺 config.json")
    else:
        print(f"  ⚠️ 模型路径不存在：{mdl}")
        print(f"     本机无 GPU 时这是预期的；训练前需把基座权重放到该路径。")
        print(f"     若模块名与基座不匹配，LoRA 会静默不生效 → 建议训练首步打印 "
              f"model.print_trainable_parameters() 并核对非零。")

    # 步数
    print(f"\n[6] 训练步数估算")
    bs = cfg["sft"]["per_device_train_batch_size"]
    ga = cfg["sft"]["gradient_accumulation_steps"]
    ep = cfg["sft"]["num_train_epochs"]
    eff = bs * ga
    steps_per_epoch = max(1, len(tr) // eff)
    total = steps_per_epoch * ep
    print(f"  有效 batch = {bs} × {ga} = {eff}")
    print(f"  每 epoch ≈ {steps_per_epoch:,} 步；{ep} epoch 共 ≈ {total:,} 步")
    ws = max(1, int(total * cfg.get("warmup_ratio", 0.03)))
    print(f"  warmup_steps ≈ {ws}（比例 {cfg.get('warmup_ratio',0.03)}；"
          f"transformers 5.x 已移除 warmup_ratio）")
    print(f"  评测：每 epoch 生成式评测 {cfg['eval_generation']['n_samples']} 条 val 样本")
    if cfg["early_stopping"]["enabled"]:
        print(f"  优化器：{'paged_adamw_32bit（QLoRA）' if cfg.get('quant')=='4bit' else 'adamw_torch（bf16 LoRA，无需 bitsandbytes）'}")
    print(f"  梯度检查点：{cfg['sft'].get('gradient_checkpointing')}")
    print(f"  早停：patience={cfg['early_stopping']['patience']}")
    check("总步数在合理区间（100–100000）", 100 <= total <= 100000, f"{total}")

    # 依赖
    print(f"\n[7] 训练依赖")
    mods = {"torch": "torch", "transformers": "transformers", "peft": "peft",
            "trl": "trl", "datasets": "datasets"}
    miss = []
    for name, imp in mods.items():
        try:
            __import__(imp)
            print(f"  ✅ {name}")
        except ImportError:
            print(f"  ❌ {name} 未安装")
            miss.append(name)
    if miss:
        print(f"  → 缺 {miss}；训练机需安装：pip install {' '.join(miss)}")

    print("\n" + "=" * 78)
    if fails:
        print(f"❌ DRY RUN 失败 {len(fails)} 项：{fails}")
        return 1
    print("✅ DRY RUN 通过" + ("（依赖未装，仅数据与配置校验）" if miss else ""))
    return 0


# ---------------------------------------------------------------------------
# 训练（需要 torch / transformers / peft / trl）
# ---------------------------------------------------------------------------

def train(cfg: dict) -> None:
    import torch
    from datasets import load_dataset
    from peft import LoraConfig, prepare_model_for_kbit_training
    from transformers import EarlyStoppingCallback, TrainerCallback
    from trl import SFTConfig, SFTTrainer

    from evaluate import diagnosis_metrics, rx_metrics
    from validate_rx import Validator

    # --- tokenizer / model ---
    # 用与 --inspect 相同的自动加载逻辑：MiMo-V2.6 是多模态架构，
    # AutoModelForCausalLM 不一定能加载，需要按优先级尝试。
    model, tok, used_cls, arch = load_model_for_inspect(
        cfg["model_id"], cfg.get("quant", "none"), cfg.get("dtype", "bfloat16"))
    print(f"  模型已加载（{used_cls} / {arch}），quant={cfg.get('quant','none')}")
    model.config.use_cache = False
    if cfg.get("quant") == "4bit":
        model = prepare_model_for_kbit_training(model)
    elif hasattr(model, "enable_input_require_grads"):
        # bf16 LoRA 配合 gradient_checkpointing 时需要，否则梯度不会回传到 LoRA
        model.enable_input_require_grads()

    # 重要：不调用 get_peft_model —— SFTTrainer 传 peft_config 已生效，
    # 重复包装会出问题（很多参考脚本里那行被注释掉的调用就是这么来的）。
    lora_kw = dict(cfg["lora"])
    if lora_kw.get("target_modules") in (None, "auto", ["auto"]):
        lora_kw["target_modules"] = auto_target_modules(model)
    if not lora_kw["target_modules"]:
        raise RuntimeError("LoRA target_modules 为空 —— 模块名探测失败，拒绝开始训练"
                           "（否则会训练出一个没有任何可训练参数的模型）")
    peft_config = LoraConfig(**lora_kw)
    # 注意：此处【不能】调用 model.print_trainable_parameters() ——
    # 那是 PEFT 的方法，而 LoRA 要等 SFTTrainer 内部应用 peft_config 之后才存在。
    # 真正的校验放到 trainer 构造之后（见下），因为"LoRA 静默不生效"是本项目
    # 踩过的真实坑（写死模块名在本模型上会漏掉 24/32 层）。

    # --- data ---
    ds = load_dataset("json", data_files={"train": cfg["train_file"],
                                          "validation": cfg["val_file"]})

    # --- 生成式评测回调：按初诊/复诊分开 ---
    ref_val = M.read_jsonl(DATA_DIR / "ref_val.jsonl")
    validator = Validator()

    class SubsetEvalCallback(TrainerCallback):
        """每个 epoch 在 val 子集上生成，按初诊/复诊分别算 Jaccard 与安全指标。

        为什么必须分开：实测初诊与复诊难度差一个量级
        （复诊照抄上次方 0.542，初诊检索 top-1 仅 0.100），
        混在一起报平均数会掩盖"初诊其实和以前一样难"。
        """

        def __init__(self, n_samples: int, max_new_tokens: int):
            self.metrics: dict = {}
            self.n = n_samples
            self.max_new = max_new_tokens
            self.history: list[dict] = []

        def on_evaluate(self, args, state, control, **kw):
            import random
            every = cfg["eval_generation"]["every_n_epochs"]
            ep = int(state.epoch or 0)
            if not cfg["eval_generation"]["enabled"] or ep % every != 0:
                return control
            rows = [r for r in ref_val if r.get("rx")]
            random.Random(cfg["seed"]).shuffle(rows)
            rows = rows[: self.n]

            # 复诊样本需要带既往处方（v2 输入）
            prompts, preds, refs = [], [], []
            for r in rows:
                txt = M.format_input_v2(r) if cfg["train_file"].endswith("_v2.jsonl") \
                    else M.format_input(r)
                msgs = [{"role": "system", "content": M.SYSTEM_PROMPT},
                        {"role": "user", "content": txt}]
                prompts.append(tok.apply_chat_template(msgs, tokenize=False,
                                                       add_generation_prompt=True))
                refs.append(r)
            model.eval()
            outs = []
            # ⚠️ 必须用 left padding：decoder-only 模型 + right padding 会让生成
            # 从 pad 之后的位置开始，产出不正确的结果。
            # 之前这里漏了设置，导致每 epoch 的 Jaccard 数字不可靠（实测教训）。
            _prev_side = tok.padding_side
            tok.padding_side = "left"
            try:
                with torch.no_grad():
                    for i in range(0, len(prompts), 4):
                        batch = tok(prompts[i:i + 4], return_tensors="pt",
                                    padding=True).to(model.device)
                        gen = model.generate(**batch, max_new_tokens=self.max_new,
                                             do_sample=False, temperature=None, top_p=None,
                                             pad_token_id=tok.pad_token_id)
                        outs += tok.batch_decode(gen[:, batch["input_ids"].shape[1]:],
                                                 skip_special_tokens=True)
            finally:
                tok.padding_side = _prev_side
            for r, o in zip(refs, outs):
                dx, rx = M.parse_assistant(o)
                preds.append((dx, rx, o))

            # 分开统计
            res = {}
            for name, m in (("全部", [True] * len(refs)),
                            ("初诊", [not r.get("prev_rx") for r in refs]),
                            ("复诊", [bool(r.get("prev_rx")) for r in refs])):
                idx = [i for i, k in enumerate(m) if k]
                if not idx:
                    continue
                prx = [preds[i][1] for i in idx]
                rrx = [refs[i]["rx"] for i in idx]
                key = set(M.top_herbs([r for r in ref_val], 100))
                vocab = set(M.top_herbs(ref_val, 10 ** 9))
                rm = rx_metrics(prx, rrx, key, vocab)
                dm = diagnosis_metrics([preds[i][0] for i in idx],
                                       [[tuple(t) for t in refs[i]["tcm_dx"]] for i in idx])
                illegal = sum(1 for i in idx
                              if any(f["code"] == "ILLEGAL_HERB"
                                     for f in validator.validate_prescription(
                                         preds[i][1], refs[i]["tcm_dx"])["findings"]))
                res[name] = {"n": len(idx), "jaccard": rm["处方_Jaccard"],
                             "zheng_f1": dm["证型_F1"], "illegal_rate": illegal / len(idx)}
            res["subset_jaccard"] = 0.5 * (res.get("初诊", {}).get("jaccard", 0.0)
                                          + res.get("复诊", {}).get("jaccard", 0.0))
            self.metrics = res
            self.history.append({"epoch": ep, **{k: v for k, v in res.items()
                                                 if isinstance(v, dict)}})
            print(f"\n[生成式评测 @epoch {ep}] "
                  f"全部 J={res['全部']['jaccard']:.3f} | "
                  f"初诊 J={res.get('初诊',{}).get('jaccard',float('nan')):.3f} | "
                  f"复诊 J={res.get('复诊',{}).get('jaccard',float('nan')):.3f} | "
                  f"证型F1={res['全部']['zheng_f1']:.3f} | "
                  f"非法药名率={res['全部']['illegal_rate']*100:.2f}%")
            model.train()
            return control

        def on_log(self, args, state, control, logs=None, **kw):
            return control

    eval_cb = SubsetEvalCallback(cfg["eval_generation"]["n_samples"],
                                 cfg["eval_generation"]["max_new_tokens"])

    class MetricInjector(TrainerCallback):
        """把回调算出的 subset_jaccard 注入 logs，供 load_best_model_at_end 使用。"""

        def on_evaluate(self, args, state, control, metrics=None, **kw):
            if metrics is not None and eval_cb.metrics:
                metrics["eval_subset_jaccard"] = eval_cb.metrics.get("subset_jaccard", 0.0)
                for k in ("初诊", "复诊"):
                    if k in eval_cb.metrics:
                        metrics[f"eval_jaccard_{k}"] = eval_cb.metrics[k]["jaccard"]
            return control

    # transformers 5.x 只接受 warmup_steps；按实际总步数把比例换算过去
    sft_kw = dict(cfg["sft"])
    sft_kw.pop("warmup_ratio", None)          # 保险：万一被回填进来
    # 优化器按量化方式选：
    #   bf16 LoRA（本项目默认，96GB 显存）→ adamw_torch，不需要 bitsandbytes
    #   QLoRA(4bit)                        → paged_adamw_32bit（省显存，需 bitsandbytes）
    if cfg.get("quant") == "4bit":
        sft_kw.setdefault("optim", "paged_adamw_32bit")
    else:
        sft_kw["optim"] = "adamw_torch"
    eff_batch = (sft_kw["per_device_train_batch_size"]
                 * sft_kw.get("gradient_accumulation_steps", 1))
    if "max_steps" in sft_kw:
        total_steps = sft_kw["max_steps"]
    else:
        total_steps = max(1, len(ds["train"]) // eff_batch) * sft_kw.get("num_train_epochs", 3)
    warmup_steps = max(1, int(total_steps * cfg.get("warmup_ratio", 0.03)))
    sft_kw["warmup_steps"] = warmup_steps
    print(f"[调度] 总步数≈{total_steps}  有效 batch={eff_batch}  "
          f"warmup_steps={warmup_steps}（比例 {cfg.get('warmup_ratio', 0.03)}，"
          f"warmup_ratio 在 transformers 5.x 已移除）")
    sft_args = SFTConfig(output_dir=cfg["output_dir"], max_length=cfg["max_length"],
                         packing=cfg["packing"], seed=cfg["seed"], **sft_kw)
    cbs = [eval_cb, MetricInjector()]
    if cfg["early_stopping"]["enabled"]:
        # 早停必须基于【与 load_best_model_at_end 同一个指标】，否则两者会打架
        cbs.append(EarlyStoppingCallback(
            early_stopping_patience=cfg["early_stopping"]["patience"],
            early_stopping_threshold=cfg["early_stopping"]["threshold"]))
    trainer = SFTTrainer(model=model, args=sft_args, train_dataset=ds["train"],
                         eval_dataset=ds["validation"], peft_config=peft_config,
                         processing_class=tok, callbacks=cbs)

    # ---- LoRA 生效性硬校验（防止"静默失效"）----
    tm = getattr(trainer.model, "peft_config", None)
    if hasattr(trainer.model, "print_trainable_parameters"):
        trainer.model.print_trainable_parameters()
    n_train = sum(p_.numel() for p_ in trainer.model.parameters() if p_.requires_grad)
    n_total = sum(p_.numel() for p_ in trainer.model.parameters())
    print(f"[LoRA 校验] 可训练 {n_train/1e6:.1f} M / 总计 {n_total/1e9:.2f} B "
          f"= {n_train/n_total*100:.3f}%")
    if n_train == 0:
        raise RuntimeError(
            "LoRA 可训练参数为 0 —— target_modules 未命中任何层，训练将毫无效果。"
            f" 当前 target_modules={lora_kw['target_modules']}。"
            " 用 `--inspect` 查看本模型真实的线性层名。")
    if n_train / n_total > 0.5:
        raise RuntimeError(
            f"可训练参数占 {n_train/n_total*100:.1f}% —— LoRA 未生效（等于全参微调）。"
            " 请检查 peft_config 是否被 SFTTrainer 接受。")
    trainer.train()
    trainer.save_model(cfg["output_dir"])
    tok.save_pretrained(cfg["output_dir"])
    # 记录实际使用的配置，便于复现与排查
    (Path(cfg["output_dir"]) / "train_run_config.json").write_text(
        json.dumps(cfg, ensure_ascii=False, indent=1), encoding="utf-8")
    (Path(cfg["output_dir"]) / "eval_history.json").write_text(
        json.dumps(eval_cb.history, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"✅ 训练完成，LoRA 权重 → {cfg['output_dir']}")
    print(f"   评测历史 → {cfg['output_dir']}/eval_history.json")
    print("   下一步：用 src/evaluate.py 在冻结测试集上评估，并区分初诊/复诊口径。")


# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="./models/Qwen3-8B-Instruct",
                    help="基座模型路径；Stage 2 换成 Qwen3-30B-Instruct")
    ap.add_argument("--train-file", default="data/train_v2.jsonl",
                    help="默认用 T3+（含既往处方）；对照可用 data/train.jsonl")
    ap.add_argument("--val-file", default="data/val_v2.jsonl")
    ap.add_argument("--output-dir", default="models/man-medical-lora")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--inspect", action="store_true",
                    help="加载模型并打印真实线性层名，用于确定 LoRA target_modules")
    ap.add_argument("--quant", choices=["none", "4bit"], default="none",
                    help="none=bf16 LoRA（96GB 显存推荐）；4bit=QLoRA")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--grad-ckpt", dest="grad_ckpt", action="store_true", default=None,
                    help="开启梯度检查点（省显存但慢 ~30-50%%）")
    ap.add_argument("--no-grad-ckpt", dest="grad_ckpt", action="store_false",
                    help="关闭梯度检查点（显存充裕时提速；96GB 单卡推荐）")
    ap.add_argument("--eval-samples", type=int, default=None,
                    help="每 epoch 生成式评测的样本数（默认 200；越小越快）")
    ap.add_argument("--eval-strategy", default=None, choices=["epoch", "steps", "no"],
                    help="覆盖评测策略；冒烟测试用 no")
    ap.add_argument("--max-steps", type=int, default=None,
                    help="覆盖最大训练步数（冒烟测试用）")
    ap.add_argument("--bs", type=int, default=None, help="per_device_train_batch_size")
    ap.add_argument("--ga", type=int, default=None, help="gradient_accumulation_steps")
    ap.add_argument("--target-modules", default=None,
                    help="逗号分隔；覆盖 LoRA 目标模块（先用 --inspect 探测）")
    args = ap.parse_args()

    cfg = default_config(args.model, args.train_file, args.val_file, args.output_dir)
    if args.epochs is not None:
        cfg["sft"]["num_train_epochs"] = args.epochs
    if args.lr is not None:
        cfg["sft"]["learning_rate"] = args.lr
    cfg["quant"] = args.quant
    cfg["dtype"] = args.dtype
    if args.grad_ckpt is not None:
        cfg["sft"]["gradient_checkpointing"] = bool(args.grad_ckpt)
    if args.bs is not None:
        cfg["sft"]["per_device_train_batch_size"] = args.bs
    if args.ga is not None:
        cfg["sft"]["gradient_accumulation_steps"] = args.ga
    if args.eval_samples is not None:
        cfg["eval_generation"]["n_samples"] = args.eval_samples
    if args.eval_strategy is not None:
        cfg["sft"]["eval_strategy"] = args.eval_strategy
        cfg["sft"]["save_strategy"] = args.eval_strategy if args.eval_strategy != "no" else "no"
        cfg["eval_generation"]["enabled"] = args.eval_strategy != "no"
        if args.eval_strategy == "no":
            # 没有评测就不能 load_best_model_at_end（Trainer 会直接报错）
            cfg["sft"]["load_best_model_at_end"] = False
            cfg["early_stopping"]["enabled"] = False
    if args.max_steps is not None:
        cfg["sft"]["max_steps"] = args.max_steps
        # max_steps 与 num_train_epochs 同时给会让 Trainer 报错，须置为默认
        del cfg["sft"]["num_train_epochs"]
    if args.target_modules:
        cfg["lora"]["target_modules"] = [t.strip() for t in args.target_modules.split(",")
                                         if t.strip()]

    if args.inspect:
        sys.exit(inspect_model(cfg))
    if args.dry_run:
        sys.exit(dry_run(cfg))
    train(cfg)


if __name__ == "__main__":
    main()
