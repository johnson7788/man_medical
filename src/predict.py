"""
predict.py — 用训练好的 LoRA 模型在冻结测试集上生成预测

这是训练闭环的最后一环：产出 `data/preds/<name>.jsonl`，格式与 `evaluate.py` 对齐
（`{"rid", "output"}`），从而和规则基线在同一口径下比较。

关键点：
  1. **接入约束解码**（`constrained_decode.make_logits_processor`）——
     药名强制落在 462 白名单内，从解码期就消灭幻觉药名（G3 硬指标）。
  2. **初诊/复诊分开**：`--subset {all,first,fu}`，因为两者难度差一个量级
     （复诊照抄 0.542 / 初诊检索 0.100），混在一起报平均数会掩盖真相。
  3. **T3+ 输入**：用 `format_input_v2`，与 `train_v2.jsonl` 的训练口径一致。
     口径不一致会导致评估数字完全失真。

用法：
  python3 src/predict.py --model models/MiMo-V2.6-Distill-Qwen-9B \
      --adapter outputs/mimo9b-tcm-lora --out data/preds/mimo9b.jsonl
  python3 src/predict.py --limit 50 ...        # 快速抽样
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import medlib as M
from medlib import DATA_DIR

BATCH = 4


def _rebuild_output(dx: list, rx: list[str]) -> str:
    """按评测入口的格式重建成文本，保证 evaluate.py 能解析。"""
    lines = [M.DX_HEADER]
    for b, z, t in dx:
        lines.append(f"{b}·{z}·{t}" if z and t else (b or ""))
    lines.append(M.RX_HEADER)
    lines.append(" ".join(rx))
    return "\n".join(lines)


def build_prompts(rows: list[dict], tok, use_v2: bool) -> list[str]:
    prompts = []
    for r in rows:
        txt = M.format_input_v2(r) if use_v2 else M.format_input(r)
        msgs = [{"role": "system", "content": M.SYSTEM_PROMPT},
                {"role": "user", "content": txt}]
        prompts.append(tok.apply_chat_template(msgs, tokenize=False,
                                               add_generation_prompt=True))
    return prompts



def constrained_second_pass(model, tok, prompt: str, first_pass: str, herbs: list[str],
                            max_new_tokens: int) -> str:
    """诊断段保留自由生成；只对【中药处方】之后的药方段施加白名单约束。

    为什么必须分段：从第一个 token 就约束，模型无法生成 `【中医诊断】` 及其内容，
    输出会退化成纯药名列表（实测踩过这个坑）。
    """
    import torch
    from constrained_decode import make_logits_processor

    marker = M.RX_HEADER if hasattr(M, "RX_HEADER") else "【中药处方】"
    if marker not in first_pass:
        # 模型没按格式出方，退回自由生成结果
        return first_pass
    head, _ = first_pass.split(marker, 1)
    prefix = prompt + head + marker + "\n"
    enc = tok([prefix], return_tensors="pt", padding=True).to(model.device)
    gen_kw = dict(max_new_tokens=max_new_tokens, do_sample=False,
                  pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
                  logits_processor=[make_logits_processor(
                      herbs, tok, prompt_len=int(enc["input_ids"].shape[1]),
                      eos_token_id=tok.eos_token_id)])
    with torch.no_grad():
        out = model.generate(**enc, **gen_kw)
    body = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)[0]
    return head + marker + "\n" + body


def make_whitelist_normalizer(whitelist):
    """构造 h -> 白名单规范名 的映射函数。

    白名单本身就是字典的规范名集合，所以对模型输出的处理是：
    直接命中 → 保留；否则按【炮制前缀/产地前缀/后缀】逐层剥离后再试。
    复用 build_external_data 里那套已验证的规则，避免两处口径不一致。
    """
    from build_external_data import ALIAS, ORIGIN_PREFIX, PROC_PREFIX, SUFFIX

    wl = set(whitelist)

    def norm(h: str) -> str:
        if h in wl:
            return h
        if h in ALIAS and ALIAS[h] in wl:
            return ALIAS[h]
        c = h
        for _ in range(3):
            c2 = PROC_PREFIX.sub("", c)
            c2 = ORIGIN_PREFIX.sub("", c2)
            c2 = SUFFIX.sub("", c2)
            if c2 in wl:
                return c2
            if c2 == c:
                break
            c = c2
        return h
    return norm


def normalize_herbs(rx: list[str], norm_map, whitelist: set) -> tuple[list[str], list[str]]:
    """把药名按字典映射归一化到白名单；返回 (归一化后, 仍未匹配的)。"""
    out, bad = [], []
    seen = set()
    for h in rx:
        n = norm_map(h)
        if n not in whitelist:
            bad.append(h)
            n = h
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out, bad

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="基座模型路径")
    ap.add_argument("--adapter", default=None, help="LoRA 适配器路径（不传则用基座）")
    ap.add_argument("--ref", default=str(DATA_DIR / "ref_test.jsonl"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--subset", choices=["all", "first", "fu"], default="all",
                    help="first=初诊，fu=复诊；默认全部（评测时再分子集）")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--sample", type=int, default=None,
                    help="按初诊/复诊分层随机抽样 N 条（固定种子，可复现）")
    ap.add_argument("--sample-seed", type=int, default=42)
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--max-new-tokens", type=int, default=320)
    ap.add_argument("--v1-input", action="store_true",
                    help="用 format_input（不含既往处方）；默认 v2")
    ap.add_argument("--constrained", action="store_true", default=False,
                    help="启用 462 白名单约束解码。注意：若从第一个 token 起约束，"
                         "会把【中医诊断】整段屏蔽掉 —— 因此本脚本默认关闭，"
                         "改用【后置归一化+校验】（见 --normalize）；"
                         "确需解码期硬保证时再用。")
    ap.add_argument("--normalize", dest="normalize", action="store_true", default=True,
                    help="后置：把药名按字典归一化到 462 白名单，并统计越界率（默认开）")
    ap.add_argument("--no-normalize", dest="normalize", action="store_false")
    ap.add_argument("--quant", choices=["none", "4bit"], default="none")
    args = ap.parse_args()

    import torch
    from transformers import AutoTokenizer

    from constrained_decode import load_whitelist, make_logits_processor
    from train_sft import load_model_for_inspect

    rows = M.read_jsonl(Path(args.ref))
    if args.subset == "first":
        rows = [r for r in rows if not r.get("prev_rx")]
    elif args.subset == "fu":
        rows = [r for r in rows if r.get("prev_rx")]
    if args.limit:
        rows = rows[: args.limit]
    if args.sample:
        # 分层抽样：按初诊/复诊各自比例抽取，避免抽样改变口径构成
        import random
        rnd = random.Random(args.sample_seed)
        fu = [r for r in rows if r.get("prev_rx")]
        fi = [r for r in rows if not r.get("prev_rx")]
        tot = len(rows)
        n_fu = int(round(args.sample * len(fu) / tot))
        n_fi = args.sample - n_fu
        rnd.shuffle(fu); rnd.shuffle(fi)
        rows = sorted(fi[:n_fi] + fu[:n_fu], key=lambda r: r["rid"])
        print(f"分层抽样 {args.sample} 条（初诊 {len(fi[:n_fi])} / 复诊 {len(fu[:n_fu])}，"
              f"seed={args.sample_seed}）")
    print(f"待预测 {len(rows)} 条（subset={args.subset}）"
          f" | 初诊 {sum(1 for r in rows if not r.get('prev_rx'))}"
          f" / 复诊 {sum(1 for r in rows if r.get('prev_rx'))}")

    # ---- 模型 ----
    print("加载模型…")
    model, tok, used_cls, arch = load_model_for_inspect(
        args.model, args.quant, "bfloat16")
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
        print(f"已挂载 LoRA 适配器：{args.adapter}")
    model.eval()
    print(f"  架构 {arch}（{used_cls}）")

    use_v2 = not args.v1_input
    prompts = build_prompts(rows, tok, use_v2)
    herbs = load_whitelist()

    # ---- 约束解码 ----
    # 关键：用 left padding，使一个 batch 内所有序列的生成起点相同，
    # 于是同一个 prompt_len 对整批都成立，约束解码可以整批启用。
    # （right padding 会让生成起点随长度变化，约束会错位 —— 错位会误屏蔽合法药名。）
    # ⚠️ 必须【无条件】设为 left padding：
    # predict.py 早期版本只在 --constrained 时设置，而该选项默认关闭，
    # 于是批量生成用 right padding —— 短 prompt 的 padding 最多，首个 token 从 pad 位置
    # 预测，导致提前 EOS、输出只有两个标题行（实测 17/64 全空，且全是短输入）。
    # 这不是样本问题：同样本 batch=1 时输出完全正常。
    tok.padding_side = "left"
    print(f"padding_side=left（短 prompt 在 right padding 下会提前 EOS）")
    if args.constrained:
        print(f"约束解码：药名限定在 {len(herbs)} 味白名单内（整批生效）")

    whitelist = set(herbs)
    norm_map = None
    if args.normalize:
        norm_map = make_whitelist_normalizer(herbs)
        print(f"后置归一化：462 白名单 + 炮制/产地/后缀剥离规则")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    n_done = 0
    stat = {"herbs": 0, "bad": 0, "samples_with_bad": 0, "dx_missing": 0}
    with out_path.open("w", encoding="utf-8") as fh:
        for i in range(0, len(prompts), args.batch):
            batch_p = prompts[i:i + args.batch]
            batch_r = rows[i:i + args.batch]
            assert tok.padding_side == "left", "生成前 padding_side 必须是 left"
            enc = tok(batch_p, return_tensors="pt", padding=True).to(model.device)
            gen_kw = dict(max_new_tokens=args.max_new_tokens, do_sample=False,
                          pad_token_id=tok.pad_token_id,
                          eos_token_id=tok.eos_token_id)
            with torch.no_grad():
                out = model.generate(**enc, **gen_kw)
            new = out[:, enc["input_ids"].shape[1]:]
            texts = tok.batch_decode(new, skip_special_tokens=True)
            if args.constrained:
                # 两段式：诊断段自由生成，药方段在约束下重生成。
                # 直接整段约束会把【中医诊断】屏蔽掉（实测踩过）。
                texts = [constrained_second_pass(
                    model, tok, p, t, herbs, args.max_new_tokens) for p, t in zip(batch_p, texts)]
            for r, t in zip(batch_r, texts):
                if args.normalize:
                    dx, rx = M.parse_assistant(t)
                    if not dx:
                        stat["dx_missing"] += 1
                    rx2, bad = normalize_herbs(rx, norm_map, whitelist)
                    stat["herbs"] += len(rx)
                    stat["bad"] += len(bad)
                    if bad:
                        stat["samples_with_bad"] += 1
                    # 重建输出，保持与评测入口一致的格式
                    t = _rebuild_output(dx, rx2)
                fh.write(json.dumps({"rid": r["rid"], "output": t},
                                    ensure_ascii=False) + "\n")
            n_done += len(batch_p)
            if (i // args.batch) % 20 == 0 or n_done == len(prompts):
                el = time.time() - t0
                eta = el / max(1, n_done) * (len(prompts) - n_done)
                print(f"  {n_done}/{len(prompts)}  已用 {el/60:.1f} min  "
                      f"预计还需 {eta/60:.1f} min", flush=True)
    if args.normalize:
        n = max(1, stat["herbs"])
        print(f"[归一化统计] 药名实例 {stat['herbs']}，归一化后仍越界 {stat['bad']} "
              f"= {stat['bad']/n*100:.2f}% | 含越界的样本 {stat['samples_with_bad']}/{n_done} "
              f"| 缺诊断段 {stat['dx_missing']}/{n_done}")
    print(f"✅ 预测已写入 {out_path}（{n_done} 条，用时 {(time.time()-t0)/60:.1f} min）")
    print(f"下一步：python3 src/evaluate.py --ref {args.ref} --pred {out_path}")


if __name__ == "__main__":
    main()
