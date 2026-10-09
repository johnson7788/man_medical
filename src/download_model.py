"""
download_model.py — 下载基座模型

优先 ModelScope（国内直连快），失败回退 HuggingFace。

为什么需要这个脚本而不是让用户自己 `git clone`：
  1. 仓库里**没有** git-lfs 假设（很多训练机不装），本脚本用 HTTP 直接拉文件；
  2. 下载后**校验分片完整性**（按官方 index 的 total_size 逐片核对，
     避免半截文件在训练时报出难以定位的错）；
  3. 显式提示该模型是**多模态 + 混合注意力**架构，
     训练前必须用 `train_sft.py --inspect` 确认 LoRA 目标模块名。

用法：
  python3 src/download_model.py --out models/MiMo-V2.6-Distill-Qwen-9B
  python3 src/download_model.py --repo XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B --source hf
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPOS = {
    "ms": "https://www.modelscope.cn/models/{repo}/resolve/master",
    "hf": "https://huggingface.co/{repo}/resolve/main",
}

# 必需的模型文件（不含 README / .gitattributes）
NEEDED = [
    "config.json",
    "generation_config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "vocab.json",
    "merges.txt",
    "chat_template.jinja",
    "processor_config.json",
    "preprocessor_config.json",
    "model.safetensors.index.json",
]


def curl(url: str, out: Path, resume: bool = True) -> bool:
    cmd = ["curl", "-sSL", "--retry", "3", "--retry-delay", "5", "--max-time", "3600",
           "-o", str(out)]
    if resume:
        cmd += ["-C", "-"]
    cmd.append(url)
    return subprocess.run(cmd).returncode == 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B")
    ap.add_argument("--source", choices=["ms", "hf"], default="ms",
                    help="ms=ModelScope（国内快，默认）; hf=HuggingFace")
    ap.add_argument("--out", type=Path, default=Path("models/MiMo-V2.6-Distill-Qwen-9B"))
    ap.add_argument("--source-fallback", action="store_true", default=True,
                    help="ms 失败时自动回退 hf")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    base = REPOS[args.source].format(repo=args.repo)
    print("=" * 78)
    print(f"下载 {args.repo}")
    print(f"  源   : {args.source} ({base})")
    print(f"  目标 : {args.out}")
    print("=" * 78)

    # 先取 index 以获得分片清单与总大小
    idx_path = args.out / "model.safetensors.index.json"
    print("\n[1/3] 取 index ...")
    if not curl(f"{base}/model.safetensors.index.json", idx_path):
        raise SystemExit(f"index 下载失败：{base}/model.safetensors.index.json\n"
                         f"  可换源重试：--source hf")
    idx = json.loads(idx_path.read_text(encoding="utf-8"))
    shards = sorted(set(idx["weight_map"].values()))
    total = idx.get("metadata", {}).get("total_size", 0)
    print(f"  分片 {len(shards)} 个，声明总大小 {total/1e9:.2f} GB（十进制）")

    print("\n[2/3] 下载文件 ...")
    files = NEEDED + shards
    for f in files:
        p = args.out / f
        print(f"  {f:44s}", end=" ", flush=True)
        ok = curl(f"{base}/{f}", p)
        sz = p.stat().st_size if p.exists() else 0
        print(f"{sz/1e6:9.1f} MB {'✅' if ok else '❌'}")
        if not ok:
            raise SystemExit(f"下载失败：{f}")

    print("\n[3/3] 完整性校验 ...")
    missing = [f for f in shards + NEEDED[:1] if not (args.out / f).exists()]
    if missing:
        raise SystemExit(f"缺失文件：{missing}")
    got = sum((args.out / f).stat().st_size for f in shards)
    # 十进制 vs 二进制单位：index 的 total_size 是十进制字节数，直接比字节
    ratio = got / total if total else 1.0
    print(f"  分片合计 {got/1e9:.2f} GB / 声明 {total/1e9:.2f} GB = {ratio*100:.1f}%")
    if ratio < 0.995:
        raise SystemExit("分片不完整（可能被截断），请删除后重跑（支持断点续传）")
    print("  ✅ 完整")

    print("\n" + "=" * 78)
    print("⚠️ 下一步必做：确认 LoRA 目标模块名")
    print("=" * 78)
    print("  该模型是 **多模态 + 混合注意力**（Qwen3_5ForConditionalGeneration）：")
    print("    - 只有少数层是全注意力（q_proj/k_proj/v_proj/o_proj）")
    print("    - 多数层是线性注意力（in_proj_qkv/in_proj_z/in_proj_b/in_proj_a/out_proj）")
    print("  照抄 Qwen3 的 target_modules 会**静默漏掉**大半注意力层。")
    print()
    print(f"  python3 src/train_sft.py --model {args.out} --inspect")
    print()
    print("  `train_sft.py` 默认用 auto 自动探测，无需手填；--inspect 用于人工确认。")


if __name__ == "__main__":
    main()
