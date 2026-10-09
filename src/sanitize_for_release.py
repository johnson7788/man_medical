"""
sanitize_for_release.py — 开源发布前的数据脱敏

背景（实测，见 docs/07_开源前必读.md）：
  对本项目 67,073 条记录的自由文本做了 PHI 扫描，结论是：
    - **无患者身份 PHI**：手机号 0、身份证 0、座机/分机 0、社交账号 0、英文姓名 0；
      "患者/姓名" 的 573 次命中经人工逐条核对，全部是 `患者自述` / `患者要求` 这类正常表述。
    - **患者 ID 已哈希**：25,103 个 pid 全部为 `P`+12 位 hex，无一残留原始病历号。
    - ⚠️ **但存在机构可识别性**：1,189 条（1.77%）记录含医院名，计 234 个不同名称，
      且高度集中于某几家医院 —— 足以推断数据来源机构。

  因此发布前对**医院名做归一化**：保留临床语义（"外院做过什么手术/检查"），
  去掉机构身份。日期按用户决定**不平移**（保留完整时间信息以支持时效性评估）。

设计取舍：
  - 只替换 `<中文名> + 医院|卫生院|诊所|门诊部` 这段，不动其余任何字符；
  - 统一替换为「外院」，因为它同时适配所有出现的句式：
      `就诊西苑医院` → `就诊外院`
      `北大医院后尿道断裂术后` → `外院后尿道断裂术后`
      `人民医院口服前列舒通胶囊` → `外院口服前列舒通胶囊`
  - **幂等**：已归一化的文本再跑一次不产生变化（"外院" 不再匹配模式）。

⚠️ 重要副作用：本项目的已训练模型是在**未脱敏**文本上训练的。
  医院名出现在 1.77% 记录的 hpi 里，脱敏后这些文本会变，
  因此"用发布数据 + 发布脚本"**无法逐位复现**已发布模型的权重与个别预测。
  两条出路见 docs/07_开源前必读.md §4：(a) 在文档中如实声明；
  (b) 用脱敏后的数据重训一次（约 3 小时），使数据与模型完全自洽。

用法:
  python3 src/sanitize_for_release.py --check      # 只扫描，不写文件
  python3 src/sanitize_for_release.py --apply      # 就地脱敏 data/*.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import medlib as M

DATA = M.DATA_DIR
REPORT_DIR = M.ROOT / "reports"

# 与 PHI 扫描一致的机构名模式
ORG_RE = re.compile(r"[\u4e00-\u9fff]{2,10}(?:医院|卫生院|诊所|门诊部)")
PLACEHOLDER = "外院"
TEXT_FIELDS = ("cc", "hpi")          # 只有自由文本需要处理
# 这些字段是药名/诊断，不含机构名，但一并扫描以便发现意外命中
SCAN_FIELDS = ("cc", "hpi", "wm_dx")


def sanitize_text(text: str) -> str:
    """把机构名替换为「外院」。幂等。"""
    if not text:
        return text
    return ORG_RE.sub(PLACEHOLDER, text)


def sanitize_record(rec: dict) -> tuple[dict, int]:
    """就地处理一条记录，返回 (记录, 替换次数)。

    必须同时覆盖两种数据形态，早期版本只处理了前者，导致训练文件漏脱敏：
      ① 结构化记录：cc / hpi / wm_dx 等顶层字段
      ② **ChatML 记录**：文本嵌在 messages[i]["content"] 里
         （train*.jsonl / val*.jsonl / test*.jsonl / *_v2.jsonl 都是这种形态）
    """
    n = 0

    def _do(v):
        nonlocal n
        if isinstance(v, str) and v:
            new = sanitize_text(v)
            if new != v:
                n += len(ORG_RE.findall(v))
            return new
        return v

    # ① 结构化字段
    for f in TEXT_FIELDS:
        v = rec.get(f)
        if isinstance(v, str):
            rec[f] = _do(v)
        elif isinstance(v, list):
            rec[f] = [(_do(x) if isinstance(x, str) else x) for x in v]
    if isinstance(rec.get("wm_dx"), list):
        rec["wm_dx"] = [(_do(x) if isinstance(x, str) else x) for x in rec["wm_dx"]]

    # ② ChatML
    msgs = rec.get("messages")
    if isinstance(msgs, list):
        for m in msgs:
            if isinstance(m, dict) and isinstance(m.get("content"), str):
                m["content"] = _do(m["content"])
    return rec, n


def iter_jsonl_files(root: Path) -> list[Path]:
    return sorted(p for p in root.glob("*.jsonl"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="就地脱敏（会先备份到 data/.pre_sanitize/）")
    ap.add_argument("--check", action="store_true", help="只扫描")
    ap.add_argument("--dir", type=Path, default=DATA)
    args = ap.parse_args()
    if not (args.apply or args.check):
        args.check = True

    files = iter_jsonl_files(args.dir)
    print("=" * 78)
    print(f"机构名归一化（{'APPLY' if args.apply else 'CHECK'}）—— {len(files)} 个 jsonl")
    print("=" * 78)

    total_repl = 0
    total_rec = 0
    names = Counter()
    per_file: list[tuple[str, int, int]] = []

    if args.apply:
        backup = args.dir / ".pre_sanitize"
        backup.mkdir(exist_ok=True)
        print(f"备份目录: {backup}")

    for path in files:
        rows = M.read_jsonl(path)
        n_repl = 0
        n_hit = 0
        out_rows = []
        for r in rows:
            # 先统计原始机构名（用于报告）：同时覆盖结构化字段与 ChatML
            for f in SCAN_FIELDS:
                v = r.get(f)
                if isinstance(v, str):
                    for m in ORG_RE.findall(v):
                        names[m] += 1
                elif isinstance(v, list):
                    for x in v:
                        if isinstance(x, str):
                            for m in ORG_RE.findall(x):
                                names[m] += 1
            for m in (r.get("messages") or []):
                if isinstance(m, dict) and isinstance(m.get("content"), str):
                    for mm in ORG_RE.findall(m["content"]):
                        names[mm] += 1
            r2, k = sanitize_record(dict(r))
            if k:
                n_hit += 1
                n_repl += k
            out_rows.append(r2)
        total_repl += n_repl
        total_rec += n_hit
        per_file.append((path.name, n_hit, n_repl))
        if args.apply and n_repl:
            shutil.copy2(path, backup / path.name)
            with path.open("w", encoding="utf-8") as fh:
                for r in out_rows:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n{'文件':38s} {'命中记录':>8s} {'替换处数':>8s}")
    for name, nh, nr in per_file:
        if nr:
            print(f"  {name:36s} {nh:>8d} {nr:>8d}")
    print(f"\n合计：命中 {total_rec:,} 条记录，替换 {total_repl:,} 处机构名")
    print(f"不同机构名 {len(names)} 个；出现最多的 10 个：")
    for k, v in names.most_common(10):
        print(f"    {v:5d}  {k}")

    # 幂等性自检
    if args.apply:
        # 幂等自检：脱敏后再读一遍，应当 0 处改动
        left = 0
        for p in files:
            for r in M.read_jsonl(p):
                _, k = sanitize_record(dict(r))
                left += k
        print(f"\n幂等自检：脱敏后重新扫描，仍会被改动的 = {left}（应为 0）")

    # 写报告
    if args.apply:
        REPORT_DIR.mkdir(exist_ok=True)
        lines = ["# 脱敏报告（机构名归一化）\n",
                 f"- 处理文件：{len(files)} 个 jsonl",
                 f"- 命中记录：{total_rec:,} 条",
                 f"- 替换处数：{total_repl:,} 处",
                 f"- 涉及不同机构名：{len(names)} 个",
                 f"- 替换目标：统一为「{PLACEHOLDER}」",
                 "", "## 出现最多的机构名", ""]
        lines += [f"- `{k}` × {v}" for k, v in names.most_common(30)]
        lines += ["", "> 备份：`data/.pre_sanitize/`（不进入版本库）",
                  "> 日期**未**平移，保留完整时间信息以支持时效性评估。"]
        (REPORT_DIR / "sanitize_report.md").write_text("\n".join(lines), encoding="utf-8")
        print(f"\n✅ 报告 -> {REPORT_DIR/'sanitize_report.md'}")


if __name__ == "__main__":
    main()
