"""
prepare_data.py — Stage 0 数据工程

产出：
  data/train.jsonl / val.jsonl / test.jsonl       T3 联合生成（主任务，患者分组切分）
  data/test_time.jsonl                            T3 时间外推测试集（患者与训练集不重叠）
  data/train_dx_only.jsonl / val_dx_only.jsonl    T1 辨证预热任务（59k，量大 2.2x）
  data/all_records.jsonl                          全量清洗后记录（供 T4/T5 复用）
  data/herb_vocab.json                            512 味药材白名单 + 频次（约束解码用）
  reports/data_audit.md                           清洗审计报告

运行:  python3 src/prepare_data.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import medlib as M
from medlib import DATA_DIR, REPORT_DIR


def emit_split(name: str, records: list[dict], with_target: bool = True) -> None:
    path = DATA_DIR / f"{name}.jsonl"
    M.write_jsonl(path, [M.to_chatml(r) for r in records])
    print(f"  {name:18s} {len(records):6d} 条  -> {path.relative_to(M.ROOT)}")


def emit_ref(name: str, records: list[dict]) -> None:
    """写出评测参考文件（含 rid 与 gold 标签）。

    训练集是 ChatML 格式（messages），评测器需要的是带 rid/gold 的扁平记录，
    两者必须分开：评测器按 rid 与预测文件对齐，不解析训练样本的 prompt。
    """
    rows = [{k: r.get(k) for k in ("rid", "pid", "date", "sex", "age", "cc", "hpi",
                                   "wm_dx", "tcm_dx", "rx", "flags",
                                   "prev_rx", "prev_gap", "visit_ix", "n_visits", "visit_ix_rx", "n_rx")}
            for r in records]
    M.write_jsonl(DATA_DIR / f"ref_{name}.jsonl", rows)
    print(f"  {'ref_' + name:18s} {len(records):6d} 条  (评测参考)")


def to_chatml_dx_only(r: dict) -> dict:
    """T1：只预测中医诊断，不含处方。"""
    dx_lines = ["·".join(t) for t in r["tcm_dx"]]
    return {"messages": [
        {"role": "system", "content": "你是中医男科专家。请根据患者四诊信息给出中医诊断，"
                                      "每行一个诊断，格式为「病名·证型·治法」。"},
        {"role": "user", "content": M.format_input(r)},
        {"role": "assistant", "content": M.DX_HEADER + "\n" + "\n".join(dx_lines)},
    ]}


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(
        description="Stage 0 数据工程。发布模式下用已有的 data/all_records.jsonl，"
                    "无需原始 Excel（原始表含院方数据，不随仓库提供）。")
    ap.add_argument("--from-xlsx", type=Path, default=None,
                    help="原始 Excel 路径（私有）。不给则读 data/all_records.jsonl")
    ap.add_argument("--records", type=Path, default=None,
                    help="发布模式下的输入记录文件，默认 data/all_records.jsonl")
    args = ap.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("Stage 0 数据工程 — 中医男科病历")
    print("=" * 78)

    records_path = args.records or (DATA_DIR / "all_records.jsonl")
    df = None
    if args.from_xlsx or not records_path.exists():
        # ---- 私有模式：从原始 Excel 全量重建 ----
        xlsx = args.from_xlsx
        if xlsx is None:
            raise SystemExit(
                f"既没有 {records_path}，也没有 --from-xlsx。\n"
                "  发布版仓库只提供派生数据 all_records.jsonl；\n"
                "  若你有原始 Excel，请显式指定：--from-xlsx <路径>")
        print(f"\n[1/6] 读取原始表 {xlsx} ...")
        df = M.load_raw(xlsx)
        print(f"  原始 {len(df):,} 行 x {len(df.columns)} 列")

        print("\n[2/6] 清洗与解析 ...")
        records = M.build_records(df)
        # 纵向关联：挂上既往处方（治疗进程），这是比中医诊断更强的预测因子
        M.link_history(records)
    else:
        # ---- 发布模式：复用已发布的派生数据 ----
        print(f"\n[1-2/6] 读取已发布的派生记录 {records_path}（无原始 Excel）")
        records = M.read_jsonl(records_path)
        print(f"  记录 {len(records):,} 条")
        print("  说明：原始 Excel 含院方数据，不随仓库提供；"
              "本模式从 all_records.jsonl 重跑切分与导出，结果与发布版一致。")
    n_fu = sum(1 for r in records if r["has_rx"] and r["prev_rx"])
    n_rx_all = sum(1 for r in records if r["has_rx"])
    print(f"  纵向关联：{n_rx_all:,} 条有处方记录中 {n_fu:,} 条有既往处方"
          f"（复诊 {n_fu/max(1,n_rx_all)*100:.1f}%）")
    flag_c = Counter(f for r in records for f in r["flags"])
    gold = [r for r in records if M.is_gold(r)]
    dx_only = [r for r in records if r["has_dx"] and r["cc"] and not r["has_rx"]]
    print(f"  全量记录      {len(records):6d}")
    print(f"  金标准 T3     {len(gold):6d}  ({len(gold)/len(records)*100:.1f}%)")
    print(f"  仅诊断 T1 池  {len(dx_only):6d}")
    print("  质量标记:", dict(flag_c.most_common()))

    # 硬校验：归一化后任何药名都不得残留 ASCII 数字/字母（剂量后缀必须剥干净）
    for r in records:
        for h in r["rx"]:
            M.assert_clean_herb_name(h)

    # 药名归一化效果（只有私有模式拿得到"归一化前"的原始名）
    vocab = Counter(h for r in records if r["has_rx"] for h in r["rx"])
    if df is not None:
        raw_names = Counter()
        for s_ in df["rx"].dropna():
            for part in str(s_).split(","):
                p = part.strip()
                if p:
                    raw_names[M._DOSE_SUFFIX.sub("", p)] += 1
        print(f"\n  药名唯一数  {len(raw_names)} -> {len(vocab)} "
              f"(归一化合并 {len(raw_names)-len(vocab)} 个变体)")
    else:
        raw_names = None
        print(f"\n  药名唯一数  {len(vocab)}（发布模式无原始名，无法对比归一化前后）")

    # 诊断段解析损耗
    if df is not None:
        n_seg_raw = n_seg_kept = 0
        for s_ in df["tcm_dx"].dropna():
            for seg in str(s_).split(","):
                seg = seg.strip()
                if not seg:
                    continue
                n_seg_raw += 1
                if len([p for p in seg.split("/") if p.strip()]) == 3:
                    n_seg_kept += 1
        print(f"  诊断段      {n_seg_raw:,} -> {n_seg_kept:,} "
              f"(丢弃非3级 {n_seg_raw-n_seg_kept:,})")
    else:
        n_seg_raw = n_seg_kept = None

    print("\n[3/6] 患者分组切分 80/10/10 ...")
    splits = M.split_by_patient(gold)
    for k in ("train", "val", "test"):
        v = splits[k]
        print(f"  {k:5s} {len(v):6d} 样本 / {len({r['pid'] for r in v}):5d} 患者")
    # 泄漏对照：如果按行随机切分会怎样
    import random
    shuffled = gold[:]
    random.Random(M.SEED).shuffle(shuffled)
    n_tr = int(len(shuffled) * 0.8)
    row_te = shuffled[n_tr:]
    overlap = {r["pid"] for r in shuffled[:n_tr]} & {r["pid"] for r in row_te}
    print(f"  ⚠ 对照：若按【行】随机切分，test 中 {len(overlap)} 名患者曾在 train 出现 "
          f"({len(overlap)/max(1,len({r['pid'] for r in row_te}))*100:.1f}% of test 患者) -> 严重泄漏")

    print("\n[4/6] 时间外推切分 (cutoff 2025-07-01, 剔除训练集患者) ...")
    tr_t, te_t = M.split_by_time(gold)
    print(f"  train {len(tr_t):6d} / test {len(te_t):6d}  患者重叠 = 0 (已强制剔除)")

    print("\n[5/6] 写出数据集 ...")
    emit_split("train", splits["train"])
    emit_split("val", splits["val"])
    emit_split("test", splits["test"])
    emit_split("test_time", te_t)
    print()
    # T3+ : 输入额外含既往处方（复诊占 63.9%，照抄上次方 Jaccard 0.539 >> Oracle 0.203）
    for name, recs in (("train", splits["train"]), ("val", splits["val"]),
                       ("test", splits["test"]), ("test_time", te_t)):
        M.write_jsonl(DATA_DIR / f"{name}_v2.jsonl", [M.to_chatml(r, v2=True) for r in recs])
        print(f"  {name + '_v2':18s} {len(recs):6d} 条  (T3+ 含既往处方)")
    print()
    emit_ref("train", splits["train"])
    emit_ref("val", splits["val"])
    emit_ref("test", splits["test"])
    emit_ref("test_time", te_t)
    print()

    # T1 辨证预热集 = 金标准切分 ∪ 无处方池
    # 无处方记录只能进 train，且其患者不得出现在 T3 的 val/test 中，否则会污染 T3 评估。
    eval_pids = {r["pid"] for r in splits["val"]} | {r["pid"] for r in splits["test"]}
    extra = [r for r in dx_only if r["pid"] not in eval_pids
             and not M.FATAL_FLAGS.intersection(r["flags"])]
    t1_train = splits["train"] + extra
    M.write_jsonl(DATA_DIR / "train_dx_only.jsonl", [to_chatml_dx_only(r) for r in t1_train])
    M.write_jsonl(DATA_DIR / "val_dx_only.jsonl", [to_chatml_dx_only(r) for r in splits["val"]])
    print(f"  {'train_dx_only':18s} {len(t1_train):6d} 条  "
          f"(金标准 {len(splits['train']):,} + 无处方 {len(extra):,})  T1 预热")
    skipped = len(dx_only) - len(extra)
    print(f"  {'':18s} 跳过 {skipped:,} 条无处方记录（患者落在 T3 val/test 内，避免污染）")
    if df is not None:
        M.write_jsonl(DATA_DIR / "all_records.jsonl", records)
        print(f"  {'all_records':18s} {len(records):6d} 条  (T4/T5 复用)")
    else:
        print(f"  {'all_records':18s} {len(records):6d} 条  (发布模式：沿用现有文件)")

    # 药材白名单 + 关键药（top100，用于「关键药 Recall」指标与约束解码）
    vocab_sorted = [h for h, _ in sorted(vocab.items(), key=lambda x: (-x[1], x[0]))]
    key_herbs = M.top_herbs(splits["train"], 100)          # 只用训练集统计，避免泄漏
    (DATA_DIR / "herb_vocab.json").write_text(json.dumps({
        "n_unique": len(vocab_sorted),
        "vocab": vocab_sorted,
        "freq": dict(sorted(vocab.items(), key=lambda x: (-x[1], x[0]))),
        "key_herbs_top100_from_train": key_herbs,
        "non_therapeutic": sorted(M.NON_THERAPEUTIC),
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  {'herb_vocab.json':18s} {len(vocab_sorted)} 味 + top100 关键药")

    print("\n[6/6] 审计报告 ...")
    src_line = (f"- 原始: {len(df):,} 行 x {len(df.columns)} 列" if df is not None
                else "- 原始表: **未提供**（发布模式，从 all_records.jsonl 重跑）")
    herb_line = (f"- 药名: {len(raw_names)} 原始 -> **{len(vocab)} 归一化**"
                 if raw_names is not None else f"- 药名: **{len(vocab)}** 归一化后")
    seg_line = (f"- 诊断段: {n_seg_raw:,} -> {n_seg_kept:,} "
                f"(丢弃非3级 {n_seg_raw-n_seg_kept:,})"
                if n_seg_raw is not None else "- 诊断段: 发布模式下不可统计")
    lines = ["# 数据清洗审计报告", "",
             src_line,
             f"- 全量清洗后记录: {len(records):,}",
             f"- **金标准 T3 样本: {len(gold):,}**",
             f"- T1 仅诊断池: {len(dx_only):,}",
             herb_line,
             seg_line, "",
             "## 患者分组切分", "",
             "| 集合 | 样本 | 患者 |", "|---|---|---|"]
    for k in ("train", "val", "test"):
        lines.append(f"| {k} | {len(splits[k]):,} | {len({r['pid'] for r in splits[k]}):,} |")
    lines += ["", "## 质量标记", "", "| 标记 | 数量 |", "|---|---|"]
    for f, c in flag_c.most_common():
        lines.append(f"| {f} | {c:,} |")
    lines += ["", "## 泄漏对照（必须写进评审材料）", "",
              f"- 患者分组切分下，test 患者与 train 零重叠。",
              f"- 若改为按行随机切分，test 中有 **{len(overlap):,}** 名患者已在 train 出现"
              f"（占 test 患者 {len(overlap)/max(1,len({r['pid'] for r in row_te}))*100:.1f}%）。",
              "- 计划 §1.6 实测：两种口径下基线 Jaccard 相差 +88%（0.164 -> 0.308）。", "",
              "## 剂量字段", "",
              "- 原始 410,554 条剂量条目中 99.7% 为常量 `1.000`，无信息量，**已整体丢弃**。",
              "- 因此模型不生成剂量；部署时由外部《中国药典》剂量表填充。"]
    (REPORT_DIR / "data_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  reports/data_audit.md")

    print("\n" + "=" * 78)
    print(f"完成。金标准 {len(gold):,} 条，切分 train/val/test = "
          f"{len(splits['train']):,}/{len(splits['val']):,}/{len(splits['test']):,}")
    print("=" * 78)


if __name__ == "__main__":
    main()
