"""
evaluate.py — 统一评测器

设计要点：
  * 评测入口只有 M.parse_assistant —— 模型输出与基线输出走完全相同的解析路径，
    保证「基线分数」与「模型分数」可比。
  * 主指标是集合级 Jaccard / P / R / F1，不用完全匹配率（计划 §1.6 实测：诚实
    口径下完全匹配率 = 0%，用它评估必然误判模型无效）。
  * 支持按子集报告：处方长度异常、处方文本重复、患者多次就诊 —— 区分「记忆」与「泛化」。

用法：
  # 评测一个预测文件（每行 {"rid": int, "output": str}）
  python3 src/evaluate.py --ref data/test.jsonl --pred preds/model.jsonl

  # 同时评测时间外推集
  python3 src/evaluate.py --ref data/test.jsonl --pred preds/model.jsonl \
                          --ref-time data/test_time.jsonl --pred-time preds/model_time.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import medlib as M
from medlib import DATA_DIR, REPORT_DIR

DIAG_FIELDS = {"病名": 0, "证型": 1, "治法": 2}


# ---------------------------------------------------------------------------
# 指标原语
# ---------------------------------------------------------------------------

def micro_prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def macro_f1(preds: list[set], refs: list[set], min_support: int = 20) -> tuple[float, int]:
    """按标签计算 F1 后取平均，只纳入支持度 >= min_support 的标签。

    长尾标签（如出现 <5 次的稀有药/罕见证型）F1 必为 0，纳入 macro 会把指标
    拖成噪声；计划 §1.4 明确要求剔除。
    """
    support = Counter(lbl for ref in refs for lbl in ref)
    keep = {lbl for lbl, n in support.items() if n >= min_support}
    if not keep:
        return 0.0, 0
    tp = Counter(); fp = Counter(); fn = Counter()
    for p, g in zip(preds, refs):
        p, g = p & keep, g & keep
        for x in p & g: tp[x] += 1
        for x in p - g: fp[x] += 1
        for x in g - p: fn[x] += 1
    scores = []
    for lbl in keep:
        _, _, f = micro_prf(tp[lbl], fp[lbl], fn[lbl])
        scores.append(f)
    return sum(scores) / len(scores), len(keep)


def diagnosis_metrics(pred_dx: list[list[tuple]], ref_dx: list[list[tuple]]) -> dict:
    out: dict = {}
    for name, idx in DIAG_FIELDS.items():
        pp = [{(t[idx] if len(t) > idx else "") for t in p} - {""} for p in pred_dx]
        rr = [{(t[idx] if len(t) > idx else "") for t in g} - {""} for g in ref_dx]
        tp = sum(len(a & b) for a, b in zip(pp, rr))
        fp = sum(len(a - b) for a, b in zip(pp, rr))
        fn = sum(len(b - a) for a, b in zip(pp, rr))
        p, r, f = micro_prf(tp, fp, fn)
        out[f"{name}_P"] = p
        out[f"{name}_R"] = r
        out[f"{name}_F1"] = f
        if name == "证型":
            mf, n = macro_f1(pp, rr)
            out["证型_macroF1"] = mf
            out["证型_macroF1_类数"] = n
    # 诊断三元组完全一致率（整行诊断集合）
    out["诊断集合完全一致"] = sum(
        1 for p, g in zip(pred_dx, ref_dx) if set(p) == set(g) and g
    ) / max(1, len(ref_dx))
    return out


def rx_metrics(pred_rx: list[list[str]], ref_rx: list[list[str]],
               key_herbs: set[str], vocab: set[str]) -> dict:
    out: dict = {}
    ps = [set(x) for x in pred_rx]
    gs = [set(x) for x in ref_rx]

    # 集合级 micro P/R/F1
    tp = sum(len(a & b) for a, b in zip(ps, gs))
    fp = sum(len(a - b) for a, b in zip(ps, gs))
    fn = sum(len(b - a) for a, b in zip(ps, gs))
    p, r, f = micro_prf(tp, fp, fn)
    out["处方_P"] = p
    out["处方_R"] = r
    out["处方_F1"] = f

    # 主指标：逐样本 Jaccard
    jac = [len(a & b) / len(a | b) if (a | b) else 0.0 for a, b in zip(ps, gs)]
    out["处方_Jaccard"] = sum(jac) / max(1, len(jac))
    out["处方_Jaccard中位数"] = sorted(jac)[len(jac) // 2] if jac else 0.0
    out["处方_完全匹配"] = sum(1 for a, b in zip(ps, gs) if a == b and b) / max(1, len(gs))

    # 关键药召回（只看 top100 高频药，临床意义更强）
    kh = key_herbs
    hit = sum(len((a & kh) & (b & kh)) for a, b in zip(ps, gs))
    tot = sum(len(b & kh) for b in gs)
    out["关键药_Recall"] = hit / tot if tot else 0.0

    # 结构健康度
    empty = sum(1 for a in ps if not a) / max(1, len(ps))
    illegal = sum(len(a - vocab) for a in ps) / max(1, sum(len(a) for a in ps))
    out["空方率"] = empty
    out["非法药名率"] = illegal
    out["重复药率"] = sum(
        len(x) - len(set(x)) for x in pred_rx
    ) / max(1, sum(len(x) for x in pred_rx))
    out["预测平均药味数"] = sum(len(a) for a in ps) / max(1, len(ps))
    out["真实平均药味数"] = sum(len(b) for b in gs) / max(1, len(gs))
    return out


def safety_metrics(pred_rx: list[list[str]], ref_rx: list[list[str]]) -> dict:
    """安全性检查：毒性药越界 + 寒热矛盾。

    说明：这是粗筛，不是临床判定。上线前仍需人工盲评（G4）。
    """
    TOXIC = {"细辛", "黑顺片", "制川乌", "制草乌", "麻黄", "法半夏", "姜半夏",
             "清半夏", "全蝎", "蜈蚣", "川楝子", "罂粟壳", "雄黄", "朱砂"}
    COLD = {"石膏", "知母", "黄连", "黄芩片", "黄柏", "栀子", "龙胆", "金银花",
            "蒲公英", "连翘", "生地黄", "玄参", "淡竹叶"}
    HOT = {"黑顺片", "制川乌", "肉桂", "干姜", "桂枝", "细辛", "麻黄", "制吴茱萸"}
    out: dict = {}
    toxic_n = sum(1 for a in pred_rx if TOXIC & set(a))
    out["含毒性药样本率"] = toxic_n / max(1, len(pred_rx))
    out["含毒性药样本率_参考"] = sum(1 for b in ref_rx if TOXIC & set(b)) / max(1, len(ref_rx))
    contra = sum(1 for a in pred_rx if (COLD & set(a)) and (HOT & set(a)))
    out["寒热同用样本率"] = contra / max(1, len(pred_rx))
    out["寒热同用样本率_参考"] = sum(
        1 for b in ref_rx if (COLD & set(b)) and (HOT & set(b))
    ) / max(1, len(ref_rx))
    # 超过 40 味（大复方异常）
    out["药味数>40率"] = sum(1 for a in pred_rx if len(set(a)) > 40) / max(1, len(pred_rx))
    return out


# ---------------------------------------------------------------------------
# 评测主流程
# ---------------------------------------------------------------------------

def subset_masks(refs: list[dict]) -> dict[str, list[bool]]:
    """构造子集掩码：区分记忆与泛化，以及初诊/复诊两个难度区间。

    初诊 vs 复诊是本数据集最重要的分界：复诊占 ~64%，且「上次处方」的预测力
    （照抄 Jaccard 0.539）远超整套中医诊断（Oracle 上界 0.203）。
    两者难度差一个量级，混在一起报平均数会掩盖真相。
    """
    rx_text = [" ".join(r["rx"]) for r in refs]
    cnt = Counter(rx_text)
    by_pid: dict[str, int] = defaultdict(int)
    for r in refs:
        by_pid[r["pid"]] += 1
    return {
        "全部": [True] * len(refs),
        "初诊(无既往方)": [not r.get("prev_rx") for r in refs],
        "复诊(有既往方)": [bool(r.get("prev_rx")) for r in refs],
        "处方重复子集": [cnt[t] > 1 for t in rx_text],
        "处方非重复子集": [cnt[t] == 1 for t in rx_text],
        "患者多次就诊子集": [by_pid[r["pid"]] > 1 for r in refs],
        "患者单次就诊子集": [by_pid[r["pid"]] == 1 for r in refs],
        "常规药味数(5-40)": ["rx_len_outlier" not in r.get("flags", []) for r in refs],
        "异常药味数(<5或>40)": ["rx_len_outlier" in r.get("flags", []) for r in refs],
        "主诉非现病史前缀": ["hpi_has_cc_prefix" not in r.get("flags", [])
                       and "cc_eq_hpi" not in r.get("flags", []) for r in refs],
    }


def evaluate(ref_path: Path, pred_path: Path, train_records: list[dict],
             label: str = "test") -> dict:
    refs = M.read_jsonl(ref_path)
    preds = M.read_jsonl(pred_path)
    pmap = {p["rid"]: p.get("output", "") for p in preds}

    missing = [r["rid"] for r in refs if r["rid"] not in pmap]
    if missing:
        print(f"  ⚠ 预测缺 {len(missing)} 条（{label}），按空输出计：{missing[:5]}")

    vocab = set(M.top_herbs(train_records, 10**9))
    key = set(M.top_herbs(train_records, 100))

    pdx, prx = [], []
    for r in refs:
        d, x = M.parse_assistant(pmap.get(r["rid"], ""))
        pdx.append(d)
        prx.append(x)
    rdx = [ [tuple(t) for t in r["tcm_dx"]] for r in refs ]
    rrx = [ r["rx"] for r in refs ]

    res: dict = {"n": len(refs), "label": label}
    res.update(diagnosis_metrics(pdx, rdx))
    res.update(rx_metrics(prx, rrx, key, vocab))
    res.update(safety_metrics(prx, rrx))

    # 子集报告
    masks = subset_masks(refs)
    sub: dict = {}
    for name, m in masks.items():
        n = sum(m)
        if n < 30:
            continue
        sp = [x for x, keep in zip(prx, m) if keep]
        sr = [x for x, keep in zip(rrx, m) if keep]
        jac = [len(set(a) & set(b)) / len(set(a) | set(b)) if (set(a) | set(b)) else 0.0
               for a, b in zip(sp, sr)]
        sub[name] = {
            "n": n,
            "Jaccard": sum(jac) / len(jac),
            "Recall": sum(len(set(a) & set(b)) for a, b in zip(sp, sr))
                      / max(1, sum(len(set(b)) for b in sr)),
            "完全匹配": sum(1 for a, b in zip(sp, sr) if set(a) == set(b) and b) / n,
        }
    res["subsets"] = sub
    return res


def print_report(res: dict) -> None:
    lab = res["label"]
    print(f"\n{'='*78}\n评测结果 [{lab}]  样本数 = {res['n']:,}\n{'='*78}")
    print("\n【中医诊断】(micro)")
    for name in DIAG_FIELDS:
        print(f"  {name}  P={res[f'{name}_P']:.3f}  R={res[f'{name}_R']:.3f}  F1={res[f'{name}_F1']:.3f}")
    print(f"  证型 macro-F1 = {res['证型_macroF1']:.3f} (纳入 {res['证型_macroF1_类数']} 类, 支持度>=20)")
    print(f"  诊断集合完全一致率 = {res['诊断集合完全一致']*100:.2f}%")

    print("\n【中药处方】")
    print(f"  ★ Jaccard(主指标) = {res['处方_Jaccard']:.3f}   中位数 = {res['处方_Jaccard中位数']:.3f}")
    print(f"  micro  P={res['处方_P']:.3f}  R={res['处方_R']:.3f}  F1={res['处方_F1']:.3f}")
    print(f"  ★ 关键药 Recall(top100) = {res['关键药_Recall']:.3f}")
    print(f"  完全匹配率 = {res['处方_完全匹配']*100:.2f}%  (仅观察项，不作 Gate)")
    print(f"  平均药味数 预测={res['预测平均药味数']:.1f} 真实={res['真实平均药味数']:.1f}")

    print("\n【结构健康度】")
    print(f"  空方率={res['空方率']*100:.2f}%  非法药名率={res['非法药名率']*100:.2f}%  "
          f"重复药率={res['重复药率']*100:.2f}%")

    print("\n【安全性】(粗筛，上线前仍需人工盲评)")
    print(f"  含毒性药样本率 预测={res['含毒性药样本率']*100:.1f}%  参考={res['含毒性药样本率_参考']*100:.1f}%")
    print(f"  寒热同用样本率 预测={res['寒热同用样本率']*100:.1f}%  参考={res['寒热同用样本率_参考']*100:.1f}%")
    print(f"  药味数>40 率   预测={res['药味数>40率']*100:.2f}%")

    print("\n【子集分析】(区分记忆与泛化)")
    print(f"  {'子集':28s} {'n':>6s} {'Jaccard':>9s} {'Recall':>8s} {'完全匹配':>9s}")
    for name, v in res["subsets"].items():
        print(f"  {name:28s} {v['n']:6d} {v['Jaccard']:9.3f} {v['Recall']:8.3f} "
              f"{v['完全匹配']*100:8.2f}%")


def gate_check(res: dict) -> None:
    """按 medlib.GATES 判定（唯一定义处，与基线报告锚点一致）。"""
    print(f"\n【Gate 判定】[{res['label']}]")
    for key, spec in M.GATES.items():
        if key not in res:
            continue
        v = res[key]
        ok = v >= spec["必达"] if spec["方向"] == ">=" else v <= spec["必达"]
        tgt = v >= spec["目标"] if spec["方向"] == ">=" else v <= spec["目标"]
        scale = 100 if key.endswith("率") else 1
        unit = "%" if scale == 100 else ""
        mark = "✅" if ok else "❌"
        star = "  (达标)" if tgt else ""
        print(f"  {mark} {spec['名称']:30s} 必达 {spec['方向']}"
              f"{spec['必达']*scale:g}{unit}  实际 {v*scale:.2f}{unit}{star}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", type=Path, required=True)
    ap.add_argument("--pred", type=Path, required=True)
    ap.add_argument("--ref-time", type=Path, default=None)
    ap.add_argument("--pred-time", type=Path, default=None)
    # 默认只用【训练集】统计词表与关键药，避免把测试集出现过的药算进白名单（轻微泄漏）
    ap.add_argument("--train", type=Path, default=DATA_DIR / "ref_train.jsonl")
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    # 参考统计只用训练集（避免把测试集出现过的药算进白名单，造成轻微泄漏）
    train_records = [r for r in M.read_jsonl(args.train) if r.get("rx")]
    print(f"参考统计口径来源: {args.train} ({len(train_records):,} 条有处方记录)")

    all_res = []
    res = evaluate(args.ref, args.pred, train_records, label=args.ref.stem)
    print_report(res); gate_check(res)
    all_res.append(res)

    if args.ref_time and args.pred_time:
        res_t = evaluate(args.ref_time, args.pred_time, train_records,
                         label=args.ref_time.stem)
        print_report(res_t); gate_check(res_t)
        d = res["处方_Jaccard"] - res_t["处方_Jaccard"]
        rel = d / res["处方_Jaccard"] if res["处方_Jaccard"] else 0
        print(f"\n【G5 时效漂移】随机切分 Jaccard={res['处方_Jaccard']:.3f} vs "
              f"时间外推={res_t['处方_Jaccard']:.3f}  相对下降 {rel*100:.1f}%  "
              f"{'✅' if rel <= M.G5_MAX_DROP else '❌'} <= {M.G5_MAX_DROP*100:.0f}%")
        all_res.append(res_t)

    if args.json_out:
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(all_res, ensure_ascii=False, indent=1),
                                 encoding="utf-8")
        print(f"\nJSON 结果 -> {args.json_out}")


if __name__ == "__main__":
    main()
