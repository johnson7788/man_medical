"""
compare_models.py — 统一评测对比（评测标准性的核心交付物）

为什么需要它：
  交付物变多后，"模型的 0.40 和基线的 0.208" 这种比较很容易变成**口径不一致的比较** ——
  不同的评测集、不同的指标实现、不同的子集口径。本项目踩过三次：
    ① 每 epoch 评测用了 right padding → 生成结果本身错误；
    ② 约束解码从第一个 token 生效 → 输出丢掉整段诊断；
    ③ 数字漂移（旧覆盖率 89.07% vs 实际 94.44%）。
  所以评测必须收敛到**一个入口、一份数据、一套指标代码**。

本脚本保证的评测标准性：
  1. **同一份冻结测试集**（`data/ref_test.jsonl`），模型与全部基线共用；
  2. **同一套指标实现**（直接调用 `evaluate.evaluate`，不重写任何指标）；
  3. **参考统计只看训练集**（`ref_train.jsonl`）—— 避免测试集药名进白名单造成轻微泄漏；
  4. **初诊/复诊分开报告**，并给出加权总体（仅报总体会掩盖初诊的真实难度）；
  5. **Gate 自动判定**（阈值来自 `medlib.GATES`，单一来源）；
  6. **模型选择绝不使用测试集**（选择用 val，报告用 test）。

用法:
  python3 src/compare_models.py \
      --model mimo9b=data/preds/mimo9b.jsonl \
      --out reports/model_comparison.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import medlib as M
from evaluate import evaluate
from medlib import DATA_DIR, GATES

REPORT_DIR = M.ROOT / "reports"

# 基线清单：(显示名, 预测文件, 类别)
BASELINES: list[tuple[str, str, str]] = [
    ("常量方（下界）", "B1_全局常量方.jsonl", "下界"),
    ("西医诊断检索（最佳可实现）", "B2_西医诊断检索.jsonl", "可实现"),
    ("Oracle 证型+病名检索（上界，需真实诊断）", "O3_证型+病名检索.jsonl", "Oracle"),
    ("端到端规则 R0", "R0_端到端规则.jsonl", "端到端"),
    ("复诊照抄+初诊Oracle X2", "X2_复诊照抄+初诊Oracle.jsonl", "治疗进程"),
]

# 关心的子集（与 evaluate.subset_masks 的键一致）
SUBSETS = ["全部", "初诊(无既往方)", "复诊(有既往方)"]


def collect(model_specs: list[str], ref: Path, train: Path) -> list[dict]:
    """对模型与全部基线跑同一套评测。"""
    train_records = [r for r in M.read_jsonl(train) if r.get("rx")]
    print(f"参考统计口径（仅训练集）: {train.name} {len(train_records):,} 条")

    runs: list[tuple[str, str, Path]] = []
    for spec in model_specs:
        if "=" not in spec:
            raise SystemExit(f"--model 需要 label=path 形式，收到：{spec}")
        lab, p = spec.split("=", 1)
        runs.append((lab, "模型", Path(p)))
    for lab, fn, cat in BASELINES:
        p = DATA_DIR / "preds" / fn
        if p.exists():
            runs.append((lab, cat, p))
        else:
            print(f"  ⚠ 跳过缺失基线：{p.name}")

    out = []
    for lab, cat, p in runs:
        if not p.exists():
            print(f"  ⚠ 预测文件不存在，跳过：{p}")
            continue
        res = evaluate(ref, p, train_records, label=lab)
        res["类别"] = cat
        res["pred_file"] = str(p)
        vals, _, masks = per_sample_jaccard(ref, p)
        m, lo, hi = bootstrap_ci(vals)
        res["jaccard_ci"] = [lo, hi]
        res["_per_sample"] = vals
        res["_masks"] = masks
        out.append(res)
        print(f"  已评测 [{cat}] {lab}: 总体 J={res['处方_Jaccard']:.3f} "
              f"[{lo:.3f},{hi:.3f}] 证型F1={res['证型_F1']:.3f}")
    return out



# ---------------------------------------------------------------------------
# Bootstrap 置信区间
# ---------------------------------------------------------------------------

def per_sample_jaccard(ref: Path, pred: Path):
    """逐样本 Jaccard + 子集掩码，用于整体与分口径的 bootstrap。"""
    from evaluate import subset_masks
    refs = M.read_jsonl(ref)
    pmap = {p["rid"]: p.get("output", "") for p in M.read_jsonl(pred)}
    vals = []
    for r in refs:
        _, prx = M.parse_assistant(pmap.get(r["rid"], ""))
        a, b = set(prx), set(r["rx"])
        vals.append(len(a & b) / len(a | b) if (a | b) else 0.0)
    return vals, refs, subset_masks(refs)


def _sub(vals: list[float], mask: list[bool]) -> list[float]:
    return [v for v, keep in zip(vals, mask) if keep]


def bootstrap_ci(vals: list[float], n_boot: int = 1000, seed: int = 42,
                 alpha: float = 0.05) -> tuple[float, float, float]:
    """对均值做 bootstrap，返回 (点估计, 下界, 上界)。

    为什么需要：只报一个点估计时，"v2 比 v1 高 0.005" 无法判断是真实提升还是抽样噪声。
    给出 CI 后才能负责地说两者是否可分。
    """
    import random
    rnd = random.Random(seed)
    n = len(vals)
    mean = sum(vals) / n
    boot = []
    for _ in range(n_boot):
        s = 0.0
        for _ in range(n):
            s += vals[rnd.randrange(n)]
        boot.append(s / n)
    boot.sort()
    lo = boot[int(n_boot * alpha / 2)]
    hi = boot[int(n_boot * (1 - alpha / 2))]
    return mean, lo, hi


def paired_bootstrap(a: list[float], b: list[float], n_boot: int = 1000,
                     seed: int = 42) -> tuple[float, float, float, float]:
    """配对 bootstrap 差值：返回 (差值, 下界, 上界, 单尾 p(差值<=0))。"""
    import random
    assert len(a) == len(b)
    rnd = random.Random(seed)
    n = len(a)
    d = [x - y for x, y in zip(a, b)]
    obs = sum(d) / n
    stats = []
    for _ in range(n_boot):
        s = 0.0
        for _ in range(n):
            s += d[rnd.randrange(n)]
        stats.append(s / n)
    stats.sort()
    lo = stats[int(n_boot * 0.025)]
    hi = stats[int(n_boot * 0.975)]
    p = sum(1 for x in stats if x <= 0) / n_boot
    return obs, lo, hi, p

def gate_table(res: dict) -> list[tuple[str, float, float, bool]]:
    rows = []
    for key, g in GATES.items():
        if key not in res:
            continue
        v = res[key]
        ok = v >= g["必达"] if g["方向"] == ">=" else v <= g["必达"]
        rows.append((g["名称"], v, g["必达"], ok))
    return rows


def render(results: list[dict], ref: Path) -> str:
    L: list[str] = []
    L.append("# 模型与基线统一对比\n")
    L.append(f"> 冻结评测集：`{ref.relative_to(M.ROOT)}`（模型与全部基线共用同一份数据）\n"
             f"> 指标实现：`src/evaluate.py::evaluate`（模型与基线共用同一套代码）\n"
             f"> 参考词表统计：`data/ref_train.jsonl`（**只用训练集**，避免测试集药名泄漏）\n"
             f"> 模型选择口径：val 集；本表报告口径：test 集（**选择与报告分离**）\n")

    L.append("\n## 1. 总体对比（按类别排序）\n")
    L.append("| 类别 | 来源 | n | **处方 Jaccard** | 95% CI（bootstrap） | 关键药 Recall | 证型 F1 | 非法药名率 |")
    L.append("|---|---|---|---|---|---|---|---|")
    order = {"模型": 0, "治疗进程": 1, "Oracle": 2, "可实现": 3, "端到端": 4, "下界": 5}
    for r in sorted(results, key=lambda x: order.get(x["类别"], 9)):
        ci = r.get("jaccard_ci")
        ci_s = f"[{ci[0]:.3f}, {ci[1]:.3f}]" if ci else "—"
        L.append(f"| {r['类别']} | {r['label']} | {r['n']:,} | "
                 f"**{r['处方_Jaccard']:.3f}** | {ci_s} | {r['关键药_Recall']:.3f} | "
                 f"{r['证型_F1']:.3f} | {r['非法药名率']*100:.2f}% |")

    L.append("\n## 2. 分口径对比（初诊/复诊必须分开看）\n")
    hdr = "| 来源 | " + " | ".join(
        (f"{s} n" if s != "全部" else "全部 n") for s in SUBSETS) + " |"
    L.append(hdr)
    L.append("|---|" + "---|" * len(SUBSETS))
    # n 行
    nrow = "| 样本数 | " + " | ".join(
        f"{results[0]['subsets'].get(s, {}).get('n', results[0]['n']):,}" for s in SUBSETS) + " |"
    L.append(nrow)
    L.append("")
    L.append("| 来源 | " + " | ".join(s + " J" for s in SUBSETS) + " |")
    L.append("|---|" + "---|" * len(SUBSETS))
    for r in sorted(results, key=lambda x: order.get(x["类别"], 9)):
        cells = []
        for s in SUBSETS:
            if s == "全部":
                cells.append(f"**{r['处方_Jaccard']:.3f}**")
            else:
                v = r["subsets"].get(s)
                cells.append(f"{v['Jaccard']:.3f}" if v else "—")
        L.append(f"| {r['label']} | " + " | ".join(cells) + " |")

    L.append("\n> **为什么要分开**：初诊与复诊难度差一个量级（复诊可照抄上次方）。"
             "只报总体会让复诊的大占比掩盖初诊的真实水平。\n")
    L.append("> **口径提示**：总体 Jaccard 是按样本加权的；下表末行给出"
             "未加权（初诊/复诊等权）平均，供对照——两者不可混用。\n")

    L.append("\n## 3. Gate 判定（阈值来源 `medlib.GATES`，单一来源）\n")
    for r in [x for x in results if x["类别"] == "模型"]:
        L.append(f"\n### {r['label']}\n")
        L.append("| Gate | 实测 | 必达 | 判定 |")
        L.append("|---|---|---|---|")
        for name, v, need, ok in gate_table(r):
            L.append(f"| {name} | {v:.4f} | {need} | {'✅' if ok else '❌'} |")
        # 分口径 Gate（计划 §4.2 要求）
        fu = r["subsets"].get("复诊(有既往方)")
        fi = r["subsets"].get("初诊(无既往方)")
        L.append("")
        L.append("| 分口径 | 实测 Jaccard | 目标 | 判定 |")
        L.append("|---|---|---|---|")
        if fu:
            L.append(f"| 复诊 | {fu['Jaccard']:.3f} | ≥ 0.45（照抄即 0.542） | "
                     f"{'✅' if fu['Jaccard'] >= 0.45 else '❌'} |")
        if fi:
            L.append(f"| 初诊 | {fi['Jaccard']:.3f} | ≥ 0.22（须超 Oracle 0.208） | "
                     f"{'✅' if fi['Jaccard'] >= 0.22 else '❌'} |")
        if fu and fi:
            un = 0.5 * (fu["Jaccard"] + fi["Jaccard"])
            L.append(f"| 未加权平均（对照） | {un:.3f} | — | — |")

    models = [x for x in results if x["类别"] == "模型"]
    if len(models) >= 2:
        L.append("\n## 3b. 模型间配对 bootstrap（判断差异是否真实）\n")
        L.append("| 对比 | 差值 | 95% CI | 单尾 p(差值≤0) | 判定 |")
        L.append("|---|---|---|---|---|")
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                a, b = models[i], models[j]
                obs, lo, hi, pv = paired_bootstrap(a["_per_sample"], b["_per_sample"])
                sig = pv < 0.05 or (1 - pv) < 0.05
                L.append(f"| {a['label']} − {b['label']} | {obs:+.4f} | "
                         f"[{lo:+.4f}, {hi:+.4f}] | {pv:.4f} | "
                         f"{'**显著**' if sig else '不显著（含 0）'} |")
        L.append("\n> 配对 bootstrap：同一样本上比较两模型，消除样本难度差异，"
                 "比各自 CI 是否重叠更灵敏。\n")

    if len(models) >= 2:
        L.append("\n## 3c. 分口径配对 bootstrap（总体打平后，要看差在哪）\n")
        L.append("| 对比 | 口径 | n | 差值 | 95% CI | 单尾 p | 判定 |")
        L.append("|---|---|---|---|---|---|---|")
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                a, b = models[i], models[j]
                for sub in ["全部", "初诊(无既往方)", "复诊(有既往方)"]:
                    ma = a["_masks"].get(sub)
                    if ma is None:
                        continue
                    va, vb = _sub(a["_per_sample"], ma), _sub(b["_per_sample"], ma)
                    if len(va) < 30:
                        continue
                    obs, lo, hi, pv = paired_bootstrap(va, vb)
                    sig = pv < 0.05 or (1 - pv) < 0.05
                    L.append(f"| {a['label']} − {b['label']} | {sub} | {len(va)} | "
                             f"{obs:+.4f} | [{lo:+.4f}, {hi:+.4f}] | {pv:.4f} | "
                             f"{'**显著**' if sig else '不显著（含 0）'} |")
        L.append("")

    L.append("\n## 4. 其它子集（记忆 vs 泛化）\n")
    L.append("| 来源 | 处方重复子集 J | 处方非重复子集 J | 患者多次 J | 患者单次 J |")
    L.append("|---|---|---|---|---|")
    for r in sorted(results, key=lambda x: order.get(x["类别"], 9)):
        cells = []
        for s in ["处方重复子集", "处方非重复子集", "患者多次就诊子集", "患者单次就诊子集"]:
            v = r["subsets"].get(s)
            cells.append(f"{v['Jaccard']:.3f}" if v else "—")
        L.append(f"| {r['label']} | " + " | ".join(cells) + " |")

    L.append("\n## 5. 复现命令\n")
    L.append("```bash\n"
             "# 1) 用训练好的 LoRA 在冻结测试集上生成（greedy，与训练同 prompt 口径）\n"
             "python3 src/predict.py --model <base> --adapter <lora> \\\n"
             "    --ref data/ref_test.jsonl --out data/preds/<name>.jsonl\n"
             "# 2) 统一对比（含 Gate 判定）\n"
             "python3 src/compare_models.py --model <name>=data/preds/<name>.jsonl\n"
             "```\n")
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", default=[],
                    help="label=path，可重复")
    ap.add_argument("--ref", type=Path, default=DATA_DIR / "ref_test.jsonl")
    ap.add_argument("--train", type=Path, default=DATA_DIR / "ref_train.jsonl")
    ap.add_argument("--out", type=Path, default=REPORT_DIR / "model_comparison.md")
    ap.add_argument("--json-out", type=Path, default=REPORT_DIR / "model_comparison.json")
    args = ap.parse_args()

    print("=" * 78)
    print("统一评测对比")
    print("=" * 78)
    results = collect(args.model, args.ref, args.train)
    if not results:
        raise SystemExit("没有可评测的对象")

    md = render(results, args.ref)
    args.out.write_text(md, encoding="utf-8")
    print(f"\n✅ 报告 -> {args.out}")

    slim = [{k: v for k, v in r.items() if k not in ("subsets", "_per_sample")} |
            {"subsets": r["subsets"]} for r in results]
    args.json_out.write_text(json.dumps(slim, ensure_ascii=False, indent=1),
                             encoding="utf-8")
    print(f"✅ JSON -> {args.json_out}")

    # 结论行：模型在关键口径上是否超过基线
    print("\n" + "=" * 78)
    print("关键结论")
    print("=" * 78)
    for r in [x for x in results if x["类别"] == "模型"]:
        b = {x["label"]: x for x in results}
        const = next((x for x in results if x["类别"] == "下界"), None)
        orac = next((x for x in results if x["类别"] == "Oracle"), None)
        carry = next((x for x in results if x["类别"] == "治疗进程"), None)
        print(f"\n[{r['label']}]")
        if const:
            d = r["处方_Jaccard"] - const["处方_Jaccard"]
            print(f"  vs 常量方 {const['处方_Jaccard']:.3f}: {d:+.3f} "
                  f"{'✅ 有效' if d > 0 else '❌ 无效（低于常量方）'}")
        if orac:
            d = r["处方_Jaccard"] - orac["处方_Jaccard"]
            print(f"  vs Oracle 上界 {orac['处方_Jaccard']:.3f}: {d:+.3f} "
                  f"{'✅ 超过（证明真读了自由文本）' if d > 0 else '❌ 未超过（疑似在查表）'}")
        # 分口径必须与对方【同口径】比：早期版本拿模型复诊比对方"总体"，得出错误结论
        fu_r = r["subsets"].get("复诊(有既往方)")
        fi_r = r["subsets"].get("初诊(无既往方)")
        if carry:
            cfu = carry["subsets"].get("复诊(有既往方)")
            if fu_r and cfu:
                d = fu_r["Jaccard"] - cfu["Jaccard"]
                print(f"  复诊 vs 照抄上次方(复诊口径 {cfu['Jaccard']:.3f}): {d:+.3f} "
                      f"{'✅ 优于照抄' if d > 0 else '❌ 不如直接照抄'}")
            cfi = carry["subsets"].get("初诊(无既往方)")
            if fi_r and cfi:
                d = fi_r["Jaccard"] - cfi["Jaccard"]
                print(f"  初诊 vs 初诊Oracle(初诊口径 {cfi['Jaccard']:.3f}): {d:+.3f} "
                      f"{'✅ 超过（Oracle 用了真实诊断）' if d > 0 else '❌ 未超过'}")


if __name__ == "__main__":
    main()
