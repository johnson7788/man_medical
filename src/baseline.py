"""
baseline.py — 规则 / 检索式基线（Stage 0）

为什么必须有这一步（计划 §5 Stage 0）：
  没有基线，之后无法判断微调模型是否真的有增益；也无法发现评测器本身的口径错误。
  本脚本产出的预测走与模型完全相同的评测入口（evaluate.py）。

⚠ 方法论语义：必须严格区分「Oracle」与「可实现」两类基线。

  Oracle 类（检索键 = **真实**中医诊断，现实中模型拿不到，只作上界参考）
    O1 证型检索 / O2 病名检索 / O3 证型+病名检索 / O4 治法检索
    O5 三级回退（O3 -> O1 -> 全局），最佳 Oracle
  可实现类（只用输入字段：主诉/现病史/西医诊断/性别/年龄）
    B1 全局常量方          —— 下界
    B2 西医诊断检索
    B3 性别年龄检索
    B4 西医诊断+性别年龄检索
  诊断基线（只用输入字段）
    D1 全局常量诊断 / D2 西医诊断检索 / D3 性别年龄检索
  端到端（真实链路：先预测诊断，再用【预测诊断】检索处方）
    R0 = D2(预测诊断) -> 处方检索     真正的可实现对照线

用法:
  python3 src/baseline.py [--k 15] [--n-dx 3]
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import evaluate as E
import medlib as M
from medlib import DATA_DIR, REPORT_DIR

# --- 输入侧 key（非 Oracle） ---
const_key = lambda r: "__const__"                                        # noqa: E731
wm_key = lambda r: tuple(sorted(r["wm_dx"]))                             # noqa: E731
demo_key = lambda r: (r["sex"], min((r["age"] or 0) // 10 * 10, 80))     # noqa: E731
wm_demo_key = lambda r: (wm_key(r), demo_key(r))                         # noqa: E731

# --- 标签侧 key（Oracle：使用真实中医诊断） ---
zheng_key = lambda r: tuple(sorted({t[1] for t in r["tcm_dx"] if t[1]}))  # noqa: E731
bing_key = lambda r: tuple(sorted({t[0] for t in r["tcm_dx"] if t[0]}))   # noqa: E731
zhi_key = lambda r: tuple(sorted({t[2] for t in r["tcm_dx"] if t[2]}))    # noqa: E731
zheng_bing_key = lambda r: tuple(                                     # noqa: E731
    sorted({(t[0], t[1]) for t in r["tcm_dx"] if t[1]}))


def top_k_padded(counter: Counter, fallback: list[str], k: int) -> list[str]:
    """取前 k 味；不足时用全局高频补足。

    为什么必须补足（实测踩坑）：稀有证型的先验可能只有 1-3 味药，
    `most_common(15)` 便只输出 3 味 —— Precision 很高但 Recall 崩掉，
    实测 Jaccard 反而【低于】不做个性化的全局常量方（0.156）。
    检索式基线必须控制输出长度，否则会得出「检索不如常量」的错误结论。
    """
    out = ([h for h, _ in sorted(counter.items(), key=lambda x: (-x[1], x[0]))[:k]]
           if counter else [])
    if len(out) < k:
        seen = set(out)
        for h in fallback:
            if h not in seen:
                out.append(h)
                seen.add(h)
            if len(out) >= k:
                break
    return out[:k]


# ---------------------------------------------------------------------------
# 处方预测器
# ---------------------------------------------------------------------------

def make_rx_predictor(train: list[dict], keyfn, k: int, fallback: list[str]):
    prior: dict = {}
    for r in train:
        prior.setdefault(keyfn(r), Counter()).update(r["rx"])

    def predict(r: dict) -> list[str]:
        return top_k_padded(prior.get(keyfn(r), Counter()), fallback, k)

    return predict


def make_rx_oracle(train: list[dict], k: int):
    """Oracle：用【真实】中医诊断检索，(病名,证型) -> 证型 -> 全局三级回退。"""
    prior_bz: dict = {}
    prior_z: dict = {}
    for r in train:
        prior_bz.setdefault(zheng_bing_key(r), Counter()).update(r["rx"])
        prior_z.setdefault(zheng_key(r), Counter()).update(r["rx"])
    global_c = Counter(h for r in train for h in r["rx"])
    fallback = [h for h, _ in sorted(global_c.items(), key=lambda x: (-x[1], x[0]))[:k]]

    def predict(r: dict) -> list[str]:
        c = prior_bz.get(zheng_bing_key(r)) or prior_z.get(zheng_key(r)) or global_c
        return top_k_padded(c, fallback, k)

    return predict


# ---------------------------------------------------------------------------
# 诊断预测器
# ---------------------------------------------------------------------------

def make_dx_predictor(train: list[dict], keyfn, n: int):
    """按 keyfn 检索训练集最常见的 n 个诊断三元组。"""
    prior: dict = {}
    global_c: Counter = Counter()
    for r in train:
        for t in r["tcm_dx"]:
            tt = tuple(t)
            prior.setdefault(keyfn(r), Counter())[tt] += 1
            global_c[tt] += 1

    def predict(r: dict) -> list[tuple[str, str, str]]:
        c = prior.get(keyfn(r)) or global_c
        return [t for t, _ in sorted(c.items(), key=lambda x: (-x[1], x[0]))[:n]]

    return predict


def make_rx_from_pred_dx(train: list[dict], k: int, fallback: list[str]):
    """端到端专用：用【预测出来的】诊断三元组检索处方。

    这是 R0 与 Oracle 的唯一区别 —— Oracle 用真实诊断，R0 用预测诊断。
    两者之差即「诊断误差的代价」（计划 §4.3 消融）。
    """
    prior_bz: dict = {}
    prior_z: dict = {}
    for r in train:
        prior_bz.setdefault(zheng_bing_key(r), Counter()).update(r["rx"])
        prior_z.setdefault(zheng_key(r), Counter()).update(r["rx"])

    def predict(dx_pred: list[tuple]) -> list[str]:
        bz = tuple(sorted({(t[0], t[1]) for t in dx_pred if len(t) >= 2 and t[1]}))
        z = tuple(sorted({t[1] for t in dx_pred if len(t) >= 2 and t[1]}))
        c = prior_bz.get(bz) or prior_z.get(z) or Counter()
        return top_k_padded(c, fallback, k)

    return predict


# ---------------------------------------------------------------------------
# 组装
# ---------------------------------------------------------------------------

def render(dx: list[tuple], rx: list[str]) -> str:
    return (f"{M.DX_HEADER}\n" + "\n".join("·".join(t) for t in dx) +
            f"\n{M.RX_HEADER}\n" + " ".join(rx))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=15, help="处方输出药味数")
    ap.add_argument("--n-dx", type=int, default=3, help="诊断输出条数")
    args = ap.parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    pred_dir = DATA_DIR / "preds"
    # 清空旧预测：基线命名会随口径调整而变化，残留文件会与报告对不上，误导后续排查
    if pred_dir.exists():
        stale = list(pred_dir.glob("*.jsonl"))
        for f in stale:
            f.unlink()
        if stale:
            print(f"清空旧预测文件 {len(stale)} 个")
    pred_dir.mkdir(parents=True, exist_ok=True)

    print("读取数据 ...")
    all_records = M.read_jsonl(DATA_DIR / "all_records.jsonl")
    gold = [r for r in all_records if M.is_gold(r)]
    sp = M.split_by_patient(gold)
    train, test = sp["train"], sp["test"]
    print(f"  train {len(train):,} / test {len(test):,}  (患者分组，零重叠)")

    vocab = set(M.top_herbs(train, 10 ** 9))
    key_herbs = set(M.top_herbs(train, 100))
    rx_fallback = M.top_herbs(train, args.k)
    ref_rx = [r["rx"] for r in test]
    ref_dx = [[tuple(t) for t in r["tcm_dx"]] for r in test]

    # ---------------- 可实现处方基线（只用输入字段） ----------------
    real_defs = [
        ("B1_全局常量方", const_key),
        ("B2_西医诊断检索", wm_key),
        ("B3_性别年龄检索", demo_key),
        ("B4_西医诊断+性别年龄", wm_demo_key),
    ]
    print(f"\n【可实现处方基线】只用输入字段，n={len(test):,}")
    print(f"  {'基线':26s} {'Jaccard':>9s} {'P':>7s} {'R':>7s} {'完全匹配':>9s}")
    print("  " + "-" * 64)
    real_results, real_preds = [], {}
    for name, keyfn in real_defs:
        pred = make_rx_predictor(train, keyfn, args.k, rx_fallback)
        prx = [pred(r) for r in test]
        real_preds[name] = prx
        m = E.rx_metrics(prx, ref_rx, key_herbs, vocab)
        real_results.append((name, m))
        print(f"  {name:26s} {m['处方_Jaccard']:9.3f} {m['处方_P']:7.3f} "
              f"{m['处方_R']:7.3f} {m['处方_完全匹配']*100:8.2f}%")

    # ---------------- Oracle 处方基线（用真实中医诊断） ----------------
    oracle_defs = [
        ("O1_证型检索", zheng_key),
        ("O2_病名检索", bing_key),
        ("O3_证型+病名检索", zheng_bing_key),
        ("O4_治法检索", zhi_key),
    ]
    print(f"\n【Oracle 处方基线】检索键 = 真实中医诊断（模型拿不到，仅作上界）")
    print(f"  {'基线':26s} {'Jaccard':>9s} {'P':>7s} {'R':>7s} {'完全匹配':>9s}")
    print("  " + "-" * 64)
    oracle_results, oracle_preds = [], {}
    for name, keyfn in oracle_defs:
        pred = make_rx_predictor(train, keyfn, args.k, rx_fallback)
        prx = [pred(r) for r in test]
        oracle_preds[name] = prx
        m = E.rx_metrics(prx, ref_rx, key_herbs, vocab)
        oracle_results.append((name, m))
        print(f"  {name:26s} {m['处方_Jaccard']:9.3f} {m['处方_P']:7.3f} "
              f"{m['处方_R']:7.3f} {m['处方_完全匹配']*100:8.2f}%")

    pred_o5 = make_rx_oracle(train, args.k)
    prx_o5 = [pred_o5(r) for r in test]
    oracle_preds["O5_三级回退"] = prx_o5
    m_o5 = E.rx_metrics(prx_o5, ref_rx, key_herbs, vocab)
    oracle_results.append(("O5_三级回退(最佳Oracle)", m_o5))
    print(f"  {'O5_三级回退(最佳Oracle)':26s} {m_o5['处方_Jaccard']:9.3f} {m_o5['处方_P']:7.3f} "
          f"{m_o5['处方_R']:7.3f} {m_o5['处方_完全匹配']*100:8.2f}%")

    # k 敏感度
    print("\n【k 敏感度】O1 证型检索")
    k_sens = []
    for k in (10, 15, 20, 25, 30):
        p = make_rx_predictor(train, zheng_key, k, M.top_herbs(train, k))
        m = E.rx_metrics([p(r) for r in test], ref_rx, key_herbs, vocab)
        k_sens.append((k, m))
        print(f"  k={k:2d}  Jaccard={m['处方_Jaccard']:.3f}  P={m['处方_P']:.3f}  R={m['处方_R']:.3f}")

    # ---------------- 诊断基线 ----------------
    dx_defs = [("D1_全局常量诊断", const_key), ("D2_西医诊断检索", wm_key),
               ("D3_性别年龄检索", demo_key)]
    print(f"\n【诊断基线】只用输入字段，输出 {args.n_dx} 条")
    print(f"  {'基线':26s} {'证型F1':>8s} {'病名F1':>8s} {'治法F1':>8s} {'集合完全一致':>12s}")
    print("  " + "-" * 68)
    dx_results, dx_preds = [], {}
    for name, keyfn in dx_defs:
        pred = make_dx_predictor(train, keyfn, args.n_dx)
        pdx = [pred(r) for r in test]
        dx_preds[name] = pdx
        m = E.diagnosis_metrics(pdx, ref_dx)
        dx_results.append((name, m))
        print(f"  {name:26s} {m['证型_F1']:8.3f} {m['病名_F1']:8.3f} {m['治法_F1']:8.3f} "
              f"{m['诊断集合完全一致']*100:11.2f}%")

    best_dx = max(dx_results, key=lambda x: x[1]["证型_F1"])
    print(f"  -> 最佳诊断基线: {best_dx[0]} (证型 F1 = {best_dx[1]['证型_F1']:.3f})")

    # ---------------- 端到端 R0：预测诊断 -> 用【预测诊断】检索处方 ----------------
    rx_from_dx = make_rx_from_pred_dx(train, args.k, rx_fallback)
    rx_e2e = [rx_from_dx(pdx) for pdx in dx_preds[best_dx[0]]]
    m_e2e = E.rx_metrics(rx_e2e, ref_rx, key_herbs, vocab)
    print(f"\n【端到端 R0】{best_dx[0]} -> 用预测诊断检索处方")
    print(f"  Jaccard={m_e2e['处方_Jaccard']:.3f}  P={m_e2e['处方_P']:.3f}  "
          f"R={m_e2e['处方_R']:.3f}")
    print(f"  诊断误差代价 = Oracle {m_o5['处方_Jaccard']:.3f} - 端到端 "
          f"{m_e2e['处方_Jaccard']:.3f} = {m_o5['处方_Jaccard']-m_e2e['处方_Jaccard']:.3f}")

    # ---------------- X：既往处方基线（治疗进程）----------------
    # 这是本数据集最重要的基线。实测同一患者相邻两次就诊的处方 Jaccard = 0.539，
    # 远超整套中医诊断的检索上界 0.203 —— 「上次开的方」比所有证型信息都强。
    fu = [r for r in test if r.get("prev_rx")]
    cold = [r for r in test if not r.get("prev_rx")]
    print(f"\n【X 既往处方基线】复诊 {len(fu):,} ({len(fu)/len(test)*100:.1f}%) / "
          f"初诊 {len(cold):,} ({len(cold)/len(test)*100:.1f}%)")
    m_x1 = E.rx_metrics([r["prev_rx"] for r in fu], [r["rx"] for r in fu], key_herbs, vocab)
    print(f"  X1 照抄上次处方（仅复诊）  Jaccard={m_x1['处方_Jaccard']:.3f}  "
          f"P={m_x1['处方_P']:.3f}  R={m_x1['处方_R']:.3f}")
    # 复诊走照抄、初诊走 Oracle 的组合
    rx_combo = [list(r["prev_rx"]) if r.get("prev_rx") else pred_o5(r) for r in test]
    m_combo = E.rx_metrics(rx_combo, ref_rx, key_herbs, vocab)
    print(f"  X2 复诊照抄 + 初诊 Oracle  Jaccard={m_combo['处方_Jaccard']:.3f}  "
          f"P={m_combo['处方_P']:.3f}  R={m_combo['处方_R']:.3f}  <- 新上界")
    m_cold_o = E.rx_metrics([pred_o5(r) for r in cold], [r["rx"] for r in cold],
                            key_herbs, vocab)
    print(f"  参考：初诊子集 Oracle Jaccard={m_cold_o['处方_Jaccard']:.3f}；"
          f"复诊子集 Oracle Jaccard="
          f"{E.rx_metrics([pred_o5(r) for r in fu], [r['rx'] for r in fu], key_herbs, vocab)['处方_Jaccard']:.3f}"
          "  <- 复诊上 Oracle 反而更差")
    print(f"  复诊调方幅度：完全不改方占 "
          f"{sum(1 for r in fu if set(r['rx']) == set(r['prev_rx']))/max(1,len(fu))*100:.1f}%")

    # ---------------- 写出预测文件 ----------------
    files = {}
    for name, prx in real_preds.items():
        files[name] = [{"rid": r["rid"], "output": render([], prx)}
                       for r, prx in zip(test, prx)]
    for name, prx in oracle_preds.items():
        files[name] = [{"rid": r["rid"], "output": render([], prx)}
                       for r, prx in zip(test, prx)]
    for name, pdx in dx_preds.items():
        files[name] = [{"rid": r["rid"], "output": render(pdx, [])}
                       for r, pdx in zip(test, pdx)]
    files["R0_端到端规则"] = [{"rid": r["rid"], "output": render(pdx, prx)}
                          for r, pdx, prx in zip(test, dx_preds[best_dx[0]], rx_e2e)]
    files["X2_复诊照抄+初诊Oracle"] = [{"rid": r["rid"], "output": render([], prx)}
                                   for r, prx in zip(test, rx_combo)]
    for name, rows in files.items():
        M.write_jsonl(pred_dir / f"{name}.jsonl", rows)

    # ---------------- 时间外推集的 R0（G5 用） ----------------
    tr_t, te_t = M.split_by_time(gold)
    print(f"\n【时间外推 R0】train(d<2025-07-01)={len(tr_t):,}  "
          f"test={len(te_t):,}  患者零重叠")
    key_t = set(M.top_herbs(tr_t, 100))
    vocab_t = set(M.top_herbs(tr_t, 10 ** 9))
    fb_t = M.top_herbs(tr_t, args.k)
    dx_pred_t = make_dx_predictor(tr_t, wm_key, args.n_dx)
    rx_from_dx_t = make_rx_from_pred_dx(tr_t, args.k, fb_t)
    pdx_t = [dx_pred_t(r) for r in te_t]
    prx_t = [rx_from_dx_t(p) for p in pdx_t]
    m_t = E.rx_metrics(prx_t, [r["rx"] for r in te_t], key_t, vocab_t)
    m_dx_t = E.diagnosis_metrics(pdx_t, [[tuple(t) for t in r["tcm_dx"]] for r in te_t])
    print(f"  证型 F1={m_dx_t['证型_F1']:.3f}  Jaccard={m_t['处方_Jaccard']:.3f}  "
          f"P={m_t['处方_P']:.3f}  R={m_t['处方_R']:.3f}")
    M.write_jsonl(pred_dir / "R0_端到端规则_time.jsonl",
                  [{"rid": r["rid"], "output": render(p, x)}
                   for r, p, x in zip(te_t, pdx_t, prx_t)])
    drift = (m_e2e["处方_Jaccard"] - m_t["处方_Jaccard"]) / max(1e-9, m_e2e["处方_Jaccard"])
    print(f"  G5 时效漂移：随机切分 {m_e2e['处方_Jaccard']:.3f} vs 时间外推 "
          f"{m_t['处方_Jaccard']:.3f}  相对下降 {drift*100:.1f}%")

    # ---------------- 报告 ----------------
    b1 = dict(real_results)["B1_全局常量方"]["处方_Jaccard"]
    best_real = max(real_results, key=lambda x: x[1]["处方_Jaccard"])
    L = ["# 规则 / 检索式基线报告（Stage 0）", "",
         f"- 训练集 **{len(train):,}** 条 / 测试集 **{len(test):,}** 条",
         "- 切分：**按患者 ID 分组 80/10/10**，train/test 患者零重叠",
         "- 先验统计仅使用 train 部分（无泄漏）",
         "- 评测入口与微调模型完全相同（`src/evaluate.py`）", "",
         "## 可实现处方基线（只用输入字段）", "",
         "| 基线 | Jaccard（主） | P | R | 完全匹配 |", "|---|---|---|---|---|"]
    for n, m in sorted(real_results, key=lambda x: -x[1]["处方_Jaccard"]):
        L.append(f"| {n} | **{m['处方_Jaccard']:.3f}** | {m['处方_P']:.3f} | "
                 f"{m['处方_R']:.3f} | {m['处方_完全匹配']*100:.2f}% |")
    L += ["", "## 既往处方基线（治疗进程；本数据集最重要的对照）", "",
          f"- 复诊 {len(fu):,} 条（{len(fu)/len(test)*100:.1f}%）/ "
          f"初诊 {len(cold):,} 条（{len(cold)/len(test)*100:.1f}%）",
          "- 既往处方在生产 HIS 系统里本来就能拿到，不是标签泄漏；"
          "它只是被当前「四诊 → 处方」的任务定义丢掉了。", "",
          "| 基线 | 适用范围 | Jaccard | P | R |", "|---|---|---|---|---|",
          f"| X1 照抄上次处方 | 复诊 {len(fu):,} | **{m_x1['处方_Jaccard']:.3f}** | "
          f"{m_x1['处方_P']:.3f} | {m_x1['处方_R']:.3f} |",
          f"| X2 复诊照抄 + 初诊 Oracle | 全部 {len(test):,} | "
          f"**{m_combo['处方_Jaccard']:.3f}** | {m_combo['处方_P']:.3f} | "
          f"{m_combo['处方_R']:.3f} |", "",
          f"**X1 = {m_x1['处方_Jaccard']:.3f}，是 Oracle 检索上界 "
          f"{m_o5['处方_Jaccard']:.3f} 的 "
          f"{m_x1['处方_Jaccard']/max(1e-9,m_o5['处方_Jaccard']):.1f} 倍。**",
          "复诊时「上次开的方」比整套中医诊断更有预测力 —— 因为处方是"
          "**治疗进程中的一次调整**，不是从零生成。", "",
          "## Oracle 处方基线（检索键 = 真实中医诊断，**不可实现**）", "",
          "| 基线 | Jaccard | P | R | 完全匹配 |", "|---|---|---|---|---|"]
    for n, m in sorted(oracle_results, key=lambda x: -x[1]["处方_Jaccard"]):
        L.append(f"| {n} | {m['处方_Jaccard']:.3f} | {m['处方_P']:.3f} | "
                 f"{m['处方_R']:.3f} | {m['处方_完全匹配']*100:.2f}% |")
    L += ["", "## 诊断基线（只用输入字段）", "",
          "| 基线 | 证型 F1 | 病名 F1 | 治法 F1 | 集合完全一致 |", "|---|---|---|---|---|"]
    for n, m in sorted(dx_results, key=lambda x: -x[1]["证型_F1"]):
        L.append(f"| {n} | **{m['证型_F1']:.3f}** | {m['病名_F1']:.3f} | "
                 f"{m['治法_F1']:.3f} | {m['诊断集合完全一致']*100:.2f}% |")
    L += ["", "## 端到端 R0（预测诊断 -> 用预测诊断检索处方）", "",
          f"- 链路：`{best_dx[0]}` -> 处方检索",
          f"- **Jaccard = {m_e2e['处方_Jaccard']:.3f}**，P = {m_e2e['处方_P']:.3f}，"
          f"R = {m_e2e['处方_R']:.3f}",
          f"- 诊断误差代价 = Oracle {m_o5['处方_Jaccard']:.3f} - 端到端 "
          f"{m_e2e['处方_Jaccard']:.3f} = **{m_o5['处方_Jaccard']-m_e2e['处方_Jaccard']:.3f}**", "",
          "## k 敏感度（O1 证型检索）", "", "| k | Jaccard | P | R |", "|---|---|---|---|"]
    for k, m in k_sens:
        L.append(f"| {k} | {m['处方_Jaccard']:.3f} | {m['处方_P']:.3f} | {m['处方_R']:.3f} |")

    g = M.GATES
    L += ["", "## 解读（写进评审材料）", "",
          f"1. **全局常量方 = {b1:.3f} Jaccard**。不做任何个性化已有此分，"
          f"任何 Jaccard 低于 {b1:.3f} 的模型都是无效的。",
          f"2. 最佳可实现基线 **{best_real[0]} = "
          f"{best_real[1]['处方_Jaccard']:.3f}**；端到端 R0 = "
          f"{m_e2e['处方_Jaccard']:.3f}（**低于常量方**，诊断误差代价 "
          f"{m_o5['处方_Jaccard'] - m_e2e['处方_Jaccard']:.3f}）。",
          f"3. **Oracle 上界（诊断完全正确）= {m_o5['处方_Jaccard']:.3f}**。"
          "这是「只用离散证型标签做查表」的天花板 —— 微调模型必须突破它，",
          "   而唯一的突破口是**从主诉/现病史自由文本里做个性化推断**，"
          "这正是本项目的核心价值主张。",
          "4. **完全匹配率在所有基线上都是 0%**（Oracle 亦为 0%）—— "
          "再次确认主指标必须是 Jaccard / 集合 F1。",
          f"5. 诊断侧：最佳 {best_dx[0]} 证型 F1 = {best_dx[1]['证型_F1']:.3f}；"
          "D2 与 D1 差距很小说明西医诊断对中医证型的判别力有限。", "",
          "## Gate 锚点（阈值定义与 medlib.GATES 同源）", "",
          "| 指标 | 常量下界 | 最佳可实现 | Oracle 上界 | 必达 | 目标 |",
          "|---|---|---|---|---|---|",
          f"| 处方 Jaccard | {b1:.3f} | {best_real[1]['处方_Jaccard']:.3f} | "
          f"{m_o5['处方_Jaccard']:.3f} | **>= {g['处方_Jaccard']['必达']}** | "
          f">= {g['处方_Jaccard']['目标']} |",
          f"| 关键药 Recall | {dict(real_results)['B1_全局常量方']['关键药_Recall']:.3f} | "
          f"{best_real[1]['关键药_Recall']:.3f} | {m_o5['关键药_Recall']:.3f} | "
          f"**>= {g['关键药_Recall']['必达']}** | >= {g['关键药_Recall']['目标']} |",
          f"| 证型 micro-F1 | {dict(dx_results)['D1_全局常量诊断']['证型_F1']:.3f} | "
          f"{best_dx[1]['证型_F1']:.3f} | - | >= {g['证型_F1']['必达']} | "
          f">= {g['证型_F1']['目标']} |", "",
          f"> 两个「必达」线都设在 **Oracle 上界之上**"
          f"（Jaccard {g['处方_Jaccard']['必达']} > {m_o5['处方_Jaccard']:.3f}；"
          f"关键药 {g['关键药_Recall']['必达']} > {m_o5['关键药_Recall']:.3f}）——",
          "> 达不到就说明模型只是学会了「证型 → 常用方」的查表，没有真正读病史。"]
    (REPORT_DIR / "baseline_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")

    print(f"\n报告 -> reports/baseline_report.md")
    print(f"预测 -> data/preds/  ({len(files)} 个文件)")
    print("\n用完整评测器复核（含 Gate 判定与子集分析）：")
    print("  python3 src/evaluate.py --ref data/ref_test.jsonl \\")
    print("      --pred data/preds/R0_端到端规则.jsonl \\")
    print("      --ref-time data/ref_test_time.jsonl \\")
    print("      --pred-time data/preds/R0_端到端规则_time.jsonl \\")
    print("      --json-out reports/stage0_baseline.json")


if __name__ == "__main__":
    main()
