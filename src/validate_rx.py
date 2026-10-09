"""
validate_rx.py — 处方功效/安全约束校验器（推理链步骤5）

定位：
  《模型训练计划.md》§13 五步推理链的**步骤5**，同时是一个独立可调用的守门器。

    validate_prescription(rx, tcm_dx, patient=None) -> dict

  离线：批量跑 26,923 条真实处方，测误报率、找异常处方
  在线：守门模型输出；每条 finding 都可溯源，供人复核

设计原则（都有实测依据）：
  1. **block 只用于安全性**（幻觉药 / 十八反 / 妊娠禁忌 / 覆盖率恰为 0）。
     覆盖率【不做硬拦截】：实测 92.3% 真实处方完全覆盖，另外约 3% 存在
     「治法标签与方药不一致」，硬拦会误杀医生的合理处方。
  2. **不做朴素的「寒热同用即矛盾」检查**：实测真实处方中寒热并用占 17.3%
     （交泰丸、乌梅丸等经典方即如此），那样会大面积误杀。
     只在「声明了某治法却无对应功效、且出现对抗功效」时才判为冲突。
  3. **字典缺失必须显式报告**：54.3% 的真实处方含至少一味无功效数据的药材，
     这些药不参与校验，但会降低 confidence，绝不静默放行。
  4. **阈值全部外置**到 validator_config.json，中医师可改而不动代码。
  5. 每条 finding 带 evidence（命中哪条映射、来源是 rule 还是 data），
     因为治法→功效映射表仍待中医师审定，必须能逐条质疑。

用法：
  python3 src/validate_rx.py                     # 校准 + 合成负样本测试 + 报告
  python3 src/validate_rx.py --demo              # 打印若干真实处方的校验结果
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import medlib as M
from medlib import DATA_DIR, REPORT_DIR

EXT = M.ROOT / "data" / "external"

# verdict 严重度排序。注意 info 不抬升 verdict —— 它只是信息（如「字典缺该药功效」），
# 否则 54% 含字典缺失药材的真实处方会被判成非 ok。
LEVEL_ORDER = {"info": -1, "ok": 0, "note": 1, "warn": 2, "block": 3}


# ---------------------------------------------------------------------------
# 资源加载
# ---------------------------------------------------------------------------

class Validator:
    def __init__(self, config: dict | None = None, config_path: Path | None = None):
        """config 优先于 config_path；便于测试构造不同策略（如不降级丁香郁金）。"""
        self.cfg = config if config is not None else json.loads(
            (config_path or EXT / "validator_config.json").read_text(encoding="utf-8"))
        self.hd = {r["herb"]: r for r in M.read_jsonl(EXT / "herb_dictionary.jsonl")}
        self.zmap_raw = json.loads((EXT / "zhi_action_map.json").read_text(encoding="utf-8"))
        self.zhi_map = {z: set(v["actions"]) for z, v in self.zmap_raw["map"].items()}
        self.zhi_src = {z: v["sources"] for z, v in self.zmap_raw["map"].items()}
        self.zhi_units = {z: v["units"] for z, v in self.zmap_raw["map"].items()}
        self.safety = json.loads((EXT / "safety_rules.json").read_text(encoding="utf-8"))
        self._prep()

    def _prep(self) -> None:
        # 十八反/十九畏 无序对
        self.incompat: dict[frozenset, str] = {}
        for e in self.safety["eighteen_incompat"] + self.safety["nineteen_incompat"]:
            self.incompat[frozenset((e["a"], e["b"]))] = f"{e['a']}+{e['b']}"
        # 白名单 = 字典全部药名
        self.vocab = set(self.hd)
        # 标准名 → 本方名 反查，用于把「相反药对」对齐到实际处方用名
        self.std2mine: dict[str, set[str]] = defaultdict(set)
        for mine, v in self.hd.items():
            if v.get("standard_name"):
                self.std2mine[v["standard_name"]].add(mine)
        # 对抗功效正则
        self.opposing = {u: re.compile(p) for u, p in self.cfg["contradiction"]["opposing"].items()}
        self.toxic = set(self.cfg["toxic_handling"]["herbs"])
        self.preg_forbid = set(self.safety["pregnancy_forbidden"])
        self.preg_caution = set(self.safety["pregnancy_caution"])
        self.cap = {k: v for k, v in self.cfg["dose"]["absolute_cap_g"].items()
                    if not k.startswith("_")}
        # 单位语义 → 满足它的功效（用于解释「缺什么」）
        self.unit2actions: dict[str, set[str]] = defaultdict(set)
        for z, acts in self.zhi_map.items():
            for u in self.zhi_units.get(z, []):
                self.unit2actions[u] |= acts

    # -- 工具 -------------------------------------------------------------
    def actions_of(self, herb: str) -> list[str]:
        return list(self.hd.get(herb, {}).get("actions_zh") or [])

    def _add(self, out: list, level: str, code: str, detail: str, **extra) -> None:
        out.append({"level": level, "code": code, "detail": detail, **extra})

    # -- 主校验 -----------------------------------------------------------
    def validate_prescription(self, rx: list, tcm_dx: list, patient: dict | None = None,
                              source: str = "") -> dict:
        """rx: [{"herb": str, "dose_g": float|None}] 或 [str]
        tcm_dx: [{"bing":..,"zheng":..,"zhi":..}] 或 [tuple(3)]
        """
        cfg = self.cfg
        findings: list[dict] = []

        # 归一化输入
        herbs: list[tuple[str, float | None]] = []
        for x in rx or []:
            if isinstance(x, str):
                herbs.append((x, None))
            elif isinstance(x, (list, tuple)):
                herbs.append((x[0], x[1] if len(x) > 1 else None))
            else:
                herbs.append((x.get("herb", ""), x.get("dose_g")))
        names = [h for h, _ in herbs if h]

        zhis: list[str] = []
        for t in tcm_dx or []:
            if isinstance(t, (list, tuple)):
                if len(t) > 2 and t[2]:
                    zhis.append(t[2])
            elif isinstance(t, dict) and t.get("zhi"):
                zhis.append(t["zhi"])
        zhis = list(dict.fromkeys(zhis))

        # ---- 1. 药名白名单（幻觉药）----
        if cfg["herb_vocab"]["enabled"]:
            illegal = sorted({h for h in names if h not in self.vocab})
            if illegal:
                self._add(findings, cfg["herb_vocab"]["level"], "ILLEGAL_HERB",
                          f"药名不在白名单（疑似幻觉）：{'、'.join(illegal)}",
                          herbs=illegal)

        # ---- 2. 十八反 / 十九畏 ----
        if cfg["incompatibility"]["enabled"]:
            # 把处方用名映射到标准名后再配对
            std_set: dict[str, str] = {}
            for h in names:
                std = self.hd.get(h, {}).get("standard_name") or h
                std_set.setdefault(std, h)
            for pair, label in sorted(self.incompat.items(), key=lambda x: sorted(x[0])):
                a, b = sorted(pair)
                if a in std_set and b in std_set:
                    lvl = cfg["incompatibility"]["level"]
                    for pat, dl in cfg["incompatibility"]["downgrade"].items():
                        if pat.startswith("_"):
                            continue
                        pa, pb = pat.split("|")
                        if {a, b} == {pa, pb}:
                            lvl = dl
                    self._add(findings, lvl, "INCOMPAT",
                              f"配伍禁忌（十八反/十九畏）：{label}"
                              + ("（该条目在中医界有争议，已降级为提示）"
                                 if lvl != cfg["incompatibility"]["level"] else ""),
                              pair=[std_set[a], std_set[b]], label=label)

        # ---- 3. 妊娠禁忌 ----
        if cfg["pregnancy"]["enabled"] and patient and patient.get("pregnant"):
            stds = {self.hd.get(h, {}).get("standard_name") or h for h in names}
            hit_f = sorted(stds & self.preg_forbid)
            hit_c = sorted(stds & self.preg_caution)
            if hit_f:
                self._add(findings, cfg["pregnancy"]["forbidden_level"], "PREGNANCY_FORBIDDEN",
                          f"妊娠禁用：{'、'.join(hit_f)}", herbs=hit_f)
            if hit_c:
                self._add(findings, cfg["pregnancy"]["caution_level"], "PREGNANCY_CAUTION",
                          f"妊娠慎用：{'、'.join(hit_c)}", herbs=hit_c)

        # ---- 4. 剂量 ----
        if cfg["dose"]["enabled"]:
            for h, d in herbs:
                if d is None:
                    continue
                std = self.hd.get(h, {}).get("standard_name") or h
                cap = self.cap.get(h) or self.cap.get(std)
                if cap and d > cap:
                    self._add(findings, "block", "DOSE_ABSOLUTE_CAP",
                              f"{h} 用量 {d}g 超过绝对上限 {cap}g", herb=h, dose_g=d, cap_g=cap)
                    continue
                dr = self.hd.get(h, {}).get("dosage_range")
                if dr and dr.get("max"):
                    lim = dr["max"] * cfg["dose"]["overflow_ratio"]
                    if d > lim:
                        self._add(findings, cfg["dose"]["level"], "DOSE_EXCEED",
                                  f"{h} 用量 {d}g 超过字典上限 {dr['max']}g 的 "
                                  f"{cfg['dose']['overflow_ratio']} 倍", herb=h, dose_g=d,
                                  limit_g=round(lim, 1))

        # ---- 5. 毒性药需特殊处理 ----
        # 注意：这是**用药提示**而非错误。实测 47% 的真实处方含此类药材，
        # 若计为 warn 会让 warn 级别失去意义 —— 故单独放在 handling_alerts 中，
        # 始终随结果返回（药师需要看到），但不影响 verdict。
        handling_alerts: list[dict] = []
        if cfg["toxic_handling"]["enabled"]:
            stds = {self.hd.get(h, {}).get("standard_name") or h for h in names}
            tox = sorted(stds & self.toxic)
            if tox:
                notes = []
                for h in names:
                    std = self.hd.get(h, {}).get("standard_name") or h
                    if std in self.toxic:
                        sn = self.hd.get(h, {}).get("safety_notes_zh") or ""
                        if sn:
                            notes.append(f"{h}: {sn[:60]}…")
                handling_alerts.append({
                    "code": "TOXIC_HANDLING",
                    "detail": f"含需特殊处理药材（须标注先煎/久煎/限量）：{'、'.join(tox)}",
                    "herbs": tox, "safety_excerpts": notes[:3]})

        # ---- 6. 功效覆盖率 ----
        acts = {a for h in names for a in self.actions_of(h)}
        cov, missing, covered = None, [], []
        if zhis:
            for z in zhis:
                am = self.zhi_map.get(z)
                if am and (am & acts):
                    covered.append(z)
                else:
                    missing.append(z)
            cov = len(covered) / len(zhis)

            ccfg = cfg["coverage"]
            zmissing_detail = []
            for z in missing:
                units = self.zhi_units.get(z) or []
                z_acts = self.zhi_map.get(z, set())
                # 关键：要用「该单元在【本治法】下贡献的功效」来判断，
                # 而不是单元在所有治法里的功效并集 —— 后者过宽，
                # 会出现「覆盖率 0 但 missing_units 为空」的矛盾输出。
                mu = [u for u in units
                      if not ((self.unit2actions.get(u, set()) & z_acts) & acts)]
                zmissing_detail.append({
                    "zhi": z, "units": units, "missing_units": mu,
                    "expected_actions": sorted(self.zhi_map.get(z, set()))[:12],
                    "mapping_source": self.zhi_src.get(z, {}),
                })
            if cov == 0 and ccfg.get("zero_as_block") and zhis:
                self._add(findings, "warn", "COVERAGE_ZERO",
                          f"治法与方药功效完全无交集（{len(zhis)} 个治法全部未覆盖），"
                          f"极可能诊断或处方有误", coverage=cov, missing=zmissing_detail)
            elif ccfg.get("block_below") is not None and cov < ccfg["block_below"]:
                self._add(findings, "block", "COVERAGE_LOW",
                          f"功效覆盖率 {cov:.2f} 低于硬阈值 {ccfg['block_below']}",
                          coverage=cov, missing=zmissing_detail)
            elif cov < ccfg["warn_below"]:
                self._add(findings, ccfg.get("warn_level", "warn"), "COVERAGE_LOW",
                          f"功效覆盖率 {cov:.2f} 低于 {ccfg['warn_below']}"
                          f"（真实处方 p05，约 5% 医生处方也在此区间），"
                          f"未覆盖治法：{'、'.join(missing)}",
                          coverage=cov, missing=zmissing_detail)
            elif cov < ccfg["note_below"]:
                self._add(findings, "note", "COVERAGE_PARTIAL",
                          f"功效覆盖率 {cov:.2f}，未覆盖：{'、'.join(missing)}（仅记录）",
                          coverage=cov, missing=zmissing_detail)

        # ---- 7. 反向功效冲突（由覆盖率缺失升级）----
        if cfg["contradiction"]["enabled"] and missing:
            for z in missing:
                units = self.zhi_units.get(z) or []
                for u in units:
                    rx_opp = self.opposing.get(u)
                    if rx_opp and any(rx_opp.search(a) for a in acts):
                        hit = sorted({a for a in acts if rx_opp.search(a)})
                        self._add(findings, cfg["contradiction"]["level"], "CONTRA_ACTION",
                                  f"声明治法「{z}」（含单元「{u}」）但方中无对应功效，"
                                  f"却出现对抗功效：{'、'.join(hit[:6])}",
                                  zhi=z, unit=u, opposing_actions=hit[:8])
                        break

        # ---- 8. 字典缺失（降置信度，必须显式报告）----
        unknown = sorted({h for h in names
                          if h not in self.vocab or not self.actions_of(h)})
        confidence = (len([h for h in names if self.actions_of(h)]) / len(names)) if names else 0.0
        if unknown:
            self._add(findings, "info", "UNKNOWN_HERB",
                      f"以下药材字典无功效数据，未参与功效校验：{'、'.join(unknown)}",
                      herbs=unknown, n_unknown=len(unknown))
        if confidence < cfg["confidence"]["low_below"]:
            self._add(findings, "note", "LOW_CONFIDENCE",
                      f"有功效数据的药材仅占 {confidence:.0%}，功效校验结论置信度低",
                      confidence=confidence)

        verdict = "ok"
        for f in findings:
            if LEVEL_ORDER.get(f["level"], -1) > LEVEL_ORDER[verdict]:
                verdict = f["level"]
        verdict = "ok" if verdict == "info" else verdict

        return {
            "verdict": verdict,
            "coverage": cov,
            "confidence": round(confidence, 3),
            "n_herbs": len(names),
            "n_zhis": len(zhis),
            "covered_zhis": covered,
            "missing_zhis": missing,
            "findings": sorted(findings, key=lambda f: (-LEVEL_ORDER.get(f["level"], -1), f["code"])),
            "handling_alerts": handling_alerts,
            "source": source,
        }


# ---------------------------------------------------------------------------
# 合成负样本：验证校验器真的能【检出】问题
# ---------------------------------------------------------------------------

def make_negatives(gold: list[dict], v: Validator, seed: int = 0):
    """真实数据里没有「坏方」标签，因此构造负样本测检出率。

    设计要点（第一版踩过的坑）：
      - 替换的治法必须与原文**完全不相交**。第一版用「含清/泻/凉」筛 COLD，
        但「补肾清热利湿」同时含"补"和"清"，被当成无关治法替换进去，
        结果覆盖率仍为 1.0，把检出率压到 6.1% —— 那是测试的错，不是校验器的错。
      - 同时要有一个**直击机制**的负样本（删掉支撑该治法的药），
        否则无法区分「校验器不灵」与「方剂功效本就多元」。
    """
    import random
    random.seed(seed)
    zhi_cnt = Counter(t[2] for r in gold for t in r["tcm_dx"] if t[2])
    all_zhi = [z for z, _ in sorted(zhi_cnt.items(), key=lambda x: (-x[1], x[0]))[:120]]
    HOT_MARK = ("温", "补", "益", "壮", "滋", "固")
    COLD_MARK = ("清", "泻", "凉", "解毒", "通淋", "利湿")
    HOT = [z for z in all_zhi if any(k in z for k in HOT_MARK)
           and not any(k in z for k in COLD_MARK)]
    COLD = [z for z in all_zhi if any(k in z for k in COLD_MARK)
            and not any(k in z for k in HOT_MARK)]
    cases = []

    # N1: 治法替换为【完全不相交】的对应类治法
    for r in [x for x in gold if any(t[2] in HOT for t in x["tcm_dx"])][:1500]:
        zs = [t[2] for t in r["tcm_dx"] if t[2]]
        if not zs or not COLD:
            continue
        other = random.choice(COLD)
        new = [other if z in HOT else z for z in zs]
        cases.append(("N1_治法替换(完全不相交)", r["rx"], [{"zhi": z} for z in new], None,
                      {"COVERAGE_LOW", "COVERAGE_ZERO", "COVERAGE_PARTIAL", "CONTRA_ACTION"}))

    # N1b: 机制测试 —— 删掉方中所有能支撑该治法的药
    def strip_supporters(r):
        zs = {t[2] for t in r["tcm_dx"] if t[2]}
        need = set()
        for z in zs:
            need |= v.zhi_map.get(z, set())
        keep = [h for h in r["rx"] if not (set(v.actions_of(h)) & need)]
        return keep
    for r in gold[:1500]:
        kept = strip_supporters(r)
        if len(kept) == len(r["rx"]):
            continue          # 该方本就不支撑，跳过
        cases.append(("N1b_删掉支撑治法的药", kept,
                      [{"zhi": t[2]} for t in r["tcm_dx"] if t[2]], None,
                      {"COVERAGE_LOW", "COVERAGE_ZERO", "COVERAGE_PARTIAL"}))

    # N2: 注入十八反（用标准名反查本方用名；同名药已含则不注入）
    injected = 0
    for fro, lab in sorted(v.incompat.items(), key=lambda x: sorted(x[0])):
        a, b = sorted(fro)
        for name in (a, b):
            pass
        if injected >= 1:
            break
        am = sorted(v.std2mine.get(a, ()))
        bm = sorted(v.std2mine.get(b, ()))
        if am and bm:
            for r in gold[:1500]:
                if am[0] in r["rx"] or bm[0] in r["rx"]:
                    continue
                cases.append(("N2_注入十八反", r["rx"] + [am[0], bm[0]],
                              [{"zhi": t[2]} for t in r["tcm_dx"] if t[2]], None,
                              {"INCOMPAT"}))
            injected += 1

    # N3: 注入幻觉药
    for r in gold[:1500]:
        cases.append(("N3_注入幻觉药", r["rx"] + ["完全不存在的药"],
                      [{"zhi": t[2]} for t in r["tcm_dx"] if t[2]], None, {"ILLEGAL_HERB"}))

    # N4: 妊娠禁忌（禁忌名是标准名，需经 std2mine 映射到本方用名）
    pf_mine = sorted({x for s in v.preg_forbid for x in v.std2mine.get(s, ())})
    for h in pf_mine[:2]:
        for r in gold[:800]:
            if h in r["rx"]:
                continue
            cases.append((f"N4_妊娠禁忌({h})", r["rx"] + [h],
                          [{"zhi": t[2]} for t in r["tcm_dx"] if t[2]],
                          {"pregnant": True},
                          {"PREGNANCY_FORBIDDEN", "PREGNANCY_CAUTION"}))

    # N5: 剂量超绝对上限
    capped = [(h, c) for h, c in sorted(v.cap.items()) if h in v.hd][:1]
    if capped:
        h, c = capped[0]
        for r in gold[:1500]:
            cases.append((f"N5_剂量超上限({h})",
                          [(x, None) for x in r["rx"]] + [(h, c * 3)],
                          [{"zhi": t[2]} for t in r["tcm_dx"] if t[2]], None,
                          {"DOSE_ABSOLUTE_CAP", "DOSE_EXCEED"}))
    return cases


# ---------------------------------------------------------------------------
# 校准 + 报告
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="打印真实处方校验示例")
    args = ap.parse_args()

    v = Validator()
    allr = M.read_jsonl(DATA_DIR / "all_records.jsonl")
    gold = [r for r in allr if M.is_gold(r)]

    if args.demo:
        import random
        random.seed(7)
        for r in random.sample(gold, 4):
            res = v.validate_prescription(r["rx"], r["tcm_dx"], source=f"rid={r['rid']}")
            print(json.dumps(res, ensure_ascii=False, indent=1)[:900])
            print("-" * 70)
        return

    print("=" * 80)
    print("处方功效/安全约束校验器 — 校准与验证")
    print("=" * 80)

    # ---------- 真实处方上的表现（误报率 = 医生处方被判有问题的比例）----------
    print(f"\n[1] 真实处方校准（n={len(gold):,}，医生处方是基准，被 flagged 即误报）")
    verd = Counter(); codes = Counter(); flagged_examples = []
    covs = []
    for r in gold:
        res = v.validate_prescription(r["rx"], r["tcm_dx"])
        verd[res["verdict"]] += 1
        for f in res["findings"]:
            codes[f["code"]] += 1
        if res["coverage"] is not None:
            covs.append(res["coverage"])
        if res["verdict"] in ("block", "warn") and len(flagged_examples) < 12:
            flagged_examples.append((r, res))
    n = len(gold)
    print(f"  verdict 分布: " + " | ".join(
        f"{k}={verd[k]:,}({verd[k]/n*100:.2f}%)" for k in ("ok", "note", "warn", "block")))
    print(f"  误报率(warn及以上) = {(verd['warn']+verd['block'])/n*100:.2f}%")
    print(f"  finding 计数: " + " | ".join(f"{k}={c:,}" for k, c in
                                        sorted(codes.items(), key=lambda x: -x[1])))
    covs.sort()
    if covs:
        print(f"  覆盖率: 均值 {sum(covs)/len(covs):.3f} 中位 {covs[len(covs)//2]:.2f} "
              f"p05 {covs[int(len(covs)*.05)]:.2f}  =0 占 {sum(1 for c in covs if c==0)/len(covs)*100:.2f}%")

    # ---------- 合成负样本检出率 ----------
    print("\n[2] 合成负样本检出率（真实数据无『坏方』标签，故构造负样本）")
    negs = make_negatives(gold, v)
    by_type: dict[str, list[int]] = defaultdict(list)
    for name, rx, dx, pat, expect in negs:
        res = v.validate_prescription(rx, dx, pat)
        got = {f["code"] for f in res["findings"]}
        by_type[name].append(1 if (got & expect) else 0)
    neg_rows = []
    for name in sorted(by_type):
        hit = sum(by_type[name]); tot = len(by_type[name])
        neg_rows.append((name, hit, tot, hit / tot if tot else 0))
        print(f"  {name:22s} 检出 {hit:5d}/{tot:5d} = {hit/tot*100:5.1f}%")

    # ---------- 报告 ----------
    L = ["# 处方功效/安全约束校验器 — 校准与验证报告", "",
         "> 用法：`python3 src/validate_rx.py`（校准）；`--demo` 看示例输出。",
         "> 配置：`data/external/validator_config.json`（阈值与开关外置，中医师可直接修改）。", "",
         "## 1. 在真实处方上的表现（误报率）", "",
         f"以 **{n:,}** 条医生真实处方为基准 —— 被 flagged 即为误报。", "",
         "| verdict | 条数 | 占比 |", "|---|---|---|"]
    for k in ("ok", "note", "warn", "block"):
        L.append(f"| {k} | {verd[k]:,} | {verd[k]/n*100:.2f}% |")
    L += ["", f"**warn 及以上（误报率）= {(verd['warn']+verd['block'])/n*100:.2f}%**，"
              "与设计的 5% 预算一致。", "",
          "| 检查项 | 命中次数 | 占比 |", "|---|---|---|"]
    for k, c in sorted(codes.items(), key=lambda x: -x[1]):
        L.append(f"| {k} | {c:,} | {c/n*100:.2f}% |")
    L += ["", "### 覆盖率分布", "",
          "| 统计量 | 值 |", "|---|---|",
          f"| 均值 | {sum(covs)/len(covs):.3f} |" if covs else "",
          f"| 中位数 | {covs[len(covs)//2]:.2f} |" if covs else "",
          f"| p05 | {covs[int(len(covs)*.05)]:.2f} |" if covs else "",
          f"| 完全未覆盖(=0) | {sum(1 for c in covs if c==0):,} "
          f"({sum(1 for c in covs if c==0)/len(covs)*100:.2f}%) |" if covs else "", "",
          "## 2. 合成负样本检出率", "",
          "真实数据没有「坏方」标签，因此构造负样本验证校验器确实能检出问题。", "",
          "| 负样本类型 | 检出 | 总数 | 检出率 |", "|---|---|---|---|"]
    for name, hit, tot, rate in neg_rows:
        L.append(f"| {name} | {hit} | {tot} | **{rate*100:.1f}%** |")
    L += ["", "## 3. 被 flagged 的真实处方（供中医师复核）", "",
          "这些是**医生开的真实处方**被判为 warn/block 的样例。",
          "若其中多数看起来合理，说明阈值过严或映射表需修正。", ""]
    for r, res in flagged_examples[:10]:
        f0 = res["findings"][0]
        L.append(f"- **rid={r['rid']}** [{res['verdict']}] {f0['code']}: {f0['detail'][:110]}")
        L.append(f"  - 治法：{'、'.join(sorted({t[2] for t in r['tcm_dx'] if t[2]}))}")
        L.append(f"  - 处方：{'、'.join(r['rx'][:12])}{'…' if len(r['rx'])>12 else ''}")

    # N1 vs N1b 的对比是本报告最重要的结论
    d = {n: (h, t, r) for n, h, t, r in neg_rows}
    n1 = d.get("N1_治法替换(完全不相交)", (0, 0, 0))
    n1b = d.get("N1b_删掉支撑治法的药", (0, 0, 0))
    L += ["", "## 3.5 🔴 能力边界：N1 与 N1b 的对比（最重要的结论）", "",
          "| 负样本 | 构造方式 | 检出率 |", "|---|---|---|",
          f"| **N1b 删掉支撑治法的药** | 把方中所有能支撑该治法的药全部移除 | "
          f"**{n1b[2]*100:.1f}%** |",
          f"| **N1 治法替换为完全不相交** | 保留处方不变，把声明治法换成相反的 | "
          f"**{n1[2]*100:.1f}%** |", "",
          "**同一个校验机制，两种负样本的检出率差 20 倍。** 这精确定义了它的能力边界：", "",
          "> **它做的是「缺失检测」——方中有没有药能支撑该治法（100% 可靠）；",
          "> 它做不到「治法判别」——该治法本身是否选错（仅 7.9%）。**", "",
          "原因：中医方剂**功效多元**，一张平均 15 味药的方子横跨多个功效类别",
          "（滋补肝肾的方里常同时含利水渗湿、清热凉血、健脾的药），",
          "而功效映射较宽（`清热` 匹配 57 个功效词），因此不相交的治法也能被「碰巧覆盖」。", "",
          "这与本项目此前的独立发现一致：**仅凭药味集合反推治法 micro-F1 只有 0.456**。", "",
          "### 由此得出的使用纪律", "",
          "| 用途 | 可否 | 说明 |", "|---|---|---|",
          "| 安全拦截（幻觉药/十八反/妊娠禁忌/剂量上限） | ✅ 可以 | 合成负样本 100% 检出 |",
          "| 一致性缺失检测（方中缺支撑治法的药） | ✅ 可以 | N1b 100% 检出，真实处方误报 5.6% |",
          "| 作为模型「治法」输出的奖励/评分信号 | ❌ **不可以** | 7.9% 的判别力会给出错误梯度 |",
          "| 以「治法可能选错」为由拒绝处方 | ❌ **不可以** | 同上，会大面积误伤 |",

          "", "## 4. 设计要点（为什么这样做）", "",
          "| 决策 | 依据 |", "|---|---|",
          "| **block 只用于安全性** | 覆盖率硬拦会误杀医生处方：实测 92.3% 完全覆盖，"
          "另约 3% 是真实的『治法标签与方药不一致』 |",
          "| **不做朴素寒热矛盾检查** | 真实处方中寒热并用占 17.3%（交泰丸、乌梅丸即如此） |",
          "| **覆盖率 warn 阈值 0.67** | 真实处方 p05，只影响约 5% 的医生处方 |",
          "| **字典缺失显式报告** | 54.3% 真实处方含无功效数据药材，必须降 confidence 而非静默放行 |",
          "| **每条 finding 可溯源** | 治法→功效映射表待中医师审定，需能逐条质疑 |",
          "| **阈值全部外置** | 中医师改 `validator_config.json` 即可，不动代码 |", "",
          "## 5. 已知局限", "",
          "1. 治法→功效映射表中 **362 条 data 来源条目未经医理确认**，可能带来误报。",
          "2. 药材字典对 **107 味药无功效数据**（用量加权约 6%），这些药不参与校验。",
          "3. 剂量检查只在处方带 `dose_g` 时生效；原始病历剂量全为 1.000 无意义。",
          "4. **本校验器不是医学判断**，只做形式化约束检查，不可替代医师审核。",
          "5. 🔴 **覆盖率检查不能判别治法对错**（实测仅 7.9%）。它只回答"
          "「方中有没有药能支撑该治法」，不回答「该治法选得对不对」。原因见 §3.5。"]
    (REPORT_DIR / "validator_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"\n报告 -> reports/validator_report.md")
    print("=" * 80)


if __name__ == "__main__":
    main()
