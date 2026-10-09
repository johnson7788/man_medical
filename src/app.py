"""
app.py — Gradio 演示（B4）

两个页签，定位不同：

  Tab 1 「处方校验器」 —— **今天就能用**。粘贴一张方子 + 声明治法，
          得到剂量填充、功效覆盖、十八反/妊娠/毒性药/幻觉药检查。
          这条链路已全部交付（§15），不依赖模型。

  Tab 2 「端到端推理」 —— 四诊 → 诊断 + 处方。微调模型尚未训练（§16 A 组），
          因此当前只能跑**规则基线**，并把它的实测水平显式标注出来
          （复诊照抄 0.542 / 初诊检索 0.100 / 端到端规则 0.124），
          以免把基线误当成模型效果。

启动：
  python3 src/app.py                 # http://127.0.0.1:7860
  python3 src/app.py --port 7861
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import medlib as M
from dose_filler import DISCLAIMER, DoseFiller
from validate_rx import LEVEL_ORDER, Validator

ROOT = M.ROOT

# ---------------------------------------------------------------------------
# 输入解析
# ---------------------------------------------------------------------------

# 药名 = 连续中文；其后可跟剂量。用 findall 整体提取，避免「茯苓 12」被空格切碎
# （早期版本按空白切分，导致 "12" 被当成一味药名）。
_HERB_DOSE_RE = re.compile(r"([\u4e00-\u9fff]+)\s*(\d+(?:\.\d+)?)?\s*(?:g|G|克)?")
_SEP_RE = re.compile(r"[,，、;；\s\n]+")


def parse_rx_text(text: str) -> list[dict]:
    """解析用户粘贴的处方文本。

    支持 `北柴胡10g, 茯苓 12, 甘草` / `北柴胡10 白芍12` / 换行分隔。
    药名不含 ASCII 数字（已校验），所以"药名后紧跟数字=剂量"的判定是安全的。
    非中文 token 会被保留为药名，交给校验器按「幻觉药」报出，不静默丢弃。
    """
    text = text or ""
    out: list[dict] = []
    consumed = []
    for m in _HERB_DOSE_RE.finditer(text):
        herb, dose = m.group(1), m.group(2)
        out.append({"herb": herb, "dose_g": float(dose) if dose else None})
        consumed.append((m.start(), m.end()))
    # 收集未被中文模式吃掉的 token（如英文名、错别字），让校验器能报非法药名
    leftover = list(text)
    for a, b in consumed:
        for i in range(a, b):
            leftover[i] = " "
    for tok in _SEP_RE.split("".join(leftover)):
        t = tok.strip()
        if t and not t.isdigit():
            out.append({"herb": t, "dose_g": None})
    return out


def parse_zhi_text(text: str) -> list[dict]:
    items = [z.strip() for z in re.split(r"[,，、;；\s\n]+", text or "") if z.strip()]
    return [{"zhi": z} for z in items]


# ---------------------------------------------------------------------------
# 规则基线（Tab 2，模型未就绪时的降级方案）
# ---------------------------------------------------------------------------

class RuleBaseline:
    """诚实标注的规则基线：西医诊断→诊断三元组（D2），处方用"照抄上次方 / 全局常用方"。

    实测水平（§1.6 / §1.8，患者分组切分）：
      端到端规则 R0 = 0.124 ；复诊照抄 = 0.542 ；初诊检索 top-1 = 0.100
    """

    def __init__(self):
        import baseline as B
        allr = M.read_jsonl(M.DATA_DIR / "all_records.jsonl")
        gold = [r for r in allr if M.is_gold(r)]
        self.train, self.test = M.split_by_patient(gold)["train"], M.split_by_patient(gold)["test"]
        self.dx_pred = B.make_dx_predictor(self.train, B.wm_key, 3)
        self.constant_rx = M.top_herbs(self.train, 15)
        # 证型 → 治法 的确定性映射（实测段级 top1 占比 85.7%）
        from collections import Counter as _C
        zhi_cnt: dict[str, _C] = {}
        for r in self.train:
            for t in r["tcm_dx"]:
                if t[1] and t[2]:
                    zhi_cnt.setdefault(t[1], _C())[t[2]] += 1
        self.zhi_of = {z: dict(c) for z, c in zhi_cnt.items()}

    def predict(self, sex: str, age: int, cc: str, hpi: str, wm_dx: str,
                prev_rx: str) -> dict:
        rec = {"sex": sex, "age": int(age or 0), "cc": cc, "hpi": hpi,
               "wm_dx": [x.strip() for x in re.split(r"[,，;；]", wm_dx or "") if x.strip()],
               "tcm_dx": []}
        dx = self.dx_pred(rec)
        # 证型 → 治法（数据里的确定性映射）
        zhis = []
        for b, z, t in dx:
            cand = self.zhi_of.get(z)
            if cand:
                best = sorted(cand.items(), key=lambda x: (-x[1], x[0]))[0][0]
                zhis.append((b, z, best))
            else:
                zhis.append((b, z, t))
        prev = parse_rx_text(prev_rx) if prev_rx else []
        if prev:
            rx = [p["herb"] for p in prev]
            source = "复诊：沿用上次处方（规则基线，实测 Jaccard 0.542）"
        else:
            rx = list(self.constant_rx)
            source = "初诊：全局常用方（规则基线，实测 Jaccard 0.159）"
        return {"dx": dx, "zhis": zhis, "rx": rx, "source": source}


# ---------------------------------------------------------------------------
# 呈现
# ---------------------------------------------------------------------------

LEVEL_ICON = {"block": "🔴", "warn": "🟠", "note": "🟡", "info": "🔵"}


def render_validation(res: dict) -> str:
    L = [f"### 校验结论：`{res['verdict']}`",
         f"功效覆盖率 **{res['coverage']:.2f}**" if res["coverage"] is not None else "功效覆盖率 —（未提供治法）",
         f"字典覆盖置信度 {res['confidence']:.0%}（{res['n_herbs']} 味中 "
         f"{round(res['confidence']*res['n_herbs'])} 味有功效数据）", ""]
    if res["missing_zhis"]:
        L.append(f"**未覆盖治法**：{'、'.join(res['missing_zhis'])}")
    if res["findings"]:
        L.append("")
        L.append("| 级别 | 检查项 | 说明 |")
        L.append("|---|---|---|")
        for f in res["findings"]:
            L.append(f"| {LEVEL_ICON.get(f['level'],'')} {f['level']} | `{f['code']}` | "
                     f"{f['detail'][:220]} |")
    else:
        L.append("\n未发现问题。")
    if res.get("handling_alerts"):
        L.append("")
        L.append("**用药提示**（不计入结论）：")
        for a in res["handling_alerts"]:
            L.append(f"- {a['detail']}")
            for s in a.get("safety_excerpts", [])[:2]:
                L.append(f"    - {s}")
    # 能力边界必须写清楚，否则会被误读成"处方质量已通过审核"
    L.append("")
    L.append("> ⚠️ 本校验器**只做缺失检测**（方中有没有药支撑该治法），"
             "**不做治法判别**（该治法选得对不对）——实测判别力仅 7.9%。"
             "它不能替代医师审方。")
    return "\n".join(L)


def make_app():
    import gradio as gr

    v = Validator()
    dfiller = DoseFiller()
    rule = RuleBaseline()

    with gr.Blocks(title="中医男科 · 处方辅助与校验") as demo:
        gr.Markdown(
            "# 中医男科 · 处方辅助与校验\n"
            "> ⚠️ **本系统为医师辅助/教学工具，不是自动开方，不构成医疗建议。**\n"
            "> 剂量由标准量表机械填充，**须由执业医师核定**。")

        with gr.Tabs():
            # ---------------- Tab 1：校验器（今天可用） ----------------
            with gr.Tab("① 处方校验器（已交付，可直接用）"):
                gr.Markdown(
                    "粘贴一张处方与声明治法，检查：**药名白名单 / 十八反十九畏 / 妊娠禁忌 / "
                    "剂量上限 / 毒性药提示 / 功效是否覆盖治法**。\n\n"
                    "治法留空则跳过功效检查。")
                with gr.Row():
                    with gr.Column():
                        rx_in = gr.Textbox(
                            label="处方（逗号或空格分隔，可带剂量如 北柴胡10g）", lines=5,
                            value="北柴胡10 白芍12 麸炒枳实10 炙甘草6 当归10 "
                                  "茯苓15 麸炒苍术10 干姜6 细辛3")
                        zhi_in = gr.Textbox(label="声明治法（逗号分隔，可留空）", value="疏肝解郁,温阳散寒")
                        preg_in = gr.Checkbox(label="妊娠期", value=False)
                        btn1 = gr.Button("校验", variant="primary")
                    with gr.Column():
                        out1 = gr.Markdown()
                tbl1 = gr.Dataframe(label="剂量填充（来自本草典标准量表）",
                                    headers=["药名", "标准名", "建议剂量(g)", "字典区间",
                                             "有毒/需特殊处理", "依据", "原文说明"],
                                    wrap=True)

                def run_tab1(rx_text, zhi_text, preg):
                    rx = parse_rx_text(rx_text)
                    res = v.validate_prescription(rx, parse_zhi_text(zhi_text),
                                                  {"pregnant": bool(preg)})
                    filled = dfiller.fill([x["herb"] for x in rx])
                    rows = []
                    for it in filled["items"]:
                        rows.append([
                            it["herb"], it.get("standard_name") or "—",
                            f"{it['dose_g']:g}" if it["dose_g"] is not None else "待人工填写",
                            f"{it['range'][0]:g}–{it['range'][1]:g}{it['unit']}"
                            if it["range"] else "—",
                            "⚠️ 是" if it["is_toxic"] else "否",
                            it["reason"], (it.get("notes") or "")[:80],
                        ])
                    md = render_validation(res)
                    md += (f"\n\n---\n### 剂量填充汇总\n共 {filled['n_herbs']} 味，"
                           f"填充 {filled['n_filled']} 味，合计 {filled['total_g']}g"
                           + (f"，**{filled['n_unknown']} 味缺剂量数据**："
                              f"{'、'.join(filled['unknown_herbs'])}" if filled["n_unknown"] else ""))
                    md += f"\n\n> {DISCLAIMER}"
                    return md, rows

                btn1.click(run_tab1, [rx_in, zhi_in, preg_in], [out1, tbl1])
                demo.load(run_tab1, [rx_in, zhi_in, preg_in], [out1, tbl1])

            # ---------------- Tab 2：端到端（模型未训练） ----------------
            with gr.Tab("② 端到端推理（模型未训练，当前为规则基线）"):
                gr.Markdown(
                    "#### ⚠️ 微调模型尚未训练\n"
                    "本页当前只能跑**规则基线**，其实测水平如下（患者分组切分，详见 §1.6/§1.8）：\n\n"
                    "| 口径 | 处方 Jaccard |\n|---|---|\n"
                    "| 复诊沿用上次方 | 0.542 |\n"
                    "| 初诊（全局常用方） | 0.159 |\n"
                    "| 端到端规则 R0（先预测诊断再检索） | 0.124 |\n\n"
                    "**这些数字是基线，不是模型效果。** 模型目标见 §4.2（复诊 ≥0.45 / 初诊 ≥0.22）。\n"
                    "缺算力见 §16 A 组；训练脚本 `src/train_sft.py` 已就绪。")
                with gr.Row():
                    with gr.Column():
                        sex = gr.Dropdown(["男", "女"], value="男", label="性别")
                        age = gr.Number(value=41, label="年龄", precision=0)
                        cc = gr.Textbox(label="主诉", value="乏力困倦一年")
                        hpi = gr.Textbox(label="现病史", lines=4,
                                         value="乏力困倦，腰痛，怕冷，下午头痛，性功能减退，"
                                               "勃起不坚，时间短，无阴囊潮湿，大便不成形黏，睡眠多梦，易怒")
                        wm = gr.Textbox(label="西医诊断（逗号分隔）",
                                        value="慢性前列腺炎,男性性腺功能低下")
                        prev = gr.Textbox(label="既往处方（复诊时填，初诊留空）", lines=2, value="")
                        btn2 = gr.Button("推理", variant="primary")
                    with gr.Column():
                        out2 = gr.Markdown()
                        tbl2 = gr.Dataframe(label="处方（含剂量填充）",
                                            headers=["药名", "建议剂量(g)", "字典区间",
                                                     "有毒/需特殊处理", "依据"], wrap=True)

                def run_tab2(sex, age, cc, hpi, wm, prev):
                    p = rule.predict(sex, age, cc, hpi, wm, prev)
                    dx_md = "\n".join(f"- {b} · {z} · {t}" for b, z, t in p["zhis"])
                    zhis = [{"zhi": t} for _, _, t in p["zhis"]]
                    res = v.validate_prescription([{"herb": h} for h in p["rx"]], zhis)
                    filled = dfiller.fill(p["rx"])
                    rows = [[it["herb"],
                             f"{it['dose_g']:g}" if it["dose_g"] is not None else "待人工填写",
                             f"{it['range'][0]:g}–{it['range'][1]:g}{it['unit']}"
                             if it["range"] else "—",
                             "⚠️ 是" if it["is_toxic"] else "否", it["reason"]]
                            for it in filled["items"]]
                    md = (f"### 中医诊断（规则基线）\n{dx_md}\n\n"
                          f"### 处方\n**来源**：{p['source']}\n\n"
                          f"`{'、'.join(p['rx'])}`\n\n---\n")
                    md += render_validation(res)
                    md += (f"\n\n---\n> ⚠️ 上述为**规则基线**输出，非模型结果。\n> {DISCLAIMER}")
                    return md, rows

                btn2.click(run_tab2, [sex, age, cc, hpi, wm, prev], [out2, tbl2])

    return demo


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="只构建界面并校验依赖，不启动服务（供 CI/流水线使用）")
    args = ap.parse_args()
    app = make_app()
    if args.smoke:
        n = len(app.blocks)
        print(f"✅ 界面构建成功：{n} 个组件；启动命令 python3 src/app.py --port {args.port}")
        return
    app.launch(server_name=args.host, server_port=args.port, share=args.share,
               show_error=True)


if __name__ == "__main__":
    main()
