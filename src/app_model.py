"""
app_model.py — 接入微调模型的 Gradio 演示

与 `app.py` 的区别：`app.py` 的推理页签只能跑**规则基线**；本脚本加载
**MiMo-V2.6-Distill-Qwen-9B + LoRA 适配器**，做真正的端到端推理。

设计要点（都是为了"演示的结论可信"）：

  1. **输入构造与训练完全一致**：复用 `medlib.format_input_v2`。
     口径不一致会让演示效果与评测数字对不上（本项目踩过：口径错一次，数字全废）。
  2. **无条件 left padding**：decoder-only + right padding 会让短 prompt 提前 EOS。
     实测踩过：17/64 条输出只剩两个空标题行，且全是短输入的初诊。
  3. **后置药名归一化**：把炮制/产地变体映射回 462 白名单，并报告越界率。
     不做这一步，模型偶发的写法差异会被误当成"幻觉药名"。
  4. **剂量不由模型生成**：模型只出药味，剂量由标准量表填充并标注需医师核定。
  5. **复诊显示"相对上次方的增删"**：本数据集 63.9% 是复诊，其最强预测因子
     就是上次处方。只给一张新方无法判断模型是"照抄"还是"调整"。
  6. **诚实标注能力边界**：校验器的治法判别力实测仅 7.9%，页面上必须说清楚，
     不能让它看起来像"处方质量已通过审核"。

启动：
  python3 src/app_model.py \
      --base models/MiMo-V2.6-Distill-Qwen-9B \
      --adapter outputs/mimo9b-tcm-lora --port 7860

  无 GPU 时只跑校验器页签：
  python3 src/app_model.py --no-model
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import medlib as M
from app import LEVEL_ICON
from dose_filler import DISCLAIMER, DoseFiller
from validate_rx import Validator

MAX_NEW_TOKENS = 192

# ---------------------------------------------------------------------------
# 后端：模型 + 校验器 + 剂量表
# ---------------------------------------------------------------------------

class Backend:
    """把模型推理、药名归一化、剂量填充、校验串成一条链。"""

    def __init__(self, base: str | None, adapter: str | None):
        self.validator = Validator()
        self.filler = DoseFiller()
        self.herbs = self._load_whitelist()
        self.norm = self._make_normalizer()
        self.tok = None
        self.model = None
        self.base, self.adapter = base, adapter
        if base:
            self._load_model(base, adapter)

    # -- 白名单 / 归一化 -------------------------------------------------
    @staticmethod
    def _load_whitelist() -> list[str]:
        from constrained_decode import load_whitelist
        return load_whitelist()

    def _make_normalizer(self):
        """模型输出的药名 → 462 白名单规范名（复用字典那套剥离规则）。"""
        from build_external_data import ALIAS, ORIGIN_PREFIX, PROC_PREFIX, SUFFIX
        wl = set(self.herbs)

        def norm(h: str) -> str:
            if h in wl:
                return h
            if h in ALIAS and ALIAS[h] in wl:
                return ALIAS[h]
            c = h
            for _ in range(3):
                c2 = SUFFIX.sub("", ORIGIN_PREFIX.sub("", PROC_PREFIX.sub("", c)))
                if c2 in wl:
                    return c2
                if c2 == c:
                    break
                c = c2
            return h
        return norm

    # -- 模型 -------------------------------------------------------------
    def _load_model(self, base: str, adapter: str | None) -> None:
        from train_sft import load_model_for_inspect
        t0 = time.time()
        self.model, self.tok, used, arch = load_model_for_inspect(base, "none", "bfloat16")
        if adapter:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter)
        # 必须 left padding，见模块 docstring 第 2 条
        self.tok.padding_side = "left"
        self.model.eval()
        print(f"[后端] 模型已加载（{used} / {arch}），适配器={adapter}，"
              f"用时 {time.time()-t0:.1f}s")

    @property
    def ready(self) -> bool:
        return self.model is not None

    # -- 生成 -------------------------------------------------------------
    def generate(self, sex: str, age: int, cc: str, hpi: str, wm_dx: str,
                 prev_rx: str) -> dict:
        import torch

        rec = self._to_record(sex, age, cc, hpi, wm_dx, prev_rx)
        user = M.format_input_v2(rec)
        msgs = [{"role": "system", "content": M.SYSTEM_PROMPT},
                {"role": "user", "content": user}]
        prompt = self.tok.apply_chat_template(msgs, tokenize=False,
                                              add_generation_prompt=True)
        assert self.tok.padding_side == "left"
        enc = self.tok([prompt], return_tensors="pt", padding=True).to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                pad_token_id=self.tok.pad_token_id, eos_token_id=self.tok.eos_token_id)
        raw = self.tok.batch_decode(out[:, enc["input_ids"].shape[1]:],
                                    skip_special_tokens=True)[0]
        return self.postprocess(raw, rec, user)

    def _to_record(self, sex, age, cc, hpi, wm_dx, prev_rx) -> dict:
        prev = [h for h in re.split(r"[,，、;；\s\n]+", prev_rx or "") if h.strip()]
        rec = {"rid": -1, "sex": sex, "age": int(age or 0), "cc": cc or "",
               "hpi": hpi or "", "wm_dx": [x.strip() for x in
                                           re.split(r"[,，;；]", wm_dx or "") if x.strip()],
               "tcm_dx": [], "rx": [], "prev_rx": [], "prev_gap": None,
               "visit_ix": 0}
        if prev:
            rec["prev_rx"] = prev
        return rec

    def postprocess(self, raw: str, rec: dict, user_text: str) -> dict:
        text = re.sub(r"<think>.*?</think>", "", raw, flags=re.S)
        text = text.replace("<think>", "").replace("</think>", "").strip()
        dx, rx_raw = M.parse_assistant(text)
        rx, bad = [], []
        for h in rx_raw:
            n = self.norm(h)
            if n not in set(self.herbs):
                bad.append(h)
            if n not in rx:
                rx.append(n)
        zhis = [{"zhi": t} for _, _, t in dx if t]
        val = self.validator.validate_prescription([{"herb": h} for h in rx], zhis)
        filled = self.filler.fill(rx)
        result = {"user": user_text, "raw": raw, "text": text, "dx": dx,
                  "rx": rx, "bad_herbs": bad, "validation": val, "doses": filled}
        if rec.get("prev_rx"):
            result["diff"] = self._diff(rec["prev_rx"], rx)
        return result

    @staticmethod
    def _diff(prev: list[str], pred: list[str]) -> dict:
        a, b = set(prev), set(pred)
        inter = a & b
        return {"jaccard": len(inter) / len(a | b) if (a | b) else 0.0,
                "kept": [h for h in prev if h in inter],
                "added": [h for h in pred if h not in a],
                "removed": [h for h in prev if h not in b]}

    # -- 校验器单独入口（Tab 2，不需要模型）------------------------------
    def validate_only(self, rx_text: str, zhi_text: str, pregnant: bool) -> tuple[str, list]:
        from app import parse_rx_text, parse_zhi_text, render_validation
        rx = parse_rx_text(rx_text)
        res = self.validator.validate_prescription(rx, parse_zhi_text(zhi_text),
                                                   {"pregnant": bool(pregnant)})
        filled = self.filler.fill([x["herb"] for x in rx])
        rows = [[it["herb"], it.get("standard_name") or "—",
                 f"{it['dose_g']:g}" if it["dose_g"] is not None else "待人工填写",
                 f"{it['range'][0]:g}–{it['range'][1]:g}{it['unit']}" if it["range"] else "—",
                 "⚠️ 是" if it["is_toxic"] else "否", it["reason"]]
                for it in filled["items"]]
        md = render_validation(res)
        md += (f"\n\n---\n### 剂量填充\n共 {filled['n_herbs']} 味，填充 {filled['n_filled']} 味，"
               f"合计 {filled['total_g']}g")
        md += f"\n\n> {DISCLAIMER}"
        return md, rows


# ---------------------------------------------------------------------------
# 呈现
# ---------------------------------------------------------------------------

def render_diagnosis(res: dict) -> str:
    if not res["dx"]:
        return "⚠️ **模型未输出可解析的中医诊断**（输出被截断或格式偏离）"
    L = ["| 病名 | 证型 | 治法 |", "|---|---|---|"]
    for b, z, t in res["dx"]:
        L.append(f"| {b or '—'} | {z or '—'} | {t or '—'} |")
    return "\n".join(L)


def render_rx(res: dict) -> str:
    if not res["rx"]:
        return "⚠️ **模型未输出可解析的处方**"
    L = [f"**共 {len(res['rx'])} 味**", "", "`" + " ".join(res["rx"]) + "`"]
    if res["bad_herbs"]:
        L += ["", f"⚠️ 归一化后仍在白名单外的药名（**需人工确认**）："
                  f"{'、'.join(res['bad_herbs'])}"]
    d = res.get("diff")
    if d:
        L += ["", "### 相对上次处方的调整（复诊）",
              f"- 与上次方重合度 Jaccard = **{d['jaccard']:.3f}**",
              f"- 沿用 {len(d['kept'])} 味",
              f"- **新增** {len(d['added'])} 味：{'、'.join(d['added']) or '无'}",
              f"- **去掉** {len(d['removed'])} 味：{'、'.join(d['removed']) or '无'}",
              "", "> 复诊场景下最强基线就是「照抄上次方」（Jaccard 0.542，见评测报告）。"
                  "请重点核对上面的**新增/去掉**是否合理。"]
    return "\n".join(L)


def render_validation_md(val: dict) -> str:
    """复用 `app.render_validation`，保证两个页签的校验呈现与免责提示完全一致。

    早期版本在 app_model 里另写了一份，导致 Tab2 缺了「校验器只做缺失检测、
    不做治法判别（判别力 7.9%）」这条能力边界提示 —— 同一事实两处实现必然漂移。
    """
    from app import render_validation
    return render_validation(val)


# ---------------------------------------------------------------------------
# 界面
# ---------------------------------------------------------------------------

EXAMPLES = [
    ["男", 33, "腰痛乏力复诊",
     "腰痛乏力，勃起不坚，晨勃（+），汗不多，口不干，手心出汗，脚凉，夜尿不多，纳可，大便日1行，眠可，心悸。",
     "男性性腺功能低下", ""],
    ["男", 31, "尿频2月",
     "尿频，腰痛，射精快，睡眠正常，排尿正常，大便粘，口干口苦，盗汗",
     "男性性腺功能低下", ""],
    ["男", 28, "腰痛伴中途疲软3天",
     "腰痛伴中途疲软3天 盗汗", "腰痛",
     "炙甘草 北柴胡 麸炒枳壳 枸杞子 菟丝子 韭菜子 白芍 制远志 酒苁蓉 "
     "石菖蒲 炒蒺藜 当归 炙淫羊藿 川芎 锁阳 制巴戟天"],
]


def make_app(be: Backend):
    import gradio as gr

    with gr.Blocks(title="中医男科 · 诊断与处方辅助") as demo:
        gr.Markdown(
            "# 中医男科 · 诊断与处方辅助\n"
            "> ⚠️ **医师辅助 / 教学工具，不是自动开方，不构成医疗建议。**\n"
            "> 剂量由标准量表机械填充，**须由执业医师核定**。")

        with gr.Tabs():
            # ---------------- Tab 1：模型推理 ----------------
            with gr.Tab("① 模型推理（四诊 → 诊断 + 处方）"):
                if be.ready:
                    gr.Markdown(
                        "模型：**MiMo-V2.6-Distill-Qwen-9B + LoRA**"
                        f"（适配器 `{be.adapter}`）\n\n"
                        "实测水平（冻结测试集 2,764 条，患者分组切分）："
                        "**总体 Jaccard 0.429**、初诊 **0.301**（超过 Oracle 上界 0.254）、"
                        "复诊 0.501（照抄基线 0.542）。")
                else:
                    gr.Markdown("⚠️ **未加载模型**（用 `--no-model` 启动），"
                                "本页不可用；请用下面的「处方校验器」。")

                with gr.Row():
                    with gr.Column(scale=1):
                        sex = gr.Dropdown(["男", "女"], value="男", label="性别")
                        age = gr.Number(value=33, label="年龄", precision=0)
                        cc = gr.Textbox(label="主诉", value="", lines=2)
                        hpi = gr.Textbox(label="现病史", value="", lines=5)
                        wm = gr.Textbox(label="西医诊断（逗号分隔）", value="")
                        prev = gr.Textbox(
                            label="既往处方（复诊时填，初诊留空）", lines=3, value="",
                            info="填了会按复诊处理，并给出「相对上次方的增删」")
                        btn = gr.Button("生成", variant="primary")
                        gr.Examples(examples=EXAMPLES,
                                    inputs=[sex, age, cc, hpi, wm, prev],
                                    label="载入数据集中的真实病例")
                    with gr.Column(scale=2):
                        out_dx = gr.Markdown(label="中医诊断")
                        out_rx = gr.Markdown(label="中药处方")
                        out_dose = gr.Dataframe(
                            headers=["药名", "建议剂量(g)", "字典区间", "有毒/需特殊处理", "依据"],
                            label="剂量填充（来自药典标准量表）", wrap=True)
                        out_val = gr.Markdown(label="处方校验")

                def run(sex, age, cc, hpi, wm, prev):
                    if not be.ready:
                        return ("模型未加载", "", [], "")
                    if not (cc or hpi):
                        return ("请至少填写主诉或现病史", "", [], "")
                    r = be.generate(sex, age, cc, hpi, wm, prev)
                    rows = [[it["herb"],
                             f"{it['dose_g']:g}" if it["dose_g"] is not None else "待人工填写",
                             f"{it['range'][0]:g}–{it['range'][1]:g}{it['unit']}"
                             if it["range"] else "—",
                             "⚠️ 是" if it["is_toxic"] else "否",
                             it["reason"]] for it in r["doses"]["items"]]
                    dose_md = (f"共 {r['doses']['n_herbs']} 味，填充 {r['doses']['n_filled']} 味，"
                               f"合计 {r['doses']['total_g']}g"
                               + (f"，**{r['doses']['n_unknown']} 味缺剂量数据**"
                                  if r["doses"]["n_unknown"] else ""))
                    return (render_diagnosis(r),
                            render_rx(r) + f"\n\n---\n### 剂量\n{dose_md}\n\n> {DISCLAIMER}",
                            rows, render_validation_md(r["validation"]))

                btn.click(run, [sex, age, cc, hpi, wm, prev],
                          [out_dx, out_rx, out_dose, out_val])

            # ---------------- Tab 2：校验器（不需要模型）----------------
            with gr.Tab("② 处方校验器（手工输入）"):
                gr.Markdown("粘贴任意处方，检查：药名白名单 / 十八反十九畏 / 妊娠禁忌 / "
                            "剂量上限 / 毒性药提示 / 功效是否覆盖治法。")
                with gr.Row():
                    with gr.Column():
                        rx_in = gr.Textbox(label="处方（逗号或空格分隔，可带剂量）", lines=5,
                                           value="北柴胡10 白芍12 麸炒枳实10 炙甘草6 "
                                                 "当归10 茯苓15 麸炒苍术10 干姜6 细辛3")
                        zhi_in = gr.Textbox(label="声明治法（逗号分隔，可留空）",
                                            value="疏肝解郁,温阳散寒")
                        preg = gr.Checkbox(label="妊娠期", value=False)
                        b2 = gr.Button("校验", variant="primary")
                    with gr.Column():
                        o2 = gr.Markdown()
                t2 = gr.Dataframe(headers=["药名", "标准名", "建议剂量(g)", "字典区间",
                                           "有毒/需特殊处理", "依据"], wrap=True)
                b2.click(be.validate_only, [rx_in, zhi_in, preg], [o2, t2])

    return demo


def main() -> None:
    ap = argparse.ArgumentParser()
    # 默认值与 README/训练指南保持一致（下载到 models/、训练产出到 outputs/）
    ap.add_argument("--base", default="models/MiMo-V2.6-Distill-Qwen-9B")
    ap.add_argument("--adapter", default="outputs/mimo9b-tcm-lora")
    ap.add_argument("--no-model", action="store_true", help="只启动校验器页签")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="只构建界面并退出（CI 用）")
    ap.add_argument("--auth", default=None, metavar="USER:PASS",
                    help="启用基础认证。公网暴露时强烈建议开启，"
                         "否则任何人都能白用你的 GPU。本地使用不必开。")
    args = ap.parse_args()

    be = Backend(None if args.no_model else args.base,
                 None if args.no_model else args.adapter)
    app = make_app(be)
    if args.smoke:
        print(f"✅ 界面构建成功：{len(app.blocks)} 个组件；模型就绪={be.ready}")
        return
    auth = None
    if args.auth:
        if ":" not in args.auth:
            raise SystemExit("--auth 格式应为 USER:PASS")
        u, pw = args.auth.split(":", 1)
        auth = (u, pw)
        print(f"已启用基础认证（用户 {u}）")
    app.launch(server_name=args.host, server_port=args.port, share=args.share,
               auth=auth, show_error=True)


if __name__ == "__main__":
    main()
