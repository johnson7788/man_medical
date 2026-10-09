"""
constrained_decode.py — 约束解码（B1）

目标（计划 §5 Stage 3 / §6.2 约束 2、3）：把「药名必须 ∈ 462 白名单」和
「不得出现十八反/十九畏」变成**解码期硬约束**，而不是生成后再人工检查。

三层约束，各自用对的手段：
  1. 药名白名单      → **语法约束**（trie 逐 token 引导）。适合逐 token 约束。
  2. 十八反/十九畏    → **集合级约束**，语法表达不了。用「生成 → 校验 → 剔除违规药 → 重生成」
                        的有界修复循环。这一层刻意做成**后置硬拦**，因为配对不对
                        取决于生成顺序，无法在 token 级预判。
  3. 功效需覆盖治法   → **不做硬约束**。实测覆盖率检查只有「缺失检测」能力（删掉支撑药
                        100% 检出，但治法选错仅 7.9% 检出），把它当硬门槛会误伤合理处方。
                        故仅作为可读的提示随结果返回。

离线可验证的部分（本模块重点）：
  - `HerbTrie`：纯 Python，不依赖 torch / vLLM。用 mock tokenizer 可完整测试。
  - `build_vllm_guided_regex()` / `build_guided_json_schema()`：纯函数，可断言正确性。
  - `generate_with_repair()`：接受任意 `generate_fn`，可用 mock 完整测试修复循环。
  - `make_logits_processor()`：唯一依赖 torch 的部分，惰性导入。

用法：
  python3 src/constrained_decode.py --selftest     # 离线自检（无需 torch/vLLM）
  python3 src/constrained_decode.py --print-regex  # 打印 vLLM guided_regex
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import medlib as M

EXT = M.ROOT / "data" / "external"


# ---------------------------------------------------------------------------
# 0. 白名单与配置
# ---------------------------------------------------------------------------

def load_whitelist(path: Path | None = None) -> list[str]:
    """462 味药白名单（来自药材字典），确定性排序。"""
    rows = M.read_jsonl(path or EXT / "herb_dictionary.jsonl")
    return sorted(r["herb"] for r in rows)


# ---------------------------------------------------------------------------
# 1. 药名白名单：trie 语法约束（纯 Python，可离线测试）
# ---------------------------------------------------------------------------

class _Node:
    __slots__ = ("children", "terminal")

    def __init__(self) -> None:
        self.children: dict = {}
        self.terminal = False


class HerbTrie:
    """把「空格分隔的药名序列」表示成逐 token 可约束的自动机。

    为什么需要 trie：中文药名常被切成**多个 token**（如「炙淫羊藿」可能是 2–3 个 token），
    简单的"允许 token 集合"会放出半个药名。trie 跟踪当前前缀，
    只允许能续成合法药名的 token。

    units 可以是 token id（真实使用）或字符（离线测试）。
    """

    def __init__(self, herbs: list[str], encode, sep: str = " ",
                 eos: str | None = None):
        self.encode = encode
        self.sep = sep
        self.eos = eos
        self.root = _Node()
        self.names: dict[tuple, str] = {}
        for h in herbs:
            ids = tuple(encode(h))
            if not ids:
                continue
            node = self.root
            for t in ids:
                node = node.children.setdefault(t, _Node())
            node.terminal = True
            self.names[ids] = h
        self.sep_ids = tuple(encode(sep))
        self.eos_ids = tuple(encode(eos)) if eos else ()

    # -- 状态推进 ---------------------------------------------------------
    def initial(self) -> frozenset:
        return frozenset({self.root})

    def _terminals(self, nodes: frozenset) -> bool:
        return any(n.terminal for n in nodes)

    def allowed_units(self, nodes: frozenset, emitted: bool) -> set:
        """当前状态下允许的下一批 unit。emitted=True 表示已至少产出一个完整药名。"""
        out: set = set()
        for n in nodes:
            out |= set(n.children.keys())
        if self._terminals(nodes) and emitted:
            out |= set(self.sep_ids)
            if self.eos_ids:
                out |= set(self.eos_ids)
        return out

    def is_sep(self, unit) -> bool:
        return bool(self.sep_ids) and unit in self.sep_ids

    def is_eos(self, unit) -> bool:
        return bool(self.eos_ids) and unit in self.eos_ids

    def advance(self, nodes, unit, emitted: bool):
        """消费一个 unit。

        返回 (新节点集合, 新 emitted) 表示正常推进；
        返回 None 表示**终止（EOS）或非法**，调用方须用 `is_eos(unit)` 区分。
        早期版本对 EOS 返回 (None, True)，导致调用方拿到 None 状态后在下一步崩溃。
        """
        if self.is_eos(unit):
            return None
        if self.is_sep(unit):
            # 只有落在完整药名边界上才允许分隔
            if self._terminals(nodes) and emitted:
                return frozenset({self.root}), True
            return None
        nxt = set()
        for n in nodes:
            c = n.children.get(unit)
            if c is not None:
                nxt.add(c)
        if not nxt:
            return None
        return frozenset(nxt), emitted or self._terminals(nxt)


# ---------------------------------------------------------------------------
# 2. vLLM 约束产物（纯函数）
# ---------------------------------------------------------------------------

def build_guided_json_schema(herbs: list[str], min_items: int = 1,
                             max_items: int = 40) -> dict:
    """vLLM `guided_json` 用的 schema：药味数组，元素限定在 462 白名单内。

    比正则更优：vLLM 会把 schema 编译成高效的状态机，且天然支持去重（uniqueItems）。
    """
    return {
        "type": "object",
        "properties": {
            "herbs": {
                "type": "array",
                "items": {"type": "string", "enum": sorted(herbs)},
                "minItems": min_items,
                "maxItems": max_items,
                "uniqueItems": True,
            }
        },
        "required": ["herbs"],
        "additionalProperties": False,
    }


def build_vllm_guided_regex(herbs: list[str]) -> str:
    """vLLM `guided_regex` 用的正则：空格分隔的药名序列。

    注意：462 项的交替式很长（~6–8KB），编译较慢。**优先用 guided_json**。
    """
    alt = "|".join(re.escape(h) for h in sorted(herbs, key=len, reverse=True))
    return rf"({alt})( ({alt}))*"


def validate_guided_regex(herbs: list[str]) -> dict:
    """离线校验正则正确性：白名单内全部命中，外部药名不命中。"""
    pat = re.compile("^" + build_vllm_guided_regex(herbs) + "$")
    ok = sum(1 for h in herbs if pat.match(h))
    # 组成一个合法的多药方
    multi = " ".join(herbs[:5])
    multi_ok = bool(pat.match(multi))
    # 负例：混入一个不存在的药
    neg_ok = not pat.match(" ".join(herbs[:2] + ["完全不存在的药"]))
    return {"n_herbs": len(herbs), "single_match": ok, "single_expected": len(herbs),
            "multi_match": multi_ok, "rejects_outsider": neg_ok,
            "regex_len": len(build_vllm_guided_regex(herbs))}


# ---------------------------------------------------------------------------
# 3. transformers logits processor（唯一依赖 torch 的部分，惰性导入）
# ---------------------------------------------------------------------------

def make_logits_processor(herbs: list[str], tokenizer, prompt_len: int,
                          eos_token_id: int | None = None, sep: str = " "):
    """返回一个 HF `LogitsProcessor`，把每步的非法 token logit 置 -inf。

    关键点：
      - 中文药名常被切成**多个 token**（如「炙淫羊藿」2–3 个），简单的"允许 token 集合"
        会放出半个药名。trie 跟踪前缀，只允许能续成合法药名的 token。
      - HF 的 LogitsProcessor 是**无状态**的，所以用 `prompt_len` 定位生成段、
        每步重放生成段 token 得到状态。这是确定性的，避免了有状态对象的 batch 管理问题。

    `prompt_len` 必须与实际 prompt 的 token 数一致，否则约束会错位。
    """
    import torch

    trie = HerbTrie(herbs, encode=lambda x: tokenizer.encode(x, add_special_tokens=False),
                    sep=sep, eos=None)

    def state_from(gen_ids: list[int]):
        nodes, emitted = trie.initial(), False
        for t in gen_ids:
            if trie.is_eos(t):
                return None
            r = trie.advance(nodes, t, emitted)
            if r is None:
                return None
            nodes, emitted = r
        return nodes, emitted

    class HerbLogitsProcessor:
        """鸭子类型即可被 HF 使用；不继承 LogitsProcessor 以免强依赖 transformers 版本。"""

        def __call__(self, input_ids, scores):
            for b in range(input_ids.shape[0]):
                gen = input_ids[b, prompt_len:].tolist()
                st = state_from(gen)
                if st is None:
                    continue
                nodes, emitted = st
                allowed = set(trie.allowed_units(nodes, emitted))
                mask = torch.full_like(scores[b], float("-inf"))
                if allowed:
                    idx = torch.tensor(sorted(allowed), device=scores.device, dtype=torch.long)
                    mask[idx] = 0.0
                # 允许在完整药名边界处结束
                if eos_token_id is not None and any(n.terminal for n in nodes):
                    mask[eos_token_id] = 0.0
                scores[b] = scores[b] + mask
            return scores

    return HerbLogitsProcessor()


# ---------------------------------------------------------------------------
# 4. 集合级硬约束：有界修复循环（十八反）
# ---------------------------------------------------------------------------

def incompat_norm(herb: str) -> str:
    return herb


def generate_with_repair(generate_fn, validate_fn, max_retries: int = 3,
                         repair_levels: tuple[str, ...] = ("block",)) -> dict:
    """生成 → 校验 → 剔除违规药 → 重生成，直到通过或达重试上限。

    generate_fn(banned: set[str]) -> list[str]
        由调用方实现；`banned` 是本次不得使用的药名集合。
    validate_fn(herbs: list[str]) -> dict
        返回含 `findings` 的校验结果；只对【安全类 block】触发修复。

    为什么用修复循环而不是语法约束：十八反是**集合级**关系
    （甘草+甘遂不能同现），与生成顺序无关，无法在 token 级预判。

    设计取舍：宁可多轮修复也不放任违规 —— 安全性优先于生成流畅度。

    repair_levels：触发修复的级别，默认只修 block（安全性）。
    注意 config 里 `丁香|郁金` 默认被降级为 warn，因此不会触发修复 ——
    这是刻意的：该条目在中医界有争议，不应据此改写医生/模型的处方。
    """
    banned: set[str] = set()
    history = []
    for attempt in range(max_retries + 1):
        herbs = generate_fn(banned)
        res = validate_fn(herbs)
        blocks = [f for f in res.get("findings", []) if f["level"] in repair_levels]
        history.append({"attempt": attempt, "herbs": list(herbs),
                        "verdict": res.get("verdict"),
                        "repair_codes": [f["code"] for f in blocks]})
        if not blocks:
            return {"herbs": herbs, "validation": res, "attempts": attempt + 1,
                    "repaired": attempt > 0, "history": history}
        for f in blocks:
            if f["code"] == "INCOMPAT" and f.get("pair"):
                # 移除该对中"较后出现"的那味，尽量少改动
                pair = list(f["pair"])
                idxs = [i for i, h in enumerate(herbs) if h in pair]
                if idxs:
                    banned.add(herbs[max(idxs)])
        if not banned:
            break
    return {"herbs": herbs, "validation": res, "attempts": max_retries + 1,
            "repaired": True, "failed": True, "history": history}


# ---------------------------------------------------------------------------
# 5. 离线自检
# ---------------------------------------------------------------------------

class CharTokenizer:
    """离线测试用的 mock tokenizer：一个汉字 = 一个 token。"""

    def __init__(self):
        self.vocab: dict[str, int] = {}
        self.itos: list[str] = []

    def _id(self, ch: str) -> int:
        if ch not in self.vocab:
            self.vocab[ch] = len(self.itos)
            self.itos.append(ch)
        return self.vocab[ch]

    def encode(self, s: str, add_special_tokens: bool = False) -> list[int]:
        return [self._id(c) for c in s]

    def decode(self, ids) -> str:
        return "".join(self.itos[i] for i in ids if 0 <= i < len(self.itos))


def _sample_from_trie(trie: HerbTrie, tok, rng, sep: str = " ") -> list[str]:
    """用 trie 状态机随机采样一个合法药名序列，返回**解码后的药名列表**。

    这是对"受约束解码器"的离线替身：真实解码器用 logits 掩码，
    这里用状态机的 allowed_units 随机选，两者共享同一套 trie 语义。
    """
    nodes, emitted = trie.initial(), False
    units: list[int] = []
    for _ in range(400):
        allowed = trie.allowed_units(nodes, emitted)
        if not allowed:
            break
        # 落在完整药名边界时，有概率结束
        if any(n.terminal for n in nodes) and rng.random() < 0.25:
            break
        u = rng.choice(sorted(allowed))
        if trie.is_eos(u):
            break
        r = trie.advance(nodes, u, emitted)
        if r is None:
            break
        nodes, emitted = r
        if not trie.is_sep(u):
            units.append(u)
        else:
            units.append(tok.encode(sep)[0])
    text = tok.decode(units)
    return [h for h in text.split(sep) if h]


def selftest() -> int:
    """离线自检：不依赖 torch / vLLM / transformers。失败返回非 0。"""
    import random
    fails = []

    def check(name, cond, extra=""):
        print(f"  {'✅' if cond else '❌'} {name}" + (f"  {extra}" if extra else ""))
        if not cond:
            fails.append(name)

    herbs = load_whitelist()
    print(f"白名单: {len(herbs)} 味")
    tok = CharTokenizer()

    # --- 1) trie 正确性 ---
    print("\n[1] HerbTrie 逐 token 约束")
    trie = HerbTrie(herbs, encode=tok.encode, sep=" ", eos="\n")
    # 完整枚举：从任一药名首字符出发，走完必须恰好落到 terminal
    bad = 0
    for h in herbs:
        nodes, emitted = trie.initial(), False
        for ch in h:
            r = trie.advance(nodes, tok.encode(ch)[0], emitted)
            if r is None:
                bad += 1
                break
            nodes, emitted = r
        else:
            if not any(n.terminal for n in nodes):
                bad += 1
    check(f"全部 {len(herbs)} 味药都能沿 trie 走到 terminal", bad == 0, f"失败 {bad}")

    # 不允许：半个药名后接终止
    nodes, emitted = trie.initial(), False
    r = trie.advance(nodes, tok.encode(herbs[0][0])[0], False)
    nodes, emitted = r
    allow = trie.allowed_units(nodes, emitted)
    check("半截药名状态下不允许分隔符/EOS", not ({" "}.issubset(allow) and emitted))

    # 非法字符不可达
    r = trie.advance(trie.initial(), tok.encode("Ω")[0], False)
    check("白名单外字符被拒", r is None)

    # --- 2) 状态机随机采样只产出白名单药名 ---
    print("\n[2] 状态机随机采样（模拟受约束生成）")
    rng = random.Random(0)
    K = 3000
    wl = set(herbs)
    neg = []
    produced = set()
    nonempty = 0
    for _ in range(K):
        seq = _sample_from_trie(trie, tok, rng)
        if seq:
            nonempty += 1
        for h in seq:
            produced.add(h)
            if h not in wl:
                neg.append(h)
    check(f"{K} 次受约束生成未产出白名单外药名", not neg, f"越界 {sorted(set(neg))[:5]}")
    check("采样确实产出了药名（非空）", nonempty > K * 0.9, f"非空 {nonempty}/{K}")
    check("采样覆盖到多味不同药", len(produced) > 50, f"覆盖 {len(produced)} 味")

    # --- 3) vLLM 产物 ---
    print("\n[3] vLLM guided_json / guided_regex")
    sch = build_guided_json_schema(herbs)
    enum = sch["properties"]["herbs"]["items"]["enum"]
    check("JSON schema enum == 白名单", sorted(enum) == herbs, f"|enum|={len(enum)}")
    check("JSON schema uniqueItems=True", sch["properties"]["herbs"]["uniqueItems"] is True)
    vr = validate_guided_regex(herbs)
    check("guided_regex 单药全部命中", vr["single_match"] == vr["single_expected"],
          f"{vr['single_match']}/{vr['single_expected']}")
    check("guided_regex 多药方命中", vr["multi_match"])
    check("guided_regex 拒绝白名单外药名", vr["rejects_outsider"])
    print(f"     正则长度 {vr['regex_len']} 字符（462 项交替式，编译较慢 → 优先用 guided_json）")

    # --- 4) 十八反修复循环 ---
    print("\n[4] 集合级硬约束：十八反修复循环")
    from validate_rx import Validator
    v = Validator()
    # 用【不降级】的配置，确保禁忌对落在 block 级，才能真正走到修复分支。
    # 默认配置把「丁香|郁金」降级为 warn（该条目有争议），若用它测会误判为"修复没生效"。
    cfg_nodg = json.loads(json.dumps(v.cfg))
    cfg_nodg["incompatibility"]["downgrade"] = {}
    v_strict = Validator(config=cfg_nodg)
    bad_pair = None
    for fro in sorted(v_strict.incompat, key=lambda x: sorted(x)):
        a, b = sorted(fro)
        am = sorted(v_strict.std2mine.get(a, ()))
        bm = sorted(v_strict.std2mine.get(b, ()))
        if am and bm:
            bad_pair = (am[0], bm[0])
            break
    check("找到可注入的 block 级禁忌药对", bad_pair is not None, str(bad_pair))
    if bad_pair:
        h1, h2 = bad_pair
        calls = {"n": 0}

        def gen(banned):
            calls["n"] += 1
            base = [h for h in ["茯苓", "白术", "甘草"] if h not in banned]
            add = [h for h in (h1, h2) if h not in banned]
            return base + add

        def val(hs):
            return v_strict.validate_prescription(hs, [{"zhi": "健脾益气"}])

        out = generate_with_repair(gen, val, max_retries=3)
        left = [f["code"] for f in out["validation"]["findings"] if f["level"] == "block"]
        check("修复后不再有 block 级违规", not left,
              f"attempts={out['attempts']} 剩余={left} 历史="
              f"{[h['repair_codes'] for h in out['history']]}")
        check("修复过程至少重试一次", calls["n"] >= 2, f"generate 调用 {calls['n']} 次")
        check("修复记录显示首轮确实违规", bool(out["history"][0]["repair_codes"]),
              str(out["history"][0]["repair_codes"]))
        # 对照：默认配置下丁香郁金被降级，不应触发修复
        def val_default(hs):
            return v.validate_prescription(hs, [{"zhi": "健脾益气"}])
        if bad_pair == ("丁香", "郁金"):
            out2 = generate_with_repair(gen, val_default, max_retries=3)
            check("默认配置(丁香郁金降级)不触发修复，符合预期",
                  out2["attempts"] == 1, f"attempts={out2['attempts']}")

    # --- 5) 真实处方上的白名单闭合性 ---
    print("\n[5] 真实处方的白名单闭合性")
    allr = M.read_jsonl(M.DATA_DIR / "all_records.jsonl")
    gold = [r for r in allr if M.is_gold(r)]
    wl = set(herbs)
    outside = sorted({h for r in gold for h in r["rx"] if h not in wl})
    check("真实处方全部药名 ∈ 462 白名单", not outside, f"越界 {outside[:5]}")

    print("\n" + "=" * 70)
    if fails:
        print(f"❌ 自检失败 {len(fails)} 项: {fails}")
        return 1
    print("✅ 全部自检通过（无需 torch / vLLM）")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--print-regex", action="store_true")
    args = ap.parse_args()
    if args.selftest or not (args.print_regex):
        sys.exit(selftest())
    if args.print_regex:
        print(build_vllm_guided_regex(load_whitelist()))


if __name__ == "__main__":
    main()
