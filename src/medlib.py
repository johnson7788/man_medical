"""
medlib.py — 中医男科病历数据集的共享工具库

统一收口：读取、清洗、归一化、解析、切分、格式化。
prepare_data.py / baseline.py / evaluate.py 全部从这里导入，保证训练与评测口径一致。

关键口径（与《模型训练计划.md》§1.5 / §3.1 一致）：
  1. 剂量字段整体丢弃 —— 原始数据 99.7% 剂量为 "1.000"，无信息量。
  2. 药名归一化 —— 去尾部标点/空格，剥离残留剂量后缀（如 "酒黄精10g"）。
  3. 诊断段级去重 —— 21.1% 的行内存在重复诊断段。
  4. 只保留 3 级诊断段（病名/证型/治法）。
  5. 切分按「病历号」分组 —— 79.9% 样本来自多次就诊患者，按行切分必然泄漏。
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_XLSX = ROOT / "男科(2).xlsx"
DATA_DIR = ROOT / "data"
REPORT_DIR = ROOT / "reports"

COLUMNS = [
    "pid", "pid_masked", "sex", "age_raw", "cc", "hpi",
    "wm_dx", "tcm_dx", "rx", "date_raw",
]

SEED = 42
SPLIT_RATIO = (0.8, 0.1, 0.1)

# ---------------------------------------------------------------------------
# Gate 通过线 —— 唯一定义处，evaluate.py 与 baseline.py 共同引用
# 所有阈值均按 Stage 0 实测基线校准（见 reports/baseline_report.md）：
#                     常量下界 | 最佳可实现 | Oracle上界 | 端到端
#   处方 Jaccard       0.159  |   0.183   |   0.208   |  0.124
#   关键药 Recall      0.352  |   0.378   |   0.403   |   -
#   证型 micro-F1      0.321  |   0.368   |    -      |  0.368
# 关键设计：两个「必达」线都设在 Oracle 上界【之上】。
#   达不到就说明模型只是学会了「证型 -> 常用方」的查表，没有真正读病史。
# ---------------------------------------------------------------------------
GATES: dict[str, dict] = {
    "处方_Jaccard":  {"名称": "G2 处方 Jaccard",         "必达": 0.22,  "目标": 0.30,  "方向": ">="},
    "关键药_Recall": {"名称": "G2 关键药 Recall(top100)", "必达": 0.45,  "目标": 0.55,  "方向": ">="},
    "证型_F1":       {"名称": "G1 证型 micro-F1",         "必达": 0.55,  "目标": 0.65,  "方向": ">="},
    "非法药名率":     {"名称": "G3 非法药名率",            "必达": 0.005, "目标": 0.0,   "方向": "<="},
    "重复药率":       {"名称": "G3 重复药率",              "必达": 0.005, "目标": 0.0,   "方向": "<="},
    "空方率":         {"名称": "G3 空方率",                "必达": 0.005, "目标": 0.0,   "方向": "<="},
}
# 各指标的实测锚点，供报告与文档引用（常量下界 / 最佳可实现 / Oracle上界）
BASELINE_ANCHORS: dict[str, tuple[float, float, float]] = {
    "处方_Jaccard":  (0.159, 0.183, 0.208),
    "关键药_Recall": (0.352, 0.378, 0.403),
    "证型_F1":       (0.321, 0.368, float("nan")),
}
# 时间外推相对下降上限（G5）
G5_MAX_DROP = 0.20

# 药味数区间外【不排除】，只打标：实测 1,793 条短方是真实小方/药对风格
# （原始字符串完整、878 名患者全部处方均 <5 味、逐年均匀 7-8%），不是录入截断。
# 排除会丢 6.5% 真实标签并造成处方长度分布偏移；评测时按子集分别报告。
RX_LEN_MIN, RX_LEN_MAX = 5, 40
AGE_MIN, AGE_MAX = 18, 105

# ---------------------------------------------------------------------------
# 药名归一化
# ---------------------------------------------------------------------------

_PUNCT = re.compile(r"[\s.．。、,，;；:：\-—_/\\]+")
# 剂量后缀必须能【连续】匹配多段：真实数据里存在 `酒黄精10g1.000`
# —— 药名 + 真实克数 + 平台标准化的 1.000。只剥一层会残留 "酒黄精10g"，
# 导致 51 味药被劈成两个名字、2,900 条记录（10.8%）药名失真。
# 实测形如 <药名><5g|10g|6g...><1.000>；中文药名不会以 ASCII 数字结尾，故可放心全剥。
_DOSE_SUFFIX = re.compile(r"(?:\d+(?:\.\d+)?\s*(?:g|G|克|mg|ml|ML)?)+$")
_LEADING_NOISE = re.compile(r"^[\s.．。、,，;；]+")
# 归一化后仍含 ASCII 数字/字母 = 清洗失败，直接报错而不是静默放过
_HAS_ASCII_ALNUM = re.compile(r"[0-9A-Za-z]")


def assert_clean_herb_name(h: str) -> None:
    if _HAS_ASCII_ALNUM.search(h):
        raise ValueError(f"药名归一化失败，仍含数字/字母：{h!r}（剂量后缀剥离不完整）")

# 仅合并「书写变体」，绝不合并不同炮制品。
# 生地黄/熟地黄、甘草片/炙甘草、白芍/赤芍 在中医里是不同药，必须保持区分。
MANUAL_HERB_MAP: dict[str, str] = {
    "大枣": "大枣",
    "地黄": "地黄",
    "甜叶菊叶": "甜叶菊叶",
}

# 非治疗性矫味/赋形药：保留在处方里，但不计入「关键药」核心指标
NON_THERAPEUTIC = {"甜叶菊叶", "甜叶菊", "蔗糖", "蜂蜜", "饴糖", "木糖醇"}


def normalize_herb(raw: str) -> str:
    """把原始药名片段（可能带剂量、标点、空格）归一到标准药名。"""
    h = str(raw).strip()
    h = _DOSE_SUFFIX.sub("", h)   # 北柴胡1.000 -> 北柴胡 ; 酒黄精10g1.000 -> 酒黄精
    h = _PUNCT.sub("", h)         # 白芍. + 空格 -> 白芍
    h = _LEADING_NOISE.sub("", h)
    return MANUAL_HERB_MAP.get(h, h)


def parse_rx(raw) -> list[str]:
    """解析中药处方字符串为有序、去重的药名列表。

    输入: "陈皮1.000,丹参1.000,当归1.000"
    输出: ["陈皮", "丹参", "当归"]
    剂量一律丢弃（原始数据剂量无信息量）。
    """
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return []
    out: list[str] = []
    for part in str(raw).split(","):
        h = normalize_herb(part)
        if h:
            out.append(h)
    return list(dict.fromkeys(out))          # 去重且保序


# ---------------------------------------------------------------------------
# 中医诊断解析
# ---------------------------------------------------------------------------

def parse_dx_segments(raw) -> list[tuple[str, str, str]]:
    """解析中医诊断字符串为段级三元组列表。

    输入: "腰痛/肾气不充证/补肾扶元,精浊/肾气不充证/补肾扶元"
    输出: [("腰痛","肾气不充证","补肾扶元"), ("精浊","肾气不充证","补肾扶元")]

    只保留恰好 3 级的段；段级去重（原始数据 21.1% 行内重复）。
    """
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return []
    segs: list[tuple[str, str, str]] = []
    for seg in str(raw).split(","):
        seg = seg.strip()
        if not seg:
            continue
        parts = [p.strip() for p in seg.split("/")]
        if len(parts) != 3:
            continue                          # 丢弃 1/2 级与 4-7 级脏段
        if not all(parts):
            continue
        segs.append((parts[0], parts[1], parts[2]))
    return list(dict.fromkeys(segs))


def parse_wm_dx(raw) -> list[str]:
    """解析西医诊断（逗号/中文逗号/分号分隔）。"""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return []
    out = [x.strip() for x in re.split(r"[,，;；]", str(raw)) if x.strip()]
    return list(dict.fromkeys(out))


def parse_age(raw) -> int | None:
    """'89岁' -> 89"""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    m = re.search(r"\d+", str(raw))
    return int(m.group()) if m else None


def hash_pid(pid: str) -> str:
    """真实病历号脱敏为不可逆短哈希，供训练数据与 jsonl 落盘使用。"""
    return "P" + hashlib.sha256(f"man_medical::{pid}".encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# 读取与构建
# ---------------------------------------------------------------------------

def load_raw(path: Path | None = None) -> pd.DataFrame:
    """读取原始 Excel。

    `path` 必须显式传入：原始表含院方数据，**不随开源仓库提供**，
    因此不再有默认路径（早期版本默认 RAW_XLSX，发布后会直接 FileNotFoundError）。
    """
    p = Path(path) if path else RAW_XLSX
    if not p.exists():
        raise FileNotFoundError(
            f"原始表不存在：{p}\n"
            "  开源发布版不提供原始 Excel。请改用发布模式：\n"
            "      python3 src/prepare_data.py            # 读 data/all_records.jsonl\n"
            "  若你持有原始表，请显式指定：--from-xlsx <路径>")
    df = pd.read_excel(p, dtype=str)
    if len(df.columns) != len(COLUMNS):
        raise ValueError(f"列数不符：期望 {len(COLUMNS)}，实际 {len(df.columns)}")
    df.columns = COLUMNS
    return df


def build_records(df: pd.DataFrame) -> list[dict]:
    """清洗全表，返回 records 列表（含无处方记录，便于 T1/T5 任务复用）。

    每条 record:
      {rid, pid, sex, age, cc, hpi, wm_dx[], tcm_dx[[b,z,t]...], rx[],
       date, has_dx, has_rx, flags[]}
    """
    # 日期统一解析为补零 ISO 字符串（必须补零：'2025-6-30' < '2025-07-01' 的
    # 字典序比较会得到 False，导致单数字月份被错误分到时间测试集）
    dt = pd.to_datetime(df["date_raw"], errors="coerce")
    dates = dt.dt.strftime("%Y-%m-%d").fillna("").tolist()

    records: list[dict] = []
    for i, row in enumerate(df.itertuples(index=False)):
        flags: list[str] = []

        age = parse_age(row.age_raw)
        if age is not None and not (AGE_MIN <= age <= AGE_MAX):
            flags.append("age_outlier")
        sex = str(row.sex).strip()
        if sex != "男":
            flags.append("sex_not_male")

        dx = parse_dx_segments(row.tcm_dx)
        rx = parse_rx(row.rx)

        if rx and not (RX_LEN_MIN <= len(rx) <= RX_LEN_MAX):
            flags.append("rx_len_outlier")

        # 主诉与现病史的冗余标记（计划 §1.5 P1）
        cc = "" if row.cc is None else str(row.cc).strip()
        hpi = "" if row.hpi is None else str(row.hpi).strip()
        if cc and cc == hpi:
            flags.append("cc_eq_hpi")
        elif cc and hpi.startswith(cc):
            flags.append("hpi_has_cc_prefix")

        records.append({
            "rid": i,
            "pid": hash_pid(row.pid),
            "sex": sex,
            "age": age,
            "cc": cc,
            "hpi": hpi,
            "wm_dx": parse_wm_dx(row.wm_dx),
            "tcm_dx": [list(t) for t in dx],
            "rx": rx,
            "date": dates[i],
            "has_dx": bool(dx),
            "has_rx": bool(rx),
            "flags": flags,
        })
    return records


# 致命标记：命中则不进训练集（性别不符科室定位 / 年龄明显错误）
FATAL_FLAGS = {"sex_not_male", "age_outlier"}


def is_gold(r: dict) -> bool:
    """金标准样本：能构造出完整监督信号。

    要求：有主诉、有中医诊断、有处方，且无致命标记。
    rx_len_outlier 不排除（见 RX_LEN_MIN 注释），只作为评测子集标签保留。
    """
    if not (r["cc"] and r["has_dx"] and r["has_rx"]):
        return False
    return not FATAL_FLAGS.intersection(r["flags"])


# ---------------------------------------------------------------------------
# 切分
# ---------------------------------------------------------------------------

def split_by_patient(records: list[dict], seed: int = SEED,
                     ratios: tuple[float, float, float] = SPLIT_RATIO
                     ) -> dict[str, list[dict]]:
    """按患者 ID 分组做 train/val/test 切分。同一患者只出现在一个集合。"""
    import random

    pids = sorted({r["pid"] for r in records})
    random.Random(seed).shuffle(pids)

    n = len(pids)
    n_tr = int(n * ratios[0])
    n_va = int(n * ratios[1])
    tr_pids = set(pids[:n_tr])
    va_pids = set(pids[n_tr:n_tr + n_va])
    te_pids = set(pids[n_tr + n_va:])

    splits = {"train": [], "val": [], "test": []}
    for r in records:
        if r["pid"] in tr_pids:
            splits["train"].append(r)
        elif r["pid"] in va_pids:
            splits["val"].append(r)
        elif r["pid"] in te_pids:
            splits["test"].append(r)

    # 硬断言：患者零重叠
    sets = {k: {r["pid"] for r in v} for k, v in splits.items()}
    assert not (sets["train"] & sets["val"]), "train/val 患者重叠"
    assert not (sets["train"] & sets["test"]), "train/test 患者重叠"
    assert not (sets["val"] & sets["test"]), "val/test 患者重叠"
    return splits


def split_by_time(records: list[dict], cutoff: str = "2025-07-01"
                  ) -> tuple[list[dict], list[dict]]:
    """时间外推切分：训练集取 cutoff 之前，测试集取之后【且患者不在训练集】。

    计划 §3.2：纯时间切分天然有 18%-30% 患者重叠，必须叠加剔除训练集患者。
    依赖 record["date"] 为补零 ISO（YYYY-MM-DD）格式，否则字典序比较会出错。
    """
    bad = [r["rid"] for r in records if len(r["date"]) != 10 or r["date"][4] != "-"]
    if bad:
        raise ValueError(f"{len(bad)} 条记录的 date 不是补零 ISO 格式，例如 rid={bad[:5]}")
    tr = [r for r in records if r["date"] < cutoff]
    tr_pids = {r["pid"] for r in tr}
    te = [r for r in records if r["date"] >= cutoff and r["pid"] not in tr_pids]
    return tr, te


# ---------------------------------------------------------------------------
# Prompt / 输出格式化（训练与评测共用，保证两侧完全一致）
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "你是中医男科专家。请根据患者四诊信息进行辨证论治："
    "先给出中医诊断，每行一个诊断，格式为「病名·证型·治法」；"
    "再给出中药处方，输出药味名称，不含剂量。"
)

DX_HEADER = "【中医诊断】"
RX_HEADER = "【中药处方】"


def format_input(r: dict) -> str:
    """四诊信息 -> user prompt 正文（T3 基础版，不含既往处方）。"""
    lines = [f"【基本信息】{r['sex']}，{r['age']}岁"]
    lines.append(f"【主诉】{r['cc']}")
    if r["hpi"]:
        lines.append(f"【现病史】{r['hpi']}")
    if r["wm_dx"]:
        lines.append(f"【西医诊断】{','.join(r['wm_dx'])}")
    return "\n".join(lines)


def format_input_v2(r: dict) -> str:
    """T3+ 版本：额外给出该患者上一次就诊的处方与间隔。

    依据实测：复诊占 63.9%，其"上次处方"的预测力（照抄 Jaccard 0.539）
    远超整套中医诊断（Oracle 上界 0.203）。不提供这个字段，
    等于让医生在不知道上次开了什么、吃了多久的情况下开方 —— 现实中不会这样。
    """
    base = format_input(r)
    if not r.get("prev_rx"):
        return base + "\n【既往处方】无（初诊）"
    gap = r.get("prev_gap")
    gap_s = f"，间隔 {gap} 天" if gap is not None else ""
    return (base +
            f"\n【既往处方】第 {r.get('visit_ix', 0)} 次复诊{gap_s}，上次处方："
            + " ".join(r["prev_rx"]))


def format_target(r: dict) -> str:
    """中医诊断 + 处方 -> assistant 输出正文（评测解析的唯一依据）。"""
    dx_lines = ["·".join(t) for t in r["tcm_dx"]]
    return (
        f"{DX_HEADER}\n" + "\n".join(dx_lines) +
        f"\n{RX_HEADER}\n" + " ".join(r["rx"])
    )


def parse_assistant(text: str) -> tuple[list[tuple[str, str, str]], list[str]]:
    """从模型输出（或参考输出）文本中解析出诊断三元组与药名列表。

    容错设计：允许诊断用 · / 、- 分隔；处方用空格/逗号/顿号分隔。
    这是评测的唯一入口 —— 模型输出只要能被这里解析就算合法。
    """
    if not text:
        return [], []

    dx_part, rx_part = "", ""
    if DX_HEADER in text:
        after = text.split(DX_HEADER, 1)[1]
        if RX_HEADER in after:
            dx_part, rx_part = after.split(RX_HEADER, 1)
        else:
            dx_part = after
    elif RX_HEADER in text:
        pre, rx_part = text.split(RX_HEADER, 1)
        dx_part = pre
    else:
        dx_part = text

    dx: list[tuple[str, str, str]] = []
    for line in dx_part.splitlines():
        line = line.strip().strip("【】")
        if not line or line.startswith("中医诊断"):
            continue
        parts = [p.strip() for p in re.split(r"[·/、|]", line) if p.strip()]
        if len(parts) == 3:
            dx.append(tuple(parts))          # type: ignore[arg-type]
        elif len(parts) == 2:
            dx.append((parts[0], parts[1], ""))   # type: ignore[arg-type]
    dx = list(dict.fromkeys(dx))

    rx = [normalize_herb(x) for x in re.split(r"[\s,，、;；]+", rx_part) if x.strip()]
    rx = [h for h in rx if h]
    return dx, list(dict.fromkeys(rx))


def to_chatml(r: dict, v2: bool = False) -> dict:
    """v2=True 时使用 T3+ 输入（含既往处方）。"""
    return {"messages": [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": format_input_v2(r) if v2 else format_input(r)},
        {"role": "assistant", "content": format_target(r)},
    ]}


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------

def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def top_herbs(records: list[dict], k: int) -> list[str]:
    """取最高频 k 味药。

    ⚠️ 不能直接用 Counter.most_common：并列名次依赖 set/dict 的遍历顺序，
    而 Python 对 str 的哈希默认随机化（PYTHONHASHSEED），会导致同一份数据
    在不同进程里产出不同的 top-k、进而让评测结果不可复现（实测出现过
    显著关联数 12 vs 13 的漂移）。必须显式按 (频次降序, 药名升序) 排序。
    """
    c = Counter(h for r in records for h in r["rx"])
    return [h for h, _ in sorted(c.items(), key=lambda x: (-x[1], x[0]))[:k]]


# ---------------------------------------------------------------------------
# 纵向关联：既往处方（治疗进程）
# ---------------------------------------------------------------------------

def link_history(records: list[dict]) -> None:
    """就地给每条记录挂上「同一患者上一次就诊」的处方与间隔天数。

    为什么这是本数据集最关键的一步：
      实测 97.8% 的「相同输入却不同处方」其实都是**同一患者的复诊**
      （主诉/现病史被医生复制粘贴，文本逐字相同，但患者已治疗过若干轮）。
      同一患者相邻两次就诊的处方 Jaccard = 0.539，
      **远高于任何基于证型的检索上界 0.203** ——
      也就是说「上次开的方」是比整套中医诊断更强的预测因子。

    这不是标签泄漏：既往处方来自**更早就诊**，生产环境的 HIS 系统里本来就能拿到。
    它只是被当前「四诊 → 处方」的任务定义丢掉了。

    口径说明：prev_rx 指向**上一次开出处方的就诊**，而不是"上一次就诊"。
    临床上「上次的方」指的是最后一次真正开出的方；中间若有只开中成药/西药的
    就诊，不应把它当成"没有既往方"。这样才能正确区分初诊与复诊。

    写入字段：
      prev_rx     上一次开出处方的处方（该患者首次开方为 None）
      prev_gap    与那次开方的间隔天数（首次为 None）
      visit_ix    就诊序号（0 = 首次就诊）
      n_visits    该患者的总就诊次数
      visit_ix_rx 开方序号（0 = 首次开方）
      n_rx        该患者的总开方次数
    """
    import datetime as _dt

    by_pid: dict[str, list[dict]] = {}
    for r in records:
        by_pid.setdefault(r["pid"], []).append(r)

    for lst in by_pid.values():
        lst.sort(key=lambda x: (x["date"], x["rid"]))
        last_rx: dict | None = None
        n_rx = 0
        for i, r in enumerate(lst):
            r["visit_ix"] = i
            r["n_visits"] = len(lst)
            if r["has_rx"]:
                r["visit_ix_rx"] = n_rx
                n_rx += 1
            else:
                r["visit_ix_rx"] = None
            if last_rx is None:
                r["prev_rx"] = None
                r["prev_gap"] = None
            else:
                r["prev_rx"] = list(last_rx["rx"])
                try:
                    d0 = _dt.date.fromisoformat(last_rx["date"])
                    d1 = _dt.date.fromisoformat(r["date"])
                    r["prev_gap"] = (d1 - d0).days
                except ValueError:
                    r["prev_gap"] = None
            if r["has_rx"]:
                last_rx = r
        for r in lst:
            r["n_rx"] = n_rx


def is_followup(r: dict) -> bool:
    """复诊（有既往处方）—— 与初诊是两个难度完全不同的子任务。"""
    return bool(r.get("prev_rx"))
