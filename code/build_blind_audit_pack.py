# build_blind_audit_pack.py
# 只读: 生成盲标材料包（不含第一标注者判定），供第二标注者独立判断
# 输出:
#   data/blind_audit_pack.txt   - 给第二标注者的材料
#   data/blind_audit_key.json   - case_id -> (cell, iid, end) 映射，第一标注者保留
# 运行: uv run --with datasets python code/build_blind_audit_pack.py

import json, os, random, re, sys
from collections import Counter, defaultdict
from datasets import load_from_disk

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

DATA = os.environ.get("TRAJGAP_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))
ROLE_ASSIST = {"assistant", "ai", "agent"}

FAIL_PATTERNS = [
    ("search_no_match", [r"No matches found for"]),
    ("usage_error", [r"^Usage: ", r"Error: File name", r"Error: "]),
    ("traceback", [r"Traceback \(most recent call last\)"]),
    ("timeout", [r"EXECUTION TIMED OUT"]),
    ("syntax_error", [r"syntax error", r"invalid syntax", r"IndentationError", r"unmatched"]),
    ("cmd_not_found", [r"command not found"]),
    ("no_such_file", [r"No such file or directory", r"cannot access", r"not found"]),
]

def load_jsonl(p):
    with open(p, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def obs_features(obs_list):
    obs_list = [o for o in obs_list if o and o.strip()]
    n = len(obs_list)
    if n < 2:
        return None
    exact_dup_pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            if obs_list[i] == obs_list[j]:
                exact_dup_pairs += 1
    fail_counts = Counter()
    for o in obs_list:
        hit = None
        for label, pats in FAIL_PATTERNS:
            if any(re.search(p, o) for p in pats):
                hit = label; break
        if hit:
            fail_counts[hit] += 1
    max_fail_label, max_fail_rep = (None, 0)
    if fail_counts:
        max_fail_label, max_fail_rep = max(fail_counts.items(), key=lambda x: x[1])
    return {"n_obs": n, "exact_dup_pairs": exact_dup_pairs,
            "max_fail_rep": max_fail_rep, "max_fail_label": max_fail_label}

def strict_label(f):
    return int(f["exact_dup_pairs"] >= 20 or f["max_fail_rep"] >= 10)

def extract_obs(w, ds):
    ridx = w["row_idx"]
    try:
        row = ds[ridx]
    except Exception:
        return None
    if row.get("instance_id") != w["iid"]:
        return None
    steps = row["trajectory"]
    ai_pos = [i for i, s in enumerate(steps) if s.get("role") in ROLE_ASSIST]
    raw = w.get("win_steps", [])
    seen = set(); win_steps = []
    for s in raw:
        if s not in seen:
            seen.add(s); win_steps.append(s)
    obs_list = []
    for ai_idx in win_steps:
        if ai_idx >= len(ai_pos):
            obs_list.append(""); continue
        pos = ai_pos[ai_idx]
        o = ""
        for j in range(pos + 1, min(pos + 4, len(steps))):
            if steps[j].get("role") == "user":
                o = steps[j].get("text", ""); break
        obs_list.append(o)
    return obs_list

PROTOCOL = """\
# 判据定义

你要判断的是：一个 LLM agent 轨迹窗口（连续 15 步的观测序列）是否处于
「参数变体型语义死循环」(parameter-varying semantic loop)。

定义：动作参数每次都在变，但语义上卡在同一个地方，持续消耗 token 却无进展。

具体判据（三个条件全部满足才判 YES）：
1. 从窗口某一步开始，agent 的语义子目标不再变化（例如：一直在找同一个函数、
   一直在试同一个 API 的参数、一直在同一个目录下反复 ls 不同文件名）；
2. 该区间内动作参数或观测文本持续变化——不是精确重复（精确重复是低一层的
   死循环，不在本次判据的 YES 范围，但也不是 NO）；
3. 观测显示持续无进展：同一失败模式重复出现（例如连续 No matches /
   command not found / syntax error），或反复在同类对象上失败而没有获得
   新信息。

判定为 NO 的情形：
- 窗口内有信息增益：agent 每次失败后打开了新的文件、获得了新的信息、
  缩小了搜索范围，或状态在推进（文件行数在增长且内容在变）。
- agent 在正常探索，每次观测提供了新的语义内容。

判定为 MAYBE 的情形：
- 前段在推进，后段卡住（需窗口外上下文才能定）；
- 或观测模板不变但底层对象在变化，无法从窗口内判断是否获得新信息。

输出格式要求（严格）：
对每个 case 输出一行，格式为 `Cxx: YES` 或 `Cxx: MAYBE` 或 `Cxx: NO`。
不要输出解释，不要输出其他内容。共 48 行。

# 材料

以下是 48 个 case，编号已随机打乱。每个 case 只有 iid、窗口终止步（end）、
和 15 步观测序列（obs[00] 到 obs[14]）。观测文本被截断到约 200 字符，
行间换行被替换为空格。

"""

def main():
    random.seed(2026)  # 与四格审计一致
    wins = load_jsonl(os.path.join(DATA, "windows_v2.jsonl"))
    rmt = load_jsonl(os.path.join(DATA, "rmt-samples.jsonl"))
    rmt_map = {(r["iid"], r["end"]): r for r in rmt}
    ds = load_from_disk(os.path.join(DATA, "subset2000"))

    cells = {(0,0): [], (0,1): [], (1,0): [], (1,1): []}
    for w in wins:
        key = (w["iid"], w["end"])
        if key not in rmt_map:
            continue
        obs = extract_obs(w, ds)
        if obs is None:
            continue
        f = obs_features(obs)
        if f is None:
            continue
        y_old = 1 - int(w["future_has_new_res"])
        y_strict = strict_label(f)
        cells[(y_old, y_strict)].append({
            "iid": w["iid"], "end": w["end"],
            "exact_dup_pairs": f["exact_dup_pairs"],
            "max_fail_rep": f["max_fail_rep"],
            "_obs": obs,
        })

    all_cases = []
    for key in [(0,0), (0,1), (1,0), (1,1)]:
        pool = cells[key]
        if not pool:
            continue
        by_iid = defaultdict(list)
        for r in pool:
            by_iid[r["iid"]].append(r)
        best = []
        for iid, rows in by_iid.items():
            rows.sort(key=lambda r: (-r["max_fail_rep"], -r["exact_dup_pairs"]))
            best.append(rows[0])
        best.sort(key=lambda r: (-r["max_fail_rep"], -r["exact_dup_pairs"]))
        for r in best[:12]:
            all_cases.append({"cell": key, "iid": r["iid"], "end": r["end"],
                              "_obs": r["_obs"]})

    print("[collected] %d cases" % len(all_cases))

    # 随机打乱 case 顺序 + 分配 case_id，避免第二标注者按 cell 分组
    random.seed(999)
    random.shuffle(all_cases)
    case_ids = ["C%02d" % (i + 1) for i in range(len(all_cases))]

    pack_path = os.path.join(DATA, "blind_audit_pack.txt")
    key_path = os.path.join(DATA, "blind_audit_key.json")

    with open(pack_path, "w", encoding="utf-8") as f:
        f.write(PROTOCOL)
        for cid, case in zip(case_ids, all_cases):
            f.write("\n### %s\n" % cid)
            f.write("iid=%s  end=%d  n_obs=%d\n" % (case["iid"], case["end"], len(case["_obs"])))
            for k, o in enumerate(case["_obs"]):
                flat = o.replace("\n", " ").replace("\r", " ").strip()
                f.write("  [%02d] %s\n" % (k, flat[:220]))
        f.write("\n# 结束。请输出 48 行 `Cxx: YES|MAYBE|NO`。\n")

    key = {cid: {"cell": "%d%d" % case["cell"], "iid": case["iid"], "end": case["end"]}
           for cid, case in zip(case_ids, all_cases)}
    with open(key_path, "w", encoding="utf-8") as f:
        json.dump(key, f, ensure_ascii=False, indent=2)

    print("[saved] %s" % pack_path)
    print("[saved] %s" % key_path)
    print("[next] 打开 pack 文件，整段复制，贴给第二标注者 (不同家族的 LLM)")
    print("       让它按判据输出 48 行 `Cxx: YES|MAYBE|NO`")
    print("       把它的 48 行输出贴回来，我算 Cohen's kappa")

if __name__ == "__main__":
    main()