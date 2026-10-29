# probe_four_cell_audit.py
# 只读: Table 1 四格分层人工审计素材
# 运行: uv run --with datasets python code/probe_four_cell_audit.py

import json, os, re, sys
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
    return {
        "n_obs": n,
        "exact_dup_pairs": exact_dup_pairs,
        "max_fail_rep": max_fail_rep,
        "max_fail_label": max_fail_label,
    }

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

def main():
    wins = load_jsonl(os.path.join(DATA, "windows_v2.jsonl"))
    rmt = load_jsonl(os.path.join(DATA, "rmt-samples.jsonl"))
    rmt_map = {(r["iid"], r["end"]): r for r in rmt}

    ds = load_from_disk(os.path.join(DATA, "subset2000"))

    cells = {
        (0, 0): [], (0, 1): [], (1, 0): [], (1, 1): [],
    }
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
            "iid": w["iid"], "end": w["end"], "row_idx": w["row_idx"],
            "n_ai": w.get("n_ai"),
            "exact_dup_pairs": f["exact_dup_pairs"],
            "max_fail_rep": f["max_fail_rep"],
            "max_fail_label": f["max_fail_label"],
            "_obs": obs,
        })

    print("[cell sizes]")
    for k in sorted(cells.keys()):
        iids = set(r["iid"] for r in cells[k])
        print("  old=%d strict=%d: windows=%4d  iids=%3d"
              % (k[0], k[1], len(cells[k]), len(iids)))

    N_PER_CELL = 12
    cell_names = {
        (0, 0): "old0_strict0  (both say PROG)",
        (0, 1): "old0_strict1  (proxy=PROG, RFH=loop)",
        (1, 0): "old1_strict0  (proxy=LOOP, RFH=not-loop)  *** largest cell ***",
        (1, 1): "old1_strict1  (both say LOOP)",
    }

    for key in [(0, 0), (0, 1), (1, 0), (1, 1)]:
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
        picks = best[:N_PER_CELL]

        print("\n\n" + "#" * 100)
        print("# CELL %s" % cell_names[key])
        print("# n_windows=%d  n_iids=%d  picked=%d (each from distinct iid, strongest window)"
              % (len(pool), len(by_iid), len(picks)))
        print("#" * 100)

        for r in picks:
            print("\n" + "=" * 90)
            print("iid=%s  end=%d  n_ai=%s" % (r["iid"], r["end"], r["n_ai"]))
            print("  exact_dup=%d  max_fail=%d(%s)"
                  % (r["exact_dup_pairs"], r["max_fail_rep"], r["max_fail_label"]))
            print("-" * 90)
            for k, o in enumerate(r["_obs"]):
                flat = o.replace("\n", " \\n ").replace("\r", " ")
                print("  obs[%02d]: %s" % (k, flat[:88]))

    print("\n\n" + "#" * 100)
    print("# SUMMARY TABLE  (fill CONFIRM column: YES / NO / MAYBE)")
    print("#" * 100)
    print("\n  cell     iid                                    end   fail  exact  CONFIRM")
    print("  " + "-" * 95)
    for key in [(0, 0), (0, 1), (1, 0), (1, 1)]:
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
        for r in best[:N_PER_CELL]:
            tag = "%d%d" % key
            print("  %-7s  %-38s %4d  %4d  %5d  ____"
                  % (tag, r["iid"], r["end"], r["max_fail_rep"], r["exact_dup_pairs"]))
    print("\n  cell codes: 00=old0_strict0  01=old0_strict1  10=old1_strict0  11=old1_strict1")

if __name__ == "__main__":
    main()