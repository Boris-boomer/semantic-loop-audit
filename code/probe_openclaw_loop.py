# probe_openclaw_loop.py
# 只读: OpenClawBench 里 loop 形态的存在性 + 与 anomaly 标签的关系
# 运行: uv run --with pandas python code/probe_openclaw_loop.py

import json, os, re, sys
from collections import Counter

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = os.environ.get("OPENCLAW_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "openclaw_bench-4378"))
PKT = os.path.join(ROOT, "dataset", "gold_iaa_300", "trajectory_packets")

FAIL_PATTERNS = [
    ("error_status", [r'"status":\s*"error"', r"status:\s*error"]),
    ("cmd_not_found", [r"command not found"]),
    ("no_such_file", [r"No such file or directory", r"cannot access", r"not found"]),
    ("traceback", [r"Traceback \(most recent call last\)"]),
    ("timeout", [r"timeout", r"timed out", r"EXECUTION TIMED OUT"]),
    ("gateway_closed", [r"gateway closed", r"abnormal closure"]),
    ("http_error", [r"HTTP/[0-9.]+ (4\d\d|5\d\d)", r"\b403\b", r"\b404\b", r"\b500\b"]),
    ("exit_code_nonzero", [r"exited with code [1-9]"]),
]

def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def tool_name(action_text):
    if not action_text:
        return None
    m = re.match(r"\[toolCall\s+([A-Za-z0-9_\-]+)\]", action_text.strip())
    if m:
        return m.group(1)
    return None

def tool_args(action_text):
    if not action_text:
        return None
    m = re.match(r"\[toolCall\s+[A-Za-z0-9_\-]+\]\s*(\{.*\})", action_text.strip(), re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            return m.group(1)
    return None

def fail_hits(text):
    hits = []
    for label, pats in FAIL_PATTERNS:
        if any(re.search(p, text) for p in pats):
            hits.append(label)
    return hits

def traj_features(traj):
    steps = traj.get("steps", [])
    n = len(steps)
    if n < 2:
        return None
    actions = [tool_name(s.get("action_text") or "") for s in steps]
    obs = [(s.get("reflection_text") or "") for s in steps]
    exact_dup_pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            if obs[i] and obs[j] and obs[i] == obs[j]:
                exact_dup_pairs += 1
    fail_counts = Counter()
    for o in obs:
        for h in fail_hits(o):
            fail_counts[h] += 1
    max_fail_label, max_fail_rep = (None, 0)
    if fail_counts:
        max_fail_label, max_fail_rep = max(fail_counts.items(), key=lambda x: x[1])
    at = [a for a in actions if a]
    action_rep = 0
    action_top = None
    if at:
        c = Counter(at)
        action_top, action_rep = c.most_common(1)[0]
    param_variants = 0
    if at:
        by_tool = {}
        for s in steps:
            t = tool_name(s.get("action_text") or "")
            if not t:
                continue
            a = tool_args(s.get("action_text") or "")
            by_tool.setdefault(t, []).append(json.dumps(a, sort_keys=True) if isinstance(a, (dict, list)) else str(a))
        for t, args in by_tool.items():
            if len(args) >= 2:
                uniq = len(set(args))
                if uniq >= 2:
                    param_variants += 1
    return {
        "n_steps": n,
        "exact_dup_pairs": exact_dup_pairs,
        "max_fail_rep": max_fail_rep,
        "max_fail_label": max_fail_label,
        "max_fail_ratio": max_fail_rep / n if n else 0.0,
        "action_top": action_top,
        "action_rep": action_rep,
        "action_rep_ratio": action_rep / n if n else 0.0,
        "param_variants": param_variants,
        "actions": actions,
        "obs": obs,
    }

def main():
    files = sorted(f for f in os.listdir(PKT) if f.endswith(".jsonl"))
    print("[packets] %d files" % len(files))
    all_rows = []
    for fn in files:
        rows = load_jsonl(os.path.join(PKT, fn))
        print("  %-40s %d" % (fn, len(rows)))
        all_rows.extend(rows)
    print("[total] %d trajectories" % len(all_rows))

    feats = []
    for r in all_rows:
        f = traj_features(r)
        if f is None:
            continue
        f["trajectory_id"] = r.get("trajectory_id")
        f["source_model"] = r.get("source_model")
        f["is_anom"] = bool(r.get("silver_is_anom"))
        f["l2"] = r.get("silver_l2")
        feats.append(f)
    print("[usable] %d" % len(feats))

    print("\n" + "=" * 88)
    print("[1] step count distribution by anomaly label")
    print("=" * 88)
    for label, pool in [("normal", [f for f in feats if not f["is_anom"]]),
                        ("anom",   [f for f in feats if f["is_anom"]])]:
        ns = sorted(f["n_steps"] for f in pool)
        if not ns:
            print("  %-8s empty" % label); continue
        def q(p):
            return ns[min(len(ns)-1, int(p*len(ns)))]
        print("  %-8s n=%3d  p50=%d  p90=%d  max=%d"
              % (label, len(ns), q(0.5), q(0.9), ns[-1]))

    print("\n" + "=" * 88)
    print("[2] loop-morphology feature distribution by anomaly label")
    print("=" * 88)
    for label, pool in [("normal", [f for f in feats if not f["is_anom"]]),
                        ("anom",   [f for f in feats if f["is_anom"]])]:
        if not pool: continue
        print("\n  [%s] n=%d" % (label, len(pool)))
        for k in ["exact_dup_pairs", "max_fail_rep", "action_rep_ratio", "param_variants"]:
            vals = sorted(f[k] for f in pool)
            def q(p): return vals[min(len(vals)-1, int(p*len(vals)))]
            print("    %-18s p25=%.2f p50=%.2f p75=%.2f p90=%.2f max=%.2f"
                  % (k, q(0.25), q(0.5), q(0.75), q(0.9), vals[-1]))

    print("\n" + "=" * 88)
    print("[3] loop indicator hit rates by anomaly label")
    print("=" * 88)
    def loop_variants(f):
        return {
            "any_fail3": f["max_fail_rep"] >= 3,
            "any_fail_ratio50": f["max_fail_rep"] >= 2 and f["max_fail_ratio"] >= 0.5,
            "exactdup3": f["exact_dup_pairs"] >= 3,
            "action_rep_ratio50": f["action_rep_ratio"] >= 0.5,
            "param_variant_loop": f["param_variants"] >= 1 and f["max_fail_rep"] >= 2,
            "any_of_above": (f["max_fail_rep"] >= 3 or f["exact_dup_pairs"] >= 3
                             or (f["action_rep_ratio"] >= 0.5 and f["max_fail_rep"] >= 1)),
        }
    normal = [f for f in feats if not f["is_anom"]]
    anom = [f for f in feats if f["is_anom"]]
    print("  %-22s  normal   anom" % "judge")
    print("  " + "-" * 40)
    for j in ["any_fail3", "any_fail_ratio50", "exactdup3",
              "action_rep_ratio50", "param_variant_loop", "any_of_above"]:
        rn = sum(1 for f in normal if loop_variants(f)[j])
        ra = sum(1 for f in anom if loop_variants(f)[j])
        print("  %-22s  %3d/%3d  %3d/%3d"
              % (j, rn, len(normal), ra, len(anom)))

    print("\n" + "=" * 88)
    print("[4] strongest loop hits among normal")
    print("=" * 88)
    normal_loop = [f for f in normal if loop_variants(f)["any_of_above"]]
    normal_loop.sort(key=lambda f: -(f["max_fail_rep"] * 100 + f["exact_dup_pairs"]))
    for f in normal_loop[:8]:
        print("\n--- %s  model=%s  n_steps=%d  fail=%d(%s)  exactdup=%d  actionrep=%s*%d"
              % (f["trajectory_id"], f["source_model"], f["n_steps"],
                 f["max_fail_rep"], f["max_fail_label"], f["exact_dup_pairs"],
                 f["action_top"], f["action_rep"]))
        for k in range(min(5, f["n_steps"])):
            a = (f["actions"][k] or "")[:40]
            o = (f["obs"][k] or "").replace("\n", " ")[:100]
            print("    step%d action=%-12s obs=%s" % (k+1, a, o))

    print("\n" + "=" * 88)
    print("[5] strongest loop hits among anomalous")
    print("=" * 88)
    anom_loop = [f for f in anom if loop_variants(f)["any_of_above"]]
    anom_loop.sort(key=lambda f: -(f["max_fail_rep"] * 100 + f["exact_dup_pairs"]))
    for f in anom_loop[:5]:
        print("\n--- %s  model=%s  l2=%s  n_steps=%d  fail=%d(%s)  exactdup=%d"
              % (f["trajectory_id"], f["source_model"], f["l2"], f["n_steps"],
                 f["max_fail_rep"], f["max_fail_label"], f["exact_dup_pairs"]))
        for k in range(min(5, f["n_steps"])):
            a = (f["actions"][k] or "")[:40]
            o = (f["obs"][k] or "").replace("\n", " ")[:100]
            print("    step%d action=%-12s obs=%s" % (k+1, a, o))

if __name__ == "__main__":
    main()