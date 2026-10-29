# eval_semantic_label.py
# 构造语义标签 + Group CV 重估 baseline
# 运行: uv run --with datasets --with scikit-learn --with scipy python code/eval_semantic_label.py

import json, os, re, sys
import numpy as np
from collections import Counter
from datasets import load_from_disk
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.stats import spearmanr

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

def shingles(text, n=8):
    t = re.sub(r"\s+", " ", text.strip().lower())
    if len(t) < n:
        return set([t]) if t else set()
    return set(t[i:i+n] for i in range(len(t) - n + 1))

def obs_features(obs_list):
    obs_list = [o for o in obs_list if o and o.strip()]
    n = len(obs_list)
    if n < 2:
        return None
    sh = [shingles(o) for o in obs_list]
    all_sh = set(); total = 0
    for s in sh:
        all_sh |= s; total += len(s)
    global_dup = 1 - len(all_sh) / total if total else 0.0
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
    max_fail_ratio = max_fail_rep / n if n else 0.0
    return {
        "n_obs": n, "global_dup": global_dup,
        "exact_dup_pairs": exact_dup_pairs,
        "max_fail_rep": max_fail_rep,
        "max_fail_ratio": max_fail_ratio,
    }

def is_strict(f):
    return f["exact_dup_pairs"] >= 20 or f["max_fail_rep"] >= 10

def is_loose(f):
    if is_strict(f): return True
    return f["exact_dup_pairs"] >= 5 or f["max_fail_rep"] >= 5

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

def eval_group_cv(rows, y_key, feat_keys, n_splits=5):
    iids = np.array([r["iid"] for r in rows])
    y = np.array([r[y_key] for r in rows])
    if len(set(y)) < 2:
        return float("nan"), float("nan"), len(y), int(y.sum())
    X = np.array([[r[k] for k in feat_keys] for r in rows], dtype=float)
    gkf = GroupKFold(n_splits=n_splits)
    oof = np.zeros(len(y))
    try:
        for tr, te in gkf.split(X, y, groups=iids):
            if len(set(y[tr])) < 2:
                oof[te] = y[tr].mean()
                continue
            sc = StandardScaler().fit(X[tr])
            clf = LogisticRegression(max_iter=2000).fit(sc.transform(X[tr]), y[tr])
            oof[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]
        auc = roc_auc_score(y, oof)
        fpr, tpr, _ = roc_curve(y, oof)
        idx = np.searchsorted(fpr, 0.05, side="left")
        idx = min(idx, len(tpr) - 1)
        tpr5 = tpr[idx]
        return auc, tpr5, len(y), int(y.sum())
    except Exception as e:
        return float("nan"), float("nan"), len(y), int(y.sum())

def main():
    wins = load_jsonl(os.path.join(DATA, "windows_v2.jsonl"))
    rmt = load_jsonl(os.path.join(DATA, "rmt-samples.jsonl"))
    sheaf = load_jsonl(os.path.join(DATA, "sheaf-samples.jsonl"))
    rmt_map = {(r["iid"], r["end"]): r for r in rmt}
    sheaf_map = {(r["iid"], r["end"]): r for r in sheaf}
    print("[load] wins=%d rmt=%d sheaf=%d" % (len(wins), len(rmt), len(sheaf)))

    ds = load_from_disk(os.path.join(DATA, "subset2000"))
    rows = []
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
        r = rmt_map[key]; s = sheaf_map.get(key, {})
        rows.append({
            "iid": w["iid"], "end": w["end"],
            "y_old": 1 - int(w["future_has_new_res"]),
            "y_strict": int(is_strict(f)),
            "y_loose": int(is_loose(f)),
            "cmd_novelty": float(w["cmd_novelty"]),
            "hist_nn_last": float(r.get("hist_nn_last", 0.0)),
            "exact_revisit_rate": float(s.get("exact_revisit_rate", 0.0)),
            "n_revisit_edges": float(s.get("n_revisit_edges", 0)),
            "n_exact_edges": float(s.get("n_exact_edges", 0)),
            "n_nonexact_edges": float(s.get("n_nonexact_edges", 0)),
            "mean_nonexact_angle": float(s.get("mean_nonexact_angle", 0.0)),
            "global_dup": f["global_dup"],
            "exact_dup_pairs": float(f["exact_dup_pairs"]),
            "max_fail_rep": float(f["max_fail_rep"]),
        })
    print("[rows] usable=%d  unique iids=%d" % (len(rows), len(set(r["iid"] for r in rows))))

    # --- 1. 标签分布 ---
    print("\n" + "=" * 80)
    print("[1] label distribution")
    print("=" * 80)
    n = len(rows)
    for k in ["y_old", "y_strict", "y_loose"]:
        pos = sum(r[k] for r in rows)
        print("  %-8s pos=%4d/%4d  rate=%.1f%%" % (k, pos, n, pos / n * 100))

    # --- 2. y_old x y_strict 混淆矩阵 ---
    print("\n" + "=" * 80)
    print("[2] confusion: y_old (rows) x y_strict (cols)")
    print("=" * 80)
    mat = {(0,0):0,(0,1):0,(1,0):0,(1,1):0}
    for r in rows:
        mat[(r["y_old"], r["y_strict"])] += 1
    print("                strict=0   strict=1")
    print("  old=0(PROG)   %6d     %6d" % (mat[(0,0)], mat[(0,1)]))
    print("  old=1(LOOP)   %6d     %6d" % (mat[(1,0)], mat[(1,1)]))
    flips = mat[(0,1)] + mat[(1,0)]
    print("  flips = %d / %d (%.1f%%)" % (flips, n, flips / n * 100))

    # --- 3. 特征与标签的相关性 (循环论证诊断) ---
    print("\n" + "=" * 80)
    print("[3] feature-label correlation (spearman rho, cycles check)")
    print("=" * 80)
    feat_keys = ["cmd_novelty", "hist_nn_last", "exact_revisit_rate",
                 "n_revisit_edges", "n_exact_edges", "n_nonexact_edges",
                 "mean_nonexact_angle"]
    for fk in feat_keys:
        xs = np.array([r[fk] for r in rows], dtype=float)
        out = []
        for yk in ["y_old", "y_strict", "y_loose"]:
            ys = np.array([r[yk] for r in rows], dtype=float)
            if len(set(ys)) < 2:
                out.append("  n/a  ")
                continue
            try:
                rho, _ = spearmanr(xs, ys)
                out.append("%+.3f" % rho)
            except Exception:
                out.append("  err  ")
        print("  %-22s old=%s  strict=%s  loose=%s" % (fk, out[0], out[1], out[2]))

    # --- 4. Group CV AUC 对比 ---
    print("\n" + "=" * 80)
    print("[4] Group CV by iid (5-fold), AUC / TPR@5%FPR")
    print("=" * 80)
    models = {
        "M1_cmd": ["cmd_novelty"],
        "M2_cmd+hist": ["cmd_novelty", "hist_nn_last"],
        "M3_cmd+topo": ["cmd_novelty", "n_revisit_edges", "n_exact_edges",
                        "n_nonexact_edges", "mean_nonexact_angle"],
        "M4_cmd+hist+topo": ["cmd_novelty", "hist_nn_last",
                             "n_revisit_edges", "n_exact_edges",
                             "n_nonexact_edges", "mean_nonexact_angle"],
    }
    print("\n  label     model              n     pos    AUC     TPR@5%")
    print("  " + "-" * 70)
    for yk in ["y_old", "y_strict", "y_loose"]:
        for mk, fks in models.items():
            auc, tpr5, nn, npos = eval_group_cv(rows, yk, fks)
            print("  %-8s  %-18s %4d  %4d   %s  %s"
                  % (yk, mk, nn, npos,
                     ("%.4f" % auc) if auc == auc else " nan  ",
                     ("%.4f" % tpr5) if tpr5 == tpr5 else " nan  "))
        print("")

    # --- 5. 关键对比：同一模型跨标签 ---
    print("=" * 80)
    print("[5] delta: same model, label changed (y_old -> y_strict / y_loose)")
    print("=" * 80)
    for mk, fks in models.items():
        a_old, _, _, _ = eval_group_cv(rows, "y_old", fks)
        a_st, _, _, _ = eval_group_cv(rows, "y_strict", fks)
        a_lo, _, _, _ = eval_group_cv(rows, "y_loose", fks)
        d1 = (a_st - a_old) if (a_st == a_st and a_old == a_old) else float("nan")
        d2 = (a_lo - a_old) if (a_lo == a_lo and a_old == a_old) else float("nan")
        print("  %-18s old=%.4f  strict=%.4f (d=%+.4f)  loose=%.4f (d=%+.4f)"
              % (mk,
                 a_old if a_old == a_old else -1,
                 a_st if a_st == a_st else -1, d1,
                 a_lo if a_lo == a_lo else -1, d2))

    # --- 6. 保存新标签 ---
    out_path = os.path.join(DATA, "semantic-labels.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({
                "iid": r["iid"], "end": r["end"],
                "y_old": r["y_old"], "y_strict": r["y_strict"], "y_loose": r["y_loose"],
            }, ensure_ascii=False) + "\n")
    print("\n[saved] %s  (%d rows)" % (out_path, len(rows)))

if __name__ == "__main__":
    main()