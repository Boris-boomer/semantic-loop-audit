import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import json, re
from collections import defaultdict
from datasets import load_from_disk

# ============ 参数 ============
DATA = os.environ.get("TRAJGAP_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))
SUBSET = f"{DATA}/subset2000"
OUT = f"{DATA}/windows_v2.jsonl"
W, K, STRIDE = 15, 10, 5
# =============================

KNOWN_EDITOR = {"open","goto","scroll_down","scroll_up","create","submit","search_dir","search_file","edit"}
KNOWN_SHELL = {
    "ls","grep","find","cat","cd","rm","mkdir","pip","pytest","python","bash",
    "echo","mv","cp","chmod","touch","curl","wget","export","sudo","apt-get",
    "make","docker","git","sed","awk","head","tail","sort","uniq","wc","diff",
    "tar","unzip","which","pwd","env","source","conda","npm","node","yarn",
    "flake8","mypy","pylint","black","isort","pre-commit","coverage","tox",
    "dvc","gh","jq","yq","xargs","tee","less","more","nano","vim","emacs",
}
KNOWN = KNOWN_EDITOR | KNOWN_SHELL

def normalize_cmd(token):
    base = token.split("/")[-1]
    if base in KNOWN:
        return base
    for cmd in sorted(KNOWN, key=len, reverse=True):
        if token.startswith(cmd) and len(token) > len(cmd):
            if token[len(cmd)] in ".-/":
                return cmd
    return "other"

def parse_commands(text):
    cmds = []
    for block in re.findall(r"```(.*?)```", text or "", flags=re.S):
        lines = [ln.strip() for ln in block.strip().splitlines() if ln.strip()]
        if not lines:
            continue
        parts = lines[0].split()
        if not parts:
            continue
        cmds.append((normalize_cmd(parts[0]), parts[1:]))
    return cmds

def basename(path):
    return path.rstrip("/").split("/")[-1] if path else "unknown"

def resource_of(cmd, args):
    if cmd in {"open", "create", "search_dir", "search_file"}:
        return basename(args[0].strip("'\"`")) if args else "none"
    if cmd in {"edit", "goto", "scroll_down", "scroll_up", "search_file"}:
        return "currentfile"
    if cmd in {"python", "pytest", "bash", "pip", "make", "tox", "flake8", "mypy"}:
        return basename(args[0].strip("'\"`")) if args else cmd
    if cmd in {"search_dir", "ls", "cd", "find", "grep", "cat", "rm", "mkdir", "touch", "mv", "cp", "chmod"}:
        return args[0] if args else "cwd"
    if cmd == "submit":
        return "none"
    return args[0] if args else "none"


# ---------------- 选行规则 ----------------
# 对每个 iid，选 ai_steps 最大的 row；tie 取 row_idx 最小。
def select_rows(ds):
    iid_to_rows = defaultdict(list)
    for row_idx, row in enumerate(ds):
        iid = row["instance_id"]
        n_ai = sum(1 for s in row["trajectory"] if s.get("role") == "ai")
        iid_to_rows[iid].append((row_idx, n_ai))

    selected = {}
    n_tie = 0
    for iid, items in iid_to_rows.items():
        max_n = max(n for _, n in items)
        candidates = [idx for idx, n in items if n == max_n]
        chosen = min(candidates)
        is_tie = int(len(candidates) > 1)
        if is_tie:
            n_tie += 1
        selected[iid] = {
            "row_idx": chosen,
            "n_ai": max_n,
            "tie": is_tie,
            "n_rows_for_iid": len(items),
        }

    stats = {
        "n_iid": len(iid_to_rows),
        "n_dup_iid": sum(1 for v in iid_to_rows.values() if len(v) > 1),
        "n_tie": n_tie,
        "n_selected": len(selected),
    }
    return selected, stats


def main():
    ds = load_from_disk(SUBSET)
    selected, stats = select_rows(ds)
    print(f"[select] iid={stats['n_iid']}, 重复 iid={stats['n_dup_iid']}, "
          f"tie={stats['n_tie']}, 选中={stats['n_selected']}")

    out = []
    skipped_short = 0
    for iid, info in selected.items():
        row = ds[info["row_idx"]]
        assert row["instance_id"] == iid, f"row_idx 错配: {iid} vs {row['instance_id']}"
        ai_steps = [s for s in row["trajectory"] if s.get("role") == "ai"]
        assert len(ai_steps) == info["n_ai"], \
            f"{iid}: ai_steps 数 {len(ai_steps)} != 记录 {info['n_ai']}"

        cmds = []
        for step_idx, s in enumerate(ai_steps):
            for cmd, args in parse_commands(s.get("text")):
                res = resource_of(cmd, args)
                cmds.append({"cmd": cmd, "resource": res,
                             "state": f"{cmd}|{res}", "step": step_idx})
        n = len(cmds)
        if n < W + K + 1:
            skipped_short += 1
            continue
        for end in range(W, n - K + 1, STRIDE):
            win, fut = cmds[end-W:end], cmds[end:end+K]
            wr = set(c["resource"] for c in win)
            fr = set(c["resource"] for c in fut)
            states = [c["state"] for c in win]
            out.append({
                "iid": iid,
                "model_name": row["model_name"],
                "row_idx": info["row_idx"],
                "tie": info["tie"],
                "n_ai": info["n_ai"],
                "end": end,
                "rel_pos": end / n,
                "cmd_novelty": len(set(states)) / len(states),
                "win_steps": [c["step"] for c in win],
                "win_states": states,
                "future_has_new_res": int(len(fr - wr) > 0),
            })

    with open(OUT, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_win_iids = len(set(r["iid"] for r in out))
    pos_rate = sum(r["future_has_new_res"] for r in out) / max(len(out), 1)
    print(f"[windows] 写出={len(out)} 到 {OUT}")
    print(f"[windows] 覆盖 iid={n_win_iids}, 跳过(命令数不足)={skipped_short}, "
          f"正例率={pos_rate:.4f}")
    print(f"[windows] tie 窗口数={sum(1 for r in out if r['tie'])}")


if __name__ == "__main__":
    main()