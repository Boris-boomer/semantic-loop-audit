# probe_cua_debugger.py
# 只读: CUA Debugger 标注层探底 (taxonomy distribution + R-series evidence)
# 内置镜像 + 直链降级，绕过 OfflineModeIsEnabled
# 运行: uv run --with datasets --with pyarrow python code/probe_cua_debugger.py

import os
import sys

# --- 必须在 import datasets 之前设置 ---
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ.pop("HF_HUB_OFFLINE", None)
os.environ.pop("TRANSFORMERS_OFFLINE", None)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from collections import Counter

DATA = os.environ.get("TRAJGAP_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))

def preview(obj, n=600):
    s = str(obj).replace("\n", " ")
    return s[:n]

def load_ds():
    from datasets import load_dataset

    # --- try 1: repo id (via HF_ENDPOINT) ---
    try:
        print("[load] try 1: load_dataset('CyT1ng/cua_debugger_traj', split='train')")
        ds = load_dataset("CyT1ng/cua_debugger_traj", split="train")
        print("[load] success via repo id")
        return ds
    except Exception as e:
        print("[load] try 1 failed: %r" % e)

    # --- try 2: direct parquet URL (via hf-mirror) ---
    urls = [
        "https://hf-mirror.com/datasets/CyT1ng/cua_debugger_traj/resolve/main/data/train.parquet",
        "https://hf-mirror.com/datasets/CyT1ng/cua_debugger_traj/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet",
    ]
    for i, url in enumerate(urls, 1):
        try:
            print("[load] try 2.%d: %s" % (i, url))
            ds = load_dataset("parquet", data_files={"train": url}, split="train")
            print("[load] success via direct URL")
            return ds
        except Exception as e:
            print("[load] try 2.%d failed: %r" % (i, e))

    # --- try 3: local parquet under TRAJGAP_DATA ---
    local_candidates = [
        os.path.join(DATA, "cua_debugger", "data", "train.parquet"),
        os.path.join(DATA, "cua_debugger", "train.parquet"),
    ]
    for p in local_candidates:
        if os.path.exists(p):
            try:
                print("[load] try 3: local file %s" % p)
                ds = load_dataset("parquet", data_files={"train": p}, split="train")
                print("[load] success via local file")
                return ds
            except Exception as e:
                print("[load] try 3 failed: %r" % e)

    return None

def main():
    print("[loading] CyT1ng/cua_debugger_traj ...")
    ds = load_ds()
    if ds is None:
        print("\n[FATAL] all load paths failed.")
        print("Manual download:")
        print('  $env:HF_ENDPOINT = "https://hf-mirror.com"')
        print('  uv run --with huggingface_hub huggingface-cli download --repo-type dataset --resume-download CyT1ng/cua_debugger_traj data/train.parquet --local-dir <TRAJGAP_DATA>/cua_debugger')
        return

    print("[loaded] n=%d" % len(ds))
    print("[columns] %s" % ds.column_names)

    print("\n" + "=" * 80)
    print("[1] first record, all fields")
    print("=" * 80)
    r0 = ds[0]
    for k, v in r0.items():
        if isinstance(v, str):
            print("  %-20s str len=%d  %s" % (k, len(v), preview(v, 300)))
        elif isinstance(v, list):
            print("  %-20s list len=%d  %s" % (k, len(v), preview(v, 300)))
        else:
            print("  %-20s %s = %s" % (k, type(v).__name__, preview(v, 300)))

    tax_key = None
    for cand in ["taxonomy_tag", "taxonomy", "root_cause_tag", "label", "error_type"]:
        if cand in ds.column_names:
            tax_key = cand
            break
    if tax_key is None:
        print("\n[WARN] taxonomy field not found; columns = %s" % ds.column_names)
        return
    print("\n[taxonomy key] %s" % tax_key)

    print("\n" + "=" * 80)
    print("[2] taxonomy distribution")
    print("=" * 80)
    tags = Counter(str(r[tax_key]) for r in ds)
    for tag, cnt in tags.most_common():
        print("  %-30s %3d" % (tag, cnt))

    print("\n" + "=" * 80)
    print("[3] R-series entries: tag + evidence (first 200 chars)")
    print("=" * 80)
    r_rows = [r for r in ds if str(r[tax_key]).upper().startswith("R")]
    print("[R series] n=%d" % len(r_rows))
    by_tag = {}
    for r in r_rows:
        by_tag.setdefault(str(r[tax_key]), []).append(r)
    for tag in sorted(by_tag.keys()):
        pool = by_tag[tag]
        print("\n--- %s  (n=%d) ---" % (tag, len(pool)))
        for r in pool[:5]:
            ev = r.get("evidence", "")
            print("  root_step=%s  evidence=%s"
                  % (r.get("root_error_step"), preview(ev, 200)))

    print("\n" + "=" * 80)
    print("[4] R4 entries: full evidence")
    print("=" * 80)
    r4 = [r for r in ds if str(r[tax_key]).strip().upper() == "R4"]
    if not r4:
        print("  (no R4; R-series tags = %s)" % sorted(by_tag.keys()))
    else:
        for i, r in enumerate(r4):
            print("\n--- R4 #%d  root_step=%s" % (i, r.get("root_error_step")))
            print("  evidence: %s" % preview(r.get("evidence", ""), 800))
            print("  correction: %s" % preview(r.get("correction", ""), 400))

    print("\n" + "=" * 80)
    print("[5] evidence entries containing loop/repeat/redundant/stuck")
    print("=" * 80)
    KWS = ["loop", "repeat", "redundant", "again", "same action", "retry", "stuck"]
    hits = []
    for r in ds:
        ev = str(r.get("evidence", "")).lower()
        matched = [k for k in KWS if k in ev]
        if matched:
            hits.append((matched, r))
    print("[keyword hits] n=%d" % len(hits))
    for matched, r in hits[:10]:
        print("\n--- tag=%s  root_step=%s  matched=%s"
              % (r.get(tax_key), r.get("root_error_step"), matched))
        print("  evidence: %s" % preview(r.get("evidence", ""), 300))

if __name__ == "__main__":
    main()