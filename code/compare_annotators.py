# compare_annotators.py
# 只读: 两个标注者的一致性分析
# 运行: uv run python code/compare_annotators.py

import json, os, sys
from collections import Counter, defaultdict

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

DATA = os.environ.get("TRAJGAP_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))
KEY_PATH = os.path.join(DATA, "blind_audit_key.json")

# ============ 第一标注者 (作者) ============
# key: (cell, iid, end) -> YES / MAYBE / NO
FIRST = {
    # Cell 00
    ("00", "asottile__pyupgrade-120", 50): "YES",
    ("00", "iterative__dvc-5362", 55): "YES",
    ("00", "cunla__fakeredis-py-127", 85): "YES",
    ("00", "iterative__dvc-8251", 35): "MAYBE",
    ("00", "SAP__python-pyodata-232", 30): "NO",
    ("00", "zalando-stups__senza-397", 15): "YES",
    ("00", "erikrose__parsimonious-211", 30): "NO",
    ("00", "wireservice__agate-excel-25", 25): "NO",
    ("00", "3YOURMIND__django-migration-linter-47", 20): "YES",
    ("00", "internap__python-hostsresolver-5", 115): "YES",
    ("00", "praw-dev__praw-1960", 45): "YES",
    ("00", "beetbox__confuse-97", 40): "MAYBE",
    # Cell 01
    ("01", "dwavesystems__dwave-cloud-client-417", 85): "YES",
    ("01", "asottile__pyupgrade-217", 195): "YES",
    ("01", "joke2k__faker-1569", 35): "YES",
    ("01", "MycroftAI__lingua-franca-226", 45): "YES",
    ("01", "scikit-hep__particle-494", 15): "YES",
    ("01", "iterative__dvc-3329", 20): "YES",
    ("01", "macbre__sql-metadata-316", 20): "YES",
    ("01", "iterative__dvc-8251", 25): "YES",
    ("01", "3YOURMIND__django-migration-linter-47", 30): "YES",
    ("01", "networkx__networkx-5394", 20): "YES",
    ("01", "pydantic__pydantic-2176", 30): "YES",
    ("01", "zopefoundation__persistent-161", 15): "YES",
    # Cell 10
    ("10", "pydantic__pydantic-1210", 45): "MAYBE",
    ("10", "pydantic__pydantic-625", 20): "MAYBE",
    ("10", "sclorg__spec2scl-22", 20): "MAYBE",
    ("10", "pydantic__pydantic-3176", 25): "MAYBE",
    ("10", "RDFLib__rdflib-1012", 20): "MAYBE",
    ("10", "pydantic__pydantic-2618", 25): "MAYBE",
    ("10", "atscub__nautapy-10", 20): "YES",
    ("10", "Pylons__webob-372", 20): "YES",
    ("10", "nipy__heudiconv-523", 15): "YES",
    ("10", "tobymao__sqlglot-1703", 15): "MAYBE",
    ("10", "lark-parser__lark-650", 35): "MAYBE",
    ("10", "just-work__fffw-68", 15): "MAYBE",
    # Cell 11
    ("11", "getnikola__nikola-3504", 20): "YES",
    ("11", "hylang__hy-2423", 35): "YES",
    ("11", "awisu2__imagenatepy-3", 35): "YES",
    ("11", "iterative__dvc-3299", 50): "YES",
    ("11", "airspeed-velocity__asv-733", 25): "YES",
    ("11", "networkx__networkx-5394", 40): "YES",
    ("11", "paver__paver-174", 30): "YES",
    ("11", "sclorg__spec2scl-22", 30): "YES",
    ("11", "pydicom__pydicom-901", 40): "YES",
    ("11", "pydantic__pydantic-3176", 35): "YES",
    ("11", "pydantic__pydantic-4963", 45): "YES",
    ("11", "DCC-Lab__RayTracing-202", 25): "YES",
}

# ============ 第二标注者 (blind LLM) ============
SECOND = {
    "C01": "YES",  "C02": "YES",  "C03": "NO",   "C04": "MAYBE",
    "C05": "YES",  "C06": "YES",  "C07": "MAYBE","C08": "MAYBE",
    "C09": "MAYBE","C10": "MAYBE","C11": "YES",  "C12": "NO",
    "C13": "MAYBE","C14": "MAYBE","C15": "MAYBE","C16": "NO",
    "C17": "MAYBE","C18": "MAYBE","C19": "MAYBE","C20": "NO",
    "C21": "MAYBE","C22": "MAYBE","C23": "MAYBE","C24": "MAYBE",
    "C25": "MAYBE","C26": "YES",  "C27": "MAYBE","C28": "NO",
    "C29": "MAYBE","C30": "MAYBE","C31": "MAYBE","C32": "YES",
    "C33": "MAYBE","C34": "MAYBE","C35": "NO",   "C36": "YES",
    "C37": "YES",  "C38": "YES",  "C39": "NO",   "C40": "YES",
    "C41": "MAYBE","C42": "MAYBE","C43": "MAYBE","C44": "NO",
    "C45": "MAYBE","C46": "MAYBE","C47": "MAYBE","C48": "MAYBE",
}

def cohen_kappa_3(a, b):
    """a, b: list of labels. 3-class Cohen's kappa."""
    labels = ["YES", "MAYBE", "NO"]
    n = len(a)
    if n == 0:
        return float("nan"), 0.0, 0.0
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    ca = Counter(a); cb = Counter(b)
    pe = sum((ca[l] / n) * (cb[l] / n) for l in labels)
    if pe == 1.0:
        return 1.0, po, pe
    kappa = (po - pe) / (1 - pe)
    return kappa, po, pe

def cohen_kappa_binary(a, b):
    """YES + MAYBE -> positive, NO -> negative."""
    def to_bin(x):
        return 1 if x in ("YES", "MAYBE") else 0
    ab = [to_bin(x) for x in a]
    bb = [to_bin(x) for x in b]
    n = len(ab)
    if n == 0:
        return float("nan"), 0.0, 0.0
    po = sum(1 for x, y in zip(ab, bb) if x == y) / n
    pa = sum(ab) / n; pb = sum(bb) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    if pe == 1.0:
        return 1.0, po, pe
    kappa = (po - pe) / (1 - pe)
    return kappa, po, pe

def main():
    with open(KEY_PATH, "r", encoding="utf-8") as f:
        key = json.load(f)

    print("[loaded] %d key entries, %d first-annot, %d second-annot"
          % (len(key), len(FIRST), len(SECOND)))

    # 配对
    pairs = []
    missing_first = []
    missing_second = []
    for cid, meta in key.items():
        cell = meta["cell"]; iid = meta["iid"]; end = meta["end"]
        f_lab = FIRST.get((cell, iid, end))
        s_lab = SECOND.get(cid)
        if f_lab is None:
            missing_first.append((cid, cell, iid, end))
            continue
        if s_lab is None:
            missing_second.append((cid, cell, iid, end))
            continue
        pairs.append({"cid": cid, "cell": cell, "iid": iid, "end": end,
                      "first": f_lab, "second": s_lab})

    print("[paired] %d" % len(pairs))
    if missing_first:
        print("[missing first] %d: %s" % (len(missing_first), missing_first[:3]))
    if missing_second:
        print("[missing second] %d: %s" % (len(missing_second), missing_second[:3]))

    if not pairs:
        return

    # 全体一致性
    print("\n" + "=" * 80)
    print("[1] overall agreement (3-class)")
    print("=" * 80)
    a = [p["first"] for p in pairs]
    b = [p["second"] for p in pairs]
    k3, po3, pe3 = cohen_kappa_3(a, b)
    print("  n=%d  Po=%.3f  Pe=%.3f  Cohen's kappa=%.3f" % (len(pairs), po3, pe3, k3))
    print("  first  dist: %s" % dict(Counter(a)))
    print("  second dist: %s" % dict(Counter(b)))

    print("\n" + "=" * 80)
    print("[2] overall agreement (2-class: YES+MAYBE vs NO)")
    print("=" * 80)
    kb, pob, peb = cohen_kappa_binary(a, b)
    print("  n=%d  Po=%.3f  Pe=%.3f  Cohen's kappa=%.3f" % (len(pairs), pob, peb, kb))

    # 3x3 confusion matrix
    print("\n" + "=" * 80)
    print("[3] 3x3 confusion matrix  (rows=first annotator, cols=second annotator)")
    print("=" * 80)
    labels = ["YES", "MAYBE", "NO"]
    mat = {fa: {sa: 0 for sa in labels} for fa in labels}
    for p in pairs:
        mat[p["first"]][p["second"]] += 1
    print("         " + "  ".join("%6s" % l for l in labels) + "   total")
    for fa in labels:
        row = mat[fa]
        total = sum(row.values())
        print("  %-5s  " % fa + "  ".join("%6d" % row[sa] for sa in labels) + "   %5d" % total)
    # 对角一致
    diag = sum(mat[l][l] for l in labels)
    print("  diagonal agreement: %d/%d = %.1f%%" % (diag, len(pairs), diag / len(pairs) * 100))

    # 分格
    print("\n" + "=" * 80)
    print("[4] by cell")
    print("=" * 80)
    by_cell = defaultdict(list)
    for p in pairs:
        by_cell[p["cell"]].append(p)
    for cell in ["00", "01", "10", "11"]:
        ps = by_cell.get(cell, [])
        if not ps:
            continue
        aa = [p["first"] for p in ps]
        bb = [p["second"] for p in ps]
        k, po, pe = cohen_kappa_3(aa, bb)
        kb_, pob_, peb_ = cohen_kappa_binary(aa, bb)
        diag_n = sum(1 for x, y in zip(aa, bb) if x == y)
        print("\n  Cell %s: n=%d" % (cell, len(ps)))
        print("    first  dist: %s" % dict(Counter(aa)))
        print("    second dist: %s" % dict(Counter(bb)))
        print("    3-class:  Po=%.3f  Pe=%.3f  kappa=%s"
              % (po, pe, ("%.3f" % k) if k == k else "nan"))
        print("    2-class:  Po=%.3f  Pe=%.3f  kappa=%s"
              % (pob_, peb_, ("%.3f" % kb_) if kb_ == kb_ else "nan"))
        print("    diagonal: %d/%d = %.1f%%" % (diag_n, len(ps), diag_n / len(ps) * 100))
        # 列出不一致的 case
        mism = [(p["cid"], p["iid"], p["first"], p["second"])
                for p in ps if p["first"] != p["second"]]
        if mism:
            print("    mismatches:")
            for cid, iid, f1, f2 in mism:
                print("      %s  %-40s  first=%-5s  second=%-5s" % (cid, iid, f1, f2))

    # Cell 00 的 YES-YES 一致率（关键）
    print("\n" + "=" * 80)
    print("[5] Cell 00: how many first=YES are also second=YES?")
    print("=" * 80)
    cell00 = by_cell.get("00", [])
    f_yes = [p for p in cell00 if p["first"] == "YES"]
    print("  first=YES in Cell 00: %d" % len(f_yes))
    agree = sum(1 for p in f_yes if p["second"] == "YES")
    maybe = sum(1 for p in f_yes if p["second"] == "MAYBE")
    no = sum(1 for p in f_yes if p["second"] == "NO")
    print("    second=YES:   %d/%d" % (agree, len(f_yes)))
    print("    second=MAYBE: %d/%d" % (maybe, len(f_yes)))
    print("    second=NO:    %d/%d" % (no, len(f_yes)))
    for p in f_yes:
        print("    %s  %-40s  first=YES  second=%s" % (p["cid"], p["iid"], p["second"]))

if __name__ == "__main__":
    main()