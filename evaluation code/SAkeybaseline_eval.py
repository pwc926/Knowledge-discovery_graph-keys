# sakey_eval_baseline.py
# "Most basic" SAKey baseline evaluation:
#   - No typed blocking (no rdf:type grouping)
#   - No empty-signature filtering
#   - No bucket-size cap
#   - No literal normalization
#
# It evaluates SAKey keys (e.g., -1-almost keys) by:
#   1) building signature buckets in Abox1 and Abox2
#   2) predicting links for equal signatures (cartesian product within each shared bucket)
#   3) computing Precision / Recall / F1 against refalign.rdf gold
#
# Usage (PowerShell example):
#   python sakey_eval_baseline.py `
#     --abox1 SPIMBENCH_small/Abox1.nt --abox2 SPIMBENCH_small/Abox2.nt `
#     --refalign SPIMBENCH_small/refalign.rdf `
#     --sakey1 sakey_abox1.txt --sakey2 sakey_abox2.txt `
#     --almost -1 --try_top 200 --max_keys_parse 500
#
from __future__ import annotations
import argparse
import re
from typing import List, Tuple, Set, Dict

from rdflib import Graph, URIRef
from rdflib.namespace import OWL

ALIGN_E1 = URIRef("http://knowledgeweb.semanticweb.org/heterogeneity/alignmententity1")
ALIGN_E2 = URIRef("http://knowledgeweb.semanticweb.org/heterogeneity/alignmententity2")


def load_nt(path: str) -> Graph:
    g = Graph()
    g.parse(path, format="nt")
    return g


def load_gold_links(refalign_path: str) -> Set[Tuple[str, str]]:
    """
    Supports BOTH:
      - Alignment format: alignmententity1/alignmententity2
      - owl:sameAs (fallback)
    Returns canonical pairs (a,b) with a<b (lexicographically).
    """
    g = Graph()
    g.parse(refalign_path)

    gold: Set[Tuple[str, str]] = set()
    e1, e2 = {}, {}

    for cell, _, ent1 in g.triples((None, ALIGN_E1, None)):
        e1[cell] = ent1
    for cell, _, ent2 in g.triples((None, ALIGN_E2, None)):
        e2[cell] = ent2

    for cell in set(e1.keys()) & set(e2.keys()):
        a, b = sorted([str(e1[cell]), str(e2[cell])])
        gold.add((a, b))

    if not gold:
        for s, _, o in g.triples((None, OWL.sameAs, None)):
            a, b = sorted([str(s), str(o)])
            gold.add((a, b))

    return gold


def prf1(pred: Set[Tuple[str, str]], gold: Set[Tuple[str, str]]):
    tp = len(pred & gold)
    fp = len(pred - gold)
    fn = len(gold - pred)
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * p * r / (p + r)) if (p + r) else 0.0
    return p, r, f1, tp, fp, fn


def parse_sakey_keys(txt_path: str, almost_level: int = -1, max_keys: int = 500) -> List[List[str]]:
    """
    Parse SAKey output:
      -1-almost keys:[[...],[...]]
    Return list of keys (each key is list of predicate URIs).
    """
    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        s = f.read().replace("\x00", "")

    pat = rf"{re.escape(str(almost_level))}-almost keys:\s*(\[\[.*?\]\])"
    m = re.search(pat, s, flags=re.S)
    if not m:
        found = sorted(set(re.findall(r"(-?\d+)-almost keys:", s)))
        raise ValueError(f"Cannot find '{almost_level}-almost keys' in {txt_path}. Found: {found}")

    block = m.group(1)
    key_strs = re.findall(r"\[([^\[\]]+?)\]", block)

    keys: List[List[str]] = []
    for ks in key_strs:
        preds = [x.strip() for x in ks.split(",")]
        preds = [p for p in preds if p.startswith("http")]
        if preds:
            keys.append(preds)

    # dedup + truncate
    out: List[List[str]] = []
    seen = set()
    for k in keys:
        t = tuple(k)
        if t not in seen:
            seen.add(t)
            out.append(k)
        if len(out) >= max_keys:
            break
    return out


def merge_and_dedup_keys(keys1: List[List[str]], keys2: List[List[str]]) -> List[List[str]]:
    out = []
    seen = set()
    for k in keys1 + keys2:
        t = tuple(k)
        if t not in seen:
            seen.add(t)
            out.append(k)
    return out


def all_subject_uris(g: Graph) -> List[URIRef]:
    """
    Baseline: consider ALL URI subjects in the graph (no rdf:type restriction).
    """
    subs = set()
    for s, _, _ in g.triples((None, None, None)):
        if isinstance(s, URIRef):
            subs.add(s)
    return sorted(subs, key=str)


def values_for_pred(g: Graph, subj: URIRef, pred_uri: str) -> Tuple[str, ...]:
    p = URIRef(pred_uri)
    vals = [str(o) for _, _, o in g.triples((subj, p, None))]
    vals.sort()
    return tuple(vals)


def signature_flat(g: Graph, subj: URIRef, key_preds: List[str]) -> Tuple[Tuple[str, ...], ...]:
    return tuple(values_for_pred(g, subj, p) for p in key_preds)


def predict_links_baseline(g1: Graph, g2: Graph, key_preds: List[str]) -> Set[Tuple[str, str]]:
    """
    Baseline matching:
      - bucket Abox1 entities by signature
      - bucket Abox2 entities by signature
      - for each shared signature, output cartesian product of entity pairs
    """
    subs1 = all_subject_uris(g1)
    subs2 = all_subject_uris(g2)

    b1: Dict[Tuple[Tuple[str, ...], ...], List[str]] = {}
    for s in subs1:
        sig = signature_flat(g1, s, key_preds)
        b1.setdefault(sig, []).append(str(s))

    b2: Dict[Tuple[Tuple[str, ...], ...], List[str]] = {}
    for s in subs2:
        sig = signature_flat(g2, s, key_preds)
        b2.setdefault(sig, []).append(str(s))

    pred: Set[Tuple[str, str]] = set()
    for sig in set(b1.keys()) & set(b2.keys()):
        for a in b1[sig]:
            for b in b2[sig]:
                x, y = sorted([a, b])
                pred.add((x, y))
    return pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--abox1", default="SPIMBENCH_small/Abox1.nt")
    ap.add_argument("--abox2", default="SPIMBENCH_small/Abox2.nt")
    ap.add_argument("--refalign", default="SPIMBENCH_small/refalign.rdf")
    ap.add_argument("--sakey1", default="sakey_abox1.txt")
    ap.add_argument("--sakey2", default="sakey_abox2.txt")
    ap.add_argument("--almost", type=int, default=-1)
    ap.add_argument("--try_top", type=int, default=200)
    ap.add_argument("--max_keys_parse", type=int, default=500)
    args = ap.parse_args()

    g1 = load_nt(args.abox1)
    g2 = load_nt(args.abox2)
    gold = load_gold_links(args.refalign)
    print(f"Gold links: {len(gold)}")

    k1 = parse_sakey_keys(args.sakey1, almost_level=args.almost, max_keys=args.max_keys_parse)
    k2 = parse_sakey_keys(args.sakey2, almost_level=args.almost, max_keys=args.max_keys_parse)
    keys = merge_and_dedup_keys(k1, k2)
    if not keys:
        raise RuntimeError("No SAKey keys parsed. Check --almost and SAKey output files.")

    best = None
    N = max(1, min(args.try_top, len(keys)))

    for i, key in enumerate(keys[:N]):
        pred = predict_links_baseline(g1, g2, key)
        p, r, f1, tp, fp, fn = prf1(pred, gold)

        if best is None or f1 > best["f1"]:
            best = {
                "idx": i,
                "key": key,
                "pred_n": len(pred),
                "p": p,
                "r": r,
                "f1": f1,
                "tp": tp,
                "fp": fp,
                "fn": fn,
            }

    print(
        f"[SAKey-baseline-best] idx={best['idx']} pred={best['pred_n']} "
        f"P={best['p']:.3f} R={best['r']:.3f} F1={best['f1']:.3f} "
        f"TP={best['tp']} FP={best['fp']} FN={best['fn']}"
    )
    print("  key =", best["key"])


if __name__ == "__main__":
    main()