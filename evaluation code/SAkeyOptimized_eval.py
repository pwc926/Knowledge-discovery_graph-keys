# sakey_eval_typed.py (patched)
# SAKey baseline evaluation WITH rdf:type filtering (recommended).
# Improvements:
#   (1) Drop entities whose signature is ALL empty (prevents empty-bucket explosion)
#   (2) Optional: drop too-large buckets (max_bucket_size)
#   (3) Optional: include untyped entities in a virtual class

from __future__ import annotations
import argparse
import re
from typing import List, Tuple, Set, Dict, Optional

from rdflib import Graph, URIRef, Literal
from rdflib.namespace import RDF, OWL

ALIGN_E1 = URIRef("http://knowledgeweb.semanticweb.org/heterogeneity/alignmententity1")
ALIGN_E2 = URIRef("http://knowledgeweb.semanticweb.org/heterogeneity/alignmententity2")

UNTYPED = "__UNTYPED__"


def load_gold_links(refalign_path: str) -> Set[Tuple[str, str]]:
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


def parse_sakey_keys(txt_path: str, almost_level: int = -1, max_keys: int = 200) -> List[List[str]]:
    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        s = f.read()
    s = s.replace("\x00", "")  # important for SAKey output on Windows

    pat = rf"{re.escape(str(almost_level))}-almost keys:\s*(\[\[.*?\]\])"
    m = re.search(pat, s, flags=re.S)
    if not m:
        found = re.findall(r"(-?\d+)-almost keys:", s)
        raise ValueError(
            f"Cannot find '{almost_level}-almost keys' in {txt_path}. Found: {sorted(set(found))}"
        )

    block = m.group(1)
    key_strs = re.findall(r"\[([^\[\]]+?)\]", block)

    keys: List[List[str]] = []
    for ks in key_strs:
        preds = [x.strip() for x in ks.split(",")]
        preds = [p for p in preds if p.startswith("http")]
        if preds:
            keys.append(preds)

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


def load_nt(path: str) -> Graph:
    g = Graph()
    g.parse(path, format="nt")
    return g


def subjects_by_type(g: Graph, include_untyped: bool = False) -> Dict[str, List[URIRef]]:
    """
    Map class_uri -> list of URI subjects with rdf:type = class_uri
    If include_untyped=True, put subjects without rdf:type into a virtual class.
    """
    typed: Dict[str, Set[URIRef]] = {}
    all_subjects: Set[URIRef] = set()

    for s, _, _ in g.triples((None, None, None)):
        if isinstance(s, URIRef):
            all_subjects.add(s)

    has_type: Set[URIRef] = set()
    for s, _, c in g.triples((None, RDF.type, None)):
        if isinstance(s, URIRef) and isinstance(c, URIRef):
            typed.setdefault(str(c), set()).add(s)
            has_type.add(s)

    if include_untyped:
        untyped = sorted(all_subjects - has_type, key=str)
        if untyped:
            typed.setdefault(UNTYPED, set()).update(untyped)

    # stable order
    out: Dict[str, List[URIRef]] = {}
    for k, vs in typed.items():
        out[k] = sorted(vs, key=str)
    return out


def _normalize_node(o) -> str:
    """
    Minimal normalization (safe):
    - Literal: use lexical form only (drops lang/datatype differences)
    - URIRef/BNode: str(o)
    You can expand this if desired.
    """
    if isinstance(o, Literal):
        return str(o.value).strip()
    return str(o).strip()


def values_for_pred(g: Graph, subj: URIRef, pred_uri: str, normalize: bool) -> Tuple[str, ...]:
    p = URIRef(pred_uri)
    vals = []
    for _, _, o in g.triples((subj, p, None)):
        vals.append(_normalize_node(o) if normalize else str(o))
    vals.sort()
    return tuple(vals)


def signature_flat(g: Graph, subj: URIRef, key_preds: List[str], normalize: bool) -> Tuple[Tuple[str, ...], ...]:
    return tuple(values_for_pred(g, subj, p, normalize=normalize) for p in key_preds)


def _is_all_empty_signature(sig: Tuple[Tuple[str, ...], ...]) -> bool:
    return all(len(part) == 0 for part in sig)


def predict_links_flat_typed(
    g1: Graph,
    g2: Graph,
    key_preds: List[str],
    *,
    include_untyped: bool = False,
    drop_all_empty_signature: bool = True,
    max_bucket_size: Optional[int] = None,
    normalize_literals: bool = False,
) -> Set[Tuple[str, str]]:
    """
    Only compare entities within the SAME rdf:type (and optionally include untyped).
    """
    t1 = subjects_by_type(g1, include_untyped=include_untyped)
    t2 = subjects_by_type(g2, include_untyped=include_untyped)
    common_types = set(t1.keys()) & set(t2.keys())

    preds: Set[Tuple[str, str]] = set()

    for cls in common_types:
        inst1 = t1[cls]
        inst2 = t2[cls]

        buckets1: Dict[Tuple[Tuple[str, ...], ...], List[str]] = {}
        for s in inst1:
            sig = signature_flat(g1, s, key_preds, normalize=normalize_literals)
            if drop_all_empty_signature and _is_all_empty_signature(sig):
                continue
            buckets1.setdefault(sig, []).append(str(s))

        buckets2: Dict[Tuple[Tuple[str, ...], ...], List[str]] = {}
        for s in inst2:
            sig = signature_flat(g2, s, key_preds, normalize=normalize_literals)
            if drop_all_empty_signature and _is_all_empty_signature(sig):
                continue
            buckets2.setdefault(sig, []).append(str(s))

        if max_bucket_size is not None:
            buckets1 = {sig: lst for sig, lst in buckets1.items() if len(lst) <= max_bucket_size}
            buckets2 = {sig: lst for sig, lst in buckets2.items() if len(lst) <= max_bucket_size}

        for sig in set(buckets1.keys()) & set(buckets2.keys()):
            for a in buckets1[sig]:
                for b in buckets2[sig]:
                    x, y = sorted([a, b])
                    preds.add((x, y))

    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--abox1", default="SPIMBENCH_small/Abox1.nt")
    ap.add_argument("--abox2", default="SPIMBENCH_small/Abox2.nt")
    ap.add_argument("--refalign", default="SPIMBENCH_small/refalign.rdf")
    ap.add_argument("--sakey1", default="sakey_abox1.txt")
    ap.add_argument("--sakey2", default="sakey_abox2.txt")
    ap.add_argument("--almost", type=int, default=-1)
    ap.add_argument("--try_top", type=int, default=50)
    ap.add_argument("--max_keys_parse", type=int, default=200)

    ap.add_argument("--include_untyped", action="store_true")
    ap.add_argument("--keep_all_empty_signature", action="store_true")
    ap.add_argument("--max_bucket_size", type=int, default=0)
    ap.add_argument("--normalize_literals", action="store_true")

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

    max_bucket = args.max_bucket_size if args.max_bucket_size > 0 else None
    drop_empty = not args.keep_all_empty_signature

    best = None
    N = max(1, min(args.try_top, len(keys)))
    for i, key in enumerate(keys[:N]):
        pred = predict_links_flat_typed(
            g1,
            g2,
            key,
            include_untyped=args.include_untyped,
            drop_all_empty_signature=drop_empty,
            max_bucket_size=max_bucket,
            normalize_literals=args.normalize_literals,
        )
        p, r, f1, tp, fp, fn = prf1(pred, gold)
        if best is None or f1 > best["f1"]:
            best = {"idx": i, "key": key, "pred_n": len(pred), "p": p, "r": r, "f1": f1, "tp": tp, "fp": fp, "fn": fn}

    print(
        f"[SAKey-typed-best] idx={best['idx']} pred={best['pred_n']} "
        f"P={best['p']:.3f} R={best['r']:.3f} F1={best['f1']:.3f} "
        f"TP={best['tp']} FP={best['fp']} FN={best['fn']}"
    )
    print("  key =", best["key"])


if __name__ == "__main__":
    main()