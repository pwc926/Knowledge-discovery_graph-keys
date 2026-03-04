# graphkey_eval_typed.py
# GraphKey evaluation WITH rdf:type filtering (typed blocking),
# reading SAKey-style GraphKey output files:
#   0-non keys: [[...],[...]]
#   -1-almost keys:[[...],[...]]
#
# Path format:
#   "p||q" means graph path p/q (1-hop object property p then property q on the target node)
#
# Key fixes to improve precision stability:
#   (1) Drop entities whose signature is ALL empty (prevents huge empty-bucket cartesian explosion)
#   (2) Optional: drop too-large buckets (max_bucket_size)
#   (3) Optional: evaluate only "real" graph keys (must contain at least one p||q)
#
# Usage:
#   python graphkey_eval_typed.py ^
#     --abox1 SPIMBENCH_small/Abox1.nt --abox2 SPIMBENCH_small/Abox2.nt ^
#     --refalign SPIMBENCH_small/refalign.rdf ^
#     --gk1 graphkey_abox1.txt --gk2 graphkey_abox2.txt ^
#     --almost -1 --try_top 200 --max_keys_parse 300 ^
#     --only_real_graphkeys --max_bucket_size 50
#
from __future__ import annotations
import argparse
import re
from typing import List, Tuple, Set, Dict, Optional

from rdflib import Graph, URIRef, BNode
from rdflib.namespace import RDF, OWL

ALIGN_E1 = URIRef("http://knowledgeweb.semanticweb.org/heterogeneity/alignmententity1")
ALIGN_E2 = URIRef("http://knowledgeweb.semanticweb.org/heterogeneity/alignmententity2")

Path = Tuple[str, ...]      # (p,) or (p,q) where (p,q) means p||q
GraphKey = Tuple[Path, ...]


# -----------------------------
# IO: graphs + gold
# -----------------------------
def load_nt(path: str) -> Graph:
    g = Graph()
    g.parse(path, format="nt")
    return g


def load_gold_links(refalign_path: str) -> Set[Tuple[str, str]]:
    """
    Supports BOTH:
      - Alignment format: alignmententity1/alignmententity2
      - owl:sameAs (fallback)
    Return canonical pairs (a,b) with a<b
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


# -----------------------------
# Metrics
# -----------------------------
def prf1(pred: Set[Tuple[str, str]], gold: Set[Tuple[str, str]]):
    tp = len(pred & gold)
    fp = len(pred - gold)
    fn = len(gold - pred)
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * p * r / (p + r)) if (p + r) else 0.0
    return p, r, f1, tp, fp, fn


# -----------------------------
# Typed blocking
# -----------------------------
def subjects_by_type(g: Graph) -> Dict[str, List[URIRef]]:
    """
    Map class_uri -> list of URI subjects with rdf:type = class_uri
    """
    m: Dict[str, List[URIRef]] = {}
    for s, _, c in g.triples((None, RDF.type, None)):
        if isinstance(s, URIRef) and isinstance(c, URIRef):
            m.setdefault(str(c), []).append(s)

    for k in m:
        m[k] = sorted(set(m[k]), key=str)
    return m


# -----------------------------
# GraphKey parsing (SAKey-like blocks)
# -----------------------------
def _parse_list_of_lists_block(block_text: str, max_keys: int) -> List[List[str]]:
    """
    Given text like: [[a,b],[c],[d,e]]
    return list of list[str]
    """
    key_strs = re.findall(r"\[([^\[\]]+?)\]", block_text)
    keys: List[List[str]] = []
    for ks in key_strs:
        preds = [x.strip() for x in ks.split(",")]
        preds = [p for p in preds if p.startswith("http")]  # includes "p||q" (starts with http)
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


def parse_graphkey_block(txt_path: str, almost_level: int = -1, max_keys: int = 200) -> List[GraphKey]:
    """
    Parse GraphKeys from SAKey-like GraphKey output file:
      -1-almost keys:[[...],[...]]
    Each key item can be:
      - "p"      -> (p,)
      - "p||q"   -> (p,q)
    Return list[GraphKey]
    """
    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        s = f.read().replace("\x00", "")

    pat = rf"{re.escape(str(almost_level))}-almost keys:\s*(\[\[.*?\]\])"
    m = re.search(pat, s, flags=re.S)
    if not m:
        found = sorted(set(re.findall(r"(-?\d+)-almost keys:", s)))
        raise ValueError(
            f"Cannot find '{almost_level}-almost keys' in {txt_path}. Found: {found}"
        )

    keys_raw = _parse_list_of_lists_block(m.group(1), max_keys=max_keys)

    gks: List[GraphKey] = []
    for key in keys_raw:
        paths: List[Path] = []
        for token in key:
            if "||" in token:
                p, q = token.split("||", 1)
                paths.append((p.strip(), q.strip()))
            else:
                paths.append((token.strip(),))
        gks.append(tuple(sorted(paths)))

    # dedup
    return list(dict.fromkeys(gks))


def merge_and_dedup_graphkeys(gks1: List[GraphKey], gks2: List[GraphKey]) -> List[GraphKey]:
    return list(dict.fromkeys(gks1 + gks2))


# -----------------------------
# Signature + prediction
# -----------------------------
def values_1hop(g: Graph, s: URIRef, p: URIRef) -> Tuple[str, ...]:
    vals = [str(o) for _, _, o in g.triples((s, p, None))]
    vals.sort()
    return tuple(vals)


def values_2hop(g: Graph, s: URIRef, p: URIRef, q: URIRef) -> Tuple[str, ...]:
    vals: List[str] = []
    for _, _, x in g.triples((s, p, None)):
        if not isinstance(x, (URIRef, BNode)):
            continue
        for _, _, v in g.triples((x, q, None)):
            vals.append(str(v))
    vals.sort()
    return tuple(vals)


def signature_graphkey(g: Graph, s: URIRef, gk: GraphKey) -> Tuple[Tuple[str, ...], ...]:
    parts: List[Tuple[str, ...]] = []
    for path in gk:
        if len(path) == 1:
            parts.append(values_1hop(g, s, URIRef(path[0])))
        else:
            parts.append(values_2hop(g, s, URIRef(path[0]), URIRef(path[1])))
    return tuple(parts)


def _is_all_empty_signature(sig: Tuple[Tuple[str, ...], ...]) -> bool:
    return all(len(part) == 0 for part in sig)


def predict_links_graphkey_typed(
    g1: Graph,
    g2: Graph,
    gk: GraphKey,
    *,
    drop_all_empty_signature: bool = True,
    max_bucket_size: Optional[int] = None,
) -> Set[Tuple[str, str]]:
    """
    Only compare entities within the SAME rdf:type.
    Improvements:
      - drop entities whose signature is all empty (default True)
      - optionally drop too-large buckets in each side before cartesian product
    """
    t1 = subjects_by_type(g1)
    t2 = subjects_by_type(g2)
    common_types = set(t1.keys()) & set(t2.keys())

    preds: Set[Tuple[str, str]] = set()

    for cls in common_types:
        inst1 = t1[cls]
        inst2 = t2[cls]

        b1: Dict[Tuple[Tuple[str, ...], ...], List[str]] = {}
        for s in inst1:
            sig = signature_graphkey(g1, s, gk)
            if drop_all_empty_signature and _is_all_empty_signature(sig):
                continue
            b1.setdefault(sig, []).append(str(s))

        b2: Dict[Tuple[Tuple[str, ...], ...], List[str]] = {}
        for s in inst2:
            sig = signature_graphkey(g2, s, gk)
            if drop_all_empty_signature and _is_all_empty_signature(sig):
                continue
            b2.setdefault(sig, []).append(str(s))

        # optional bucket filtering: remove huge buckets to avoid cartesian explosion
        if max_bucket_size is not None:
            b1 = {sig: lst for sig, lst in b1.items() if len(lst) <= max_bucket_size}
            b2 = {sig: lst for sig, lst in b2.items() if len(lst) <= max_bucket_size}

        for sig in set(b1.keys()) & set(b2.keys()):
            for a in b1[sig]:
                for b in b2[sig]:
                    x, y = sorted([a, b])
                    preds.add((x, y))

    return preds


# -----------------------------
# Main (SAKey-like output style)
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--abox1", default="SPIMBENCH_small/Abox1.nt")
    ap.add_argument("--abox2", default="SPIMBENCH_small/Abox2.nt")
    ap.add_argument("--refalign", default="SPIMBENCH_small/refalign.rdf")
    ap.add_argument("--gk1", default="graphkey_abox1.txt")
    ap.add_argument("--gk2", default="graphkey_abox2.txt")
    ap.add_argument("--almost", type=int, default=-1)
    ap.add_argument("--try_top", type=int, default=200)
    ap.add_argument("--max_keys_parse", type=int, default=300)

    ap.add_argument("--only_real_graphkeys", action="store_true",
                    help="Only evaluate keys that contain at least one path p||q")
    ap.add_argument("--max_bucket_size", type=int, default=0,
                    help="If >0, drop buckets larger than this size on either side (reduces FP).")
    ap.add_argument("--keep_all_empty_signature", action="store_true",
                    help="Disable dropping entities whose signature is all empty (NOT recommended).")

    args = ap.parse_args()

    g1 = load_nt(args.abox1)
    g2 = load_nt(args.abox2)
    gold = load_gold_links(args.refalign)
    print(f"Gold links: {len(gold)}")

    gks1 = parse_graphkey_block(args.gk1, almost_level=args.almost, max_keys=args.max_keys_parse)
    gks2 = parse_graphkey_block(args.gk2, almost_level=args.almost, max_keys=args.max_keys_parse)
    gks = merge_and_dedup_graphkeys(gks1, gks2)

    if args.only_real_graphkeys:
        gks = [gk for gk in gks if any(len(path) == 2 for path in gk)]

    if not gks:
        raise RuntimeError("No GraphKeys loaded (after filtering). Check graphkey_abox*.txt and --almost")

    max_bucket = args.max_bucket_size if args.max_bucket_size > 0 else None
    drop_empty = not args.keep_all_empty_signature

    best = None
    N = max(1, min(args.try_top, len(gks)))

    for i, gk in enumerate(gks[:N]):
        pred = predict_links_graphkey_typed(
            g1,
            g2,
            gk,
            drop_all_empty_signature=drop_empty,
            max_bucket_size=max_bucket,
        )
        p, r, f1, tp, fp, fn = prf1(pred, gold)

        if best is None or f1 > best["f1"]:
            best = {
                "idx": i,
                "gk": gk,
                "pred_n": len(pred),
                "p": p,
                "r": r,
                "f1": f1,
                "tp": tp,
                "fp": fp,
                "fn": fn,
            }

    assert best is not None
    print(
        f"[GraphKey-typed-best] idx={best['idx']} pred={best['pred_n']} "
        f"P={best['p']:.3f} R={best['r']:.3f} F1={best['f1']:.3f} "
        f"TP={best['tp']} FP={best['fp']} FN={best['fn']}"
    )

    pretty = []
    for path in best["gk"]:
        if len(path) == 2:
            pretty.append(path[0] + "||" + path[1])
        else:
            pretty.append(path[0])
    print("  graph_key =", pretty)


if __name__ == "__main__":
    main()