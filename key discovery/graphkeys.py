# graphkeys.py
# Generate GraphKey outputs in SAKey-like text format:
#   0-non keys: [[...],[...]]
#   -1-almost keys:[[...],[...]]
#
# Representation for path keys:
#   p||q means the path p/q (1-hop object property p followed by property q on the target node)
#
# Usage:
#   python graphkeys.py ^
#     --abox1 SPIMBENCH_small/Abox1.nt --abox2 SPIMBENCH_small/Abox2.nt ^
#     --sakey1 sakey_abox1.txt --sakey2 sakey_abox2.txt ^
#     --out1 graphkey_abox1.txt --out2 graphkey_abox2.txt ^
#     --almost -1 --top_k_single 30 --limit_base 200 --max_out 200
#
from __future__ import annotations
import argparse
import re
from typing import List, Tuple, Set, Dict

from rdflib import Graph, URIRef, BNode

Key = List[str]  # list of predicate strings or path strings (p or p||q)


def load_nt(path: str) -> Graph:
    g = Graph()
    g.parse(path, format="nt")
    return g


def parse_sakey_block(txt_path: str, block_level: int, max_keys: int = 500) -> List[List[str]]:
    """
    Parse a SAKey-like block:
      "<level>-almost keys:[[...],[...]]"   when block_level=-1
      "0-non keys: [[...],[...]]"          when block_level=0 and block name is "non"
    We will:
      - remove NUL bytes
      - extract the list-of-lists and return each inner list as list of URIs
    """
    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        s = f.read().replace("\x00", "")

    if block_level == 0:
        # 0-non keys:
        pat = r"0-non keys:\s*(\[\[.*?\]\])"
    else:
        pat = rf"{re.escape(str(block_level))}-almost keys:\s*(\[\[.*?\]\])"

    m = re.search(pat, s, flags=re.S)
    if not m:
        if block_level == 0:
            raise ValueError(f"Cannot find '0-non keys' in {txt_path}")
        found = sorted(set(re.findall(r"(-?\d+)-almost keys:", s)))
        raise ValueError(f"Cannot find '{block_level}-almost keys' in {txt_path}. Found: {found}")

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


def is_object_property_in_data(g: Graph, p: URIRef) -> bool:
    for _, _, o in g.triples((None, p, None)):
        if isinstance(o, (URIRef, BNode)):
            return True
    return False


def is_literal_property_in_data(g: Graph, p: URIRef) -> bool:
    for _, _, o in g.triples((None, p, None)):
        if not isinstance(o, (URIRef, BNode)):
            return True
    return False


def pick_q_candidates_from_sakey(keys: List[List[str]], g_all: Graph, top_k_single: int) -> List[str]:
    """
    Use SAKey single-property keys as q candidates, prefer those that appear as literal props in data.
    """
    singles = [k[0] for k in keys if len(k) == 1]
    seen = set()
    q = []
    for p in singles:
        if p in seen:
            continue
        seen.add(p)
        if is_literal_property_in_data(g_all, URIRef(p)):
            q.append(p)
        if len(q) >= max(1, top_k_single):
            break

    if not q:  # fallback
        q = singles[: max(1, top_k_single)]
    return q


def build_graphkeys_depth1_from_basekeys(
    base_keys: List[List[str]],
    g_all: Graph,
    q_candidates: List[str],
    max_out: int,
) -> List[Key]:
    """
    For each base key [p1,p2,...]:
      - datatype prop pi -> keep "pi"
      - object prop pi   -> replace by "pi||q" (choose ONE q per pi, default choose first q)
    Output keys are lists of strings (some can be "p||q").
    """
    out: List[Key] = []
    seen = set()

    for key in base_keys:
        new_key: List[str] = []
        for p_str in key:
            p = URIRef(p_str)
            if is_object_property_in_data(g_all, p) and q_candidates:
                # choose first q by default (simple); you can also generate multiple variants
                q = q_candidates[0]
                new_key.append(f"{p_str}||{q}")
            else:
                new_key.append(p_str)

        # normalize order for stable representation
        new_key_sorted = sorted(new_key)
        t = tuple(new_key_sorted)
        if t not in seen:
            seen.add(t)
            out.append(new_key_sorted)

        if len(out) >= max_out:
            break

    return out


def format_sakey_style(non_keys: List[Key], almost_keys: List[Key], almost_level: int) -> str:
    """
    Return SAKey-like text content.
    """
    def fmt(keys: List[Key]) -> str:
        inner = []
        for k in keys:
            inner.append("[" + ", ".join(k) + "]")
        return "[" + ", ".join(inner) + "]"

    s = ""
    s += f"0-non keys: {fmt(non_keys)}\n\n"
    s += f"{almost_level}-almost keys:{fmt(almost_keys)}\n"
    return s


def write_text(path: str, content: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--abox1", default="SPIMBENCH_small/Abox1.nt")
    ap.add_argument("--abox2", default="SPIMBENCH_small/Abox2.nt")
    ap.add_argument("--sakey1", default="sakey_abox1.txt")
    ap.add_argument("--sakey2", default="sakey_abox2.txt")
    ap.add_argument("--out1", default="graphkey_abox1.txt")
    ap.add_argument("--out2", default="graphkey_abox2.txt")
    ap.add_argument("--almost", type=int, default=-1)
    ap.add_argument("--top_k_single", type=int, default=30)
    ap.add_argument("--limit_base", type=int, default=200)
    ap.add_argument("--max_out", type=int, default=200)
    ap.add_argument("--max_parse", type=int, default=500)
    args = ap.parse_args()

    # Load data
    g1 = load_nt(args.abox1)
    g2 = load_nt(args.abox2)
    g_all = Graph()
    for t in g1:
        g_all.add(t)
    for t in g2:
        g_all.add(t)

    # Parse SAKey blocks
    non1 = parse_sakey_block(args.sakey1, block_level=0, max_keys=args.max_parse)
    non2 = parse_sakey_block(args.sakey2, block_level=0, max_keys=args.max_parse)
    alm1 = parse_sakey_block(args.sakey1, block_level=args.almost, max_keys=args.max_parse)
    alm2 = parse_sakey_block(args.sakey2, block_level=args.almost, max_keys=args.max_parse)

    non = merge_and_dedup_keys(non1, non2)
    alm = merge_and_dedup_keys(alm1, alm2)

    # pick q candidates (single-prop keys)
    q_candidates = pick_q_candidates_from_sakey(alm, g_all, args.top_k_single)

    # build graph keys from top base keys
    base_non = non[: max(1, args.limit_base)]
    base_alm = alm[: max(1, args.limit_base)]

    graph_non = build_graphkeys_depth1_from_basekeys(base_non, g_all, q_candidates, max_out=args.max_out)
    graph_alm = build_graphkeys_depth1_from_basekeys(base_alm, g_all, q_candidates, max_out=args.max_out)

    content = format_sakey_style(graph_non, graph_alm, almost_level=args.almost)

    # same content for both (because we're generating from merged keys + merged data)
    write_text(args.out1, content)
    write_text(args.out2, content)

    print("Wrote SAKey-style GraphKey files:")
    print(" ", args.out1)
    print(" ", args.out2)
    print("q_candidates (first used):", q_candidates[:10])


if __name__ == "__main__":
    main()