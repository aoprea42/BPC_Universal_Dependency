#!/usr/bin/env python
"""
Convert few-shot UD parsing outputs (results/*.json) to CoNLL-U files.

For every results/<model>__<tb>__<src>__N<N>__s<seed>.json it writes
  conllu_out/<same name>.conllu     predicted annotation (10 columns)
and, once per treebank,
  conllu_out/gold/<tb>.gold.conllu  gold sentences in the same order (for conll18_ud_eval.py)

Gold tokenization is used: FORM comes from the gold file, only LEMMA/UPOS/XPOS/FEATS/HEAD/DEPREL
come from the model. Multiword-token range lines are copied from the gold file so that
the official evaluator can align tokens. Invalid trees (missing rows, no/multiple roots,
cycles, out-of-range heads) are repaired so that the output is always valid CoNLL-U.
"""
import argparse, glob, json, os
from collections import Counter
import conllu


# ---------------------------------------------------------------- parsing (same logic as notebook)
def norm_feats(f):
    if not f or f == "_":
        return "_"
    parts = sorted(p.strip() for p in f.split("|") if p.strip() and p.strip() != "_")
    return "|".join(parts) if parts else "_"


def parse_table(txt, n):
    out = dict(lemmas=[None] * n, upos=[None] * n, xpos=[None] * n,
               feats=[None] * n, heads=[None] * n, labs=[None] * n)
    for line in txt.split("\n"):
        p = line.split("\t")
        if len(p) < 8:
            p = line.split()
            if len(p) < 8:
                continue
        if not p[0].strip().isdigit():
            continue
        i = int(p[0])
        h = p[6].strip()
        if 1 <= i <= n and out["heads"][i - 1] is None and h.isdigit():
            out["lemmas"][i - 1] = p[2].strip()
            out["upos"][i - 1] = p[3].strip().upper()
            out["xpos"][i - 1] = p[4].strip()
            out["feats"][i - 1] = norm_feats(p[5].strip())
            out["heads"][i - 1] = int(h)
            out["labs"][i - 1] = p[7].strip().lower()
    return out


# ---------------------------------------------------------------- tree repair
def repair_tree(heads, rels):
    """heads: list[int|None] (1-based ids, 0=root). Returns (heads, rels, n_fixed)."""
    n = len(heads)
    heads, rels = list(heads), list(rels)
    fixed = 0

    # missing / out-of-range / self-loop heads -> mark invalid
    bad = [h is None or not (0 <= h <= n) or h == i + 1 for i, h in enumerate(heads)]

    roots = [i for i in range(n) if not bad[i] and heads[i] == 0]
    root = roots[0] if roots else 0          # fall back to first token
    heads[root], rels[root] = 0, "root"
    bad[root] = False

    for i in range(n):
        if i == root:
            continue
        if bad[i] or heads[i] == 0:          # invalid or extra root
            heads[i] = root + 1
            rels[i] = "dep" if (bad[i] or rels[i] in (None, "root")) else rels[i]
            fixed += 1

    # break cycles: any node that cannot reach 0 is attached to root
    for i in range(n):
        seen, j = set(), i
        while j != root and j not in seen:
            seen.add(j)
            j = heads[j] - 1
        if j != root:
            heads[i], rels[i] = root + 1, "dep"
            fixed += 1
    return heads, rels, fixed


# ---------------------------------------------------------------- gold handling
def is_word(t):
    return isinstance(t["id"], int)


def is_range(t):
    return isinstance(t["id"], tuple) and len(t["id"]) == 3 and t["id"][1] == "-"


def load_gold(path):
    with open(path, encoding="utf8") as f:
        return {tl.metadata.get("sent_id"): tl for tl in conllu.parse_incr(f)}


def render_prediction(gold_tl, pred_txt):
    words = [t for t in gold_tl if is_word(t)]
    n = len(words)
    p = parse_table(pred_txt, n)
    n_missing = sum(h is None for h in p["heads"])

    rels = [r if r is not None else "dep" for r in p["labs"]]
    heads, rels, n_fixed = repair_tree(p["heads"], rels)

    lines = []
    for k in ("sent_id", "text"):
        if gold_tl.metadata.get(k) is not None:
            lines.append(f"# {k} = {gold_tl.metadata[k]}")
    w = 0
    for t in gold_tl:
        if is_range(t):
            lines.append(f"{t['id'][0]}-{t['id'][2]}\t{t['form']}\t_\t_\t_\t_\t_\t_\t_\t_")
        elif is_word(t):
            lem = p["lemmas"][w] or "_"
            row = [str(t["id"]), t["form"], lem, p["upos"][w] or "_", p["xpos"][w] or "_",
                   p["feats"][w] or "_", str(heads[w]), rels[w], "_", "_"]
            lines.append("\t".join(row))
            w += 1
    return "\n".join(lines) + "\n\n", n_missing, n_fixed


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--data", default="data/data", help="dir with <tb>-ud-test.conllu")
    ap.add_argument("--out", default="conllu_out")
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    os.makedirs(os.path.join(args.out, "gold"), exist_ok=True)
    gold_cache, gold_written = {}, set()
    stats = Counter()

    for fp in sorted(glob.glob(os.path.join(args.results, "*.json"))):
        j = json.load(open(fp, encoding="utf8"))
        tb = j["treebank"]
        if tb not in gold_cache:
            gpath = os.path.join(args.data, f"{tb}-ud-{args.split}.conllu")
            if not os.path.exists(gpath):
                print(f"[skip] gold file missing: {gpath}")
                gold_cache[tb] = None
            else:
                gold_cache[tb] = load_gold(gpath)
        gold = gold_cache[tb]
        if gold is None:
            continue

        chunks, gold_chunks, miss_sents, fix_sents, skipped = [], [], 0, 0, 0
        for item in j["preds"]:
            tl = gold.get(item["id"])
            if item["id"] is None or tl is None:
                skipped += 1
                continue
            text, n_missing, n_fixed = render_prediction(tl, item["pred"] or "")
            chunks.append(text)
            gold_chunks.append(tl.serialize())
            miss_sents += n_missing > 0
            fix_sents += n_fixed > 0

        out_path = os.path.join(args.out, os.path.basename(fp).replace(".json", ".conllu"))
        with open(out_path, "w", encoding="utf8") as f:
            f.write("".join(chunks))

        if tb not in gold_written:
            with open(os.path.join(args.out, "gold", f"{tb}.gold.conllu"), "w", encoding="utf8") as f:
                f.write("".join(gold_chunks))
            gold_written.add(tb)

        stats["files"] += 1
        print(f"{os.path.basename(out_path):60s} sents={len(chunks):4d} "
              f"incomplete={miss_sents:3d} repaired={fix_sents:3d} skipped={skipped}")

    print(f"\nDone: {stats['files']} files -> {args.out}/")


if __name__ == "__main__":
    main()