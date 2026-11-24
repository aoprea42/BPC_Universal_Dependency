#!/usr/bin/env python3
"""
Standalone UD evaluation (UDPipe 2.0 Table 5 metrics, gold tokenization).

Metrics: UPOS, XPOS, UFeats, AllTags, Lemmas, UAS, LAS, MLAS, BLEX
(same definitions as the evaluation section of fewshot_ud_parsing.ipynb).

Modes
-----
1) files    Compare a predicted CoNLL-U file against a gold CoNLL-U file.
      python eval_ud.py files --gold en_ewt-ud-test.conllu --pred pred.conllu

2) results  Re-score the JSON files written by the notebook (results/*.json).
            Gold sentences are looked up by sent_id in the treebank test file
            (downloaded from GitHub if missing).
      python eval_ud.py results --results results --data data/data --csv summary.csv

3) selftest Oracle check: gold scored against itself must give 100.0 everywhere.
      python eval_ud.py selftest --gold en_ewt-ud-test.conllu

Requires: pip install conllu
"""
import argparse
import csv
import glob
import json
import os
import sys
import urllib.request
from collections import defaultdict

import conllu

METRICS = ["UPOS", "XPOS", "UFeats", "AllTags", "Lemmas", "UAS", "LAS", "MLAS", "BLEX"]

# CoNLL 2018 functional / non-content relations for MLAS and BLEX
FUNC_RELS = {"aux", "cop", "mark", "det", "clf", "case"}
NON_CONTENT_RELS = FUNC_RELS | {"punct", "goeswith", "reparandum", "cc", "flat",
                                "compound", "list", "orphan", "fixed"}

# treebank key -> (GitHub repo, file prefix); extend as needed
TREEBANKS = {
    "en_ewt":       ("UD_English-EWT", "en_ewt"),
    "en_gum":       ("UD_English-GUM", "en_gum"),
    "de_gsd":       ("UD_German-GSD", "de_gsd"),
    "de_hdt":       ("UD_German-HDT", "de_hdt"),
    "af_afribooms": ("UD_Afrikaans-AfriBooms", "af_afribooms"),
    "fo_farpahc":   ("UD_Faroese-FarPaHC", "fo_farpahc"),
    "fo_oft":       ("UD_Faroese-OFT", "fo_oft"),
    "bar_maibaam":  ("UD_Bavarian-MaiBaam", "bar_maibaam"),
    "hy_armtdp":    ("UD_Armenian-ArmTDP", "hy_armtdp"),
    "sq_staf":      ("UD_Albanian-STAF", "sq_staf"),
    "fa_perdt":     ("UD_Persian-PerDT", "fa_perdt"),
}


# ----------------------------------------------------------------------------
# Reading CoNLL-U
# ----------------------------------------------------------------------------
def clean(val):
    if val is None or val == "":
        return "_"
    return str(val).replace(" ", "_").replace("\t", "_").replace("\n", "")


def format_feats(feats):
    if not feats:
        return "_"
    if isinstance(feats, str):
        return clean(feats)
    return "|".join(f"{k}={v}" for k, v in sorted(feats.items()))


def read_conllu(path):
    """Read gold sentences; skip sentences with multiword/empty-node ID gaps,
    missing heads, or != 1 root (identical filtering to the notebook)."""
    sents = []
    with open(path, encoding="utf8") as f:
        for tl in conllu.parse_incr(f):
            toks = [t for t in tl if isinstance(t["id"], int)]
            if not toks or [t["id"] for t in toks] != list(range(1, len(toks) + 1)):
                continue
            heads = [t["head"] for t in toks]
            if any(h is None for h in heads) or sum(h == 0 for h in heads) != 1:
                continue
            sents.append(dict(
                id=tl.metadata.get("sent_id"),
                forms=[clean(t["form"]) for t in toks],
                lemmas=[clean(t.get("lemma")) for t in toks],
                upos=[clean(t.get("upos") or t.get("upostag")) for t in toks],
                xpos=[clean(t.get("xpos") or t.get("xpostag")) for t in toks],
                feats=[format_feats(t.get("feats")) for t in toks],
                heads=heads,
                deprels=[clean(t["deprel"]) for t in toks],
            ))
    return sents


def read_pred_conllu_as_tables(path):
    """Read a predicted CoNLL-U file leniently (missing heads allowed) and return
    a list of (sent_id, n_tokens, table_text) so it can go through parse_table."""
    out = []
    with open(path, encoding="utf8") as f:
        for tl in conllu.parse_incr(f):
            toks = [t for t in tl if isinstance(t["id"], int)]
            lines = []
            for t in toks:
                head = t["head"] if t["head"] is not None else "_"
                lines.append("\t".join([
                    str(t["id"]), clean(t["form"]), clean(t.get("lemma")),
                    clean(t.get("upos") or t.get("upostag")),
                    clean(t.get("xpos") or t.get("xpostag")),
                    format_feats(t.get("feats")), str(head), clean(t["deprel"]),
                ]))
            out.append((tl.metadata.get("sent_id"), len(toks), "\n".join(lines)))
    return out


def fetch_treebank_test(name, data_dir):
    repo, prefix = TREEBANKS[name]
    os.makedirs(data_dir, exist_ok=True)
    path = f"{data_dir}/{prefix}-ud-test.conllu"
    if not (os.path.exists(path) and os.path.getsize(path) > 0):
        url = (f"https://raw.githubusercontent.com/UniversalDependencies/"
               f"{repo}/master/{prefix}-ud-test.conllu")
        urllib.request.urlretrieve(url, path)
    return path


# ----------------------------------------------------------------------------
# Table format, parsing, scoring
# ----------------------------------------------------------------------------
def base(label):
    return label.split(":")[0]


def norm_feats(f):
    if not f or f == "_":
        return "_"
    parts = sorted(p.strip() for p in f.split("|") if p.strip() and p.strip() != "_")
    return "|".join(parts) if parts else "_"


def table_of(s, strip=True):
    """Gold sentence -> 8-column tab-separated table (used for the oracle check)."""
    rows = []
    for i in range(len(s["forms"])):
        rel = (base(s["deprels"][i]) if strip else s["deprels"][i]).lower()
        rows.append(
            f"{i+1}\t{s['forms'][i]}\t{s['lemmas'][i]}\t{s['upos'][i]}\t"
            f"{s['xpos'][i]}\t{s['feats'][i]}\t{s['heads'][i]}\t{rel}"
        )
    return "\n".join(rows)


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


def _func_children(heads, rels, upos, feats):
    n = len(heads)
    fc = [[] for _ in range(n)]
    for j in range(n):
        h = heads[j]
        if h is not None and 1 <= h <= n and rels[j] in FUNC_RELS:
            fc[h - 1].append((rels[j], upos[j], feats[j]))
    return [sorted(x) for x in fc]


def evaluate(s, pred, strip=True):
    """Score one predicted table against one gold sentence. Returns raw counts."""
    n = len(s["forms"])
    p = parse_table(pred, n)
    parsed = sum(h is not None for h in p["heads"])
    f = base if strip else (lambda x: x)

    g_upos = [u.upper() for u in s["upos"]]
    g_xpos = s["xpos"]
    g_feats = [norm_feats(ft) for ft in s["feats"]]
    g_lemma = s["lemmas"]
    g_heads = s["heads"]
    g_rels = [f(d).lower() for d in s["deprels"]]
    p_rels = [f(r).lower() if r is not None else None for r in p["labs"]]

    g_fc = _func_children(g_heads, g_rels, g_upos, g_feats)
    p_fc = (_func_children(p["heads"], p_rels, p["upos"], p["feats"])
            if parsed > 0 else [[] for _ in range(n)])

    c_upos = c_xpos = c_feats = c_alltags = c_lemma = c_head = c_las = 0
    n_content = c_mlas = c_blex = 0

    for j in range(n):
        is_content = g_rels[j] not in NON_CONTENT_RELS
        if is_content:
            n_content += 1
        if parsed == 0 or p["heads"][j] is None:
            continue

        u_ok = p["upos"][j] == g_upos[j]
        x_ok = p["xpos"][j] == g_xpos[j]
        f_ok = p["feats"][j] == g_feats[j]
        l_ok = p["lemmas"][j] == g_lemma[j]
        h_ok = p["heads"][j] == g_heads[j]
        las_ok = h_ok and (p_rels[j] == g_rels[j])

        c_upos += int(u_ok)
        c_xpos += int(x_ok)
        c_feats += int(f_ok)
        c_alltags += int(u_ok and x_ok and f_ok)
        c_lemma += int(l_ok)
        c_head += int(h_ok)
        c_las += int(las_ok)

        if is_content and las_ok:
            if u_ok and f_ok and p_fc[j] == g_fc[j]:
                c_mlas += 1
            if l_ok:
                c_blex += 1

    return dict(
        n_gold=n, n_content=n_content, valid=parsed > 0,
        c_upos=c_upos, c_xpos=c_xpos, c_feats=c_feats, c_alltags=c_alltags,
        c_lemma=c_lemma, c_head=c_head, c_las=c_las, c_mlas=c_mlas, c_blex=c_blex,
    )


def summarize(rs):
    """Pool counts over sentences -> percentage scores."""
    g = sum(r["n_gold"] for r in rs)
    gc = sum(r["n_content"] for r in rs)
    return dict(
        UPOS=100 * sum(r["c_upos"] for r in rs) / g,
        XPOS=100 * sum(r["c_xpos"] for r in rs) / g,
        UFeats=100 * sum(r["c_feats"] for r in rs) / g,
        AllTags=100 * sum(r["c_alltags"] for r in rs) / g,
        Lemmas=100 * sum(r["c_lemma"] for r in rs) / g,
        UAS=100 * sum(r["c_head"] for r in rs) / g,
        LAS=100 * sum(r["c_las"] for r in rs) / g,
        MLAS=100 * sum(r["c_mlas"] for r in rs) / max(gc, 1),
        BLEX=100 * sum(r["c_blex"] for r in rs) / max(gc, 1),
    )


# ----------------------------------------------------------------------------
# Printing helpers
# ----------------------------------------------------------------------------
def print_table(rows, label_cols, title=None):
    """rows: list of dicts with label_cols + METRICS."""
    if title:
        print(f"\n=== {title} ===")
    widths = [max(len(c), *(len(str(r[c])) for r in rows)) for c in label_cols]
    header = "  ".join(c.ljust(w) for c, w in zip(label_cols, widths))
    header += "  " + "  ".join(f"{m:>7s}" for m in METRICS)
    print(header)
    print("-" * len(header))
    for r in rows:
        line = "  ".join(str(r[c]).ljust(w) for c, w in zip(label_cols, widths))
        line += "  " + "  ".join(f"{r[m]:7.2f}" for m in METRICS)
        print(line)


def mean_rows(rows):
    return {m: sum(r[m] for r in rows) / len(rows) for m in METRICS}


# ----------------------------------------------------------------------------
# Modes
# ----------------------------------------------------------------------------
def cmd_files(args):
    gold = read_conllu(args.gold)
    preds = read_pred_conllu_as_tables(args.pred)

    gold_ids_ok = all(s["id"] is not None for s in gold)
    pred_by_id = {sid: (n, t) for sid, n, t in preds if sid is not None}
    if gold_ids_ok and all(s["id"] in pred_by_id for s in gold):
        aligned = [(s, pred_by_id[s["id"]][1]) for s in gold]
    else:
        if len(gold) != len(preds):
            sys.exit(f"Cannot align: {len(gold)} gold sentences vs {len(preds)} predicted "
                     f"and sent_ids do not match.")
        print("sent_ids do not match; aligning by sentence order.", file=sys.stderr)
        aligned = [(s, t) for s, (_, _, t) in zip(gold, preds)]

    res = [evaluate(s, t, not args.keep_subtypes) for s, t in aligned]
    summ = summarize(res)
    print(f"Sentences: {len(res)}  (valid predictions: {sum(r['valid'] for r in res)})")
    print_table([dict(file=os.path.basename(args.pred), **summ)], ["file"])


def cmd_selftest(args):
    sents = read_conllu(args.gold)
    strip = not args.keep_subtypes
    summ = summarize([evaluate(s, table_of(s, strip), strip) for s in sents])
    for m, v in summ.items():
        assert abs(v - 100.0) < 1e-6, (m, v)
    print(f"Oracle check passed on {len(sents)} sentences: all 9 metrics = 100.0")


def cmd_results(args):
    files = sorted(glob.glob(os.path.join(args.results, "*.json")))
    if not files:
        sys.exit(f"No JSON files found in {args.results}")

    strip = not args.keep_subtypes
    gold_cache = {}

    def gold_for(tb):
        if tb not in gold_cache:
            path = fetch_treebank_test(tb, args.data)
            gold_cache[tb] = {s["id"]: s for s in read_conllu(path)}
        return gold_cache[tb]

    runs = []
    for fp in files:
        j = json.load(open(fp, encoding="utf8"))
        tb = j["treebank"]
        gold = gold_for(tb)
        res = []
        for item in j["preds"]:
            if item["id"] not in gold:
                sys.exit(f"{fp}: sent_id {item['id']!r} not found in {tb} test file")
            res.append(evaluate(gold[item["id"]], item["pred"], strip))
        summ = summarize(res)
        runs.append(dict(model=j["model"], N=j["N"], seed=j["seed"], treebank=tb, **summ))

        if args.check_stored and "summary" in j:
            diff = max(abs(summ[m] - j["summary"][m]) for m in METRICS)
            if diff > 1e-6:
                print(f"WARNING {os.path.basename(fp)}: differs from stored summary "
                      f"(max abs diff {diff:.4f})", file=sys.stderr)

    # per (model, N, treebank): mean over seeds
    grp = defaultdict(list)
    for r in runs:
        grp[(r["model"], r["N"], r["treebank"])].append(r)
    per_tb = [dict(model=m, N=N, treebank=tb, **mean_rows(rs))
              for (m, N, tb), rs in sorted(grp.items())]

    # macro over treebanks
    grp2 = defaultdict(list)
    for r in per_tb:
        grp2[(r["model"], r["N"])].append(r)
    macro = [dict(model=m, N=N, treebank="macro-avg", **mean_rows(rs))
             for (m, N), rs in sorted(grp2.items())]

    print_table(macro, ["model", "N"], "Macro-average across treebanks")
    for (m, N), rs in sorted(grp2.items()):
        print_table(rs + [dict(model=m, N=N, treebank="macro-avg", **mean_rows(rs))],
                    ["treebank"], f"Model: {m} | N={N}")

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf8") as f:
            w = csv.DictWriter(f, fieldnames=["model", "N", "treebank"] + METRICS)
            w.writeheader()
            for r in per_tb + macro:
                w.writerow({k: (round(v, 2) if k in METRICS else v) for k, v in r.items()})
        print(f"\nWrote {args.csv}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--keep-subtypes", action="store_true",
                    help="keep UD relation subtypes (default: strip, e.g. nsubj:pass -> nsubj)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("files", help="score a predicted CoNLL-U file against gold")
    p1.add_argument("--gold", required=True)
    p1.add_argument("--pred", required=True)
    p1.set_defaults(fn=cmd_files)

    p2 = sub.add_parser("results", help="re-score notebook results/*.json")
    p2.add_argument("--results", default="results")
    p2.add_argument("--data", default="data/data")
    p2.add_argument("--csv", default=None)
    p2.add_argument("--check-stored", action="store_true",
                    help="warn if re-computed scores differ from the stored summary")
    p2.set_defaults(fn=cmd_results)

    p3 = sub.add_parser("selftest", help="oracle check on a gold file")
    p3.add_argument("--gold", required=True)
    p3.set_defaults(fn=cmd_selftest)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
