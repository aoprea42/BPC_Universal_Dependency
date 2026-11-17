#!/usr/bin/env python3
"""Evaluate bracketed UD-tree predictions (U-DepPLLaMA format) against a gold CoNLL-U file.

Subcommands
  score      gold .conllu + predictions -> UAS / LAS / F1 / invalid rate
  table      all result JSONs in ./results -> table of scores (printed + saved as CSV/Markdown); default command
  aggregate  directory of notebook result JSONs -> mean/std table (+ optional plot)
  selftest   oracle and corruption checks on a gold file

Examples
  python evaluate_ud.py                      # = table over ./results, writes results/scores.csv and scores.md
  python evaluate_ud.py table --results ./results --csv out.csv --md out.md
  python evaluate_ud.py score --gold en_ewt-ud-test.conllu --pred fewshot_ud/run.json
  python evaluate_ud.py score --gold en_ewt-ud-test.conllu --pred trees.txt --max-words 40
  python evaluate_ud.py aggregate --results fewshot_ud --plot las_vs_shots.png
  python evaluate_ud.py selftest --gold en_ewt-ud-test.conllu

Prediction formats
  *.json   notebook output: {"preds": [{"id": <sent_id>, "pred": <tree>}, ...]}  (matched by sent_id)
  other    plain text, one tree per line, in gold order (after --max-words filtering)

Metrics (micro-averaged over words; UD subtypes stripped unless --keep-subtypes):
  UAS = c_head / n_gold,  LAS = c_las / n_gold,  P = c_las / n_sys,  R = c_las / n_gold,  F1 = 2PR/(P+R)
Invalid (unparsable) trees count as all-wrong.
"""
import argparse, glob, json, math, re, sys
from collections import defaultdict

# ------------------------------------------------------------------ data
def clean(form):
    return str(form).replace("[", "{").replace("]", "}").replace(" ", "_")

def read_conllu(path, max_words=None):
    import conllu
    sents = []
    with open(path, encoding="utf8") as f:
        for tl in conllu.parse_incr(f):
            toks = [t for t in tl if isinstance(t["id"], int)]      # drop multiword/empty nodes
            if not toks or [t["id"] for t in toks] != list(range(1, len(toks) + 1)): continue
            heads = [t["head"] for t in toks]
            if any(h is None for h in heads) or sum(h == 0 for h in heads) != 1: continue
            if max_words and len(toks) > max_words: continue
            forms = [clean(t["form"]) for t in toks]
            sents.append(dict(id=tl.metadata.get("sent_id"), forms=forms, heads=heads,
                              deprels=[t["deprel"] for t in toks]))
    return sents

def linearize(s):
    n = len(s["forms"]); kids = [[] for _ in range(n + 1)]
    for i, h in enumerate(s["heads"], 1): kids[h].append(i)
    def rec(i):
        out = ["[" + s["deprels"][i-1].upper()]
        out += [rec(c) for c in kids[i] if c < i]
        out.append("[" + s["forms"][i-1] + "]")
        out += [rec(c) for c in kids[i] if c > i]
        out.append("]")
        return "".join(out)
    return rec(kids[0][0])

# ------------------------------------------------------------------ parsing and scoring
TOK = re.compile(r"\[|\]|[^\[\]]+")

def parse_tree(txt):
    """Bracketed tree -> nodes in in-order order. Raises ValueError/IndexError on malformed input."""
    toks = TOK.findall(txt.strip()); pos = 0
    def group():
        nonlocal pos
        if toks[pos] != "[": raise ValueError("expected [")
        pos += 1
        if toks[pos] in ("[", "]"): raise ValueError("missing label")
        text = toks[pos].strip(); pos += 1
        if not text: raise ValueError("empty text")
        if toks[pos] == "]": pos += 1; return ("w", text)
        items = []
        while toks[pos] != "]": items.append(group())
        pos += 1
        if sum(k == "w" for k, _ in items) != 1: raise ValueError("node needs exactly one word")
        return ("n", dict(label=text, items=items))
    kind, root = group()
    if kind != "n": raise ValueError("root must be a node")
    order = []
    def rec(node, parent):
        node["parent"] = parent
        for k, x in node["items"]:
            if k == "w": node["word"] = x; node["pos"] = len(order); order.append(node)
            else: rec(x, node)
    rec(root, None)
    return order

def base(label): return label.split(":")[0]

def align(gold_forms, sys_forms):
    """LCS of forms first (exact for projective trees), then leftover words matched by nearest position."""
    from difflib import SequenceMatcher
    m = {}
    for a, b, size in SequenceMatcher(None, gold_forms, sys_forms, autojunk=False).get_matching_blocks():
        for t in range(size): m[a + t] = b + t
    used = set(m.values()); left = defaultdict(list)
    for k, f in enumerate(sys_forms):
        if k not in used: left[f].append(k)
    for j, f in enumerate(gold_forms):
        if j not in m and left[f]:
            k = min(left[f], key=lambda k: abs(k - j)); left[f].remove(k); m[j] = k
    return m

def evaluate(s, pred, strip=True):
    n = len(s["forms"])
    r = dict(n_gold=n, n_sys=0, c_head=0, c_las=0, valid=True, recovered=False)
    try: order = parse_tree(pred)
    except Exception: r["valid"] = False; return r
    r["n_sys"] = len(order)
    m = align(s["forms"], [x["word"] for x in order])
    inv = {k: j for j, k in m.items()}
    r["recovered"] = (len(m) < n) or (len(order) != n)
    f = base if strip else (lambda x: x)
    for j in range(n):
        gh, gl = s["heads"][j], f(s["deprels"][j].upper())
        if j not in m:                                   # missing word: attach to root, label unknown
            r["c_head"] += int(gh == 0); continue
        node = order[m[j]]
        sh = 0 if node["parent"] is None else inv.get(node["parent"]["pos"], -1) + 1
        ok = (sh == gh) and sh >= 0
        r["c_head"] += int(ok); r["c_las"] += int(ok and f(node["label"]) == gl)
    return r

def summarize(rs):
    g = sum(r["n_gold"] for r in rs); sy = sum(r["n_sys"] for r in rs)
    h = sum(r["c_head"] for r in rs); l = sum(r["c_las"] for r in rs)
    P, R = (l / sy if sy else 0.0), (l / g if g else 0.0)
    k = max(len(rs), 1)
    return dict(UAS=100 * h / max(g, 1), LAS=100 * l / max(g, 1),
                F1=100 * (2 * P * R / (P + R) if P + R else 0.0),
                invalid=100 * sum(not r["valid"] for r in rs) / k,
                recovered=100 * sum(r["recovered"] for r in rs) / k, n=len(rs))

# ------------------------------------------------------------------ commands
def load_preds(path, gold):
    """Return a list of (gold_sentence, predicted_tree)."""
    if path.endswith(".json"):
        j = json.load(open(path, encoding="utf8")); by_id = {s["id"]: s for s in gold}
        pairs = []
        for p in j["preds"]:
            if p["id"] not in by_id: sys.exit(f"sent_id {p['id']!r} not in gold file")
            pairs.append((by_id[p["id"]], p["pred"] or ""))
        return pairs
    lines = open(path, encoding="utf8").read().split("\n")
    if lines and lines[-1] == "": lines.pop()
    if len(lines) != len(gold):
        sys.exit(f"{len(lines)} prediction lines but {len(gold)} gold sentences "
                 f"(use the same --max-words as when generating)")
    return list(zip(gold, lines))

def cmd_score(a):
    gold = read_conllu(a.gold, a.max_words)
    pairs = load_preds(a.pred, gold)
    res = [evaluate(s, p, not a.keep_subtypes) for s, p in pairs]
    out = summarize(res)
    if a.json: print(json.dumps(out, indent=2)); return
    print(f"sentences : {out['n']}")
    print(f"UAS       : {out['UAS']:.2f}")
    print(f"LAS       : {out['LAS']:.2f}")
    print(f"F1        : {out['F1']:.2f}")
    print(f"invalid   : {out['invalid']:.2f}%")
    print(f"recovered : {out['recovered']:.2f}%  (valid trees needing reorder/missing-word repair)")

def cmd_selftest(a):
    gold = read_conllu(a.gold, a.max_words or 40)[:500]
    o = summarize([evaluate(s, linearize(s)) for s in gold])
    print("oracle:", {k: round(v, 2) for k, v in o.items()})
    assert o["LAS"] > 99.5 and o["invalid"] == 0, "oracle should be ~100"
    assert evaluate(gold[0], "garbage")["valid"] is False
    assert evaluate(gold[0], linearize(gold[0])[:-3])["valid"] is False
    print("selftest OK")

def cmd_table(a):
    import pandas as pd, os
    files = sorted(glob.glob(os.path.join(a.results, "*.json")))
    if not files: sys.exit(f"no result JSONs in {a.results}")
    rows = []
    for f in files:
        j = json.load(open(f, encoding="utf8")); s = j["summary"]
        rows.append(dict(model=j["model"], treebank=j["treebank"], mode=j.get("mode", ""), N=j["N"], seed=j["seed"],
                         UAS=s["UAS"], LAS=s["LAS"], F1=s["F1"], invalid=s["invalid"], recovered=s["recovered"],
                         demos_used=s.get("demos_used", float("nan")), n=s["n"]))
    df = pd.DataFrame(rows).sort_values(["model", "treebank", "N", "seed"]).reset_index(drop=True)
    num = ["UAS", "LAS", "F1", "invalid", "recovered", "demos_used"]
    df[num] = df[num].round(2)
    mean = (df.groupby(["model", "treebank", "mode", "N"])
              .agg(runs=("seed", "count"), UAS=("UAS", "mean"), LAS=("LAS", "mean"), LAS_sd=("LAS", "std"),
                   F1=("F1", "mean"), invalid=("invalid", "mean"), demos_used=("demos_used", "mean"))
              .round(2).reset_index())
    pd.set_option("display.width", 220); pd.set_option("display.max_rows", None); pd.set_option("display.max_columns", None)
    print(f"{len(df)} runs from {a.results}\n\n== per run ==\n{df.to_string(index=False)}")
    print(f"\n== mean over seeds ==\n{mean.to_string(index=False)}")
    csv = a.csv or os.path.join(a.results, "scores.csv"); md = a.md or os.path.join(a.results, "scores.md")
    df.to_csv(csv, index=False); mean.to_csv(csv.replace(".csv", "_mean.csv"), index=False)
    def to_md(d):
        cols = list(d.columns)
        return "\n".join(["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"] +
                          ["| " + " | ".join("" if (isinstance(v, float) and math.isnan(v)) else str(v) for v in r) + " |"
                           for r in d.itertuples(index=False)])
    open(md, "w", encoding="utf8").write("## Per run\n\n" + to_md(df) + "\n\n## Mean over seeds\n\n" + to_md(mean) + "\n")
    print(f"\nsaved {csv}, {csv.replace('.csv', '_mean.csv')}, {md}")

def cmd_aggregate(a):
    import numpy as np, pandas as pd
    rows = []
    for f in glob.glob(f"{a.results}/*.json"):
        j = json.load(open(f))
        rows.append(dict(model=j["model"], treebank=j["treebank"], N=j["N"], seed=j["seed"], **j["summary"]))
    if not rows: sys.exit(f"no result JSONs in {a.results}")
    df = pd.DataFrame(rows)
    agg = df.groupby(["model", "treebank", "N"]).agg(
        LAS=("LAS", "mean"), LAS_sd=("LAS", "std"), UAS=("UAS", "mean"), F1=("F1", "mean"),
        invalid=("invalid", "mean"), recovered=("recovered", "mean")).round(2).reset_index()
    pd.set_option("display.width", 200); pd.set_option("display.max_rows", None)
    for metric in ["LAS", "UAS", "invalid"]:
        print(f"\n== {metric} (mean over seeds) ==")
        print(agg.pivot_table(index=["model", "treebank"], columns="N", values=metric).round(1))
    # ranking: low-resource treebanks weigh more (edit LEVEL for your language set)
    LEVEL = dict(en_ewt="HR", en_gum="HR", de_gsd="HR", de_hdt="HR", af_afribooms="MR", hy_armtdp="MR",
                 fo_farpahc="LR", fo_oft="LR", bar_maibaam="LR")
    W = dict(HR=1.0, MR=1.5, LR=2.0)
    agg["w"] = agg.treebank.map(lambda t: W[LEVEL.get(t, "HR")])
    Nmax = agg.N.max(); rank = {}
    for m, g in agg.groupby("model"):
        top = g[g.N == Nmax]; row = dict(wLAS_Nmax=np.average(top.LAS, weights=top.w), invalid_pct=top.invalid.mean())
        if 0 in g.N.values:
            z = g[g.N == 0]; row["wLAS_0"] = np.average(z.LAS, weights=z.w)
            row["gain_per_shot"] = (row["wLAS_Nmax"] - row["wLAS_0"]) / Nmax
        rank[m] = row
    print(f"\n== ranking (N={Nmax}) =="); print(pd.DataFrame(rank).T.sort_values("wLAS_Nmax", ascending=False).round(2))
    if a.plot:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        tbs = sorted(df.treebank.unique()); ncol = 3; nrow = math.ceil(len(tbs) / ncol)
        fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 3.6 * nrow), squeeze=False)
        for ax, tb in zip(axes.flat, tbs):
            for m, g in agg[agg.treebank == tb].groupby("model"):
                ax.errorbar(g.N, g.LAS, yerr=g.LAS_sd.fillna(0), marker="o", capsize=3, label=m)
            ax.set_title(tb); ax.set_xlabel("N shots"); ax.set_ylabel("LAS"); ax.set_xticks(sorted(df.N.unique()))
        axes.flat[0].legend(fontsize=7)
        for ax in list(axes.flat)[len(tbs):]: ax.axis("off")
        plt.tight_layout(); plt.savefig(a.plot, dpi=150); print("saved", a.plot)

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd")
    s = sp.add_parser("score"); s.add_argument("--gold", required=True); s.add_argument("--pred", required=True)
    s.add_argument("--max-words", type=int, default=None); s.add_argument("--keep-subtypes", action="store_true")
    s.add_argument("--json", action="store_true", help="print metrics as JSON"); s.set_defaults(f=cmd_score)
    tb = sp.add_parser("table"); tb.add_argument("--results", default="./results")
    tb.add_argument("--csv"); tb.add_argument("--md"); tb.set_defaults(f=cmd_table)
    g = sp.add_parser("aggregate"); g.add_argument("--results", required=True); g.add_argument("--plot")
    g.set_defaults(f=cmd_aggregate)
    t = sp.add_parser("selftest"); t.add_argument("--gold", required=True); t.add_argument("--max-words", type=int)
    t.set_defaults(f=cmd_selftest)
    a = ap.parse_args()
    if a.cmd is None:                       # default: table over ./results
        a = ap.parse_args(["table"])
    a.f(a)

if __name__ == "__main__":
    main()
