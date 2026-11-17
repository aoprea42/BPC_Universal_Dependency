#!/usr/bin/env python3
"""
Few-shot Universal Dependency parsing with LLMs, local version for an AMD RX 7900 XTX (24 GB, gfx1100, ROCm).

Setup (Linux, ROCm installed; check pytorch.org for the current ROCm wheel index):
    python -m venv .venv && source .venv/bin/activate
    pip install torch --index-url https://download.pytorch.org/whl/rocm6.4
    pip install conllu "transformers>=4.44" accelerate pandas matplotlib tqdm huggingface_hub
    export HF_TOKEN=hf_...            # only needed for gated models (Llama, Aya)

Usage:
    python fewshot_ud_parsing.py --smoke                      # tiny sanity run (en_ewt, N=10)
    python fewshot_ud_parsing.py --models llama-3.1           # full grid
    python fewshot_ud_parsing.py --models llama-3.1 aya-expanse-8b --treebanks en_ewt de_gsd
    python fewshot_ud_parsing.py --report-only                # aggregate cached results, plot, rank
    python fewshot_ud_parsing.py --quant 4bit                 # optional bitsandbytes NF4 (ROCm support is experimental)

Memory: an 8B model in bf16 needs about 8e9 * 2 B = 16 GB, which fits in 24 GB with room for the KV cache
at batch_size=8. 4-bit quantization is therefore off by default.
"""
import os
# Must be set before torch is imported. Reduces fragmentation on long, variable-length prompts.
os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF", "expandable_segments:True")

import re, sys, json, random, math, glob, time, gc, argparse, urllib.request
from collections import Counter, defaultdict
from difflib import SequenceMatcher

import numpy as np
import pandas as pd
import torch
import conllu

# ---------------------------------------------------------------- CONFIG
CFG = dict(
    models={
        "Qwen3.5-9B":     "Qwen/Qwen3.5-9B-Base",
        "llama-3.1":      r"C:\Users\alexa\Desktop\TUM\NLP\llama-3.1-8b",#"meta-llama/Llama-3.1-8B",
        "aya-expanse-8b": "CohereLabs/aya-expanse-8b",
        "bolmo-7b":       "allenai/Bolmo-7B",
    },
    run_models=["llama-3.1"],
    treebanks=["en_ewt", "fo_oft"],
    shots=[0, 3, 5, 10],
    seeds=[0],                 # demonstration seeds (0-shot is run once)
    mode="gold",                     # "gold": gold tokenization | "raw": raw text input
    n_eval=200,
    max_eval_words=40,
    max_demo_words=25,
    batch_size=8,
    demo_source={},                  # ablation, e.g. {"bar_maibaam": "de_gsd"}
    held_out_demos=50,
    strip_subtypes=True,
    repetition_penalty=1.15,         # as in the notebook; set to 1.0 to disable (see note in generate())
    quant="none",                    # "none" (bf16/fp16) | "4bit" (bitsandbytes)
    out_dir="./results/unconstrained",
    data_dir="./ud_data",
)

LEVEL = dict(en_ewt="HR", en_gum="HR", de_gsd="HR", de_hdt="HR",
             af_afribooms="MR", hy_armtdp="MR", fo_farpahc="LR", fo_oft="LR", bar_maibaam="LR")
WEIGHT = dict(HR=1.0, MR=1.5, LR=2.0)

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
}

# Reference LAS: (UDPipe 2.0++, U-DepPLLaMA 7B, U-DepPLLaMA 13B), UD 2.3
REF = {"en_ewt": (90.40, 89.91, 90.05), "en_gum": (88.80, 87.92, 87.47), "de_gsd": (84.06, 84.16, 84.14)}


# ---------------------------------------------------------------- DATA
def fetch(repo, prefix, split):
    path = f"{CFG['data_dir']}/{prefix}-ud-{split}.conllu"
    if not (os.path.exists(path) and os.path.getsize(path) > 0):
        url = f"https://raw.githubusercontent.com/UniversalDependencies/{repo}/master/{prefix}-ud-{split}.conllu"
        try:
            urllib.request.urlretrieve(url, path)
        except Exception:
            if os.path.exists(path):
                os.remove(path)
            return None
    return path


def clean(form):
    # brackets would break the tree format; spaces break word alignment
    return str(form).replace("[", "{").replace("]", "}").replace(" ", "_")


def read_conllu(path):
    sents = []
    with open(path, encoding="utf8") as f:
        for tl in conllu.parse_incr(f):
            toks = [t for t in tl if isinstance(t["id"], int)]      # drop multiword/empty nodes
            if not toks or [t["id"] for t in toks] != list(range(1, len(toks) + 1)):
                continue
            heads = [t["head"] for t in toks]
            if any(h is None for h in heads) or sum(h == 0 for h in heads) != 1:
                continue
            forms = [clean(t["form"]) for t in toks]
            sents.append(dict(id=tl.metadata.get("sent_id"),
                              text=tl.metadata.get("text", " ".join(forms)),
                              forms=forms, heads=heads,
                              deprels=[t["deprel"] for t in toks]))
    return sents


DATA = {}


def load_treebank(name):
    repo, prefix = TREEBANKS[name]
    paths = {s: fetch(repo, prefix, s) for s in ["train", "dev", "test"]}
    test = read_conllu(paths["test"])
    if paths["train"]:
        pool, src = read_conllu(paths["train"]), "train"
    elif paths["dev"]:
        pool, src = read_conllu(paths["dev"]), "dev"
    else:                # no train/dev: carve a disjoint demo pool out of test
        rng = random.Random(123); rng.shuffle(test)
        k = CFG["held_out_demos"]; pool, test, src = test[:k], test[k:], "test-heldout"
    pool = [s for s in pool if len(s["forms"]) <= CFG["max_demo_words"]]
    ev = [s for s in test if len(s["forms"]) <= CFG["max_eval_words"]]
    random.Random(123).shuffle(ev); ev = ev[:CFG["n_eval"]]
    DATA[name] = dict(pool=pool, eval=ev, pool_src=src)
    print(f"{name:14s} pool={len(pool):6d} ({src})  eval={len(ev)}")


# ---------------------------------------------------------------- LINEARIZATION / PARSING / SCORING
def linearize(s):
    # U-DepPLLaMA bracketed format: node = [DEPREL <left children> [word] <right children>]
    n = len(s["forms"]); kids = [[] for _ in range(n + 1)]
    for i, h in enumerate(s["heads"], 1):
        kids[h].append(i)

    def rec(i):
        out = ["[" + s["deprels"][i - 1].upper()]
        out += [rec(c) for c in kids[i] if c < i]
        out.append("[" + s["forms"][i - 1] + "]")
        out += [rec(c) for c in kids[i] if c > i]
        out.append("]")
        return "".join(out)
    return rec(kids[0][0])


TOKRE = re.compile(r"\[|\]|[^\[\]]+")


def parse_tree(txt):
    # returns nodes in in-order (surface-like) order; raises ValueError on malformed trees
    toks = TOKRE.findall(txt.strip()); pos = 0

    def group():
        nonlocal pos
        if toks[pos] != "[":
            raise ValueError("expected [")
        pos += 1
        if toks[pos] in ("[", "]"):
            raise ValueError("missing label")
        text = toks[pos].strip(); pos += 1
        if not text:
            raise ValueError("empty text")
        if toks[pos] == "]":
            pos += 1; return ("w", text)          # word leaf
        items = []
        while toks[pos] != "]":
            items.append(group())
        pos += 1
        if sum(k == "w" for k, _ in items) != 1:
            raise ValueError("node needs exactly one word")
        return ("n", dict(label=text, items=items))
    kind, root = group()
    if kind != "n":
        raise ValueError("root must be a node")
    order = []

    def rec(node, parent):
        node["parent"] = parent
        for k, x in node["items"]:
            if k == "w":
                node["word"] = x; node["pos"] = len(order); order.append(node)
            else:
                rec(x, node)
    rec(root, None)
    return order


def base(label):
    return label.split(":")[0]


def align(gold_forms, sys_forms):
    # 1) longest common subsequence of forms (handles projective trees and repeated words exactly)
    # 2) leftover words (subtrees moved by non-projective linearization) are matched in order per form
    m = {}
    for a, b, size in SequenceMatcher(None, gold_forms, sys_forms, autojunk=False).get_matching_blocks():
        for t in range(size):
            m[a + t] = b + t
    used = set(m.values()); left = defaultdict(list)
    for k, f in enumerate(sys_forms):
        if k not in used:
            left[f].append(k)
    for j, f in enumerate(gold_forms):
        if j not in m and left[f]:
            k = min(left[f], key=lambda k: abs(k - j)); left[f].remove(k); m[j] = k
    return m


def evaluate(s, pred, strip=True):
    n = len(s["forms"])
    r = dict(n_gold=n, n_sys=0, c_head=0, c_las=0, valid=True, recovered=False)
    try:
        order = parse_tree(pred)
    except Exception:
        r["valid"] = False; return r        # invalid => all wrong
    r["n_sys"] = len(order)
    m = align(s["forms"], [x["word"] for x in order])
    inv = {k: j for j, k in m.items()}
    r["recovered"] = (len(m) < n) or (len(order) != n)
    f = base if strip else (lambda x: x)
    for j in range(n):
        gh, gl = s["heads"][j], f(s["deprels"][j].upper())
        if j not in m:                                     # missing word: attach to root, label unknown
            r["c_head"] += int(gh == 0); continue
        node = order[m[j]]
        sh = 0 if node["parent"] is None else inv.get(node["parent"]["pos"], -1) + 1
        ok = (sh == gh) and sh >= 0
        r["c_head"] += int(ok); r["c_las"] += int(ok and f(node["label"]) == gl)
    return r


def summarize(rs):
    g = sum(r["n_gold"] for r in rs); sy = sum(r["n_sys"] for r in rs)
    h = sum(r["c_head"] for r in rs); l = sum(r["c_las"] for r in rs)
    P, R = (l / sy if sy else 0.0), l / g
    return dict(UAS=100 * h / g, LAS=100 * l / g, F1=100 * (2 * P * R / (P + R) if P + R else 0.0),
                invalid=100 * np.mean([not r["valid"] for r in rs]),
                recovered=100 * np.mean([r["recovered"] for r in rs]), n=len(rs))


def run_unit_tests():
    """No GPU needed: the oracle must score 100, corruptions must not."""
    s = DATA[CFG["treebanks"][0]]["eval"][0]
    print(s["text"]); print(linearize(s))
    for tb in CFG["treebanks"]:
        las = summarize([evaluate(x, linearize(x)) for x in DATA[tb]["eval"]])["LAS"]
        print(f"  oracle {tb:14s} LAS = {las:.2f}")
    assert evaluate(s, "garbage")["valid"] is False
    assert evaluate(s, linearize(s)[:-3])["valid"] is False              # truncated -> invalid


def trivial_baseline(name):
    """Each word attaches to its left neighbour; label = most frequent relation in the demo pool."""
    d = DATA[name]
    lab = Counter(base(r.upper()) for s in d["pool"] for r in s["deprels"]).most_common(1)[0][0]
    h = l = g = 0
    for s in d["eval"]:
        for j, (gh, gr) in enumerate(zip(s["heads"], s["deprels"])):
            ok = gh == j; h += ok; l += ok and base(gr.upper()) == lab; g += 1
    return dict(UAS=100 * h / g, LAS=100 * l / g)


# ---------------------------------------------------------------- PROMPTS
INSTR = ("Parse each sentence into a bracketed Universal Dependencies tree.\n"
         "Format: [DEPREL ...]. Every node starts with its dependency relation (the root node is ROOT). "
         "The node's own word appears in its own brackets, placed between its left children and its right children, "
         "e.g. [ROOT[NSUBJ[He]][slept]]. Copy every word of the sentence exactly once, in tree order.\n\n")


def sent_line(s, mode):
    return " ".join(s["forms"]) if mode == "gold" else s["text"]


def build_prompt(demos, s, mode):
    p = INSTR
    for d in demos:
        p += f"Sentence: {sent_line(d, mode)}\nTree: {linearize(d)}\n\n"
    return p + f"Sentence: {sent_line(s, mode)}\nTree:"


def get_demos(tb_eval, N, seed):
    # nested sets: demos(3) subset of demos(5) subset of demos(10) for a given seed; independent of the model
    src = CFG["demo_source"].get(tb_eval, tb_eval)
    pool = list(DATA[src]["pool"]); random.Random(f"{src}-{seed}").shuffle(pool)
    return pool[:N]


# ---------------------------------------------------------------- MODEL / GENERATION
MODEL = TOK_ = None


def free_gpu():
    global MODEL, TOK_
    MODEL = TOK_ = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_model(hf_id):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    global MODEL, TOK_
    free_gpu()
    dt = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    tok = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    kw = dict(trust_remote_code=True, device_map={"": 0}, attn_implementation="sdpa")
    if CFG["quant"] == "4bit":
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dt)
    else:
        kw["torch_dtype"] = dt
    try:
        model = AutoModelForCausalLM.from_pretrained(hf_id, **kw)
    except (ValueError, ImportError) as e:        # some custom-code models do not accept attn_implementation
        print("retrying without attn_implementation:", e)
        kw.pop("attn_implementation"); model = AutoModelForCausalLM.from_pretrained(hf_id, **kw)
    model.eval(); MODEL, TOK_ = model, tok
    ctx = getattr(model.config, "max_position_embeddings", 4096)
    print(f"{hf_id} loaded ({CFG['quant']}, {dt}); context = {ctx}")
    return ctx


@torch.no_grad()
def generate(prompts, max_new):
    enc = TOK_(prompts, return_tensors="pt", padding=True).to(MODEL.device)
    # stop as soon as the model emits a blank line (end of the tree) or EOS
    stop_ids = set(TOK_.encode("\n\n", add_special_tokens=False))
    if TOK_.eos_token_id is not None:
        stop_ids.add(TOK_.eos_token_id)
    # NOTE: repetition_penalty > 1 also penalizes legitimate copying of words/brackets from the prompt.
    # It is kept at the notebook's value for comparability; try 1.0 if outputs look degraded.
    out = MODEL.generate(**enc, max_new_tokens=max_new, do_sample=False,
                         pad_token_id=TOK_.pad_token_id,
                         repetition_penalty=CFG["repetition_penalty"],
                         eos_token_id=sorted(stop_ids))
    texts = TOK_.batch_decode(out[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
    return [t.lstrip("\n").split("\n")[0].strip() for t in texts]


def fit_demos(demos, s, ctx, reserve):
    # drop trailing demos if the prompt would overflow the context (recorded as demos_used)
    while True:
        p = build_prompt(demos, s, CFG["mode"])
        if len(TOK_(p).input_ids) + reserve <= ctx or not demos:
            return demos, p
        demos = demos[:-1]


def run_batch(batch, prompts):
    reserve = max(len(TOK_(linearize(s)).input_ids) for s in batch)
    max_new = min(int(reserve * 1.4) + 16, 512)
    try:
        return generate(prompts, max_new)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return [generate([p], max_new)[0] for p in prompts]


# ---------------------------------------------------------------- EXPERIMENT LOOP (resumable)
def run_key(m, tb, N, seed):
    return f"{m}__{tb}__{CFG['mode']}__{CFG['demo_source'].get(tb, tb)}__N{N}__s{seed}"


def run_all():
    from tqdm import tqdm
    out = CFG["out_dir"]
    for m in CFG["run_models"]:
        todo = [(tb, N, sd) for tb in CFG["treebanks"] for N in CFG["shots"]
                for sd in (CFG["seeds"] if N > 0 else [0])
                if not os.path.exists(f"{out}/{run_key(m, tb, N, sd)}.json")]
        if not todo:
            print(m, ": all done"); continue
        ctx = load_model(CFG["models"][m])
        for tb, N, sd in todo:
            t0 = time.time(); ev = DATA[tb]["eval"]; demos0 = get_demos(tb, N, sd)
            items = []
            for s in ev:
                d, p = fit_demos(demos0, s, ctx, reserve=int(len(TOK_(linearize(s)).input_ids) * 1.4) + 16)
                items.append((s, p, len(d)))
            order = sorted(range(len(items)), key=lambda i: len(items[i][1]))
            preds = [None] * len(items)

            print(f"\nProcessing: {m} on {tb} (N={N}, seed={sd})")
            for b in tqdm(range(0, len(order), CFG["batch_size"]), desc="batches"):
                idx = order[b:b + CFG["batch_size"]]
                outs = run_batch([items[i][0] for i in idx], [items[i][1] for i in idx])
                for i, o in zip(idx, outs):
                    preds[i] = o

            res = [evaluate(s, pr, CFG["strip_subtypes"]) for (s, _, _), pr in zip(items, preds)]
            summ = summarize(res); summ["demos_used"] = float(np.mean([k for _, _, k in items]))
            with open(f"{out}/{run_key(m, tb, N, sd)}.json", "w") as f:
                json.dump(dict(model=m, treebank=tb, N=N, seed=sd, mode=CFG["mode"], summary=summ,
                               preds=[dict(id=s["id"], pred=pr) for (s, _, _), pr in zip(items, preds)]), f)
            print(f"Result: {m} {tb:13s} N={N:2d} s={sd}  LAS={summ['LAS']:5.1f} UAS={summ['UAS']:5.1f} "
                  f"inv={summ['invalid']:4.1f}% demos={summ['demos_used']:.1f}  {time.time() - t0:.0f}s")
        free_gpu()


# ---------------------------------------------------------------- REPORT
def report():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = CFG["out_dir"]
    rows = []
    for f in glob.glob(f"{out}/*.json"):
        j = json.load(open(f))
        rows.append(dict(model=j["model"], treebank=j["treebank"], N=j["N"], seed=j["seed"], **j["summary"]))
    if not rows:
        print("no result files in", out); return
    df = pd.DataFrame(rows)
    agg = df.groupby(["model", "treebank", "N"]).agg(
        LAS=("LAS", "mean"), LAS_sd=("LAS", "std"), UAS=("UAS", "mean"), F1=("F1", "mean"),
        invalid=("invalid", "mean"), recovered=("recovered", "mean"),
        demos_used=("demos_used", "mean")).round(2).reset_index()
    agg.to_csv(f"{out}/aggregate.csv", index=False)
    pd.set_option("display.width", 200)

    for metric in ["LAS", "UAS", "invalid"]:
        print(f"\n== {metric} (mean over seeds) ==")
        print(agg.pivot_table(index=["model", "treebank"], columns="N", values=metric).round(1))

    tbs = sorted(df.treebank.unique())
    for tb in tbs:
        if tb not in DATA:
            load_treebank(tb)
    trivial = {tb: trivial_baseline(tb) for tb in tbs}

    ncol = 3; nrow = math.ceil(len(tbs) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 3.6 * nrow), squeeze=False)
    for ax, tb in zip(axes.flat, tbs):
        for m, g in agg[agg.treebank == tb].groupby("model"):
            ax.errorbar(g.N, g.LAS, yerr=g.LAS_sd.fillna(0), marker="o", capsize=3, label=m)
        ax.axhline(trivial[tb]["LAS"], ls=":", c="gray", label="trivial")
        if tb in REF:
            ax.axhline(REF[tb][0], ls="--", c="k", lw=.8, label="UDPipe2.0++")
            ax.axhline(REF[tb][1], ls="--", c="r", lw=.8, label="U-DepPLLaMA 7B")
        ax.set_title(f"{tb} ({LEVEL[tb]})"); ax.set_xlabel("N shots"); ax.set_ylabel("LAS")
        ax.set_xticks(sorted(df.N.unique()))
    axes.flat[0].legend(fontsize=7)
    for ax in list(axes.flat)[len(tbs):]:
        ax.axis("off")
    plt.tight_layout(); plt.savefig(f"{out}/las_vs_shots.png", dpi=150)
    print("saved", f"{out}/las_vs_shots.png")

    # Decision rule: weighted mean LAS over treebanks (LR languages weigh more)
    Nmax = agg.N.max()
    best = agg[agg.N == Nmax].copy(); best["w"] = best.treebank.map(LEVEL).map(WEIGHT)
    rank = pd.DataFrame({
        "wLAS@Nmax": {m: np.average(g.LAS, weights=g.w) for m, g in best.groupby("model")},
        "wUAS@Nmax": {m: np.average(g.UAS, weights=g.w) for m, g in best.groupby("model")},
        "invalid%":  {m: g.invalid.mean() for m, g in best.groupby("model")}})
    if 0 in agg.N.values:
        zero = agg[agg.N == 0].copy(); zero["w"] = zero.treebank.map(LEVEL).map(WEIGHT)
        rank["wLAS@0"] = pd.Series({m: np.average(g.LAS, weights=g.w) for m, g in zero.groupby("model")})
        rank["gain/shot"] = (rank["wLAS@Nmax"] - rank["wLAS@0"]) / Nmax
    print("\n== Ranking ==")
    print(rank.sort_values("wLAS@Nmax", ascending=False).round(2))


# ---------------------------------------------------------------- MAIN
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", help="keys of CFG['models']")
    ap.add_argument("--treebanks", nargs="+")
    ap.add_argument("--shots", nargs="+", type=int)
    ap.add_argument("--seeds", nargs="+", type=int)
    ap.add_argument("--mode", choices=["gold", "raw"])
    ap.add_argument("--n-eval", type=int)
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--quant", choices=["none", "4bit"])
    ap.add_argument("--rep-penalty", type=float)
    ap.add_argument("--out-dir")
    ap.add_argument("--data-dir")
    ap.add_argument("--smoke", action="store_true", help="en_ewt, N=10, seed 0, results in ./smoke")
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--skip-tests", action="store_true")
    a = ap.parse_args()

    if a.models:      CFG["run_models"] = a.models
    if a.treebanks:   CFG["treebanks"] = a.treebanks
    if a.shots:       CFG["shots"] = a.shots
    if a.seeds:       CFG["seeds"] = a.seeds
    if a.mode:        CFG["mode"] = a.mode
    if a.n_eval:      CFG["n_eval"] = a.n_eval
    if a.batch_size:  CFG["batch_size"] = a.batch_size
    if a.quant:       CFG["quant"] = a.quant
    if a.rep_penalty is not None: CFG["repetition_penalty"] = a.rep_penalty
    if a.out_dir:     CFG["out_dir"] = a.out_dir
    if a.data_dir:    CFG["data_dir"] = a.data_dir
    if a.smoke:
        CFG.update(treebanks=["en_ewt"], shots=[10], seeds=[0])
        if not a.out_dir:
            CFG["out_dir"] = "./smoke"
    for m in CFG["run_models"]:
        if m not in CFG["models"]:
            sys.exit(f"unknown model '{m}'; choices: {list(CFG['models'])}")

    os.makedirs(CFG["data_dir"], exist_ok=True); os.makedirs(CFG["out_dir"], exist_ok=True)

    if a.report_only:
        report(); return

    for tb in sorted(set(CFG["treebanks"]) | set(CFG["demo_source"].values())):
        load_treebank(tb)
    if not a.skip_tests:
        run_unit_tests()

    if not torch.cuda.is_available():
        sys.exit("No GPU visible to PyTorch. Check the ROCm install: python -c 'import torch; print(torch.version.hip)'")
    free, total = torch.cuda.mem_get_info()
    print(f"device: {torch.cuda.get_device_name(0)} | HIP {torch.version.hip} | free {free / 1e9:.1f} / {total / 1e9:.1f} GB")

    run_all()
    report()


if __name__ == "__main__":
    main()