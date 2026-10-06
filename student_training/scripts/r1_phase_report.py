"""
r1_phase_report.py - performance record of one training phase (no GPU needed; reads the bundle downloaded from the pod).

1. curves:  train loss and validation loss per epoch (+ wrong-video / blank-video gap per epoch) -> curves.png, epoch_metrics.csv
2. text metrics on the generated validation answers (generations.jsonl of each gates folder) -> text_metrics.md/.json,
   per_window_text_metrics.csv. Every metric is also computed for two reference points so it can be read:
     floor    = the same generated answer scored against the correct text of ANOTHER random window of the group
     baseline = the most common training text, written for every window (DADA only)

Metrics (all compare one generated answer with the window's correct text):
  exact event / cause   part before / after ';' identical after lower-casing and removing punctuation (DADA)
  token F1              overlap of words (bag of words), harmonic mean of precision and recall
  ROUGE-L F1            longest common word subsequence (word order matters), F-measure
  BERTScore F1          each word matched to its most similar word in the other text using contextual embeddings
                        (roberta-large, layer 17, rescaled with the published baseline so ~0 = unrelated text)
  embedding cosine      cosine similarity of whole-sentence embeddings (google/embeddinggemma-300m)
  retrieval R@1 / R@5   rank of the correct text among all DIFFERENT correct texts of the group by embedding cosine
                        (chance = 1 / number of different texts)
  list-class accuracy   (DADA) the generated event / cause mapped to the nearest phrase of the fixed DADA list by embedding
                        cosine; correct if it is the window's phrase

    python r1_phase_report.py --run-dir ../../outputs/r1_week1/pod_phase1_2026-10-06 --phase 1
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import random
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
GROUPS = {1: [("gates_phase1_dada", "dada", "A: DADA val, encoder flags (P>=0.5)"),
              ("gates_phase1_dada_lowp", "dada", "B: DADA val, encoder does not flag (P<0.5)"),
              ("gates_phase1_nexar_zeroshot", "nexar", "C: Nexar val, zero-shot")]}
VAL_KEYS = {1: [("val_dada", "A: DADA val P>=0.5"), ("val_dada_lowp", "B: DADA val P<0.5"), ("val_nexar_zeroshot", "C: Nexar val zero-shot")]}


def norm(t):
    return re.sub(r"[^a-z0-9 ]", "", str(t).lower()).strip()


def split_ec(t):
    p = [x.strip() for x in str(t).split(";", 1)]
    return p[0], (p[1] if len(p) > 1 else "")


def token_f1(a, b):
    a, b = norm(a).split(), norm(b).split()
    c = sum((collections.Counter(a) & collections.Counter(b)).values())
    if not a or not b or c == 0:
        return 0.0
    p, r = c / len(a), c / len(b)
    return 2 * p * r / (p + r)


def rouge_l(a, b):
    a, b = norm(a).split(), norm(b).split()
    if not a or not b:
        return 0.0
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a)):
        for j in range(len(b)):
            dp[i + 1][j + 1] = dp[i][j] + 1 if a[i] == b[j] else max(dp[i][j + 1], dp[i + 1][j])
    l = dp[-1][-1]
    if l == 0:
        return 0.0
    p, r = l / len(a), l / len(b)
    return 2 * p * r / (p + r)


# ----------------------------------------------------------------------------- 1. curves
def curves(run, phase, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    log = [json.loads(l) for l in open(run / f"phase{phase}" / "train_log.jsonl", encoding="utf-8")]
    rows = []
    for r in log:
        row = {"epoch": r["epoch"], "step": r["step"], "train_loss": r["train_loss"]}
        for k, _ in VAL_KEYS[phase]:
            v = r.get(k, {})
            for m in ("loss_real", "loss_wrong", "loss_blank", "gap_wrong", "gap_blank"):
                row[f"{k}_{m}"] = v.get(m)
            row[f"{k}_gap_wrong_lo"], row[f"{k}_gap_wrong_hi"] = (v.get("gap_wrong_ci") or [None, None])
        rows.append(row)
    with open(out / "epoch_metrics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    ep = [r["epoch"] for r in rows]
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
    ax[0].plot(ep, [r["train_loss"] for r in rows], "k-o", ms=3, label="train (mean over the epoch's training windows)")
    for (k, name), c in zip(VAL_KEYS[phase][:2], ("tab:blue", "tab:orange")):
        ax[0].plot(ep, [r[f"{k}_loss_real"] for r in rows], "-o", ms=3, color=c, label=f"val {name}")
    best = min(rows, key=lambda r: r[f"{VAL_KEYS[phase][0][0]}_loss_real"])
    ax[0].axvline(best["epoch"], color="tab:blue", ls=":", lw=1)
    ax[0].set(title="Loss per epoch (DADA)", xlabel="epoch", ylabel="loss = mean −ln p(correct word-piece)", yscale="log")
    ax[0].legend(fontsize=7)
    k, name = VAL_KEYS[phase][2]
    ax[1].plot(ep, [r[f"{k}_loss_real"] for r in rows], "-o", ms=3, color="tab:green", label=f"val {name}, own video")
    ax[1].plot(ep, [r[f"{k}_loss_blank"] for r in rows], "--", color="tab:green", alpha=.6, label="same, blank video")
    ax[1].set(title="Loss per epoch (Nexar, never trained on)", xlabel="epoch", ylabel="loss")
    ax[1].legend(fontsize=7)
    for (k, name), c in zip(VAL_KEYS[phase], ("tab:blue", "tab:orange", "tab:green")):
        y = [r[f"{k}_gap_wrong"] for r in rows]
        lo = [r[f"{k}_gap_wrong_lo"] for r in rows]
        hi = [r[f"{k}_gap_wrong_hi"] for r in rows]
        ax[2].plot(ep, y, "-o", ms=3, color=c, label=name)
        ax[2].fill_between(ep, lo, hi, color=c, alpha=.15)
    ax[2].axhline(0, color="k", lw=.8)
    ax[2].set(title="Wrong-video gap per epoch (95% CI band)", xlabel="epoch",
              ylabel="loss(other clip's video) − loss(own video)")
    ax[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "curves.png", dpi=130)
    return rows, best


# ----------------------------------------------------------------------------- 2. text metrics
def text_metrics(run, phase, out, seed=0):
    from bert_score import BERTScorer
    from sentence_transformers import SentenceTransformer
    bs = BERTScorer(model_type="roberta-large", lang="en", rescale_with_baseline=True)
    emb = SentenceTransformer("google/embeddinggemma-300m")
    man = [json.loads(l) for l in open(ROOT / "dataset" / "manifests" / "r1_mmau_dada_windows.jsonl", encoding="utf-8")]
    crash = [r for r in man if r["label"] == 1 and r["valid"]]
    ev_list = sorted({r["gt"]["event"] for r in crash})
    ca_list = sorted({r["gt"]["cause"] for r in crash})
    tr = [r for r in crash if r["split"] == "train"]
    top_txt = collections.Counter(r["targets"]["phase1"] for r in tr).most_common(1)[0][0]
    E = lambda xs: emb.encode(xs, normalize_embeddings=True, batch_size=32)   # noqa: E731
    ev_e, ca_e = E(ev_list), E(ca_list)
    rng = random.Random(seed)
    report, per_rows = {}, []
    for g, src, title in GROUPS[phase]:
        G = [json.loads(l) for l in open(run / g / "generations.jsonl", encoding="utf-8")]
        gen = [x["real"] for x in G]
        tgt = [x["target"] for x in G]
        perm = list(range(len(G)))
        while any(i == j or tgt[i] == tgt[j] for i, j in enumerate(perm)):   # floor: a DIFFERENT window's text
            rng.shuffle(perm)
        shuf = [tgt[j] for j in perm]
        variants = {"model": (gen, tgt), "floor (random other window's text)": (gen, shuf)}
        if src == "dada":
            variants["baseline (always most common train text)"] = ([top_txt] * len(G), tgt)
        res = {}
        for name, (c, r) in variants.items():
            P, R, F = bs.score(c, r)
            ce, re_ = E(c), E(r)
            m = {"token F1": float(np.mean([token_f1(a, b) for a, b in zip(c, r)])),
                 "ROUGE-L F1": float(np.mean([rouge_l(a, b) for a, b in zip(c, r)])),
                 "BERTScore F1 (rescaled)": float(F.mean()),
                 "embedding cosine": float(np.mean(np.sum(ce * re_, 1)))}
            if src == "dada":
                m["exact event"] = float(np.mean([norm(split_ec(a)[0]) == norm(split_ec(b)[0]) for a, b in zip(c, r)]))
                m["exact cause"] = float(np.mean([norm(split_ec(a)[1]) == norm(split_ec(b)[1]) for a, b in zip(c, r)]))
                me = E([split_ec(a)[0] for a in c]) @ ev_e.T
                mc = E([split_ec(a)[1] or a for a in c]) @ ca_e.T
                m["list-class event acc"] = float(np.mean([ev_list[int(me[i].argmax())] == split_ec(r[i])[0] for i in range(len(c))]))
                m["list-class cause acc"] = float(np.mean([ca_list[int(mc[i].argmax())] == split_ec(r[i])[1] for i in range(len(c))]))
            if name == "model":
                uniq = sorted(set(tgt))
                ue = E(uniq)
                sims = E(gen) @ ue.T
                ranks = [int((sims[i] > sims[i, uniq.index(tgt[i])]).sum()) + 1 for i in range(len(G))]
                m["retrieval R@1"] = float(np.mean([k == 1 for k in ranks]))
                m["retrieval R@5"] = float(np.mean([k <= 5 for k in ranks]))
                m["retrieval chance R@1"] = 1 / len(uniq)
                for i, x in enumerate(G):
                    per_rows.append([g, x["id"], x["target"], x["real"], round(token_f1(gen[i], tgt[i]), 3),
                                     round(rouge_l(gen[i], tgt[i]), 3), round(float(F[i]), 3), round(float(np.sum(ce[i] * re_[i])), 3), ranks[i]])
            res[name] = m
        report[title] = {"n": len(G), "different correct texts": len(set(tgt)), **res}
    (out / "text_metrics.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    with open(out / "per_window_text_metrics.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["group", "window_id", "correct_text", "model_output", "token_f1", "rouge_l_f1", "bertscore_f1_rescaled",
                    "embedding_cosine", "retrieval_rank"])
        w.writerows(per_rows)
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--phase", type=int, default=1)
    args = ap.parse_args()
    run = Path(args.run_dir)
    out = run / "report"
    out.mkdir(parents=True, exist_ok=True)
    rows, best = curves(run, args.phase, out)
    rep = text_metrics(run, args.phase, out)
    L = [f"# Phase {args.phase} performance record", "",
         "Files: `curves.png` (loss and wrong-video gap per epoch), `epoch_metrics.csv` (all per-epoch numbers), "
         "`text_metrics.json`, `per_window_text_metrics.csv` (every generated answer with its scores).", "",
         f"Lowest group-A validation loss: epoch {best['epoch']} ({best['val_dada_loss_real']:.3f}); "
         f"last epoch {rows[-1]['epoch']}: {rows[-1]['val_dada_loss_real']:.3f}; training loss last epoch {rows[-1]['train_loss']:.4f}.", "",
         "## Text metrics on the generated answers (50 random windows per group)", "",
         "Each cell = average over the 50 windows. *floor* = the same answers scored against another window's correct text "
         "(what an answer unrelated to the clip gets). *baseline* = writing the most common training text for every window.", ""]
    for title, r in rep.items():
        names = [k for k in r if isinstance(r[k], dict)]
        keys = list(dict.fromkeys(k for n in names for k in r[n]))
        L += [f"### {title}  (n={r['n']}, different correct texts={r['different correct texts']})", "",
              "| metric | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
        for k in keys:
            L.append(f"| {k} | " + " | ".join(f"{r[n][k]:.3f}" if k in r[n] else "" for n in names) + " |")
        L.append("")
    (out / "README.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
