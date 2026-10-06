"""
r1_review_phase2.py - Phase-2 record: review file of ALL validation outputs, curves, verdict calibration, text metrics.

    python r1_review_phase2.py --run-dir ../../outputs/r1_week1/pod_phase2_2026-10-06 [--n-frames 30]
      -> <run-dir>/review/phase2_validation_outputs.md/.csv, frames/<id>.png (sample only)
         <run-dir>/report/curves.png, epoch_metrics.csv, verdict_by_group.md, text_metrics.md/.json

Text metrics (compare the written answer with the correct text, after parsing the 'Collision / Time to impact / Event / Cause'
answer): exact event / cause (DADA), BERTScore F1 (rescaled), token F1, embedding cosine of the EVENT sentence vs the correct event
(DADA phrase / Nexar V12 caption); each also against a floor = the same answers vs a random other window's correct text.
"""
from __future__ import annotations

import argparse
import collections
import csv
import io
import json
import random
import re
import tarfile
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download

from r1_common import parse_phase2
from r1_data_review import R, feature_scores, rows_of, sheet
from r1_phase_report import norm, rouge_l, token_f1

SRC = [("gates_phase2_nexar", "nexar", "Nexar validation (V12 teacher caption as event; cause not annotated)"),
       ("gates_phase2_dada", "dada", "DADA validation (human event / cause)")]


def load(run):
    out = {}
    for g, src, _ in SRC:
        G = [json.loads(l) for l in open(run / g / "generations.jsonl", encoding="utf-8")]
        V = {json.loads(l)["id"]: json.loads(l) for l in open(run / g / "verdicts.jsonl", encoding="utf-8")}
        for x in G:
            x["p_yes"] = V[x["id"]]["p_yes_llm"]
            x["parsed"] = parse_phase2(x["real"])
            x["tgt"] = parse_phase2(x["target"])
        out[g] = G
    return out


def curves(run, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    log = [json.loads(l) for l in open(run / "phase2" / "train_log.jsonl", encoding="utf-8")]
    ep = [r["epoch"] for r in log]
    rows = []
    for r in log:
        row = {"epoch": r["epoch"], "step": r["step"], "train_loss": r["train_loss"]}
        for k in ("val_dada", "val_nexar"):
            v = r[k]
            for m in ("loss_real", "loss_blank", "loss_wrong", "gap_wrong", "gap_blank"):
                row[f"{k}_{m}"] = v[m]
            row[f"{k}_gap_wrong_lo"], row[f"{k}_gap_wrong_hi"] = v["gap_wrong_ci"]
        rows.append(row)
    with open(out / "epoch_metrics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
    ax[0].plot(ep, [r["train_loss"] for r in rows], "k-o", label="train")
    ax[0].plot(ep, [r["val_nexar_loss_real"] for r in rows], "-o", color="tab:green", label="val Nexar (monitored)")
    ax[0].plot(ep, [r["val_dada_loss_real"] for r in rows], "-o", color="tab:blue", label="val DADA")
    ax[0].axvline(3, color="r", ls=":", lw=1)
    ax[0].set(title="Loss per epoch (red dotted = epoch kept)", xlabel="epoch", ylabel="mean −ln p(correct word-piece)")
    ax[0].legend(fontsize=8)
    for k, name, c in (("val_nexar", "Nexar val", "tab:green"), ("val_dada", "DADA val", "tab:blue")):
        y = [r[f"{k}_gap_wrong"] for r in rows]
        ax[1].plot(ep, y, "-o", color=c, label=name)
        ax[1].fill_between(ep, [r[f"{k}_gap_wrong_lo"] for r in rows], [r[f"{k}_gap_wrong_hi"] for r in rows], color=c, alpha=.15)
    ax[1].axhline(0, color="k", lw=.8)
    ax[1].set(title="Wrong-video gap per epoch (95% CI)", xlabel="epoch", ylabel="loss(other clip) − loss(own clip)")
    ax[1].legend(fontsize=8)
    for k, name, c in (("val_nexar", "Nexar val", "tab:green"), ("val_dada", "DADA val", "tab:blue")):
        ax[2].plot(ep, [r[f"{k}_gap_blank"] for r in rows], "-o", color=c, label=name)
    ax[2].set(title="Blank-video gap per epoch", xlabel="epoch", ylabel="loss(blank) − loss(own clip)")
    ax[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "curves.png", dpi=130)


def verdict_groups(data, out):
    from sklearn.metrics import roc_auc_score
    man = {json.loads(l)["id"]: json.loads(l) for l in open(Path(__file__).resolve().parents[2] / "dataset" / "manifests" /
                                                              "r1_mmau_dada_windows.jsonl", encoding="utf-8")}
    L = ["# Phase-2 yes/no (P(yes) read from the answer 'Collision: yes|no'; threshold 0.5) by group", "",
         "`share said yes` = windows with P(yes) ≥ 0.5. `A1 yes` = windows where the frozen crash model has P(collision) ≥ 0.5.", "",
         "| set | group | n | LLM says yes | A1 says yes | LLM AUC | A1 AUC |", "|---|---|---|---|---|---|---|"]
    for g, src, _ in SRC:
        G = data[g]
        groups = [("all", G), ("crash TTE 0.5", [x for x in G if x["label"] == 1 and x["tte"] == 0.5]),
                  ("crash TTE 1.0", [x for x in G if x["label"] == 1 and x["tte"] == 1.0]),
                  ("crash TTE 1.5", [x for x in G if x["label"] == 1 and x["tte"] == 1.5]),
                  ("no-crash", [x for x in G if x["label"] == 0])]
        for name, S in groups:
            if not S:
                continue
            auc = lambda k: (f"{roc_auc_score([x['label'] for x in G], [x[k] for x in G]):.3f}" if name == "all" else "")  # noqa: E731
            L.append(f"| {src} | {name} | {len(S)} | {np.mean([x['p_yes'] >= .5 for x in S]):.0%} | {np.mean([x['a1_p'] >= .5 for x in S]):.0%} "
                     f"| {auc('p_yes')} | {auc('a1_p')} |")
    (out / "verdict_by_group.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    return L


def text_metrics(data, out):
    from bert_score import BERTScorer
    from sentence_transformers import SentenceTransformer
    bs = BERTScorer(model_type="roberta-large", lang="en", rescale_with_baseline=True)
    emb = SentenceTransformer("google/embeddinggemma-300m")
    E = lambda xs: emb.encode(xs, normalize_embeddings=True, batch_size=32)  # noqa: E731
    rng = random.Random(0)
    rep, L = {}, ["# Phase-2 text metrics on the written answers (all validation windows)", "",
                  "Only windows whose correct answer has an event text are scored (crash windows; DADA no-crash windows have the fixed text "
                  "'normal driving before the incident'). Event = the sentence after `Event:`. Floor = the same answers against another "
                  "window's correct event. The Nexar correct event is the V12 teacher caption (a different wording style from "
                  "anything else the model saw), so only the paraphrase-tolerant scores are meaningful there.", ""]
    for g, src, title in SRC:
        G = [x for x in data[g] if x["label"] == 1 and x["tgt"]["event"]]
        gen = [x["parsed"]["event"] or "" for x in G]
        ref = [x["tgt"]["event"] for x in G]
        floor_ref = [rng.choice([j for j in range(len(G)) if norm(ref[j]) != norm(ref[i])]) for i in range(len(G))]
        floor_ref = [ref[j] for j in floor_ref]
        res = {}
        for name, r in (("model", ref), ("floor", floor_ref)):
            P, Rr, F = bs.score(gen, r)
            ce, re_ = E(gen), E(r)
            res[name] = {"BERTScore F1": float(F.mean()), "token F1": float(np.mean([token_f1(a, b) for a, b in zip(gen, r)])),
                         "ROUGE-L F1": float(np.mean([rouge_l(a, b) for a, b in zip(gen, r)])),
                         "embedding cosine": float(np.mean(np.sum(ce * re_, 1)))}
        if src == "dada":
            res["model"]["exact event"] = float(np.mean([norm(a) == norm(b) for a, b in zip(gen, ref)]))
            gc = [x["parsed"]["cause"] or "" for x in G]
            rc = [x["tgt"]["cause"] for x in G]
            res["model"]["exact cause"] = float(np.mean([norm(a) == norm(b) for a, b in zip(gc, rc)]))
            res["model"]["exact event AND cause"] = float(np.mean([norm(a) == norm(b) and norm(c) == norm(d)
                                                                  for a, b, c, d in zip(gen, ref, gc, rc)]))
            top_e = collections.Counter(norm(x) for x in ref).most_common(1)[0]
            top_c = collections.Counter(norm(x) for x in rc).most_common(1)[0]
            res["baseline"] = {"exact event (always the most common event of THIS set)": top_e[1] / len(G),
                               "exact cause (always the most common cause of THIS set)": top_c[1] / len(G)}
        rep[title] = {"n": len(G), **res}
        names = list(res)
        keys = list(dict.fromkeys(k for n in names for k in res[n]))
        L += [f"## {title} (n={len(G)} crash windows)", "", "| metric | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
        for k in keys:
            L.append(f"| {k} | " + " | ".join(f"{res[n][k]:.3f}" if k in res[n] else "" for n in names) + " |")
        L.append("")
    (out / "text_metrics.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    (out / "text_metrics.md").write_text("\n".join(L), encoding="utf-8")
    return L


def review(data, run, out, n_frames):
    scores = feature_scores()
    rows = {s: {r["window_id"]: r for r in rows_of(s)} for s in ("dada", "nexar")}
    rng = random.Random(0)
    chosen = {}
    for g, src, _ in SRC:
        pos = [x["id"] for x in data[g] if x["label"] == 1]
        neg = [x["id"] for x in data[g] if x["label"] == 0]
        chosen[g] = set(rng.sample(pos, min(n_frames // 2, len(pos))) + rng.sample(neg, min(n_frames // 2, len(neg))))
    (out / "frames").mkdir(parents=True, exist_ok=True)
    for g, src, _ in SRC:
        need = collections.defaultdict(set)
        for w in chosen[g]:
            need[rows[src][w]["shard"]].add(w)
        for shard, ids in need.items():
            with tarfile.open(hf_hub_download(R[src], shard, repo_type="dataset")) as tf:
                for m in tf:
                    w = m.name[:-4]
                    if w in ids and not (out / "frames" / f"{w}.png").exists():
                        fr = np.load(io.BytesIO(tf.extractfile(m).read()))["frames"]
                        sheet(rows[src][w], fr, scores.get(w), out / "frames" / f"{w}.png")
    md = ["# Phase 2 — model outputs on ALL validation windows", "",
          "How to read it:",
          "- **Question** asked for every window: \"Is a collision coming? Give: collision yes/no, time to impact, event, cause.\"",
          "- **Correct** = the ground-truth answer we trained toward (DADA: human event and cause; Nexar: V12 teacher caption as event, "
          "'Cause: not annotated'). **Model** = what the model wrote after seeing the window (greedy decoding).",
          "- **P(yes)** = the model's own probability of answering 'yes' to the collision question (0–1), read from the first word of its answer; "
          "**A1 P** = the frozen crash model's probability for the same window.",
          "- ✓/✗ columns: verdict = the model's written yes/no equals the true label; event/cause = word-for-word identical (DADA only).",
          "- Windows with a frame-grid link were drawn at random (seed 0) for visual checking: 15 crash + 15 no-crash per dataset.", ""]
    csv_rows = []
    for g, src, title in SRC:
        G = data[g]
        md += [f"## {title}", ""]
        md += ["| # | window | label | TTE | A1 P | P(yes) | model said | verdict | correct answer | model answer |", "|---|---|---|---|---|---|---|---|---|---|"]
        for k, x in enumerate(G, 1):
            pv = x["parsed"]["collision"]
            ok = "✓" if pv == x["label"] else "✗"
            link = f"[{x['id']}](frames/{x['id']}.png)" if x["id"] in chosen[g] else x["id"]
            md.append(f"| {k} | {link} | {'crash' if x['label'] else 'no-crash'} | {x['tte']} | {x['a1_p']:.2f} | {x['p_yes']:.2f} | "
                      f"{ {1: 'yes', 0: 'no', None: '?'}[pv] } | {ok} | {x['target']} | {x['real']} |")
            csv_rows.append([g, x["id"], x["label"], x["tte"], round(x["a1_p"], 4), x["p_yes"], pv, x["target"], x["real"]])
        md.append("")
    (out / "phase2_validation_outputs.md").write_text("\n".join(md), encoding="utf-8")
    with open(out / "phase2_validation_outputs.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["set", "window_id", "label", "tte", "a1_p_collision", "llm_p_yes", "llm_written_verdict", "correct_answer", "model_answer"])
        w.writerows(csv_rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--n-frames", type=int, default=30)
    args = ap.parse_args()
    run = Path(args.run_dir)
    (run / "report").mkdir(exist_ok=True)
    (run / "review").mkdir(exist_ok=True)
    data = load(run)
    curves(run, run / "report")
    print("\n".join(verdict_groups(data, run / "report")))
    print("\n".join(text_metrics(data, run / "report")))
    review(data, run, run / "review", args.n_frames)
    print("review written")


if __name__ == "__main__":
    main()
