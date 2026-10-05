"""
r1_eval_gates.py - the week-1 go/no-go gates on held-out windows (plan 2026-10-05_Plan-Week1-GoNoGo-rev3, section 5).

Run on one checkpoint (merger [+ LoRA]) and one validation set:
  GATE  wrong-video gap     loss(true text | another clip's tokens) - loss(true text | own tokens); bootstrap CI > 0
  GATE  blank-video gap     same with all-zero V-JEPA features (the merger's 'no information' output); CI > 0
  GATE  retrieval@1 / 10    rank the true clip among 10 candidates by the text's likelihood (chance 10%)
  GATE  diversity           distinct generated answers out of N windows (>= 90% distinct)
  GATE  facts vs blank      verdict acc, time-to-impact acc, event/cause match, ArA 5-option cause choice:
                            real tokens must beat the blank-video run
  report verdict agreement with the BADAS score (faithfulness preview) and caption-overlap metrics.

Validation sets (user decision 2026-10-05): phase-1 val = DADA val VALID CRASH (TP) windows only. The no-crash
hallucination check is deferred to the next session.

  python r1_eval_gates.py --phase 1 --ckpt ../../outputs/r1_week1/phase1/best.pt --source dada --cache-dir ... --out-dir ...
  python r1_eval_gates.py --phase 2 --ckpt ../../outputs/r1_week1/phase2/best.pt --source nexar ...
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "models"))
from r1_bridge import R1Bridge  # noqa: E402
from r1_common import parse_phase2  # noqa: E402
from r1_data import Cache, feats_for, load_items, make_batch  # noqa: E402
from r1_train import QWEN_ID, bootstrap_ci, evaluate, load_lm  # noqa: E402


def norm(t):
    return re.sub(r"[^a-z0-9 ]", "", str(t).lower()).strip()


def token_f1(a, b):
    a, b = norm(a).split(), norm(b).split()
    if not a or not b:
        return 0.0
    c = sum((Counter(a) & Counter(b)).values())
    if c == 0:
        return 0.0
    p, r = c / len(a), c / len(b)
    return 2 * p * r / (p + r)


def distinct_n(texts, n):
    grams = []
    for t in texts:
        w = norm(t).split()
        grams += [tuple(w[i:i + n]) for i in range(len(w) - n + 1)]
    return len(set(grams)) / max(1, len(grams))


@torch.no_grad()
def retrieval_at10(bridge, items, caches, dev, n_cand=10, seed=0, bs=10):
    """For window i: loss of ITS answer given own tokens vs 9 other windows' tokens (different video, different
    answer text so identical list-sentences cannot tie). Returns top-1 rate and chance."""
    rng = random.Random(seed)
    hits, n = 0, 0
    for i, it in enumerate(items):
        pool = [j for j, o in enumerate(items) if o["video_key"] != it["video_key"] and o["answer"] != it["answer"]]
        if len(pool) < n_cand - 1:
            continue
        cand = [i] + rng.sample(pool, n_cand - 1)
        feats = feats_for([items[j] for j in cand], caches).to(dev)
        batch = [t.to(dev) for t in make_batch(bridge.prompts, [it] * len(cand))]
        loss = bridge.loss_per_sample(feats, batch).float().cpu()
        hits += int(int(loss.argmin()) == 0)
        n += 1
    return {"top1": hits / max(1, n), "n": n, "chance": 1 / n_cand}


@torch.no_grad()
def ara_choice(bridge, items, caches, dev, phase, mode="real"):
    """DADA only: the official ArA question with 5 options. Each option is scored by the likelihood of the
    target text with that option as the cause; choose the lowest loss. chance = 20%."""
    ok, n = 0, 0
    for it in items:
        ara = it["gt"].get("ara") if it["source"] == "dada" else None
        if not ara or it["label"] != 1:
            continue
        feats = feats_for([it], caches).to(dev).expand(5, -1, -1)
        cands = []
        for opt in ara["options"]:
            c = dict(it)
            if phase == 1:
                c["answer"] = f"{it['gt']['event']}; {opt}"
            else:
                c["answer"] = f"Collision: yes. Time to impact: about {it['tte']:.1f} s. Event: {it['gt']['event']}. Cause: {opt}."
            cands.append(c)
        batch = [t.to(dev) for t in make_batch(bridge.prompts, cands)]
        loss = bridge.loss_per_sample(feats, batch, mode=mode).float().cpu()
        ok += int(int(loss.argmin()) == ara["answer"])
        n += 1
    return {"acc": ok / max(1, n), "n": n, "chance": 0.2}


@torch.no_grad()
def generate_all(bridge, items, caches, dev, mode, max_new=80):
    out = []
    for it in items:
        f = feats_for([it], caches).to(dev)
        out.append(bridge.generate(f, it["tag"], it["question"], max_new_tokens=max_new, mode=mode))
    return out


def facts(items, gens, phase, caches=None):
    """Compare generated answers with ground truth (ground-truth fields only)."""
    res = {"n": len(items)}
    if phase == 1:
        ev, ca, f1 = [], [], []
        for it, g in zip(items, gens):
            f1.append(token_f1(g, it["answer"]))
            if it["source"] != "dada":          # Nexar zero-shot items: V12 description only (no event/cause fields)
                continue
            parts = [x.strip() for x in g.split(";", 1)]
            ev.append(norm(parts[0]) == norm(it["gt"]["event"]))
            ca.append(len(parts) > 1 and norm(parts[1]) == norm(it["gt"]["cause"]))
        res.update(event_exact=float(np.mean(ev)) if ev else None, cause_exact=float(np.mean(ca)) if ca else None,
                   token_f1=float(np.mean(f1)))
        return res
    verdict, tti, event, cause, agree = [], [], [], [], []
    for it, g in zip(items, gens):
        p = parse_phase2(g)
        verdict.append(p["collision"] == it["label"])
        if it["label"] == 1:
            tti.append(p["tti"] is not None and abs(p["tti"] - it["tte"]) < 0.26)
        if it["source"] == "dada":
            event.append(norm(p["event"] or "") == norm(it["gt"]["event"]))
            if it["label"] == 1:
                cause.append(norm(p["cause"] or "") == norm(it["gt"]["cause"]))
        else:
            event.append(token_f1(p["event"] or "", it["gt"]["v12_caption"]))
        if caches is not None and p["collision"] is not None:
            agree.append(int(p["collision"] == int(caches[it["source"]].p_collision(it["id"]) >= 0.5)))
    m = lambda x: float(np.mean(x)) if x else None
    res.update(verdict_acc=m(verdict), tti_acc=m(tti), event_match=m(event), cause_exact=m(cause),
               agree_with_badas_score=m(agree))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", type=int, required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--source", default="dada", choices=["dada", "nexar"])
    ap.add_argument("--split", default="val")
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--lm", default=QWEN_ID)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--n-gen", type=int, default=50, help="windows to generate for diversity / facts")
    ap.add_argument("--max-ret", type=int, default=0, help="limit retrieval windows (0 = all)")
    ap.add_argument("--min-p", type=float, default=0, help="only windows with A1 P(collision) >= this (0 = off)")
    ap.add_argument("--max-p", type=float, default=0, help="only windows with A1 P(collision) < this (0 = off)")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device(args.device)
    torch.manual_seed(0)

    qwen, tok = load_lm(args.lm, dev)
    bridge = R1Bridge(qwen, tok).to(dev)
    sd = torch.load(args.ckpt, map_location="cpu")
    if "lora" in sd:
        bridge.enable_lora(r=args.lora_r)
    bridge.load(args.ckpt)
    bridge.eval()

    caches = {s: Cache(args.cache_dir, s) for s in ("dada", "nexar")}
    items = load_items(args.source, args.phase, args.split, caches[args.source],
                       min_p=args.min_p if args.min_p > 0 else None, max_p=args.max_p if args.max_p > 0 else None)
    print(f"[gates] phase {args.phase} {args.source}/{args.split}: {len(items)} windows "
          f"({'valid crash only' if args.phase == 1 else 'valid crash + no-crash'})")
    assert items, "no windows"

    rep = {"phase": args.phase, "source": args.source, "split": args.split, "ckpt": args.ckpt, "n_windows": len(items)}
    rep["loss_gaps"] = evaluate(bridge, items, caches, dev)
    sub = items[:args.max_ret] if args.max_ret else items
    rep["retrieval_at_10"] = retrieval_at10(bridge, sub, caches, dev)
    rep["ara_real"] = ara_choice(bridge, items, caches, dev, args.phase, "real")
    rep["ara_blank"] = ara_choice(bridge, items, caches, dev, args.phase, "blank")

    rg = random.Random(0)                      # generations: half crash / half no-crash when both exist
    pos, neg = [i for i in items if i["label"] == 1], [i for i in items if i["label"] == 0]
    if pos and neg:
        k = min(args.n_gen // 2, len(pos), len(neg))
        gen_items = rg.sample(pos, k) + rg.sample(neg, k)
    else:
        gen_items = rg.sample(items, min(args.n_gen, len(items)))
    gens = {m: generate_all(bridge, gen_items, caches, dev, m) for m in ("real", "blank")}
    rep["diversity"] = {m: {"distinct_of_n": len(set(g)), "n": len(g), "distinct_1": distinct_n(g, 1),
                            "distinct_2": distinct_n(g, 2)} for m, g in gens.items()}
    rep["facts"] = {m: facts(gen_items, g, args.phase, caches) for m, g in gens.items()}
    with open(out / "generations.jsonl", "w", encoding="utf-8") as f:
        for it, a, b in zip(gen_items, gens["real"], gens["blank"]):
            f.write(json.dumps({"id": it["id"], "label": it["label"], "tte": it["tte"], "target": it["answer"],
                                "real": a, "blank": b}, ensure_ascii=False) + "\n")

    g = rep["loss_gaps"]
    gate = {
        "wrong_video_gap_ci_gt0": g["gap_wrong_ci"][0] > 0,
        "blank_video_gap_ci_gt0": g["gap_blank_ci"][0] > 0,
        "retrieval_at10_ge_2.5x_chance": rep["retrieval_at_10"]["top1"] >= 2.5 * rep["retrieval_at_10"]["chance"],
        "diversity_ge_90pct": rep["diversity"]["real"]["distinct_of_n"] >= 0.9 * rep["diversity"]["real"]["n"],
        "ara_real_gt_blank": rep["ara_real"]["acc"] > rep["ara_blank"]["acc"] if rep["ara_real"]["n"] else None,
    }
    rep["gates"] = gate
    rep["gates_pass_all"] = all(v for v in gate.values() if v is not None)
    (out / "gates.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")

    lines = [f"# gates - phase {args.phase} - {args.source}/{args.split} - {len(items)} windows", "",
             f"checkpoint: `{args.ckpt}`", "",
             "| gate | value | pass |", "|---|---|---|",
             f"| wrong-video gap | {g['gap_wrong']:+.3f} CI {g['gap_wrong_ci'][0]:+.3f}..{g['gap_wrong_ci'][1]:+.3f} (loss real {g['loss_real']:.3f}, wrong {g['loss_wrong']:.3f}) | {gate['wrong_video_gap_ci_gt0']} |",
             f"| blank-video gap | {g['gap_blank']:+.3f} CI {g['gap_blank_ci'][0]:+.3f}..{g['gap_blank_ci'][1]:+.3f} (loss blank {g['loss_blank']:.3f}) | {gate['blank_video_gap_ci_gt0']} |",
             f"| retrieval@1 of 10 | {rep['retrieval_at_10']['top1']:.3f} (chance 0.10, n={rep['retrieval_at_10']['n']}) | {gate['retrieval_at10_ge_2.5x_chance']} |",
             f"| diversity (real) | {rep['diversity']['real']['distinct_of_n']}/{rep['diversity']['real']['n']} distinct (blank: {rep['diversity']['blank']['distinct_of_n']}) | {gate['diversity_ge_90pct']} |",
             f"| ArA 5-option cause | real {rep['ara_real']['acc']:.3f} vs blank {rep['ara_blank']['acc']:.3f} (chance 0.20, n={rep['ara_real']['n']}) | {gate['ara_real_gt_blank']} |",
             "", "facts (real vs blank): " + json.dumps(rep["facts"]), "",
             f"**all gates pass: {rep['gates_pass_all']}**"]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
