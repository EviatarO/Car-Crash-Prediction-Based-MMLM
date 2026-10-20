"""
r1_eval_reasoning.py - evaluation of the reasoning model on the Nexar reasoning dataset (plan 2026-10-12_Plan-Phase2-Reasoning-Retrain, section 6).

  python r1_eval_reasoning.py --ckpt <phase2r>/best.pt --cache-dir <cache> --out-dir <dir> [--order reason_first|verdict_first|facts_first]
         [--facts-cards fact_cards_teacher.jsonl] [--drop-prealert]   (plan 2026-10-15)
         [--before-ckpt <phase1>/best.pt] [--old-ckpt <old phase2>/best.pt] [--limit N] [--lm tiny --device cpu]   (smoke)

Writes predictions.jsonl (one row per validation window), metrics.json, summary.md.
Decision scores are log-odds read at the label "Collision:" (r1_reasoning.py). Every thresholded number is at threshold 0.5
(score >= 0 for the language model, crash-model P >= 0.5).
  Run 1 (reason_first):  s_before (reasoning cut out), s_teacher (teacher reasoning forced), s_after (model's own reasoning)
  Run 2 (verdict_first): s_first
  s_old   : old Phase-2 model (--old-ckpt, its own question "Is a collision coming?...", same windows)
  a1_p    : the frozen crash model (vision path)
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "models"))
from r1_bridge import R1Bridge  # noqa: E402
from r1_common import FACT_SLOTS, ORDERS, Q_PHASE2, parse_reasoning  # noqa: E402
from r1_data import Cache, load_cards, load_items  # noqa: E402
from r1_eval_gates import retrieval_at10, token_f1  # noqa: E402
from r1_reasoning import auc_ap, logodds, prefix_for, score_kinds  # noqa: E402
from r1_train import QWEN_ID, evaluate, load_lm  # noqa: E402

GROUPS = ("normal", "pre_alert", "tte1.5", "tte1.25", "tte1.0", "tte0.75", "tte0.5")          # tte0.75 / tte1.25: the extra windows of plan 2026-10-19
ROAD_USERS = [("pedestrian", r"pedestrian|person|worker|walker|child"), ("cyclist", r"cyclist|bicycl|bike\b|scooter"),
              ("motorcycle", r"motorcycl|motorbike"), ("bus", r"\bbus\b"), ("truck", r"truck|lorry|dump|semi\b|trailer|tractor"),
              ("van", r"\bvan\b|minivan"), ("suv", r"\bsuv\b|sport utility|crossover|jeep"), ("pickup", r"pickup"),
              ("car", r"\bcar\b|sedan|hatchback|coupe|wagon|taxi|vehicle|cab\b")]
NEG_OUTCOME = r"no collision|without|not collid|avoid|safe|clear|no contact|no impact|passes|remain|neither|\bnor\b|does not|doesn't"


def road_user(text):
    """First road-user class mentioned in a text (position-based), None if none."""
    best = None
    for name, pat in ROAD_USERS:
        m = re.search(pat, (text or "").lower())
        if m and (best is None or m.start() < best[0]):
            best = (m.start(), name)
    return best[1] if best else None


def outcome_says_collision(text):
    t = (text or "").lower()
    if re.search(NEG_OUTCOME, t):
        return False
    return bool(re.search(r"collid|collision|crash|\bhit|strike|contact|impact|rear-end|runs? into|\bram(s|med)?\b|smash|plow", t))


def fmt(x, k=3):
    return 'n/a' if x is None else f'{x:.{k}f}'


def load_ckpt(bridge, path, lora_r):
    sd = torch.load(path, map_location="cpu")
    if "lora" in sd and not hasattr(bridge.text, "peft_config"):
        bridge.enable_lora(r=lora_r)
    bridge.load(path)
    bridge.eval()


def loss_summary(bridge, items, caches, dev):
    g = evaluate(bridge, items, caches, dev)
    g["perplexity_real"] = float(math.exp(g["loss_real"]))
    return g


@torch.no_grad()
def generate_all(bridge, items, caches, dev, mode, max_new):
    from r1_data import feats_for
    out = []
    for i, it in enumerate(items):
        out.append(bridge.generate(feats_for([it], caches).to(dev), it["tag"], it["question"], max_new_tokens=max_new, mode=mode))
        if (i + 1) % 25 == 0:
            print(f"  [gen {mode}] {i + 1}/{len(items)}", flush=True)
    return out


# ----------------------------------------------------------------------------- metrics (also used on the PC)
def conf(y, pred):
    y, pred = np.asarray(y), np.asarray(pred)
    return {"TP": int(((pred == 1) & (y == 1)).sum()), "FN": int(((pred == 0) & (y == 1)).sum()),
            "FP": int(((pred == 1) & (y == 0)).sum()), "TN": int(((pred == 0) & (y == 0)).sum())}


def mcnemar_exact(ok_a, ok_b):
    """Exact two-sided McNemar on paired correctness vectors: b = A right & B wrong, c = A wrong & B right."""
    from scipy.stats import binomtest
    ok_a, ok_b = np.asarray(ok_a, bool), np.asarray(ok_b, bool)
    b, c = int((ok_a & ~ok_b).sum()), int((~ok_a & ok_b).sum())
    return {"only_first_right": b, "only_second_right": c, "p": float(binomtest(b, b + c, 0.5).pvalue) if b + c else 1.0}


def boot_auc_diff(rows, key_a, key_b, label="label", n=2000, seed=0):
    """Paired bootstrap over VIDEOS of AUC(a) - AUC(b)."""
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    by_v = defaultdict(list)
    for i, r in enumerate(rows):
        by_v[r["video_key"]].append(i)
    vids = list(by_v)
    y = np.array([r[label] for r in rows])
    a = np.array([r[key_a] for r in rows], float)
    b = np.array([r[key_b] for r in rows], float)
    diffs = []
    for _ in range(n):
        idx = np.concatenate([by_v[vids[k]] for k in rng.integers(0, len(vids), len(vids))])
        if y[idx].min() == y[idx].max():
            continue
        diffs.append(roc_auc_score(y[idx], a[idx]) - roc_auc_score(y[idx], b[idx]))
    d0 = roc_auc_score(y, a) - roc_auc_score(y, b)
    if not diffs:                                      # tiny samples (smoke tests): every resample had a single class
        return {"diff": float(d0), "ci95": [None, None]}
    return {"diff": float(d0), "ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]}


def compute_metrics(rows, score_keys):
    """rows: predictions. score_keys: list of log-odds keys (>= 0 means 'yes'); 'a1_p' (probability, >= 0.5) is always added."""
    m = {"n": len(rows), "decision": {}, "by_group": {}, "paired": {}}
    y = np.array([r["label"] for r in rows])
    pv = np.array([r["label_nexar"] for r in rows])
    keep = np.array([not r["pre_alert"] for r in rows])        # windows comparable with the old numbers (Nexar label)
    sc = {k: np.array([r[k] for r in rows], float) for k in score_keys}
    sc["a1_p"] = np.array([r["a1_p"] for r in rows], float)
    thr = {k: (0.0 if k != "a1_p" else 0.5) for k in sc}
    for k, s in sc.items():
        auc, ap = auc_ap(y, s)
        auc_n, ap_n = auc_ap(pv[keep], s[keep])
        pred = (s >= thr[k]).astype(int)
        c = conf(y, pred)
        m["decision"][k] = {"auc_vs_label_vis": auc, "ap_vs_label_vis": ap, "auc_no_prealert_vs_nexar_label": auc_n, "ap_no_prealert_vs_nexar_label": ap_n,
                            "threshold": ("log-odds 0 (P=0.5)" if k != "a1_p" else "P=0.5"), "acc_at_thr": float((pred == y).mean()),
                            "balanced_acc_at_thr": float(0.5 * (c["TP"] / max(1, c["TP"] + c["FN"]) + c["TN"] / max(1, c["TN"] + c["FP"]))),
                            "confusion_at_thr": c}
    neg = y == 0
    for g in GROUPS:
        gi = np.array([r["group"] == g for r in rows])
        if not gi.any():
            continue
        e = {"n": int(gi.sum()), "n_pos_label_vis": int(y[gi].sum())}
        for k, s in sc.items():
            pred = (s >= thr[k]).astype(int)
            e[k] = {"confusion": conf(y[gi], pred[gi]), "acc": float((pred[gi] == y[gi]).mean())}
            if g.startswith("tte"):                              # AUC of this TTE's positives against ALL negatives
                sel = gi | neg
                a, _ = auc_ap(y[sel], s[sel])
                e[k]["auc_vs_all_negatives"] = a
        m["by_group"][g] = e
    main = "s_after" if "s_after" in sc else score_keys[0]
    for k in list(sc):
        if k in (main, "a1_p"):
            continue
        ok_a = (sc[main] >= thr[main]).astype(int) == y
        ok_b = (sc[k] >= thr[k]).astype(int) == y
        m["paired"][f"{main}_vs_{k}"] = {"mcnemar_at_thr": mcnemar_exact(ok_a, ok_b), "auc_diff": boot_auc_diff(rows, main, k, "label")}
    ok_a = (sc[main] >= thr[main]).astype(int) == y
    ok_b = (sc["a1_p"] >= 0.5).astype(int) == y
    m["paired"][f"{main}_vs_a1_p"] = {"mcnemar_at_thr": mcnemar_exact(ok_a, ok_b), "auc_diff": boot_auc_diff(rows, main, "a1_p", "label"),
                                      "rescues_by_group": {g: {"reasoning_right_crash_model_wrong": int(sum(1 for i, r in enumerate(rows) if r["group"] == g and ok_a[i] and not ok_b[i])),
                                                               "crash_model_right_reasoning_wrong": int(sum(1 for i, r in enumerate(rows) if r["group"] == g and ok_b[i] and not ok_a[i]))}
                                                           for g in GROUPS}}
    return m


def missed_crashes(rows):
    """Plan 2026-10-15 gate number: per crash TTE group, the share of visible-hazard crash windows (label 1) that each path calls 'no':
    the written verdict (Collision: no), the decision score at 0.5 (s_after or s_first) and the crash model (P < 0.5)."""
    main = "s_after" if rows and "s_after" in rows[0] else ("s_first" if rows and "s_first" in rows[0] else None)
    out = {}
    for g in ("tte0.5", "tte0.75", "tte1.0", "tte1.25", "tte1.5", "all"):
        rs = [r for r in rows if r["label"] == 1 and (r["group"] == g or (g == "all" and r["group"].startswith("tte")))]
        if not rs:
            continue
        out[g] = {"n": len(rs),
                  "written_no": float(np.mean([r["gen_collision"] != 1 for r in rs])) if any(r.get("gen_text") for r in rs) else None,
                  "score_no": float(np.mean([r[main] < 0 for r in rs])) if main else None,
                  "crash_model_no": float(np.mean([r["a1_p"] < 0.5 for r in rs]))}
    return out


def facts_line_accuracy(rows):
    """facts_first: the generated Facts line vs the window's card, per slot (windows whose card slot is not_stated are skipped), with the
    majority baseline of the same windows; crash windows (TTE groups) and all windows."""
    out = {}
    for name, sel in (("all", lambda r: True), ("crash", lambda r: r["group"].startswith("tte"))):
        rs = [r for r in rows if sel(r) and r.get("teacher_facts")]
        e = {"n": len(rs), "line_present": float(np.mean([r.get("gen_facts") is not None for r in rs])) if rs else None}
        for s in FACT_SLOTS:
            k = [(r["teacher_facts"][s], (r.get("gen_facts") or {}).get(s)) for r in rs if r["teacher_facts"][s] != "not_stated"]
            if not k:
                continue
            maj = Counter(a for a, _ in k).most_common(1)[0]
            e[s] = {"n": len(k), "acc": float(np.mean([a == b for a, b in k])), "majority": maj[0], "majority_acc": maj[1] / len(k)}
        out[name] = e
    return out


def text_metrics(rows):
    """Cheap text numbers; BERTScore / ROUGE / BLEU / METEOR are computed on the PC from predictions.jsonl."""
    f1 = [token_f1(r["gen_explanation"] or "", r["teacher_explanation"]) for r in rows]
    ru = [(road_user(r["gen_explanation"]), road_user(r["teacher_explanation"])) for r in rows]
    wv = [r["gen_collision"] for r in rows]
    parsed = [r["gen_collision"] is not None and r["gen_explanation"] and r["gen_outcome"] for r in rows]
    cons = [(r["gen_collision"] == int(outcome_says_collision(r["gen_outcome"]))) for r in rows if r["gen_collision"] is not None and r["gen_outcome"]]
    return {"token_f1_vs_teacher": float(np.mean(f1)),
            "road_user_match": float(np.mean([a == b for a, b in ru if b is not None])) if any(b is not None for _, b in ru) else None, "road_user_n": int(sum(b is not None for _, b in ru)),
            "format_ok": float(np.mean([bool(p) for p in parsed])),
            "written_verdict_acc_vs_label_vis": float(np.mean([w == r["label"] for w, r in zip(wv, rows) if w is not None])) if any(w is not None for w in wv) else None,
            "written_verdict_vs_score_agree": float(np.mean([(r["gen_collision"] == int(r["s_after"] >= 0)) for r in rows if r["gen_collision"] is not None and "s_after" in r])) if rows and "s_after" in rows[0] else None,
            "verdict_vs_outcome_consistent(heuristic)": float(np.mean(cons)) if cons else None,
            "distinct_explanations": int(len({r["gen_explanation"] for r in rows})), "n": len(rows)}


def stratified_sample(items, n, seed=0):
    """n windows drawn at random within each (label, group) stratum, proportionally (largest remainders), in a fixed order."""
    import random as _r
    if n >= len(items):
        return sorted(items, key=lambda it: it["id"])
    strata = {}
    for it in items:
        strata.setdefault((it["label"], it["group"]), []).append(it)
    keys = sorted(strata, key=str)
    exact = {k: n * len(strata[k]) / len(items) for k in keys}
    take = {k: int(exact[k]) for k in keys}
    for k in sorted(keys, key=lambda k: exact[k] - take[k], reverse=True)[:n - sum(take.values())]:
        take[k] += 1
    rng = _r.Random(seed)
    out = [it for k in keys for it in rng.sample(strata[k], take[k])]
    return sorted(out, key=lambda it: it["id"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--order", default="reason_first", choices=list(ORDERS))
    ap.add_argument("--facts-cards", default=None, help="fact_cards_teacher.jsonl: needed for --order facts_first (also scores the generated Facts line)")
    ap.add_argument("--drop-prealert", action="store_true", help="score without the pre-alert windows (plan 2026-10-15)")
    ap.add_argument("--split", default="val")
    ap.add_argument("--source", default="nexar_reasoning", choices=["nexar_reasoning", "nexar_reasoning_test", "nexar_reasoning_test_public"],
                    help="windows to score (cache name = source); nexar_reasoning_test = the private test windows (plan 2026-10-17), _public = the public ones")
    ap.add_argument("--no-ref", action="store_true",
                    help="the windows have no reference text yet (test): skip loss / perplexity / gaps / retrieval / blank-video / text metrics; the decision scores, "
                         "missed crashes and the generated answers are still written")
    ap.add_argument("--lm", default=QWEN_ID)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--before-ckpt", default=None, help="Phase-1 checkpoint: loss / perplexity / gaps on the new targets (the 'before' row)")
    ap.add_argument("--old-ckpt", default=None, help="old Phase-2 checkpoint: its verdict score on the same windows (its own question)")
    ap.add_argument("--input", default="last", choices=["last", "last+attn", "l11+last", "l11+last+attn"], help="merger input of the checkpoint")
    ap.add_argument("--box-hint", action="store_true", help="option 3 model: the prompt carries the position of the attended road user")
    ap.add_argument("--train-threshold", type=int, default=0,
                    help="N>0: choose the 'before reasoning' decision threshold on N random TRAINING windows (max balanced accuracy) and report it on validation")
    ap.add_argument("--max-new", type=int, default=160)
    ap.add_argument("--n-blank", type=int, default=20)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-extra-train", action="store_true", help="plan 2026-10-19, chain A: leave the extra TTE windows out of the TRAIN split (the model never trained on them)")
    ap.add_argument("--no-generate", action="store_true", help="scoring only (plan 2026-10-19): the decision scores, no generated text and no text metrics; verdict_first only. A scoring pass takes minutes, a generating one about an hour per 650 windows")
    ap.add_argument("--sample", type=int, default=0, help="N>0: a random subset of N windows, stratified by (label, group) (plan 2026-10-18: the train-split report)")
    ap.add_argument("--sample-seed", type=int, default=0)
    ap.add_argument("--max-ret", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device(args.device)
    torch.manual_seed(0)
    qwen, tok = load_lm(args.lm, dev)
    bridge = R1Bridge(qwen, tok, inp=args.input).to(dev)
    caches = {"nexar_reasoning": Cache(args.cache_dir, args.source, args.input), args.source: Cache(args.cache_dir, args.source, args.input)}
    cache = caches[args.source]
    cards = load_cards(args.facts_cards) if args.facts_cards else None
    items = load_items(args.source, 2, "test" if args.source.startswith("nexar_reasoning_test") else args.split, cache, order=args.order, box_hint=args.box_hint, facts=cards,
                       drop_prealert=args.drop_prealert, drop_extra_train=args.no_extra_train)
    if args.sample and args.sample < len(items):
        items = stratified_sample(items, args.sample, args.sample_seed)
    if args.limit:
        items = items[:args.limit]
    assert items, "no validation windows in the cache"
    print(f"[eval] {len(items)} windows (order {args.order})", flush=True)
    rep = {"order": args.order, "ckpt": args.ckpt, "n_windows": len(items)}

    if args.before_ckpt and not args.no_ref:                          # (1) Phase-1 model, no LoRA yet
        load_ckpt(bridge, args.before_ckpt, args.lora_r)
        rep["before_phase1"] = loss_summary(bridge, items, caches, dev)
        print("[before]", json.dumps(rep["before_phase1"]), flush=True)

    load_ckpt(bridge, args.ckpt, args.lora_r)                         # (2) the trained model
    if args.no_ref:
        rep["after"], rep["retrieval_at_10"] = None, None
    else:
        rep["after"] = loss_summary(bridge, items, caches, dev)
        sub = items[:args.max_ret] if args.max_ret else items
        rep["retrieval_at_10"] = retrieval_at10(bridge, sub, caches, dev)
        print("[after]", json.dumps(rep["after"]), json.dumps(rep["retrieval_at_10"]), flush=True)

    rows = [{"id": it["id"], "video_key": it["video_key"], "frames_dir": it["gt"].get("frames_dir") or it["id"].replace("nexar_", "", 1),
             "label": it["label"], "label_nexar": it["label_nexar"], "tte": it["tte"], "pre_alert": it["pre_alert"], "group": it["group"],
             "a1_p": cache.p_collision(it["id"]), "teacher_explanation": it["gt"]["explanation"], "teacher_outcome": it["gt"]["prediction"],
             "teacher_risk": it["gt"]["risk"]} for it in items]
    if cards:
        for r, it in zip(rows, items):
            r["teacher_facts"] = it["facts"]
    score_keys = []
    kinds = tuple(k for k in score_kinds(args.order) if not (args.no_ref and k == "teacher"))     # no teacher text on the test windows
    for kind in kinds:
        sc = logodds(bridge, items, [prefix_for(kind, it) for it in items], caches, dev)
        for r, s in zip(rows, sc):
            r[f"s_{kind}"] = s
        score_keys.append(f"s_{kind}")
    if args.train_threshold:
        import random as _r
        tr_all = load_items("nexar_reasoning", 2, "train", cache, order=args.order, box_hint=args.box_hint, facts=cards, drop_prealert=args.drop_prealert, drop_extra_train=args.no_extra_train)
        tr_items = _r.Random(0).sample(tr_all, min(args.train_threshold, len(tr_all)))
        kind0 = kinds[0]
        s_tr = np.array(logodds(bridge, tr_items, [prefix_for(kind0, it) for it in tr_items], caches, dev))
        y_tr = np.array([it["label"] for it in tr_items])
        cand = np.unique(s_tr)
        bal = [0.5 * ((s_tr[y_tr == 1] >= t).mean() + (s_tr[y_tr == 0] < t).mean()) for t in cand]
        thr = float(cand[int(np.argmax(bal))])
        s_va = np.array([r[f"s_{kind0}"] for r in rows])
        y_va = np.array([r["label"] for r in rows])
        pred = (s_va >= thr).astype(int)
        rep["train_threshold"] = {"score": f"s_{kind0}", "n_train_windows": len(tr_items), "log_odds_threshold": thr,
                                  "p_threshold": float(1 / (1 + np.exp(-thr))), "balanced_acc_on_train_windows": float(max(bal)),
                                  "val_acc": float((pred == y_va).mean()), "val_confusion": conf(y_va, pred),
                                  "val_balanced_acc": float(0.5 * ((pred[y_va == 1] == 1).mean() + (pred[y_va == 0] == 0).mean()))}
        print("[train-threshold]", json.dumps(rep["train_threshold"]), flush=True)
    if args.no_generate:
        assert args.order == "verdict_first", "--no-generate needs --order verdict_first (reason_first scores the model's own written reasoning)"
    gens = [""] * len(items) if args.no_generate else generate_all(bridge, items, caches, dev, "real", args.max_new)
    parsed = [parse_reasoning(g) for g in gens]
    for r, g, p in zip(rows, gens, parsed):
        r.update(gen_text=g, gen_explanation=p["explanation"], gen_collision=p["collision"], gen_outcome=p["outcome"])
        if args.order == "facts_first":
            r["gen_facts"] = p["facts"]
    if args.order in ("reason_first", "facts_first"):
        expl = [p["explanation"] or g.split("Collision:")[0].split("Reasoning:")[-1].strip() for g, p in zip(gens, parsed)]
        fl = [(re.search(r"Facts:[^\n]*", g).group(0) if re.search(r"Facts:[^\n]*", g) else None) for g in gens]
        sc = logodds(bridge, items, [prefix_for("after", it, e, f) for it, e, f in zip(items, expl, fl)], caches, dev)
        for r, s in zip(rows, sc):
            r["s_after"] = s
        score_keys.append("s_after")
    nb = 0 if (args.no_ref or args.no_generate) else min(args.n_blank, len(items))
    rep["blank_generations"] = [{"id": it["id"], "text": g} for it, g in zip(items[:nb], generate_all(bridge, items[:nb], caches, dev, "blank", args.max_new))]

    if args.old_ckpt:                                                 # (3) old Phase-2 model on the same windows with its own question
        load_ckpt(bridge, args.old_ckpt, args.lora_r)
        old_items = [dict(it, question=Q_PHASE2, tag="[Nexar]") for it in items]
        sc = logodds(bridge, old_items, ["Collision:"] * len(items), caches, dev)
        for r, s in zip(rows, sc):
            r["s_old_phase2"] = s
        score_keys.append("s_old_phase2")

    rep.update(metrics=compute_metrics(rows, score_keys), text=None if (args.no_ref or args.no_generate) else text_metrics(rows), missed_crashes=missed_crashes(rows))
    if args.order == "facts_first":
        rep["facts_line"] = facts_line_accuracy(rows)
    with open(out / "predictions.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (out / "metrics.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    d = rep["metrics"]["decision"]
    lines = [f"# reasoning model evaluation - {len(rows)} validation windows - order {args.order}", "", f"checkpoint `{args.ckpt}`", "",
             "| score | AUC vs label_vis | AUC (no pre-alert) vs Nexar label | acc @ thr | balanced acc @ thr |", "|---|---|---|---|---|"]
    for k, v in d.items():
        lines.append(f"| {k} | {fmt(v['auc_vs_label_vis'])} | {fmt(v['auc_no_prealert_vs_nexar_label'])} | {v['acc_at_thr']:.3f} ({v['threshold']}) | {v['balanced_acc_at_thr']:.3f} |")
    if args.no_ref:
        lines += ["", "(no reference text: loss, gaps, retrieval and text metrics skipped)", "",
                  "missed crashes (share of visible-hazard crash windows called 'no'; thresholds 0.5): " + json.dumps(rep["missed_crashes"])]
        (out / "summary.md").write_text(chr(10).join(lines), encoding="utf-8")
        print(chr(10).join(lines))
        return
    lines += ["", f"loss / perplexity (teacher text): after {rep['after']['loss_real']:.3f} / {rep['after']['perplexity_real']:.2f}"
              + (f" , before (Phase 1) {rep['before_phase1']['loss_real']:.3f} / {rep['before_phase1']['perplexity_real']:.2f}" if 'before_phase1' in rep else ""),
              f"wrong-video gap {rep['after']['gap_wrong']:+.3f} {rep['after']['gap_wrong_ci']}; blank-video gap {rep['after']['gap_blank']:+.3f} {rep['after']['gap_blank_ci']}",
              f"retrieval@1 of 10: {rep['retrieval_at_10']['top1']:.3f} (chance 0.10)", "", "text: " + json.dumps(rep["text"]),
              "", "missed crashes (share of visible-hazard crash windows called 'no'; thresholds 0.5): " + json.dumps(rep["missed_crashes"])]
    if "facts_line" in rep:
        lines += ["", "Facts line vs the window's card: " + json.dumps(rep["facts_line"])]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
