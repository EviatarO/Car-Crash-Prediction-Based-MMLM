"""
r1_data.py - data plumbing for r1_train.py / r1_eval_gates.py: manifest rows + cached V-JEPA tokens ->
(question, answer) items and padded batches. Only windows that are (a) valid under the window visibility
rule and (b) present in the feature cache are used.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch

import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from r1_common import Q_PHASE1, Q_PHASE2, TAG, q_reason, reasoning_group, reasoning_target  # noqa: E402

ROOT = HERE.parents[1]
MANIFEST = {"nexar": ROOT / "dataset" / "manifests" / "r1_nexar_v12_windows.jsonl",
            "dada": ROOT / "dataset" / "manifests" / "r1_mmau_dada_windows.jsonl",
            "nexar_reasoning": ROOT / "dataset" / "manifests" / "r1_nexar_reasoning_windows.jsonl",
            "nexar_reasoning_shifted": ROOT / "dataset" / "manifests" / "r1_nexar_reasoning_shifted.jsonl"}


class Cache:
    """Cached tokens for one source: features_<source>.npy (N,2048,1024) fp16 memmap + index_<source>.jsonl.
    inp (plan 2026-10-13) selects what get() returns, concatenated on the channel axis:
      last          (B,2048,1024)  last-layer tokens (the original input)
      last+attn     (B,2048,1025)  + the crash head's attention per token (attn_<source>.npy), scaled to mean 1 per window, log1p
      l11+last      (B,2048,2048)  layer-11 tokens (features_<source>_l11.npy) first, then the last layer
      l11+last+attn (B,2048,2049)"""

    def __init__(self, cache_dir, source, inp="last"):
        d = Path(cache_dir)
        self.source, self.inp = source, inp
        self.rows = {}
        self.mm = self.mm_l11 = self.mm_attn = None
        ix = d / f"index_{source}.jsonl"
        if ix.exists():
            for l in open(ix, encoding="utf-8"):
                r = json.loads(l)
                self.rows[r["id"]] = r
            self.mm = np.load(d / f"features_{source}.npy", mmap_mode="r")
            if inp.startswith("l11"):
                self.mm_l11 = np.load(d / f"features_{source}_l11.npy", mmap_mode="r")
            if inp.endswith("attn"):
                self.mm_attn = np.load(d / f"attn_{source}.npy", mmap_mode="r")

    def __contains__(self, wid):
        return wid in self.rows

    def get(self, ids):
        srcs = [self.rows[i]["row"] for i in ids]
        parts = []
        if self.mm_l11 is not None:
            parts.append(torch.from_numpy(np.stack([np.asarray(self.mm_l11[r]) for r in srcs])).float())
        parts.append(torch.from_numpy(np.stack([np.asarray(self.mm[r]) for r in srcs])).float())
        if self.mm_attn is not None:
            a = torch.from_numpy(np.stack([np.asarray(self.mm_attn[r]) for r in srcs])).float()      # (B, 2048) attention received per token
            a = torch.log1p(a / a.mean(1, keepdim=True).clamp_min(1e-12))
            parts.append(a.unsqueeze(-1))
        return parts[0] if len(parts) == 1 else torch.cat(parts, -1)

    def p_collision(self, wid):
        return self.rows[wid]["p_collision"]


def load_items(source, phase, split, cache: Cache, min_p=None, max_p=None, nocrash_only=False, order="reason_first", box_hint=False):
    """phase 1: DADA valid crash windows, Q_PHASE1 -> phase1 target. phase 2: valid crash + no-crash, Q_PHASE2.
    min_p / max_p: keep only windows whose cached A1 crash score is >= min_p / < max_p (option b, 2026-10-06:
    Phase 1 trains on DADA crash windows the frozen encoder itself flags; the rest is a separate evaluation set)."""
    out = []
    for l in open(MANIFEST[source], encoding="utf-8"):
        r = json.loads(l)
        if not r["valid"] or r["split"] != split or r["id"] not in cache:
            continue
        if source.startswith("nexar_reasoning"):      # reasoning dataset (and its frame-shifted copies): the class is the VISIBLE-HAZARD label
            g = r["gt"]
            out.append({"id": r["id"], "source": source, "video_key": r["video_key"], "label": g["label_vis"], "label_nexar": r["label"],
                        "tte": r["tte"], "pre_alert": g["pre_alert"], "group": reasoning_group(r["label"], r["tte"], g["pre_alert"]),
                        "src_id": r.get("source_id", r["id"]), "shift": r.get("shift", 0), "split": r["split"],
                        "question": q_reason(order) + ("\n" + g["box_hint"] if box_hint else ""), "tag": TAG[source], "order": order, "gt": g,
                        "answer": reasoning_target(g["explanation"], g["prediction"], g["label_vis"], order)})
            continue
        if nocrash_only:              # phase-1 hallucination check: no-crash windows, no target text (generation only)
            if r["label"] != 0:
                continue
            out.append({"id": r["id"], "source": source, "video_key": r["video_key"], "label": 0, "tte": None,
                        "question": Q_PHASE1, "answer": "", "tag": TAG[source], "gt": r["gt"]})
            continue
        if min_p is not None and cache.p_collision(r["id"]) < min_p:
            continue
        if max_p is not None and cache.p_collision(r["id"]) >= max_p:
            continue
        if phase == 1:
            if source == "nexar":      # zero-shot evaluation only (never trained on in phase 1):
                q, a = Q_PHASE1, r["gt"]["v12_caption"]   # V12 literal description vs the phase-1 question
            elif r["label"] != 1:
                continue
            else:
                q, a = Q_PHASE1, r["targets"]["phase1"]
        else:
            q, a = Q_PHASE2, r["targets"]["phase2"]
        if a is None:
            continue
        out.append({"id": r["id"], "source": source, "video_key": r["video_key"], "label": r["label"],
                    "tte": r["tte"], "question": q, "answer": a, "tag": TAG[source], "gt": r["gt"]})
    return out


def epoch_items(items, rng: random.Random, balance: bool):
    """Phase 2 sampler: 50/50 on `label`. With k = min(#neg, #pos), draw k from BOTH classes (old DADA+Nexar data: negatives are the
    minority, so all of them are kept - unchanged behaviour). If negatives are the majority (reasoning dataset) the k negatives are
    drawn proportionally per `group` (normal / pre_alert), so the hard negatives keep their share. Phase 1: all."""
    items = list(items)
    if any("variants" in i for i in items):                # augmentation (plan 2026-10-14): realised AFTER the 50/50 draw
        return [realize(i, rng) if "variants" in i else i for i in _epoch_base(items, rng, balance)]
    return _epoch_base(items, rng, balance)


def _epoch_base(items, rng, balance):
    if any("keep_p" in i for i in items):                  # option 2: template cap (items with a frequent opening kept with prob cap/count)
        items = [i for i in items if rng.random() < i.get("keep_p", 1.0)]
    if balance:
        neg = [i for i in items if i["label"] == 0]
        pos = [i for i in items if i["label"] == 1]
        if neg and pos:
            k = min(len(neg), len(pos))
            items = _draw(neg, k, rng) + _draw(pos, k, rng)
    rng.shuffle(items)
    return items


def load_texts(path):
    """paraphrases.jsonl (r1_paraphrase_check.py) -> {original window id: [(explanation, outcome) of rephrased v1, v2, v3 ...]} (accepted versions only)"""
    out = {}
    for l in open(path, encoding="utf-8"):
        r = json.loads(l)
        out.setdefault(r["id"], []).append((r["source"], r["explanation"], r["outcome"]))
    return {k: [(e, o) for _, e, o in sorted(v)] for k, v in out.items()}


def attach_aug(items, shifted_items, texts, p_teacher=0.4):
    """Give every ORIGINAL training item its alternatives: `variants` = [(source, id), ...] (the original window + its shifted copies that exist in the
    cache) and `texts` = [(explanation, outcome), ...] of the rephrased versions. realize() draws one window version and one text version."""
    by_src = {}
    for s in shifted_items:
        by_src.setdefault(s["src_id"], []).append(s)
    for it in items:
        it["variants"] = [(it["source"], it["id"])] + [(s["source"], s["id"]) for s in sorted(by_src.get(it["id"], []), key=lambda x: x["shift"])]
        it["texts"] = texts.get(it["id"], []) if texts else []
        it["p_teacher"] = p_teacher
    return items


def realize(it, rng):
    """One (window version, text version) pair of an original item. The pair is always correct: every shifted copy shows the same scene (<= 0.1 s apart)
    and every text version states the same facts. The teacher's own text is drawn with probability p_teacher (style anchor), else a rephrased one."""
    src, wid = rng.choice(it["variants"])
    out = dict(it, source=src, id=wid)
    for k in ("variants", "texts", "p_teacher"):
        out.pop(k, None)
    if it["texts"] and rng.random() >= it["p_teacher"]:
        e, o = rng.choice(it["texts"])
        out["answer"] = reasoning_target(e, o, it["label"], it["order"])
    return out


def val_pairs(items, shifted_items, texts):
    """Enlarged validation (deterministic): original window k=0..3 (the original, then its shifts 1..3 that exist) with text version k (the teacher's,
    then rephrased v1..v3 that exist; else the teacher's). 612 originals -> about 2,448 pairs."""
    by_src = {}
    for s in shifted_items:
        by_src.setdefault(s["src_id"], []).append(s)
    out = []
    for it in items:
        wins = [it] + sorted(by_src.get(it["id"], []), key=lambda x: x["shift"])
        tx = [None] + (texts.get(it["id"], []) if texts else [])
        for k in range(max(len(wins), len(tx))):
            w = wins[k] if k < len(wins) else wins[0]
            t = tx[k] if k < len(tx) else None
            c = dict(w, src_id=it["id"])
            if t is not None:
                c["answer"] = reasoning_target(t[0], t[1], it["label"], it["order"])
            out.append(c)
    return out


def _draw(cls_items, k, rng):
    """k items of one class; stratified by `group` when the items have one (largest-remainder rounding), plain sample otherwise."""
    if k >= len(cls_items):
        return list(cls_items)
    groups = {}
    for it in cls_items:
        groups.setdefault(it.get("group"), []).append(it)
    if len(groups) == 1:
        return rng.sample(cls_items, k)
    quota = {g: k * len(v) / len(cls_items) for g, v in groups.items()}
    take = {g: int(q) for g, q in quota.items()}
    for g in sorted(quota, key=lambda g: quota[g] - take[g], reverse=True)[:k - sum(take.values())]:
        take[g] += 1
    return [x for g, v in groups.items() for x in rng.sample(v, take[g])]


def other_index(items, seed=0):
    """For wrong-video tests: for each item a partner window of a DIFFERENT video (fixed by seed)."""
    rng = random.Random(seed)
    n = len(items)
    out = []
    for i, it in enumerate(items):
        for _ in range(50):
            j = rng.randrange(n)
            if items[j]["video_key"] != it["video_key"]:
                break
        else:
            j = (i + 1) % n
        out.append(j)
    return out


def make_batch(prompts, items, with_answer=True, end=True):
    enc = [prompts.encode(it["tag"], it["question"], it["answer"] if with_answer else None, end=end) for it in items]
    return prompts.collate(enc)


def feats_for(items, caches):
    parts = []
    for it in items:
        parts.append(caches[it["source"]].get([it["id"]]))
    return torch.cat(parts, 0)
