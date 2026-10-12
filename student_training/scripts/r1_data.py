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
            "nexar_reasoning": ROOT / "dataset" / "manifests" / "r1_nexar_reasoning_windows.jsonl"}


class Cache:
    """Cached tokens for one source: features_<source>.npy (N,2048,1024) fp16 memmap + index_<source>.jsonl."""

    def __init__(self, cache_dir, source):
        d = Path(cache_dir)
        self.source = source
        self.rows = {}
        ix = d / f"index_{source}.jsonl"
        if ix.exists():
            for l in open(ix, encoding="utf-8"):
                r = json.loads(l)
                self.rows[r["id"]] = r
            self.mm = np.load(d / f"features_{source}.npy", mmap_mode="r")
        else:
            self.mm = None

    def __contains__(self, wid):
        return wid in self.rows

    def get(self, ids):
        srcs = [self.rows[i]["row"] for i in ids]
        return torch.from_numpy(np.stack([np.asarray(self.mm[r]) for r in srcs])).float()

    def p_collision(self, wid):
        return self.rows[wid]["p_collision"]


def load_items(source, phase, split, cache: Cache, min_p=None, max_p=None, nocrash_only=False, order="reason_first"):
    """phase 1: DADA valid crash windows, Q_PHASE1 -> phase1 target. phase 2: valid crash + no-crash, Q_PHASE2.
    min_p / max_p: keep only windows whose cached A1 crash score is >= min_p / < max_p (option b, 2026-10-06:
    Phase 1 trains on DADA crash windows the frozen encoder itself flags; the rest is a separate evaluation set)."""
    out = []
    for l in open(MANIFEST[source], encoding="utf-8"):
        r = json.loads(l)
        if not r["valid"] or r["split"] != split or r["id"] not in cache:
            continue
        if source == "nexar_reasoning":      # reasoning dataset: the class is the VISIBLE-HAZARD label; the Nexar label is kept for comparison
            g = r["gt"]
            out.append({"id": r["id"], "source": source, "video_key": r["video_key"], "label": g["label_vis"], "label_nexar": r["label"],
                        "tte": r["tte"], "pre_alert": g["pre_alert"], "group": reasoning_group(r["label"], r["tte"], g["pre_alert"]),
                        "question": q_reason(order), "tag": TAG[source], "order": order, "gt": g,
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
    if balance:
        neg = [i for i in items if i["label"] == 0]
        pos = [i for i in items if i["label"] == 1]
        if neg and pos:
            k = min(len(neg), len(pos))
            items = _draw(neg, k, rng) + _draw(pos, k, rng)
    rng.shuffle(items)
    return items


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
