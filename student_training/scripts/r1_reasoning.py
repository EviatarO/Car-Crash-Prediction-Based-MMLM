"""
r1_reasoning.py - decision scores of the reasoning model (plan 2026-10-12_Plan-Phase2-Reasoning-Retrain, section 4).

The score is NOT part of the prompt and is never trained on. It is a measurement: the answer is cut right after the label
"Collision:" and we read the language model's next-piece probabilities for " yes" and " no":

    score = log P(" yes" | video, prompt, answer-so-far) - log P(" no" | same)        (log-odds; P(yes) = sigmoid(score))

Computed exactly: one forward pass over [video, prompt, answer prefix], log-softmax of the language-model head at the last prefix position
(the old Phase-2 `verdict_probs` obtained the same number from the difference of two summed answer log-probabilities).

Where the answer is cut (order "reason_first", Run 1):
    before   "Reasoning:\\nCollision:"                 the reasoning is removed -> the decision from the features alone (off-format cut)
    teacher  "Reasoning: <teacher text>\\nCollision:"   the teacher's reasoning is forced in -> upper bound
    after    "Reasoning: <model's own text>\\nCollision:" the real use: decision after the model's own reasoning (text from greedy generation)
Order "verdict_first" (Run 2): the decision is the first piece of the answer, prefix "Collision:" (score name "first").
"""
from __future__ import annotations

import numpy as np
import torch

from r1_common import facts_line
from r1_data import feats_for


def prefix_for(kind: str, it: dict, expl: str | None = None, facts: str | None = None) -> str:
    """facts_first (plan 2026-10-15): before = 'Facts:\\nReasoning:\\nCollision:' (both cut out), teacher = the window's card + the teacher text,
    after = the model's own Facts line (`facts`, the generated line) + its own reasoning."""
    if kind == "first":
        return "Collision:"
    ff = it.get("order") == "facts_first"
    if kind == "before":
        return "Facts:\nReasoning:\nCollision:" if ff else "Reasoning:\nCollision:"
    text = it["gt"]["explanation"] if kind == "teacher" else (expl or "")
    if not ff:
        return f"Reasoning: {text.strip()}\nCollision:"
    head = facts_line(it["facts"]) if kind == "teacher" else (facts or "Facts:")
    return f"{head.strip()}\nReasoning: {text.strip()}\nCollision:"


@torch.no_grad()
def logodds(bridge, items, prefixes, caches, dev, bs=8):
    """log P(' yes') - log P(' no') right after each prefix (which must end at 'Collision:'). One value per item.
    One forward pass over [video, prompt, prefix]; the next-piece distribution is read at the last real position (exact)."""
    tok = bridge.prompts.tok
    yes_id, no_id = tok(" yes", add_special_tokens=False)["input_ids"], tok(" no", add_special_tokens=False)["input_ids"]
    assert len(yes_id) == len(no_id) == 1, "' yes' / ' no' must be single word-pieces"
    out = []
    for s in range(0, len(items), bs):
        chunk, pf = items[s:s + bs], prefixes[s:s + bs]
        feats = feats_for(chunk, caches).to(dev)
        enc = [bridge.prompts.encode(it["tag"], it["question"], p, end=False) for it, p in zip(chunk, pf)]
        ids, _, mm, att = [t.to(dev) for t in bridge.prompts.collate(enc)]
        h = bridge.hidden_vis(bridge.visual_tokens(feats), ids, mm, att)
        last = att.sum(1) - 1                                              # right-padded batches
        lp = torch.log_softmax(bridge.qwen.lm_head(h[torch.arange(len(chunk), device=dev), last]).float(), -1)
        out += (lp[:, yes_id[0]] - lp[:, no_id[0]]).cpu().tolist()
    return out


def score_kinds(order: str):
    return ("first",) if order == "verdict_first" else ("before", "teacher")


def auc_ap(y, s):
    from sklearn.metrics import average_precision_score, roc_auc_score
    y, s = np.asarray(y), np.asarray(s)
    if y.min() == y.max():
        return None, None
    return float(roc_auc_score(y, s)), float(average_precision_score(y, s))


@torch.no_grad()
def epoch_decision_report(bridge, items, caches, dev, order="reason_first", bs=8):
    """Cheap per-epoch decision AUC on validation windows (no generation): before / teacher (Run 1) or first (Run 2)."""
    bridge.eval()
    rep = {}
    y = [it["label"] for it in items]
    for kind in score_kinds(order):
        sc = logodds(bridge, items, [prefix_for(kind, it) for it in items], caches, dev, bs)
        auc, ap = auc_ap(y, sc)
        rep[kind] = {"auc": auc, "ap": ap, "acc_at_0": float(np.mean((np.asarray(sc) >= 0) == np.asarray(y)))}
    return rep
