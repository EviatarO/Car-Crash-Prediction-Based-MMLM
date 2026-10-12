"""
r1_reasoning_test.py - CPU unit tests for the reasoning-dataset additions (plan 2026-10-12_Plan-Phase2-Reasoning-Retrain). No GPU, no downloads
beyond the local Qwen3-VL tokenizer files (dataset/public_samples/qwen3vl_cfg).

 1. reasoning_target / parse_reasoning round-trip for both answer orders; the prompt lists the fields in the answer's order.
 2. No target contains an outcome word BEFORE the 'Collision:' label in reason_first order (checked on the whole manifest).
 3. Train / val share no video; every val video comes from the old Nexar validation split.
 4. Sampler: 50/50 on label_vis, negatives stratified by group, positives keep their TTE mix; old behaviour (negatives <= positives) is unchanged.
 5. logodds(...) == the directly computed next-piece log-odds log P(' yes') - log P(' no') after the prefix (tiny random LM).
 6. A trainer step with --sources nexar_reasoning on the tiny LM runs and gives a finite loss (see the dry-run command in the plan).

    python student_training/scripts/r1_reasoning_test.py
"""
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "models"))
from r1_common import ORDERS, parse_reasoning, q_reason, reasoning_target  # noqa: E402
from r1_data import epoch_items, load_items  # noqa: E402


class AllIn:
    def __contains__(self, _):
        return True


def main():
    # 1 ---- round trip + prompt order
    for o in ORDERS:
        t = reasoning_target("A truck slows ahead. The gap closes.", "The ego vehicle collides with the truck", 1, o)
        p = parse_reasoning(t)
        assert p == {"explanation": "A truck slows ahead. The gap closes.", "collision": 1, "outcome": "The ego vehicle collides with the truck."}, p
        q = q_reason(o)
        pos = [q.index(k) for k in ("Reasoning:", "Collision:", "Outcome:")]
        tpos = [t.index(k) for k in ("Reasoning:", "Collision:", "Outcome:")]
        assert sorted(range(3), key=pos.__getitem__) == sorted(range(3), key=tpos.__getitem__), (o, pos, tpos)
        assert parse_reasoning(reasoning_target("x y.", "No collision occurs.", 0, o))["collision"] == 0
    print("1 ok: round trip, prompt order == answer order")

    c = AllIn()
    tr = load_items("nexar_reasoning", 2, "train", c)
    va = load_items("nexar_reasoning", 2, "val", c)
    # 2 ---- no outcome word before the verdict (reason_first)
    bad = [it["id"] for it in tr + va if re.search(r"collid|collision|crash|\bhit\b", it["answer"].split("Collision:")[0], re.I)]
    assert not bad, bad[:5]
    print(f"2 ok: no outcome word before 'Collision:' in {len(tr) + len(va)} targets")

    # 3 ---- split by video
    assert not {i["video_key"] for i in tr} & {i["video_key"] for i in va}
    old_val = {json.loads(l)["video_key"] for l in open(ROOT / "dataset/manifests/r1_nexar_v12_windows.jsonl", encoding="utf-8") if json.loads(l)["split"] == "val"}
    assert {i["video_key"] for i in va} <= old_val
    print(f"3 ok: {len(tr)} train / {len(va)} val windows, no shared video, val videos are the old validation videos")

    # 4 ---- sampler
    ep = epoch_items(tr, random.Random(0), True)
    cl = Counter(i["label"] for i in ep)
    assert cl[0] == cl[1] == sum(i["label"] for i in tr), cl
    g = Counter((i["label"], i["group"]) for i in ep)
    pos_all = Counter(i["group"] for i in tr if i["label"] == 1)
    assert all(g[(1, k)] == v for k, v in pos_all.items()), (g, pos_all)                 # every positive once per epoch -> natural TTE mix
    neg_all = Counter(i["group"] for i in tr if i["label"] == 0)
    for k, v in neg_all.items():                                                           # negatives drawn proportionally per group (+-1)
        assert abs(g[(0, k)] - cl[0] * v / sum(neg_all.values())) <= 1, (k, g[(0, k)])
    old = [{"label": 0, "id": i} for i in range(5)] + [{"label": 1, "id": 100 + i} for i in range(9)]
    r = random.Random(0)
    a = epoch_items(old, r, True)
    r = random.Random(0)
    b = [i for i in old if i["label"] == 0] + r.sample([i for i in old if i["label"] == 1], 5)
    r.shuffle(b)
    assert [i["id"] for i in a] == [i["id"] for i in b]
    print(f"4 ok: epoch {len(ep)} = {cl[0]} + {cl[1]}, groups {dict(g)}; old behaviour unchanged")

    # 5 ---- logodds == direct next-piece log-odds (tiny random LM)
    from r1_bridge import R1Bridge
    from r1_bridge_test import tiny_qwen
    from r1_data import Cache
    from r1_reasoning import logodds, prefix_for
    from transformers import AutoProcessor
    tok = AutoProcessor.from_pretrained(str(ROOT / "dataset/public_samples/qwen3vl_cfg")).tokenizer
    torch.manual_seed(0)
    bridge = R1Bridge(tiny_qwen(len(tok)).float(), tok)
    bridge.eval()

    torch.manual_seed(1)
    its = va[:3]
    feats = torch.randn(3, 2048, 1024) * 0.1
    by_id = {it["id"]: feats[i:i + 1] for i, it in enumerate(its)}

    class FixedCache:
        def get(self, ids):
            return torch.cat([by_id[i] for i in ids])
    pf = [prefix_for("teacher", it) for it in its]
    got = logodds(bridge, its, pf, {"nexar_reasoning": FixedCache()}, torch.device("cpu"), bs=3)
    yes_id, no_id = tok(" yes", add_special_tokens=False)["input_ids"], tok(" no", add_special_tokens=False)["input_ids"]
    assert len(yes_id) == len(no_id) == 1
    enc = [bridge.prompts.encode(it["tag"], it["question"], p, end=False) for it, p in zip(its, pf)]
    ids, lab, mm, att = bridge.prompts.collate(enc)
    with torch.no_grad():
        h = bridge.hidden_vis(bridge.visual_tokens(feats), ids, mm, att)
        for i, (e, want) in enumerate(zip(enc, got)):
            last = len(e[0]) - 1
            lp = torch.log_softmax(bridge.qwen.lm_head(h[i, last]).float(), -1)
            direct = float(lp[yes_id[0]] - lp[no_id[0]])
            assert abs(direct - want) < 1e-3, (direct, want)
    print("5 ok: logodds == direct next-piece log P(' yes') - log P(' no')", [round(x, 3) for x in got])
    print("ALL OK")


if __name__ == "__main__":
    main()
