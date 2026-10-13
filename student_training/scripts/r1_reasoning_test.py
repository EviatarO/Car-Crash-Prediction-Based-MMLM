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

    # 6 ---- merger input types: shapes, original parameters unchanged, extra channels matter
    import tempfile
    import numpy as np
    from r1_bridge import INPUTS, Merger
    for name, (nl, at) in INPUTS.items():
        m = Merger(1024, 64, nl, at)
        C = nl * 1024 + int(at)
        x = torch.randn(2, 2048, C)
        y = m(x)
        assert y.shape == (2, 512, 64), (name, y.shape)
    base = Merger(1024, 64)
    assert set(base.state_dict()) == {"norm.weight", "norm.bias", "linear_fc1.weight", "linear_fc1.bias", "linear_fc2.weight", "linear_fc2.bias"}
    m2 = Merger(1024, 64, 1, True)
    with torch.no_grad():
        m2.fuse.weight[:, 1024] = 0.5                     # identity start ignores the attention channel; check that the channel is wired
    x = torch.randn(1, 2048, 1025)
    x2 = x.clone()
    x2[..., 1024] += 3.0
    assert not torch.allclose(m2(x), m2(x2)), "attention channel has no effect"
    # identity start: with the same norm / fc weights, every extended merger equals the original one on the last-layer part
    torch.manual_seed(1)
    ref = Merger(1024, 64)
    xl = torch.randn(2, 2048, 1024)
    for name, (nl, at) in INPUTS.items():
        if name == "last":
            continue
        m = Merger(1024, 64, nl, at)
        m.load_state_dict({**m.state_dict(), **ref.state_dict()})
        extra_l11 = torch.randn(2, 2048, 1024) if nl == 2 else None
        parts = ([extra_l11] if nl == 2 else []) + [xl] + ([torch.rand(2, 2048, 1)] if at else [])
        assert torch.allclose(m(torch.cat(parts, -1)), ref(xl), atol=1e-5), name
        w = m.fuse.weight
        k = 1024 if nl == 2 else 0
        assert torch.equal(w[:, k:k + 1024], torch.eye(1024)) and float(w.abs().sum()) == 1024.0
    print("6 ok: merger shapes for", list(INPUTS), "; original parameters unchanged; attention channel acts; extended mergers start identical to the original")

    # 7 ---- Cache returns the right channel layout for each input
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        n = 3
        rng = np.random.default_rng(0)
        last, l11 = rng.standard_normal((n, 2048, 1024)).astype(np.float16), rng.standard_normal((n, 2048, 1024)).astype(np.float16)
        att = np.abs(rng.standard_normal((n, 2048))).astype(np.float16) + 0.01
        np.save(td / "features_nexar_reasoning.npy", last)
        np.save(td / "features_nexar_reasoning_l11.npy", l11)
        np.save(td / "attn_nexar_reasoning.npy", att)
        (td / "index_nexar_reasoning.jsonl").write_text("".join(json.dumps({"id": f"w{i}", "row": i, "p_collision": 0.5}) + chr(10) for i in range(n)))
        from r1_data import Cache
        for name, C in (("last", 1024), ("last+attn", 1025), ("l11+last", 2048), ("l11+last+attn", 2049)):
            g = Cache(td, "nexar_reasoning", name).get(["w2", "w0"])
            assert g.shape == (2, 2048, C), (name, g.shape)
        g = Cache(td, "nexar_reasoning", "l11+last+attn").get(["w1"])
        assert torch.allclose(g[0, :, :1024], torch.from_numpy(l11[1]).float()) and torch.allclose(g[0, :, 1024:2048], torch.from_numpy(last[1]).float())
        assert abs(float(torch.expm1(g[0, :, 2048]).mean()) - 1.0) < 1e-3, "attention not scaled to mean 1"
    print("7 ok: Cache channel layout [l11 | last | attn] and attention scaling")

    # 8 ---- fact extractor spot checks
    from r1_fact_eval import facts
    f = facts("A white SUV directly ahead in the same lane brakes hard. The gap closes rapidly.")
    assert f == {"type": "suv", "side": "ahead", "motion": "braking", "gap": "closing"}, f
    f = facts("A pickup truck from the left lane merges into the ego lane while the distance decreases.")
    assert f["side"] == "left" and f["motion"] == "cutting_in" and f["gap"] == "closing", f
    assert facts("A car is parked along the right curb. The ego vehicle passes it, keeping a steady distance.") ==         {"type": "car", "side": "right", "motion": "parked", "gap": "stable"}
    print("8 ok: fact extractor")

    # 9 ---- probe learns a separable synthetic fact and stays near chance on noise
    from r1_fact_probe import bal_acc, fit_probe
    g = torch.Generator().manual_seed(0)
    y = torch.randint(0, 3, (900,), generator=g)
    sig = torch.randn(3, 512, generator=g)
    X = torch.randn(900, 512, generator=g) * 2 + sig[y]
    net, f = fit_probe(X[:600], y[:600], X[600:750], y[600:750], 3, torch.device("cpu"))
    acc = bal_acc(y[750:].numpy(), net(f(X[750:])).argmax(1).numpy(), 3)
    Xn = torch.randn(900, 512, generator=g)
    net2, f2 = fit_probe(Xn[:600], y[:600], Xn[600:750], y[600:750], 3, torch.device("cpu"))
    acc_n = bal_acc(y[750:].numpy(), net2(f2(Xn[750:])).argmax(1).numpy(), 3)
    assert acc > 0.85 and acc_n < 0.5, (acc, acc_n)
    print(f"9 ok: probe on a separable fact {acc:.2f}, on noise {acc_n:.2f} (chance 0.33)")
    print("ALL OK")


if __name__ == "__main__":
    main()
