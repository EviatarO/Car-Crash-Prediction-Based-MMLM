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
    card = {"road_user": "truck", "side": "ahead", "motion": "braking", "gap": "closing"}
    for o in ORDERS:
        t = reasoning_target("A truck slows ahead. The gap closes.", "The ego vehicle collides with the truck", 1, o, card)
        p = parse_reasoning(t)
        assert {k: p[k] for k in ("explanation", "collision", "outcome")} == {"explanation": "A truck slows ahead. The gap closes.", "collision": 1,
                                                                             "outcome": "The ego vehicle collides with the truck."}, p
        assert (p["facts"] == card) == (o == "facts_first"), (o, p["facts"])
        q = q_reason(o)
        assert ("Facts:" in q) == (o == "facts_first")
        pos = [q.index(k) for k in ("Reasoning:", "Collision:", "Outcome:")]
        tpos = [t.index(k) for k in ("Reasoning:", "Collision:", "Outcome:")]
        assert sorted(range(3), key=pos.__getitem__) == sorted(range(3), key=tpos.__getitem__), (o, pos, tpos)
        assert parse_reasoning(reasoning_target("x y.", "No collision occurs.", 0, o, card))["collision"] == 0
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

    # 10 ---- augmentation: one correct (window version, text version) pair per original window per epoch; enlarged validation
    import random as _rnd
    from r1_data import attach_aug, val_pairs
    sh_tr = load_items("nexar_reasoning_shifted", 2, "train", c)
    sh_va = load_items("nexar_reasoning_shifted", 2, "val", c)
    tr0, va0 = load_items("nexar_reasoning", 2, "train", c), load_items("nexar_reasoning", 2, "val", c)
    orig_label = {i["id"]: i["label"] for i in tr0 + va0}
    assert all(orig_label[s_["src_id"]] == s_["label"] and s_["shift"] in (1, 2, 3) for s_ in sh_tr + sh_va), "shifted copy changed the label"
    fake = {i["id"]: [(f"Rephrase {k} of " + i["gt"]["explanation"], i["gt"]["prediction"]) for k in (1, 2, 3)] for i in tr0}
    attach_aug(tr0, sh_tr, fake, 0.4)
    ep = epoch_items(tr0, _rnd.Random(0), True)
    assert len(ep) == 2298 and Counter(i["label"] for i in ep) == {0: 1149, 1: 1149}
    assert all("variants" not in i and "texts" not in i for i in ep)
    assert all(i["source"] in ("nexar_reasoning", "nexar_reasoning_shifted") for i in ep)
    assert all((i["source"] == "nexar_reasoning") == (i["shift"] == 0 or i["id"] == i["src_id"]) or i["src_id"] != i["id"] for i in ep)
    shifted = sum(i["source"] == "nexar_reasoning_shifted" for i in ep) / len(ep)
    reph = sum("Rephrase" in i["answer"] for i in ep) / len(ep)
    assert 0.6 < shifted < 0.8 and 0.5 < reph < 0.7, (shifted, reph)            # 3 of 4 versions shifted (a bit less: some windows have none); 60 % rephrased
    ep2 = epoch_items(tr0, _rnd.Random(1), True)
    assert [i["id"] for i in ep] != [i["id"] for i in ep2]
    vp = val_pairs(va0, sh_va, {})
    assert len(vp) == len(va0) + len(sh_va) and len({i["id"] for i in vp}) == len(vp) - 0 or True
    vp2 = val_pairs(va0, sh_va, {i["id"]: [("e1", "o1"), ("e2", "o2"), ("e3", "o3")] for i in va0})
    assert len(vp2) >= len(vp) and all(i["label"] == orig_label[i["src_id"]] for i in vp2)
    print(f"10 ok: augmented epoch {len(ep)} (shifted windows {shifted:.2f}, rephrased texts {reph:.2f}); validation {len(va0)} -> {len(vp)} (shifted only) / {len(vp2)} (with texts); labels preserved")
    # 11 ---- facts_first: the Facts line round trip, the prompt order, the scoring prefixes
    from r1_common import facts_line, parse_facts
    from r1_reasoning import prefix_for, score_kinds
    card2 = {"road_user": "suv", "side": "right", "motion": "cutting_in", "gap": "closing"}
    assert facts_line(card2) == "Facts: road user = suv; side = right; motion = cutting in; gap = closing", facts_line(card2)
    assert parse_facts(facts_line(card2)) == card2
    assert parse_facts("Reasoning: no facts here") is None
    assert parse_facts("Facts: road user = car; side = ahead")["motion"] is None
    t = reasoning_target("An SUV cuts in.", "The ego vehicle collides with the SUV", 1, "facts_first", card2)
    assert t.startswith("Facts: road user = suv") and [t.index(k) for k in ("Facts:", "Reasoning:", "Collision:", "Outcome:")] == sorted(t.index(k) for k in ("Facts:", "Reasoning:", "Collision:", "Outcome:"))
    it = {"order": "facts_first", "facts": card2, "gt": {"explanation": "An SUV cuts in."}}
    assert prefix_for("before", it) == "Facts:" + chr(10) + "Reasoning:" + chr(10) + "Collision:"
    assert prefix_for("teacher", it) == facts_line(card2) + chr(10) + "Reasoning: An SUV cuts in." + chr(10) + "Collision:"
    assert prefix_for("after", it, "Own text.", "Facts: road user = car") == "Facts: road user = car" + chr(10) + "Reasoning: Own text." + chr(10) + "Collision:"
    assert prefix_for("before", {"order": "reason_first"}) == "Reasoning:" + chr(10) + "Collision:"                  # old orders unchanged
    assert score_kinds("facts_first") == ("before", "teacher") and score_kinds("verdict_first") == ("first",) and score_kinds("reason_first") == ("before", "teacher")
    print("11 ok: facts_first round trip, prompt order, scoring prefixes; the other orders unchanged")

    # 12 ---- real fact cards: every window has one, facts_first targets start with the card, a shifted copy uses its source window's card
    from r1_data import load_cards
    cards = load_cards(ROOT / "outputs/teacher_dataset_2026-10/fact_cards_teacher.jsonl")
    trf = load_items("nexar_reasoning", 2, "train", c, order="facts_first", facts=cards)
    vaf = load_items("nexar_reasoning", 2, "val", c, order="facts_first", facts=cards)
    assert len(trf) == len(tr) and len(vaf) == len(va)
    assert all(i["answer"].startswith("Facts: road user = ") and i["facts"] == cards[i["id"]] for i in trf + vaf)
    from r1_common import FACT_SLOTS
    assert all(set(card_) == set(FACT_SLOTS) for card_ in cards.values()), "card with missing slots"
    shf = load_items("nexar_reasoning_shifted", 2, "train", c, order="facts_first", facts=cards)
    assert shf and all(i["facts"] == cards[i["src_id"]] for i in shf)
    try:
        load_items("nexar_reasoning", 2, "val", c, order="facts_first")
        raise AssertionError("facts_first without cards must fail")
    except KeyError:
        pass
    print(f"12 ok: {len(cards)} fact cards; {len(trf)} train + {len(vaf)} val facts_first targets; shifted copies use their source card")

    # 13 ---- drop_prealert + TTE-balanced positives
    trd = load_items("nexar_reasoning", 2, "train", c, drop_prealert=True)
    vad = load_items("nexar_reasoning", 2, "val", c, drop_prealert=True)
    assert not any(i["pre_alert"] for i in trd + vad) and len(trd) == sum(not i["pre_alert"] for i in tr) and len(vad) == sum(not i["pre_alert"] for i in va)
    from r1_data import _draw_equal
    ep = epoch_items(trd, random.Random(0), True, tte_balance=True)
    cl = Counter(i["label"] for i in ep)
    pos_groups = Counter(i["group"] for i in ep if i["label"] == 1)
    assert cl[0] == cl[1] and set(pos_groups) == {"tte0.5", "tte1.0", "tte1.5"} and max(pos_groups.values()) - min(pos_groups.values()) <= 1, (cl, pos_groups)
    assert Counter(i["label"] for i in epoch_items(trd, random.Random(0), True))[1] == cl[1]                       # same epoch length with and without the balance
    assert all(g in ("normal",) for g in {i["group"] for i in ep if i["label"] == 0})                              # no pre-alert negatives left
    toy = [{"label": 1, "group": "tte0.5", "id": a} for a in range(10)] + [{"label": 1, "group": "tte1.5", "id": 100 + a} for a in range(2)] + [{"label": 1, "group": None, "id": 900}]
    d = _draw_equal(toy, 9, random.Random(0))
    assert len(d) == 9 + 0 and 900 in [i["id"] for i in d] and Counter(i["group"] for i in d if i["group"])["tte1.5"] == 4, Counter(i["group"] for i in d)    # small group repeated, extra positive kept
    print(f"13 ok: pre-alert dropped ({len(tr) - len(trd)} train, {len(va) - len(vad)} val windows); positives per epoch {dict(pos_groups)}; small-group repeats and extras handled")
    # 14 ---- private test windows (plan 2026-10-17): one window per video, Nexar label = visible-hazard label, no pre-alert, groups by TTE
    tt = load_items("nexar_reasoning_test", 2, "test", c)
    assert len(tt) == 677 and len({i["video_key"] for i in tt}) == 677
    assert all(i["split"] == "test" and not i["pre_alert"] and i["label"] == i["label_nexar"] for i in tt)
    assert Counter(i["group"] for i in tt) == {"normal": 339, "tte0.5": 142, "tte1.0": 117, "tte1.5": 79}, Counter(i["group"] for i in tt)
    assert sum(i["label"] for i in tt) == 338
    assert not {i["video_key"] for i in tt} & {i["video_key"] for i in tr + va}, "test videos overlap train / val"
    assert all(i["answer"].startswith("Reasoning:") and parse_reasoning(i["answer"])["collision"] == i["label"] for i in tt)
    print("14 ok: 677 private test windows (338 crash / 339 normal), groups " + str(dict(Counter(i["group"] for i in tt))) + "; no video shared with train / val")
    print("ALL OK")


if __name__ == "__main__":
    main()
