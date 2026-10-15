"""
r1_cards_review_page.py - review pages for the fact cards (plan 2026-10-15, review points A and B). Writes static HTML into
outputs/teacher_dataset_2026-10/ (open it through the website server).

  A : 30 random P1aug validation windows. For each: the teacher's text and the model's text, the card Gemini read from each, and what the old
      word-list grader read from the same text. Shows where the two readers disagree (the reason for replacing the word list).
  B : 10 random TEACHER cards next to the text and the frames (training will use these cards as the Facts line).

    python r1_cards_review_page.py A|B [--seed 0]
"""
from __future__ import annotations

import argparse
import html
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from r1_fact_eval import facts  # noqa: E402

D = ROOT / "outputs" / "teacher_dataset_2026-10"
P1A = ROOT / "outputs" / "r1_phase1aug_2026-10-14"
SLOTS = ["road_user", "side", "motion", "gap"]
OLD = {"road_user": "type", "side": "side", "motion": "motion", "gap": "gap"}
CSS = """body{background:#0e1222;color:#e6ebf5;font:15px/1.5 system-ui,sans-serif;margin:0;padding:16px;max-width:1250px}h1{font-size:20px}.card{border:1px solid #2a3560;border-radius:10px;
padding:12px 14px;margin:14px 0;background:#121831}h3{margin:0 0 6px;font-size:15px;font-family:monospace}.m{color:#a8b3cc;font-size:13.5px}img{width:100%;max-width:1100px;border-radius:6px;display:block;margin:6px 0}
table{border-collapse:collapse;margin:6px 0}td,th{border:1px solid #2a3560;padding:3px 9px;font-size:13.5px;text-align:left}th{color:#7fd1ff}.dis{background:#3a1d28}.ok{color:#8fe3a8}b{color:#fff}.t{margin:4px 0}"""


def esc(s):
    return html.escape(str(s))


def card_table(rows):
    """rows: [(label, card dict or None)] -> table (one row per reader)"""
    h = "<table><tr><th>reader</th>" + "".join(f"<th>{s}</th>" for s in SLOTS) + "</tr>"
    for lab, c in rows:
        h += f"<tr><td>{esc(lab)}</td>" + "".join(f"<td>{esc((c or {}).get(s, '-'))}</td>" for s in SLOTS) + "</tr>"
    return h + "</table>"


def old_card(text):
    f = facts(text or "")
    return {"road_user": f["type"] or "not_stated", "side": f["side"] or "not_stated", "motion": f["motion"] or "not_stated", "gap": f["gap"] or "not_stated"}


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def page(title, intro, cards):
    return (f"<!doctype html><html><head><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'><title>{esc(title)}</title><style>{CSS}</style></head>"
            f"<body><h1>{esc(title)}</h1><p class=m>{intro}</p>{''.join(cards)}</body></html>")


def page_a(seed):
    preds = {r["id"]: r for r in jl(P1A / "phase1aug_2026-10-14" / "P1aug" / "results" / "predictions.jsonl")}
    mc = {r["key"]: r for r in jl(P1A / "cards_P1aug.jsonl")}
    tc = {r["key"]: r for r in jl(D / "fact_cards_teacher.jsonl")}
    ids = [i for i in preds if i in mc and i in tc]
    random.Random(seed).shuffle(ids)
    cards = []
    n_dis_old = n_dis_new = 0
    for i in ids[:30]:
        p = preds[i]
        new_t, new_m, old_t, old_m = tc[i], mc[i], old_card(p["teacher_explanation"]), old_card(p["gen_explanation"])
        cards.append(f"""<div class=card><h3>{esc(i)} <span class=m>group {esc(p['group'])} · crash model P = {p['a1_p']:.2f} · P1aug verdict {'yes' if p['gen_collision'] == 1 else 'no'} · label {'yes' if p['label'] else 'no'}</span></h3>
<div class=t><b>Teacher:</b> {esc(p['teacher_explanation'])}</div>{card_table([('Gemini reads teacher', new_t), ('old word list reads teacher', old_t)])}
<div class=t><b>P1aug wrote:</b> {esc(p['gen_explanation'])}</div>{card_table([('Gemini reads model text', new_m), ('old word list reads model text', old_m)])}
<p class=m>Do you agree with Gemini's two cards? (road user / side / motion of the MAIN road user / gap)</p></div>""")
    return page("Review A: fact cards on 30 random P1aug validation windows",
                "For each window: the same four slots read two ways, by Gemini (closed lists, reads the text) and by the old word list. "
                "Judge which reader is right, and whether Gemini's cards are acceptable for grading. 'not_stated' = the text does not say.", cards)


def page_b(seed):
    tc = {r["key"]: r for r in jl(D / "fact_cards_teacher.jsonl")}
    man = {json.loads(l)["id"]: json.loads(l) for l in open(ROOT / "dataset" / "manifests" / "r1_nexar_reasoning_windows.jsonl", encoding="utf-8")}
    ids = [i for i in tc if i in man and man[i]["split"] == "train"]
    random.Random(seed).shuffle(ids)
    cards = []
    for i in ids[:10]:
        m = man[i]
        fd = m["frames_dir"]
        cards.append(f"""<div class=card><h3>{esc(i)} <span class=m>train · TTE {esc(m['tte'])} · {'crash' if m['label'] else 'normal'} · verdict {'yes' if m['gt']['label_vis'] else 'no'}</span></h3>
<a href="grid_large/{esc(fd)}.jpg" target=_blank><img loading=lazy src="grid_large/{esc(fd)}.jpg"></a>
<div class=t><b>Teacher:</b> {esc(m['gt']['explanation'])} &rarr; {esc(m['gt']['prediction'])}</div>{card_table([('card (the Facts line training will use)', tc[i])])}</div>""")
    return page("Review B: 10 random teacher fact cards (training windows)", "Frames (16, left-to-right, top-to-bottom; red box = the road user the crash model attends to), the teacher text, and the card. Is the card what the text says?", cards)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=["A", "B"])
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = D / ("cards_review_A.html" if a.which == "A" else "cards_review_B.html")
    out.write_text(page_a(a.seed) if a.which == "A" else page_b(a.seed), encoding="utf-8")
    print(out)
