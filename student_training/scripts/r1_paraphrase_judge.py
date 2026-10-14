"""
r1_paraphrase_judge.py - independent judge of the rephrased reasoning versions (Claude Sonnet 5.5 on OpenRouter; the writer is Gemini / Claude agents).

Every version that passed the automatic check (paraphrases.jsonl, r1_paraphrase_check.py) is compared with the teacher's original by the judge, with a
strict rubric: same road user, side, motion / state, gap trend; nothing added (colour, object, place, cause, intensity, collision type), nothing dropped;
outcome keeps its meaning; explanation does not announce the outcome. ANY doubt = reject.

  judge    : judge every not-yet-judged version -> paraphrase_judge.jsonl (one row per version: key, id, ok, problem); resumable
  finalize : paraphrases_judged.jsonl = up to 3 versions per window that passed BOTH (sources renumbered rephrased_v1..v3), training input;
             paraphrase_good_counts.json (for the rewrite round), paraphrase_judge_rejects.jsonl (hints for the rewrite)
  sample   : stratified audit sample for the Opus audit -> paraphrase_audit_sample.jsonl

    python r1_paraphrase_judge.py judge --concurrency 16
    python r1_paraphrase_judge.py finalize
    python r1_paraphrase_judge.py sample --n 300
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
D = ROOT / "outputs" / "teacher_dataset_2026-10"
ACC, JUDGE = D / "paraphrases.jsonl", D / "paraphrase_judge.jsonl"
BATCHES = D / "paraphrase_batches"
IN_PER_M, OUT_PER_M = 2.0, 10.0                         # anthropic/claude-sonnet-5.5 list price, USD per million tokens

RUBRIC = """You check rewordings of short dashcam-scene descriptions for factual faithfulness. The ORIGINAL was written by someone who watched the video; you and the rewriter did not. A REWORDING must state exactly the same facts in different words. Be strict: when in doubt, reject.

Reject (ok=false) if the rewording, compared with its original:
1. changes or adds a road user or object kind (a synonym of the same kind is fine: sedan = car, SUV = sport utility vehicle; car -> truck is not);
2. changes or drops the position / side (ahead, same lane, left, right, adjacent lane, oncoming, crossing, shoulder ...). Equivalent wording is fine ("in front" = "ahead");
3. changes the motion or state (stopped -> parked, slows -> brakes hard, drifts -> swerves, crossing -> turning), or adds a motion the original does not state;
4. changes the gap trend (closing / widening / stable) or its strength (gradually -> rapidly, or the reverse);
5. adds any detail absent from the original: colour, place, road feature, weather, light, traffic signal, cause, intention, a number, a collision type ("rear-end") ...;
6. drops a fact the original states (a colour, the place, a second road user, the motion, the gap trend);
7. the OUTCOME changes meaning (collision <-> no collision, or a different object is involved);
8. the EXPLANATION already says that a collision happens (that belongs only in the outcome).
Pure style changes (sentence order, voice, synonyms with the same meaning, splitting sentences) are fine.

Reply with ONLY JSON lines, one per item, in the same order, nothing else:
{"key": "<key>", "ok": true}
{"key": "<key>", "ok": false, "problem": "<at most 12 words: rule number and what changed>"}"""


def key_of(r):
    return hashlib.sha1(f'{r["id"]}|{r["explanation"]}|{r["outcome"]}'.encode("utf-8")).hexdigest()[:12]


def originals():
    return {json.loads(l)["id"]: json.loads(l) for p in sorted(BATCHES.glob("batch_*.jsonl")) for l in open(p, encoding="utf-8")}


def judged():
    out = {}
    if JUDGE.exists():
        for l in open(JUDGE, encoding="utf-8"):
            try:
                x = json.loads(l)
                out[x["key"]] = x
            except Exception:                                                      # noqa: BLE001
                pass
    return out


def user_msg(chunk, orig):
    parts = []
    for r in chunk:
        o = orig[r["id"]]
        parts.append(f'key: {key_of(r)}\nORIGINAL EXPLANATION: {o["explanation"]}\nORIGINAL OUTCOME: {o["outcome"]}\n'
                     f'REWORDED EXPLANATION: {r["explanation"]}\nREWORDED OUTCOME: {r["outcome"]}')
    return "Items:\n\n" + "\n\n".join(parts)


def call(client, model, msg, timeout, retries):
    last = None
    for a in range(1, retries + 1):
        try:
            r = client.chat.completions.create(model=model, temperature=0, timeout=timeout, max_tokens=2000,
                                               messages=[{"role": "system", "content": RUBRIC}, {"role": "user", "content": msg}])
            u = r.usage.model_dump() if getattr(r, "usage", None) else {}
            return r.choices[0].message.content or "", u, None
        except Exception as e:                                                     # noqa: BLE001
            last = f"{type(e).__name__}: {str(e)[:200]}"
            time.sleep(min(60, 4 * a * a))
    return "", {}, last


def parse(txt, keys):
    out = {}
    for l in txt.splitlines():
        l = l.strip().strip("`")
        if not l.startswith("{"):
            continue
        try:
            x = json.loads(l)
        except json.JSONDecodeError:
            continue
        if x.get("key") in keys and isinstance(x.get("ok"), bool):
            out[x["key"]] = {"ok": x["ok"], "problem": str(x.get("problem", ""))[:200]}
    return out


def judge(a):
    orig, done = originals(), judged()
    todo = [r for r in map(json.loads, open(ACC, encoding="utf-8")) if key_of(r) not in done]
    if a.limit:
        todo = todo[:a.limit]
    chunks = [todo[i:i + a.per_request] for i in range(0, len(todo), a.per_request)]
    est_in = sum(len(RUBRIC) / 4 + len(user_msg(c, orig)) / 4 for c in chunks)
    print(f"[judge] versions to judge {len(todo)} ({len(done)} already judged); requests {len(chunks)}; est cost ${(est_in * IN_PER_M + len(todo) * 25 * OUT_PER_M) / 1e6:.2f}",
          flush=True)
    if a.dry_run or not chunks:
        if chunks:
            print("\n===== SYSTEM =====\n" + RUBRIC + "\n\n===== USER =====\n" + user_msg(chunks[0][:2], orig))
        return
    from dotenv import load_dotenv
    from openai import OpenAI
    load_dotenv(ROOT / ".env")
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])
    cost, n_ok, n_bad, miss, t0 = 0.0, 0, 0, 0, time.time()
    with open(JUDGE, "a", encoding="utf-8") as f, ThreadPoolExecutor(a.concurrency) as pool:
        futs = {pool.submit(call, client, a.model, user_msg(c, orig), a.timeout, a.retries): c for c in chunks}
        for k, fu in enumerate(as_completed(futs), 1):
            c = futs[fu]
            txt, u, err = fu.result()
            cost += float(u.get("cost") or 0)
            res = parse(txt, {key_of(r) for r in c})
            for r in c:
                kk = key_of(r)
                if kk in res:
                    f.write(json.dumps({"key": kk, "id": r["id"], **res[kk], "model": a.model}, ensure_ascii=False) + "\n")
                    n_ok += res[kk]["ok"]
                    n_bad += not res[kk]["ok"]
                else:
                    miss += 1
            f.flush()
            if k % 20 == 0 or k == len(chunks):
                print(f"[judge] {k}/{len(chunks)} requests; ok {n_ok}, rejected {n_bad} ({n_bad / max(1, n_ok + n_bad):.1%}), unanswered {miss}; "
                      f"cost ${cost:.2f}; {time.time() - t0:.0f}s" + (f"; last error {err}" if err else ""), flush=True)
    print(f"[judge] done: ok {n_ok}, rejected {n_bad}, unanswered {miss} (rerun judges them); cost ${cost:.2f}")


def finalize(a):
    orig, jd = originals(), judged()
    acc = list(map(json.loads, open(ACC, encoding="utf-8")))
    good, rej, unjudged = {}, [], 0
    for r in acc:
        j = jd.get(key_of(r))
        if j is None:
            unjudged += 1
            continue
        if j["ok"]:
            good.setdefault(r["id"], []).append(r)
        else:
            rej.append({"id": r["id"], "explanation": r["explanation"], "outcome": r["outcome"], "problem": j["problem"]})
    out = []
    for i, rs in good.items():
        for k, r in enumerate(rs[:3], 1):
            out.append({"id": i, "source": f"rephrased_v{k}", "explanation": r["explanation"], "outcome": r["outcome"], "judge": "ok"})
    (D / "paraphrases_judged.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out), encoding="utf-8")
    (D / "paraphrase_judge_rejects.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rej), encoding="utf-8")
    counts = {i: min(3, len(good.get(i, []))) for i in orig}
    (D / "paraphrase_good_counts.json").write_text(json.dumps(counts), encoding="utf-8")
    c = Counter(counts.values())
    probs = Counter((r["problem"].split(":")[0].split(" ")[0] if r["problem"] else "?") for r in rej)
    print(f"[finalize] versions after auto check {len(acc)}: judged ok {sum(len(v) for v in good.values())}, judge-rejected {len(rej)} "
          f"({len(rej) / max(1, len(acc) - unjudged):.1%}), not judged yet {unjudged}")
    print(f"           windows by number of good versions (0/1/2/3): {c[0]}/{c[1]}/{c[2]}/{c[3]}; written {len(out)} versions -> paraphrases_judged.jsonl")
    print("           judge reject reasons (first word):", dict(probs.most_common(10)))


def sample(a):
    """Stratified audit sample for Opus: judge-accepted versions (the risky side: a miss there goes into training) and judge-rejected ones."""
    orig, jd = originals(), judged()
    acc = list(map(json.loads, open(ACC, encoding="utf-8")))
    rng = random.Random(0)
    okv = [r for r in acc if jd.get(key_of(r), {}).get("ok") is True]
    bad = [r for r in acc if jd.get(key_of(r), {}).get("ok") is False]
    n_bad = min(len(bad), a.n // 6)
    pick = rng.sample(okv, min(len(okv), a.n - n_bad)) + rng.sample(bad, n_bad)
    rng.shuffle(pick)
    rows = [{"key": key_of(r), "id": r["id"], "original_explanation": orig[r["id"]]["explanation"], "original_outcome": orig[r["id"]]["outcome"],
             "explanation": r["explanation"], "outcome": r["outcome"], "judge_ok": jd[key_of(r)]["ok"], "judge_problem": jd[key_of(r)].get("problem", "")}
            for r in pick]
    (D / "paraphrase_audit_sample.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(f"[sample] {len(rows)} versions ({len(rows) - n_bad} judge-accepted, {n_bad} judge-rejected) -> paraphrase_audit_sample.jsonl")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["judge", "finalize", "sample"])
    ap.add_argument("--model", default="anthropic/claude-sonnet-5.5")
    ap.add_argument("--per-request", type=int, default=10)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--timeout", type=float, default=180)
    ap.add_argument("--retries", type=int, default=4)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    {"judge": judge, "finalize": finalize, "sample": sample}[a.cmd](a)
