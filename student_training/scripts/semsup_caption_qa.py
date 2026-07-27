"""
semsup_caption_qa.py
=====================
Gate 0 (token compliance + content validity) for the prompt-bakeoff captions,
and arm_a/arm_b/arm_c construction. See PLAN: prompt-bakeoff-harness (2026-07-27).

Two modes, auto-detected from the input file's schema:

  NEW pipeline  (row has "caption_neutral"): the raw output of captioning the
    sampled manifest with prompts/PROMPT_SEMSUP_V2.py. Runs full Gate 0 +
    builds arm_a.jsonl / arm_b.jsonl / arm_c.jsonl.

  LEGACY self-test  (row has "caption", no "caption_neutral"): the existing
    outputs/semantic_captions/Caption_Train_All_Clips.jsonl. Runs the same
    token-count and verdict-leakage logic as a calibration check (plan
    verification step 2) - expected to reproduce 0% truncation and flag the
    known 267/267 verdict-leakage artifact. Builds nothing (nothing to build).

Why the banned-word / duplicate / leakage checks exist (not just style advice):
  - Banned words in caption_neutral would make Arm A and Arm B the SAME text,
    destroying the one contrast (risk-clause present vs absent) the whole
    prompt-bakeoff experiment is designed to measure.
  - Duplicate sentences collapse embeddings toward the mean (the exact collapse
    mechanism that broke the original cosine-regression B1 run - see
    docs_agents/EXPERIMENTS.md's A-1 finding).
  - >=99% verdict-vs-GT agreement is the known signature of the teacher being
    told (directly or via TTE) that an event exists - see EXPERIMENTS.md's
    documentation-consistency finding on the incumbent 267-caption set.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "student_training" / "scripts"))

BANNED_WORDS = ("risk", "danger", "collision", "crash", "imminent", "safe",
                "avoid", "impact", "hazard", "accident")
BANNED_RE = re.compile(r"\b(" + "|".join(BANNED_WORDS) + r")\b", re.IGNORECASE)

MAX_TOKENS = 64
TOKEN_FAIL_FRACTION = 0.05  # hard-fail if more than 5% of rows exceed MAX_TOKENS


def _norm_verdict(v):
    if v is None:
        return None
    s = str(v).strip().upper()
    if s in ("1", "YES", "TRUE"):
        return 1
    if s in ("0", "NO", "FALSE"):
        return 0
    return None


def _normalize_for_dup(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()


def token_report(texts: list, tok, label: str) -> dict:
    lens = [len(tok(t)["input_ids"]) for t in texts]
    over = [(t, n) for t, n in zip(texts, lens) if n > MAX_TOKENS]
    frac_over = len(over) / len(lens) if lens else 0.0
    lens_sorted = sorted(lens)
    n = len(lens_sorted)
    print(f"  [{label}] n={n}  tokens: min={lens_sorted[0] if n else 0} "
          f"median={lens_sorted[n // 2] if n else 0} max={lens_sorted[-1] if n else 0}  "
          f"over-{MAX_TOKENS}={len(over)} ({frac_over:.1%})")
    if over:
        worst_text, worst_n = max(over, key=lambda x: x[1])
        print(f"    worst offender ({worst_n} tokens): {worst_text[:100]}")
    return {"label": label, "n": n, "min": lens_sorted[0] if n else 0,
            "median": lens_sorted[n // 2] if n else 0, "max": lens_sorted[-1] if n else 0,
            "n_over_64": len(over), "frac_over_64": frac_over,
            "PASS": frac_over <= TOKEN_FAIL_FRACTION}


def banned_word_report(texts_with_ids: list) -> dict:
    hits = [(vid, t) for vid, t in texts_with_ids if BANNED_RE.search(t)]
    print(f"  [banned-word check on caption_neutral] {len(hits)} / {len(texts_with_ids)} rows flagged")
    for vid, t in hits[:10]:
        matched = BANNED_RE.findall(t)
        print(f"    {vid}: contains {matched} -> {t[:90]}")
    if len(hits) > 10:
        print(f"    ... and {len(hits) - 10} more")
    return {"n_flagged": len(hits), "flagged_ids": [vid for vid, _ in hits]}


def duplicate_report(texts_with_ids: list) -> dict:
    groups: dict = {}
    for vid, t in texts_with_ids:
        groups.setdefault(_normalize_for_dup(t), []).append(vid)
    dups = {k: v for k, v in groups.items() if len(v) > 1}
    n_dup_rows = sum(len(v) for v in dups.values())
    print(f"  [duplicate-sentence check] {len(dups)} distinct sentences reused across "
          f"{n_dup_rows} rows")
    for sent, vids in list(dups.items())[:5]:
        print(f"    {vids}: \"{sent[:80]}\"")
    return {"n_duplicate_groups": len(dups), "n_duplicate_rows": n_dup_rows}


def verdict_leakage_report(pairs: list) -> dict:
    """pairs: list of (predicted_verdict 0/1, gt 0/1)."""
    n = len(pairs)
    n_match = sum(1 for p, g in pairs if p == g)
    neg_pairs = [(p, g) for p, g in pairs if g == 0]
    n_neg = len(neg_pairs)
    n_neg_match = sum(1 for p, g in neg_pairs if p == g)
    overall = n_match / n if n else 0.0
    neg_acc = n_neg_match / n_neg if n_neg else float("nan")
    print(f"  [verdict QA] overall agreement: {n_match}/{n} ({overall:.1%})   "
          f"negatives-only: {n_neg_match}/{n_neg} ({neg_acc:.1%})")
    suspected_leak = overall >= 0.99
    if suspected_leak:
        print("  *** SUSPECTED PROMPT LEAKAGE: >=99% verdict agreement. The teacher is very "
              "likely inferring the answer from something other than the 16 frames alone "
              "(e.g. TTE mentioned in context, or a prior-generation caption reused). "
              "This is NOT being treated as success. ***")
    return {"n": n, "overall_agreement": overall, "negatives_only_agreement": neg_acc,
            "suspected_leakage": suspected_leak}


def run_legacy_selftest(path: Path, tok):
    print(f"\n=== LEGACY SELF-TEST mode: {path.name} ===")
    print("(calibration check against the known incumbent 267-caption set - see plan "
          "verification step 2. Builds nothing.)\n")
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    print(f"loaded {len(rows)} rows")

    captions = [r["caption"] for r in rows]
    tr = token_report(captions, tok, "caption (legacy single-field)")

    pairs = []
    for r in rows:
        gt = _norm_verdict(r.get("gt_verdict"))
        fv = _norm_verdict(r.get("final_verdict"))
        if gt is not None and fv is not None:
            pairs.append((fv, gt))
    vr = verdict_leakage_report(pairs)

    print("\n--- calibration expectations ---")
    print(f"  expected: 0% over-64-token  -> got {tr['frac_over_64']:.1%} "
          f"{'OK' if tr['frac_over_64'] == 0.0 else 'MISMATCH'}")
    print(f"  expected: >=99% leakage flag fires -> "
          f"{'FIRED (as expected)' if vr['suspected_leakage'] else 'DID NOT FIRE (unexpected!)'}")


def run_new_pipeline(raw_path: Path, manifest_path: Path, out_dir: Path, tok):
    print(f"\n=== NEW PIPELINE mode: {raw_path.name} + {manifest_path.name} ===\n")
    raw_rows = [json.loads(l) for l in open(raw_path, encoding="utf-8") if l.strip()]
    manifest_rows = [json.loads(l) for l in open(manifest_path, encoding="utf-8") if l.strip()]
    manifest_by_vid = {r["video_id"]: r for r in manifest_rows}

    missing = [r["video_id"] for r in raw_rows if r["video_id"] not in manifest_by_vid]
    if missing:
        print(f"WARNING: {len(missing)} raw-caption rows have no matching manifest row "
              f"(dropped): {missing[:10]}")
    joined = [(r, manifest_by_vid[r["video_id"]]) for r in raw_rows if r["video_id"] in manifest_by_vid]
    print(f"joined {len(joined)} / {len(raw_rows)} raw rows against the manifest")

    # Gate 0: token compliance
    print("\n--- Gate 0: token compliance ---")
    neutral_texts = [r["caption_neutral"] for r, _ in joined]
    combined_texts = [r["caption_neutral"] + ", " + r["risk_clause"] for r, _ in joined]
    tr_a = token_report(neutral_texts, tok, "arm_a (caption_neutral)")
    tr_b = token_report(combined_texts, tok, "arm_b (caption_neutral + risk_clause)")
    gate0_pass = tr_a["PASS"] and tr_b["PASS"]
    print(f"  Gate 0: {'PASS' if gate0_pass else 'FAIL'}")

    # banned-word + duplicate checks on caption_neutral only
    print("\n--- content validity (caption_neutral) ---")
    ids_texts = [(r["video_id"], r["caption_neutral"]) for r, _ in joined]
    bw = banned_word_report(ids_texts)
    dup = duplicate_report(ids_texts)

    # verdict QA (sanity only)
    print("\n--- verdict QA (sanity only, never used for arm selection) ---")
    pairs = [(_norm_verdict(r.get("verdict")), m["target"]) for r, m in joined
             if _norm_verdict(r.get("verdict")) is not None]
    vr = verdict_leakage_report(pairs)

    # build arms
    out_dir.mkdir(parents=True, exist_ok=True)
    tte_bucket_to_seconds = {"TTE_0.5": 0.5, "TTE_1.0": 1.0, "TTE_1.5": 1.5}

    def arm_c_text(target: int, horizon_label: str) -> str:
        if target == 1:
            secs = tte_bucket_to_seconds.get(horizon_label, "an unknown number of")
            return f"collision imminent in {secs} seconds"
        return "normal driving, no collision"

    n_written = {"a": 0, "b": 0, "c": 0}
    files = {k: open(out_dir / f"arm_{k}.jsonl", "w", encoding="utf-8") for k in ("a", "b", "c")}
    try:
        for r, m in joined:
            gt_verdict = "YES" if m["target"] == 1 else "NO"
            base = {
                "video_id": r["video_id"],
                "requested_time_to_event": m["requested_time_to_event"],
                "gt_verdict": gt_verdict,
                "frames_dir": m["frames_dir"],
            }
            files["a"].write(json.dumps({**base, "caption": r["caption_neutral"]},
                                         ensure_ascii=False) + "\n")
            files["b"].write(json.dumps(
                {**base, "caption": r["caption_neutral"] + ", " + r["risk_clause"]},
                ensure_ascii=False) + "\n")
            files["c"].write(json.dumps(
                {**base, "caption": arm_c_text(m["target"], m["horizon_label"])},
                ensure_ascii=False) + "\n")
            for k in n_written:
                n_written[k] += 1
    finally:
        for f in files.values():
            f.close()
    print(f"\nwrote arm_a.jsonl / arm_b.jsonl / arm_c.jsonl ({n_written['a']} rows each) -> {out_dir}")

    return {
        "n_joined": len(joined), "gate0_pass": gate0_pass,
        "token_report_a": tr_a, "token_report_b": tr_b,
        "banned_word": bw, "duplicate": dup, "verdict_qa": vr,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True,
                     help="raw teacher-output JSONL (new schema) OR the legacy "
                          "Caption_Train_All_Clips.jsonl (self-test mode)")
    ap.add_argument("--manifest", default=None,
                     help="required in new-pipeline mode: the sampled clip manifest "
                          "from semsup_sample_clips.py")
    ap.add_argument("--out-dir", default=str(PROJECT_ROOT / "outputs" / "semantic_captions" / "promptbakeoff"))
    ap.add_argument("--siglip-model", default="google/siglip-base-patch16-224")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.siglip_model)

    input_path = Path(args.input)
    first_row = json.loads(open(input_path, encoding="utf-8").readline())
    is_new_pipeline = "caption_neutral" in first_row

    if is_new_pipeline:
        if not args.manifest:
            raise SystemExit("--manifest is required in new-pipeline mode (row has 'caption_neutral')")
        run_new_pipeline(input_path, Path(args.manifest), Path(args.out_dir), tok)
    else:
        run_legacy_selftest(input_path, tok)


if __name__ == "__main__":
    main()
