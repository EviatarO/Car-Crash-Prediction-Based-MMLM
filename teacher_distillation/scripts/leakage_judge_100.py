"""leakage_judge_100.py -- re-run the V12 leakage judge at n=100 (18 val clips +
82 freshly-sampled distinct clips from the 4,446-window pool) instead of n=18.

WHY: at n=18, V12's leakage-judge result (12/18=66.7%) was NOT statistically
distinguishable from chance (one-sided exact binomial P(X>=12|n=18,p=0.5)~=0.12).
n=100 (50 pos/50 neg) narrows the CI enough to actually decide whether V12 still
leaks the label, using the exact same judge mechanic as score_val18_neutral.py
(fresh context, captions only, no images) -- just generalized to >26 items via
leakage_judge_100's run_leakage_judge(), which now uses zero-padded numeric IDs
instead of A-Z letters (the old scheme silently caps at 26).

Prerequisite: run these two captioning passes first (see the 2026-08-11 plan) --
  1. student_training/scripts/sample_val_check_clips.py  -> val82_v12_check.jsonl
  2. semsup_caption_promptbakeoff.py --prompt v12 on that manifest
     -> raw_v12_extra82.jsonl

No grounding/neutrality score here (the 82 extra clips have no gt_reasoning_en,
so val18_gt_slots-based slot_recall cannot be computed for them) -- this script
answers exactly the question asked: is the leakage judge result at n=18 real,
and does it hold up at n=100.

Writes:
  outputs/prompt_bakeoff/semsup_val18_neutral/leakage_judge_n100.md
"""
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
import score_val18_neutral as S  # noqa: E402

VAL18_MANIFEST = REPO_ROOT / "dataset" / "manifests" / "val_e3a.jsonl"
VAL82_MANIFEST = REPO_ROOT / "dataset" / "manifests" / "val82_v12_check.jsonl"
VAL18_CAPTIONS = REPO_ROOT / "outputs" / "prompt_bakeoff" / "semsup_val18_neutral" / "raw_v12_gemini.jsonl"
VAL82_CAPTIONS = REPO_ROOT / "outputs" / "prompt_bakeoff" / "semsup_val18_neutral" / "raw_v12_extra82.jsonl"
OUT_MD = REPO_ROOT / "outputs" / "prompt_bakeoff" / "semsup_val18_neutral" / "leakage_judge_n100.md"


def binom_test_one_sided(k: int, n: int, p: float = 0.5) -> float:
    """Exact P(X >= k | n, p) via direct summation (n<=~200 is trivially fast,
    no scipy dependency needed on the pod)."""
    from math import comb
    return sum(comb(n, i) * (p ** i) * ((1 - p) ** (n - i)) for i in range(k, n + 1))


def main():
    val18 = {r["video_id"]: r for r in S._load_jsonl(VAL18_MANIFEST)}
    val82 = {r["video_id"]: r for r in S._load_jsonl(VAL82_MANIFEST)}
    cap18 = {r["video_id"]: r for r in S._load_jsonl(VAL18_CAPTIONS)}
    cap82 = {r["video_id"]: r for r in S._load_jsonl(VAL82_CAPTIONS)}

    missing82 = set(val82) - set(cap82)
    if missing82:
        raise SystemExit(f"{len(missing82)}/82 extra clips have no caption yet "
                          f"(captioning run may still be in progress): "
                          f"{sorted(missing82)[:5]}")

    captions, labels = {}, {}
    for vid, r in val18.items():
        captions[vid] = cap18[vid]["caption_neutral"]
        labels[vid] = "YES" if int(r["target"]) == 1 else "NO"
    for vid, r in val82.items():
        captions[vid] = cap82[vid]["caption_neutral"]
        labels[vid] = "YES" if int(r["event_occurs"]) == 1 else "NO"

    n_pos = sum(1 for v in labels.values() if v == "YES")
    n_neg = sum(1 for v in labels.values() if v == "NO")
    print(f"[pool] n={len(captions)}  positives={n_pos}  negatives={n_neg}")
    assert len(captions) == 100, f"expected 100 clips, got {len(captions)}"

    from dotenv import load_dotenv
    from openai import OpenAI
    import os
    load_dotenv()
    client = OpenAI(base_url="https://openrouter.ai/api/v1",
                     api_key=os.environ["OPENROUTER_API_KEY"],
                     default_headers={"HTTP-Referer": "http://localhost",
                                      "X-Title": "MMLM_Semsup_LeakageJudge100"})

    model = "google/gemini-3.6-flash"
    pred = S.run_leakage_judge(client, model, captions)
    n_answered = len(pred)
    correct = sum(1 for vid, p in pred.items() if p == labels[vid])

    ci_lo, ci_hi = S.binom_ci(correct, n_answered)
    p_value = binom_test_one_sided(correct, n_answered, 0.5)

    # breakdown: original 18 vs the new 82, to check the two batches agree
    v18_correct = sum(1 for vid in val18 if vid in pred and pred[vid] == labels[vid])
    v18_n = sum(1 for vid in val18 if vid in pred)
    v82_correct = sum(1 for vid in val82 if vid in pred and pred[vid] == labels[vid])
    v82_n = sum(1 for vid in val82 if vid in pred)

    lines = []
    lines.append("# V12 leakage judge at n=100 (18 val + 82 fresh distinct clips)\n")
    lines.append("**Question:** the n=18 leakage-judge result (12/18=66.7%) was not "
                  "statistically distinguishable from chance (one-sided exact binomial "
                  "P(X>=12|n=18,p=0.5)~=0.12). Does the result hold up with 5.5x the "
                  "sample?\n")
    lines.append(f"**Sample:** {n_answered}/100 clips answered by the judge "
                  f"({n_pos} positive / {n_neg} negative ground truth, balanced by "
                  "construction). 82 new clips sampled distinct-video, balanced "
                  "41 pos/41 neg, from the 1,482-distinct-clip pool "
                  "(`train4500_hires.jsonl`), zero overlap with the 18 val clips.\n")
    lines.append("## Result\n")
    lines.append(f"- **Judge accuracy: {correct}/{n_answered} = {100*correct/n_answered:.1f}%**")
    lines.append(f"- 95% CI (normal approx): [{100*ci_lo:.1f}%, {100*ci_hi:.1f}%]")
    lines.append(f"- One-sided exact binomial P(X>={correct} | n={n_answered}, p=0.5) "
                  f"= **{p_value:.4f}**")
    lines.append(f"- Verdict: {'SIGNIFICANT above chance -- residual leakage confirmed, not noise' if p_value < 0.05 else 'NOT significant at alpha=0.05 -- consistent with a small amount of leakage or with pure chance; cannot fully distinguish at this n either' if correct/n_answered > 0.5 else 'at or below chance'}")
    lines.append("")
    lines.append("**Batch consistency check** (do the original 18 and the new 82 agree?):")
    lines.append(f"- val18 subset: {v18_correct}/{v18_n} = "
                  f"{100*v18_correct/v18_n:.1f}%" if v18_n else "- val18 subset: n/a")
    lines.append(f"- extra82 subset: {v82_correct}/{v82_n} = "
                  f"{100*v82_correct/v82_n:.1f}%" if v82_n else "- extra82 subset: n/a")
    lines.append("")
    lines.append("## Interpretation\n")
    if p_value < 0.05:
        lines.append(f"The result is now statistically real: V12 still leaks the label "
                      f"at {100*correct/n_answered:.1f}% judge accuracy, well above the "
                      f"50% neutral target, though far below V10's corpus-level AUC=0.964 "
                      f"leak. The prompt reduces leakage substantially but does not "
                      f"eliminate it -- recommend a further prompt iteration before "
                      f"committing to the full re-caption, OR proceed with the "
                      f"understanding that the full-corpus TF-IDF AUC gate (task 1.6) "
                      f"is the final arbiter and may still fail.")
    else:
        lines.append(f"At n={n_answered} the result is not statistically distinguishable "
                      f"from chance. This is the strongest evidence so far that V12 "
                      f"is register-neutral (or close to it) -- recommend proceeding to "
                      f"the full re-caption, with the full-corpus TF-IDF AUC test (task "
                      f"1.6, n~1,400+ via GroupKFold) as final confirmation rather than "
                      f"further n=100-scale iteration.")
    lines.append("")
    lines.append("## Files\n")
    lines.append("- `val82_v12_check.jsonl` -- the 82 sampled clips (manifest)")
    lines.append("- `raw_v12_extra82.jsonl` -- their V12 captions")
    lines.append("- `sample_val_check_clips.py` -- the balanced-distinct-clip sampler")
    lines.append("- `leakage_judge_100.py` -- this script")

    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote {OUT_MD}")
    for l in lines:
        print(l)


if __name__ == "__main__":
    main()
