# PROMPT_G_OPT_v6_balanced (unmodified) vs google/gemini-3.6-flash — 18-clip screen

**Question:** does the ORIGINAL, unmodified v6 teacher prompt — the 7-step CoT-gated,
"prefer NO"-laden prompt this entire nine-round session moved away from — do better on
Gemini 3.6 Flash than V9 (the minimal, evidence-distilled prompt this session converged on)?
Same model, same 18 clips, only the prompt differs.

**Settings:** `semsup_v6_control_rerun.py`, unmodified `PROMPT_G_OPT_v6_balanced.py` (no
`caption_neutral`/`risk_clause` — native fields `collision_verdict`/`verdict_reasoning`/
`scene_context`/`dynamic_objects`/`temporal_analysis`), `google/gemini-3.6-flash`, native
1280×720, `detail="high"`, `temperature=0.1`, `max_tokens=8192`. 18/18 succeeded first attempt,
0 failures. Wall 184s.

**Scorer disclosure:** hand-scores assigned by Claude. `verdict_reasoning` + `temporal_analysis`
serve as the caption-equivalent text for this comparison, since this prompt produces no
`caption_neutral` field.

## Headline: this is the best result of the entire nine-round session, on both axes

| | V9 / Qwen3-VL-235B | V9 / Gemini 3.6 Flash | **v6_balanced / Gemini 3.6 Flash** |
|---|---|---|---|
| Accuracy | 72.2% | 50.0% | **72.2%** |
| Recall | 0.56 | 0.22 | 0.44 |
| Precision | 0.83 | 0.50 | **1.00** |
| TP/FP/TN/FN | 5/1/8/4 | 2/2/7/7 | **4/0/9/5** |
| reasoning MATCH | 4/18 | 4/18 | **9/18** |
| reasoning CONTRADICT | 5/18 | 9/18 | **4/18** |
| mean hand-score | 4.33 | 4.22 | **5.22** |

v6_balanced on Gemini 3.6 Flash matches V9-on-Qwen's best-yet accuracy (72.2%), does it with
**zero false positives** (the only round in nine to combine a non-trivial recall with perfect
precision), and produces the highest-fidelity reasoning of any round or model tested all
session — half its captions substantively match GT's stated mechanism, versus roughly a
quarter for every other prompt/model pair tried.

## Same model, same clips, only the prompt differs — and the gap is the sharpest seen all session

McNemar (v6_balanced vs V9, both on Gemini 3.6 Flash): **4 clips v6_balanced-only-correct, 0
clips V9-only-correct, p = 0.125.** Every other prompt comparison this session landed at
p = 0.25-1.00 with roughly balanced flips in both directions; this is the first comparison
where the flips are entirely one-directional. Still short of conventional significance at
n=18 (as flagged throughout this session, that bar is unlikely to be cleared by any single
18-clip comparison), but it is the least-ambiguous result produced in nine rounds.

## Two specific things this run got right that broke almost everything else this session

- **`00687`, `00529`, `00474`**: this is the one place where the result is genuinely mixed —
  these three remain false negatives, with confident CONTRADICT-tier reasoning (e.g. `00687`:
  "No trajectory conflict exists. The grey SUV... is being safely passed by ego, receding" — the
  literal opposite of GT's SUV-drifts-in-and-closes-rapidly mechanism). This is the same
  apparent-motion misread chased across V6/V7/V8's scaffolding all session, asserted here with
  no hedge and no grounding check at all.
- **`01153`**: correctly resolved without the fabricated crossing-sedan hallucination that broke
  Qwen3-VL on this identical clip in the V5, V6, V8 and V9 rounds. Matches V9-on-Gemini's result
  on this same clip — a second, independent confirmation that this specific hallucination is a
  Qwen3-VL-family prior on this scene type, not a defect of any prompt tried this session.
- **`01552`, `01643`**: neither of the two persistent minor fabrications tracked since V4 (the
  "school bus" on `01552`, the invented road-work sign on `01643`) reproduces here.
- **`00283`**: correctly identifies a pickup truck (matching GT's actual object) rather than the
  SUV every single Qwen3-VL round reported on this clip, regardless of prompt version.

## Why this result plausibly makes sense, stated carefully

This session's earlier finding was that heavy, conservative-gated CoT prompts caused severe
under-calling on *Flash-class* Qwen and GPT models (Qwen3.7 Flash and GPT-5.6 Luna Pro both
recorded TP=2/FP=0/TN=9/FN=7, recall 0.22, on this exact unmodified v6_balanced prompt). Gemini
3.6 Flash on the same prompt does not reproduce that collapse — recall is 0.44, twice those
models' rate, while keeping their zero-false-positive discipline. That is evidence the earlier
finding was about *those specific model families' interaction with heavy gating*, not a general
law that "long CoT + conservative gates always under-call on Flash-tier models." Model family
matters as much as prompt design, which is the more general lesson worth carrying forward.

## What this does NOT settle

- **n=18.** This is the sharpest single comparison of the session and still not clean by
  conventional standards. The right next step, unchanged from every prior summary this session,
  is a frozen comparison on held-out clips from `dataset/manifests/semsup_promptbakeoff.jsonl`.
- **v6_balanced has no `caption_neutral` field**, so it cannot be adopted for the actual
  captioning pipeline (`semsup_caption_qa.py` needs `caption_neutral`/`risk_clause` to build the
  A/B arms) without modification. This result is informative about *prompt structure and
  model-family interaction*, not a drop-in replacement — a variant that adds V9's caption-first
  output while keeping v6_balanced's CoT-gate structure would be the natural next test if this
  finding is worth pursuing further.
- The three false negatives that remain (`00474`, `00529`, `00687`) share a lateral-drift
  mechanism this session has repeatedly struggled to detect reliably across every prompt and
  model combination tried — this run does not solve that problem, it is simply the best
  trade-off against it recorded so far.

## Files

- Per-clip data + reasoning-alignment verdicts: `reasoning_analysis_v6_gemini36flash_val18.xlsx`
- Raw output: `raw_v6_gemini36flash.jsonl`
- Prompt (unmodified): `prompts/PROMPT_G_OPT_v6_balanced.py`
- Runner: `student_training/scripts/semsup_v6_control_rerun.py`
- Analysis script: `teacher_distillation/scripts/reasoning_analysis_v6_gemini36flash_val18.py`
- Direct comparison point: `v9_gemini36flash_summary.md` (V9 / Gemini 3.6 Flash)
