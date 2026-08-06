# V10/V10Q GT-informed mechanism captioning — 2×2 bake-off, 18-clip val set

> **Correction notice (2026-08-02):** the first version of this document recommended
> **Qwen3-VL-235B-Thinking**, on the grounds that it "never fabricated" and used the
> `mechanism_visible=false` escape hatch honestly. **Both claims were wrong** and are retracted
> below — Qwen reproduced a known fabrication (`01552` school bus) and was misusing the escape
> hatch, not being honest. The corrected recommendation is **Gemini 3.6 Flash**.

**Question:** does giving the teacher the binary GT label (train-only, never at inference) and
asking it to explain the mechanism — instead of predicting the verdict — produce better,
non-fabricated captions than blind prediction? And which model should caption the 587
train4500 failure windows?

**Settings:** native 1280×720, `detail="high"`, `temperature=0.1`. Gemini: no `max_tokens` cap
(~4 min/18 clips). Qwen (thinking model): `max_tokens=16000` (~8-10 min/18 clips). 72/72 calls
succeeded, 0 schema failures across all 4 arms. Per-clip data:
`reasoning_analysis_v10_gt_val18.xlsx`.

**Scorer disclosure:** automated, not hand-scored (a first for this project's prompt rounds).
`slot_recall` = keyword overlap against GT slots extracted from `gt_reasoning_en`
(`dataset/manifests/val18_gt_slots.json`), calibrated against the known hand-scored CONTRADICT
set (`00319/00372/00474/00529/00687`) before being trusted. **It is exact-phrase brittle and
systematically undercounts** — `00319`'s GT-arm caption is a clear semantic match but scored
0.25 because the GT keyword `"vehicle from the right"` doesn't appear in that word order. Read
`slot_recall` as a noisy floor; the CONTRADICT flag and manual review are the reliable signals.

## Headline

| | Gemini/GT | Gemini/blind | Qwen/GT | Qwen/blind |
|---|---|---|---|---|
| mean slot_recall | 0.417 | **0.468** | 0.347 | 0.352 |
| MATCH / PARTIAL / MISS | 11/4/3 | **12/4/1** | 9/5/4 | 7/8/3 |
| CONTRADICT (positive clips) | **0** | 1 (`00319`) | **0** | 0 |
| confirmed fabrication | `01643` (ambiguous) | — | **`01552` school bus** | — |
| blind verdict acc / AP / AUC | — | **66.7% / 0.737 / 0.722** | — | 44.4% / 0.548 / 0.556 |

Row meanings: **slot_recall** = fraction of GT's 4 mechanism slots (which vehicle / what it did /
where / how the gap changed) the caption recovered. **MATCH/PARTIAL/MISS** = per-clip grade from
that. **CONTRADICT** = on a clip that *does* crash, the caption claims everything is fine — the
catastrophic case, since it trains the model on the opposite of reality. **blind verdict** =
crash/no-crash prediction quality without being told; a sanity check on raw perception, not on
caption quality.

## Model choice: Gemini 3.6 Flash

Gemini wins on every measurable quality axis — higher slot_recall in both modes, more MATCH,
better raw perception (blind AP 0.737 vs 0.548), 2× faster. Qwen wins only on price.

**Why the original Qwen recommendation was wrong:**

1. **The escape hatch wasn't honesty, it was schema misuse.** Qwen set
   `mechanism_visible=false` on 7/9 negatives, which looked like admirable restraint. Only **2**
   (`01643`, `01737`) are genuinely empty scenes. On the other 5 it flagged "no mechanism
   visible" *and then described vehicles anyway* — e.g. `02117`: `false`, caption *"silver sedan
   ahead maintains consistent following distance…"*. It read the field as "no crash happens"
   (trivially true — the GT block just told it so) rather than "no relevant agent," which is what
   the prompt asked for.
2. **Qwen did fabricate.** On `01552` it wrote *"school bus stationary at gas station"*. There is
   no school bus in GT. This is the **known Qwen-family hallucination on that exact clip**,
   tracked since V4 — and the earlier v6 round explicitly noted Gemini does *not* reproduce it.

Gemini's one suspect case (`01643`, `hazard_agent: "opposing vehicles in left lane"` where GT
says *"no vehicles around it"*) is **unresolved, not confirmed** — Qwen independently reported
"parked cars on left" on the same clip, so either both models see something the GT text glosses
over, or both err. Not adjudicable from text alone; would need the frames.

## GT vs blind: use GT on positives, blind on negatives

**This is the most useful result, and it is not "GT is better."** Paired per-clip on Gemini:
**GT better on 3 clips, blind better on 4, tied on 11** — a wash overall. Splitting by class
shows why the aggregate hides the real effect:

| | GT mean_recall | blind mean_recall | GT MATCH | blind MATCH | GT CONTRA | blind CONTRA |
|---|---|---|---|---|---|---|
| **Positives (9)** | **0.528** | 0.500 | 6 | 6 | **0** | 1 |
| **Negatives (9)** | 0.306 | **0.435** | 5 | **6** | 0 | 0 |

- **On positives GT helps** — marginally on recall, and it eliminates the one CONTRADICT
  (`00319`, where the blind arm asserted no conflict on a clip that crashes). That contradiction
  is exactly the failure mode the GT block was designed to remove, and it removed it.
- **On negatives GT actively hurts** (0.306 vs 0.435, 3 MISS vs 1). Mechanism: telling the model
  *"this does NOT end in a collision, identify the dominant benign dynamic"* pressures it to
  nominate **some** focal agent when there isn't a meaningful one — so it picks a wrong or
  irrelevant one (`01281`: named a Lexus SUV instead of GT's blue pickup; `01643`: "opposing
  vehicles" on a road GT calls empty). Blind mode just describes the scene and lands closer.

**Recommendation: hybrid.** Run `--gt-mode gt` on the 269 FN (positive) windows and
`--gt-mode blind` on the 318 FP (negative) windows. Costs nothing extra — it's two invocations
of the existing runner against two filtered manifests.

**Caveat, stated plainly:** n=18 (9 per class). The positive-side CONTRADICT gain is literally
one clip, and the negative-side gap rests on 2-3 clips. This is weak evidence with a coherent
mechanism behind it, not a settled result. It is cheap to act on and cheap to reverse.

### Confirmed by an independent metric: the negative-clip fabrication check

Added after the first version of this document, precisely because the two fabrications above
were found by hand and would not survive 587-window scale. Two triage rules (see
`fabrication_check()` in the scorer): **EMPTY_SCENE** (GT asserts no vehicles present, model
names an agent anyway) and **SPECIFIC_TYPE** (model names a vehicle type absent from GT, applied
only when GT itself names at least one specific type — otherwise GT vagueness would flag
everything). Both hand-found cases are caught with correct reasoning; the `01153` false positive
from a first draft is gone.

| Arm | negatives flagged |
|---|---|
| Gemini / **GT** | **3 / 9** (`01281`, `01643`, `02117`) |
| Gemini / **blind** | **1 / 9** (`01281`) |
| Qwen / GT | 1 / 9 (`01552`) |
| Qwen / blind | 2 / 9 |

**Gemini's GT arm fabricates 3× more on negatives than its own blind arm** — a second,
independent metric agreeing with the slot_recall split. The hybrid recommendation now rests on
two measurements rather than one, which is what moved it from "weak signal" to "act on it."

## What this does NOT settle

- n=18, as with every prior round in this thread.
- The fabrication check is **triage, not a verdict**. It cannot distinguish a genuine
  hallucination from a synonym mismatch (GT "jeep" vs model "SUV" on `01281`) or from the model
  picking a real-but-wrong agent. It also cannot catch a fabrication phrased only in generic
  terms when GT is itself generic. Treat flags as a review queue.
- `mechanism_visible` is **not reliable as a quality signal** — Qwen misread it as "no crash
  happens" rather than "no relevant agent". If it is kept for the 587 run it should be treated
  as a diagnostic to inspect, not a field to filter on.
- Whether Qwen's price advantage is real — never verified against current OpenRouter rates,
  and moot now that Gemini is the pick on quality.

## Next step

Caption the 587 train4500 failure windows with **Gemini 3.6 Flash**, hybrid GT mode:

```
# 269 FN (positive) windows
--prompt v10 --gt-mode gt    --model google/gemini-3.6-flash
# 318 FP (negative) windows
--prompt v10 --gt-mode blind --model google/gemini-3.6-flash
```

Then run the fabrication check over the negative half and hand-review only the flagged rows
(expected ~1-3 per 9 negatives ≈ 35-100 rows of 318, based on this round's blind-arm rate),
rather than reading all 587. Then A1_587-vs-B_587, paired, on the identical window set.

## Files

- Raw captions (4 arms): `raw_v10_gemini_gt.jsonl`, `raw_v10_gemini_blind.jsonl`,
  `raw_v10q_qwen_gt.jsonl`, `raw_v10q_qwen_blind.jsonl`
- **Human-review sheet (GT vs blind, with caption text):**
  `compare_gemini_gt_vs_blind_val18.xlsx` — built by
  `teacher_distillation/scripts/compare_gemini_gt_vs_blind_val18.py`
- Per-clip scored data, all 4 arms (metrics only, no caption text):
  `reasoning_analysis_v10_gt_val18.xlsx`
- GT slots (built once, reusable): `dataset/manifests/val18_gt_slots.json`
- Prompts: `prompts/PROMPT_SEMSUP_V10_GT.py` (winner), `prompts/PROMPT_SEMSUP_V10Q_GT.py`
- Runner: `student_training/scripts/semsup_caption_promptbakeoff.py`
  (`--prompt v10 --gt-mode gt|blind`)
- Scorer: `teacher_distillation/scripts/reasoning_analysis_v10_gt_val18.py`
  (`--calibrate` re-verifies against the known CONTRADICT set)
- Plan: `~/.claude/plans/CCP based BADAS/2026-08-02_Plan-V10-GT-informed-captioning-prompts-2x2-bakeoff-val18.md`
