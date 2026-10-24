# Where the experiments stand (2026-10-24)

A plain-language account of every experiment on the reasoning path: what each one asked, why we ran it, what was done, what came out, and what we concluded.
All numbers are on Nexar data. AP and AUC are threshold-free (they measure how well crash clips are ranked above normal clips); every thresholded number states its threshold.
"Validation" = the 524 original validation windows (core) or 641 with the 117 extra 0.75 / 1.25 s windows (full). "Test" = Nexar private 677 + public 667 = pooled 1,344, scored only at the end.
Detailed ledgers: `docs_agents/EXPERIMENTS.md`; per-experiment folders `outputs/<experiment>/summary.md`.

## The goal in one paragraph
One vision pass should give both the collision alert and its explanation. The frozen V-JEPA2 / BADAS encoder (with the crash model's LoRA, "A1-compress256") turns a 16-frame dashcam window into 2,048 tokens;
a small trained projector maps them into Qwen3-VL-4B, which (with its own LoRA) writes "Collision: yes/no", a short reasoning sentence and an outcome sentence. The decision score is read at the word after "Collision:"
(log P(yes) − log P(no)). The bar: the frozen crash model (AP 0.911 pooled test) and BADAS (Kaggle mAP: BADAS-1.0 0.925, BADAS-2.0 0.940).

## 1. The crash model (the baseline, frozen since September)
- **Asked:** how good is a vision-only crash score on this data, and how noisy is it?
- **Done:** the published BADAS-Open model, then our fine-tune (LoRA on the encoder, a better crop: "A1-compress256"); later 4 more training seeds of the same recipe.
- **Result:** test AP 0.913 private / 0.910 public (pooled 0.911, Kaggle mAP 0.916). The other 4 seeds score 0.897-0.910 pooled: the champion is the best of five (seed mean 0.906 ± 0.006).
- **Concluded:** this is the reference model; its own seed noise is about ±0.006 AP.

## 2. Building the reasoning model (October, weeks 1-2)
- **Projector + language model, first runs (6-12 Oct).** Asked: can the language model read the encoder's tokens at all? Phase 1 trains only the projector (alignment), Phase 2 adds the language-model LoRA.
  Result: the text depends on the video (a wrong clip raises the text loss), the yes/no decision is close to the crash model on validation, but the first reasoning texts were generic (template collapse: one sentence repeated 147 times).
- **Teacher dataset (8-12 Oct).** Asked: what should the model learn to write? Gemini wrote a blind description of each training window ("E3"); where its verdict was wrong, a second look with a correct neighbouring window (E6) or the frames after the window (E7) fixed it; 30 windows were hand-edited.
  Result: 4,169 windows with explanation + outcome sentence (~$63). Then 3 rewordings per text (same facts, other words), checked automatically and by a second model.
- **Fresh Phase 1 on the Nexar texts + richer inputs (13-14 Oct).** Asked: does training Phase 1 on our own texts (not DADA) fix the generic text, and do extra inputs (crash-head attention, a middle encoder layer) help?
  Result: template collapse gone (536 distinct explanations instead of 259), text depends ~3x more on the video, side of the hazard named correctly 73 % vs 49 %. Extra inputs added little.
- **Phase 1 enlargement "P1aug" (14-15 Oct).** Asked: does more variety help the projector? Every window also as copies shifted by 1-3 frames, and the rewordings as extra texts. Result: P1aug became the starting point of every later model of that week.
- **Three answer formats (15-16 Oct).** Asked: should the verdict come before or after the reasoning? Result: verdict first (**P2V**) ranks best; reasoning first hurts the decision. P2V: validation AUC 0.953.
- **P2V on the test sets (17-18 Oct).** Pooled test AP **0.926** vs the crash model's 0.911 (+0.015 [+0.005, +0.025]). One seed only, so not yet a claim.

## 3. Trying to improve the decision (19-22 Oct, validation only)
Each of these asked "does change X move the decision beyond noise?". The noise between recipes on validation is ~0.007 AUC.
- **Extra time-to-impact windows + gentler training ("chain B").** More windows at 0.75 / 1.25 s. Result: worse (val 0.942). Closed.
- **Stage 1: R2 (regularised Phase 2) and D2 (relaxed rewordings, 27 % more texts).** Result: 0.951 / 0.952 vs P2V 0.953, no gain. R2 helped the "gap" fact (+0.055).
- **Fusion of crash model + reasoning model.** Asked: does combining both scores help? Result: at most +0.0024 AUC. Closed (the gain was below the +0.005 bar).
- **B4 vs control.** Asked: does feeding the crash head's attention to the projector help? Result: B4 helps the motion / gap facts but not the decision (0.953 vs control 0.959) and needs the crash head at inference, so it is not a final candidate.
  The control (D2's recipe with a fresh two-step Phase 1) became the base.
- **Single-stage vs control.** Asked: can one projector stage replace the two-step Phase 1? Result: worse (0.948 vs 0.959). The two steps stay.
- **Stage 3: train the encoder's LoRA with the language model's loss (22 Oct).** Asked: if the encoder adapts to the language model, does the decision improve? Two arms (all parts / encoder only).
  Result: no point beat the starting model (0.9593); the joint arm over-fitted, the encoder-only arm stayed flat. Negative; the encoder stays frozen.

## 4. Stage 4: five seeds of the final recipe, then the test sets once (22-23 Oct)
- **Asked:** what does the final recipe really score, with error bars, on test, for the decision and the text?
- **Done:** the control recipe trained 5 times (seed 0 = the existing run, seeds 1-4 new, two pods ~11 h, ~$15), each scored on validation and on both test halves; the teacher's blind text was made for the public test windows ($8.4);
  text scored against the teacher where the teacher's verdict was right (private 431, public 460 windows). Five rules (R1-R5) were written before any test number.
- **Validation:** 0.952 ± 0.005 core AUC. Seed 0's 0.959 was a lucky draw (it won the recipe selection on these same windows).
- **Test decision (pooled 1,344):**

| | AP | AUC | Kaggle mAP |
|---|---|---|---|
| crash model (champion) | 0.911 | 0.913 | 0.916 |
| crash model, 5 seeds averaged | 0.908 | 0.909 | 0.911 |
| reasoning model, single seed (mean ± SD of 5) | 0.913 ± 0.003 | 0.918 ± 0.003 | 0.915 ± 0.005 |
| **reasoning model, 5 seeds averaged** | **0.927** | **0.930** | **0.932** [0.915, 0.947] |
| P2V (15 Oct, one seed) | 0.926 | 0.928 | 0.928 |

- **Rules:** R1 (one model beats the crash model by >= +0.010 AP): **no** (+0.002). R2 (was P2V lucky?): its 0.926 is above all 5 seeds (0.908-0.917). R3 (is the 5-model average worth it?): **yes** (0.927 vs best seed 0.917).
  R4 (vs BADAS-2.0's 0.940): below. R5 (is the text about the clip?): **yes** for every seed and both halves.
- **Is the averaging gain just "ensembling"?** Checked afterwards with the 5 existing crash-model seeds: averaging them gains only +0.002, averaging the reasoning models +0.014.
  The reasoning ensemble beats the crash ensemble by **+0.019 AP [+0.010, +0.029]** (paired bootstrap). Likely reason: the crash seeds share a frozen encoder and head and make the same mistakes; each reasoning model starts from a random projector.
- **Text on test:** BERTScore 0.422 / 0.421 (private / public) vs a floor of ~0.33 (the same answer against another clip's teacher text); the four facts (road user, side, motion, gap) agree with the teacher 0.60 / 0.62 vs majority baselines 0.49 / 0.53;
  a wrong clip raises the teacher text's loss by +0.29 to +0.36 nats; the right clip is found among 10 in 30-50 % (chance 10 %); format valid on 100 %; the written verdict matches the score's verdict on >= 98.8 %.
- **Concluded:** one reasoning model = the crash model on the decision, plus a clip-specific explanation; five averaged clearly beat five averaged crash models and are numerically above BADAS-1.0 (0.932 vs 0.925, inside our uncertainty), below BADAS-2.0.

## 5. Open question now: was P2V a lucky run, or is its recipe better? (step B)
P2V (0.926) sits ~4 SD above the 5 new seeds, while its validation (0.953) was inside their validation noise. It differs only in Phase 1 (frame-shifted windows + strictly judged rewordings).
Step B trains 4 new full seeds of the P2V recipe (two pods, ~14 h, ~$20) and compares the recipes seed against seed with a rule fixed beforehand (P2V itself not counted, because it triggered the question).
Plan: `~/.claude/plans/CCP based BADAS/2026-10-24_Plan-StepB-P2V-Recipe-Seeds.md`.

## What the thesis can say today
- A single vision pass gives the alert and a clip-specific explanation; the decision matches the vision-only crash model (AP 0.913 vs 0.911).
- Averaging five reasoning models raises the decision to AP 0.927 / Kaggle mAP 0.932, clearly above five averaged crash models (+0.019 AP), numerically above BADAS-1.0 and below BADAS-2.0; it costs five passes at inference.
- Every claim is from one test pass, with rules fixed in advance; the crash-ensemble comparison and step B are follow-ups and are labelled as such.
