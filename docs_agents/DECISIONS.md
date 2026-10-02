<!-- handoff-month: 2026-10 -->
# Decisions

One line per rejected option so it is not re-proposed. Full reasoning, evidence tables and the chronology of every item:
`history/2026-09_DECISIONS.md` (covers Jun → 30 Sep 2026; `history/2026-08_DECISIONS.md` is a 16-Sep snapshot).

## Rejected options

### Scope decisions by the user
- **Keep optimising the crash-score path** → frozen 2026-10-02 (champion 0.9128/0.9096, midneg mean 0.916); further ViT-L gains are "very hard". 1.5 s TTE ceiling accepted.
- **V13 captions for any new use** → rejected by the user: 96.9 % of 4,446 captions share one of 3 openers (template collapse, distinctiveness −32.5 % vs V12).
- **Datasets older than ~3 years as primary reasoning sources** → user asked for 0–3-year-old datasets only (DAD 2016, CCD 2020, SUTD-TrafficQA 2021, BDD-X 2018 are old; BDD-X is kept in the draft only as an optional Wave-2 alignment source — confirm).
- **Calibrating the decision threshold for reported CMs** → 0.5 stays the standard; AP/Kaggle mAP are threshold-free.

### Crash-score methods (all measured; see EXPERIMENTS.md §1b)
- **Semantic supervision of any kind (caption InfoNCE/cosine, two-stage P1, pooled-tap term, concept heads, V13 redesign)** → every arm lost or tied A1; recovery family indistinguishable from the 0.0009 noise floor. Not retired by the crop bug (common-mode). `--sem-pooled-weight` exists, default off, **never run**.
- **Attention/gradient-location supervision (Stage AA token-relevance, AA-H `attn_rank`/`attn_mass`, `gradcam` unrun)** → attention was moved 2×–40× with AP flat; the relevance label never encodes "does the trajectory imply a hit".
- **Loss-based look-ahead head (λ 15.3 and 1.0)** → `z_hat = f(z_now)` carries no information beyond `z_now`; a trained linear probe on `z_now` already beats the frozen head on the predicted future. Rule kept: any "feed the head a derived feature" gate needs a label-trained readout of the same input as a control.
- **Horizon-weighted crash loss as the 1.5 s fix** → positives-only shifts the operating point; symmetric is inside seed noise.
- **Full 4,446-window pool with the old negative sampling** → −0.0091 AP; the gain came from midpoint-aligned negatives, not pool size. **A1-v2 recipe bundle (full pool + cosine LR + encoder-only LoRA + dropout 0.10)** → 0.868–0.888 < 0.900.
- **Unfreezing the crash head with default LR settings** → head moved < 0.05 % (cosine decay × 0.1 multiplier); needs `--head-lr-schedule constant` and a higher multiplier; full unfreeze rejected in favour of LoRA on the head (not built).
- **Full-unfreeze of the ViT-L trunk** → too large for ≤4.5 k windows. **Training on only the 587 A0-failure windows** → AP 0.333 (inversion).
- **Dropping teacher-flagged "unexplainable" windows or high-confidence-wrong A0 windows as a fix** → circular (filters on our own model's output).
- **Run `A1-crop-rerun`** → deliberately cancelled; G1 re-score already ruled out environment inflation.
- **Generic VLM as the crash predictor / VL-JEPA predict+reason route / ReverseBERT decoder / off-the-shelf BLIP-2 or LLaVA projectors** → Nexar's own tests lost clearly; VL-JEPA decoder was domain-locked; off-the-shelf projectors are trained on CLIP/SigLIP image features, geometrically incompatible with V-JEPA2 patches.

### Reporting / methodology rules (still binding)
- Never select a checkpoint/epoch on test (including "fewer FN" picks); fix one epoch in advance or from val; never report the single best checkpoint — report mean ± sd over ≥3 seeds (5 for headline claims).
- Compare arms by paired bootstrap against the same-seed twin and by FPR at matched recall, not FP/FN at 0.5 across different seeds; pool private+public.
- Clip-level val AP over row-level; rank-1 val_ap selection is untrustworthy (picked attn_mass's worst epoch).
- AP computed from rounded per-clip dumps differs from the summary AP (ties); summaries are authoritative for AP/AUC, dumps for the confusion matrix.
- Any hardcoded drift pin (EXPECTED confusion matrices) must be re-derived when scores are regenerated.

### Teacher / caption lessons
- V10 (GT-informed branch) leaks the label (TF-IDF AUC 0.9643); V12 neutral reached 0.764 — fine for semantic supervision, **wrong target for explanations** (outcome words banned).
- Caption-length rules need a floor as well as a ceiling; a single worked example is copied as a template (V13) — use several structurally different examples.
- Prompts mixing a banned-word list and a required vocabulary must check they do not collide ("hazard"/`hazards_on`).
- A GT-informed block makes the teacher fabricate hazards on no-crash clips (V11 lesson) → blind mode for negatives.
- 18-clip screens cannot rank prompts (McNemar p ≥ 0.125); do not iterate prompt versions on n=18. Always pass `--model` explicitly; never compare to a stored baseline for a `preview` model alias.

### Reasoning-path plan review (2026-10-02/03, the Gemini plan)
- **"Bounding boxes extracted from V-JEPA2" / "radar detected a hazard" in the prompt** → fabricated (V-JEPA2 has no detection head; no radar). Boxes in this repo come from the AA pipeline.
- **Detector boxes in the inference prompt** → rebuilds the intermediary BADAS-2.0 says to remove and breaks the "explanation from the features that drove the score" claim; boxes only for training targets, grounding eval, one ablation arm.
- **Average-pool + 2-layer MLP projector** → loses position ("car on the left"); keep `ResamplerProjector` (passed the e4 gate).
- **Uniformly sampling 16 frames over a whole 10–40 s clip** → off-distribution temporal statistics for an encoder trained on 2 s @ 8 fps; use 2 s windows (multi-window encoding for long captions).
- **Spatial augmentation / horizontal flip in Stage 1** → pointless with a frozen cached encoder and flips corrupt left/right text.
- **"Audit that the fast-path AP stays above 0.86" as a final check** → vacuous (score path frozen and separate); replace with faithfulness metrics.
- **CCD/"DAD" for language alignment** → CCD ships no text; the cited repo is CCD, not DAD (Chan 2016). **DRAMA via `github.com/mtli/DRAMA`** → 404 (real source: Honda, university request). **LingoQA** → 4 s @ 1 Hz, incompatible with 16 frames @ 8 fps. **TAR/CrashSight** → CCTV; **AccidentBench** → MCQ mixed land/air/water; **TAU-106K** → annotation method unverified; **SUTD-TrafficQA** → 2021, mixed viewpoints (unverified).
- **MCQ (MM-AU reason labels, CAViAR/VRU MCQ) as a generation target** → trains a classifier, not text; evaluation only.
- **Teacher "reasoning" for positives** → unnecessary for the 749 CAViAR clips (human-written, 37–40-word answers); only negatives and MM-AU need teacher text.
- **Treating the ~35 k–75 k-pair projector-only Stage 1 as sufficient by assumption** → Meta used 88.5 M pairs and trained the LLM in stages 2–3; sufficiency is a hypothesis (H0) tested by a learning curve, never assumed.

## Unresolved — reasoning path (needs the user)
1. **Approve / modify the draft plan** (`~/.claude/plans/CCP based BADAS/2026-10-02_Plan-Reasoning-Path-Public-Pretrain-DRAFT.md`). ExitPlanMode was rejected five times; the user's last open questions (answered in the plan text, not yet confirmed by them): (a) cross-dataset generalisation — plan answer: source tag in the instruction + held-out Nexar evaluation after every wave, drop a wave that does not help; (b) what the H0 curve is — plan answer: a learning curve of held-out Nexar alignment quality (Δ-PPL vs random projector, ΔCE-shuffle, distinct outputs/50) against 1k/5k/20k/50k Stage-1 pairs; (c) is Wave 1 + Stage 1 realistic in about a week — plan answer ≈ 3–4 working days of wall time at ~70 % confidence, minimal path = Wave 1 only (MM-AU ~30 k windows + Nexar ~4.4 k captions), Wave 2 needs the BDD100K download, Wave 3 is TB-scale.
2. **Does a projector-only alignment of a language-free encoder to an LLM work at ~35–75 k pairs?** (H0, the load-bearing assumption). Escalation ladder if the curve is flat: LoRA on the LLM during Stage 1 → larger resampler → natively multimodal LM. Encoder frozen in every option.
3. **Which LLM:** Qwen3-4B-Instruct-2507 (known-good control) vs Qwen3-VL-4B's LM (BADAS-Reason's, makes the thesis comparison controlled) vs Qwen3.5-4B re-tested. Hypothesis (unverified): the weak Qwen3.5 result came from feeding a natively multimodal model generic `pad` placeholders; native injection must be implemented per model.
4. **Which tokens to feed:** all 2560 (2048 real + 512 BADAS-predicted "future" tokens) or only the 2048 real tokens — the concat order is unverified and e4 cached all 2560.
5. **Encoder identity:** A1-compress256 trunk (required for the faithfulness claim) vs stock BADAS-Open (ablation). Every e4 projector/cache must be rebuilt under `compress256`.
6. **MM-AU access:** HF listing is public (cc-by-nc-4.0) but the GitHub README says academic-only/email — confirm terms before pulling ~345 GB (CAP+DADA). RoadSafe365 availability unknown (no link). CAViAR's CCD annotations are not released.
7. **Spend approvals (none given):** ~$25 Nexar descriptive captions, ~$5–10 negatives, ~$60 MM-AU anchored reasoning, Wave-3 captions only if triggered.
8. **Run the never-run e4 ep5/ep7 generation eval first (~1 pod-hour)?** If later epochs are already clip-specific, the June collapse was checkpoint selection, which weakens the case for large pre-training.
9. **Text format:** one shared format across all reasoning data (CAViAR's question set: description → primary reason → at-fault → violated rule), phased/decomposed short steps rather than one free paragraph (TAR's only published ablation: +1.5 pp from reasoning traces; our confidence that a structured chain beats a paragraph at this scale ≈ 65 %).
10. **Baseline before SFT (user request):** text-only floor (visual tokens zeroed), Stage-1 zero-shot, BADAS-Reason-style Qwen3-VL-4B zero-shot on a held-out CAViAR split — every later model must beat the floor.

## Unresolved — crash-path leftovers (low priority while frozen)
- Which fixed epoch is the recipe? Epoch 1 has the best 3-seed mean (0.9185 vs 0.9157 / 0.9133) but that ranking used the test set; fix from val (`--dump-val-scores`) or declare, then confirm on fresh seeds.
- Seed-to-seed calibration jitter: stabiliser candidates untried — cosine LR decay or EMA/SWA instead of constant 2e-4; trainable logit scale+bias (or unfrozen head bias); seed ensembling. Attributing it to LoRA init vs data order needs a code change (both drawn from `--init-seed`).
- Is A0 a clean baseline on the 1,761 pool? Probably not (Nexar's own train split; A0 AP 0.9535 there vs 0.853 held-out) — treat A0's pool column as contaminated.
- Should the training pool stay failure-enriched (one-third mined failures)? Not isolated from the A1-v2 recipe bundle.
- 1.5 s TTE options never taken because the path is frozen: longer input window/temporal stride, kinematic (detection/lane) targets (Stage 4), or accept the ceiling (the current choice).
- Unconfirmed GT partner labels for several AA.1 clips (gen18 00903/00932/01035, dev 00283) — only matters if the AA pipeline is reused for grounding evaluation.
- The pod checkpoints `e2_lora_100clips`, `e3a_lora_89clips`, `e3b_lora_267clips` exist only on the old volume (not on HF); not deleted pending explicit confirmation.
