<!-- handoff-month: 2026-10 -->
# Decisions

One line per rejected option so it is not re-proposed. Full reasoning for June–September items: `history/2026-09_DECISIONS.md`.

## Rejected options

### Scope (user decisions)
- **Keep optimising the crash-score path** → frozen 2026-10-02 (A1-compress256 AP 0.913 / 0.910); 1.5 s TTE ceiling accepted.
- **CAViAR text as training targets (and requesting its CCD part)** → templated (Violation == Reason 744/749, repeated phrases); dropped 2026-10-03.
- **V13 captions** → template collapse (96.9 % share 3 openers).
- **Nexar text from CAViAR / human reasoning** → Nexar text will be our own V12-style teacher captions + detection boxes (BADAS-2.0 recipe), later.
- **Teacher rewriting of public data (MM-AU, BDD-X)** → no API spend on public data; teacher cost only for Nexar.
- **Ask the model to predict / explain a crash from a window that ends before the hazard is visible** → window visibility rule (memory
  `window-visibility-rule`): such windows are excluded from all phases, not relabelled as no-crash.

### Reasoning-path design
- **Text-only Qwen3-4B + 64-query position-less resampler (June e4 design)** → LM never saw visual tokens; resampler had no positional information.
  Replaced by Qwen3-VL-4B's language model + its merger structure + native video tokens / M-RoPE.
- **DeepStack (Qwen3-VL layers 5/11/17 → first 3 LM layers) in week 1** → extra parameters, not needed for a go/no-go; later ablation only.
- **A bigger/deeper projector** → Qwen3-VL's merger is already a 2-layer MLP after a 2×2 merge; deeper connectors show no consistent gain (Prismatic).
- **Reusing Qwen's merger weights as fact** → untested across different encoders; decided by a 2-epoch random-vs-Qwen init A/B in Phase 1.
- **Selecting checkpoints by validation loss** → e4 collapse lesson; select by wrong-video gap (+ diversity in Phase 2).
- **3 epochs for the projector from scratch** → only ~350 steps on ~1.85 k windows; up to 15 epochs (~1,750 steps) with early stopping.
- **Training Nexar in Phase 1** → Nexar is Phase 2 (SFT) only; Phase 1 sees it only zero-shot.
- **Training no-crash windows in Phase 1** → MM-AU has no description of normal driving; ~1.3 k identical template targets would teach a label
  shortcut. Hallucination check deferred to the next session.
- **Different text style for crash vs no-crash** → style would reveal the label (V10 leak lesson); one schema, different content.
- **Invented reasoning chains (frame-by-frame CoT, at-fault from rules) as targets** → no ground truth; only GT-backed fields are used.
- **MM-AU weather/light/scene/road codes in the targets** → codebook exists only in headers and looks inconsistent; not in the GT text.
- **MCQ (ArA, CAViAR MCQ, AccidentBench) as generation targets** → trains a classifier; ArA used only as a 5-option evaluation.
- **Crop for DADA (2.4:1) to match Nexar 16:9** → cuts ~13 % per side (the crop-bug lesson); full-frame squash (`compress256`) instead.
- **3 s / stride-2 windows for 10 fps sources** → off-distribution motion per frame for the frozen encoder; nearest-frame 2 s / 7.5 fps instead.
- **MM-AU CAP part in week 1** → per-video fps unknown (no fps column); DADA (30 fps) only until fps is resolved.
- **"Projector-only stage on ~35–75 k pairs is enough" as an assumption** → Meta used 18–88.5 M pairs with LLM training; tested by gates and a learning curve.

### Data sources (excluded after checking)
- DRAMA-X (machine-written, single frames), AccidentBench (MCQ, land/air/sea), DriveLM (no crashes), TAR (CCTV), Real-Collide (unreleased,
  contains Nexar), SUTD-TrafficQA (2021, mixed views), LingoQA (1 Hz), CrashChat text (= MM-AU list) and **CrashChat's Nexar clips**
  (renumbered, possible test contamination), SafeAuto-BDDX mirror (3 fps, 455×256), Gemini plan's `github.com/mtli/DRAMA` (404).
- LLaVA-Video-178K → only if the learning curve asks for generic coarse data.

### Infrastructure
- **Keeping datasets on the RunPod network volume** → quota full (56 GB), pins EU-RO-1; durable data goes to private HF repos instead.
- **Google Drive for datasets** → no reliable CLI for large files on pods, rate limits; HF chosen.
- **HTTP-streaming the DADA tar parts on the pod** → a stream dropped mid-part; download parts with retry + back-pressure.
- **Reading the HF cache via symlinks on Windows** → `WinError 1314`; use `local_dir`.

### Carried over (still binding)
- Never select a checkpoint on test; report mean ± sd over ≥3 seeds (5 for headline claims); paired bootstrap against the same-seed twin;
  FPR at matched recall rather than FP at 0.5 across seeds; summaries are authoritative for AP; re-derive any pinned EXPECTED confusion
  matrix when scores are regenerated.
- Teacher lessons: blind mode for negatives; several structurally different examples in prompts; ≥ 2 prompt versions not ranked on n=18;
  always pass `--model` explicitly.
- Generic VLM as the crash predictor; VL-JEPA predict-and-decode route; off-the-shelf CLIP/SigLIP projectors on V-JEPA2 features → rejected (Jun–Jul).

### Evaluation (user decisions 2026-10-06)
- **Word-for-word match as the only text metric** → misses paraphrases and the DADA text is one coarse phrase per video; use BERTScore / embedding retrieval / list-class accuracy
  (with floor and always-most-common baselines) and the user's reading of the review files. LLM judge only for inference-time evaluation, never in training.
- **BERTScore (or any sentence metric) as a training loss** → needs sampling + RL; unstable; not now. Training loss stays masked cross-entropy (answer pieces only).
- **Choosing a checkpoint by the wrong-video gap alone** → rewards over-confidence (Phase 1 picked the most over-fitted epoch). Phase 2 keeps every epoch and stops when validation loss rises 2 epochs in a row.
- **Per-word-piece probability breakdown (actor/action words)** → postponed by the user.
- **Wrong-video partner chosen at random** is a weak test (a similar scene can match the text); only proves "uses the video", not detail accuracy. Better design (same-category partners for detail) not built.
- **Phase-1 diversity gate "≥ 90 % distinct of N"** → invalid for DADA's closed phrase list; compared with the targets' own diversity instead.

### Teacher (user decisions 2026-10-08, after the boxed-teacher pilot)
- **Teacher yes/no verdict field** → removed: blind teachers under-call crashes (pilot 8–10/27; August 0.22 recall); the student's verdict comes from the label.
- **Blind explanation for crash windows** → rejected: a harmless explanation paired with the label "yes" contradicts itself. Next prompt TELLS the teacher the
  outcome and asks for the visible cues only, with "not visible" when it cannot see them (fabrication guard; such windows are dropped or marked).
- **Standard tier** → Flex chosen: half the uncached price ($0.0117 vs $0.024 per window), same latency with 16 parallel calls.
- **Removing the attention box** → kept, but first a paired no-box A/B on the same 51 windows (Flex) and a check of the chosen object against the 14 hand-labelled clips.
- **Explanation text that copies the closed-list words** → rejected; next prompt requires natural sentences.
- **Window selection from the 1,761 pool** → rejected for teacher pilots (failure-enriched); use videos outside it.
- **Teacher prompt study (user, 2026-10-08):** extra arms (Gemini Pro model, outcome + hindsight frames) → declined; only the user's E1-E4. Box stays in E2-E4. Set B (51 more windows) → not used.
- **Fine-tuning the teacher on Nexar** → not now: Nexar has labels and times but no reasoning targets, and tuning Gemini Flash is not confirmed available. Alternatives kept for later: in-context
  corrected examples (e.g. 00013), self-consistency (3 samples), a stronger model only where Flash conflicts with the label.
- **"V12 was the best prompt" as a verdict prompt** → false: V12 never predicted a verdict (caption-only). Most balanced blind verdict prompt measured at scale: `PROMPT_G_OPT_v6_balanced` on Gemini 3.1 Pro
  (e3a: recall 0.75 / specificity 0.785; private test 677: 0.78 / 0.58); on Flash and other models it under-calls (Gemini 3.6 Flash 4/9, Qwen / GPT 2/9, Qwen3-VL-235B 0/18).
- **Debate / two-pass recovery as a measure of teacher quality** → it is routed by the ground-truth label (mismatches only), so its accuracy (67.9 % to 90 % on the private test) rises by construction;
  recovered explanations scored lower (mean 3.6/10) and sometimes kept the first pass's invented agent. Use only to get a label-consistent explanation, never as a perception metric.

### Teacher dataset (user decisions 2026-10-10..12)
- **Training target for crash windows that end before the hazard is visible** → `label_vis` = 0 (the Nexar `label` stays 1 for the crash model); E7-pre turns it 1 only when hindsight shows the hazard already visible (18 windows).
- **Option (a): E7 visibility check on every anchored window** → rejected (too expensive); keep the E3 → E6 / E7 flowchart.
- **Removing 66 flagged windows** → only the 35 genuine crash outliers removed; the 31 normal windows were a chain-logic bug (for a normal window "hazard not visible" is the correct state) and were fixed.
- **Dropping the 99 crash videos with no kept window** → done in the manifest (4,203 windows from 1,401 videos); the 18 former val_e3a videos stay in the dataset.
- **Kaggle for the pod videos** → replaced by the HF Nexar dataset; the pod stops itself only after the HF push is verified.
- **Prediction sentence** → outcome only, 4-15 words (not a summary of the explanation).
- **Hand edits override the teacher** → `edits.jsonl` is applied last and an edited window is never dropped as an outlier; edited crash windows are `label_vis` 1.
- **Professor-facing documents** → no run names (E3/E6/E7) or file paths.

### Reasoning retrain (user decisions 2026-10-12..14)
- **Mixing DADA into the new Phase 2 / evaluating on DADA** -> no: Nexar only, Phase-1 start; DADA forgetting check skipped.
- **Starting Phase 2 from the old Phase-2 checkpoint** -> no: it carries the old template and captions; and the old DADA Phase 1 itself was the bottleneck (fresh Phase 1 on the Nexar texts is the big gain).
- **Verdict before the reasoning inside the main model** -> no: reasoning would become a justification and S_after = S_before by construction; verdict-first is a separate Run 2 (kept in the plan, NOT run yet - remind the user).
- **Imitating the teacher's risk_score as the student's score** -> no: verbal self-report from a closed model, clustered and label-informed; the score is read from the token probabilities at "Collision:".
- **The teacher's long prompt for the student** -> no: a short field-announcing prompt (the format is learned from 3.5 k answers; a long prompt doubles the sequence).
- **QLoRA / small learning rate / more LoRA checks** -> no: memory is not a constraint (11 GB), 2e-4 is validated, 144 steps per epoch.
- **Zero-shot Qwen with its own image encoder as an information test** -> dropped: not fine-tuned on Nexar, uninformative; replaced by the trained probe.
- **Box position as text in the prompt (option 3)** -> no: the model leans on the hint (smallest wrong-video gap), type does not improve.
- **Object crop / detector at inference** -> not now (user: wrong approach at this stage); an attention-guided crop would add a second vision path - a later design decision.
- **Four separate fact prompts** -> no (one answer format for every window); a structured fact field at the start of the SAME answer or an auxiliary fact loss are candidates after Phase 1.
- **Checkpoint chosen by the wrong-video gap** -> rewards the over-fitted epoch; Session 1 / Phase 1 use validation loss (`--select val_loss`); neither rule alone predicts generation quality.
- **More than 3 rephrasings per text** -> risk of drift and of learning Claude's style; teacher text drawn with p 0.4 each epoch; try 5 only if the 4x set helps.
- **Rephrasing validated only with the word-list extractor** -> too noisy for motion / gap (e.g. "across the frames"): hard checks on length, outcome words, type / side sets and polarity from the verdict, rest flagged for an Opus reader.
- **Many parallel Claude subagents for bulk text** -> hit the session limit; use <= 4 at a time or an API.

### Formats, evaluation, extra windows, pods (user decisions 2026-10-15..20)
- **Rephrasing writer** -> Gemini 3.8 Flash via OpenRouter (text-only) replaced Claude subagents (session limit); Sonnet 5.5 judge + Opus blind audit; gate <= 1 % major errors among judge-accepted.
- **Reasoning before the verdict as the headline format** -> verdict-first (P2V) is the headline: reasoning-first makes the verdict a justification (S_after saturated, ranking worse).
- **Chain A (the wider recipe grid) and multi-seed grids for every arm** -> not run: "we care only about P2V"; only chain B, 1 seed, early stop on P2V-v2 only.
- **Judging the new data by train/val per-TTE AP** -> not comparable (crash TTE k vs all normals ~1:10 vs test 1:1); use per-TTE AUC; Kaggle mAP only on test.
- **Choosing a P2V checkpoint / threshold on test** -> never; threshold from validation (free, open item). Train-sample threshold P = 0.56 is stricter and lowers recall (earlier opposite claim withdrawn).
- **Training on, or selecting with, the test teacher texts** -> never; final evaluation only. Pass-2 test reasoning only after the user's review of the 20-window pilot.
- **Windows with a contradicting teacher verdict** (user spotted 00200, 00232 and others, 10 in total) -> removed; `check`-status windows removed in `r1_extra_merge.py`.
- **Network volume for pods** -> not used (63 GB $4.41/mo, pins the datacenter, double cost with the 80 GB container disk); safety against pod loss = background HF sync + restore. The user decides any volume deletion in the console.
- **Pod scheduling tricks / Windows power settings** -> never modify power settings; keep-awake tool lasts ~5 min.
- **Retraining P1aug per seed** (3.5 h, needs the deleted frame-shifted features) -> seeds vary only P1fixR + P2V; stated as a limit.
- **Lower learning rates to delay over-fitting** -> tried in chain B, did not help.

## Unresolved (need the user)
000. **Rewordings: style vs diversity.** A lot of rewordings are rejected by the judge; the user suspects slang / colloquial style rather than added information. Should training text be one uniform format, or does diversity help? Needs the rejection-reason breakdown first, then possibly relaxed rules or the spare (4th+) versions, tested by a training run. Pending answer.
001. **Seeds plan:** 4 new seeds (recommended) or 2; reuse the current pod or a new one. Asked 2026-10-20.
002. **After chain B + seeds:** ensemble of P2V seeds as the final model, a regularised Phase 2 (best epoch is always the first), or the write-up; fusion with the crash model; threshold on validation.
003. **Pass 2 of the test teacher reasoning** after the user's review of the 20-window pilot.
01. **Continuation list** (Phase-1 plan section 5): direct fact signal (structured fields vs auxiliary loss), short reasoning, attention / layer 11 on the new base; ViT-L LoRA fine-tune of the encoder; motion/gap loss.
02. (superseded by 003) test-set teacher reasoning on hold.
03. **Attention-guided object crop** (a second small vision path) for road-user type: allowed only if the user accepts the extra path.
00. **Student retrain on the new dataset:** mix DADA back in or Nexar only; targets and split by video as in PROJECT_STATE TODO 1. (Teacher prompt question resolved: E3 → E6/E7 flow, box kept, Flex.)
0. **Next experiment after the Phase-2 result** (user reads `outputs/r1_week1/pod_phase2_2026-10-06/review/phase2_validation_outputs.md` first): (a) Nexar-focused Phase 2 from the
   Phase-1 checkpoint with explicit verdict / time-to-impact fields (≈1 h), (b) re-weight DADA no-crash to cut the 83 % false-alarm rate, (c) more / richer Nexar text (BADAS-2.0 recipe with
   detection boxes; teacher spend), (d) calibrate the DADA threshold (AUC 0.735 is fine, 0.5 is not).
1. **Volume clean-up:** the old volume `0hnvco2s4j` (64 GB, old runs, 41 GB of Nexar frames) was scanned read-only 2026-10-05; nothing deleted, user said keep for now.
2. (resolved) HF write scope, merger init (random), pod type (RTX PRO 4500, 60 GB disk), Phase-1 no-crash hallucination measured (97 % / 91 % crash phrases).
6. **Enlarge decision after week 1:** MM-AU CAP (needs fps from the authors or estimation), BDD-X (needs BDD100K access), TAU-106K
  (~106 GB YouTube + Bilibili; 12 % unavailable; no hazard-start field; windows must stay inside the scene clip), VRU (needs the
  authors' source mapping).
7. **Unsent access requests:** DRAMA form, WTS form, RoadSafe365 email, MM-AU authors email; Nexar HF gate (full clips).
8. **Later LM arms:** Cosmos-Reason2-2B and Qwen3.5-4B (same interface) and the text-only control — not in week 1.
9. Known minor shift accepted for now: DADA squashed from 2.4:1 to 256×256 is ~1.24× more horizontally compressed than Nexar 16:9.
