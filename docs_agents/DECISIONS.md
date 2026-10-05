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

## Unresolved (need the user)
1. **Open the data-prep pod** (recommended A40 48 GB, ≥ 150 GB container disk, volume attached) and send the SSH command; later training pod
   L40S (or A100 SXM). Claude cannot start pods.
2. **Volume clean-up list:** what may be deleted after the read-only scan (nothing without explicit approval).
3. **HF token scope:** can the fine-grained token create private repos? If not, the user adds the scope.
4. **Phase-1 no-crash hallucination:** if the rate is high after Phase 1 → re-weight Phase 2 or add no-crash windows to Phase 1 at ~20 %?
5. **Merger init:** random vs Qwen weights (A/B result decides).
6. **Enlarge decision after week 1:** MM-AU CAP (needs fps from the authors or estimation), BDD-X (needs BDD100K access), TAU-106K
  (~106 GB YouTube + Bilibili; 12 % unavailable; no hazard-start field; windows must stay inside the scene clip), VRU (needs the
  authors' source mapping).
7. **Unsent access requests:** DRAMA form, WTS form, RoadSafe365 email, MM-AU authors email; Nexar HF gate (full clips).
8. **Later LM arms:** Cosmos-Reason2-2B and Qwen3.5-4B (same interface) and the text-only control — not in week 1.
9. Known minor shift accepted for now: DADA squashed from 2.4:1 to 256×256 is ~1.24× more horizontally compressed than Nexar 16:9.
