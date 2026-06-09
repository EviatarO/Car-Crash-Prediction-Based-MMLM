# AGENTS.md

## Project Snapshot
- This repo implements a Teacher -> Student collision-anticipation pipeline over 16-frame Nexar clips.
- Teacher stage (`teacher_distillation/scripts/Teacher_dataset_distill_v11.py`) calls OpenRouter Gemini twice (Pass-1 `PROMPT_G`, Pass-2 debate only on mismatches) and writes `dataset/teacher_labels/teacher_dataset_v11.jsonl` + `.xlsx`.
- Student stage runs either zero-shot (`student_training/scripts/zero_shot_eval.py`) or LoRA fine-tuning (`student_training/scripts/train_lora.py`) on InternVL3.5-4B-Flash.
- Primary thesis metric is Average Precision (AP), computed in `student_training/scripts/evaluate_metrics.py`.

## Repo Map (Where To Work)
- Prompts live in one source of truth: `prompts/templates.py` (`PROMPT_G`, `PROMPT_DEBATE_G`).
- Teacher data prep: `teacher_distillation/scripts/build_teacher_manifest.py` -> manifest JSONL with `frame_indices` + labels.
- Student data prep: `student_training/scripts/build_test_manifest.py` -> `dataset/manifests/test_manifest_private.jsonl`.
- Training dataset/token wiring: `student_training/data/collision_dataset.py`.
- Model wrapper + LoRA + ScoreHead + multi-GPU fixes: `student_training/models/internvl_lora.py`.
- Evaluation/review artifacts: `student_training/scripts/evaluate_metrics.py` and `student_training/scripts/zero_shot_to_xlsx.py`.

## Data Contracts You Must Preserve
- Clip format is fixed across scripts: 16 frames, stride 4, filename pattern `frame_{:05d}.jpg`.
- Zero-shot and trained eval outputs intentionally share schema (`video_id`, `ground_truth`, `score`, `collision_verdict`, `verdict_reasoning`, `parse_error`, etc.). Keep this stable so downstream scripts remain plug-compatible.
- Teacher JSONL fields (`final_reasoning`, Pass-1 `scene_context/...`, Pass-2 `p2_*`) are consumed by `CollisionDataset`; do not rename without updating dataset parsing.
- `CollisionDataset` inserts `<IMG_CONTEXT>` token IDs directly into `input_ids`; this depends on `_probe_and_set_img_context_token_id()` in `internvl_lora.py`.

## Critical Workflows (Known-Good Commands)
- Install deps: `pip install -r requirements.txt` (plus optional `pip install flash-attn --no-build-isolation` on GPU hosts).
- Build test manifest: run `python student_training/scripts/build_test_manifest.py --test_csv ... --metadata_csv ... --frames_root ... --output dataset/manifests/test_manifest_private.jsonl`.
- Zero-shot baseline: run `python student_training/scripts/zero_shot_eval.py --manifest ... --frames_root ... --output outputs/e0_zero_shot_baseline/*.jsonl --config student_training/configs/zero_shot.yaml`.
- Metrics + plots: run `python student_training/scripts/evaluate_metrics.py --results <jsonl> --out_dir outputs/<experiment>/metrics/<tag> --tag <name>`.
- LoRA training: run `python student_training/scripts/train_lora.py --jsonl dataset/teacher_labels/teacher_dataset_v11.jsonl --frames_root ... --config student_training/configs/train_lora.yaml [--resume]`.
- Trained eval: run `python student_training/scripts/trained_eval.py --checkpoint outputs/checkpoints/e2_lora_100clips/step_XXXXXX --manifest dataset/manifests/test_manifest_private.jsonl --frames_root ... --output outputs/trained/*.jsonl --config student_training/configs/train_lora.yaml`.

## Project-Specific Conventions
- Reuse preprocessing exactly (`build_transform` with ImageNet mean/std at 448x448) across training and inference.
- Resume behavior is file-based in long jobs (`--resume` reads existing output JSONL/checkpoints and skips completed clips).
- Mismatch review convention: Excel writers in teacher/student highlight bad rows fully red (`PatternFill("FFFF9999")`).
- Training is intentionally overfit-style (`num_epochs: 50`); best checkpoint is selected from `epoch_metrics.jsonl` by validation metrics, not final epoch.
- `train_lora.py` saves checkpoint folders as `step_XXXXXX/` containing LoRA adapter files, `score_head.pt`, and `training_state.pt`; `trained_eval.py` expects this layout.

## Folder Structure Convention (effective 2026-05-26)
- `dataset/` — raw data (frames in train/ and test/), ground-truth labels (GT xlsx files), teacher-generated labels (`teacher_labels/`), and all manifest JSONL files (`manifests/`).
- `outputs/<experiment_name>/` — experiment results only, organized by experiment code: `e0_zero_shot_baseline/`, `prompt_experiment/`, `prompt_bakeoff/`, `apo/`, `e2_lora_100clip_pilot/`, `e3a_student_90clips/`, `checkpoints/`, `archives/`. Each subfolder has a README.md describing the experiment.
- `reports/` — human-readable documents: `presentations/` (.pptx), `progress_reports/` (.docx/.pdf/.md), `status/` (external files), `figures/`, `_scripts/` (report generators). Report files use `YYYY-MM_short-description` naming.
- `prompts/` — active prompt templates (source of truth: `templates.py`). Historical versions in `old prompts/`.
- When creating new experiment outputs, create a new subfolder under `outputs/` with a README.md. Never dump loose files in `outputs/` root.
- When creating new reports, place in the appropriate `reports/` subfolder and use the `YYYY-MM_short-description` naming convention.

## Integrations and Environment
- Teacher API secrets come from `.env` (`OPENROUTER_API_KEY`, optional `OPENROUTER_HTTP_REFERER`, `OPENROUTER_APP_TITLE`; see `.env.example`).
- Student model defaults to `OpenGVLab/InternVL3_5-4B-Flash` (configs in `student_training/configs/zero_shot.yaml` and `student_training/configs/train_lora.yaml`).
- Multi-GPU behavior is patched in `internvl_lora.py` (CPU `image_flags`, projector-device hook); keep these patches when refactoring distributed loading.
- Large generated outputs/checkpoints are mostly gitignored (`.gitignore`), while manifests and key metrics JSON are meant to stay trackable.

