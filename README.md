# Car-Crash Prediction Based MMLM

**MSc Thesis Project — Explainable Collision Anticipation using Teacher-Student Knowledge Distillation**

> Predict the probability of a collision event within a 0–3 second horizon from dashcam footage, and provide a human-readable chain-of-thought (CoT) justification for each prediction.

---

## Project Overview

This project implements a **Teacher-Student distillation pipeline** for collision anticipation:

- **Teacher** (`gemini-3.1-pro-preview` via OpenRouter) — Generates high-fidelity CoT reasoning using ground-truth labels and a Semi-Oracle Debate architecture (Pass-1 blind prediction + Pass-2 correction on mismatches).
- **Student** (`InternVL3.5-4B-Flash`) — A compact 4.7B VLM fine-tuned on the Teacher's outputs using LoRA. Produces both a collision probability score and structured reasoning text.

**Dataset:** Nexar large-scale dashcam dataset (1,500 training videos, 1,344 test videos).

**Success metric:** Average Precision (AP) > Teacher zero-shot baseline.

---

## Repository Structure

```
MMLM_AI/
│
├── teacher_distillation/           # Teacher pipeline (Gemini API distillation)
│   ├── scripts/
│   │   ├── Teacher_dataset_distill_v11.py   # Latest distillation script (v11, 100 clips)
│   │   ├── build_teacher_manifest.py        # Builds clip manifests for distillation
│   │   ├── prompt_experiment*.py            # Prompt ablation experiments
│   │   ├── teacher_benchmark.py             # Benchmarks teacher output quality
│   │   ├── benchmark_analysis.py            # Analyses benchmark results
│   │   ├── jsonl_to_excel.py                # Converts JSONL outputs to Excel
│   │   ├── generate_report*.py              # Progress and summary reports
│   │   └── fix_jsonl.py                     # Repairs malformed JSONL files
│   ├── teacher/
│   │   └── distill.py                       # Core teacher distillation logic
│   └── slurm/                               # HPC SLURM job scripts
│
├── student_training/               # Student model (InternVL3.5-4B-Flash)
│   ├── configs/
│   │   └── zero_shot.yaml                   # Config for zero-shot evaluation
│   ├── scripts/
│   │   ├── build_test_manifest.py           # Builds test clip manifest from test.csv
│   │   ├── zero_shot_eval.py                # Zero-shot evaluation (no training)
│   │   └── evaluate_metrics.py             # Metrics + graphs (AP, AUC, CM, etc.)
│   ├── models/                              # Student model architecture (to be built)
│   ├── data/                                # Dataset loaders (to be built)
│   └── inference/                           # Inference utilities (to be built)
│
├── prompts/                        # Shared prompt templates (Teacher + Student)
│   ├── templates.py                         # All prompt versions (A–K, PROMPT_G, DEBATE_G)
│   └── PROMPT_G.py                          # Standalone PROMPT_G export
│
├── outputs/                        # All generated outputs (gitignored for large files)
│   ├── teacher_dataset_v11.jsonl            # Latest teacher distillation (100 clips)
│   ├── manifest_v11_100clips.jsonl          # Clip manifest for 100 teacher clips
│   ├── test_manifest_private.jsonl          # Test manifest (677 Private clips)
│   ├── zero_shot/                           # Zero-shot evaluation results
│   └── metrics/                             # Metric JSON + graphs per experiment
│
├── reports/                        # Thesis documents
├── requirements.txt                # Python dependencies
├── .env                            # API keys (not committed)
└── .env.example                    # Template for .env
```

---

## Architecture

### Teacher Pipeline (Semi-Oracle Debate)

```
Nexar clip (16 frames, stride=4, ~2s window)
        |
        v
  [Pass-1: PROMPT_G]  ──> Gemini-3.1-pro-preview
        |
   Match GT? ──YES──> Use Pass-1 reasoning
        |
        NO
        |
  [Pass-2: PROMPT_DEBATE_G]  ──> Gemini (argue correct position)
        |
        v
  final_verdict + final_reasoning  ──> teacher_dataset_v11.jsonl
```

### Student Architecture (InternVL3.5-4B-Flash, Native Multi-Image)

```
16 frames  (448x448 each)
    |
    v
[InternViT-300M]  FROZEN  -- per-frame patch extraction
    |
[ViR Compression]  FROZEN  -- 64 tokens/frame (pixel unshuffle)
    |   16 separate <image> blocks
[Projector]  TRAINED  -- vision_dim -> LLM_dim
    |
[Qwen3-4B LLM + LoRA]  -- causal attention over all tokens
    |                  \
    v                   v
[Score Head]        [Reasoning Generation]
 TRAINED             LoRA-adapted LM head
    |                   |
P(collision)        CoT reasoning text
  in [0,1]          (mirrors PROMPT_G structure)

Loss = alpha * BCE(score, target) + (1-alpha) * CE(reasoning, teacher_text)
```

**Key design decisions:**
- Native multi-image mode — no custom TemporalMixer (preserves frame boundaries for PROMPT_G's frame-range comparisons)
- Binary `target` labels (0/1), not soft `target_risk`
- `final_reasoning` from JSONL used as reasoning supervision (auto-selects Pass-1 or Pass-2)
- LoRA: r=16, alpha=32, dropout=0.1, targets: `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`

---

## Setup

### Prerequisites

- Python 3.10+
- GPU with >= 16GB VRAM for training (RunPod RTX 4090 recommended)
- GPU with >= 10GB VRAM for zero-shot inference

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Environment Variables

Copy `.env.example` to `.env` and fill in:

```bash
cp .env.example .env
```

```
OPENROUTER_API_KEY=your_key_here        # For Teacher distillation (Gemini via OpenRouter)
OPENROUTER_HTTP_REFERER=your_project    # Optional referer header
```

---

## Usage

### Step 1 — Build Test Manifest

Builds a JSONL manifest of test clips (677 Private videos from test.csv):

```bash
python student_training/scripts/build_test_manifest.py \
  --test_csv      /data/Nexar_DataSet/test.csv \
  --metadata_csv  /data/test_frames_metadata.csv \
  --frames_root   /data/Nexar_DataSet/test_frames256 \
  --output        outputs/test_manifest_private.jsonl
```

**Output fields per record:**
- `video_id` — zero-padded 5-digit ID
- `event_occurs` — 0 (no collision) or 1 (collision)
- `group` — 0=0.5s, 1=1.0s, 2=1.5s before event (Nexar pre-cut timing)
- `frame_indices` — 16 frame indices (last window, stride=4)

### Step 2 — Run Zero-Shot Evaluation (RunPod)

Evaluates InternVL3.5-4B-Flash with no training using PROMPT_G:

```bash
# On 100 teacher clips (training set baseline)
python student_training/scripts/zero_shot_eval.py \
  --manifest     outputs/manifest_v11_100clips.jsonl \
  --frames_root  /data/train_frames256 \
  --output       outputs/zero_shot/zero_shot_teacher_100.jsonl \
  --config       student_training/configs/zero_shot.yaml

# On 677 test clips (held-out test baseline)
python student_training/scripts/zero_shot_eval.py \
  --manifest     outputs/test_manifest_private.jsonl \
  --frames_root  /data/test_frames256 \
  --output       outputs/zero_shot/zero_shot_test.jsonl \
  --config       student_training/configs/zero_shot.yaml
```

Add `--resume` to continue an interrupted run.
Add `--load_in_4bit` for GPUs with < 10GB VRAM.

### Step 3 — Compute Metrics & Generate Graphs

```bash
python student_training/scripts/evaluate_metrics.py \
  --results  outputs/zero_shot/zero_shot_test.jsonl \
  --out_dir  outputs/metrics/zero_shot_test \
  --tag      "Zero-Shot Baseline"
```

**Outputs:**
| File | Description |
|------|-------------|
| `metrics.json` | All scalar metrics (AP, AUC, F1, CM, etc.) |
| `confusion_matrix.png` | 2x2 heatmap with counts and % |
| `roc_curve.png` | ROC curve with AUC annotation |
| `pr_curve.png` | Precision-Recall curve with AP annotation |
| `score_distribution.png` | Histogram: positive vs negative score distributions |
| `group_ap_bar.png` | AP per time-to-event group (0.5s / 1.0s / 1.5s) |
| `examples_table.txt` | Top 5 correct + top 5 wrong predictions |

---

## Configuration

### `student_training/configs/zero_shot.yaml`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_id` | `OpenGVLab/InternVL3_5-4B-Flash` | HuggingFace model ID |
| `load_in_4bit` | `false` | 4-bit quantization (use for < 10GB VRAM) |
| `torch_dtype` | `bfloat16` | Model weight precision |
| `window_size` | `16` | Frames per clip |
| `stride` | `4` | Frame sampling stride |
| `frame_size` | `448` | Input resolution (InternVL native) |
| `max_new_tokens` | `600` | Max tokens in model response |
| `temperature` | `0.0` | Greedy decoding for reproducibility |

---

## Experiments

### Experiment Sequence

| Experiment | Description | Comparison baseline |
|------------|-------------|---------------------|
| **E0** | Zero-shot (no training) on test set | — (starting point) |
| **E1** | Zero-shot on 100 teacher clips | E0 (same model, train-set clips) |
| **E2** | Fine-tuned on 100 clips | E0 (improvement from training) |
| **E3** | Fine-tuned on 200 clips | E2 (scaling signal) |
| **E4** | Fine-tuned on 500 clips | E3 (target scale) |
| **E5** | Ablation: LoRA rank r={4,8,16,32} | E4 |
| **E6** | Ablation: loss alpha={0.3,0.5,0.7,1.0} | E4 |
| **E7** | Ablation: +TemporalMixer | E4 |

### Key Thesis Graphs

1. **AP vs Dataset Size** — scaling curve (E1→E4)
2. **ROC curves** — overlay zero-shot vs trained models
3. **Confusion matrix heatmaps** — zero-shot vs best trained
4. **Training loss curves** — L_score + L_reasoning
5. **Score distribution** — positive vs negative clip histograms
6. **Group AP bar** — AP at 0.5s / 1.0s / 1.5s before event
7. **Ablation bar charts**

---

## RunPod Workflow

```
[Local PC]                        [RunPod Pod — RTX 4090]
──────────────────────────────    ────────────────────────────────
1. Write / edit code          →
2. git push                   →   3. git pull (latest code)
                                  4. pip install -r requirements.txt
                                  5. Model auto-downloads from HuggingFace
                                     (cached on Network Volume, ~10GB)
                                  6. Frames already on Network Volume
                                  7. python zero_shot_eval.py ...
                                  8. python evaluate_metrics.py ...
← 9. git pull results JSONL       (commit + push outputs)
10. View graphs locally
```

**Cost estimate:**
- Network Volume (50GB): ~$2.50/month
- Zero-shot run (2 hrs): ~$0.88
- Training run (3–4 hrs): ~$1.30–1.75
- Full experiment set (~30 runs): ~$40–75

---

## Teacher Dataset Versions

| Version | Clips | Model | Key Feature |
|---------|-------|-------|-------------|
| v2 | 18 | gemini-3-pro | First blind eval (PROMPT_G) |
| v8 | 18 | gemini-3-pro | Semi-Oracle Debate architecture |
| v9 | 18 | gemini-3.1-pro | Temperature=0.1, improved TN recall |
| v10 | 18 | gemini-3.1-pro | Refined prompts |
| **v11** | **100** | **gemini-3.1-pro** | **Current — scaling to 500+** |

---

## Hardware Requirements

| Task | Min VRAM | Recommended |
|------|----------|-------------|
| Zero-shot inference | 10 GB | RTX 4090 (24GB) |
| Fine-tuning (LoRA) | 14 GB | RTX 4090 (24GB) |
| Fine-tuning (QLoRA 4-bit) | 10 GB | RTX 4090 (24GB) |
| Data prep / metrics | CPU only | Local PC |

*Local PC (RTX 1000 Ada, 6GB): sufficient for data preparation and metrics only.*

---

## Citation / Acknowledgements

- InternVL3.5: [OpenGVLab/InternVL](https://github.com/OpenGVLab/InternVL)
- Nexar Dataset: [nexar.com](https://nexar.com)
- LoRA: Hu et al., 2021 — *LoRA: Low-Rank Adaptation of Large Language Models*
