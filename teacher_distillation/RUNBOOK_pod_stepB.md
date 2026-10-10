# Runbook — Step B on a RunPod pod (tracks + A1 attention for the 4,203 teacher windows)

Plan: `~/.claude/plans/CCP based BADAS/2026-10-10_Plan-Teacher-Full-Run.md`. Driver: `student_training/scripts/r1_teacher_pod.sh <stage>`.
The pod does the GPU work only and returns two small archives (`tracks.tar.gz`, `attn.tar.gz`) through the private HF repo
`eviatarO-org/teacher-dataset-2026-10`. The PC then draws the boxes on its own full-resolution frames and runs the teacher.

**Pod:** 24–48 GB GPU (RTX PRO 4500 / A40 / L40S), container disk 80 GB, no network volume, PyTorch template.
**Pod environment variables:** `KAGGLE_USERNAME`, `KAGGLE_KEY` (Kaggle → Settings → API; accept the Nexar competition rules on your account first).
**Bundle (from the PC):** `outputs/teacher_dataset_2026-10/pod_bundle.tar.gz` -> `/root/r1`, `windows.jsonl` -> `/root/teacher_dataset/windows.jsonl`,
HF token -> `/root/.cache/huggingface/token`.

| stage | what it does | check |
|---|---|---|
| `setup` | venv with the pinned versions (transformers 5.8.0, peft 0.19.1, trackers, kaggle) | torch / GPU name printed; windows.jsonl has 4,203 lines |
| `kaggle_check` | read-only: token accepted? file layout of the competition | `train/<id>.mp4` names (else set `NEXAR_PREFIX`) |
| `videos` | the 1,401 needed train videos, one file per call, 8 in parallel, resumable | "videos present: 1401 of 1401" |
| `cut` | the 4,203 windows at native resolution (JPEG q95), 4 processes | 4203 window folders |
| `smoke` | tracks + attention for 3 videos; prints seconds per video / window | read before the full run |
| `tracks_attn 3` | 3 parallel processes: tracks then attention, resumable | tracks 1401, attn 4203 |
| `push` | `tracks.tar.gz`, `attn.tar.gz`, `windows.jsonl`, logs -> private HF repo | files listed, repo private |
| `check_stop` / `stop` | can the pod stop itself; stop it (only after `push.done`) | |

Local check before any pod (done 2026-10-10): the same cut → tracks → attention path reproduced the pilot's numbers for video 00013
(A1 P(collision), top track and track counts identical) and ran from a clean extract of the bundle without the `dataset/` folder.
