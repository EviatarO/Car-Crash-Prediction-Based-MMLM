"""
semsup_common.py
=================
Shared plumbing for the semantic-supervision experiments (B1 / A1 / B):
  - resolve (video_id, TTE) -> frames_dir from the teacher_labels manifests
  - load the 267-row Caption_Train_All_Clips.jsonl as a training set
  - TrainableBadasWrapper: loads BADAS-Open (V-JEPA2 ViT-L), optionally applies
    LoRA to the trunk, and exposes forward() returning (logits, patch_grid)
    WITHOUT detaching patches -> gradients can flow into the (LoRA) trunk.
  - frozen SigLIP text encoder for the semantic targets.
  - a --dry-run-modules helper to print nn_model.named_modules() on the pod
    BEFORE committing to a LoRA target_modules list (BADAS internals are only
    knowable at runtime - see plan risk note).

Reuses e4_stageA_badas_open_eval.py (load_badas, preprocess_clip) and
vjepa_reason.py (ResamplerProjector) rather than reimplementing them.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "student_training" / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "student_training" / "models"))

CAPTIONS_JSONL = PROJECT_ROOT / "outputs" / "semantic_captions" / "Caption_Train_All_Clips.jsonl"
TEACHER_LABELS_GLOB = str(PROJECT_ROOT / "dataset" / "teacher_labels" / "*.jsonl")
TRAIN_FRAMES_ROOT = PROJECT_ROOT / "dataset" / "train"


def _norm_verdict(v):
    if v is None:
        return None
    s = str(v).strip().upper()
    if s in ("1", "YES", "TRUE"):
        return "YES"
    if s in ("0", "NO", "FALSE"):
        return "NO"
    return s or None


# =============================================================================
# Data: resolve frames_dir, load the caption/label training set
# =============================================================================

def build_frames_dir_index() -> dict:
    """(video_id, str(requested_time_to_event)) -> frames_dir, from teacher_labels/*.jsonl."""
    import glob
    idx = {}
    for fp in glob.glob(TEACHER_LABELS_GLOB):
        with open(fp, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                vid, tte, fd = r.get("video_id"), r.get("requested_time_to_event"), r.get("frames_dir")
                if vid and fd:
                    idx[(vid, str(tte))] = fd
    return idx


def load_training_examples(limit: int = 0, require_frames: bool = True) -> list:
    """Load Caption_Train_All_Clips.jsonl, resolve frames_dir, verify 16 frames on
    disk, attach label (0/1 from gt_verdict). Skips unresolvable/missing rows."""
    idx = build_frames_dir_index()
    rows = [json.loads(l) for l in open(CAPTIONS_JSONL, encoding="utf-8") if l.strip()]
    out, skipped = [], 0
    for r in rows:
        key = (r["video_id"], str(r["requested_time_to_event"]))
        fd = idx.get(key)
        if not fd:
            skipped += 1
            continue
        frame_dir = TRAIN_FRAMES_ROOT / fd
        paths = [frame_dir / f"frame_{i:05d}.jpg" for i in range(1, 17)]
        if require_frames and not all(p.exists() for p in paths):
            skipped += 1
            continue
        gt = _norm_verdict(r.get("gt_verdict"))
        if gt not in ("YES", "NO"):
            skipped += 1
            continue
        out.append({
            "video_id": r["video_id"],
            "tte": r["requested_time_to_event"],
            "frames_dir": fd,
            "frame_paths": [str(p) for p in paths],
            "caption": r["caption"],
            "label": 1 if gt == "YES" else 0,
        })
    print(f"[data] loaded {len(out)} examples ({skipped} skipped: unresolved/missing frames)")
    if limit:
        out = out[:limit]
    return out


def clip_level_split(examples: list, val_frac: float = 0.2, seed: int = 0):
    """Split by unique video_id (not by row) so no clip leaks across train/val."""
    import random
    vids = sorted({e["video_id"] for e in examples})
    random.Random(seed).shuffle(vids)
    n_val = max(1, int(len(vids) * val_frac))
    val_vids = set(vids[:n_val])
    train = [e for e in examples if e["video_id"] not in val_vids]
    val = [e for e in examples if e["video_id"] in val_vids]
    return train, val


# =============================================================================
# BADAS (trainable): logits + patch grid WITH gradients
# =============================================================================

class TrainableBadasWrapper:
    """Loads BADAS-Open; optionally wraps nn_model with LoRA (peft). forward()
    returns (logits (1,2), patches (P,D)) where patches keeps its gradient link
    to the trunk (no .detach(), unlike the frozen VJEPA2FeatureExtractor) so a
    semantic loss can backprop into the LoRA-unfrozen ViT-L.
    """

    def __init__(self, stagea_cfg: dict, lora_target_modules: list | None = None,
                 lora_r: int = 16, lora_alpha: int = 32, lora_dropout: float = 0.05):
        from e4_stageA_badas_open_eval import load_badas, preprocess_clip
        self._preprocess_clip = preprocess_clip
        self.vjepa, self.nn_model, self.device = load_badas(stagea_cfg)

        probe = getattr(self.nn_model, "temporal_processor", None)
        if probe is None:
            probe = getattr(self.nn_model, "pooler", None)
        if probe is None:
            for name, mod in self.nn_model.named_modules():
                low = name.lower()
                if ("temporal" in low or low.endswith("pooler")
                        or "probe" in low or "attentive" in low):
                    probe = mod
                    print(f"  [wrapper] hooking probe module by search: '{name}'")
                    break
        if probe is None:
            raise RuntimeError(
                "Could not locate the attentive-probe module on BADAS-Open. "
                "Run --dry-run-modules on the pod and set the tap point manually."
            )
        self._captured = {}

        def _pre_hook(_module, args):
            self._captured["patches"] = args[0]   # NOTE: no .detach() -> keeps grad

        probe.register_forward_pre_hook(_pre_hook)

        self.lora_enabled = lora_target_modules is not None
        if self.lora_enabled:
            from peft import LoraConfig, get_peft_model
            cfg = LoraConfig(
                r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, bias="none",
                target_modules=lora_target_modules,
            )
            self.nn_model = get_peft_model(self.nn_model, cfg)
            trainable = sum(p.numel() for p in self.nn_model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.nn_model.parameters())
            print(f"  [wrapper] LoRA applied: trainable={trainable:,} / total={total:,} "
                  f"({100*trainable/total:.2f}%)")
            if trainable == 0:
                raise RuntimeError(
                    f"LoRA target_modules={lora_target_modules} matched ZERO parameters. "
                    "Re-run --dry-run-modules and pick real module name substrings."
                )
        else:
            for p in self.nn_model.parameters():
                p.requires_grad = False
            self.nn_model.eval()

    def forward(self, frame_paths: list):
        clip = self._preprocess_clip(self.vjepa, frame_paths).to(self.device)
        self._captured.clear()
        logits = self.nn_model(clip)                    # (1, 2) - grads flow if LoRA on
        patches = self._captured.get("patches")
        if patches is None:
            raise RuntimeError("probe pre-hook did not fire - tap point is wrong.")
        return logits, patches[0]                        # (1,2), (P, D)


def dry_run_modules(cfg_path: str, out_path: str):
    """Load BADAS (no LoRA), dump nn_model.named_modules() so the real LoRA
    target_modules list can be chosen before any training run. No training."""
    import yaml
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    from e4_stageA_badas_open_eval import load_badas
    _, nn_model, _ = load_badas(cfg)
    with open(out_path, "w", encoding="utf-8") as f:
        for name, mod in nn_model.named_modules():
            f.write(f"{name}\t{type(mod).__name__}\n")
    print(f"[dry-run] wrote module list -> {out_path}")
    print("[dry-run] look for Linear layers inside attention blocks (e.g. containing "
          "'qkv'/'q_proj'/'k_proj'/'v_proj'/'proj'/'fc1'/'fc2') and pass their common "
          "substring(s) as --lora-target-modules to semsup_train.py")


# =============================================================================
# SigLIP (frozen) text encoder for semantic targets
# =============================================================================

def load_siglip(model_id: str = "google/siglip-base-patch16-224", device: str = "cuda"):
    import torch
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, tok


def siglip_text_embed(texts: list, siglip_model, tokenizer, device: str):
    """L2-normalized (B, Dt) SigLIP text embeddings. Always frozen/no_grad."""
    import torch
    inputs = tokenizer(texts, padding="max_length", truncation=True,
                        max_length=64, return_tensors="pt").to(device)
    with torch.no_grad():
        out = siglip_model.get_text_features(**inputs)
        # Some transformers versions wrap this in an output object instead of
        # returning a plain tensor - handle both shapes robustly.
        if torch.is_tensor(out):
            feats = out
        elif hasattr(out, "text_embeds") and out.text_embeds is not None:
            feats = out.text_embeds          # projected shared vision-text space (preferred)
        elif hasattr(out, "pooler_output") and out.pooler_output is not None:
            feats = out.pooler_output         # raw pooled, unprojected (fallback)
        else:
            feats = out.last_hidden_state[:, 0]
        feats = torch.nn.functional.normalize(feats, dim=-1)
    return feats


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run-modules", action="store_true")
    ap.add_argument("--config", default=str(PROJECT_ROOT / "student_training" / "configs" / "e4_stageA.yaml"))
    ap.add_argument("--out", default=str(PROJECT_ROOT / "outputs" / "semantic_captions" / "badas_named_modules.txt"))
    args = ap.parse_args()
    if args.dry_run_modules:
        dry_run_modules(args.config, args.out)
    else:
        examples = load_training_examples(require_frames=True)
        tr, va = clip_level_split(examples)
        print(f"train={len(tr)} val={len(va)} (clip-level split)")
