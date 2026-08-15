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

DEFAULT_LABEL_FILES = ["teacher_dataset_e3b.jsonl"]  # covers all 267 caption keys (verified)


def _norm_tte(tte) -> str:
    """Normalize a TTE value to a stable string key. Numeric TTEs (0.5, 1, 1.0, 1.5)
    must collapse to one key regardless of how a given writer serialized the float
    ('1' vs '1.0') - otherwise two label files can silently disagree on the same
    clip+TTE without ever comparing equal. Non-numeric TTEs (offset labels like
    '-4.0_offset', 'TN_MIDPOINT') pass through unchanged."""
    try:
        return str(float(tte))
    except (TypeError, ValueError):
        return str(tte)


def build_frames_dir_index(label_files: list | None = None) -> dict:
    """(video_id, _norm_tte(requested_time_to_event)) -> frames_dir.

    Reads only `label_files` (default: DEFAULT_LABEL_FILES, which alone covers all
    267 current caption keys) rather than globbing every file under
    dataset/teacher_labels/ - that directory holds 28 files across many experiment
    generations, and blindly merging all of them risks a silent collision the
    moment two files disagree on the same (video_id, TTE). Pass label_files=None
    and TEACHER_LABELS_GLOB is still available for an explicit full-glob call.

    Raises on a genuine conflict (same key, different frames_dir across files) -
    a silently-resolved-wrong frames_dir is a much worse failure than a crash.
    """
    if label_files is None:
        label_files = DEFAULT_LABEL_FILES
    fps = [str(PROJECT_ROOT / "dataset" / "teacher_labels" / name) for name in label_files]

    idx: dict = {}
    source: dict = {}
    for fp in fps:
        with open(fp, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                vid, tte, fd = r.get("video_id"), r.get("requested_time_to_event"), r.get("frames_dir")
                if not vid or not fd:
                    continue
                key = (vid, _norm_tte(tte))
                prev = idx.get(key)
                if prev is not None and prev != fd:
                    raise ValueError(
                        f"frames_dir conflict for {key}: {prev!r} (from {source[key]}) "
                        f"vs {fd!r} (from {fp})"
                    )
                idx[key] = fd
                source[key] = fp
    return idx


def load_training_examples(limit: int = 0, require_frames: bool = True,
                            captions_path=None) -> list:
    """Load a caption-schema JSONL (default: Caption_Train_All_Clips.jsonl),
    resolve frames_dir, verify 16 frames on disk, attach label (0/1 from
    gt_verdict). Skips unresolvable/missing rows.

    captions_path: override the caption file (e.g. one of the prompt-bakeoff
    arm_{a,b,c}.jsonl files - see semsup_caption_qa.py). If a row already
    carries an explicit 'frames_dir' field (the bakeoff arm files always do,
    since they're built directly from the sampler's manifest), that value is
    used AS-IS instead of going through build_frames_dir_index(). This matters
    because the default index only covers teacher_dataset_e3b.jsonl's 267 keys
    - a fresh distinct-video sample drawn from other teacher_labels generations
    would not resolve through it, and merging all 29 label files into one
    index risks the exact silent-collision failure build_frames_dir_index()
    was written to prevent (see its docstring). Rows WITHOUT an explicit
    'frames_dir' (the original 267-caption file) are resolved exactly as
    before - this is a strict superset of the old behavior, not a change to it.
    """
    path = Path(captions_path) if captions_path else CAPTIONS_JSONL
    idx = build_frames_dir_index()
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    out, skipped = [], 0
    for r in rows:
        fd = r.get("frames_dir")
        if not fd:
            key = (r["video_id"], _norm_tte(r["requested_time_to_event"]))
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

        def _post_hook(_module, _args, output):
            # The probe's OUTPUT: the single pooled vector the classifier consumes.
            # Verified 2026-08-13 on BADAS-Open: input (1, 2560, 1024) -> output
            # (1, 1024), i.e. all 2560 spatiotemporal tokens collapse to ONE vector.
            # That vector is the entire basis for the crash decision, so anything a
            # semantic loss shapes OUTSIDE it is invisible to the classifier - which
            # is exactly what the `pooled` tap in semsup_b1_probe.py measures.
            # No .detach() here either, for the same reason as the pre-hook.
            self._captured["pooled"] = output[0] if isinstance(output, (tuple, list)) else output

        probe.register_forward_pre_hook(_pre_hook)
        probe.register_forward_hook(_post_hook)

        self.lora_enabled = lora_target_modules is not None
        if self.lora_enabled:
            from peft import LoraConfig, get_peft_model

            # peft accepts EITHER a list of name-substrings OR a single regex string.
            # Plain substrings are dangerous on this model: "query,key,value" matches
            # 108 Linears, not the 72 intended - 72 under backbone.encoder.layer.{0-23}
            # (wanted) plus 36 under backbone.predictor.layer.{0-11}, the V-JEPA2
            # latent-forecast head used during SSL pretraining and NOT part of the
            # classification path. That is 442,368 of 2,801,664 LoRA params (15.8%)
            # either receiving no gradient at all or adapting a module irrelevant to
            # the crash task. Passing a regex scopes it to the encoder stack.
            targets = lora_target_modules
            if isinstance(targets, str):
                print(f"  [wrapper] LoRA target regex: {targets}")
            cfg = LoraConfig(
                r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, bias="none",
                target_modules=targets,
            )
            self.nn_model = get_peft_model(self.nn_model, cfg)
            trainable = sum(p.numel() for p in self.nn_model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.nn_model.parameters())

            # Report WHERE the adapters landed, not just how many there are. The bare
            # count cannot distinguish "72 encoder modules" from "72 encoder + 36
            # predictor", which is exactly the ambiguity that hid the 15.8% above.
            from collections import Counter
            hit = Counter()
            for name, _ in self.nn_model.named_modules():
                if name.endswith("lora_A.default"):
                    base = name.split(".lora_A")[0]
                    if ".encoder.layer." in base:
                        hit["backbone.encoder"] += 1
                    elif ".predictor.layer." in base:
                        hit["backbone.predictor"] += 1
                    else:
                        hit["other"] += 1
            print(f"  [wrapper] LoRA applied: trainable={trainable:,} / total={total:,} "
                  f"({100*trainable/total:.2f}%)")
            print(f"  [wrapper] adapters by stack: {dict(hit)}")
            if hit.get("backbone.predictor"):
                print(f"  [wrapper] NOTE: {hit['backbone.predictor']} adapters are on the "
                      f"V-JEPA2 predictor (latent-forecast) stack, not the encoder. "
                      f"Pass a regex like "
                      f"r'backbone\\.encoder\\.layer\\.\\d+\\.attention\\.(query|key|value)' "
                      f"to scope them out.")
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
        return self.forward_clip(clip)

    def forward_clip(self, clip):
        """Run the model on an ALREADY-preprocessed clip tensor (moved to
        self.device by the caller). Split out of forward() so a prefetch
        pipeline (see prefetch_clips()) can do the preprocessing - file read +
        JPEG decode + resize/normalize - in background threads while the GPU
        works on the previous window's forward_clip() call, instead of doing
        preprocessing and GPU compute serially for every window.

        WHY THIS MATTERS (measured 2026-08-11 on a real pod via direct
        profiling, not inferred from utilization graphs): per window, raw file
        read = ~670ms, +decode/resize = ~503ms, +GPU forward = ~0ms
        (unmeasurable above noise). Preprocessing is ~100% of per-window wall
        time on a slow network volume; GPU compute is negligible by
        comparison. Overlapping decode with GPU compute alone (the originally
        planned fix) would therefore buy almost nothing - the actual fix is
        PARALLELIZING the I/O itself across multiple threads, which is what
        prefetch_clips() does."""
        self._captured.clear()
        logits = self.nn_model(clip)                    # (1, 2) - grads flow if LoRA on
        patches = self._captured.get("patches")
        if patches is None:
            raise RuntimeError("probe pre-hook did not fire - tap point is wrong.")
        return logits, patches[0]                        # (1,2), (P, D)

    def prefetch_clips(self, examples, num_workers=8, prefetch=16, key="frame_paths"):
        """Yields (i, ex, clip_or_None, error_or_None) in ORDER over `examples`,
        with `num_workers` background threads doing preprocessing (file read +
        decode + resize) concurrently, `prefetch` windows ahead of what the
        caller has consumed.

        WHY THREADS GIVE REAL CONCURRENCY DESPITE THE GIL: file I/O (open/
        read) and PIL's JPEG decompression both release the GIL during their
        C-level work, so ThreadPoolExecutor gives genuine wall-clock
        concurrency here even though CPython can't parallelize pure-Python
        bytecode. This is not multiprocessing and needs no pickling/IPC
        overhead.

        ERROR HANDLING matches the caller's existing per-window try/except
        (OSError, RuntimeError): continue contract exactly - a failed window
        yields (i, ex, None, exc) instead of raising, so one bad frame can't
        kill the whole prefetch pipeline or silently desync the futures dict."""
        from concurrent.futures import ThreadPoolExecutor

        def _one(ex):
            try:
                return self._preprocess_clip(self.vjepa, ex[key]), None
            except (OSError, RuntimeError) as e:
                return None, e

        n = len(examples)
        if num_workers <= 0:
            # Debugging escape hatch: fully serial, no thread pool at all -
            # exactly the old inline behavior, for isolating whether a bug is
            # in the pipeline itself vs. the underlying preprocessing/model.
            for i, ex in enumerate(examples):
                clip, err = _one(ex)
                yield i, ex, clip, err
            return
        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            futures = {}
            next_submit = 0
            for i in range(min(prefetch, n)):
                futures[i] = pool.submit(_one, examples[i])
                next_submit = i + 1
            for i in range(n):
                clip, err = futures.pop(i).result()
                if next_submit < n:
                    futures[next_submit] = pool.submit(_one, examples[next_submit])
                    next_submit += 1
                yield i, examples[i], clip, err


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
