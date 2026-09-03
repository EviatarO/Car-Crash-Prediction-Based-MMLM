"""
extract_token_heatmaps.py
=========================
Spatial/temporal heat maps over the trunk's (1, 2560, 1024) token grid, for one or
more LoRA checkpoints, so the semantic branch's effect can be located IN THE FRAME
rather than only measured in direction space.

WHY THIS EXISTS (read before running - it is not a repeat of P3)
---------------------------------------------------------------
Two existing results already answer "does the semantic gradient reach the classifier":
  - B1 pooled-tap probe: caption info survives the 2560->1 compression (retrieval
    9.95% at `pooled` vs 14.03% at `patches`, chance 0.45%), and pooled ~= meanpool,
    so crash-tuned attention is not selectively discarding caption directions.
  - P3: ||dPooled||/||dPatches|| = 0.00341 real vs 0.00186 for an equal-norm random
    perturbation, paired 95% CI [0.00143, 0.00163], excludes zero.
Conclusion on record: the signal reaches the decision path, it just does not help
there. Changing the TAP POINT is therefore already-tested and rejected.

What has never been measured is WHERE - which spatial positions and which of the 8
temporal slices carry the crash decision, and whether the semantic term moves those
same tokens or different ones. That is what this script extracts.

WHAT IT WRITES (per clip, per checkpoint) - all flat over the P tokens, reshaped for
display by the consumer using the `layout` block:
  attn  (P,)  attention weight the crash head's attentive probe puts on each token.
              "Where the model looks." Only present if the probe exposes or permits
              reconstruction of its attention - see --dump-modules.
  proj  (P,)  <h_i, pooled> / ||pooled||: how much each token projects onto the exact
              vector the classifier reads. Always available (uses the existing hooks).
              Deliberately NOT the raw token norm: ViT token norms are dominated by
              high-norm register/artifact tokens in background regions (Darcet et al.,
              ICLR 2024), which would produce a confident-looking but meaningless map.
  drep  (P,)  per PAIR of checkpoints, ||h_B[i] - h_A[i]||: where the representation
              actually moved. Computed for every ordered pair so the consumer can
              choose any baseline.

THE TOKEN LAYOUT IS DISCOVERED, NOT ASSUMED
-------------------------------------------
Nothing in this repo has ever reshaped these 2560 tokens, and the available sources
disagree about the input geometry: badas_loader.py loads backbone
`vjepa2-vitl-fpc16-256-ssv2` (-> 8 x 16x16 = 2048 tokens), e4_stageA.yaml sets
img_size 224 (-> 8 x 14x14 = 1568) while its own comment says squash-to-256, and the
architecture figure says (1,16,3,256,320) (-> 8 x 16x20 = 2560, the only value that
matches the measured token count). `VJEPAModel(use_sliding_window=True,
window_stride=1)` also leaves open that P is a concatenation of windows rather than
one grid. So this script MEASURES the real input tensor and patch-embed geometry,
asserts T*H*W == P, and records the result in the output. If it cannot factor P, it
says so and still writes the 1-D per-token arrays - the temporal profile survives even
when a spatial grid does not.

USAGE (on the pod - the base weights and most adapters are not local)
--------------------------------------------------------------------
  # 0. FIRST: look at the probe's internals, so the attention tap is chosen, not guessed
  HF_HOME=/root/.cache/huggingface python3 extract_token_heatmaps.py \
      --config ../configs/e4_stageA.yaml --dump-modules

  # 1. Then extract. Any number of checkpoints; NAME=NONE is the frozen A0 baseline.
  HF_HOME=/root/.cache/huggingface python3 extract_token_heatmaps.py \
      --config ../configs/e4_stageA.yaml \
      --manifest ../../dataset/manifests/test_manifest_hires.jsonl \
      --frames-root ../../dataset/test \
      --checkpoints A0=NONE \
                    A1=/workspace/semsup/a1_1761/epoch_04/lora_adapter \
                    B-v3=/workspace/semsup/b_v3_1761/epoch_02/lora_adapter \
                    V12=/workspace/MMLM_AI/outputs/a1fail321/results/v12/fold_01/epoch_10/lora_adapter \
      --clips-per-cell 2 \
      --out ../../outputs/e4_vjepa_reason/token_heatmaps/heatmaps.json
"""
import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from semsup_common import TrainableBadasWrapper  # noqa: E402

GROUP_LABEL = {0: "tte_0.5s", 1: "tte_1.0s", 2: "tte_1.5s"}


def frame_paths_for(record, frames_root, pattern):
    d = Path(frames_root) / record["frames_dir"]
    return [str(d / pattern.format(i)) for i in record["frame_indices"]]


# --------------------------------------------------------------------- layout
def discover_layout(badas, clip, n_tokens):
    """Factor the P tokens into (T, H, W) from the model's own geometry.

    Returns a dict that always records what was measured, and only claims a grid
    when T*H*W actually equals P. Everything downstream keys off `ok`.
    """
    info = {"n_tokens": int(n_tokens), "ok": False, "note": None}
    info["input_shape"] = list(clip.shape)

    # Walk the backbone for the patch embedding; V-JEPA2 exposes patch/tubelet size
    # on the conv3d embedder, but the attribute names differ between releases, so
    # read the conv kernel directly when the named attributes are absent.
    patch = tubelet = None
    for name, mod in badas.nn_model.named_modules():
        if isinstance(mod, torch.nn.Conv3d):
            k = mod.kernel_size                       # (tubelet, patch_h, patch_w)
            tubelet, patch = int(k[0]), (int(k[1]), int(k[2]))
            info["patch_embed_module"] = name
            info["conv3d_kernel"] = [int(x) for x in k]
            break
    info["patch_size"] = patch
    info["tubelet_size"] = tubelet

    if patch and tubelet and len(clip.shape) == 5:
        # clip is (B, T, C, H, W) or (B, C, T, H, W); pick whichever axis order makes
        # the token count come out right rather than trusting a convention.
        b, d1, d2, d3, d4 = clip.shape
        for (T, H, W) in ((d1, d3, d4), (d2, d3, d4)):
            t = T // tubelet
            h, w = H // patch[0], W // patch[1]
            if t * h * w == n_tokens:
                info.update({"T": int(t), "H": int(h), "W": int(w), "ok": True,
                             "order": "T,H,W (row-major)"})
                return info
    if not info["ok"]:
        info["note"] = (
            f"could not factor {n_tokens} tokens from input {list(clip.shape)} with "
            f"patch={patch} tubelet={tubelet}. Spatial maps are NOT valid; only the "
            f"per-token arrays are written. Check whether use_sliding_window is "
            f"concatenating windows.")
    return info


# ------------------------------------------------------------------ attention
class AttentionTap:
    """Capture the attentive probe's attention over the P tokens.

    Implementations differ in whether they return attention weights at all, so this
    tries the cheap route first (a module that hands back weights) and otherwise
    reconstructs softmax(qk^T/sqrt(d)) from the probe's own q/k projections. If
    neither works it stays disabled and the run continues with `proj` only - a
    missing layer is far better than a fabricated one.
    """

    def __init__(self, probe):
        self.weights = None
        self.mode = "none"
        self._handles = []
        if probe is None:
            return
        for name, mod in probe.named_modules():
            if "attention" not in type(mod).__name__.lower():
                continue
            self._handles.append(mod.register_forward_hook(self._hook))
            self.mode = f"forward-hook on {type(mod).__name__} ({name or 'root'})"
            break

    def _hook(self, _m, _a, out):
        # Only trust a second element that is shaped like an attention map.
        if isinstance(out, (tuple, list)) and len(out) > 1 and torch.is_tensor(out[1]):
            self.weights = out[1].detach()

    def take(self, n_tokens):
        w = self.weights
        self.weights = None
        if w is None:
            return None
        w = w.float()
        # collapse everything that is not the key axis: batch, heads, and the query
        # axis (the probe has a single learned query, or a handful that are averaged)
        while w.dim() > 1 and w.shape[-1] != n_tokens:
            w = w.mean(dim=-1)
        if w.shape[-1] != n_tokens:
            return None
        return w.reshape(-1, n_tokens).mean(0).cpu().numpy()


# ----------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--manifest")
    ap.add_argument("--frames-root")
    ap.add_argument("--checkpoints", nargs="+",
                    help="NAME=/path/to/lora_adapter, or NAME=NONE for the frozen base")
    ap.add_argument("--clips-per-cell", type=int, default=2,
                    help="clips sampled per (group, class) cell -> 6 cells total")
    ap.add_argument("--out")
    ap.add_argument("--dump-modules", action="store_true",
                    help="print the probe's submodule tree and exit (run this first)")
    args = ap.parse_args()

    import yaml
    cfg = yaml.safe_load(open(args.config))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[setup] device={device}")

    badas = TrainableBadasWrapper(cfg, lora_target_modules=["query", "key", "value"],
                                  lora_r=16, lora_alpha=32, lora_dropout=0.05)
    badas.nn_model.eval()

    probe = (getattr(badas.nn_model, "temporal_processor", None)
             or getattr(badas.nn_model, "pooler", None))
    if args.dump_modules:
        print("\n=== attentive probe submodule tree ===")
        if probe is None:
            print("probe not found by attribute; full model tree instead")
            for n, m in badas.nn_model.named_modules():
                print(f"  {n:70s} {type(m).__name__}")
        else:
            for n, m in probe.named_modules():
                extra = ""
                if isinstance(m, torch.nn.Linear):
                    extra = f"  in={m.in_features} out={m.out_features}"
                print(f"  {n or '<root>':40s} {type(m).__name__}{extra}")
        print("\n=== Conv3d patch embedders (token geometry) ===")
        for n, m in badas.nn_model.named_modules():
            if isinstance(m, torch.nn.Conv3d):
                print(f"  {n}  kernel={m.kernel_size} stride={m.stride}")
        return

    for req in ("manifest", "frames_root", "checkpoints", "out"):
        if not getattr(args, req):
            raise SystemExit(f"--{req.replace('_', '-')} is required unless --dump-modules")

    pattern = cfg["data"]["frame_filename_pattern"]
    gt_field = cfg["data"]["gt_field"]
    records = [json.loads(l) for l in open(args.manifest, encoding="utf-8") if l.strip()]

    # Stratified sample: every (TTE group x class) cell equally represented, so the
    # maps cannot be dominated by whichever condition happens to come first.
    chosen = []
    for g in (0, 1, 2):
        for y in (1, 0):
            cell = [r for r in records if r.get("group") == g and int(r[gt_field]) == y]
            chosen.extend(cell[:args.clips_per_cell])
    print(f"[clips] {len(chosen)} sampled ({args.clips_per_cell} per group x class cell)")

    specs = []
    for spec in args.checkpoints:
        if "=" not in spec:
            raise SystemExit(f"--checkpoints entry {spec!r} must be NAME=/path or NAME=NONE")
        name, path = spec.split("=", 1)
        specs.append((name, path))

    attn_tap = AttentionTap(probe)
    print(f"[attn] capture mode: {attn_tap.mode}")

    from safetensors.torch import load_file
    from peft.utils import set_peft_model_state_dict

    layout = None
    per_ckpt = {}            # name -> {clip_id: {attn, proj}}
    patch_store = {}         # name -> {clip_id: float16 (P,D)} for the pairwise deltas

    for name, path in specs:
        if path.upper() == "NONE":
            # peft keeps the adapter attached; zeroing lora_B makes it an exact no-op,
            # which is the frozen base without rebuilding the model.
            sd = {k: torch.zeros_like(v) for k, v in badas.nn_model.state_dict().items()
                  if "lora_B" in k}
            set_peft_model_state_dict(badas.nn_model, sd)
            print(f"\n[ckpt] {name}: frozen base (LoRA zeroed)")
        else:
            p = Path(path)
            sft = (p / "adapter_model.safetensors") if p.is_dir() else p
            if not sft.exists():
                raise SystemExit(f"{name}: adapter not found at {sft}")
            set_peft_model_state_dict(badas.nn_model, load_file(str(sft)))
            print(f"\n[ckpt] {name}: {sft}")

        per_ckpt[name] = {}
        patch_store[name] = {}
        recs = [{**r, "frame_paths": frame_paths_for(r, args.frames_root, pattern)}
                for r in chosen]
        with torch.no_grad():
            for _, ex, clip, err in badas.prefetch_clips(recs, num_workers=8, prefetch=8):
                if err is not None:
                    print(f"  [warn] skipping {ex.get('video_id')}: {err}")
                    continue
                clip = clip.to(device)
                logits, patches = badas.forward_clip(clip)       # (1,2), (P,D)
                pooled = badas._captured.get("pooled")
                pooled = pooled[0] if pooled.dim() > 1 else pooled
                P = patches.shape[0]

                if layout is None:
                    layout = discover_layout(badas, clip, P)
                    print(f"[layout] {json.dumps(layout)}")

                # projection of every token onto the vector the classifier reads
                pn = pooled / (pooled.norm() + 1e-8)
                proj = (patches @ pn).float().cpu().numpy()
                a = attn_tap.take(P)

                vid = ex["video_id"]
                per_ckpt[name][vid] = {
                    "proj": np.round(proj, 5).tolist(),
                    "attn": (np.round(a, 6).tolist() if a is not None else None),
                    "score": float(torch.softmax(logits, dim=1)[0, 1]),
                }
                patch_store[name][vid] = patches.half().cpu()
        print(f"  captured {len(per_ckpt[name])} clips")

    # ---- pairwise representation deltas, every ordered pair, so the consumer can
    # ---- pick any baseline without another GPU pass
    deltas = {}
    names = [n for n, _ in specs]
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            key = f"{a}->{b}"
            deltas[key] = {}
            for vid in patch_store[a]:
                if vid not in patch_store[b]:
                    continue
                d = (patch_store[b][vid].float() - patch_store[a][vid].float())
                deltas[key][vid] = np.round(d.norm(dim=1).numpy(), 5).tolist()
            print(f"[delta] {key}: {len(deltas[key])} clips")

    meta = {v["video_id"]: {"gt": int(v[gt_field]), "group": v.get("group"),
                            "tte": GROUP_LABEL.get(v.get("group"))} for v in chosen}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"generated_from": "extract_token_heatmaps.py",
                   "layout": layout, "attn_mode": attn_tap.mode,
                   "checkpoints": names, "clips": meta,
                   "maps": per_ckpt, "deltas": deltas}, f, separators=(",", ":"))
    print(f"\n[wrote] {out}  ({out.stat().st_size / 1024 / 1024:.1f} MB)")
    if layout and not layout.get("ok"):
        print(f"[WARN] {layout['note']}")


if __name__ == "__main__":
    main()
