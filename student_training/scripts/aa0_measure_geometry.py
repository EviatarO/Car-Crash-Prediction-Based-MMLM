"""
aa0_measure_geometry.py
=======================
Gate AA.0 v2, Step 1: MEASURE (not derive) BADAS-Open's input geometry in the environment
that will actually train/score, for both preprocessing modes ("crop", "compress256").

Records, per mode:
  - input tensor shape produced by preprocess_clip()
  - the exact source-pixel box the model sees (coordinate-encoded frame: R channel = x,
    G channel = y; read back after undoing ImageNet normalization)
  - token count at the patch embedding AND at the temporal_processor input (the tap that
    once recorded 2560 on the pod - semsup_common.py's _pre_hook comment)
And once (mode-independent, model-level):
  - token flatten order: perturb one tubelet x patch cell of the INPUT TENSOR and find the
    single embedding token that changes -> verifies idx = t*H'*W' + y*W' + x
  - tubelet grouping: which temporal group a change in frame k lands in
  - environment fingerprint (python/torch/transformers/peft/GPU, processor config)

Usage:
  # local dry run (no BADAS model; processor + geometry only)
  python student_training/scripts/aa0_measure_geometry.py --processor-only \
      --out outputs/aa0/geometry_local.json

  # pod (full)
  python3 aa0_measure_geometry.py --config ../configs/e4_stageA.yaml \
      --real-window ../../dataset/test/00001_hires --out ../../outputs/aa0/geometry_pod.json
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models"))

from e4_stageA_badas_open_eval import preprocess_clip, PREPROCESS_MODES  # noqa: E402

MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])
SRC_W, SRC_H = 1280, 720


def env_fingerprint(processor):
    import torch
    fp = {"python": platform.python_version(), "torch": torch.__version__,
          "cuda_available": torch.cuda.is_available(),
          "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    for mod in ("transformers", "peft", "timm", "einops", "safetensors"):
        try:
            fp[mod] = __import__(mod).__version__
        except Exception:
            fp[mod] = None
    try:
        d = processor.to_dict()
        fp["processor"] = {k: d.get(k) for k in ("video_processor_type", "size", "crop_size",
                                                  "do_center_crop", "do_resize", "resample",
                                                  "default_to_square", "image_mean", "image_std")}
    except Exception as e:
        fp["processor"] = f"unavailable: {e}"
    return fp


def coordinate_frames(tmpdir: Path):
    xx, yy = np.meshgrid(np.arange(SRC_W), np.arange(SRC_H))
    code = np.stack([xx * 255 / (SRC_W - 1), yy * 255 / (SRC_H - 1), np.zeros_like(xx)],
                    -1).astype(np.uint8)
    paths = []
    for i in range(16):
        p = tmpdir / f"coord_{i:02d}.png"   # PNG: lossless, so the encoding survives
        Image.fromarray(code).save(p)
        paths.append(str(p))
    return paths


def visible_box(clip):
    """Source-pixel box covered by the model input, from the coordinate-encoded clip."""
    a = (clip[0, 0].permute(1, 2, 0).cpu().numpy() * STD + MEAN).clip(0, 1) * 255
    h, w = a.shape[:2]

    def src(py, px):
        return (int(round(a[py, px, 0] / 255 * (SRC_W - 1))),
                int(round(a[py, px, 1] / 255 * (SRC_H - 1))))
    tl, br = src(0, 0), src(h - 1, w - 1)
    return {"top_left_src_px": tl, "bottom_right_src_px": br,
            "width_fraction": round((br[0] - tl[0]) / (SRC_W - 1), 4),
            "height_fraction": round((br[1] - tl[1]) / (SRC_H - 1), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "configs" / "e4_stageA.yaml"))
    ap.add_argument("--processor-only", action="store_true",
                     help="local dry run: no BADAS model load (processor geometry only)")
    ap.add_argument("--real-window", default=None,
                     help="a 16-frame window dir (frame_00001..16.jpg) for the token-count forward pass")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch
    out = {"modes": {}}

    if args.processor_only:
        from transformers import AutoVideoProcessor

        class _Stub:
            pass
        vjepa = _Stub()
        vjepa.processor = AutoVideoProcessor.from_pretrained("facebook/vjepa2-vitl-fpc16-256-ssv2")
        vjepa.transform = None
        nn_model = None
    else:
        import yaml
        from e4_stageA_badas_open_eval import load_badas
        cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
        vjepa, nn_model, device = load_badas(cfg)
        nn_model.eval()

    out["environment"] = env_fingerprint(vjepa.processor)
    print(json.dumps(out["environment"], indent=2))

    with tempfile.TemporaryDirectory() as td:
        cpaths = coordinate_frames(Path(td))
        for mode in PREPROCESS_MODES:
            clip = preprocess_clip(vjepa, cpaths, mode=mode)
            rec = {"input_shape": list(clip.shape), "visible_region": visible_box(clip)}
            out["modes"][mode] = rec
            print(f"[{mode}] input {tuple(clip.shape)}  visible {rec['visible_region']}")

    if nn_model is not None:
        dev = next(nn_model.parameters()).device
        embed = nn_model.backbone.encoder.embeddings
        captured = {}
        h1 = embed.patch_embeddings.register_forward_hook(
            lambda m, i, o: captured.__setitem__("embed", tuple(o.shape)))
        h2 = nn_model.temporal_processor.register_forward_pre_hook(
            lambda m, a: captured.__setitem__("probe", tuple(a[0].shape)))

        # token counts at both taps, per mode, on a real window
        if args.real_window:
            wpaths = [str(Path(args.real_window) / f"frame_{i:05d}.jpg") for i in range(1, 17)]
            for mode in PREPROCESS_MODES:
                captured.clear()
                with torch.no_grad():
                    nn_model(preprocess_clip(vjepa, wpaths, mode=mode).to(dev))
                out["modes"][mode]["tokens_patch_embedding"] = captured.get("embed")
                out["modes"][mode]["tokens_probe_input"] = captured.get("probe")
                print(f"[{mode}] tokens: patch_embedding {captured.get('embed')}  "
                      f"probe_input {captured.get('probe')}")
        h1.remove()
        h2.remove()

        # flatten order + tubelet grouping (model-level; perturb the INPUT TENSOR so no
        # resize blurs the change across patch boundaries)
        T, H, W = 16, 256, 256
        tub = embed.config.tubelet_size if hasattr(embed, "config") else 2
        ps = embed.patch_size
        Tg, Hg, Wg = T // tub, H // ps, W // ps
        base = torch.zeros(1, T, 3, H, W, device=dev)
        with torch.no_grad():
            e0 = embed(base)

        def changed_tokens(t0, t1, r, c):
            x = base.clone()
            x[:, t0:t1, :, r * ps:(r + 1) * ps, c * ps:(c + 1) * ps] = 1.0
            with torch.no_grad():
                d = (embed(x) - e0).abs().sum(-1)[0]
            return torch.nonzero(d > 1e-6).flatten().tolist()

        probes = []
        for (tg, r, c) in [(0, 0, 0), (Tg - 1, Hg - 1, Wg - 1), (3, 5, 7), (6, 2, 13)]:
            got = changed_tokens(tg * tub, tg * tub + tub, r, c)
            expect = tg * Hg * Wg + r * Wg + c
            probes.append({"cell_t_r_c": [tg, r, c], "expected_index": expect, "changed": got,
                           "ok": got == [expect]})
        groups = []
        for k in range(min(T, 5)):
            got = changed_tokens(k, k + 1, 4, 4)
            groups.append({"frame": k, "token_group": (got[0] // (Hg * Wg)) if got else None,
                           "expected_group": k // tub, "ok": bool(got) and got[0] // (Hg * Wg) == k // tub})
        out["grid"] = {"tubelet": tub, "patch": ps, "T_H_W": [Tg, Hg, Wg], "P": Tg * Hg * Wg}
        out["flatten_order_probes"] = probes
        out["tubelet_grouping_probes"] = groups
        out["flatten_order_verified"] = all(p["ok"] for p in probes)
        out["tubelet_grouping_verified"] = all(g["ok"] for g in groups)
        print(f"grid {out['grid']}  flatten_order_verified={out['flatten_order_verified']}  "
              f"tubelet_grouping_verified={out['tubelet_grouping_verified']}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[wrote] {args.out}")


if __name__ == "__main__":
    main()
