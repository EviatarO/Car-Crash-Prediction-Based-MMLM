"""
r1_frames.py - stage 3 of plan 2026-10-22_Plan-Stage3-Encoder-LoRA: frames instead of cached features, and the V-JEPA2 encoder run ONLINE with a trainable LoRA.

The reasoning model's tokens are the V-JEPA2 encoder's own output, backbone.get_vision_features(pixel_values_videos=x) (2048 x 1024; checked in the BADAS-Open source
src/train/video_training.py: the predictor's 512 future tokens are only concatenated after them, for the crash head). So the encoder runs ALONE here: the BADAS predictor and
the crash head are never executed (user rule 2026-10-22: no crash head at all).

  build   --repo eviatarO-org/nexar-reasoning-windows --out <dir>          the 4,169 windows of the HF window repo -> frames_nexar_reasoning.npy (N,16,256,256,3) uint8 + index
  build   --frames-dirs <dir> --manifest <extra manifest> --split val --out <dir>   JPEG windows cut on the pod -> the same layout (processor resize, as the window repo)
  parity  --frames <dir> --cache <old cache dir> --n 24                     encoder-only (A1 adapter, fp32) tokens == the cached tokens; GPU normalisation == the processor
  encode  --frames <dir> --adapter <dir> --p-from <old cache dir> --out <dir>   re-encode with a (trained) encoder LoRA, encoder only, fp32 -> a feature cache r1_data.Cache reads
                                                                             (the crash-model P column is copied from the old index: the head is not run on the new encoder)
  drift   --old <cache dir> --new <cache dir> [--split val]                  mean cosine similarity of old vs new tokens per window
"""
from __future__ import annotations

import argparse
import io
import json
import math
import sys
import tarfile
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
SRC = "nexar_reasoning"
N_FR, SIDE, N_TOK, DIM = 16, 256, 2048, 1024
CFG = HERE.parent / "configs" / "e4_stageA.yaml"
A1_ADAPTER = ROOT / "outputs" / "a1_compress256" / "train" / "epoch_02" / "lora_adapter"


# ----------------------------------------------------------------------------- frames store
class FramesStore:
    """uint8 frames memmap (N,16,256,256,3) + index {id: row}; several directories separated by commas (the original windows + the extra ones)."""

    def __init__(self, frames_dir):
        self.rows, self.maps = {}, []
        for k, d in enumerate(Path(x) for x in str(frames_dir).split(",") if x.strip()):
            ix = d / f"index_{SRC}.jsonl"
            if not ix.exists():
                continue
            self.maps.append(np.load(d / f"frames_{SRC}.npy", mmap_mode="r"))
            for l in open(ix, encoding="utf-8"):
                r = json.loads(l)
                self.rows.setdefault(r["id"], (len(self.maps) - 1, r["row"]))

    def __contains__(self, wid):
        return wid in self.rows

    def get(self, ids):
        return np.stack([np.asarray(self.maps[self.rows[i][0]][self.rows[i][1]]) for i in ids])     # (B,16,256,256,3) uint8


def frames_sink(out_dir, n):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    return np.lib.format.open_memmap(out / f"frames_{SRC}.npy", mode="w+", dtype=np.uint8, shape=(n, N_FR, SIDE, SIDE, 3)), open(out / f"index_{SRC}.jsonl", "w", encoding="utf-8")


def build_from_repo(repo, out_dir, splits=("train", "val")):
    from huggingface_hub import hf_hub_download
    work = Path(out_dir) / "_repo"
    rows = [json.loads(l) for l in open(hf_hub_download(repo, "windows.jsonl", repo_type="dataset", local_dir=str(work)), encoding="utf-8")]
    rows = [r for r in rows if r["split"] in splits]
    by_shard = {}
    for r in rows:
        by_shard.setdefault(r["shard"], set()).add(r["window_id"])
    mm, ix = frames_sink(out_dir, len(rows))
    n = 0
    for shard, ids in sorted(by_shard.items()):
        p = hf_hub_download(repo, shard, repo_type="dataset", local_dir=str(work))
        with tarfile.open(p) as tf:
            for m in tf:
                wid = m.name[:-4]
                if wid not in ids:
                    continue
                fr = np.load(io.BytesIO(tf.extractfile(m).read()))["frames"]
                assert fr.shape == (N_FR, SIDE, SIDE, 3) and fr.dtype == np.uint8, (wid, fr.shape, fr.dtype)
                mm[n] = fr
                ix.write(json.dumps({"id": wid, "row": n}) + "\n")
                n += 1
        Path(p).unlink(missing_ok=True)
        print(f"[frames] {shard}: {n} / {len(rows)} windows", flush=True)
    mm.flush()
    ix.close()
    assert n == len(rows), (n, len(rows))
    print(f"[frames] done: {n} windows -> {out_dir}")


def build_from_dirs(frames_root, manifest, split, out_dir):
    """JPEG windows (cut on the pod, 1280x720) -> 256x256 uint8 with the processor's own resize (r1_build_window_repo.resize_u8, as the window repo)."""
    from PIL import Image
    from r1_build_window_repo import get_processor, resize_u8
    proc = get_processor()
    rows = [json.loads(l) for l in open(manifest, encoding="utf-8") if l.strip()]
    rows = [r for r in rows if r["valid"] and (not split or r["split"] == split)]
    mm, ix = frames_sink(out_dir, len(rows))
    for n, r in enumerate(rows):
        paths = sorted((Path(frames_root) / r["frames_dir"]).glob("frame_*.jpg"))
        assert len(paths) == N_FR, (r["frames_dir"], len(paths))
        mm[n] = resize_u8(proc, [np.array(Image.open(p).convert("RGB")) for p in paths])
        ix.write(json.dumps({"id": r["id"], "row": n}) + "\n")
    mm.flush()
    ix.close()
    print(f"[frames] {len(rows)} windows from {frames_root} -> {out_dir}")


# ----------------------------------------------------------------------------- the online encoder
def encoder_lora_names(names):
    """The trainable parameters of stage 3: LoRA matrices of the ENCODER only (backbone.encoder.layer.*), never the predictors (backbone.predictor / model.predictor)."""
    return [n for n in names if "lora_" in n and ".backbone.encoder." in n]


class OnlineEncoder(torch.nn.Module):
    """Frozen V-JEPA2 ViT-L (BADAS-Open + the crash model's LoRA as the starting point) whose ENCODER LoRA trains. forward(uint8 frames (B,16,256,256,3)) -> (B,2048,1024).
    Only backbone.get_vision_features runs (no predictor, no crash head). Normalisation on the GPU with the processor's own constants (parity-checked against
    r1_cache_features.preprocess_frames). fp32 throughout by default (TF32 matmuls while training); precision="bf16" = autocast while gradients are on."""

    def __init__(self, adapter=None, device="cuda", trainable=True, grad_ckpt=True, precision="fp32"):
        super().__init__()
        self.precision = precision                                        # "fp32" (TF32 while training, default) | "fp32_strict" | "bf16" (autocast while gradients are on)
        self.tf32_training = precision == "fp32"                          # "fp32" = TF32 matmuls ONLY while gradients are on (measured relative token change 0.4 %); "fp32_strict" = plain fp32; "bf16" = autocast (measured 13 %)
        import yaml
        from aa1_token_probe import load_probe_model
        cfg = yaml.safe_load(open(CFG, encoding="utf-8"))
        self.badas, _ = load_probe_model(cfg, "compress256", [], str(adapter or A1_ADAPTER), "query,key,value")
        self.net = self.badas.nn_model                                   # peft-wrapped EnhancedVideoClassifier
        self.backbone = next(m for n, m in self.net.named_modules() if n.endswith("backbone"))
        for m in self.net.modules():                                      # the V-JEPA2 config has stochastic depth 0.1 (+ dropouts): in train mode it would make the tokens random (measured: relative change 0.7).
            if type(m).__name__ == "VJEPA2DropPath":                       # Zero them so that train mode (needed for gradient checkpointing) gives the same tokens as eval mode.
                m.drop_prob = 0.0
            elif isinstance(m, torch.nn.Dropout):
                m.p = 0.0
        names = encoder_lora_names([n for n, _ in self.net.named_parameters()])
        assert len(names) == 24 * 3 * 2, f"expected 144 encoder LoRA matrices (24 layers x q,k,v x A,B), got {len(names)}"
        self.train_names = set(names)
        for n, p in self.net.named_parameters():
            p.requires_grad = trainable and n in self.train_names
            if n in self.train_names:
                p.data = p.data.float()
        if grad_ckpt and hasattr(self.backbone, "gradient_checkpointing_enable"):
            self.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        proc = self.badas.vjepa.processor
        self.register_buffer("mean", torch.tensor(proc.image_mean, dtype=torch.float32).view(1, 1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(proc.image_std, dtype=torch.float32).view(1, 1, 3, 1, 1), persistent=False)
        self.rescale = float(getattr(proc, "rescale_factor", 1 / 255))
        self.to(device)
        self.device = torch.device(device)

    def trainable(self):
        return [p for n, p in self.net.named_parameters() if n in self.train_names and p.requires_grad]

    def normalise(self, frames_u8):
        x = torch.as_tensor(frames_u8).to(self.device, non_blocking=True)          # (B,16,256,256,3) uint8
        x = x.permute(0, 1, 4, 2, 3).float() * self.rescale                        # (B,16,3,256,256)
        return (x - self.mean) / self.std

    def forward(self, frames_u8):
        x = self.normalise(frames_u8)
        self.net.train(torch.is_grad_enabled())                           # train mode only while gradients flow (HF gradient checkpointing needs it); drop-path / dropout are zeroed above
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = bool(torch.is_grad_enabled() and self.tf32_training)
        if torch.is_grad_enabled() and self.precision == "bf16":
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
                out = self.backbone.get_vision_features(pixel_values_videos=x)
        else:                                                              # fp32 (TF32 matmuls while training; plain fp32 for evaluation / re-encoding, as the original caching)
            out = self.backbone.get_vision_features(pixel_values_videos=x)
        out = out.last_hidden_state if hasattr(out, "last_hidden_state") else out
        assert out.shape[1:] == (N_TOK, DIM), out.shape
        return out.float()

    def save_adapter(self, path):
        """adapter_model.safetensors + adapter_config.json, the format load_probe_model reads. peft's save_pretrained cannot be used: its model-card step fails on BADAS's custom config object."""
        from peft.utils import get_peft_model_state_dict
        from safetensors.torch import save_file
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        save_file({k: v.detach().cpu().contiguous() for k, v in get_peft_model_state_dict(self.net).items()}, str(path / "adapter_model.safetensors"))
        for cfg in self.net.peft_config.values():
            cfg.save_pretrained(str(path))

    def load_adapter(self, path):
        from peft.utils import set_peft_model_state_dict
        from safetensors.torch import load_file
        set_peft_model_state_dict(self.net, load_file(str(Path(path) / "adapter_model.safetensors")))
        for n, p in self.net.named_parameters():
            if n in self.train_names:
                p.data = p.data.float()


class OnlineCache:
    """Drop-in for r1_data.Cache (rows / __contains__ / get / p_collision): the rows (and the crash-model P column) come from the OLD feature cache index, the tokens from the
    online encoder on the frames. get() keeps the gradient when gradients are enabled (training) and runs under no-grad otherwise."""

    def __init__(self, old_cache_dirs, frames: FramesStore, encoder: OnlineEncoder, chunk=None):
        self.rows = {}
        for d in (Path(x) for x in str(old_cache_dirs).split(",") if x.strip()):
            ix = d / f"index_{SRC}.jsonl"
            if ix.exists():
                for l in open(ix, encoding="utf-8"):
                    r = json.loads(l)
                    if r["id"] in frames:
                        self.rows.setdefault(r["id"], r)
        import os
        self.frames, self.encoder, self.chunk = frames, encoder, int(chunk or os.environ.get("R1_ENC_CHUNK", 4))      # windows per encoder forward when no gradient is needed (1 on a 6 GB laptop GPU)

    def __contains__(self, wid):
        return wid in self.rows

    _memo = None

    def memo(self):
        """Context: while active (and no gradient), tokens are kept on the CPU (fp32, ~8 MB per window) and reused; the encoder must not change inside it."""
        import contextlib

        @contextlib.contextmanager
        def ctx():
            self._memo = {}
            try:
                yield
            finally:
                self._memo = None
        return ctx()

    def get(self, ids):
        if torch.is_grad_enabled():
            return self.encoder(self.frames.get(ids))
        with torch.no_grad():
            if self._memo is None:
                return torch.cat([self.encoder(self.frames.get(ids[i:i + self.chunk])) for i in range(0, len(ids), self.chunk)], 0)
            todo = [w for w in dict.fromkeys(ids) if w not in self._memo]
            for i in range(0, len(todo), self.chunk):
                y = self.encoder(self.frames.get(todo[i:i + self.chunk]))
                for w, t in zip(todo[i:i + self.chunk], y):
                    self._memo[w] = t.cpu()
            dev = next(self.encoder.net.parameters()).device
            return torch.stack([self._memo[w] for w in ids]).to(dev)

    def p_collision(self, wid):
        return self.rows[wid]["p_collision"]


# ----------------------------------------------------------------------------- tools
def cmd_parity(a):
    from r1_cache_features import preprocess_frames
    fr = FramesStore(a.frames)
    old = np.load(Path(a.cache) / f"features_{SRC}.npy", mmap_mode="r")
    idx = [json.loads(l) for l in open(Path(a.cache) / f"index_{SRC}.jsonl", encoding="utf-8")]
    idx = [r for r in idx if r["id"] in fr][:a.n]
    enc = OnlineEncoder(trainable=False, grad_ckpt=False, device=a.device)
    worst_norm, rels, coss = 0.0, [], []
    with torch.no_grad():
        for r in idx:
            f = fr.get([r["id"]])
            ref = preprocess_frames(enc.badas.vjepa, list(f[0])).to(enc.device).float()
            worst_norm = max(worst_norm, float((enc.normalise(f) - ref).abs().max()))
            t = enc(f)[0].cpu().numpy()
            o = np.asarray(old[r["row"]]).astype(np.float32)
            rels.append(float(np.abs(t - o).mean() / np.abs(o).mean()))
            coss.append(float((t.ravel() @ o.ravel()) / (np.linalg.norm(t) * np.linalg.norm(o))))
    print(f"[parity] {len(idx)} windows: max |GPU normalisation - processor| = {worst_norm:.2e}; tokens vs the cache: relative mean |diff| mean {np.mean(rels):.4f} (max {np.max(rels):.4f}), "
          f"cosine min {np.min(coss):.5f}  [reference: wrong adapter epoch 23-29 %, bf16 autocast 13 %, TF32 matmuls 0.4 %]")
    assert worst_norm < 1e-4, "GPU normalisation differs from the processor"
    # The cache was built on another machine / library build and is NOT bit-reproducible (measured 2026-10-22: ~2 % relative, cosine 0.999 on three windows, the same on a laptop GPU and on the pods);
    # a wrong adapter (23-29 %), a wrong tap point or bf16 (13 %) are far outside this. The functional gate is the epoch-0 validation loss of the training run.
    assert np.mean(rels) < 0.05 and np.min(coss) > 0.995, "encoder-only tokens differ from the cached tokens beyond the numerical noise floor (wrong tap point, adapter or frames)"


def cmd_bench(a):
    """Is fp32 required? On N windows: (1) how far bf16-autocast tokens are from fp32 tokens (relative to the token scale and to the fp16 rounding the reader was trained on),
    (2) seconds per forward + backward step for fp32 (TF32 matmuls) vs bf16 at micro-batch 2, (3) peak GPU memory of each."""
    import time
    fr = FramesStore(a.frames)
    ids = list(fr.rows)[:a.n]
    out = {}
    enc = OnlineEncoder(trainable=True, grad_ckpt=True, device=a.device, precision="fp32")
    with torch.no_grad():
        ref = torch.cat([enc(fr.get(ids[i:i + 1])).cpu() for i in range(len(ids))])               # plain fp32 (as the original caching)
    ref16 = ref.half().float()
    for prec in ("fp32", "fp32_strict", "bf16"):
        enc.precision = prec
        torch.cuda.reset_peak_memory_stats()
        times = []
        for k in range(3):
            fr_b = fr.get(ids[(2 * k) % len(ids):(2 * k) % len(ids) + 2] if len(ids) > 1 else ids[:1])
            torch.cuda.synchronize()
            t0 = time.time()
            for q in enc.trainable():
                q.grad = None
            with torch.enable_grad():
                y = enc(fr_b)
                (y.float() ** 2).mean().backward()
            torch.cuda.synchronize()
            times.append(time.time() - t0)
        out[prec] = {"sec_per_step_micro_batch_2": round(float(np.median(times[1:])), 2), "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)}
    d16 = (ref16 - ref).abs()
    for prec in ("fp32", "fp32_strict", "bf16"):                   # fp32 with gradients on = TF32 matmuls (training); fp32_strict = plain fp32; bf16 = autocast
        enc.precision = prec
        with torch.enable_grad():
            enc.net.train()
            tk = torch.cat([enc(fr.get(ids[i:i + 1])).detach().cpu() for i in range(len(ids))])
        d = (tk - ref).abs()
        out[{"fp32": "tf32_training", "fp32_strict": "fp32_strict_training", "bf16": "bf16"}[prec] + "_vs_eval_fp32_tokens"] = {
            "mean_abs": float(d.mean()), "max_abs": float(d.max()), "rel_mean": float(d.mean() / ref.abs().mean()),
            "fp16_rounding_mean_abs": float(d16.mean()), "ratio_to_fp16_rounding": float(d.mean() / max(d16.mean(), 1e-12)),
            "cosine_per_window": float(torch.nn.functional.cosine_similarity(tk.flatten(1), ref.flatten(1), dim=1).mean())}
    print("[bench] " + json.dumps(out))
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")


def cmd_encode(a):
    from r1_cache_features import Sink
    fr = FramesStore(a.frames)
    old = {}
    for d in a.p_from.split(","):
        ix = Path(d) / f"index_{SRC}.jsonl"
        if ix.exists():
            for l in open(ix, encoding="utf-8"):
                r = json.loads(l)
                old.setdefault(r["id"], r)
    ids = [i for i in fr.rows if i in old]
    sink = Sink(Path(a.out), SRC, capacity=len(ids))
    enc = OnlineEncoder(adapter=a.adapter, trainable=False, grad_ckpt=False, device=a.device)
    with torch.no_grad():
        for k in range(0, len(ids), a.bs):
            chunk = [i for i in ids[k:k + a.bs] if i not in sink.done]
            if not chunk:
                continue
            tok = enc(fr.get(chunk))
            assert torch.isfinite(tok).all() and float(tok.abs().max()) < 1e4, "token magnitude check failed"
            for i, t in zip(chunk, tok.half().cpu().numpy()):
                r = old[i]
                sink.add(i, t, r["p_collision"], {"split": r.get("split"), "label": r.get("label"), "p_note": "crash-model P copied from the old cache (head not run)"})
            if (k // a.bs) % 50 == 0:
                print(f"[encode] {min(k + a.bs, len(ids))} / {len(ids)}", flush=True)
    sink.close()
    print(f"[encode] done: {len(ids)} windows -> {a.out}")


def cmd_drift(a):
    def load(d):
        ix = {json.loads(l)["id"]: json.loads(l) for l in open(Path(d) / f"index_{SRC}.jsonl", encoding="utf-8")}
        return np.load(Path(d) / f"features_{SRC}.npy", mmap_mode="r"), ix
    (mo, io_), (mn, in_) = load(a.old), load(a.new)
    ids = [i for i in in_ if i in io_ and (not a.split or in_[i].get("split") == a.split)]
    cos = []
    for i in ids:
        x, y = np.asarray(mo[io_[i]["row"]]).astype(np.float32), np.asarray(mn[in_[i]["row"]]).astype(np.float32)
        cos.append(float(((x * y).sum(1) / (np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1) + 1e-9)).mean()))
    cos = np.array(cos)
    res = {"n": len(ids), "split": a.split, "mean_token_cosine": float(cos.mean()), "p05": float(np.percentile(cos, 5)), "min": float(cos.min())}
    print("[drift] " + json.dumps(res))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--repo")
    b.add_argument("--frames-dirs")
    b.add_argument("--manifest")
    b.add_argument("--split", default="")
    b.add_argument("--out", required=True)
    p = sub.add_parser("parity")
    p.add_argument("--frames", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--n", type=int, default=24)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    e = sub.add_parser("encode")
    e.add_argument("--frames", required=True)
    e.add_argument("--adapter", required=True)
    e.add_argument("--p-from", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--bs", type=int, default=4)
    e.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    bn = sub.add_parser("bench")
    bn.add_argument("--frames", required=True)
    bn.add_argument("--n", type=int, default=8)
    bn.add_argument("--out", default=None)
    bn.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    d = sub.add_parser("drift")
    d.add_argument("--old", required=True)
    d.add_argument("--new", required=True)
    d.add_argument("--split", default="val")
    d.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.cmd == "build":
        build_from_repo(a.repo, a.out) if a.repo else build_from_dirs(a.frames_dirs, a.manifest, a.split, a.out)
    elif a.cmd == "parity":
        cmd_parity(a)
    elif a.cmd == "bench":
        cmd_bench(a)
    elif a.cmd == "encode":
        cmd_encode(a)
    else:
        cmd_drift(a)


if __name__ == "__main__":
    main()
