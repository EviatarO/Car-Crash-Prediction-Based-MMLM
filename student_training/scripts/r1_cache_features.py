"""
r1_cache_features.py - cache the frozen A1-compress256 V-JEPA2 tokens (the tokens the crash head reads)
for the week-1 windows, once, so Phase 1 / Phase 2 never run the encoder.

Per window: the first 2,048 of the 2,560 probe-input tokens (8 time x 16 x 16 x 1024 real tokens; the 512
BADAS-predicted "future" tokens are dropped) + P(collision) (T=1) for the faithfulness metric.
Preprocessing = the existing compress256 path (full frame squashed to 256x256, no crop).

Output (in --out-dir):  features_<source>.npy  float16 memmap (N, 2048, 1024)   index_<source>.jsonl
Resumable: windows already in the index are skipped.

  Nexar (local frames, valid windows of the chosen splits):
    python r1_cache_features.py --source nexar --out-dir ../../outputs/r1_week1/cache --limit 8
  DADA (streams the MM-AU DADA tar parts; --parts-dir = local copies of DADA2000.part_* downloaded in
  order, deleted after use with --delete-parts; or --url-parts N streams the first N parts over HTTP):
    python r1_cache_features.py --source dada --url-parts 1 --out-dir ../../outputs/r1_week1/cache --limit 8
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "models"))
from r1_common import window_indices  # noqa: E402

MANIFEST = {"nexar": ROOT / "dataset" / "manifests" / "r1_nexar_v12_windows.jsonl",
            "dada": ROOT / "dataset" / "manifests" / "r1_mmau_dada_windows.jsonl"}
ADAPTER = ROOT / "outputs" / "a1_compress256" / "train" / "epoch_02" / "lora_adapter"
CFG = HERE.parent / "configs" / "e4_stageA.yaml"
DADA_URL = "https://huggingface.co/datasets/JeffreyChou/MM-AU/resolve/main/DADA-2000_chunks/DADA2000.part_{}"
N_TOK, DIM = 2048, 1024


# ----------------------------------------------------------------------------- model
def load_model():
    import yaml
    from aa1_token_probe import load_probe_model
    cfg = yaml.safe_load(open(CFG, encoding="utf-8"))
    badas, _ = load_probe_model(cfg, "compress256", [], str(ADAPTER), "query,key,value")
    return badas


def preprocess_frames(vjepa, frames_rgb):
    """In-memory twin of e4_stageA_badas_open_eval.preprocess_clip(mode='compress256') (same processor call
    and overrides), for frames that did not come from JPEG files. r1_cache_features_test checks equality."""
    inputs = vjepa.processor(videos=frames_rgb, return_tensors="pt",
                             size={"height": 256, "width": 256}, do_center_crop=False)
    key = "pixel_values_videos" if "pixel_values_videos" in inputs else next(iter(inputs))
    clip = inputs[key]
    return clip.unsqueeze(0) if clip.dim() == 4 else clip


@torch.no_grad()
def encode(badas, clip):
    logits, patches = badas.forward_clip(clip.to(badas.device))
    tokens = patches[:N_TOK].float()
    assert tokens.shape == (N_TOK, DIM), tokens.shape
    assert torch.isfinite(tokens).all() and tokens.abs().max() < 1e4, "token magnitude check failed"
    p = float(torch.softmax(logits.float(), dim=1)[0, 1])
    return tokens.half().cpu().numpy(), p


# ----------------------------------------------------------------------------- sinks
class Sink:
    """Resumable fp16 memmap + jsonl index."""

    def __init__(self, out_dir: Path, source: str, capacity: int):
        out_dir.mkdir(parents=True, exist_ok=True)
        self.fx = out_dir / f"features_{source}.npy"
        self.ix = out_dir / f"index_{source}.jsonl"
        self.done = {}
        if self.ix.exists():
            for l in open(self.ix, encoding="utf-8"):
                r = json.loads(l)
                self.done[r["id"]] = r
        mode = "r+" if self.fx.exists() else "w+"
        self.mm = np.lib.format.open_memmap(self.fx, mode=mode, dtype=np.float16, shape=(capacity, N_TOK, DIM)) \
            if mode == "w+" else np.load(self.fx, mmap_mode="r+")
        self.n = len(self.done)
        self.f = open(self.ix, "a", encoding="utf-8")

    def add(self, wid, tokens, p, extra=None):
        self.mm[self.n] = tokens
        rec = {"id": wid, "row": self.n, "p_collision": round(p, 5), **(extra or {})}
        self.f.write(json.dumps(rec) + "\n")
        self.done[wid] = rec
        self.n += 1
        if self.n % 25 == 0:
            self.mm.flush()
            self.f.flush()

    def close(self):
        self.mm.flush()
        self.f.close()


def load_rows(source, splits, limit):
    rows = [json.loads(l) for l in open(MANIFEST[source], encoding="utf-8")]
    rows = [r for r in rows if r["valid"] and r["split"] in splits]
    return rows[:limit] if limit else rows


# ----------------------------------------------------------------------------- Nexar
def run_nexar(args, badas):
    from PIL import Image
    rows = load_rows("nexar", args.splits, args.limit)
    sink = Sink(Path(args.out_dir), "nexar", capacity=len(rows))
    todo = [r for r in rows if r["id"] not in sink.done]
    print(f"[nexar] {len(rows)} valid windows, {len(todo)} to encode")
    t0 = time.time()
    for i, r in enumerate(todo):
        fd = ROOT / "dataset" / "train" / r["frames_dir"]
        paths = sorted(fd.glob("frame_*.jpg"))
        assert len(paths) == 16, (fd, len(paths))
        frames = [np.array(Image.open(p).convert("RGB")) for p in paths]
        tokens, p = encode(badas, preprocess_frames(badas.vjepa, frames))
        sink.add(r["id"], tokens, p, {"split": r["split"], "label": r["label"]})
        if (i + 1) % 25 == 0 or i + 1 == len(todo):
            el = time.time() - t0
            print(f"  [{i + 1}/{len(todo)}] {el / (i + 1):.2f}s/window eta={(len(todo) - i - 1) * el / (i + 1) / 60:.1f}min", flush=True)
    sink.close()


# ----------------------------------------------------------------------------- DADA stream
class ChainedStream(io.RawIOBase):
    """Reads DADA2000.part_aa, _ab, ... as ONE byte stream (they are pieces of a single tar.gz).
    Sources are local files (optionally deleted once fully read) or HTTP URLs."""

    def __init__(self, sources, delete_local=False, wait_s=4 * 3600):
        self.sources, self.delete_local, self.wait_s, self.k, self.cur = list(sources), delete_local, wait_s, -1, None
        self._next()

    def _next(self):
        if self.cur is not None:
            is_file = isinstance(self.cur, io.BufferedReader)
            name = self.cur.name if is_file else None
            self.cur.close()
            if is_file and self.delete_local:
                Path(name).unlink(missing_ok=True)
        self.k += 1
        if self.k >= len(self.sources):
            self.cur = None
            return
        s = str(self.sources[self.k])
        if not s.startswith("http"):
            t0 = time.time()                       # a downloader may still be writing this part (the final
            while not Path(s).exists():            # file name only appears once hf_hub_download finished it)
                if time.time() - t0 > self.wait_s:
                    raise FileNotFoundError(f"part never appeared: {s}")
                time.sleep(5)
        print(f"[stream] part {self.k + 1}/{len(self.sources)}: {s}", flush=True)
        self.cur = urllib.request.urlopen(s) if s.startswith("http") else open(s, "rb")

    def readable(self):
        return True

    def readinto(self, b):
        while self.cur is not None:
            n = self.cur.readinto(b) if hasattr(self.cur, "readinto") else None
            if n is None:
                d = self.cur.read(len(b))
                n = len(d)
                b[:n] = d
            if n:
                return n
            self._next()
        return 0


def part_suffix(i):
    """0 -> 'aa', 1 -> 'ab', ... 26 -> 'ba' (split(1) naming of DADA2000.part_*)."""
    return chr(97 + i // 26) + chr(97 + i % 26)


def run_dada(args, badas):
    rows = load_rows("dada", args.splits, 0)
    by_key = {}
    for r in rows:
        by_key.setdefault((r["gt"]["type"], r["gt"]["video"]), []).append(r)
    cap = len(rows) if not args.limit else args.limit
    sink = Sink(Path(args.out_dir), "dada", capacity=cap)
    n_enc = 0

    sources = ([str(Path(args.parts_dir) / f"DADA2000.part_{part_suffix(i)}") for i in range(args.n_parts)]
               if args.parts_dir else [DADA_URL.format(part_suffix(i)) for i in range(args.url_parts)])
    tf = tarfile.open(fileobj=io.BufferedReader(ChainedStream(sources, args.delete_parts), 1 << 20), mode="r|gz")

    def flush(key, frames):
        nonlocal n_enc
        for r in by_key.get(key, []):
            if r["id"] in sink.done or (args.limit and n_enc >= args.limit):
                continue
            idx = window_indices(r["end_frame"])
            if not all(i in frames for i in idx):
                print(f"  [skip] {r['id']}: frames missing", flush=True)
                continue
            rgb = [cv2.cvtColor(cv2.imdecode(np.frombuffer(frames[i], np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
                   for i in idx]
            tokens, p = encode(badas, preprocess_frames(badas.vjepa, rgb))
            sink.add(r["id"], tokens, p, {"split": r["split"], "label": r["label"]})
            n_enc += 1
            if n_enc % 25 == 0:
                print(f"  [dada] encoded {n_enc} windows", flush=True)

    cur_key, cur = None, {}
    try:
        for m in tf:
            parts = m.name.split("/")
            if not (m.isfile() and len(parts) >= 4 and parts[-2] == "images"):
                continue
            try:
                key = (int(parts[-4]), int(parts[-3]))
            except ValueError:
                continue
            if key != cur_key:
                flush(cur_key, cur)
                cur_key, cur = key, {}
                if args.limit and n_enc >= args.limit:
                    break
            if key in by_key:
                cur[int(parts[-1].split(".")[0])] = tf.extractfile(m).read()
        else:
            flush(cur_key, cur)
    except tarfile.ReadError as e:
        if not args.allow_truncated:
            raise
        print(f"[warn] stream ended mid-member ({e}); expected when only the first parts are streamed")
    sink.close()
    print(f"[dada] encoded {n_enc} windows -> {sink.fx}")


# ----------------------------------------------------------------------------- HF window repos (chunked, no big local file)
def run_from_hf(args, badas):
    """Encode the windows of a private HF window repo (made by r1_build_window_repo.py) shard by shard.
    Per shard: download -> encode -> chunks/<source>/<shard>.npy (+ .jsonl written LAST = done marker) -> optional push to
    --push-repo under <source>/ -> delete the shard and the chunk .npy. Resumable: finished shards are skipped."""
    import collections
    from huggingface_hub import HfApi, hf_hub_download
    api, work = HfApi(), Path(args.out_dir)
    local_repo = Path(args.from_hf) if Path(args.from_hf).is_dir() else None     # a folder = dry run on a local build

    def fetch(f):
        return str(local_repo / f) if local_repo else hf_hub_download(args.from_hf, f, repo_type="dataset", local_dir=str(work / "_hf"))

    wp = fetch("windows.jsonl")
    rows = [json.loads(l) for l in open(wp, encoding="utf-8")]
    rows = [r for r in rows if r["split"] in args.splits]
    by_shard = collections.OrderedDict()
    for r in sorted(rows, key=lambda r: (r["shard"], r["window_id"])):
        by_shard.setdefault(r["shard"], {})[r["window_id"]] = r
    cdir = work / "chunks" / args.source
    cdir.mkdir(parents=True, exist_ok=True)
    remote = set()
    if args.push_repo:
        api.create_repo(args.push_repo, repo_type="dataset", private=True, exist_ok=True)
        remote = set(api.list_repo_files(args.push_repo, repo_type="dataset"))
    print(f"[hf] {len(rows)} windows in {len(by_shard)} shards from {args.from_hf}", flush=True)
    t0, n_done, worst = time.time(), 0, 0.0
    for si, (shard, meta) in enumerate(by_shard.items()):
        if args.max_shards and si >= args.max_shards:
            break
        name = Path(shard).stem
        if (cdir / f"{name}.jsonl").exists() or f"{args.source}/{name}.jsonl" in remote:
            print(f"  [skip] {name} already encoded", flush=True)
            continue
        sp = fetch(shard)
        npy = cdir / f"{name}.npy"
        arr = np.lib.format.open_memmap(npy, mode="w+", dtype=np.float16, shape=(len(meta), N_TOK, DIM))
        recs = []
        with tarfile.open(sp) as tf:
            for m in tf:
                wid = m.name[:-4]
                if wid not in meta:
                    continue
                frames = np.load(io.BytesIO(tf.extractfile(m).read()))["frames"]
                tokens, p = encode(badas, preprocess_frames(badas.vjepa, list(frames)))
                arr[len(recs)] = tokens
                ref = meta[wid].get("a1_p_collision")
                if ref is not None:
                    worst = max(worst, abs(p - ref))
                recs.append({"id": wid, "row": len(recs), "p_collision": round(p, 5), "split": meta[wid]["split"],
                             "label": meta[wid]["label"], "a1_ref": ref})
        assert len(recs) == len(meta), (name, len(recs), len(meta))
        arr.flush()
        del arr
        ix = cdir / f"{name}.jsonl"
        ix.write_text("".join(json.dumps(r) + "\n" for r in recs), encoding="utf-8")
        n_done += len(recs)
        if args.push_repo:
            api.upload_file(path_or_fileobj=str(npy), path_in_repo=f"{args.source}/{name}.npy", repo_id=args.push_repo, repo_type="dataset")
            api.upload_file(path_or_fileobj=str(ix), path_in_repo=f"{args.source}/{name}.jsonl", repo_id=args.push_repo, repo_type="dataset")
            npy.unlink()
        if not local_repo:
            Path(sp).unlink(missing_ok=True)
        el = time.time() - t0
        print(f"  [{si + 1}/{len(by_shard)}] {name}: {len(recs)} windows, {el / n_done:.2f}s/window, "
              f"max|p - recorded A1 p| so far = {worst:.4f}", flush=True)
    print(f"[hf] done; max|p - recorded A1 p| = {worst:.4f} (tolerance 0.004)", flush=True)


def consolidate(args):
    """Training pod: merge the <source>/*.npy|jsonl chunks of the features repo into the features_<source>.npy
    memmap + index_<source>.jsonl that r1_data.Cache reads. Chunks are fetched one at a time and deleted after
    merging, so peak disk = final cache + one chunk (not 2x the cache)."""
    out = Path(args.out_dir)
    local = Path(args.consolidate_from).is_dir()                     # a local chunks folder (smoke runs)
    if local:
        names = sorted(p.stem for p in (Path(args.consolidate_from) / args.source).glob("*.jsonl"))
        get = lambda f: str(Path(args.consolidate_from) / f)          # noqa: E731
    else:
        from huggingface_hub import HfApi, hf_hub_download
        names = sorted(Path(f).stem for f in HfApi().list_repo_files(args.consolidate_from, repo_type="dataset")
                       if f.startswith(f"{args.source}/") and f.endswith(".jsonl"))
        get = lambda f: hf_hub_download(args.consolidate_from, f, repo_type="dataset", local_dir=str(out / "_features_repo"))  # noqa: E731
    recs = {n: [json.loads(l) for l in open(get(f"{args.source}/{n}.jsonl"), encoding="utf-8")] for n in names}
    sink = Sink(out, args.source, capacity=sum(len(r) for r in recs.values()))
    for n in names:
        if all(r["id"] in sink.done for r in recs[n]):
            continue
        npy = get(f"{args.source}/{n}.npy")
        mm = np.load(npy, mmap_mode="r")
        for r in recs[n]:
            if r["id"] not in sink.done:
                sink.add(r["id"], np.asarray(mm[r["row"]]), r["p_collision"], {"split": r["split"], "label": r["label"]})
        del mm
        if not local:
            Path(npy).unlink(missing_ok=True)
    sink.close()
    print(f"[consolidate] {args.source}: {sink.n} windows -> {sink.fx}", flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, choices=["nexar", "dada"])
    ap.add_argument("--from-hf", default=None, help="encode the shards of this private HF window repo (chunked, resumable)")
    ap.add_argument("--max-shards", type=int, default=0, help="with --from-hf: encode only the first N shards (smoke runs)")
    ap.add_argument("--push-repo", default=None, help="with --from-hf: upload each encoded chunk here, then delete it locally")
    ap.add_argument("--consolidate-from", default=None, help="merge the chunks of this HF features repo into features_<source>.npy")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--splits", default="train,val")
    ap.add_argument("--limit", type=int, default=0, help="encode at most N windows (dry runs)")
    ap.add_argument("--parts-dir", default=None, help="dir with DADA2000.part_aa ... (downloaded in order)")
    ap.add_argument("--n-parts", type=int, default=58, help="number of DADA2000.part_* files expected in --parts-dir")
    ap.add_argument("--url-parts", type=int, default=0, help="stream the first N parts over HTTP")
    ap.add_argument("--delete-parts", action="store_true")
    ap.add_argument("--allow-truncated", action="store_true", help="dry runs on the first parts only: tolerate the cut at the part boundary")
    args = ap.parse_args()
    args.splits = set(args.splits.split(","))
    if args.consolidate_from:
        return consolidate(args)
    if args.source == "dada" and not (args.parts_dir or args.url_parts or args.from_hf):
        ap.error("dada needs --from-hf, --parts-dir or --url-parts")
    badas = load_model()
    if args.from_hf:
        return run_from_hf(args, badas)
    (run_nexar if args.source == "nexar" else run_dada)(args, badas)


if __name__ == "__main__":
    main()
