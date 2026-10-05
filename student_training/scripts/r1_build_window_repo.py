"""
r1_build_window_repo.py - cut the week-1 windows ONCE into the exact frames the frozen V-JEPA2 encoder receives
(16 frames, full frame squashed to 256x256 = "compress256"), and publish them to a PRIVATE Hugging Face dataset repo
together with one configuration file. Pods become disposable: training pulls from HF, nothing durable stays on a pod.

Repo layout (per source):
  README.md                         dataset card (source, license note, preprocessing, field dictionary, counts)
  windows.jsonl / windows.csv       one row per VALID window (the configuration / setting file)
  windows_excluded.jsonl            windows removed by the visibility rule / bounds, with drop_reason (audit only)
  shards/<split>-NNNN.tar           WebDataset-style tar, member <window_id>.npz = {frames: uint8 (16,256,256,3)}, lossless
  index/<split>-NNNN.jsonl          rows of that shard (written right after the shard upload; makes the run resumable)

Exactness: the frames are the output of the V-JEPA2 processor's OWN resize step (do_rescale/do_normalize off, uint8).
Feeding them back through the processor (resize to 256 is then an identity) gives a tensor bit-identical to the one made
from the original frames; the script asserts max|diff| == 0 on the first --verify-n windows of every run.

  Nexar (frames are already cut as 16-frame windows under dataset/train/<frames_dir>/):
    python r1_build_window_repo.py --source nexar --repo eviatarO-org/nexar-windows
  DADA (streams the 58 tar.gz parts in order; r1_download_dada_parts.py feeds --parts-dir with back-pressure):
    python r1_build_window_repo.py --source dada --repo eviatarO-org/mmau-dada-windows --parts-dir /root/dada_parts/DADA-2000_chunks
  Dry run (no upload, keeps shards in --work-dir):  add --no-upload --limit 12
"""
from __future__ import annotations

import argparse
import collections
import csv
import io
import json
import sys
import tarfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from r1_cache_features import DADA_URL, MANIFEST, ChainedStream, part_suffix  # noqa: E402
from r1_common import FPS_SRC, N_FR, STRIDE, window_indices  # noqa: E402

PROC_REPO = "facebook/vjepa2-vitl-fpc16-256-ssv2"           # the processor BADAS-Open itself uses
RESIZE = dict(size={"height": 256, "width": 256}, do_center_crop=False)
PREPROCESS = ("compress256: full frame squashed to 256x256 by the V-JEPA2 processor resize (bilinear, uint8); "
              "no crop; normalisation is applied later by the processor")
A1_SCORES = ROOT / "outputs" / "e4_vjepa_reason" / "pool1761_scores" / "A1-compress256.jsonl"


# ----------------------------------------------------------------------------- frames
def get_processor():
    from transformers import AutoVideoProcessor
    return AutoVideoProcessor.from_pretrained(PROC_REPO)


def resize_u8(proc, frames_rgb) -> np.ndarray:
    """16 RGB uint8 frames (any size) -> uint8 (16,256,256,3) using the processor's own resize."""
    r = proc(videos=frames_rgb, return_tensors="pt", do_rescale=False, do_normalize=False, **RESIZE)["pixel_values_videos"]
    assert r.dtype == torch.uint8 and r.shape == (1, N_FR, 3, 256, 256), (r.dtype, r.shape)
    return r[0].permute(0, 2, 3, 1).contiguous().numpy()


def check_exact(proc, frames_rgb, stored):
    a = proc(videos=frames_rgb, return_tensors="pt", **RESIZE)["pixel_values_videos"]
    b = proc(videos=[f for f in stored], return_tensors="pt", **RESIZE)["pixel_values_videos"]
    d = float((a - b).abs().max())
    assert d == 0.0, f"stored frames are not bit-identical through the processor (max diff {d})"


# ----------------------------------------------------------------------------- records
def base_record(row, extra):
    return {"window_id": row["id"], "source": row["source"], "video_id": row["video_key"], "split": row["split"],
            "label": row["label"], "tte_group": row["tte"], "gt_label_text": "collision" if row["label"] else "no collision",
            "n_frames": N_FR, "frame_size": 256, "frame_stride": STRIDE, "fps_source": FPS_SRC,
            "preprocess": PREPROCESS, "valid": True, "drop_reason": None,
            "r1_targets": row["targets"], "a1_p_collision": None, **extra}


def nexar_extra(row, times, a1):
    t = times.get(row["video_key"], {})
    return {"time_of_alert_s": t.get("alert"), "time_of_event_s": t.get("event"),
            "window_end_s": (round(t["event"] - row["tte"], 3) if t.get("event") is not None and row["label"] == 1 else None),
            "source_frames_dir": row["frames_dir"], "source_frame_indices": None,
            "reasoning_text": row["gt"]["v12_caption"], "reasoning_kind": "teacher V12 neutral caption (Gemini, blind)",
            "a1_p_collision": a1.get(row["frames_dir"])}


def dada_extra(row):
    g = row["gt"]
    idx = window_indices(row["end_frame"])
    return {"type": g["type"], "video": g["video"],
            "time_of_alert_s": round(g["t_ai"] / FPS_SRC, 3), "time_of_event_s": round(g["t_co"] / FPS_SRC, 3),
            "t_ai_frame": g["t_ai"], "t_co_frame": g["t_co"], "t_ae_frame": g["t_ae"], "video_total_frames": g["total"],
            "window_end_frame": row["end_frame"], "window_end_s": round(row["end_frame"] / FPS_SRC, 3),
            "source_frame_indices": idx,
            "reasoning_text": f"{g['event']}; {g['cause']}", "event": g["event"], "cause": g["cause"],
            "reasoning_kind": "human annotation (MM-AU closed list)", "ara": g["ara"]}


# ----------------------------------------------------------------------------- shard writer + uploader
class Uploader:
    def __init__(self, repo, enabled, work: Path):
        self.repo, self.enabled, self.work = repo, enabled, work
        self.pool = ThreadPoolExecutor(1)
        self.sem = threading.Semaphore(2)               # at most 2 finished shards waiting on disk
        self.futs = []
        if enabled:
            from huggingface_hub import HfApi
            self.api = HfApi()
            self.api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)

    def _up(self, local: Path, remote: str, delete: bool):
        try:
            if self.enabled:
                for attempt in range(1, 7):
                    try:
                        self.api.upload_file(path_or_fileobj=str(local), path_in_repo=remote, repo_id=self.repo, repo_type="dataset")
                        break
                    except Exception as e:
                        print(f"[up] {remote} attempt {attempt} failed: {type(e).__name__}: {e}", flush=True)
                        time.sleep(10 * attempt)
                else:
                    raise RuntimeError(f"upload failed: {remote}")
                print(f"[up] {remote} ({local.stat().st_size / 1e6:.0f} MB)", flush=True)
                if delete:
                    local.unlink(missing_ok=True)
        finally:
            if delete:
                self.sem.release()

    def submit(self, local: Path, remote: str, delete=True):
        if delete:
            self.sem.acquire()
        self.futs.append(self.pool.submit(self._up, local, remote, delete))

    def drain(self):
        for f in self.futs:
            f.result()
        self.futs = []


class ShardWriter:
    def __init__(self, work: Path, up: Uploader, shard_size: int, first_no: dict):
        self.work, self.up, self.size = work, up, shard_size
        self.no = collections.defaultdict(int, first_no)
        self.cur = {}                                   # split -> (name, TarFile, rows)
        (work / "shards").mkdir(parents=True, exist_ok=True)
        (work / "index").mkdir(parents=True, exist_ok=True)
        self.rows = []

    def add(self, rec, arr):
        sp = rec["split"]
        if sp not in self.cur:
            name = f"{sp}-{self.no[sp]:04d}"
            self.no[sp] += 1
            self.cur[sp] = (name, tarfile.open(self.work / "shards" / f"{name}.tar", "w"), [])
        name, tf, rows = self.cur[sp]
        buf = io.BytesIO()
        np.savez_compressed(buf, frames=arr)
        ti = tarfile.TarInfo(f"{rec['window_id']}.npz")
        ti.size = buf.tell()
        buf.seek(0)
        tf.addfile(ti, buf)
        rec = {**rec, "shard": f"shards/{name}.tar"}
        rows.append(rec)
        self.rows.append(rec)
        if len(rows) >= self.size:
            self._close(sp)

    def _close(self, sp):
        name, tf, rows = self.cur.pop(sp)
        tf.close()
        ix = self.work / "index" / f"{name}.jsonl"
        ix.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        self.up.submit(self.work / "shards" / f"{name}.tar", f"shards/{name}.tar")
        self.up.submit(ix, f"index/{name}.jsonl", delete=False)

    def close(self):
        for sp in list(self.cur):
            self._close(sp)
        self.up.drain()


# ----------------------------------------------------------------------------- resume
def remote_state(repo, work: Path):
    """Index files already in the repo -> (rows, next shard number per split)."""
    from huggingface_hub import HfApi, hf_hub_download
    rows, nxt = [], collections.defaultdict(int)
    try:
        files = [f for f in HfApi().list_repo_files(repo, repo_type="dataset") if f.startswith("index/") and f.endswith(".jsonl")]
    except Exception:
        return rows, nxt
    for f in sorted(files):
        p = hf_hub_download(repo, f, repo_type="dataset", local_dir=str(work / "_remote"))
        rows += [json.loads(l) for l in open(p, encoding="utf-8")]
        sp, n = Path(f).stem.rsplit("-", 1)
        nxt[sp] = max(nxt[sp], int(n) + 1)
    return rows, nxt


# ----------------------------------------------------------------------------- sources
def load_manifest(source, splits, limit):
    rows = [json.loads(l) for l in open(MANIFEST[source], encoding="utf-8")]
    excluded = [r for r in rows if not r["valid"]]
    rows = [r for r in rows if r["valid"] and r["split"] in splits]
    rows.sort(key=lambda r: (r["split"], r["id"]))
    return (rows[:limit] if limit else rows), excluded


def run_nexar(args, proc, writer, done_ids):
    import pandas as pd
    from PIL import Image
    t = pd.read_csv(ROOT / "dataset" / "train.csv")
    times = {f"{int(i):05d}": {"alert": None if pd.isna(a) else float(a), "event": None if pd.isna(e) else float(e)}
             for i, a, e in zip(t.id, t.time_of_alert, t.time_of_event)}
    a1 = {}
    if A1_SCORES.exists():
        for l in open(A1_SCORES, encoding="utf-8"):
            r = json.loads(l)
            a1[r["frames_dir"]] = round(r["score"], 5)
    rows, excluded = load_manifest("nexar", args.splits, args.limit)
    todo = [r for r in rows if r["id"] not in done_ids]
    print(f"[nexar] {len(rows)} valid windows, {len(todo)} to build, {len(excluded)} excluded by the visibility rule", flush=True)

    def work(i_row):
        i, r = i_row
        fd = ROOT / "dataset" / "train" / r["frames_dir"]
        paths = sorted(fd.glob("frame_*.jpg"))
        assert len(paths) == N_FR, (fd, len(paths))
        frames = [np.array(Image.open(p).convert("RGB")) for p in paths]
        arr = resize_u8(proc, frames)
        if i < args.verify_n:
            check_exact(proc, frames, arr)
        return base_record(r, nexar_extra(r, times, a1)), arr

    t0 = time.time()
    n = 0
    with ThreadPoolExecutor(args.threads) as ex:
        for b in range(0, len(todo), 64):                  # batches keep memory bounded (3 MB per window)
            for rec, arr in ex.map(work, enumerate(todo[b:b + 64], start=b)):
                writer.add(rec, arr)
                n += 1
            print(f"  [{n}/{len(todo)}] {(time.time() - t0) / n:.2f}s/window", flush=True)
    return rows, excluded


def run_dada(args, proc, writer, done_ids):
    rows, excluded = load_manifest("dada", args.splits, 0)
    by_key = collections.defaultdict(list)
    for r in rows:
        if r["id"] not in done_ids:
            by_key[(r["gt"]["type"], r["gt"]["video"])].append(r)
    if args.limit:                                     # dry run: first N windows (optionally only ids with a prefix)
        keep = {r["id"] for r in rows if r["id"] not in done_ids and r["id"].startswith(args.id_prefix)}
        keep = set(sorted(keep)[:args.limit])
        by_key = collections.defaultdict(list, {k: [r for r in v if r["id"] in keep] for k, v in by_key.items()})
        by_key = collections.defaultdict(list, {k: v for k, v in by_key.items() if v})
    needed = {k: {i for r in v for i in window_indices(r["end_frame"])} for k, v in by_key.items()}
    total = sum(len(v) for v in by_key.values())
    print(f"[dada] {len(rows)} valid windows, {total} to build in {len(by_key)} sequences, {len(excluded)} excluded", flush=True)

    sources = ([str(Path(args.parts_dir) / f"DADA2000.part_{part_suffix(i)}") for i in range(args.n_parts)] if args.parts_dir
               else [DADA_URL.format(part_suffix(i)) for i in range(args.url_parts)])
    tf = tarfile.open(fileobj=io.BufferedReader(ChainedStream(sources, args.delete_parts), 1 << 20), mode="r|gz")
    pool = ThreadPoolExecutor(args.threads)
    pending, built, verified = collections.deque(), [0], [0]
    t0 = time.time()

    def cut(r, frames):
        idx = window_indices(r["end_frame"])
        if not all(i in frames for i in idx):
            return None
        rgb = [cv2.cvtColor(cv2.imdecode(np.frombuffer(frames[i], np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB) for i in idx]
        arr = resize_u8(proc, rgb)
        return base_record(r, dada_extra(r)), arr, (rgb if verified[0] < args.verify_n else None)

    def drain(limit):
        while len(pending) > limit:
            res = pending.popleft().result()
            if res is None:
                print("  [skip] frames missing", flush=True)
                continue
            rec, arr, rgb = res
            if rgb is not None:
                check_exact(proc, rgb, arr)
                verified[0] += 1
            writer.add(rec, arr)
            built[0] += 1
            if built[0] % 100 == 0:
                print(f"  [dada] {built[0]}/{total} windows, {(time.time() - t0) / built[0]:.2f}s/window", flush=True)

    def flush(key, frames):
        for r in by_key.get(key, []):
            pending.append(pool.submit(cut, r, dict(frames)))
        drain(args.threads * 2)

    cur_key, cur, seen = None, {}, set()
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
                if cur_key in by_key:
                    seen.add(cur_key)
                cur_key, cur = key, {}
                if len(seen) == len(by_key):
                    break                              # every wanted sequence is done: stop reading the stream
            if key in needed:
                n = int(parts[-1].split(".")[0])
                if n in needed[key]:
                    cur[n] = tf.extractfile(m).read()
        else:
            flush(cur_key, cur)
    except tarfile.ReadError as e:
        if not args.allow_truncated:
            raise
        print(f"[warn] stream ended mid-member ({e}); expected for dry runs on the first parts")
    drain(0)
    pool.shutdown()
    print(f"[dada] built {built[0]} windows", flush=True)
    return rows, excluded


# ----------------------------------------------------------------------------- final files
CARD = """\
---
license: other
pretty_name: {name}
private: true
---
# {name}

Encoder-ready windows for the V-JEPA2 -> projector -> Qwen3-VL reasoning-path experiment (thesis, Oct 2026).
**Private; academic use only** ({lic}). Do not redistribute.

* Source: {src}
* Windows: **{n}** valid ({per_split}); excluded by the window visibility rule / bounds: {nex} (see `windows_excluded.jsonl`).
* Each window = 16 frames, every {stride}th frame of a {fps} fps source (2.0 s at 7.5 fps), stored as **uint8 (16, 256, 256, 3)**.
* Preprocessing: {prep}.
* The frames are bit-identical to what the encoder's processor sees: feeding them to the V-JEPA2 processor with
  `size=256x256, do_center_crop=False` reproduces the tensor made from the original frames (asserted, max diff 0.0).
* Window rule: a crash window ends `TTE in {{0.5, 1.0, 1.5}}` s before the event and is kept only if the hazard is already
  visible inside it ({vis}). No-crash windows: {neg}.

## Files
`windows.jsonl` / `windows.csv` (configuration; `shard` = tar holding the frames), `windows_excluded.jsonl`,
`shards/<split>-NNNN.tar` (member `<window_id>.npz`, key `frames`), `index/` (per-shard rows, used for resuming).

## Read a window
```python
import tarfile, numpy as np, io, json
rows = [json.loads(l) for l in open("windows.jsonl")]
r = rows[0]
with tarfile.open(r["shard"]) as tf:
    frames = np.load(io.BytesIO(tf.extractfile(r["window_id"] + ".npz").read()))["frames"]   # (16,256,256,3) uint8
```

## Fields
`window_id, source, video_id, split, label (1 crash / 0 no-crash), tte_group (0.5/1.0/1.5/null), time_of_alert_s, time_of_event_s,
window_end_s, source_frame_indices, fps_source, reasoning_text, reasoning_kind, r1_targets {{phase1, phase2}}, a1_p_collision
(frozen A1-compress256 crash score; filled when recorded), preprocess, valid, drop_reason, shard` + source specific fields.
"""


def finish(args, rows_all, excluded, work: Path, up: Uploader):
    rows_all = sorted(rows_all, key=lambda r: (r["split"], r["window_id"]))
    (work / "windows.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows_all), encoding="utf-8")
    (work / "windows_excluded.jsonl").write_text(
        "".join(json.dumps({"window_id": r["id"], "video_id": r["video_key"], "split": r["split"], "label": r["label"],
                            "tte_group": r["tte"], "valid": False, "drop_reason": r["drop_reason"]}) + "\n" for r in excluded), encoding="utf-8")
    cols = list(dict.fromkeys(k for r in rows_all for k in r))
    with open(work / "windows.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows_all:
            w.writerow([json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v for v in (r.get(c) for c in cols)])
    c = collections.Counter(r["split"] for r in rows_all)
    meta = {"nexar": dict(name="nexar-windows", src="Nexar collision-prediction train videos (our 1,761-window V12 pool; text = teacher V12 captions)",
                          lic="Nexar dataset terms", vis="end >= time_of_alert", neg="negative videos, 'Cause: not annotated' in the targets"),
            "dada": dict(name="mmau-dada-windows", src="MM-AU / DADA-2000 part (JeffreyChou/MM-AU, 30 fps, 1584x660 -> squashed); text = human annotations",
                         lic="MM-AU CC BY-NC", vis="end >= t_ai + 8 frames", neg="end 0.5 s before the abnormal start t_ai")}[args.source]
    (work / "README.md").write_text(CARD.format(
        n=len(rows_all), per_split=", ".join(f"{k} {v}" for k, v in sorted(c.items())), nex=len(excluded), stride=STRIDE, fps=FPS_SRC,
        prep=PREPROCESS, **meta), encoding="utf-8")
    for f in ("windows.jsonl", "windows.csv", "windows_excluded.jsonl", "README.md"):
        up.submit(work / f, f, delete=False)
    up.drain()
    print(f"[done] {len(rows_all)} windows {dict(c)}; excluded {len(excluded)}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, choices=["nexar", "dada"])
    ap.add_argument("--repo", required=True)
    ap.add_argument("--work-dir", default=None)
    ap.add_argument("--splits", default="train,val,test")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard-size", type=int, default=400)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--verify-n", type=int, default=20)
    ap.add_argument("--no-upload", action="store_true")
    ap.add_argument("--parts-dir", default=None)
    ap.add_argument("--n-parts", type=int, default=58)
    ap.add_argument("--id-prefix", default="", help="dry runs: only windows whose id starts with this (part_aa holds dada_t08_*)")
    ap.add_argument("--url-parts", type=int, default=0, help="dry runs: stream the first N parts over HTTP")
    ap.add_argument("--delete-parts", action="store_true")
    ap.add_argument("--allow-truncated", action="store_true")
    args = ap.parse_args()
    args.splits = set(args.splits.split(","))
    if args.source == "dada" and not (args.parts_dir or args.url_parts):
        ap.error("dada needs --parts-dir (or --url-parts for a dry run)")
    torch.set_num_threads(1)
    work = Path(args.work_dir or (Path.home() / "r1_window_repos" / args.source))
    work.mkdir(parents=True, exist_ok=True)
    up = Uploader(args.repo, not args.no_upload, work)
    prev, nxt = ([], {}) if args.no_upload else remote_state(args.repo, work)
    done_ids = {r["window_id"] for r in prev}
    print(f"[resume] {len(prev)} windows already in {args.repo}" if prev else "[resume] fresh repo", flush=True)
    proc = get_processor()
    writer = ShardWriter(work, up, args.shard_size, nxt)
    rows, excluded = (run_nexar if args.source == "nexar" else run_dada)(args, proc, writer, done_ids)
    writer.close()
    finish(args, prev + writer.rows, excluded, work, up)


if __name__ == "__main__":
    main()
