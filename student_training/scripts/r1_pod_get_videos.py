"""
r1_pod_get_videos.py - download the Nexar training videos needed by the teacher dataset from the Hugging Face dataset
nexar-ai/nexar_collision_prediction (train/positive/<id>.mp4, train/negative/<id>.mp4; gated "auto": the account must have accepted the
terms) into one flat folder <out>/<id>.mp4. Resumable (existing non-empty files are skipped), parallel, sizes verified against the repo.

    python r1_pod_get_videos.py --windows /root/teacher_dataset/windows.jsonl --out /root/nexar/train [--threads 16] [--check]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

REPO = "nexar-ai/nexar_collision_prediction"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--prefix", default="train/", help="repo folder: train/ (default) or test-private/ (the private test videos, plan 2026-10-17)")
    ap.add_argument("--check", action="store_true", help="only verify access and the file list (downloads one small file)")
    args = ap.parse_args()
    need = sorted({json.loads(l)["video_id"] for l in open(args.windows, encoding="utf-8")})
    sib = {f.rfilename: f.size for f in HfApi().dataset_info(REPO, files_metadata=True).siblings}
    where = {k.split("/")[-1][:-4]: k for k in sib if k.startswith(args.prefix) and k.endswith(".mp4")}
    missing = [v for v in need if v not in where]
    print(f"[videos] need {len(need)}, in the repo {len(where)} train videos, not found: {missing[:5]}", flush=True)
    assert not missing
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cache = Path(tempfile.mkdtemp(prefix="hfdl_", dir=str(out.parent)))
    if args.check:
        v = need[0]
        p = hf_hub_download(REPO, where[v], repo_type="dataset", local_dir=str(cache))
        print(f"[videos] access OK: {where[v]} -> {Path(p).stat().st_size} bytes (repo says {sib[where[v]]})")
        shutil.rmtree(cache, ignore_errors=True)
        return

    def get(v):
        dst = out / f"{v}.mp4"
        if dst.exists() and dst.stat().st_size == sib[where[v]]:
            return v, "skip"
        p = hf_hub_download(REPO, where[v], repo_type="dataset", local_dir=str(cache))
        shutil.move(p, dst)
        assert dst.stat().st_size == sib[where[v]], f"{v}: size {dst.stat().st_size} != {sib[where[v]]}"
        return v, "ok"

    n_ok = n_skip = 0
    with ThreadPoolExecutor(args.threads) as pool:
        futs = [pool.submit(get, v) for v in need]
        for k, fut in enumerate(as_completed(futs), 1):
            _, st = fut.result()
            n_ok += st == "ok"
            n_skip += st == "skip"
            if k % 100 == 0:
                print(f"  [{k}/{len(need)}] downloaded {n_ok}, already there {n_skip}", flush=True)
    shutil.rmtree(cache, ignore_errors=True)
    bad = [v for v in need if not (out / f"{v}.mp4").exists() or (out / f"{v}.mp4").stat().st_size != sib[where[v]]]
    print(f"[videos] done: {n_ok} downloaded, {n_skip} already there, {len(bad)} wrong or missing", flush=True)
    assert not bad, bad[:5]


if __name__ == "__main__":
    sys.exit(main())
