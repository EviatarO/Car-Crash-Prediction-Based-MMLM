"""
r1_hf_sync.py - keep the results of a long pod run safe on HF while it runs (plan 2026-10-19 rev 2). A pod that is deleted, crashes or runs out of credit loses its disk;
this copies what matters to a private HF repo as it appears, and lets a fresh pod continue from the last finished stage.

  push     upload every file of <dir>/B (train_log.jsonl, epoch_XX.pt, best.pt, the training-stage markers done.B_*) and <dir>/logs (small logs, the status file) that is missing on HF or has
           another size, skipping files changed in the last 20 s; one pass, or --loop SECONDS for a background loop. Sizes are verified after the upload.
  pull     download everything under <repo>/<prefix>/ into <dir> that is missing locally (the restore for a fresh pod): the stage markers `done.B_*` come back too, so the
           driver skips the finished TRAINING stages (the preparation stages are always redone: the cache and the videos are not on HF).
Files go to <repo>/<prefix>/<relative path>. The token comes from the environment (HF_TOKEN). A failed upload is printed and retried on the next pass; it never stops the run.

    python r1_hf_sync.py push --dir /root/r1_chains --repo eviatarO-org/checkpoints --prefix chains_2026-10-19/live [--loop 600]
    python r1_hf_sync.py pull --dir /root/r1_chains --repo eviatarO-org/checkpoints --prefix chains_2026-10-19/live
"""
from __future__ import annotations

import argparse
import fnmatch
import os
import sys
import time
from pathlib import Path

PATTERNS = ("train_log.jsonl", "epoch_*.pt", "best.pt", "done.B_*", "top3.txt", "*.log", "chains.status")
SUBDIRS = ("B", "logs", ".")                           # chain B, the logs, and the done.* markers at the top of <dir>


def candidates(root: Path):
    seen = set()
    for sub in SUBDIRS:
        base = root if sub == "." else root / sub
        if not base.exists():
            continue
        walk = [(base, [], [f.name for f in base.iterdir() if f.is_file()])] if sub == "." else os.walk(base)
        for d, _, names in walk:
            for n in names:
                if any(fnmatch.fnmatch(n, p) for p in PATTERNS):
                    p = Path(d) / n
                    if p not in seen:
                        seen.add(p)
                        yield p


def push_once(api, root, repo, prefix, min_age=20.0, dry=False):
    files = [p for p in candidates(root) if time.time() - p.stat().st_mtime > min_age]
    rels = {p: f"{prefix}/{p.relative_to(root).as_posix()}" for p in files}
    remote = {}
    if files and not dry:
        for i in range(0, len(files), 200):
            for x in api.get_paths_info(repo, [rels[p] for p in files[i:i + 200]], repo_type="model"):
                remote[x.path] = x.size
    todo = [p for p in files if remote.get(rels[p]) != p.stat().st_size]
    sent = 0
    for p in todo:
        if dry:
            print(f"[sync] would upload {rels[p]} ({p.stat().st_size} bytes)")
            continue
        try:
            size = p.stat().st_size
            api.upload_file(path_or_fileobj=str(p), path_in_repo=rels[p], repo_id=repo, repo_type="model")
            got = {x.path: x.size for x in api.get_paths_info(repo, [rels[p]], repo_type="model")}
            if got.get(rels[p]) == size:
                sent += 1
            else:
                print(f"[sync] size mismatch after uploading {rels[p]} (local {size}, remote {got.get(rels[p])}); retried next pass", flush=True)
        except Exception as e:                                                   # noqa: BLE001  never stop the run
            print(f"[sync] upload of {rels[p]} failed: {type(e).__name__}: {str(e)[:120]}; retried next pass", flush=True)
    print(f"[sync] {time.strftime('%H:%M:%S')}: {len(files)} candidate files, {len(todo)} to upload, {sent} uploaded", flush=True)
    return sent


def pull(api, root, repo, prefix):
    from huggingface_hub import hf_hub_download
    try:
        items = [x for x in api.list_repo_tree(repo, path_in_repo=prefix, recursive=True, repo_type="model") if hasattr(x, "size")]
    except Exception as e:                                                       # noqa: BLE001
        print(f"[sync] nothing to restore ({type(e).__name__})")
        return 0
    n = 0
    for x in items:
        rel = x.path[len(prefix) + 1:]
        dst = root / rel
        if dst.exists() and dst.stat().st_size == x.size:
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = hf_hub_download(repo, x.path, repo_type="model", local_dir=str(root / "_restore"))
        os.replace(tmp, dst)
        n += 1
    print(f"[sync] restored {n} files from {repo}/{prefix}")
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["push", "pull"])
    ap.add_argument("--dir", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--loop", type=int, default=0, help="push: repeat every N seconds until killed")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    root = Path(a.dir)
    api = None
    if not a.dry_run:
        from huggingface_hub import HfApi
        api = HfApi()
    if a.mode == "pull":
        pull(api, root, a.repo, a.prefix)
        return
    while True:
        push_once(api, root, a.repo, a.prefix, dry=a.dry_run)
        if not a.loop:
            break
        time.sleep(a.loop)


if __name__ == "__main__":
    sys.exit(main())
