"""
r1_download_dada_parts.py - download the MM-AU DADA-2000 tar.gz parts (DADA2000.part_aa ... , ~2.15 GB each,
~131 GB total) to a local directory, in order, resumably, with back-pressure so the disk never holds more
than --max-ahead unconsumed parts (r1_cache_features.py --delete-parts removes a part once it has read it).

The parts are pieces of ONE tar.gz, so they must be consumed in order. A part's final file name only appears
once hf_hub_download has finished it, so the consumer can safely wait for it.

    python r1_download_dada_parts.py --dir /workspace/dada_parts --max-ahead 6
      -> /workspace/dada_parts/DADA-2000_chunks/DADA2000.part_aa ...
"""
import argparse
import time
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

REPO = "JeffreyChou/MM-AU"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--max-ahead", type=int, default=6, help="max parts on disk not yet consumed")
    ap.add_argument("--retries", type=int, default=6)
    args = ap.parse_args()
    parts = sorted(f for f in HfApi().list_repo_files(REPO, repo_type="dataset")
                   if f.startswith("DADA-2000_chunks/DADA2000.part_"))
    print(f"[dl] {len(parts)} parts: {parts[0]} ... {parts[-1]}", flush=True)
    out = Path(args.dir) / "DADA-2000_chunks"
    for i, f in enumerate(parts):
        while len(list(out.glob("DADA2000.part_*"))) >= args.max_ahead:
            time.sleep(15)                       # consumer has not freed disk yet
        target = out / Path(f).name
        if target.exists():
            continue
        for attempt in range(1, args.retries + 1):
            try:
                hf_hub_download(REPO, f, repo_type="dataset", local_dir=args.dir)
                print(f"[dl] {i + 1}/{len(parts)} {target.name}", flush=True)
                break
            except Exception as e:                # network hiccup: retry (hf_hub_download resumes)
                print(f"[dl] {target.name} attempt {attempt} failed: {type(e).__name__}: {e}", flush=True)
                time.sleep(10 * attempt)
        else:
            raise SystemExit(f"giving up on {f}")
    print("[dl] all parts downloaded", flush=True)


if __name__ == "__main__":
    main()
