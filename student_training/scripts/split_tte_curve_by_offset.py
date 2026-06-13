"""
split_tte_curve_by_offset.py
============================
Split the 1420 TTE-curve frame dirs into 5 per-offset zips + per-offset manifests,
so each offset can be uploaded and inferred independently (pipelining).

Bottleneck is the laptop uplink (~3.7 Mbps), so we upload by scientific priority:
the long horizons (3.0/2.5/2.0s) carry the novel anticipation-decline signal; the
short ones (1.5/1.0s) are cross-checks against existing buckets.

Outputs (per offset suffix tte{10,15,20,25,30}):
  dataset/tte_zips/test_tte_curve_tte{XX}.zip
      -> arcnames: test_tte_curve/{private,public}/{vid}_hires_tte{XX}/frame_*.jpg
         (pod: unzip -d /workspace/data/  ->  /workspace/data/test_tte_curve/...)
  dataset/manifests/test_tte_curve_{private,public}_tte{XX}_manifest.jsonl

Each zip ~835 MB (284 dirs x 16 frames). Run order printed = upload priority.
"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC  = REPO / "dataset" / "test_tte_curve"
ZIPDIR = REPO / "dataset" / "tte_zips"
MANDIR = REPO / "dataset" / "manifests"

# (suffix, requested_tte_s) in UPLOAD PRIORITY order: long horizons first.
OFFSETS = [("tte30", 3.0), ("tte25", 2.5), ("tte20", 2.0), ("tte15", 1.5), ("tte10", 1.0)]
HALVES = ["private", "public"]


def split_manifests() -> None:
    for half in HALVES:
        full = MANDIR / f"test_tte_curve_{half}_manifest.jsonl"
        rows = [json.loads(l) for l in full.read_text().splitlines() if l.strip()]
        for suffix, _ in OFFSETS:
            sub = [r for r in rows if r["frames_dir"].endswith(suffix)]
            out = MANDIR / f"test_tte_curve_{half}_{suffix}_manifest.jsonl"
            out.write_text("\n".join(json.dumps(r) for r in sub) + "\n")
            print(f"  manifest {out.name}: {len(sub)} records")


def build_zip(suffix: str) -> None:
    zpath = ZIPDIR / f"test_tte_curve_{suffix}.zip"
    n_dirs = 0
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_STORED) as zf:
        for half in HALVES:
            half_dir = SRC / half
            for fdir in sorted(half_dir.glob(f"*_hires_{suffix}")):
                jpgs = sorted(fdir.glob("frame_*.jpg"))
                if not jpgs:
                    continue
                for jpg in jpgs:
                    zf.write(jpg, jpg.relative_to(SRC.parent))  # test_tte_curve/half/dir/frame
                n_dirs += 1
    size_gb = zpath.stat().st_size / 1e9
    print(f"  {zpath.name}: {n_dirs} dirs, {size_gb:.2f} GB")


def main() -> None:
    ZIPDIR.mkdir(parents=True, exist_ok=True)
    print("Splitting manifests by offset...")
    split_manifests()
    print("\nBuilding per-offset zips (upload priority order)...")
    for suffix, tte in OFFSETS:
        print(f"[{tte:>3}s] {suffix}")
        build_zip(suffix)
    print("\nDone. Upload order: tte30 -> tte25 -> tte20 -> tte15 -> tte10")


if __name__ == "__main__":
    main()
