"""
upload_tte_curve_frames.py
==========================
Zip the 1420 TTE-curve frame dirs and upload to HF dataset repo.

Run AFTER extract_tte_curve_test_frames.py completes and all 1420 dirs are verified.

  python student_training/scripts/upload_tte_curve_frames.py [--skip-zip]

--skip-zip : reuse an existing zip (e.g. if zip was already built).
"""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC  = REPO / "dataset" / "test_tte_curve"
ZIP  = REPO / "dataset" / "test_tte_curve_upload.zip"
HF_REPO = "EviatarO/test-tte-curve-frames"


def build_zip() -> None:
    dirs = sorted(SRC.rglob("*/"))
    frame_dirs = [d for d in dirs if d.is_dir() and any(d.glob("frame_*.jpg"))]
    print(f"Frame dirs to zip: {len(frame_dirs)}  (expect 1420)")

    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_STORED) as zf:
        for i, fdir in enumerate(frame_dirs, 1):
            for jpg in sorted(fdir.glob("frame_*.jpg")):
                arcname = jpg.relative_to(SRC.parent)  # dataset/test_tte_curve/private/...
                zf.write(jpg, arcname)
            if i % 100 == 0:
                print(f"  Zipped {i}/{len(frame_dirs)} dirs...")

    size_gb = ZIP.stat().st_size / 1e9
    print(f"Zip done: {ZIP}  ({size_gb:.2f} GB)")


def upload_zip() -> None:
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(HF_REPO, repo_type="dataset", private=True, exist_ok=True)
    print(f"Uploading {ZIP.name} to {HF_REPO}...")
    api.upload_file(
        path_or_fileobj=str(ZIP),
        path_in_repo="test_tte_curve_upload.zip",
        repo_id=HF_REPO,
        repo_type="dataset",
    )
    print(f"Uploaded to HF: {HF_REPO}/test_tte_curve_upload.zip")


def verify_counts() -> bool:
    priv = list((SRC / "private").iterdir()) if (SRC / "private").exists() else []
    pub  = list((SRC / "public").iterdir())  if (SRC / "public").exists()  else []
    print(f"private dirs: {len(priv)} (expect 710)")
    print(f"public  dirs: {len(pub)}  (expect 710)")
    ok = len(priv) == 710 and len(pub) == 710
    if not ok:
        print("ERROR: counts don't match — extraction incomplete.")
    return ok


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-zip", action="store_true")
    args = parser.parse_args()

    if not verify_counts():
        return

    if not args.skip_zip:
        build_zip()
    else:
        print(f"Skipping zip — using existing: {ZIP}")
        if not ZIP.exists():
            print("ERROR: --skip-zip specified but zip not found.")
            return

    upload_zip()
    print("Done.")


if __name__ == "__main__":
    main()
