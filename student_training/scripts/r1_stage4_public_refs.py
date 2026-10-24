"""
r1_stage4_public_refs.py - PC side of stage 4 (plan 2026-10-22 rev 3): the teacher's pass-1 references for the PUBLIC test windows, made while the pods train.

Steps (each skipped when its output exists; rerun to resume):
  1  download pod 2's boxes inputs (HF dataset teacher-test-2026-10/public/: tracks.tar.gz, attn.tar.gz, windows.jsonl) -> outputs/teacher_test_public_2026-10/
  2  boxed frames: r1_attention_boxes.py --stage boxes on the local public frames (dataset/test_public/<id>_hires; CPU, drawing only)
  3  pass 1: r1_teacher_test.py e3 --split public (Gemini 3.8 Flash, OpenRouter Flex, the E3 prompt of the training texts, parallel calls)
  4  the outcome sentence: r1_add_prediction.py --run E3 (text only, Flex)
  5  references: r1_test_refs.py manifest --split public (only the windows whose pass-1 verdict = the label)
  6  upload the reference manifest to HF checkpoints/stage4_2026-10-22/refs/ (one commit, size-checked): the pods fetch it before their test pass

    python r1_stage4_public_refs.py [--concurrency 64] [--dry-run]
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "outputs" / "teacher_test_public_2026-10"
FRAMES = ROOT / "dataset" / "test_public"
REFS = ROOT / "dataset" / "manifests" / "r1_nexar_reasoning_test_public_refs.jsonl"
R_TEST, R_CKPT, TOP = "eviatarO-org/teacher-test-2026-10", "eviatarO-org/checkpoints", "stage4_2026-10-22"
N = 667


def run(cmd, env=None, dry=False):
    print("  +", " ".join(map(str, cmd)), flush=True)
    if not dry:
        subprocess.run([str(c) for c in cmd], cwd=HERE, env={**os.environ, "PYTHONUTF8": "1", **(env or {})}, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concurrency", type=int, default=64)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    py = sys.executable
    env = {"R1_PILOT_OUT": str(OUT), "R1_TRAIN_DIR": str(FRAMES)}

    if not (OUT / "attn").exists() or len(list((OUT / "attn").glob("*.json"))) < N:          # 1
        print("[1] download the public boxes inputs from HF", flush=True)
        if not a.dry_run:
            from huggingface_hub import hf_hub_download
            dl = OUT / "_hf"
            for f in ("tracks.tar.gz", "attn.tar.gz", "windows.jsonl"):
                p = Path(hf_hub_download(R_TEST, f"public/{f}", repo_type="dataset", local_dir=str(dl)))
                if f.endswith(".tar.gz"):
                    with tarfile.open(p) as t:
                        members = [m for m in t.getmembers() if m.isfile() and m.name.split("/")[0] in ("tracks", "attn") and ".." not in m.name]
                        t.extractall(OUT, members=members)
                else:
                    (OUT / "windows.jsonl").write_bytes(p.read_bytes())
            print(f"    tracks {len(list((OUT / 'tracks').glob('*.json')))}, attention {len(list((OUT / 'attn').glob('*.json')))} of {N}")
    if not (OUT / "boxes.jsonl").exists():                                                   # 2
        print("[2] boxed frames", flush=True)
        run([py, "r1_attention_boxes.py", "--stage", "boxes", "--windows-file", OUT / "windows.jsonl"], env, a.dry_run)
    print("[3] pass 1 (resumes where it stopped)", flush=True)                                 # 3 + 4 resume by themselves
    run([py, "r1_teacher_test.py", "e3", "--split", "public", "--tier", "flex", "--concurrency", a.concurrency] + (["--dry-run"] if a.dry_run else []), env, False)
    print("[4] outcome sentences", flush=True)
    run([py, "r1_add_prediction.py", "--run", "E3", "--model", "google/gemini-3.8-flash", "--tier", "flex", "--concurrency", a.concurrency], env, a.dry_run)
    print("[5] reference manifest", flush=True)
    run([py, "r1_test_refs.py", "manifest", "--split", "public"], None, a.dry_run)
    print("[6] upload the references", flush=True)                                            # 6
    if a.dry_run:
        return
    from huggingface_hub import CommitOperationAdd, HfApi
    api = HfApi()
    readme = (b"# stage4_2026-10-22/refs - teacher pass-1 references of the Nexar PUBLIC test windows\n\n"
              b"Only the windows whose pass-1 verdict agrees with the Nexar label. Used ONLY to score the reasoning text of the final models; never for training or selection.\n")
    dst = f"{TOP}/refs/{REFS.name}"
    api.create_commit(R_CKPT, repo_type="model", commit_message="stage 4: public test references",
                      operations=[CommitOperationAdd(dst, str(REFS)), CommitOperationAdd(f"{TOP}/refs/README.md", readme)])
    got = {x.path: x.size for x in api.get_paths_info(R_CKPT, [dst], repo_type="model")}
    assert got.get(dst) == REFS.stat().st_size, "the reference upload is incomplete"
    print(f"    {dst}: {REFS.stat().st_size} bytes on HF")


if __name__ == "__main__":
    main()
