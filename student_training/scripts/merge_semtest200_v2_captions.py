"""
merge_semtest200_v2_captions.py
=================================
Extend the v1 Caption_semtest200_{V10,V12}.jsonl (200 rows, covering the original
pool) to the full v2 300-clip pool, in the SAME schema:
    video_id, frames_dir, horizon_label, requested_time_to_event, gt_verdict,
    caption, source ('reused'|'new'), teacher_model

For the 100 new easy_TN/easy_TP clips (select_semtest200_easy.py), each is either:
  - already present in the corresponding 1,761-row corpus (28/100 for both V10 and
    V12, checked 2026-08-29) -> source='reused', caption copied verbatim, teacher_model
    ='gemini-3.6-flash' (matches the v1 file's own convention for 1761-reused rows -
    that corpus predates the 3.7-flash switch).
  - not present (72/100, the SAME 72 clips for both corpora) -> source='new', caption
    aliased from the freshly-generated caption_neutral field (raw_v10_easy72.jsonl /
    raw_v12_easy72.jsonl, generated this session via --model google/gemini-3.7-flash),
    teacher_model='gemini-3.7-flash'.

v1's 200 rows are copied through completely unchanged - this script only ADDS rows,
never edits the original 200's captions.

Usage:
  python merge_semtest200_v2_captions.py \
      --template V10 \
      --v1-captions ../../outputs/semtest200/Caption_semtest200_V10.jsonl \
      --corpus-1761 ../../outputs/semantic_captions/Caption_Train4500_Mixed_1761.jsonl \
      --fresh ../../outputs/semtest200_v2/raw_v10_easy72.jsonl \
      --easy-selection ../../outputs/semtest200_v2/selection_easy100.jsonl \
      --out ../../outputs/semtest200_v2/Caption_semtest200_V10.jsonl
"""
import argparse
import json
from pathlib import Path


def load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v1-captions", required=True)
    ap.add_argument("--corpus-1761", required=True,
                     help="the 1,761-row corpus to check for reusable captions "
                          "(caption field may be 'caption' or 'caption_neutral')")
    ap.add_argument("--fresh", required=True,
                     help="freshly-generated raw output for the clips NOT in "
                          "--corpus-1761 (caption field: caption_neutral)")
    ap.add_argument("--easy-selection", required=True,
                     help="selection_easy100.jsonl - source of gt_verdict/horizon_label "
                          "for the new rows")
    ap.add_argument("--reused-teacher-model", default="gemini-3.6-flash")
    ap.add_argument("--fresh-teacher-model", default="gemini-3.7-flash")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    v1_rows = load(args.v1_captions)
    corpus = load(args.corpus_1761)
    fresh = load(args.fresh)
    easy_sel = {r["frames_dir"]: r for r in load(args.easy_selection)}

    def cap_text(row):
        return row.get("caption") or row.get("caption_neutral")

    corpus_by_fd = {r["frames_dir"]: r for r in corpus}
    fresh_by_fd = {r["frames_dir"]: r for r in fresh}

    out_rows = list(v1_rows)   # v1's 200 rows, byte-for-byte unchanged
    v1_fds = {r["frames_dir"] for r in v1_rows}

    n_reused = n_new = 0
    for fd, sel in easy_sel.items():
        if fd in v1_fds:
            continue   # already covered (should not happen - easy100 is pool-disjoint)
        if fd in corpus_by_fd:
            cap = cap_text(corpus_by_fd[fd])
            teacher = args.reused_teacher_model
            source = "reused"
            n_reused += 1
        elif fd in fresh_by_fd:
            cap = cap_text(fresh_by_fd[fd])
            teacher = args.fresh_teacher_model
            source = "new"
            n_new += 1
        else:
            raise SystemExit(f"{fd} (easy tier) found in neither --corpus-1761 nor "
                              f"--fresh - nothing to merge")
        out_rows.append({
            "video_id": sel["video_id"], "frames_dir": fd,
            "horizon_label": sel["horizon_label"],
            "requested_time_to_event": None,
            "gt_verdict": sel["gt_verdict"], "caption": cap,
            "source": source, "teacher_model": teacher,
        })

    print(f"[merge] v1 rows carried through: {len(v1_rows)}")
    print(f"[merge] easy clips reused from 1761 corpus: {n_reused}")
    print(f"[merge] easy clips freshly generated:       {n_new}")
    print(f"[merge] total: {len(out_rows)}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r) + "\n")
    print(f"[wrote] {out_path}")


if __name__ == "__main__":
    main()
