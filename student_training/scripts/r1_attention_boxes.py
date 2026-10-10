"""
r1_attention_boxes.py - pick, for every pilot window, the road user the frozen crash model (A1-compress256) attends to, box it
(same track ID propagated across the video's three windows) and write boxed frames + review grids for the teacher.
Plan: 2026-10-08_Plan-BoxedTeacher-Pilot-rev3.

Two resumable passes (never both models on the GPU at once):
  tracks : per video, decode the raw mp4 span of its windows; YOLOPv2 on every frame (vehicles) + Grounding-DINO-tiny every
           3rd frame with the prompt "person. bicycle. motorcycle." (vulnerable road users); class-agnostic NMS merge;
           BoT-SORT + fragment stitching + ego-hood filter (all reused from aa1_tracks.py) -> tracks/<vid>.json
  attn   : per window, A1-compress256 forward; the crash head's attention table (1,2560,2560) -> attention received per real
           token r (8 tubelets x 16 x 16); per track: attention mass inside its box cells (aa4_token_labels.box_to_cells),
           tubelets weighted toward the end (w_t ~ t+1) -> attn/<frames_dir>.json
  boxes  : target = top-attention track in the video's LAST window (TTE 0.5 / fake event -0.5 s) -- the same rule for crash and
           normal videos; the same track ID is boxed in the earlier windows when it is visible in >= 4 of their 16 frames, otherwise
           the window's own top track is used and flagged. Writes boxed/<frames_dir>/frame_*.jpg (one red box, no text, native
           resolution), boxes.jsonl, review/<frames_dir>.jpg (4x4 grid).

    python r1_attention_boxes.py --stage tracks|attn|boxes|all [--limit-videos N]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "models"))
import aa1_detect_track_rank as dtr  # noqa: E402
from aa1_scene import YOLOPv2  # noqa: E402
from aa1_tracks import extend_edge_tracks, is_ego_hood, stitch_fragments, track_from_yolop  # noqa: E402
from aa1_yolop_cache import iou_xyxy  # noqa: E402
from aa4_token_labels import box_to_cells  # noqa: E402

OUT = Path(__import__("os").environ.get("R1_PILOT_OUT") or ROOT / "outputs" / "teacher_pilot_2026-10")   # R1_PILOT_OUT: separate test folders
TRAIN = ROOT / "dataset" / "train"
VRU_PROMPT = "person. bicycle. motorcycle."
CONF = 0.30
GDINO_EVERY = 3
PRE_ROLL_S = 0.5
MIN_VISIBLE = 4                                  # frames (of 16) a propagated track must be boxed in the window
W8 = np.arange(1, 9, dtype=np.float64)
W8 /= W8.sum()


def load_windows():
    return [json.loads(l) for l in open(OUT / "windows.jsonl", encoding="utf-8")]


def nms_merge(boxes, scores, labels, thr=0.5):
    order = np.argsort(-np.asarray(scores))
    keep = []
    for i in order:
        if all(iou_xyxy(boxes[i], boxes[j]) <= thr for j in keep):
            keep.append(int(i))
    return [boxes[i] for i in keep], [scores[i] for i in keep], [labels[i] for i in keep]


@torch.no_grad()
def gdino_vru(det, frame_bgr):
    from PIL import Image
    img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    inp = det.processor(images=img, text=VRU_PROMPT, return_tensors="pt").to(det.device)
    out = det.model(**inp)
    res = det.processor.post_process_grounded_object_detection(out, inp.input_ids, threshold=dtr.BOX_THRESHOLD,
                                                               text_threshold=dtr.TEXT_THRESHOLD, target_sizes=[img.size[::-1]])[0]
    labels = res.get("text_labels", res.get("labels"))
    boxes = res["boxes"].cpu().numpy().astype(np.float32)
    scores = res["scores"].cpu().numpy().astype(np.float32)
    keep = [i for i in range(len(boxes)) if boxes[i, 2] - boxes[i, 0] >= dtr.MIN_BOX_SIDE_PX and boxes[i, 3] - boxes[i, 1] >= dtr.MIN_BOX_SIDE_PX]
    return [boxes[i] for i in keep], [float(scores[i]) for i in keep], [str(labels[i]).lower() for i in keep]


# --------------------------------------------------------------------------- pass 1
def stage_tracks(wins, limit):
    (OUT / "tracks").mkdir(parents=True, exist_ok=True)
    vids = sorted({w["video_id"] for w in wins})[: limit or None]
    todo = [v for v in vids if not (OUT / "tracks" / f"{v}.json").exists()]
    print(f"[tracks] {len(vids)} videos, {len(todo)} to do", flush=True)
    if not todo:
        return
    yp = YOLOPv2()
    gd = dtr.Detector()
    for n, vid in enumerate(todo):
        t0 = time.time()
        ends = [w["t_end_s"] for w in wins if w["video_id"] == vid]
        t_start, t_end = max(0.0, min(ends) - 2.0 - PRE_ROLL_S), max(ends)
        frames, ts, fps = dtr.decode_frames(vid, t_start, t_end)
        cache = {"fps": fps, "frames": []}
        gdet = {}
        for i, fr in enumerate(frames):
            o = yp.infer(fr, conf_thres=CONF)
            boxes = [b for b in o["boxes"]]
            scores = [float(s) for s in o["scores"]]
            labels = ["vehicle"] * len(boxes)
            if i % GDINO_EVERY == 0:
                gb, gs, gl = gdino_vru(gd, fr)
                gdet[round(ts[i], 4)] = list(zip(gb, gl))
                boxes += gb
                scores += gs
                labels += gl
            if len(boxes) > 1:
                boxes, scores, labels = nms_merge(boxes, scores, labels)
            cache["frames"].append({"t": ts[i], "boxes": [list(map(float, b)) for b in boxes], "scores": scores, "labels": labels})
        raw = track_from_yolop(cache, frames)
        stitched, _ = stitch_fragments(raw, frames, ts, fps)
        hood = {tid for tid, pts in stitched.items() if is_ego_hood(pts)}
        live = {tid: pts for tid, pts in stitched.items() if tid not in hood}
        extend_edge_tracks(live, cache)
        rec = {"video_id": vid, "fps": fps, "span": [t_start, t_end], "n_frames": len(frames), "tracks": {}}
        for tid, pts in live.items():
            votes = {}
            for t, box, _ in pts:
                for gb, gl in gdet.get(round(t, 4), []):
                    if iou_xyxy(box, gb) > 0.5:
                        votes[gl] = votes.get(gl, 0) + 1
            rec["tracks"][str(tid)] = {"class_hint": max(votes, key=votes.get) if votes else "vehicle",
                                       "points": [[round(t, 4), [float(x) for x in box], c] for t, box, c in pts]}
        (OUT / "tracks" / f"{vid}.json").write_text(json.dumps(rec), encoding="utf-8")
        print(f"[tracks {n + 1}/{len(todo)}] {vid}: {len(frames)} frames, {len(live)} tracks ({len(hood)} hood dropped), "
              f"{time.time() - t0:.0f}s", flush=True)


# --------------------------------------------------------------------------- pass 2
def attach_attention_hook(badas, store):
    mods = [(n, m) for n, m in badas.nn_model.named_modules() if n.endswith("temporal_processor.attention")]
    assert mods, "temporal_processor.attention not found"
    mods[0][1].register_forward_hook(lambda m, a, out: store.__setitem__("attn", out[1].detach()))


def load_tracks(vid):
    rec = json.loads((OUT / "tracks" / f"{vid}.json").read_text(encoding="utf-8"))
    tr = {}
    for tid, d in rec["tracks"].items():
        tr[tid] = {"class_hint": d["class_hint"], "pts": {round(p[0], 4): p[1] for p in d["points"]}}
    return rec["fps"], tr


def box_at(track, frame_index, fps):
    t = round(frame_index / fps, 4)
    return track["pts"].get(t)


def score_tracks(r, tracks, idx, fps):
    """r: (8,16,16) attention received per real token; idx: the window's 16 raw frame indices."""
    out, inside = {}, np.zeros(8)
    union_mass = np.zeros(8)
    for t in range(8):
        union = set()
        for tid, tr in tracks.items():
            cells = set()
            for k in (2 * t, 2 * t + 1):
                b = box_at(tr, idx[k], fps)
                if b is not None:
                    cells |= box_to_cells(b)
            if cells:
                mass = float(sum(r[t][rr, cc] for rr, cc in cells))
                out.setdefault(tid, np.zeros(8))[t] = mass
                union |= cells
        union_mass[t] = float(sum(r[t][rr, cc] for rr, cc in union)) if union else 0.0
    scores = {tid: float((W8 * m).sum()) for tid, m in out.items()}
    return scores, float((W8 * union_mass).sum()), float((W8 * r.reshape(8, -1).sum(1)).sum())


def stage_attn(wins, limit):
    (OUT / "attn").mkdir(parents=True, exist_ok=True)
    vids = set(sorted({w["video_id"] for w in wins})[: limit or None])
    todo = [w for w in wins if w["video_id"] in vids and not (OUT / "attn" / f"{w['frames_dir']}.json").exists()]
    print(f"[attn] {len(todo)} windows to do", flush=True)
    if not todo:
        return
    from r1_cache_features import load_model, preprocess_frames
    badas = load_model()
    store = {}
    attach_attention_hook(badas, store)
    cache = {}
    for n, w in enumerate(todo):
        t0 = time.time()
        paths = sorted((TRAIN / w["frames_dir"]).glob("frame_*.jpg"))
        assert len(paths) == 16, w["frames_dir"]
        frames_bgr = [cv2.imread(str(p)) for p in paths]
        rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames_bgr]
        with torch.no_grad():
            logits, _ = badas.forward_clip(preprocess_frames(badas.vjepa, rgb).to(badas.device))
        p_coll = float(torch.softmax(logits.float(), 1)[0, 1])
        attn = store["attn"][0].float()                                    # (2560, 2560)
        r = attn.mean(0)[:2048].reshape(8, 16, 16).cpu().numpy()
        r_future = float(attn.mean(0)[2048:].sum())
        if w["video_id"] not in cache:
            cache[w["video_id"]] = load_tracks(w["video_id"])
        fps, tracks = cache[w["video_id"]]
        # alignment check: the window's last JPEG equals the raw decoded frame at the planned index (up to JPEG noise)
        cap = cv2.VideoCapture(str(dtr.RAW_VIDEO_ROOT / f"{w['video_id']}.mp4"))
        cap.set(cv2.CAP_PROP_POS_FRAMES, w["frame_idx"][-1])
        ok, raw = cap.read()
        cap.release()
        diff = float(np.abs(raw.astype(np.float32) - frames_bgr[-1].astype(np.float32)).mean()) if ok else -1.0
        scores, inside, total = score_tracks(r, tracks, w["frame_idx"], fps)
        rec = {"frames_dir": w["frames_dir"], "video_id": w["video_id"], "a1_p_collision": round(p_coll, 4),
               "scores": {k: round(v, 5) for k, v in scores.items()}, "attn_inside_boxes": round(inside, 4),
               "attn_total_real": round(total, 4), "attn_future_tokens": round(r_future, 4), "align_diff": round(diff, 2)}
        (OUT / "attn" / f"{w['frames_dir']}.json").write_text(json.dumps(rec), encoding="utf-8")
        if (n + 1) % 10 == 0 or n == 0:
            print(f"[attn {n + 1}/{len(todo)}] {w['frames_dir']} P={p_coll:.2f} tracks={len(scores)} align_diff={diff:.1f} "
                  f"{time.time() - t0:.1f}s/window", flush=True)


# --------------------------------------------------------------------------- pass 3
def stage_boxes(wins, limit):
    (OUT / "boxed").mkdir(exist_ok=True)
    (OUT / "review").mkdir(exist_ok=True)
    vids = sorted({w["video_id"] for w in wins})[: limit or None]
    rows = []
    for vid in vids:
        vw = sorted([w for w in wins if w["video_id"] == vid], key=lambda w: w["horizon"])        # 0.5, 1.0, 1.5
        fps, tracks = load_tracks(vid)
        attn = {w["frames_dir"]: json.loads((OUT / "attn" / f"{w['frames_dir']}.json").read_text(encoding="utf-8")) for w in vw}
        last = attn[vw[0]["frames_dir"]]
        target = max(last["scores"], key=last["scores"].get) if last["scores"] else None
        for w in vw:
            a = attn[w["frames_dir"]]
            idx = w["frame_idx"]
            own = max(a["scores"], key=a["scores"].get) if a["scores"] else None
            vis = lambda tid: sum(box_at(tracks[tid], i, fps) is not None for i in idx) if tid in tracks else 0   # noqa: E731
            chosen, flag = target, "propagated" if w is not vw[0] else "attention_top"
            if target is None or vis(target) < MIN_VISIBLE:
                chosen, flag = own, ("own_top_used" if own is not None else "no_object")
            total = sum(a["scores"].values()) or 1.0
            boxes = [box_at(tracks[chosen], i, fps) if chosen is not None else None for i in idx]
            d = OUT / "boxed" / w["frames_dir"]
            d.mkdir(parents=True, exist_ok=True)
            grid = []
            for k, p in enumerate(sorted((TRAIN / w["frames_dir"]).glob("frame_*.jpg"))):
                fr = cv2.imread(str(p))
                if boxes[k] is not None:
                    x1, y1, x2, y2 = [int(round(x)) for x in boxes[k]]
                    cv2.rectangle(fr, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.imwrite(str(d / p.name), fr, [cv2.IMWRITE_JPEG_QUALITY, 95])
                grid.append(cv2.resize(fr, (320, 180)))
            rows_g = [np.hstack(grid[i:i + 4]) for i in range(0, 16, 4)]
            head = np.zeros((34, 1280, 3), np.uint8)
            cv2.putText(head, f"{w['frames_dir']}  set {w['set']}  label {w['label']}  {flag}  track {chosen}  share "
                              f"{(a['scores'].get(chosen, 0) / total):.2f}  A1 P {a['a1_p_collision']:.2f}", (8, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.imwrite(str(OUT / "review" / f"{w['frames_dir']}.jpg"), np.vstack([head] + rows_g), [cv2.IMWRITE_JPEG_QUALITY, 85])
            rows.append({"frames_dir": w["frames_dir"], "set": w["set"], "video_id": vid, "label": w["label"], "horizon": w["horizon"],
                         "track": chosen, "flag": flag, "class_hint": tracks[chosen]["class_hint"] if chosen else None,
                         "attention_share_of_tracks": round(a["scores"].get(chosen, 0) / total, 3) if chosen else None,
                         "n_tracks": len(a["scores"]), "frames_boxed": sum(b is not None for b in boxes),
                         "attn_inside_boxes": a["attn_inside_boxes"], "a1_p_collision": a["a1_p_collision"],
                         "align_diff": a["align_diff"], "boxes": [None if b is None else [round(float(x), 1) for x in b] for b in boxes]})
    with open(OUT / "boxes.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    import collections
    print("[boxes]", len(rows), "windows;", dict(collections.Counter(r["flag"] for r in rows)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["tracks", "attn", "boxes", "all"])
    ap.add_argument("--limit-videos", type=int, default=0)
    ap.add_argument("--set", default=None, choices=["A", "B", "R"], help="only the windows of this set (R = rescue test)")
    args = ap.parse_args()
    wins = load_windows()
    if args.set:
        wins = [w for w in wins if w["set"] == args.set]
    if args.stage in ("tracks", "all"):
        stage_tracks(wins, args.limit_videos)
    if args.stage in ("attn", "all"):
        stage_attn(wins, args.limit_videos)
    if args.stage in ("boxes", "all"):
        stage_boxes(wins, args.limit_videos)


if __name__ == "__main__":
    main()
