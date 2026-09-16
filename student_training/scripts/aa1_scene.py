"""
aa1_scene.py
============
YOLOPv2 wrapper for Stage AA.1 v2b (child plan 2026-09-15-AA1-v2b-YOLOPv2-lanes-tracking):
vehicle detection + drivable-area + lane-line segmentation in one pass, replacing G-DINO as the
default detector and the fitted/fallback horizon as the source of lane geometry.

Model: official TorchScript export from CAIC-AD/YOLOPv2 (MIT), downloaded from the GitHub
release asset (not a third-party re-upload) to
`third_party/yolopv2_ref/weights/yolopv2.pt`. Pre/post-processing reuses the authors' own
`third_party/yolopv2_ref/utils.py` (vendored verbatim, MIT) for the parts that are easy to get
subtly wrong (NMS, the split_for_trace_model anchor decode, the seg/lane mask crop+upsample) -
these are not reimplemented here.

Confirmed empirically (2026-09-15, this file's __main__ block):
- Detection head is 255 = 3 anchors x 85 (4 box + 1 obj + 80 COCO-style class slots), but the
  model was trained on BDD100K with a single "vehicle" class - see CLASS_ID_VEHICLE below and
  the class-histogram check in main().
- seg/lane outputs are [1,2,384,640] and [1,1,384,640] at the network's internal padded
  resolution; driving_area_mask()/lane_line_mask() crop+upsample them back to the exact 720x1280
  frame size IF the frame was pre-resized to 1280x720 before letterboxing (matches our pipeline's
  FRAME_W/FRAME_H exactly, so no extra rescale is needed for the masks - only detection boxes go
  through scale_coords).
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from third_party.yolopv2_ref.utils import (  # noqa: E402
    letterbox, split_for_trace_model, non_max_suppression, scale_coords,
    driving_area_mask, lane_line_mask,
)

WEIGHTS_PATH = REPO / "third_party" / "yolopv2_ref" / "weights" / "yolopv2.pt"
IMG_SIZE = 640          # inference size, per the authors' demo.py default
STRIDE = 32
CONF_THRES = 0.30       # matches aa1_detect_track_rank.py's BOX_THRESHOLD for a fair comparison
IOU_THRES = 0.45        # authors' demo.py default
CLASS_ID_VEHICLE = 3    # confirmed empirically 2026-09-15 on 00687 (93 frames): only class id 3
                        # ever appears - single-class "vehicle" model, matches the paper


class YOLOPv2:
    def __init__(self, weights_path: Path = WEIGHTS_PATH, device: str = "cuda"):
        self.device = torch.device(device)
        self.model = torch.jit.load(str(weights_path), map_location=self.device)
        self.model.to(self.device).eval()
        self.half = self.device.type == "cuda"
        if self.half:
            self.model.half()

    @torch.no_grad()
    def infer(self, frame_bgr: np.ndarray, conf_thres: float = CONF_THRES,
              iou_thres: float = IOU_THRES):
        """frame_bgr must already be 1280x720 (our pipeline's FRAME_W/FRAME_H) - the seg/lane
        mask crop+upsample math in the vendored utils is derived for exactly that size.
        Returns dict(boxes=(N,4) xyxy float32 in ORIGINAL 1280x720 pixels, scores=(N,) float32,
        classes=(N,) int64, drivable=(720,1280) bool, lane=(720,1280) bool)."""
        h0, w0 = frame_bgr.shape[:2]
        assert (w0, h0) == (1280, 720), f"expected 1280x720, got {w0}x{h0}"
        img_letterboxed = letterbox(frame_bgr, IMG_SIZE, stride=STRIDE)[0]
        img = img_letterboxed[:, :, ::-1].transpose(2, 0, 1)  # BGR->RGB, HWC->CHW
        img = np.ascontiguousarray(img)
        t = torch.from_numpy(img).to(self.device)
        t = t.half() if self.half else t.float()
        t /= 255.0
        t = t.unsqueeze(0)

        [pred, anchor_grid], seg, ll = self.model(t)
        pred = split_for_trace_model(pred, anchor_grid)
        pred = non_max_suppression(pred, conf_thres, iou_thres)[0]  # single image in batch

        da_mask = driving_area_mask(seg).astype(bool)
        ll_mask = lane_line_mask(ll).astype(bool)

        if len(pred):
            boxes = pred[:, :4].clone()
            boxes = scale_coords(t.shape[2:], boxes, frame_bgr.shape).round()
            boxes = boxes.cpu().numpy().astype(np.float32)
            scores = pred[:, 4].cpu().numpy().astype(np.float32)
            classes = pred[:, 5].cpu().numpy().astype(np.int64)
        else:
            boxes = np.zeros((0, 4), np.float32)
            scores = np.zeros((0,), np.float32)
            classes = np.zeros((0,), np.int64)

        return dict(boxes=boxes, scores=scores, classes=classes, drivable=da_mask, lane=ll_mask)


if __name__ == "__main__":
    # Empirical check (2026-09-15): confirm class histogram, mask alignment, and that partner
    # vehicles are found on a known clip, before wiring this into the ranking pipeline.
    import time
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from aa1_detect_track_rank import decode_frames, decode_span  # noqa: E402

    yp = YOLOPv2()
    t_start, t_end, _ = decode_span("00687")
    frames, timestamps, fps = decode_frames("00687", t_start, t_end)
    print(f"decoded {len(frames)} frames")

    all_classes = []
    t0 = time.time()
    for f in frames:
        out = yp.infer(f)
        all_classes.append(out["classes"])
    dt = time.time() - t0
    print(f"YOLOPv2: {dt:.1f}s total, {dt/len(frames)*1000:.1f}ms/frame")
    uniq = np.unique(np.concatenate(all_classes)) if all_classes else np.array([])
    print(f"unique class ids seen across {len(frames)} frames: {uniq}")

    # Render one frame with boxes + drivable + lane overlay to inspect visually.
    mid = len(frames) // 2
    out = yp.infer(frames[mid])
    vis = frames[mid].copy()
    vis[out["drivable"]] = (vis[out["drivable"]] * 0.5 + np.array([0, 255, 0]) * 0.5).astype(np.uint8)
    vis[out["lane"]] = (vis[out["lane"]] * 0.3 + np.array([0, 0, 255]) * 0.7).astype(np.uint8)
    for box, score in zip(out["boxes"], out["scores"]):
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(vis, f"{score:.2f}", (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    out_path = REPO / "outputs" / "aa1_v2_18clips"
    out_path.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path / "00687_yolopv2_check.jpg"), vis)
    print(f"n_boxes={len(out['boxes'])}  drivable_px={out['drivable'].sum()}  lane_px={out['lane'].sum()}")
    print(f"wrote {out_path / '00687_yolopv2_check.jpg'}")
