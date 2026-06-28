"""
stageb_bridge_dataset.py
========================
Stage-B dataset: serves CACHED V-JEPA2 patch features (not raw frames) paired
with a Qwen3-formatted prompt and the teacher reasoning target.

Each item:
  vis_feats       (P, in_dim)   cached patch grid (fp16 on disk -> float)
  input_ids       (L,)          [system + user header + VIS block + question
                                 + assistant header + verdict-prefix + reason]
  attention_mask  (L,)
  labels          (L,)          -100 everywhere EXCEPT the reason span (plan §3.1)
  vis_mask        (L,)          True at the `num_vis_tokens` visual positions
  meta            dict          video_id, horizon_label, target, score, split, reason

Supervision is **reason-only**: the verdict tokens are masked so the projector's
gradient comes from modelling the *explanation* given the visual tokens.

The cache manifest (one JSON per line) is produced by e4_stageB_cache_features.py
and carries: {key, video_id, frames_dir, horizon_label, target, score, split,
verdict, reason}. `key` indexes the cached tensor `<cache_dir>/<key>.pt`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import torch
from torch.utils.data import Dataset

SYSTEM_PROMPT = "You are a helpful assistant."

STAGEB_QUESTION = (
    "These are 16 sequential dashcam frames (~2 seconds) from the ego vehicle. "
    "Decide whether a collision involving the ego vehicle is about to occur, and "
    "explain the visual evidence. Respond as JSON: "
    '{"verdict": "YES" or "NO", "reason": "<one concise paragraph>"}.'
)


def _parse_target(assistant_target: str):
    """Return (verdict, reason) from the teacher `assistant_target` JSON string."""
    try:
        obj = json.loads(assistant_target)
        return str(obj.get("verdict", "")).strip(), str(obj.get("reason", "")).strip()
    except Exception:
        return "", str(assistant_target)


class StageBBridgeDataset(Dataset):
    def __init__(self, manifest_path: str, cache_dir: str, tokenizer,
                 num_vis_tokens: int = 64, max_seq_len: int = 1024,
                 split: str = None, supervise_verdict: bool = False):
        self.cache_dir = Path(cache_dir)
        self.tok = tokenizer
        self.num_vis_tokens = num_vis_tokens
        self.max_seq_len = max_seq_len
        # Stage B (False): mask the verdict, supervise the reason span only.
        # Stage C (True):  supervise the FULL assistant JSON (verdict + reason).
        self.supervise_verdict = supervise_verdict
        # Placeholder id for visual positions; its embedding is overwritten, so
        # the exact id is irrelevant — only the position matters.
        self.vis_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

        rows = []
        with open(manifest_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if split is None or r.get("split") == split:
                    rows.append(r)
        self.records = rows
        print(f"StageBBridgeDataset: {len(self.records)} records from {manifest_path}"
              f"{f' (split={split})' if split else ''}")

    def __len__(self):
        return len(self.records)

    def _enc(self, text: str) -> List[int]:
        return self.tok.encode(text, add_special_tokens=False)

    def __getitem__(self, idx: int) -> dict:
        r = self.records[idx]
        feats = torch.load(self.cache_dir / f"{r['key']}.pt").float()      # (P, in_dim)

        verdict, reason = _parse_target(r.get("assistant_target") or r.get("reason", ""))
        if not reason and r.get("reason"):
            reason = r["reason"]

        header = self._enc(
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n")
        vis_block = [self.vis_id] * self.num_vis_tokens
        question = self._enc(
            f"\n{STAGEB_QUESTION}<|im_end|>\n<|im_start|>assistant\n")

        # Visual block sits right after the header; record its span explicitly so
        # vis positions are tracked structurally (never inferred from token id).
        vstart = len(header)
        vend = vstart + self.num_vis_tokens

        if self.supervise_verdict:
            # Stage C: prompt ends at the assistant header; supervise the FULL JSON
            # (verdict + reason). Build the target in SEGMENTS so we know the exact
            # index of the verdict token (for the score-consistency anchor).
            seg_vp = self._enc('{"verdict": "')
            seg_verdict = self._enc(verdict)                         # "YES" / "NO"
            seg_rest = self._enc(f'", "reason": "{reason}"}}<|im_end|>\n')
            target = seg_vp + seg_verdict + seg_rest
            prefix = header + vis_block + question
            verdict_pos = len(prefix) + len(seg_vp)                  # first verdict token
        else:
            # Stage B: verdict prefix is MASKED; only the reason body is supervised.
            verdict_prefix = self._enc(f'{{"verdict": "{verdict}", "reason": "')
            target = self._enc(f'{reason}"}}<|im_end|>\n')
            prefix = header + vis_block + question + verdict_prefix
            verdict_pos = -1
        all_ids = prefix + target
        prefix_len = len(prefix)

        # Truncate from the FRONT only if it does not reach the visual block, so
        # the visual span and the reason span are always preserved intact.
        if len(all_ids) > self.max_seq_len:
            overflow = len(all_ids) - self.max_seq_len
            if overflow <= max(0, vstart):          # only header text is trimmed
                all_ids = all_ids[overflow:]
                vstart -= overflow
                vend -= overflow
                prefix_len -= overflow
            else:                                   # rare: drop oldest tokens, re-mark by structure
                all_ids = all_ids[overflow:]
                vstart = max(0, vstart - overflow)
                vend = vstart + self.num_vis_tokens
                prefix_len = max(0, prefix_len - overflow)
            if verdict_pos >= 0:
                verdict_pos = max(-1, verdict_pos - overflow)

        labels = [-100] * prefix_len + all_ids[prefix_len:]
        vis_mask = [False] * len(all_ids)
        for j in range(max(0, vstart), min(vend, len(all_ids))):
            vis_mask[j] = True

        score = r.get("score")
        return {
            "vis_feats":      feats,
            "input_ids":      torch.tensor(all_ids, dtype=torch.long),
            "attention_mask": torch.ones(len(all_ids), dtype=torch.long),
            "labels":         torch.tensor(labels, dtype=torch.long),
            "vis_mask":       torch.tensor(vis_mask, dtype=torch.bool),
            "verdict_pos":    torch.tensor(int(verdict_pos), dtype=torch.long),
            "vision_score":   torch.tensor(float(score) if score is not None else 0.0),
            "meta": {
                "video_id":      r.get("video_id"),
                "horizon_label": r.get("horizon_label"),
                "target":        int(r.get("target", 0)),
                "score":         r.get("score"),
                "split":         r.get("split"),
                "reason":        reason,
                "verdict":       verdict,
            },
        }

    @staticmethod
    def collate_fn(batch: List[dict]) -> dict:
        max_len = max(s["input_ids"].shape[0] for s in batch)
        pad = lambda t, v: torch.nn.functional.pad(t, (0, max_len - t.shape[0]), value=v)
        ids   = torch.stack([pad(s["input_ids"], 0)        for s in batch])
        mask  = torch.stack([pad(s["attention_mask"], 0)   for s in batch])
        lbl   = torch.stack([pad(s["labels"], -100)        for s in batch])
        vmask = torch.stack([pad(s["vis_mask"], False)     for s in batch])
        feats = torch.stack([s["vis_feats"] for s in batch])           # (B, P, in_dim)
        out = {
            "vis_feats": feats, "input_ids": ids, "attention_mask": mask,
            "labels": lbl, "vis_mask": vmask, "meta": [s["meta"] for s in batch],
        }
        if "verdict_pos" in batch[0]:                                  # Stage C score anchor
            out["verdict_pos"] = torch.stack([s["verdict_pos"] for s in batch])
            out["vision_score"] = torch.stack([s["vision_score"] for s in batch])
        return out
