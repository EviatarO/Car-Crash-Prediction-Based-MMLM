"""
e4_stageC_eval.py
=================
Stage C, step 3: run a trained LoRA checkpoint over a split and emit a per-clip
JSONL in the `evaluate_metrics.py` schema, augmented with the LLM verdict.

Per row:
  video_id, ground_truth, score (= FROZEN vision P(collision)),
  horizon_label, time_before_s,
  collision_verdict (= vision decision YES/NO at threshold),   # vision-only headline
  vision_score, llm_verdict (YES/NO), llm_p_yes (verdict-token prob),
  llm_reasoning, teacher_reasoning, verdict_reasoning (= llm_reasoning, for the
  examples table in evaluate_metrics.py).

The headline AP uses `score` (vision, frozen). LLM-verdict AP/agreement use
`llm_p_yes` and are reported separately. Nothing is fused (that is Stage D).

Usage (val):
  python student_training/scripts/e4_stageC_eval.py --config configs/e4_stageC.yaml \
      --adapter <ckpt_dir> --split val --out <out.jsonl>
Test reuses Stage-A vision scores via --vision_scores.

Env: E4_CACHE_DIR, E4_OUTPUT_DIR, E4_PROJECTOR_CKPT, E4_LLM_MODEL_ID, E4_PROJECTOR_OUT_DIM.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from student_training.data.stageb_bridge_dataset import StageBBridgeDataset

HORIZON_TBS = {"TTE_0.5": 0.5, "TTE_1.0": 1.0, "TTE_1.5": 1.5}


def resolve_cfg(cfg):
    cache_dir = os.environ.get("E4_CACHE_DIR", cfg["data"]["cache_dir"])
    out_dir = os.environ.get("E4_OUTPUT_DIR", cfg["data"]["output_dir"])
    model_id = os.environ.get("E4_LLM_MODEL_ID", cfg["llm"]["model_id"])
    if os.environ.get("E4_PROJECTOR_OUT_DIM"):
        cfg["projector"]["out_dim"] = int(os.environ["E4_PROJECTOR_OUT_DIM"])
    proj_ckpt = os.environ.get("E4_PROJECTOR_CKPT", cfg["projector"].get("ckpt"))
    return cache_dir, out_dir, model_id, proj_ckpt


def parse_json_verdict(text):
    """Pull verdict + reason from a generated JSON-ish string; robust to noise."""
    try:
        obj = json.loads(text[text.index("{"):text.rindex("}") + 1])
        return str(obj.get("verdict", "")).strip().upper(), str(obj.get("reason", "")).strip()
    except Exception:
        m_v = re.search(r'"verdict"\s*:\s*"?(YES|NO)"?', text, re.I)
        m_r = re.search(r'"reason"\s*:\s*"(.+?)"\s*[}\n]', text, re.S)
        v = m_v.group(1).upper() if m_v else ""
        r = m_r.group(1).strip() if m_r else text.strip()
        return v, r


def load_bridge(cfg, model_id, adapter_dir, proj_ckpt, device, amp_dtype):
    import torch
    from peft import PeftModel
    from student_training.models.vjepa_reason import load_llm, build_projector, StageBBridge
    print(f"Loading LLM: {model_id}  + adapter: {adapter_dir}")
    llm, tok = load_llm(model_id, dtype=amp_dtype)
    llm = PeftModel.from_pretrained(llm, str(adapter_dir))
    llm.eval()
    projector = build_projector(cfg)
    projector.load_state_dict(torch.load(proj_ckpt, map_location="cpu"))
    bridge = StageBBridge(llm, projector, freeze_llm=True,
                          match_embed_norm=cfg["projector"].get("match_embed_norm")).to(device).eval()
    return bridge, tok


def run(args, cfg):
    import torch
    cache_dir, out_dir, model_id, proj_ckpt = resolve_cfg(cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp_dtype = torch.bfloat16 if "bfloat16" in cfg.get("torch_dtype", "bfloat16") else torch.float16
    thr = args.threshold

    manifest = args.manifest or os.path.join(cache_dir, f"cache_manifest_{args.split}.jsonl")

    bridge, tok = load_bridge(cfg, model_id, args.adapter, proj_ckpt, device, amp_dtype)
    ds = StageBBridgeDataset(
        manifest, cache_dir, tok,
        num_vis_tokens=cfg["projector"]["num_queries"],
        max_seq_len=cfg["data"]["max_seq_len"], supervise_verdict=True)

    # optional external vision scores (test: reuse Stage-A badas_open_*.jsonl)
    score_override = {}
    if args.vision_scores and Path(args.vision_scores).exists():
        for l in open(args.vision_scores, encoding="utf-8"):
            if l.strip():
                r = json.loads(l)
                score_override[str(r.get("video_id"))] = float(r.get("score"))
        print(f"Loaded {len(score_override)} external vision scores from {args.vision_scores}")

    vp_ids = tok.encode('{"verdict": "', add_special_tokens=False)
    yes_ids = tok.encode("YES", add_special_tokens=False)
    no_ids = tok.encode("NO", add_special_tokens=False)

    def cond_lp(vis, prefix_ids, prefix_vm, cont_ids):
        ids = prefix_ids + cont_ids
        labels = [-100] * len(prefix_ids) + cont_ids
        vm = prefix_vm + [False] * len(cont_ids)
        t_ids = torch.tensor([ids], dtype=torch.long).to(device)
        t_lbl = torch.tensor([labels], dtype=torch.long).to(device)
        t_vm = torch.tensor([vm], dtype=torch.bool).to(device)
        am = torch.ones_like(t_ids)
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=(device == "cuda"), dtype=amp_dtype):
            loss, _ = bridge(vis, t_ids, am, t_lbl, t_vm, ablate="none")
        return -float(loss.item()) * len(cont_ids)   # total logprob of the continuation

    rows = []
    import math
    for i in range(len(ds)):
        item = ds[i]
        meta = item["meta"]
        vis = item["vis_feats"].unsqueeze(0).to(device)
        ids = item["input_ids"].tolist()
        labels = item["labels"].tolist()
        vm = item["vis_mask"].tolist()
        prompt_len = next((j for j, x in enumerate(labels) if x != -100), len(labels))
        prompt_ids = ids[:prompt_len]
        prompt_vm = vm[:prompt_len]

        # generation (prompt only -> full JSON)
        with torch.no_grad():
            out_ids = bridge.generate(
                vis, torch.tensor([prompt_ids]).to(device),
                torch.ones(1, len(prompt_ids), dtype=torch.long).to(device),
                torch.tensor([prompt_vm], dtype=torch.bool).to(device),
                max_new_tokens=cfg["gate"].get("max_new_tokens", 160))
        gen = tok.decode(out_ids[0], skip_special_tokens=True)
        llm_verdict, llm_reason = parse_json_verdict(gen)

        # P(YES): logprob of YES vs NO continuation after '{"verdict": "'
        pre_ids = prompt_ids + vp_ids
        pre_vm = prompt_vm + [False] * len(vp_ids)
        lp_yes = cond_lp(vis, pre_ids, pre_vm, yes_ids)
        lp_no = cond_lp(vis, pre_ids, pre_vm, no_ids)
        m = max(lp_yes, lp_no)
        p_yes = math.exp(lp_yes - m) / (math.exp(lp_yes - m) + math.exp(lp_no - m))

        vid = str(meta["video_id"])
        vscore = score_override.get(vid, meta["score"])
        gt = int(meta["target"])
        h = meta.get("horizon_label", "")
        rows.append({
            "video_id": vid,
            "ground_truth": gt,
            "score": round(float(vscore), 6),
            "vision_score": round(float(vscore), 6),
            "horizon_label": h,
            "time_before_s": HORIZON_TBS.get(h),
            "collision_verdict": "YES" if float(vscore) >= thr else "NO",   # vision decision
            "llm_verdict": llm_verdict or ("YES" if p_yes >= 0.5 else "NO"),
            "llm_p_yes": round(float(p_yes), 6),
            "llm_reasoning": llm_reason,
            "verdict_reasoning": llm_reason,
            "teacher_reasoning": meta.get("reason", ""),
        })
        if (i + 1) % 10 == 0 or (i + 1) == len(ds):
            print(f"  {i+1}/{len(ds)}  last vid={vid} vis={vscore:.3f} "
                  f"llm={rows[-1]['llm_verdict']} p_yes={p_yes:.3f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nWrote {len(rows)} rows -> {out_path}")

    # quick alignment summary
    n = len(rows)
    agree_vision = sum((r["llm_verdict"] == r["collision_verdict"]) for r in rows)
    agree_gt = sum((r["llm_verdict"] == ("YES" if r["ground_truth"] else "NO")) for r in rows)
    print(f"  LLM-vs-vision agreement: {agree_vision}/{n} ({100*agree_vision/n:.1f}%)")
    print(f"  LLM-vs-GT     agreement: {agree_gt}/{n} ({100*agree_gt/n:.1f}%)")


def dry_run(args, cfg):
    cache_dir = resolve_cfg(cfg)[0]
    manifest = args.manifest or os.path.join(cache_dir, f"cache_manifest_{args.split}.jsonl")
    print("\n=== DRY RUN (Stage C eval) — no model ===")
    print(f"  manifest: {manifest}  exists={Path(manifest).exists()}")
    print(f"  adapter : {args.adapter}  exists={Path(str(args.adapter)).exists() if args.adapter else False}")
    if Path(manifest).exists():
        rows = [json.loads(l) for l in open(manifest, encoding="utf-8") if l.strip()]
        print(f"  {len(rows)} rows; sample keys: {sorted(rows[0].keys()) if rows else '—'}")
    print("=== dry run OK ===\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--adapter", help="trained LoRA checkpoint dir (ckpt_epNN)")
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--manifest", help="override cache manifest path")
    ap.add_argument("--vision_scores", help="external vision-score JSONL (test: Stage-A badas_open_*.jsonl)")
    ap.add_argument("--out", help="output JSONL path")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()
    with open(args.config if os.path.isabs(args.config) else PROJECT_ROOT / args.config) as f:
        cfg = yaml.safe_load(f)
    if args.dry_run:
        dry_run(args, cfg)
        return
    if not args.adapter or not args.out:
        ap.error("--adapter and --out are required for a real run")
    run(args, cfg)


if __name__ == "__main__":
    main()
