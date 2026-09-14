"""
score_checkpoints_on_test.py
==============================
Score one or more LoRA checkpoints on the 677-clip TEST manifest (Stage 2 of the
A1-failure-recovery run).

WHY A SEPARATE SCRIPT: semsup_train.py test-scores its kept checkpoints only when
--test-manifest is passed AT TRAINING TIME. The a1fail321 runs deliberately omitted it
(the STOP gate was "review loss curves first"), so the checkpoints exist but were never
test-scored. This scores them after the fact without retraining.

SOFTMAX CONVENTION (matters, and is easy to get wrong here): this uses
softmax(logits/--temperature)[0,1], default temperature=1.0 - identical to
semsup_train.py and score_arms_on_pool1761.py at the default. e4_stageA_badas_open_eval.py
uses temperature=2.0 (the published-scorer convention; pass --temperature 2.0 here for a
directly A0-comparable run). For AP/AUC/accuracy@0.5/the confusion matrix the difference is
irrelevant - dividing logits by a constant is a monotone transform and those are all
invariant to it - but it DOES move Brier/ECE (calibration metrics), which are NOT invariant
to temperature. Project review (2026-09-06 §4.5): publishing A0's Brier/ECE (T=2) next to
every other arm's (T=1) INVERTS the calibration ranking (A0's ECE moves 0.1528 -> 0.1920,
i.e. worse than A1, not better) even though AP/AUC/CM are bit-identical across the two
conventions. Never compare Brier/ECE across runs at different --temperature.

--unfreeze-head checkpoints (project review §4.1): if a lora_adapter/'s sibling
epoch_XX/head_state.pt exists, it MUST be passed via --head-states NAME=path or this
script hard-fails - scoring an unfrozen-head checkpoint against the ORIGINAL frozen head
would silently produce numbers for a model that was never actually trained (peft's
save_pretrained() persists only the LoRA delta, never the head).

Loads BADAS ONCE and swaps adapters between checkpoints (~3 min/checkpoint of actual
scoring vs ~2 min of model load), so scoring N checkpoints costs far less than N runs.

Usage (on the pod):
  python3 score_checkpoints_on_test.py --config ../configs/e4_stageA.yaml \
      --test-manifest ../../dataset/manifests/test_manifest_hires.jsonl \
      --test-frames-root /workspace/data/test_HiRes \
      --adapters A1=/workspace/semsup/a1_1761/epoch_04/lora_adapter \
                 v12=/workspace/MMLM_AI/outputs/a1fail321/results/v12/fold_01/epoch_10/lora_adapter \
      --out-dir /workspace/MMLM_AI/outputs/a1fail321/test_scores

  # scoring an --unfreeze-head checkpoint:
  python3 score_checkpoints_on_test.py ... \
      --adapters vision=/workspace/semtest200/vision/epoch_08/lora_adapter \
      --head-states vision=/workspace/semtest200/vision/epoch_08/head_state.pt

  # an A0-comparable (T=2) run, for calibration metrics comparable to the published baseline:
  python3 score_checkpoints_on_test.py ... --temperature 2.0
"""
import argparse
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models"))

from semsup_common import (  # noqa: E402
    TrainableBadasWrapper, resolve_lora_topology, parse_lora_target_modules,
    load_lora_adapter_checked,
)
from metrics_core import metrics_from_arrays  # noqa: E402

# The two module-name substrings that make up the crash head - see
# semsup_common.py's --unfreeze-head comment. Used only to snapshot/restore the
# ORIGINAL frozen head weights between checkpoints scored in the same process, so
# an --unfreeze-head arm never leaks its mutated head into the next arm scored.
HEAD_SUBSTRINGS = ("temporal_processor", "classifier")


def frame_paths_for(record, frames_root, pattern):
    """Absolute paths of the clip's frames. frames_dir falls back to video_id.
    Same logic as e4_stageA_badas_open_eval.py / e4_stageB_cache_features.py."""
    frames_dir = record.get("frames_dir") or record["video_id"]
    return [os.path.join(frames_root, frames_dir, pattern.format(i))
            for i in record["frame_indices"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--test-manifest", required=True)
    ap.add_argument("--test-frames-root", required=True)
    ap.add_argument("--adapters", nargs="+", required=True,
                     help="one or more NAME=/path/to/lora_adapter. Use NAME=NONE to score "
                          "the frozen no-adapter baseline.")
    ap.add_argument("--head-states", nargs="+", default=[],
                     help="one or more NAME=/path/to/head_state.pt, matching a NAME in "
                          "--adapters. REQUIRED for any arm whose lora_adapter/ has a "
                          "sibling head_state.pt on disk (i.e. it was trained with "
                          "--unfreeze-head) - this script hard-fails rather than silently "
                          "score that checkpoint against the original frozen head.")
    ap.add_argument("--temperature", type=float, default=1.0,
                     help="softmax(logits/temperature). Default 1.0 matches "
                          "semsup_train.py/score_arms_on_pool1761.py. Pass 2.0 to match "
                          "e4_stageA_badas_open_eval.py's A0 convention - only needed for "
                          "a directly A0-comparable Brier/ECE; AP/AUC/CM are invariant to "
                          "this (see module docstring).")
    ap.add_argument("--preprocess", default="crop", choices=["crop", "compress256"],
                     help="frame preprocessing - MUST match how the adapter was trained "
                          "('crop' = every historical run; 'compress256' = full frame "
                          "resized to 256x256). Written into each metrics JSON.")
    ap.add_argument("--lora-target-modules", default=None,
                     help="override. Default: read from each adapter's run "
                          "train_metrics.json (legacy query,key,value if absent).")
    ap.add_argument("--lora-r", type=int, default=None, help="override; see --lora-target-modules")
    ap.add_argument("--lora-alpha", type=int, default=None, help="override; see --lora-target-modules")
    ap.add_argument("--lora-dropout", type=float, default=None, help="override; see --lora-target-modules")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--deterministic", action="store_true", default=True,
                     help="torch.use_deterministic_algorithms(True) + "
                          "cudnn.deterministic=True (default ON). Project review "
                          "(2026-09-06 §3.1): the SAME checkpoint scored twice through "
                          "this script on the same 677 clips disagreed on 677/677 of "
                          "them (max|delta|=0.097, 5 clips flipped the 0.5 boundary, "
                          "dAP=0.0009) with nothing pinning inference numerics. Pass "
                          "--no-deterministic to restore the old behavior.")
    ap.add_argument("--no-deterministic", action="store_false", dest="deterministic")
    args = ap.parse_args()

    if args.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)

    head_state_map = {}
    for spec in args.head_states:
        if "=" not in spec:
            raise SystemExit(f"--head-states entry {spec!r} must be NAME=/path")
        hname, hpath = spec.split("=", 1)
        head_state_map[hname] = hpath

    import yaml
    cfg = yaml.safe_load(open(args.config))
    pattern = cfg["data"]["frame_filename_pattern"]
    gt_field = cfg["data"]["gt_field"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    records = [json.loads(l) for l in open(args.test_manifest, encoding="utf-8") if l.strip()]
    print(f"[load] test manifest: {len(records)} records  gt_field={gt_field!r}")
    pos = sum(1 for r in records if int(r[gt_field]) == 1)
    print(f"[load] class balance: {pos} positive / {len(records) - pos} negative")

    # frame_paths precomputed once and reused across every checkpoint (records do not
    # change between checkpoints); prefetch_clips reads key="frame_paths" by default.
    records_wp = [{**r, "frame_paths": frame_paths_for(r, args.test_frames_root, pattern)}
                  for r in records]
    missing = [r for r in records_wp if not all(os.path.exists(p) for p in r["frame_paths"])]
    if missing:
        raise SystemExit(f"{len(missing)} records have missing frames on disk, "
                          f"e.g. {missing[0]['video_id']} -> {missing[0]['frame_paths'][0]}")
    print(f"[verify] all {len(records_wp)} records have every frame present on disk")

    # One wrapper serves every adapter in this invocation, so all adapters must share
    # one topology. Resolve it from each adapter's own train_metrics.json and refuse to
    # mix topologies (score those in separate invocations).
    overrides = {"lora_target_modules": args.lora_target_modules, "lora_r": args.lora_r,
                 "lora_alpha": args.lora_alpha, "lora_dropout": args.lora_dropout}
    topologies = {}
    for spec in args.adapters:
        name, path = spec.split("=", 1) if "=" in spec else (spec, None)
        if path and path.upper() != "NONE":
            topologies[name] = resolve_lora_topology(path, overrides)
    if topologies:
        first_name, (topo, source) = next(iter(topologies.items()))
        for name, (t, s) in topologies.items():
            if t != topo:
                raise SystemExit(f"adapters {first_name} and {name} were trained with different "
                                 f"LoRA topologies ({topo} vs {t}); score them in separate "
                                 f"invocations")
    else:
        topo, source = resolve_lora_topology(None, overrides)
    print(f"[setup] LoRA topology {topo}  <- {source}")
    print(f"[setup] preprocess: {args.preprocess}")
    badas = TrainableBadasWrapper(cfg, lora_target_modules=parse_lora_target_modules(topo["lora_target_modules"]),
                                   lora_r=topo["lora_r"], lora_alpha=topo["lora_alpha"],
                                   lora_dropout=topo["lora_dropout"],
                                   preprocess_mode=args.preprocess)
    badas.nn_model.eval()

    # Snapshot the ORIGINAL frozen head weights before scoring anything, so any arm
    # scored with --head-states can be restored to the frozen head afterward - without
    # this, an --unfreeze-head arm's mutated head would silently leak into the NEXT
    # arm scored in the same process if that arm has no head_state.pt of its own.
    orig_head_sd = {k: v.detach().clone() for k, v in badas.nn_model.state_dict().items()
                     if any(sub in k for sub in HEAD_SUBSTRINGS)}
    print(f"[setup] snapshotted {len(orig_head_sd)} original head-param tensors for restore")

    # SAME bug class for LoRA weights, and it is NOT hypothetical - it silently fired the
    # first time this script scored a real adapter and NAME=NONE in one invocation (AA.0,
    # 2026-09-14): "NONE" only skipped loading a new adapter, it never reset the PREVIOUS
    # one, so the "frozen baseline" silently inherited whatever adapter was scored just
    # before it (A1=... A0=NONE -> A0 scored bit-identical to A1: AP/AUC/every TP-FN-FP-TN
    # matched exactly - impossible for a real frozen-vs-tuned comparison, and it would have
    # gone unnoticed if the two numbers had merely been close instead of identical). LoRA's
    # lora_B is zero-init, so this snapshot IS the true frozen-baseline state; restoring it
    # for every NAME=NONE arm makes "NONE" mean "no adapter", not "whatever loaded last".
    orig_lora_sd = {k: v.detach().clone() for k, v in badas.nn_model.state_dict().items()
                    if "lora_" in k}
    print(f"[setup] snapshotted {len(orig_lora_sd)} original (zero-init) LoRA tensors for restore")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for spec in args.adapters:
        if "=" not in spec:
            raise SystemExit(f"--adapters entry {spec!r} must be NAME=/path (or NAME=NONE)")
        name, path = spec.split("=", 1)

        if path.upper() == "NONE":
            # Restore the zero-init LoRA state - see orig_lora_sd's comment above. Without
            # this, "NONE" scored after a real adapter in the same invocation silently
            # scores that PREVIOUS adapter again, not the frozen baseline.
            if orig_lora_sd:
                badas.nn_model.load_state_dict(orig_lora_sd, strict=False)
            print(f"\n[score] {name}: frozen baseline, no adapter attached "
                  f"({len(orig_lora_sd)} LoRA tensors reset to zero-init)")
            sibling_head = None
        else:
            adapter_path = Path(path)
            sft = (adapter_path / "adapter_model.safetensors") if adapter_path.is_dir() else adapter_path
            if not sft.exists():
                raise SystemExit(f"{name}: adapter not found at {sft}")
            print(f"\n[score] {name}:")
            # Every adapter here shares the same (verified) topology, so this fully
            # overwrites the previous checkpoint's lora_A/lora_B - no residue carries
            # between arms. strict=True: a mismatched adapter raises instead of loading
            # partially.
            load_lora_adapter_checked(badas.nn_model, sft, strict=True)
            # sibling head_state.pt convention (semsup_train.py writes both under the
            # same epoch_XX/ dir): adapter_path is .../epoch_XX/lora_adapter, so the
            # sibling is adapter_path.parent / "head_state.pt".
            candidate = adapter_path.parent / "head_state.pt" if adapter_path.is_dir() \
                else adapter_path.parent.parent / "head_state.pt"
            sibling_head = candidate if candidate.exists() else None

        # --unfreeze-head correctness gate (project review §4.1): hard-fail rather
        # than silently score against the wrong head.
        if sibling_head is not None and name not in head_state_map:
            raise SystemExit(
                f"{name}: found {sibling_head} next to this adapter (this checkpoint "
                f"was trained with --unfreeze-head) but no --head-states {name}=... was "
                f"passed. Scoring it against the ORIGINAL frozen head would silently "
                f"produce numbers for a model that was never actually trained this way. "
                f"Pass --head-states {name}={sibling_head} (or, if you deliberately want "
                f"the frozen-head numbers for this checkpoint, rename/move the sibling "
                f"head_state.pt out of the way first).")
        if name in head_state_map:
            hpath = head_state_map[name]
            if sibling_head is not None and str(sibling_head) != hpath and Path(hpath).resolve() != sibling_head.resolve():
                print(f"  [warn] {name}: --head-states path ({hpath}) differs from the "
                      f"sibling head_state.pt found next to the adapter ({sibling_head}) - "
                      f"using the explicitly-passed path.")
            badas.load_head_state(hpath)
        else:
            # No head_state for this arm - restore the ORIGINAL frozen head in case a
            # prior arm in this same process mutated it.
            badas.nn_model.load_state_dict(orig_head_sd, strict=False)

        n_failed, rows = 0, []
        with torch.no_grad():
            for _, rec, clip, err in badas.prefetch_clips(records_wp,
                                                           num_workers=args.num_workers,
                                                           prefetch=16):
                if err is not None:
                    n_failed += 1
                    print(f"  [warn] skipping {rec.get('video_id')}: {err}")
                    continue
                logits, _ = badas.forward_clip(clip.to(device))
                # See module docstring for the temperature convention (--temperature,
                # default 1.0 = "no /2.0", matching semsup_train.py's own scorer).
                score = float(torch.softmax(logits / args.temperature, dim=1)[0, 1].item())
                rows.append({
                    "arm": name,
                    "video_id": rec["video_id"],
                    "frames_dir": rec.get("frames_dir"),
                    "group": rec.get("group"),
                    "gt_verdict": "YES" if int(rec[gt_field]) == 1 else "NO",
                    "score": score,
                })
        if n_failed:
            print(f"  [warn] {n_failed}/{len(records_wp)} clips failed to score")

        out_path = out_dir / f"{name}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

        # metrics_core.metrics_from_arrays - the SAME function the training pipeline
        # and the results website use, per the project's single-metric-source rule
        # (project review §4.5/§B-5: this scorer previously hand-rolled AP/AUC/acc
        # inline and emitted no CM, no per-TTE, no n_positive/n_negative).
        y = [1 if r["gt_verdict"] == "YES" else 0 for r in rows]
        s = [r["score"] for r in rows]
        g = [r["group"] for r in rows]
        m = metrics_from_arrays(y, s, groups=g, threshold=0.5)
        metrics_path = out_dir / f"{name}.metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump({"arm": name, "temperature": args.temperature,
                       "preprocess": args.preprocess,
                       "lora_topology": None if path.upper() == "NONE" else topo,
                       "test_manifest": args.test_manifest,
                       "head_state": head_state_map.get(name), **m}, f, indent=2)
        print(f"  {name}: n={m['n_total']}  AP={m['ap']}  AUC={m['auc_roc']}  "
              f"acc@0.5={m['accuracy']}  tp={m['tp']} fn={m['fn']} fp={m['fp']} tn={m['tn']}  "
              f"Brier={m['brier']}  ECE={m['ece']}  (temperature={args.temperature})")
        print(f"  [wrote] {out_path}")
        print(f"  [wrote] {metrics_path}")


if __name__ == "__main__":
    main()
