"""
e4_stageB_eval_gate.py
======================
Stage B, step 3: compute the gate metrics (plan §5) and write the deliverables.

  (1) Delta-Perplexity  : trained vs random-init projector vs text-only   [PRIMARY GATE]
  (2) Visual-ablation   : delta-CE for zeroed / shuffled visual tokens     [GROUNDING GATE]
  (3) Discrimination    : hazard-lexicon rate gap (pos - neg) from greedy generations
  (4) Generations dump  : per-val-clip prompt/gen/teacher/GT
  (5) Per-TTE score-path: full metric suite per TTE bucket + overlaid histogram

Outputs land in <output_dir> (default outputs/e4_vjepa_reason/e4_StageB_bridge/).
Run on RunPod after training. Local --dry_run validates manifests only.
"""

import argparse
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

HAZARD_LEXICON = [
    "collision", "collide", "crash", "impact", "hit", "rear-end", "rear end",
    "brake", "braking", "swerve", "cut in", "cuts in", "cutting", "imminent",
    "unavoidable", "lose control", "loss of control", "run red", "pedestrian",
    "cyclist", "stop short", "closing", "gap collaps", "T-bone", "t-bone", "merge",
]


# ── metric helpers (compact; AP/AUC/F1/CM + bootstrap CI) ────────────────────

def metric_suite(scores, targets, n_boot=2000, seed=42):
    from sklearn.metrics import average_precision_score, roc_auc_score
    s = np.asarray(scores, float); y = np.asarray(targets, int)
    out = {"n": int(len(y)), "n_pos": int(y.sum()), "n_neg": int((1 - y).sum())}
    if len(set(y.tolist())) < 2:
        out.update(ap=None, auc=None); return out
    out["ap"] = float(average_precision_score(y, s))
    out["auc"] = float(roc_auc_score(y, s))
    # threshold 0.5 confusion + optimal-F1 threshold
    for thr_name, thr in (("0.5", 0.5), ("optf1", _best_f1_threshold(s, y))):
        pred = (s >= thr).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        out[f"thr_{thr_name}"] = {"threshold": round(float(thr), 4), "tp": tp, "fp": fp,
                                  "fn": fn, "tn": tn, "precision": round(prec, 4),
                                  "recall": round(rec, 4), "f1": round(f1, 4),
                                  "accuracy": round((tp + tn) / len(y), 4)}
    # bootstrap CI on AP
    rng = np.random.default_rng(seed); aps = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(set(y[idx].tolist())) < 2:
            continue
        aps.append(average_precision_score(y[idx], s[idx]))
    if aps:
        out["ap_ci"] = [round(float(np.percentile(aps, 2.5)), 4),
                        round(float(np.percentile(aps, 97.5)), 4)]
    return out


def _best_f1_threshold(s, y):
    from sklearn.metrics import f1_score
    best_t, best_f1 = 0.5, -1
    for t in np.linspace(0.05, 0.95, 19):
        f1 = f1_score(y, (s >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t


def hazard_rate(text):
    t = (text or "").lower()
    return 1 if any(k in t for k in HAZARD_LEXICON) else 0


# ── (5) per-TTE score-path characterization + overlaid histogram ─────────────

def per_tte_analysis(train_manifest_rows, val_rows, out_dir, neg_bucket="MID"):
    """Bucket positives by TTE, pair with the MID negatives, full suite per bucket;
    overlaid translucent per-TTE prediction histogram."""
    pos_by_tte = defaultdict(list)
    negs = []
    for r in train_manifest_rows:
        h = r["horizon_label"]
        if r["target"] == 1 and h.startswith("TTE_"):
            pos_by_tte[h].append(r)
        elif r["target"] == 0 and h == neg_bucket:
            negs.append(r)

    result = {"neg_bucket": neg_bucket, "buckets": {}}
    for tte in sorted(pos_by_tte):
        pos = pos_by_tte[tte]
        scores = [p["score"] for p in pos] + [n["score"] for n in negs]
        targets = [1] * len(pos) + [0] * len(negs)
        result["buckets"][tte] = metric_suite(scores, targets)

    # overlaid histogram
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        bins = np.linspace(0, 1, 21)
        colors = {"TTE_0.5": "#d7191c", "TTE_1.0": "#fdae61", "TTE_1.5": "#2c7bb6"}
        plt.figure(figsize=(8, 5))
        for tte in sorted(pos_by_tte):
            sc = [p["score"] for p in pos_by_tte[tte]]
            plt.hist(sc, bins=bins, alpha=0.45, density=True,
                     color=colors.get(tte, None), label=f"{tte} pos (n={len(sc)})")
            _kde_overlay(plt, sc, colors.get(tte, None))
        if negs:
            plt.hist([n["score"] for n in negs], bins=bins, alpha=0.25, density=True,
                     color="grey", label=f"{neg_bucket} neg (n={len(negs)})")
        plt.xlabel("Frozen V-JEPA2 P(collision)"); plt.ylabel("density")
        plt.title("BADAS-Open score distribution by TTE (closer event -> higher)")
        plt.legend(); plt.tight_layout()
        (Path(out_dir) / "plots").mkdir(parents=True, exist_ok=True)
        plt.savefig(Path(out_dir) / "plots" / "score_hist_by_tte.png", dpi=140)
        plt.close()
    except Exception as e:
        print(f"  [warn] histogram failed: {e}")

    # 18-val regression-guard scores (compare to Stage A externally)
    result["val_scores"] = {r["video_id"]: r["score"] for r in val_rows}
    return result


def _kde_overlay(plt, samples, color):
    if len(samples) < 3:
        return
    try:
        from scipy.stats import gaussian_kde
        xs = np.linspace(0, 1, 200)
        kde = gaussian_kde(samples)
        plt.plot(xs, kde(xs), color=color, lw=1.8)
    except Exception:
        pass


# ── gate eval (needs GPU + trained projector) ───────────────────────────────

def run_gate(cfg, out_dir):
    import torch
    from torch.utils.data import DataLoader
    from student_training.data.stageb_bridge_dataset import StageBBridgeDataset
    from student_training.models.vjepa_reason import load_llm, build_projector, StageBBridge

    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp_dtype = torch.bfloat16 if "bfloat16" in cfg.get("torch_dtype", "bfloat16") else torch.float16
    cache_dir = os.environ.get("E4_CACHE_DIR", cfg["data"]["cache_dir"])
    nq = cfg["projector"]["num_queries"]

    llm, tok = load_llm(cfg["llm"]["model_id"], dtype=amp_dtype)
    val_ds = StageBBridgeDataset(os.path.join(cache_dir, "cache_manifest_val.jsonl"),
                                 cache_dir, tok, num_vis_tokens=nq,
                                 max_seq_len=cfg["data"]["max_seq_len"])
    loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                        collate_fn=StageBBridgeDataset.collate_fn)

    def make_bridge(state_path):
        proj = build_projector(cfg)
        if state_path:
            proj.load_state_dict(torch.load(state_path, map_location="cpu"))
        return StageBBridge(llm, proj).to(device).eval()

    trained = make_bridge(Path(out_dir) / "projector_best.pt")
    random_proj = make_bridge(None)   # untrained = visual-noise floor

    def ppl(bridge, ablate="none", shuffle=False):
        tot, ntok = 0.0, 0
        feats_bank = [b["vis_feats"] for b in loader] if shuffle else None
        with torch.no_grad():
            for i, b in enumerate(loader):
                n = int((b["labels"] != -100).sum().item())
                if n == 0:
                    continue
                sf = feats_bank[(i + 1) % len(feats_bank)].to(device) if shuffle else None
                with torch.cuda.amp.autocast(enabled=(device == "cuda"), dtype=amp_dtype):
                    loss, _ = bridge(b["vis_feats"].to(device), b["input_ids"].to(device),
                                     b["attention_mask"].to(device), b["labels"].to(device),
                                     b["vis_mask"].to(device), ablate=ablate, shuffle_feats=sf)
                tot += float(loss.item()) * n; ntok += n
        ce = tot / max(ntok, 1)
        return math.exp(ce), ce

    # (1) Delta-PPL
    ppl_trained, ce_trained = ppl(trained, "none")
    ppl_random, _ = ppl(random_proj, "none")
    ppl_textonly, _ = ppl(trained, "mask")
    ppl_res = {
        "ppl_trained": round(ppl_trained, 4), "ppl_random_proj": round(ppl_random, 4),
        "ppl_text_only": round(ppl_textonly, 4),
        "gain_vs_random_pct": round(100 * (1 - ppl_trained / ppl_random), 2),
        "gain_vs_textonly_pct": round(100 * (1 - ppl_trained / ppl_textonly), 2),
    }

    # (2) ablation sensitivity (delta-CE; higher = more vision-dependent)
    _, ce_zero = ppl(trained, "zero")
    _, ce_shuf = ppl(trained, "shuffle", shuffle=True)
    abl_res = {"ce_real": round(ce_trained, 5), "ce_zeroed": round(ce_zero, 5),
               "ce_shuffled": round(ce_shuf, 5),
               "dCE_zero": round(ce_zero - ce_trained, 5),
               "dCE_shuffle": round(ce_shuf - ce_trained, 5)}

    # (3)+(4) generation discrimination
    gens = []
    pos_haz, neg_haz, lengths, reps, parseable = [], [], [], 0, 0
    with torch.no_grad():
        for b in loader:
            out_ids = trained.generate(
                b["vis_feats"].to(device), b["input_ids"].to(device),
                b["attention_mask"].to(device), b["vis_mask"].to(device),
                max_new_tokens=cfg["gate"].get("max_new_tokens", 160))
            text = tok.decode(out_ids[0], skip_special_tokens=True)
            meta = b["meta"][0]
            try:
                obj = json.loads(text[text.index("{"):text.rindex("}") + 1])
                reason = obj.get("reason", text); parseable += 1
            except Exception:
                reason = text
            (pos_haz if meta["target"] == 1 else neg_haz).append(hazard_rate(reason))
            lengths.append(len(reason.split()))
            reps += _is_degenerate(reason)
            gens.append({"video_id": meta["video_id"], "target": meta["target"],
                         "generated": text, "teacher_reason": meta["reason"]})

    disc = {
        "hazard_rate_pos": round(np.mean(pos_haz) if pos_haz else 0, 4),
        "hazard_rate_neg": round(np.mean(neg_haz) if neg_haz else 0, 4),
        "hazard_gap": round((np.mean(pos_haz) if pos_haz else 0) -
                            (np.mean(neg_haz) if neg_haz else 0), 4),
        "mean_gen_len_words": round(float(np.mean(lengths)) if lengths else 0, 1),
        "repetition_rate": round(reps / max(len(gens), 1), 3),
        "json_parseable_rate": round(parseable / max(len(gens), 1), 3),
    }

    _dump(out_dir, "metrics/ppl_results.json", ppl_res)
    _dump(out_dir, "metrics/ablation_sensitivity.json", abl_res)
    _dump(out_dir, "metrics/discrimination.json", disc)
    with open(Path(out_dir) / "generations.jsonl", "w", encoding="utf-8") as f:
        for g in gens:
            f.write(json.dumps(g, ensure_ascii=False) + "\n")

    print("\n=== GATE METRICS ===")
    print(" (1) PPL :", ppl_res)
    print(" (2) ABL :", abl_res)
    print(" (3) DISC:", disc)
    return {"ppl": ppl_res, "ablation": abl_res, "discrimination": disc}


def _is_degenerate(text, win=4):
    toks = text.split()
    if len(toks) < 2 * win:
        return 0
    return int(any(toks[i:i + win] == toks[i + win:i + 2 * win]
                   for i in range(len(toks) - 2 * win)))


def _dump(out_dir, rel, obj):
    p = Path(out_dir) / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(obj, f, indent=2)


def _load_manifest(cache_dir, split):
    p = Path(cache_dir) / f"cache_manifest_{split}.jsonl"
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()] if p.exists() else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--skip_gate", action="store_true",
                    help="Only run the §5(5) per-TTE score-path analysis (no LLM).")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()
    with open(args.config if os.path.isabs(args.config) else PROJECT_ROOT / args.config) as f:
        cfg = yaml.safe_load(f)

    out_dir = os.environ.get("E4_OUTPUT_DIR", cfg["data"]["output_dir"])
    cache_dir = os.environ.get("E4_CACHE_DIR", cfg["data"]["cache_dir"])
    train_rows = _load_manifest(cache_dir, "train")
    val_rows = _load_manifest(cache_dir, "val")

    if args.dry_run:
        print(f"train windows: {len(train_rows)}  val windows: {len(val_rows)}")
        print("horizons:", dict(Counter(r["horizon_label"] for r in train_rows)))
        return

    # (5) per-TTE — pure analysis, no model
    tte = per_tte_analysis(train_rows, val_rows, out_dir,
                           neg_bucket=cfg["gate"].get("neg_bucket", "MID"))
    _dump(out_dir, "metrics/score_path_by_tte.json", tte)
    print("\n=== PER-TTE (frozen scorer) ===")
    for k, v in tte["buckets"].items():
        ap = v.get("ap"); print(f"  {k}: AP={ap if ap is None else round(ap,4)}  n={v['n']}")

    if not args.skip_gate:
        run_gate(cfg, out_dir)
    print(f"\nOutputs in {out_dir}")


if __name__ == "__main__":
    main()
