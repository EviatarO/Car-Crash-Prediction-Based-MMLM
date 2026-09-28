"""
build_experiments_data.py
=========================
Generates website/experiments_data.js (window.EXPERIMENTS_DATA) - the per-arm payload
behind the Experiments page's detail view: description, prompt, dataset composition,
architecture configuration, hyperparameters, training curves and test results for
A0, A1, B-v1, B-v2, B-v3, P1 and the four a1fail321 recovery arms (a1cont, V10, V12,
v12shuf) - ten arms, all scored on the same 677-clip test set.

Every metric comes from student_training/scripts/metrics_core.py::metrics_from_arrays -
the same function the training pipeline itself uses - so the page cannot disagree with
the run reports by using a different formula, threshold or rounding.

TWO SOURCES OF TRUTH, ON PURPose
--------------------------------
Where a run wrote a `test_summary.json`, that file is authoritative for AP/AUC, because
`semsup_train.py` stores per-clip scores as `round(s, 4)` and the resulting ties perturb
average precision. The per-clip dump stays authoritative for the confusion matrix, which
is insensitive to those ties. Before trusting a summary we assert that its f1/recall/
specificity match the dump's - if they disagree the two files are not the same
checkpoint, and the override would be silently wrong. B-v3 is the arm where this matters
most: 409 of its 677 test scores are tied (101 sit at exactly 1.0), so its dump-derived
AP reads 0.8655 against a published 0.8784.

WHAT IS DELIBERATELY ABSENT
---------------------------
Rendered as explicit "not available" notes rather than interpolated:
  - accuracy-vs-epoch and AUC-vs-epoch: `epoch_metrics.jsonl` never logged them for any
    arm. `val_ap` is the per-epoch curve that exists, and is literally the checkpoint
    selection criterion. For the four a1fail321 arms only, per-epoch val accuracy/AUC
    ARE derived here from their `val_scores_ep*.jsonl` dumps.
  - train-split per-example scores: never dumped, so no train ROC / train confusion
    matrix exists for any arm.
  - per-TTE metrics on the TRAINING pools: TTE is perfectly confounded with the label
    there (every TTE_* window is positive, every MID-* window negative), so per-bucket
    AP/AUC is undefined. Test-set bucketing is valid and is what the page offers.

    python website/build_experiments_data.py
"""
import json
import sys
from collections import Counter, OrderedDict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_site_data import MMLM_AI  # noqa: E402

sys.path.insert(0, str(MMLM_AI / "student_training" / "scripts"))
from metrics_core import metrics_from_arrays  # noqa: E402

sys.path.insert(0, str(MMLM_AI))

OUT = Path(__file__).resolve().parent / "experiments_data.js"
E4 = MMLM_AI / "outputs" / "e4_vjepa_reason"
A1F = MMLM_AI / "outputs" / "a1fail321"
A1C = MMLM_AI / "outputs" / "a1_compress256"
AAT = MMLM_AI / "outputs" / "aa_token_aux"
AAH = MMLM_AI / "outputs" / "aa_head_attn"
OVN = MMLM_AI / "outputs" / "overnight_2026-09-27"
CAPS = MMLM_AI / "outputs" / "semantic_captions"
MANIFESTS = MMLM_AI / "dataset" / "manifests"
TEST_MANIFEST = MANIFESTS / "test_manifest_hires.jsonl"

THRESHOLD = 0.5
# Stage 1 noise floor: mean-over-8-checkpoint test AP of the two LoRA-init-seed controls
# (AA-ctrl-seed1/2, split held fixed at A1-compress256's partition). DERIVED in main() from
# those two arms' own by_epoch means, not hardcoded, so the floor and every arm's
# mean-over-8 on this page are computed the same way (from the rounded per-epoch dumps).
# The published-summary basis reads 0.8910-0.8958 for the same two runs.
NOISE_FLOOR_ARMS = ("AA-ctrl-seed1", "AA-ctrl-seed2")
# metrics_core.py's own mapping - group is an int on the test manifest and every test dump.
GROUP_LABEL = {0: "tte_0.5s", 1: "tte_1.0s", 2: "tte_1.5s"}
TTE_ORDER = ["tte_0.5s", "tte_1.0s", "tte_1.5s"]

# Expected confusion-matrix values, verified against the source artifacts. Same numbers
# as build_landing_data.py's EXPECTED dict (kept in sync by hand - the two builders read
# the same underlying score files, so a real drift trips BOTH pages, not just one).
#
# Project review 2026-09-06 §5.2: before this, only 4 of the 10 arms (those with a
# test_summary.json) had ANY drift protection here, via the f1/recall/specificity gate
# below - and that gate only fires when cfg["summary"] is set. A0, B-v1, a1cont, V10,
# V12, v12shuf had ZERO protection: a re-scored or partially-overwritten 677-row dump for
# any of those six would render new numbers on this page with no assertion catching it,
# even though build_landing_data.py's pin on the SAME data would have caught it. This
# dict closes that gap for every arm, not just the four with a summary file.
EXPECTED_CM = {
    "A0":      dict(n=677, tp=308, fn=30, fp=130, tn=209),
    "A1":      dict(n=677, tp=320, fn=18, fp=123, tn=216),
    "B-v1":    dict(n=677, tp=317, fn=21, fp=130, tn=209),
    "B-v2":    dict(n=677, tp=285, fn=53, fp=76, tn=263),
    "B-v3":    dict(n=677, tp=267, fn=71, fp=55, tn=284),
    "P1":      dict(n=677, tp=278, fn=60, fp=92, tn=247),
    "a1cont":  dict(n=677, tp=238, fn=100, fp=34, tn=305),
    "V10":     dict(n=677, tp=253, fn=85, fp=39, tn=300),
    "V12":     dict(n=677, tp=253, fn=85, fp=39, tn=300),   # build_landing_data.py: "v12"
    "v12shuf": dict(n=677, tp=244, fn=94, fp=36, tn=303),
    "A1-compress256": dict(n=677, tp=286, fn=52, fp=55, tn=284),
    "AA-occ-unfrozen": dict(n=677, tp=264, fn=74, fp=45, tn=294),
    "AA-rel-unfrozen": dict(n=677, tp=276, fn=62, fp=51, tn=288),
    "AA-rel": dict(n=677, tp=310, fn=28, fp=93, tn=246),
    "AA-occ": dict(n=677, tp=303, fn=35, fp=86, tn=253),
    "AA-rel-L23": dict(n=677, tp=283, fn=55, fp=69, tn=270),
    "AA-ctrl-unfrozen": dict(n=677, tp=258, fn=80, fp=41, tn=298),
    "AA-ctrl-seed1": dict(n=677, tp=271, fn=67, fp=60, tn=279),
    "AA-ctrl-seed2": dict(n=677, tp=267, fn=71, fp=45, tn=294),
    "AA-H-rank-R_all": dict(n=677, tp=288, fn=50, fp=57, tn=282),
    "AA-H-rank-R_pos": dict(n=677, tp=287, fn=51, fp=55, tn=284),
    "AA-H-rank-partner_pos": dict(n=677, tp=288, fn=50, fp=55, tn=284),
    "AA-H-mass-R_pos": dict(n=677, tp=277, fn=61, fp=77, tn=262),
    "midneg-seed0": dict(n=677, tp=260, fn=78, fp=45, tn=294),
    "midneg-seed1": dict(n=677, tp=269, fn=69, fp=47, tn=292),
    "midneg-seed2": dict(n=677, tp=202, fn=136, fp=16, tn=323),
    "fullpool-seed0": dict(n=677, tp=300, fn=38, fp=77, tn=262),
    "fullpool-seed1": dict(n=677, tp=288, fn=50, fp=75, tn=264),
    "fullpool-seed2": dict(n=677, tp=314, fn=24, fp=107, tn=232),
}


# --------------------------------------------------------------------------- io helpers
def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def to01(v):
    """Labels arrive as 0/1 ints ('ground_truth') or YES/NO strings ('gt_verdict')."""
    return int(v) if not isinstance(v, str) else (1 if v == "YES" else 0)


def bucket_of(row):
    """The TTE/MID horizon. Caption corpora disagree on which key holds it - V12's
    populates `requested_time_to_event` and leaves `horizon_label` null, V10's does the
    reverse - so read both rather than depending on one file's convention."""
    return row.get("requested_time_to_event") or row.get("horizon_label")


def roc_points(y, s, max_pts=140):
    """ROC as plain arrays for the SVG chart. Downsampled to a bounded number of points
    (evenly over the curve, endpoints always kept) so eight arms x four buckets stay a
    small payload instead of ~40k floats."""
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y, s)
    n = len(fpr)
    if n > max_pts:
        idx = sorted({round(i * (n - 1) / (max_pts - 1)) for i in range(max_pts)})
    else:
        idx = range(n)
    return {"fpr": [round(float(fpr[i]), 4) for i in idx],
            "tpr": [round(float(tpr[i]), 4) for i in idx]}


# ------------------------------------------------------------------------- pool metadata
def pool_stats(rows, split_key=None):
    """Composition of a training pool: labels, horizon histogram, clip count."""
    y = [to01(r["gt_verdict"]) for r in rows]
    buckets = Counter(bucket_of(r) for r in rows)
    out = {
        "n_windows": len(rows),
        "n_clips": len({r["video_id"] for r in rows}),
        "n_pos": sum(y), "n_neg": len(y) - sum(y),
        "buckets": OrderedDict(
            (k, buckets.get(k, 0))
            for k in ["TTE_0.5", "TTE_1.0", "TTE_1.5", "MID-4", "MID-8", "MID-10"]),
    }
    if split_key:
        sp = Counter(r.get(split_key) for r in rows)
        out["split"] = {"train": sp.get("train", 0), "val": sp.get("val", 0)}
    return out


def build_pools():
    pool1761 = pool_stats(load_jsonl(CAPS / "Caption_V12_Neutral_1761_fortrain.jsonl"))
    pool1761.update({
        "key": "pool1761", "name": "Pool-1761",
        "blurb": "1,761 windows mined from the 4,446-window train pool: 587 windows the "
                 "frozen A0 baseline gets wrong, plus 587 true-positive and 587 "
                 "true-negative controls it gets right.",
        "split_note": "Split by CLIP (val_frac 0.2, seed 0) → 1,413 train / 348 val "
                      "windows over 221 val clips. Identical across every arm trained "
                      "on this pool, so their val numbers are directly comparable.",
    })
    # a1fail321 carries its own train/val assignment as a field rather than deriving it.
    a1f = pool_stats(load_jsonl(A1F / "selection_a1fail321.jsonl"), split_key="split")
    a1f.update({
        "key": "a1fail321", "name": "A1-fail-321",
        "blurb": "Every window the A1 champion gets wrong at threshold 0.5 — all 321 of "
                 "them. A1's own AUC on this pool is exactly 0.0 by construction, so "
                 "there is no headroom to fake: any gain has to be a real repair.",
        "split_note": "Split by VIDEO (seed 0) → 260 train / 61 val windows, so sibling "
                      "windows of the same clip cannot leak across the split.",
    })

    test_rows = load_jsonl(TEST_MANIFEST)
    tb = Counter((r["group"], r["event_occurs"]) for r in test_rows)
    test = {
        "key": "test677", "name": "Nexar private test set",
        "n_windows": len(test_rows),
        "n_clips": len({r["video_id"] for r in test_rows}),
        "n_pos": sum(1 for r in test_rows if r["event_occurs"] == 1),
        "n_neg": sum(1 for r in test_rows if r["event_occurs"] == 0),
        "blurb": "677 held-out clips, one 16-frame window each. Never trained on by any "
                 "arm, and the only set where the TTE buckets contain both classes — "
                 "which is why per-horizon metrics are offered here and nowhere else.",
        "tte": OrderedDict(
            (GROUP_LABEL[g], {"n": tb[(g, 1)] + tb[(g, 0)],
                              "pos": tb[(g, 1)], "neg": tb[(g, 0)]})
            for g in (0, 1, 2)),
    }
    return {"pool1761": pool1761, "a1fail321": a1f, "test677": test}


# ----------------------------------------------------------------------------- prompts
def load_prompt(kind):
    """Import and CALL the real prompt builder, so the page can never drift from the
    prompt the captions were actually generated with."""
    if kind == "v10":
        from prompts.PROMPT_SEMSUP_V10_GT import build_prompt
        return {"name": "V10 · ground-truth conditioned",
                "file": "prompts/PROMPT_SEMSUP_V10_GT.py",
                "text": build_prompt("gt", True)}
    if kind == "v12":
        from prompts.PROMPT_SEMSUP_V12_NEUTRAL import build_prompt
        return {"name": "V12 · register-neutral",
                "file": "prompts/PROMPT_SEMSUP_V12_NEUTRAL.py",
                "text": build_prompt()}
    raise ValueError(kind)


# --------------------------------------------------------------------- hyperparameters
# Order matters: this is the reading order on the page, grouped adapter → objective →
# schedule → bookkeeping, not argparse's declaration order.
HYPER_KEYS = [
    "lora_target_modules", "lora_r", "lora_alpha", "lora_dropout", "lora_init",
    "predictor_init", "crash_weight", "semantic_weight", "semantic_loss",
    "infonce_tau_init", "siglip_model",
    "aux_mode", "aux_label", "aux_layer", "aux_weight", "aux_margin", "aux_schedule",
    "aux_warmup_frac", "aux_on_epochs", "aux_head_init",
    "captions_path", "bank_captions",
    "lr", "lr_schedule", "warmup_frac", "epochs", "grad_accum", "clip_grad_per_group",
    "unfreeze_head", "head_lr_mult", "head_lr_schedule",
    "early_stop_patience", "select_by", "keep_top_k", "val_frac", "seed",
]


# One clause each, sized to sit between ARGUMENT and VALUE on a single row. Kept next to
# HYPER_KEYS so the text travels with the ordering it documents rather than drifting in a
# separate file.
HYPER_DESC = {
    "lora_target_modules": "which Linear layers get an adapter",
    "lora_r": "adapter rank - the capacity added per layer",
    "lora_alpha": "adapter scaling; effective LR multiplier is alpha/r",
    "lora_dropout": "dropout inside the adapter only",
    "lora_init": "checkpoint the LoRA weights start from (blank = from scratch)",
    "predictor_init": "checkpoint the semantic Predictor starts from",
    "crash_weight": "weight on the crash cross-entropy term",
    "semantic_weight": "lambda on the semantic term; 0 makes this a crash-only arm",
    "semantic_loss": "how caption and vision embeddings are compared",
    "infonce_tau_init": "starting temperature of the InfoNCE softmax",
    "siglip_model": "frozen text encoder that embeds the captions",
    "aux_mode": "Stage AA per-token aux target (occ/rel, layer probe) or Stage AA-H "
                "crash-head aux loss (attn_rank/attn_mass/gradcam)",
    "aux_label": "Stage AA-H only: which windows count as R (R_all/R_pos/partner_pos)",
    "aux_layer": "encoder layer the aux loss reads (0-indexed); backprop reaches only "
                 "layers 0..this one — Stage AA (side probe) only",
    "aux_weight": "lambda on the aux term; sized so its gradient on the shared LoRA "
                  "layers is ~10-30% of the crash gradient's (measured, not guessed)",
    "aux_margin": "attn_rank only: required log-ratio margin before the loss goes slack",
    "aux_schedule": "constant = aux loss on for every epoch; warm_on_off = on for the "
                    "first --aux-on-epochs, then off for the rest",
    "aux_warmup_frac": "fraction of the aux-on window spent ramping lambda up from 0",
    "aux_on_epochs": "warm_on_off only: how many epochs the aux loss stays active",
    "aux_head_init": "frozen linear probe (LayerNorm+Linear) the aux loss is read "
                     "through; fit once on the untrained trunk, never updated here",
    "captions_path": "caption corpus supervising this run",
    "bank_captions": "wider corpus used only to add InfoNCE distractors",
    "lr": "peak learning rate for the trunk/LoRA group",
    "lr_schedule": "how the learning rate decays after warmup",
    "warmup_frac": "fraction of training spent warming the LR up",
    "epochs": "passes over the training pool",
    "grad_accum": "batches accumulated before an optimizer step",
    "clip_grad_per_group": "clip LoRA and Predictor on separate budgets, not one shared",
    "unfreeze_head": "whether the crash head trains too, or stays frozen",
    "head_lr_mult": "head learning rate as a multiple of the trunk's",
    "head_lr_schedule": "constant keeps the head's LR flat once the trunk's decays",
    "early_stop_patience": "epochs without improvement before stopping (0 = never)",
    "select_by": "metric that picks the reported checkpoint",
    "keep_top_k": "how many checkpoints are kept and scored",
    "val_frac": "fraction of clips held out, split by video",
    "seed": "seed for the split and initialisation",
}


def shorten(v):
    """Long absolute paths are noise in a table; the basename identifies the file."""
    if isinstance(v, str) and ("/" in v or "\\" in v) and not v.startswith("re:"):
        return v.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    return v


def hyper_rows(args):
    rows = []
    for k in HYPER_KEYS:
        if k in args and args[k] is not None:
            rows.append([k, HYPER_DESC.get(k, ""), str(shorten(args[k]))])
    return rows


# a1fail321's arms were launched from run_a1fail321_4arms.sh and interrupted before
# train_metrics.json was written, so their configuration is transcribed from that script
# (the checked-in launcher IS the record) rather than read back from an args dump.
A1FAIL_ARGS = {
    "lora_target_modules": "query,key,value", "lora_r": 16, "lora_alpha": 32,
    "lora_dropout": 0.05, "lora_init": "a1_1761/epoch_04/lora_adapter",
    "predictor_init": "b1_v2_100pct/predictor_b1.pt",
    "crash_weight": 1.0, "semantic_weight": 0.2, "semantic_loss": "infonce",
    "infonce_tau_init": 0.07, "siglip_model": "google/siglip-base-patch16-224",
    "lr": 2e-05, "lr_schedule": "cosine", "warmup_frac": 0.1, "epochs": 10,
    "grad_accum": 8, "unfreeze_head": False, "select_by": "val_ap", "keep_top_k": 10,
    "val_frac": 0.2, "seed": 0,
}


# a1cont is the same launcher recipe with the semantic branch switched off entirely: no
# caption path, no InfoNCE bank, no Predictor to warm-start. Everything else - the A1
# epoch-4 init, LR, schedule, epochs, seed - is identical, which is what makes it the
# controlled floor for the three semantic arms.
A1FAIL_CRASH_ONLY = {k: v for k, v in A1FAIL_ARGS.items()
                     if k not in ("predictor_init", "semantic_loss", "infonce_tau_init",
                                  "siglip_model")}
A1FAIL_CRASH_ONLY["semantic_weight"] = 0.0


# --------------------------------------------------------------------------- arm registry
def A(**kw):
    return kw


ARMS = [
    A(key="A0", label="A0 · Frozen baseline", order=0, family="pool1761",
      tagline="BADAS-Open exactly as published — no training of any kind.",
      hypothesis="None. A0 is the reference point, not a treatment: it fixes the number "
                 "every other arm has to beat.",
      aim="Establish the off-the-shelf capability of BADAS-Open on this task, and — "
          "because its per-window errors define which clips are worth training on — "
          "supply the mining signal for the Pool-1761 and A1-fail-321 pools.",
      method="Load the published BADAS-Open (V-JEPA2 ViT-L) checkpoint untouched and "
             "score every window. No LoRA, no gradient, no language branch. Scores are "
             "softmax(logits/2)[1]; that divisor is a monotone transform, so it changes "
             "neither AP/AUC nor any decision at threshold 0.5.",
      prompt=None,
      prompt_note="No language supervision in this arm — there is no caption branch to "
                  "prompt.",
      pool=None, train_dir=None,
      arch=dict(semantic=False, loss=False, state={"lora": "absent"},
                note="nothing is trained — every module is the published checkpoint"),
      hyper=None,
      hyper_note="Not applicable: A0 is never trained, so there is no configuration to "
                 "record.",
      test=dict(path=E4 / "StageA_scorer" / "badas_open_private.jsonl",
                gt_key="ground_truth", summary=None, epoch=None,
                source="e4_vjepa_reason/StageA_scorer/badas_open_private.jsonl")),

    A(key="A1", label="A1 · Crash-only control", order=1, family="pool1761",
      tagline="LoRA fine-tuning on crash labels alone. The champion, and the honest control.",
      hypothesis="LoRA-adapting the frozen V-JEPA2 trunk on collision labels alone is "
                 "enough to beat the off-the-shelf baseline — no language needed.",
      aim="Separate the fine-tuning contribution from the semantic contribution. The "
          "claim of interest for every B arm is B − A1, not B − A0; without this control "
          "any gain from a semantic arm could just be the LoRA doing the work.",
      method="LoRA r=16, α=32 on the trunk's query/key/value projections, crash "
             "cross-entropy only (--semantic-weight 0). The crash head (temporal "
             "processor + classifier) stays frozen, so the comparison isolates what the "
             "trunk's features do.",
      prompt=None,
      prompt_note="No language supervision in this arm. Captions were loaded (the corpus "
                  "defines the window list) but λ=0, so no caption ever reaches a gradient.",
      pool="pool1761", train_dir=E4 / "a1_1761",
      arch=dict(semantic=False, loss=True, state={}, note=None),
      hyper=E4 / "a1_1761" / "train_metrics.json", hyper_note=None,
      test=dict(path=E4 / "a1_1761" / "test_results_ep04.jsonl", gt_key="ground_truth",
                summary=E4 / "a1_1761" / "test_summary.json", epoch=4,
                source="e4_vjepa_reason/a1_1761 (epoch 4)")),

    A(key="A1-compress256", label="A1-compress256 · Full-frame preprocessing", order=1.5,
      family="pool1761",
      tagline="A1's identical recipe, but the model sees the WHOLE frame instead of a "
              "center crop that discards ~51% of the width.",
      hypothesis="Every arm to date — A0 through v12shuf — was accidentally trained and "
                 "scored on a center-cropped view (V-JEPA2's processor default: resize "
                 "shortest-edge 292, then center-crop 256×256), keeping only source "
                 "x∈[321,953] of a 1280-wide frame. On A1's own worst-missed crashes, "
                 "the collision partner sits outside that crop. Feeding the full frame "
                 "(resized to 256×256, no crop) should recover that lost context.",
      aim="Test whether A1's recipe, unchanged in every other respect, does better with "
          "the full field of view — and separately, whether the FROZEN model alone "
          "(no training) already benefits, to isolate a preprocessing effect from a "
          "fine-tuning effect.",
      method="Identical to A1: LoRA r=16, α=32 on query/key/value, crash CE only, lr 2e-4 "
             "constant, 8 epochs, same 1,761-window pool, same clip-level split "
             "(split_seed=0). The ONLY change is `--preprocess compress256`: the V-JEPA2 "
             "processor resizes the full 1280×720 frame to 256×256 with `do_center_crop"
             "=False`, instead of its default center-crop. Val-selected rank-1 = epoch 2 "
             "(val_ap=0.9528, vs A1's own best val_ap of 0.9143).",
      prompt=None,
      prompt_note="No language supervision — same as A1.",
      pool="pool1761", train_dir=A1C / "train",
      arch=dict(semantic=False, loss=True, state={}, note="preprocess=compress256 "
               "(full frame, no crop) — the only difference from A1"),
      hyper=A1C / "train" / "train_metrics.json", hyper_note=None,
      test=dict(path=A1C / "train" / "test_results_ep02.jsonl", gt_key="ground_truth",
                summary=A1C / "train" / "test_summary.json", epoch=2,
                source="a1_compress256/train (epoch 2)")),

    A(key="B-v1", label="B-v1 · Crash + semantic (parallel)", order=2, family="pool1761",
      tagline="First joint arm: crash CE and an InfoNCE caption loss trained together.",
      hypothesis="Supervising the trunk with teacher captions during training adds "
                 "information the binary crash label does not carry, and that extra "
                 "structure should show up as better collision anticipation at "
                 "inference — which stays vision-only and free.",
      aim="Test the core thesis claim for the first time at the 1,761-window scale, "
          "against A1 rather than against A0.",
      method="Crash CE + 0.05 · InfoNCE between a trainable Predictor (8 learned queries "
             "over the trunk's patch tokens, mean-pooled) and frozen SigLIP text "
             "embeddings of V10 teacher captions. Predictor cold-started.",
      prompt="v10",
      prompt_note=None,
      pool="pool1761", train_dir=E4 / "b_1761_par",
      arch=dict(semantic=True, loss=True, state={},
                note="Predictor cold-started (later found to be a defect — see B-v3)"),
      hyper=E4 / "b_1761_par" / "train_metrics.json", hyper_note=None,
      test=dict(path=E4 / "b_1761_par" / "test_results_ep04.jsonl", gt_key="ground_truth",
                summary=None, epoch=4,
                by_epoch=[(n, E4 / "b_1761_par" / f"test_results_ep{n:02d}.jsonl")
                          for n in range(1, 7)],
                source="e4_vjepa_reason/b_1761_par (epoch 4)")),

    A(key="B-v2", label="B-v2 · Same, on neutral captions", order=3, family="pool1761",
      tagline="B-v1 rerun on the register-neutral V12 corpus, to rule out caption leakage.",
      hypothesis="B-v1's loss was caused by V10's captions leaking the label through "
                 "their register (positives and negatives written in different voices). "
                 "A neutral corpus should let the semantic term help.",
      aim="Rule leakage in or out as the explanation for B-v1's result, holding "
          "everything else at A1's exact recipe.",
      method="Identical to A1's recipe (from-scratch LoRA, seed 0, constant LR, 8 epochs) "
             "plus --semantic-weight 0.05 --semantic-loss infonce, captions swapped to "
             "the V12 neutral corpus.",
      prompt="v12",
      prompt_note=None,
      pool="pool1761", train_dir=E4 / "b_v2_1761",
      arch=dict(semantic=True, loss=True, state={},
                note="Predictor cold-started; gradient-clip budget shared with LoRA"),
      hyper=E4 / "b_v2_1761" / "train_metrics.json", hyper_note=None,
      test=dict(path=E4 / "b_v2_1761" / "test_results_ep02.jsonl", gt_key="ground_truth",
                summary=E4 / "b_v2_1761" / "test_summary.json", epoch=2,
                by_epoch=[(2, E4 / "b_v2_1761" / "test_results_ep02.jsonl"),
                          (4, E4 / "b_v2_1761" / "test_results_ep04.jsonl")],
                source="e4_vjepa_reason/b_v2_1761 (epoch 2)")),

    A(key="B-v3", label="B-v3 · Both execution defects fixed", order=4, family="pool1761",
      tagline="B-v2 with a warm-started Predictor and per-group gradient clipping.",
      hypothesis="B-v2 lost because of two execution defects, not because the idea is "
                 "wrong: the Predictor was cold-started (so early semantic gradients were "
                 "noise) and LoRA shared its gradient-clip budget with the Predictor (so "
                 "LoRA's effective step size differed from A1's).",
      aim="Give the semantic arm its best honest shot — if it still loses with both "
          "defects fixed, the defects were not the explanation.",
      method="B-v2's recipe plus --predictor-init from a B1 probe trained on the V12 "
             "corpus, and --clip-grad-per-group so LoRA and Predictor are clipped on "
             "separate budgets matching A1's effective LoRA budget. Extended to 12 epochs.",
      prompt="v12",
      prompt_note=None,
      pool="pool1761", train_dir=E4 / "b_v3_1761",
      arch=dict(semantic=True, loss=True, state={},
                note="Predictor warm-started from the B1 probe; LoRA clipped on its own budget"),
      hyper=None,
      hyper_note="Not recorded. This run's train_metrics.json was never synced from the "
                 "pod, and the local epoch_metrics.jsonl was overwritten by the 12-epoch "
                 "continuation. The recipe is B-v2's plus --predictor-init and "
                 "--clip-grad-per-group, but the exact argument dump is not on disk, so "
                 "nothing is reconstructed here.",
      train_note="Only epochs 9–12 survive locally: the 12-epoch continuation overwrote "
                 "the original 1–8 log. The curve below therefore starts at epoch 9 — it "
                 "is not a run that began there.",
      test=dict(path=E4 / "b_v3_1761" / "test_results_ep10.jsonl", gt_key="ground_truth",
                summary=E4 / "b_v3_1761" / "test_summary.json", epoch=10,
                by_epoch=[(2, E4 / "b_v3_1761" / "test_results_ep02.jsonl"),
                          (10, E4 / "b_v3_1761" / "test_results_ep10.jsonl")],
                source="e4_vjepa_reason/b_v3_1761 (epoch 10, the 12-epoch continuation)")),

    A(key="P1", label="P1 · Two-stage (semantic → crash)", order=5, family="pool1761",
      tagline="Pre-train the trunk on captions only, then fine-tune on crash labels.",
      hypothesis="If a joint loss makes the two objectives fight for the same weights, "
                 "separating them in TIME should help: learn semantics first with no "
                 "crash gradient at all, then fine-tune on crash labels from that "
                 "initialization.",
      aim="Test the sequencing alternative to joint training — the last structural "
          "variant available before concluding the semantic signal simply does not "
          "transfer.",
      method="Stage A: --crash-weight 0, semantic only, 12 epochs, checkpoint selected by "
             "clip-level retrieval@1 (val_ap is uninformative when nothing optimizes it); "
             "retrieval peaked at epoch 10 at 20.81%, 46× chance. Stage B (reported here): "
             "crash-only, LoRA warm-started from Stage A epoch 10, otherwise A1's recipe.",
      prompt="v12",
      prompt_note=None,
      pool="pool1761", train_dir=E4 / "p1_stageB",
      arch=dict(semantic=False, loss=True, state={},
                note="Stage B shown: LoRA warm-started from Stage A epoch 10; no semantic "
                     "branch is constructed at this stage"),
      hyper=E4 / "p1_stageB" / "train_metrics.json", hyper_note=None,
      test=dict(path=E4 / "p1_stageB" / "test_results_ep02.jsonl", gt_key="ground_truth",
                summary=E4 / "p1_stageB" / "test_summary.json", epoch=2,
                by_epoch=[(1, E4 / "p1_stageB" / "test_results_ep01.jsonl"),
                          (2, E4 / "p1_stageB" / "test_results_ep02.jsonl")],
                source="e4_vjepa_reason/p1_stageB (epoch 2)")),

    A(key="a1cont", label="A1-cont · Recovery control (no captions)", order=6,
      family="a1fail321",
      tagline="The same weights and the same 321 windows, with the caption term switched off.",
      hypothesis="None of its own - this arm exists to make the others interpretable. If a "
                 "semantic arm beats it, the caption term did something; if it matches, "
                 "whatever the semantic arms gained came from continued crash training on "
                 "the failure pool, not from language.",
      aim="Be the controlled floor. Same A1 epoch-4 initialisation, same pool, same LR and "
          "schedule and seed as V10/V12/V12-shuffled - the only difference is that no "
          "caption ever reaches a gradient.",
      method="Identical to the recovery recipe with --semantic-weight 0: no caption corpus, "
             "no InfoNCE bank, no Predictor. Crash cross-entropy only, crash head frozen.",
      prompt=None,
      prompt_note="No language supervision in this arm - that is the whole point of it.",
      pool="a1fail321", train_dir=A1F / "results" / "a1cont" / "fold_01",
      arch=dict(semantic=False, loss=True, state={},
                note="LoRA initialized from A1 epoch 4; no semantic branch is constructed"),
      hyper=A1FAIL_CRASH_ONLY,
      hyper_note="Transcribed from run_a1fail321_4arms.sh (the crash-only variant) - this "
                 "run's train_metrics.json was never written, so the checked-in launcher "
                 "is the record.",
      test=dict(path=A1F / "test_scores" / "a1cont_ep10.jsonl", gt_key="gt_verdict",
                summary=None, epoch=10,
                source="a1fail321/test_scores/a1cont_ep10.jsonl (epoch 10)")),

    A(key="V10", label="V10 · Failure recovery, GT captions", order=7, family="a1fail321",
      tagline="Start from A1's weights and train only on the 321 windows A1 gets wrong.",
      hypothesis="Semantic supervision failed at pool scale because the signal was "
                 "diluted across mostly-easy windows. Concentrated on A1's own failures — "
                 "where there is nothing left to lose — caption content should finally "
                 "move the score.",
      aim="Two questions at once: does the semantic term repair A1's failures, and does "
          "training on them cost A1 the 0.900 test AP it already has?",
      method="LoRA warm-started from A1 epoch 4, Predictor warm-started from the B1 (V12, "
             "100%) probe, λ raised to 0.2, InfoNCE bank widened to the full 1,761-caption "
             "corpus so the contrastive task is not trivially easy. Crash head frozen — it "
             "is what A1's 0.900 was measured with.",
      prompt="v10",
      prompt_note=None,
      pool="a1fail321", train_dir=A1F / "results" / "v10" / "fold_01",
      arch=dict(semantic=True, loss=True, state={},
                note="LoRA initialized from A1 epoch 4; Predictor from the B1 (V12, 100%) probe"),
      hyper=A1FAIL_ARGS,
      hyper_note="Transcribed from run_a1fail321_4arms.sh — this run's train_metrics.json "
                 "was never written, so the checked-in launcher is the record.",
      test=dict(path=A1F / "test_scores" / "v10_ep10.jsonl", gt_key="gt_verdict",
                summary=None, epoch=10,
                source="a1fail321/test_scores/v10_ep10.jsonl (epoch 10)")),

    A(key="V12", label="V12 · Failure recovery, neutral captions", order=8, family="a1fail321",
      tagline="The same recovery run on register-neutral captions — the cleanest B-vs-A1 test.",
      hypothesis="Same as V10, on the neutral corpus: with the register leak removed and "
                 "the predictor demonstrably learning, concentrated semantic supervision "
                 "should repair A1's failures without damaging what A1 already gets right.",
      aim="The decisive arm of the recovery study. Paired against a crash-only control "
          "started from the identical weights, so any difference is attributable to the "
          "semantic term and nothing else.",
      method="Identical to V10 with V12 neutral captions. A class-preserving shuffled-"
             "caption arm ran alongside as the content control: if real and shuffled "
             "captions perform the same, the effect is caption presence, not meaning.",
      prompt="v12",
      prompt_note=None,
      pool="a1fail321", train_dir=A1F / "results" / "v12" / "fold_01",
      arch=dict(semantic=True, loss=True, state={},
                note="LoRA initialized from A1 epoch 4; Predictor from the B1 (V12, 100%) probe"),
      hyper=A1FAIL_ARGS,
      hyper_note="Transcribed from run_a1fail321_4arms.sh — this run's train_metrics.json "
                 "was never written, so the checked-in launcher is the record.",
      test=dict(path=A1F / "test_scores" / "v12_ep10.jsonl", gt_key="gt_verdict",
                summary=None, epoch=10,
                source="a1fail321/test_scores/v12_ep10.jsonl (epoch 10)")),

    A(key="v12shuf", label="V12-shuffled · Caption-content control", order=9,
      family="a1fail321",
      tagline="V12 with the captions permuted within class - same words, wrong clips.",
      hypothesis="If V12's effect comes from what the captions SAY, scrambling which clip "
                 "each caption belongs to should destroy it. If instead the effect comes "
                 "from merely having a second loss term - a regulariser that happens to be "
                 "text-shaped - scrambled captions will perform the same.",
      aim="Separate caption CONTENT from caption PRESENCE. This is the control that decides "
          "whether the thesis claim is about language or about auxiliary-loss regularisation, "
          "and no aggregate metric on V12 alone can answer it.",
      method="Byte-identical to V12 except the caption file: the same V12 captions permuted "
             "WITHIN class (YES with YES, NO with NO, enforced as a derangement), so class "
             "balance and vocabulary are untouched and only the clip-caption pairing is "
             "destroyed. Same prompt, same weights, same schedule, same seed.",
      prompt="v12",
      prompt_note="Same V12 prompt as the real arm - the prompt did not change, the caption "
                  "ASSIGNMENT did. This is not a different prompt variant.",
      pool="a1fail321", train_dir=A1F / "results" / "v12shuf" / "fold_01",
      arch=dict(semantic=True, loss=True, state={},
                note="identical to V12; only the clip-to-caption pairing is scrambled"),
      hyper=A1FAIL_ARGS,
      hyper_note="Transcribed from run_a1fail321_4arms.sh - this run's train_metrics.json "
                 "was never written, so the checked-in launcher is the record. The only "
                 "differing argument is --captions-path (the shuffled corpus).",
      test=dict(path=A1F / "test_scores" / "v12shuf_ep10.jsonl", gt_key="gt_verdict",
                summary=None, epoch=10,
                source="a1fail321/test_scores/v12shuf_ep10.jsonl (epoch 10)")),

    A(key="AA-occ-unfrozen", label="AA-occ-unfrozen · occupancy aux, unfrozen head",
      order=10, family="pool1761",
      tagline="Localisation-only control for Stage AA's token-relevance aux loss — teach "
              "the trunk 'a car is here', nothing about which car matters.",
      hypothesis="A per-token auxiliary loss attached mid-trunk, plus letting the crash "
                 "head itself train (A1's recipe freezes it), should let the model use "
                 "spatial car-presence information the frozen A1-compress256 recipe "
                 "cannot act on.",
      aim="Isolate 'the trunk knows where cars are' from 'the trunk knows which car "
          "matters' (AA-rel-unfrozen, below) — the two arms share every setting except "
          "the aux target, so any gap between them is attributable to relevance, not to "
          "unfreezing the head or to the aux loss existing at all.",
      method="A1-compress256's exact recipe (LoRA r=16 α=32 on query/key/value, lr 2e-4 "
             "constant, compress256, pool1761, seed 0) PLUS: (1) the crash head "
             "(attention-pool + MLP classifier) unfrozen at the same 2e-4 LR, and (2) a "
             "second loss term read from encoder layer 17 through a FROZEN linear probe "
             "(fit once on the untrained trunk, never updated here): balanced BCE against "
             "a per-patch-token label ('does any tracked vehicle cover this patch', from "
             "an offline YOLOPv2+BoT-SORT detection pass). Backprop from this term reaches "
             "only the LoRA in layers 0-17 — layers 18-23 and the predictor stack still "
             "learn from crash CE alone. lambda=0.59, sized so the aux term's gradient on "
             "the shared LoRA layers is ~10% of the crash gradient's (measured directly, "
             "not guessed). Only 3 of the usual 8 epochs were run — a quick check to see "
             "whether unfreezing the head does ANYTHING before committing a full 8-epoch "
             "pair, not the final recipe. CAUTION comparing to A1-compress256 above: that "
             "arm keeps the head frozen, so the honest baseline for this one is the "
             "matched no-aux control, AA-ctrl-unfrozen (rank-1 test AP 0.9070, not on this "
             "page) — against A1-compress256 directly, two things changed at once "
             "(unfreezing + the aux loss).",
      prompt=None,
      prompt_note="No language supervision in this arm — the aux target is a per-token "
                  "geometric label from detection, not a caption.",
      pool="pool1761", train_dir=AAT / "AA-occ-unfrozen-check3",
      arch=dict(semantic=False, loss=True, state={"tproc": "train", "clsf": "train"},
               note="crash head (temporal processor + classifier) UNFROZEN, training at "
                    "the same 2e-4 LR as LoRA — every other arm on this page keeps it "
                    "frozen"),
      hyper=AAT / "AA-occ-unfrozen-check3" / "train_metrics.json", hyper_note=None,
      test=dict(path=AAT / "AA-occ-unfrozen-check3" / "test_results_ep02.jsonl",
                gt_key="ground_truth",
                summary=AAT / "AA-occ-unfrozen-check3" / "test_summary.json", epoch=2,
                by_epoch=[(n, AAT / "AA-occ-unfrozen-check3" / f"test_results_ep{n:02d}.jsonl")
                          for n in range(1, 4)],
                source="aa_token_aux/AA-occ-unfrozen-check3 (epoch 2)")),

    A(key="AA-rel-unfrozen", label="AA-rel-unfrozen · relevance aux, unfrozen head",
      order=11, family="pool1761",
      tagline="Does teaching the trunk WHICH car matters (not just that one is present) "
              "improve crash prediction, once the head can actually use it?",
      hypothesis="A frozen-head run of this same idea (aux loss at layer 17, lambda sized "
                 "the same way) measurably changed layer-17 token readability for "
                 "relevance more than its own occupancy control did, but that change "
                 "never reached the final prediction (0.99 score correlation with "
                 "A1-compress256, ~16/677 label flips at threshold 0.5 — smaller than the "
                 "drift between two ordinary training epochs of the same arm). Unfreezing "
                 "the crash head, so it can actually act on relevance information instead "
                 "of a fixed frozen readout, should let that already-real internal change "
                 "surface as a prediction difference.",
      aim="Test the head-freezing explanation directly, as the other lever (moving the "
          "aux tap from layer 17 to layer 23, closer to the output) already failed to "
          "produce it. Compare against AA-occ-unfrozen above, its exact-recipe control, "
          "to attribute any gap to relevance specifically rather than to the aux loss or "
          "the unfrozen head in general.",
      method="Identical to AA-occ-unfrozen's recipe (see that arm's method for the full "
             "unfrozen-head + aux-loss description) with two changes: the per-token label "
             "is 'how relevant is this car' (0-1, continuous — a kinematic selection "
             "score: closeness x ego-lane position x approach rate, computed offline and "
             "never fit to the crash label) instead of binary occupancy, and lambda=2.43 "
             "(sized the same way — ~10% relative gradient pull — but larger because the "
             "relevance target's own gradient is smaller in magnitude). Same 3-epoch "
             "quick-check caveat and the same CAUTION about comparing to A1-compress256 "
             "directly (see AA-occ-unfrozen's method) — the fair baseline here is "
             "AA-ctrl-unfrozen (0.9070) or AA-occ-unfrozen (0.9101) above, not "
             "A1-compress256 (0.9128, frozen head).",
      prompt=None,
      prompt_note="No language supervision in this arm — the aux target is a per-token "
                  "kinematic label from detection, not a caption.",
      pool="pool1761", train_dir=AAT / "AA-rel-unfrozen-check3",
      arch=dict(semantic=False, loss=True, state={"tproc": "train", "clsf": "train"},
               note="crash head UNFROZEN, same as AA-occ-unfrozen — the two arms differ "
                    "only in the aux target"),
      hyper=AAT / "AA-rel-unfrozen-check3" / "train_metrics.json", hyper_note=None,
      test=dict(path=AAT / "AA-rel-unfrozen-check3" / "test_results_ep02.jsonl",
                gt_key="ground_truth",
                summary=AAT / "AA-rel-unfrozen-check3" / "test_summary.json", epoch=2,
                by_epoch=[(n, AAT / "AA-rel-unfrozen-check3" / f"test_results_ep{n:02d}.jsonl")
                          for n in range(1, 4)],
                source="aa_token_aux/AA-rel-unfrozen-check3 (epoch 2)")),

    # ---------------------------------------------------------------------------------
    # Stage AA: original frozen-head side-probe arms (2026-09-13/24). These ran BEFORE
    # AA-occ-unfrozen/AA-rel-unfrozen above and are why the head was unfrozen at all -
    # order placed just before them (9.1-9.4) so the page reads in the order the work
    # actually happened, not upload order.
    A(key="AA-rel", label="AA-rel · relevance aux, frozen head (layer 17)", order=9.1,
      family="pool1761",
      tagline="Does a per-token 'which car matters' loss on the trunk change the final "
              "crash prediction, with the crash head kept exactly as A1-compress256 froze it?",
      hypothesis="A per-token auxiliary loss teaching the trunk which detected car is "
                 "relevant (closeness x ego-lane position x approach rate, from an offline "
                 "YOLOPv2+tracking pass, never fit to the crash label) should sharpen the "
                 "features the frozen crash head reads, and improve test AP.",
      aim="The first test of the whole token-relevance idea, at A1-compress256's exact "
          "recipe otherwise, so any effect is attributable to the aux loss alone.",
      method="A1-compress256's exact recipe (LoRA r=16 α=32 on query/key/value, lr 2e-4 "
             "constant, compress256, pool1761, seed 0), crash head FROZEN, plus a second "
             "loss term read from encoder layer 17 through a frozen linear probe (fit "
             "once on the untrained trunk): balanced BCE against the continuous relevance "
             "label. Backprop from this term reaches only the LoRA in layers 0-17. "
             "lambda=3.6, sized so the aux gradient on shared LoRA layers is ~10% of the "
             "crash gradient's.",
      prompt=None,
      prompt_note="No language supervision - the aux target is a per-token kinematic "
                  "label from detection, not a caption.",
      pool="pool1761", train_dir=AAT / "AA-rel",
      media={"video": "assets/media/aa1_detection_example.webm",
             "image": "assets/media/aa1_detection_curves.png",
             "caption": "Video 00283: the offline YOLOPv2 detection + tracking pass "
                        "this arm's relevance label is computed from - boxes, ego-path "
                        "corridor and the per-track relevance/threat curve over time. "
                        "This runs once, offline, before training; the aux loss only "
                        "ever sees its output (a 0-1 label per patch token), never the "
                        "video or detector at train time."},
      arch=dict(semantic=False, loss=True, state={},
               auxBranch={"kind": "probe", "layer": 17, "target": "relevance (0-1)"},
               note="frozen head (unlike AA-rel-unfrozen below) - the aux loss changed "
                    "layer-17 readability for relevance but the effect never reached the "
                    "final prediction: 0.99 score correlation with A1-compress256, ~16/677 "
                    "label flips at threshold 0.5 - smaller than the drift between two "
                    "ordinary epochs of the same arm. Verdict: negative, not propagated."),
      hyper=AAT / "AA-rel" / "train_metrics.json", hyper_note=None,
      test=dict(path=AAT / "AA-rel" / "test_results_ep03.jsonl", gt_key="ground_truth",
                summary=AAT / "AA-rel" / "test_summary.json", epoch=3,
                by_epoch=[(n, AAT / "AA-rel" / f"test_results_ep{n:02d}.jsonl")
                          for n in range(1, 9)],
                source="aa_token_aux/AA-rel (epoch 3)")),

    A(key="AA-occ", label="AA-occ · occupancy aux, frozen head (layer 17)", order=9.2,
      family="pool1761",
      tagline="Localisation-only control for AA-rel - teach the trunk 'a car is here', "
              "nothing about which car matters.",
      hypothesis="If AA-rel's aux loss helps, it should help BECAUSE of relevance "
                 "specifically, not merely because any per-token supervision regularises "
                 "the trunk. A binary 'is any tracked vehicle here' target isolates that.",
      aim="Give AA-rel a same-recipe control that shares everything except which "
          "question the per-token label answers, so any gap between the two arms is "
          "attributable to relevance, not to the aux loss existing at all.",
      method="Identical to AA-rel's recipe (layer 17, frozen head, same lambda-sizing "
             "procedure) except the per-token label is binary occupancy (does any "
             "tracked vehicle cover this patch) instead of the continuous relevance "
             "score. lambda=0.59 (different target, so the 10%-gradient sizing lands on "
             "a different absolute number).",
      prompt=None,
      prompt_note="No language supervision - the aux target is a per-token geometric "
                  "label from detection, not a caption.",
      pool="pool1761", train_dir=AAT / "AA-occ",
      arch=dict(semantic=False, loss=True, state={},
               auxBranch={"kind": "probe", "layer": 17, "target": "occupancy (0/1)"},
               note="frozen head - same non-propagation result as AA-rel (see that arm)."),
      hyper=AAT / "AA-occ" / "train_metrics.json", hyper_note=None,
      test=dict(path=AAT / "AA-occ" / "test_results_ep03.jsonl", gt_key="ground_truth",
                summary=AAT / "AA-occ" / "test_summary.json", epoch=3,
                by_epoch=[(n, AAT / "AA-occ" / f"test_results_ep{n:02d}.jsonl")
                          for n in range(1, 9)],
                source="aa_token_aux/AA-occ (epoch 3)")),

    A(key="AA-rel-L23", label="AA-rel-L23 · relevance aux, frozen head (layer 23)", order=9.3,
      family="pool1761",
      tagline="Moving the tap closer to the crash head's own input - does that alone fix "
              "the non-propagation seen at layer 17?",
      hypothesis="AA-rel's aux signal might be real at layer 17 but dilute across "
                 "layers 18-23 before reaching the crash head. Reading the probe at "
                 "layer 23 instead (one layer before the head) should let more of it "
                 "survive if dilution is the explanation.",
      aim="Test the 'wrong tap depth' explanation directly, holding the aux target "
          "(relevance) and everything else fixed.",
      method="AA-rel's exact recipe with --aux-layer 23 instead of 17 (a new frozen "
             "probe fit at layer 23 on the untrained trunk). lambda=2.76.",
      prompt=None,
      prompt_note="No language supervision - same as AA-rel.",
      pool="pool1761", train_dir=AAT / "AA-rel-L23",
      arch=dict(semantic=False, loss=True, state={},
               auxBranch={"kind": "probe", "layer": 23, "target": "relevance (0-1)"},
               note="Moving the tap did not fix propagation either - see this arm's test "
                    "result vs A1-compress256/AA-rel. Ruled out as a standalone fix."),
      hyper=AAT / "AA-rel-L23" / "train_metrics.json", hyper_note=None,
      test=dict(path=AAT / "AA-rel-L23" / "test_results_ep08.jsonl", gt_key="ground_truth",
                summary=AAT / "AA-rel-L23" / "test_summary.json", epoch=8,
                by_epoch=[(n, AAT / "AA-rel-L23" / f"test_results_ep{n:02d}.jsonl")
                          for n in range(1, 9)],
                source="aa_token_aux/AA-rel-L23 (epoch 8)")),

    A(key="AA-ctrl-unfrozen", label="AA-ctrl-unfrozen · unfrozen head, no aux loss (control)",
      order=9.4, family="pool1761",
      tagline="The honest baseline for AA-occ-unfrozen/AA-rel-unfrozen: what does "
              "unfreezing the crash head do BY ITSELF, with no aux loss at all?",
      hypothesis="None - this is a control, not a treatment. It isolates 'unfreezing the "
                 "head' from 'unfreezing the head AND adding an aux loss'.",
      aim="Without this arm, AA-occ-unfrozen/AA-rel-unfrozen could only be compared "
          "against A1-compress256 (frozen head) - a comparison that changes two things "
          "at once. This arm changes only the head.",
      method="A1-compress256's exact recipe, crash head UNFROZEN at the same 2e-4 LR, "
             "aux_mode=none (no second loss term at all). Same 3-epoch quick-check as "
             "the two aux arms it controls for.",
      prompt=None, prompt_note="No language supervision in this arm.",
      pool="pool1761", train_dir=AAT / "AA-ctrl-unfrozen-check3",
      arch=dict(semantic=False, loss=True, state={"tproc": "train", "clsf": "train"},
               note="crash head UNFROZEN, no aux loss - the control AA-occ-unfrozen and "
                    "AA-rel-unfrozen above are read against, not A1-compress256."),
      hyper=AAT / "AA-ctrl-unfrozen-check3" / "train_metrics.json", hyper_note=None,
      test=dict(path=AAT / "AA-ctrl-unfrozen-check3" / "test_results_ep02.jsonl",
                gt_key="ground_truth",
                summary=AAT / "AA-ctrl-unfrozen-check3" / "test_summary.json", epoch=2,
                by_epoch=[(n, AAT / "AA-ctrl-unfrozen-check3" / f"test_results_ep{n:02d}.jsonl")
                          for n in range(1, 4)],
                source="aa_token_aux/AA-ctrl-unfrozen-check3 (epoch 2)")),

    # ---------------------------------------------------------------------------------
    # Stage AA-H: supervise the crash head's OWN attention instead of a side probe on an
    # intermediate layer (2026-09-24/26). Literature: RARE (attn_rank), FAX (attn_mass),
    # GAIN/CAMAL (gradcam, never run - see DECISIONS.md). Same A1-compress256 recipe and
    # split-seed throughout; only init_seed and the aux loss vary.
    A(key="AA-ctrl-seed1", label="AA-ctrl-seed1 · noise floor, init_seed=1", order=12,
      family="pool1761",
      tagline="How much does test AP move from LoRA-init randomness alone, holding the "
              "split fixed? Half of the Stage-1 noise-floor measurement.",
      hypothesis="None - this is a noise-floor control. Two re-runs of A1-compress256 "
                 "with only the init_seed changed bound how big a change has to be before "
                 "it means anything, for every Stage AA-H arm below.",
      aim="Give the AA-H screen a real yardstick before judging any aux-loss arm: a "
          "result within this range is a re-roll of the random init, not an effect.",
      method="A1-compress256's exact recipe and split_seed=0 (identical train/val "
             "partition), only --init-seed 1 / --seed 1 changed.",
      prompt=None, prompt_note="No language supervision - crash-only, same as A1-compress256.",
      pool="pool1761", train_dir=AAH / "AA-ctrl-seed1" / "train",
      public=dict(jsonl=AAH / "AA-ctrl-seed1" / "scores_public" / "AA-ctrl-seed1.jsonl",
                  metrics=AAH / "AA-ctrl-seed1" / "scores_public" / "AA-ctrl-seed1.metrics.json"),
      arch=dict(semantic=False, loss=True, state={},
               note="identical recipe to A1-compress256 - only the LoRA init/seed differ."),
      hyper=AAH / "AA-ctrl-seed1" / "train" / "train_metrics.json", hyper_note=None,
      # No summary= override: this run's live (unrounded) evaluation and its rounded
      # per-clip dump disagree by >1e-3 on f1 at epoch 4 (a genuine near-threshold score
      # that rounds across 0.5) - test_block's own identity gate catches this, so the
      # dump's own recomputed metrics are used throughout instead of the published ones.
      test=dict(path=AAH / "AA-ctrl-seed1" / "train" / "test_results_ep04.jsonl",
                gt_key="ground_truth", epoch=4,
                by_epoch=[(n, AAH / "AA-ctrl-seed1" / "train" / f"test_results_ep{n:02d}.jsonl")
                          for n in range(1, 9)],
                source="aa_head_attn/AA-ctrl-seed1/train (epoch 4, dump-recomputed — see note)")),

    A(key="AA-ctrl-seed2", label="AA-ctrl-seed2 · noise floor, init_seed=2", order=13,
      family="pool1761",
      tagline="The other half of the Stage-1 noise-floor pair.",
      hypothesis="None - noise-floor control, see AA-ctrl-seed1.",
      aim="Same as AA-ctrl-seed1; two seeds (not one) so the floor is a range, not a "
          "single lucky/unlucky draw.",
      method="A1-compress256's exact recipe and split_seed=0, only --init-seed 2 / "
             "--seed 2 changed.",
      prompt=None, prompt_note="No language supervision - crash-only, same as A1-compress256.",
      pool="pool1761", train_dir=AAH / "AA-ctrl-seed2" / "train",
      public=dict(jsonl=AAH / "AA-ctrl-seed2" / "scores_public" / "AA-ctrl-seed2.jsonl",
                  metrics=AAH / "AA-ctrl-seed2" / "scores_public" / "AA-ctrl-seed2.metrics.json"),
      arch=dict(semantic=False, loss=True, state={},
               note="identical recipe to A1-compress256 - only the LoRA init/seed differ."),
      hyper=AAH / "AA-ctrl-seed2" / "train" / "train_metrics.json", hyper_note=None,
      test=dict(path=AAH / "AA-ctrl-seed2" / "train" / "test_results_ep01.jsonl",
                gt_key="ground_truth",
                summary=AAH / "AA-ctrl-seed2" / "train" / "test_summary.json", epoch=1,
                by_epoch=[(n, AAH / "AA-ctrl-seed2" / "train" / f"test_results_ep{n:02d}.jsonl")
                          for n in range(1, 9)],
                source="aa_head_attn/AA-ctrl-seed2/train (epoch 1)")),

    A(key="AA-H-rank-R_all", label="AA-H-rank-R_all · attn_rank aux, R on every clip",
      order=14, family="pool1761",
      tagline="RARE-style ranking loss on the crash head's own attention: rank the "
              "relevant car's tokens above other cars and background, every window.",
      hypothesis="A side probe never reached the crash head's decision (see AA-rel "
                 "above). Supervising the head's OWN attention instead should reach it "
                 "by construction - the aux gradient flows through the same path the "
                 "crash loss does.",
      aim="Screen attn_rank against the R_all label (the existing relevance score, "
          "applied unconditionally) as the first of 3 label variants, against the "
          "Stage-1 noise floor (0.8910-0.8958 mean-over-8) rather than a single control.",
      method="A1-compress256's exact recipe, crash head frozen, plus a rank/margin loss "
             "(RARE) on the attention weights of the crash head's own temporal-processor "
             "MultiheadAttention: relevant-car tokens must out-rank other-vehicle and "
             "background tokens by a log-ratio margin of 0.5 (a thin-sample "
             "approximation - see DECISIONS.md). lambda=0.1645, sized to a 30% "
             "gradient-norm pull via a 1-epoch pilot. Ramped on epochs 1-3, off 4-8 "
             "(warm_on_off, matching HASTE/REPA's early-stop finding).",
      prompt=None, prompt_note="No language supervision - the aux target is the crash "
                  "head's own attention distribution, not a caption.",
      pool="pool1761", train_dir=AAH / "AA-H-rank-R_all" / "train",
      public=dict(jsonl=AAH / "AA-H-rank-R_all" / "scores_public" / "AA-H-rank-R_all.jsonl",
                  metrics=AAH / "AA-H-rank-R_all" / "scores_public" / "AA-H-rank-R_all.metrics.json"),
      arch=dict(semantic=False, loss=True, state={},
               auxBranch={"kind": "head", "mode": "attn_rank", "label": "R_all"},
               note="Attention moved as designed (rho_P/V, rho_P/B both rose) but AP "
                    "gain is marginal (mean-over-8 0.8991 vs control 0.8910-0.8958, both on the "
                    "published-summary basis) "
                    "and, on the pooled 1,344-clip re-analysis against the same-seed "
                    "control, statistically a null result once seed variance is "
                    "accounted for - see EXPERIMENTS.md's 2026-09-26 re-analysis."),
      hyper=AAH / "AA-H-rank-R_all" / "train" / "train_metrics.json", hyper_note=None,
      test=dict(path=AAH / "AA-H-rank-R_all" / "train" / "test_results_ep02.jsonl",
                gt_key="ground_truth",
                summary=AAH / "AA-H-rank-R_all" / "train" / "test_summary.json", epoch=2,
                by_epoch=[(n, AAH / "AA-H-rank-R_all" / "train" / f"test_results_ep{n:02d}.jsonl")
                          for n in range(1, 9)],
                source="aa_head_attn/AA-H-rank-R_all/train (epoch 2)")),

    A(key="AA-H-rank-R_pos", label="AA-H-rank-R_pos · attn_rank aux, R on crash clips only",
      order=15, family="pool1761",
      tagline="Same ranking loss, but the aux term is skipped on negative clips - does "
              "restricting it to windows with a real threat change anything?",
      hypothesis="The relevance score R answers 'which car might matter', not 'is a "
                 "crash coming' - a car merely close in normal traffic scores the same "
                 "as a real threat. Restricting the aux term to crash clips (where R's "
                 "target really is the threat) should avoid teaching 'close car -> "
                 "alarm' on ordinary negatives.",
      aim="Test whether label precision (which windows count as R) fixes attn_rank's "
          "weak result, holding the loss and every hyperparameter procedure fixed.",
      method="Identical to AA-H-rank-R_all except the aux loss is skipped entirely on "
             "negative-clip windows (~half the pool) - R only supervises attention on "
             "crash clips. lambda=0.6965 (its own pilot; skipping half the pool changes "
             "the gradient-norm ratio the pilot targets).",
      prompt=None, prompt_note="No language supervision - same as AA-H-rank-R_all.",
      pool="pool1761", train_dir=AAH / "AA-H-rank-R_pos" / "train",
      public=dict(jsonl=AAH / "AA-H-rank-R_pos" / "scores_public" / "AA-H-rank-R_pos.jsonl",
                  metrics=AAH / "AA-H-rank-R_pos" / "scores_public" / "AA-H-rank-R_pos.metrics.json"),
      arch=dict(semantic=False, loss=True, state={},
               auxBranch={"kind": "head", "mode": "attn_rank", "label": "R_pos"},
               note="Best mean-over-8 of the attn_rank screen (0.9018, published-summary "
                    "basis) and, on the "
                    "pooled 1,344-clip re-analysis at matched recall, no measurable FP "
                    "change vs its same-seed control - a null result, not a win or a "
                    "regression. See EXPERIMENTS.md's 2026-09-26 re-analysis."),
      hyper=AAH / "AA-H-rank-R_pos" / "train" / "train_metrics.json", hyper_note=None,
      test=dict(path=AAH / "AA-H-rank-R_pos" / "train" / "test_results_ep02.jsonl",
                gt_key="ground_truth",
                summary=AAH / "AA-H-rank-R_pos" / "train" / "test_summary.json", epoch=2,
                by_epoch=[(n, AAH / "AA-H-rank-R_pos" / "train" / f"test_results_ep{n:02d}.jsonl")
                          for n in range(1, 9)],
                source="aa_head_attn/AA-H-rank-R_pos/train (epoch 2)")),

    A(key="AA-H-rank-partner_pos",
      label="AA-H-rank-partner_pos · attn_rank aux, hindsight collision-partner label",
      order=16, family="pool1761",
      tagline="Replace the proximity-based relevance score with the actual car the ego "
              "collided with, found by decoding a moment past the crash.",
      hypothesis="R_pos still answers 'which car is near/entering the lane', not 'which "
                 "car did we hit'. A hindsight label - the literal collision partner, "
                 "found by tracking objects through the impact moment - removes that "
                 "ambiguity entirely, so if attention placement is the bottleneck, this "
                 "label should show the clearest gain.",
      aim="The most direct test of the plan's original FP concern: does the most "
          "precise possible 'relevant object' label fix what R_all/R_pos could not?",
      method="Identical to AA-H-rank-R_pos except the label is `partner_pos`: for each "
             "crash clip, the video is decoded ~0.8s past the cached span, detected and "
             "tracked (YOLOPv2 + a greedy IoU tracker), and the partner is the track "
             "re-identified into the original window with the largest box near the end "
             "of the extension. 543 positive videos processed, 498 found a partner "
             "(88.6% window label rate). lambda=0.5036 (own pilot).",
      prompt=None, prompt_note="No language supervision - same as AA-H-rank-R_all.",
      pool="pool1761", train_dir=AAH / "AA-H-rank-partner_pos" / "train",
      public=dict(jsonl=AAH / "AA-H-rank-partner_pos" / "scores_public" / "AA-H-rank-partner_pos.jsonl",
                  metrics=AAH / "AA-H-rank-partner_pos" / "scores_public" / "AA-H-rank-partner_pos.metrics.json"),
      arch=dict(semantic=False, loss=True, state={},
               auxBranch={"kind": "head", "mode": "attn_rank", "label": "partner_pos"},
               note="The most precise label gave the LEAST AP lift of the three (mean-"
                    "over-8 0.8951 on the published basis, inside the control range) - its "
                    "rho_P/V (attention "
                    "vs other vehicles) barely moved; its gain was entirely attention vs "
                    "background. Never learned to prefer the true partner over other "
                    "nearby traffic."),
      hyper=AAH / "AA-H-rank-partner_pos" / "train" / "train_metrics.json", hyper_note=None,
      test=dict(path=AAH / "AA-H-rank-partner_pos" / "train" / "test_results_ep02.jsonl",
                gt_key="ground_truth",
                summary=AAH / "AA-H-rank-partner_pos" / "train" / "test_summary.json", epoch=2,
                by_epoch=[(n, AAH / "AA-H-rank-partner_pos" / "train" / f"test_results_ep{n:02d}.jsonl")
                          for n in range(1, 9)],
                source="aa_head_attn/AA-H-rank-partner_pos/train (epoch 2)")),

    A(key="AA-H-mass-R_pos", label="AA-H-mass-R_pos · attn_mass aux (FAX-style), R_pos label",
      order=17, family="pool1761",
      tagline="A mechanistically different loss from attn_rank: maximise the SHARE of "
              "the head's total attention on the relevant car, not just its rank.",
      hypothesis="attn_rank is a margin loss - it stops pushing once the relevant car "
                 "out-ranks the rest, so it can be satisfied cheaply without a real "
                 "reallocation of attention. attn_mass (FAX) instead maximises "
                 "-ln(sum of attention on relevant tokens), which cannot be satisfied "
                 "that way - it should force a genuinely larger share of attention onto "
                 "the relevant object, and might succeed where attn_rank's weaker push "
                 "did not.",
      aim="Test a mechanistically different loss family with the best-performing label "
          "from the attn_rank screen (R_pos), after attn_rank itself showed no clear "
          "effect.",
      method="Same recipe as AA-H-rank-R_pos with --aux-mode attn_mass instead of "
             "attn_rank: the loss is -ln(sum of attention probability on relevant "
             "tokens), guarded by an entropy monitor against full collapse. "
             "lambda=0.5518 (own pilot, same 30%-gradient-norm target), same "
             "warm_on_off schedule.",
      prompt=None, prompt_note="No language supervision - same as AA-H-rank-R_all.",
      pool="pool1761", train_dir=AAH / "AA-H-mass-R_pos" / "train",
      public=dict(jsonl=AAH / "AA-H-mass-R_pos" / "scores_public" / "AA-H-mass-R_pos.jsonl",
                  metrics=AAH / "AA-H-mass-R_pos" / "scores_public" / "AA-H-mass-R_pos.metrics.json"),
      arch=dict(semantic=False, loss=True, state={},
               auxBranch={"kind": "head", "mode": "attn_mass", "label": "R_pos"},
               note="The mechanism prediction was right - rho_P/V reaches 195.85 (vs "
                    "attn_rank's ~4.6 peak) and barely unwinds after the loss switches "
                    "off - but val_ap picked epoch 8, the checkpoint with the WORST "
                    "private test AP of all 8 (0.8841). Epoch 4 (the genuinely best "
                    "checkpoint) scores the same AP as control at matched recall - the "
                    "large FP count at threshold 0.5 there is a calibration shift, not "
                    "worse ranking. Only the val-selected epoch 8 is a real regression."),
      hyper=AAH / "AA-H-mass-R_pos" / "train" / "train_metrics.json", hyper_note=None,
      test=dict(path=AAH / "AA-H-mass-R_pos" / "train" / "test_results_ep08.jsonl",
                gt_key="ground_truth",
                summary=AAH / "AA-H-mass-R_pos" / "train" / "test_summary.json", epoch=8,
                by_epoch=[(n, AAH / "AA-H-mass-R_pos" / "train" / f"test_results_ep{n:02d}.jsonl")
                          for n in range(1, 9)],
                source="aa_head_attn/AA-H-mass-R_pos/train (epoch 8, val-selected)")),
]


# ------------------------------------------------------------------------ train section
def train_block(arm):
    d = arm.get("train_dir")
    if d is None:
        return {"available": False,
                "note": "A0 is never trained, so there are no training curves, no "
                        "checkpoints and no epoch to select."}
    em = d / "epoch_metrics.jsonl"
    if not em.exists():
        return {"available": False, "note": f"No epoch_metrics.jsonl under {d.name}."}
    rows = load_jsonl(em)
    epochs = [r["epoch"] for r in rows]

    def series(key):
        vals = [r.get(key) for r in rows]
        return None if all(v is None for v in vals) else vals

    out = {
        "available": True,
        "epochs": epochs,
        "series": {k: v for k, v in {
            "train_total": series("train_total_loss"),
            "val_total": series("val_total_loss"),
            "train_crash": series("crash_loss"),
            "val_crash": series("val_crash_loss"),
            "train_sem": series("sem_loss"),
            "val_sem": series("val_sem_loss"),
            "val_ap": series("val_ap"),
            "train_val_gap": series("train_val_gap"),
            "lr": series("lr"),
            "grad_cos": series("grad_cos_mean"),
            "train_aux": series("aux_loss"),
            "aux_grad_cos": series("aux_grad_cos_mean"),
            "aux_h_rho_pv": series("aux_h_rho_pv"),
            "aux_h_rho_pb": series("aux_h_rho_pb"),
            "aux_h_entropy": series("aux_h_entropy"),
            "aux_h_lambda": series("aux_h_lambda_last"),
        }.items() if v is not None},
        "selection_metric": rows[0].get("select_by") or "val_ap",
        "note": arm.get("train_note"),
        "acc_note": "Accuracy and ROC-AUC were never logged per epoch by the trainer — "
                    "only val AP, which is the criterion that selects the checkpoint.",
    }
    # A semantic arm whose sem_loss is all-zero is really a crash-only run; don't plot a
    # flat zero line and imply a semantic branch was active.
    if out["series"].get("train_sem") and not any(out["series"]["train_sem"]):
        out["series"].pop("train_sem", None)
        out["series"].pop("val_sem", None)

    sel = arm.get("test", {}) or {}
    out["selected_epoch"] = sel.get("epoch")

    # Per-epoch val scores exist only for the a1fail321 family (--dump-val-scores post-dates
    # the pool-1761 runs), so val accuracy / AUC / ROC are derivable there and nowhere else.
    dumps = sorted(d.glob("val_scores_ep*.jsonl"))
    if dumps:
        per_epoch, curves = {}, {"epochs": [], "val_acc": [], "val_auc": [], "val_ap": []}
        for p in dumps:
            ep = int(p.stem.split("ep")[-1])
            vr = load_jsonl(p)
            y = [int(r["label"]) for r in vr]
            s = [float(r["score"]) for r in vr]
            if len(set(y)) < 2:
                continue
            m = metrics_from_arrays(y, s, threshold=THRESHOLD)
            per_epoch[ep] = {"metrics": m, "roc": roc_points(y, s), "n": len(y)}
            curves["epochs"].append(ep)
            curves["val_acc"].append(m["accuracy"])
            curves["val_auc"].append(m["auc_roc"])
            curves["val_ap"].append(m["ap"])
        if per_epoch:
            best = out["selected_epoch"] if out["selected_epoch"] in per_epoch \
                else max(per_epoch, key=lambda e: per_epoch[e]["metrics"]["ap"] or 0)
            out["val_eval"] = {
                "available": True, "epoch": best, "n": per_epoch[best]["n"],
                "metrics": per_epoch[best]["metrics"], "roc": per_epoch[best]["roc"],
                "curves": curves, "threshold": THRESHOLD,
                "note": "Derived here from this arm's per-epoch val score dumps — the "
                        "trainer itself logged only val AP.",
            }
    if "val_eval" not in out:
        out["val_eval"] = {
            "available": False,
            "note": "No per-clip validation scores were dumped for this run (it predates "
                    "--dump-val-scores, or ran with it off), so no ROC curve and no "
                    "confusion matrix can be drawn for its training phase. Only the loss "
                    "and val-AP curves above survive.",
        }
    out["train_roc_note"] = ("There is no train-split ROC for any arm in this project — "
                             "the trainer never dumps per-example scores for the training "
                             "split, only aggregate loss.")
    return out


# ------------------------------------------------------------------------- test section
def test_metrics_bundle(y, s, groups, test_group_by_vid=None):
    """Full metrics + ROC for the whole set and for each TTE bucket, so the page's mode
    selector can switch between them without recomputing anything in the browser."""
    bundle = {"all": {"metrics": metrics_from_arrays(y, s, groups=groups,
                                                     threshold=THRESHOLD),
                      "roc": roc_points(y, s), "n": len(y)}}
    for g, name in GROUP_LABEL.items():
        idx = [i for i, gg in enumerate(groups) if gg == g]
        yy = [y[i] for i in idx]
        ss = [s[i] for i in idx]
        if len(set(yy)) < 2:
            continue
        bundle[name] = {"metrics": metrics_from_arrays(yy, ss, threshold=THRESHOLD),
                        "roc": roc_points(yy, ss), "n": len(yy)}
    return bundle


def test_block(arm, group_by_vid):
    cfg = arm.get("test")
    if not cfg:
        return {"available": False, "note": arm.get("test_note")}
    rows = load_jsonl(cfg["path"])
    y = [to01(r[cfg["gt_key"]]) for r in rows]
    s = [float(r["score"]) for r in rows]
    # Always take the horizon from the manifest rather than the dump: the a1fail321
    # score files carry no group field at all, and joining by video_id makes every arm
    # use one definition of the buckets.
    missing = [r["video_id"] for r in rows if r["video_id"] not in group_by_vid]
    assert not missing, f"{arm['key']}: {len(missing)} test rows not in the manifest"
    groups = [group_by_vid[r["video_id"]] for r in rows]
    assert len(rows) == 677, f"{arm['key']}: expected 677 test rows, got {len(rows)}"

    bundle = test_metrics_bundle(y, s, groups)
    m = bundle["all"]["metrics"]
    ap, auc, published = m["ap"], m["auc_roc"], None

    # EVERY arm gets its confusion matrix checked against EXPECTED_CM, not only the four
    # with a test_summary.json (project review §5.2 - previously six arms had zero drift
    # protection on this page even though build_landing_data.py pins the same numbers).
    exp = EXPECTED_CM.get(arm["key"])
    assert exp is not None, \
        f"{arm['key']}: no entry in EXPECTED_CM - add one (see build_landing_data.py's " \
        f"EXPECTED for the paste-ready values) before this arm can ship on the page."
    got = dict(n=m["n_total"], tp=m["tp"], fn=m["fn"], fp=m["fp"], tn=m["tn"])
    assert got == exp, \
        f"{arm['key']}: confusion matrix drifted from the expected/verified values.\n" \
        f"  expected {exp}\n  got      {got}\n  (source: {cfg['path']})"

    if cfg.get("summary"):
        best = json.load(open(cfg["summary"], encoding="utf-8"))["checkpoints"]
        entry = next((c for c in best if c["epoch"] == cfg["epoch"]), None)
        assert entry, f"{arm['key']}: epoch {cfg['epoch']} not in {cfg['summary'].name}"
        # Identity gate: these three are reproducible from the rounded dump. If they
        # disagree, the summary describes a different checkpoint and its AP/AUC must not
        # be pasted onto this dump's confusion matrix.
        for key, mine in (("f1", m["f1"]), ("recall", m["recall_sensitivity_tpr"]),
                          ("specificity", m["specificity_tnr"])):
            assert abs(entry[key] - mine) < 1e-3, \
                f"{arm['key']}: {cfg['summary'].name} {key}={entry[key]} disagrees with " \
                f"the per-clip dump ({mine}) — not the same checkpoint."
        published = {"ap": round(float(entry["test_ap"]), 4),
                     "auc": round(float(entry["auc_roc"]), 4),
                     "f1_optimal": entry.get("f1_optimal"),
                     "brier": entry.get("brier"), "ece": entry.get("ece")}
        ap, auc = published["ap"], published["auc"]
        # The run also published per-horizon AP. Carry it so the bucket rows quote the
        # same source as the headline row - otherwise "all" would read published and the
        # buckets recomputed, and a reader checking against the run report would find a
        # mismatch in one place but not the other.
        per = entry.get("per_tte_ap") or {}
        for lab, blk in per.items():
            if lab in bundle:
                assert blk["n"] == bundle[lab]["n"], \
                    f"{arm['key']}: {lab} n={blk['n']} in the summary vs " \
                    f"{bundle[lab]['n']} in the dump"
                bundle[lab]["published_ap"] = blk["ap"]

    out = {
        "available": True, "epoch": cfg.get("epoch"), "source": cfg["source"],
        "threshold": THRESHOLD, "n": len(rows),
        "buckets": bundle, "ap": ap, "auc": auc, "published": published,
    }
    # The curve is drawn from the rounded dump; say so when that visibly disagrees with
    # the published AUC rather than letting the reader assume the chart is the number.
    if published and abs(bundle["all"]["metrics"]["auc_roc"] - published["auc"]) > 1e-3:
        out["roc_note"] = (
            f"AP/AUC above are this run's published values. The curve is drawn from the "
            f"per-clip dump, whose scores are stored rounded to 4 decimals; recomputed "
            f"from it the same curve reads AP {bundle['all']['metrics']['ap']:.4f} / AUC "
            f"{bundle['all']['metrics']['auc_roc']:.4f}. The confusion matrix is "
            f"unaffected — it agrees exactly.")

    by_epoch = []
    for ep, p in cfg.get("by_epoch", []) or []:
        if not p.exists():
            continue
        er = load_jsonl(p)
        if len(er) != 677:          # b_1761_par/test_results_ep07.jsonl is truncated at 522
            continue
        em = metrics_from_arrays([to01(r[cfg["gt_key"]]) for r in er],
                                 [float(r["score"]) for r in er], threshold=THRESHOLD)
        by_epoch.append({"epoch": ep, "ap": em["ap"], "auc": em["auc_roc"],
                         "accuracy": em["accuracy"], "f1": em["f1"]})
    out["by_epoch"] = by_epoch
    out["by_epoch_note"] = (
        "Test scoring runs only on the checkpoints selected by val AP, not every epoch — "
        "so this is a handful of points, not a training curve." if len(by_epoch) > 1
        else "Only one checkpoint of this arm was ever scored on the test set, so there "
             "is no metric-vs-epoch curve to draw.")

    # Mean-over-N-checkpoints AP: the reliable metric for judging whether an arm's whole
    # trajectory moved (Stage 1 finding — rank-1/val-selected AP alone is noisy, range
    # 0.0204 across seeds vs 0.0048 for the mean). Computed from the SAME by_epoch dump
    # recomputation above (not re-read from test_summary.json), so it can never disagree
    # with the per-epoch curve on this same page. Only shown for a full 8/8-epoch run,
    # not a handful of hand-picked checkpoints.
    if len(by_epoch) >= 8:
        aps = [r["ap"] for r in by_epoch]
        out["mean_ap_over_epochs"] = {
            "n_epochs": len(aps), "mean_ap": round(sum(aps) / len(aps), 4),
            "note": f"average over all {len(aps)} checkpoints, recomputed from the "
                    "per-epoch dumps. This is the metric the Stage AA-H screen is judged "
                    "on; the single val-selected checkpoint is much noisier. Can differ by "
                    "a few thousandths from published summaries (scores stored rounded).",
        }
    return out


# ------------------------------------------------------------------------------ public
# The 667-clip Public half of the Nexar test set (disjoint from the 677-clip Private set
# used everywhere else on this page — no video_id overlap). Only scored for the Stage
# AA-H arms and their noise-floor controls; every other arm on this page has never been
# scored on it. Numbers are read from the run's own scores_public/*.metrics.json (already
# computed by score_checkpoints_on_test.py at the SAME threshold=0.5), not recomputed —
# the per-clip dump is used only to verify the metrics file describes the same scores.
EXPECTED_CM_PUBLIC = {
    "AA-ctrl-seed1":         dict(n=667, tp=271, fn=63, fp=62, tn=271),
    "AA-ctrl-seed2":         dict(n=667, tp=268, fn=66, fp=54, tn=279),
    "AA-H-rank-R_all":       dict(n=667, tp=284, fn=50, fp=67, tn=266),
    "AA-H-rank-R_pos":       dict(n=667, tp=282, fn=52, fp=66, tn=267),
    "AA-H-rank-partner_pos": dict(n=667, tp=282, fn=52, fp=66, tn=267),
    "AA-H-mass-R_pos":       dict(n=667, tp=278, fn=56, fp=79, tn=254),
    "midneg-seed0":   dict(n=667, tp=259, fn=75, fp=48, tn=285),
    "midneg-seed1":   dict(n=667, tp=269, fn=65, fp=52, tn=281),
    "midneg-seed2":   dict(n=667, tp=205, fn=129, fp=17, tn=316),
    "fullpool-seed0": dict(n=667, tp=297, fn=37, fp=91, tn=242),
    "fullpool-seed1": dict(n=667, tp=288, fn=46, fp=82, tn=251),
    "fullpool-seed2": dict(n=667, tp=309, fn=25, fp=118, tn=215),
}


def public_block(arm):
    cfg = arm.get("public")
    if not cfg:
        return None
    rows = load_jsonl(cfg["jsonl"])
    assert len(rows) == 667, f"{arm['key']}: expected 667 public-test rows, got {len(rows)}"
    y = [to01(r["gt_verdict"]) for r in rows]
    s = [float(r["score"]) for r in rows]
    m = metrics_from_arrays(y, s, threshold=THRESHOLD)

    exp = EXPECTED_CM_PUBLIC.get(arm["key"])
    assert exp is not None, f"{arm['key']}: no entry in EXPECTED_CM_PUBLIC"
    got = dict(n=m["n_total"], tp=m["tp"], fn=m["fn"], fp=m["fp"], tn=m["tn"])
    assert got == exp, f"{arm['key']} (public): drifted.\n  expected {exp}\n  got {got}"

    published = json.load(open(cfg["metrics"], encoding="utf-8"))
    # Identity gate, same purpose as the private-test one above: these are reproducible
    # from the per-clip dump, so if they disagree the metrics.json describes a different
    # checkpoint than this jsonl.
    for key, mine in (("fp", m["fp"]), ("tp", m["tp"]), ("tn", m["tn"]), ("fn", m["fn"])):
        assert published[key] == mine, \
            f"{arm['key']} (public): metrics.json {key}={published[key]} disagrees with " \
            f"the per-clip dump ({mine})"
    return {
        "available": True, "n": 667, "threshold": THRESHOLD,
        "source": str(cfg["jsonl"]).rsplit("outputs", 1)[-1].replace("\\", "/"),
        "ap": published["ap"], "auc": published["auc_roc"],
        "tp": published["tp"], "fp": published["fp"],
        "tn": published["tn"], "fn": published["fn"],
        "precision": published["precision"],
        "recall": published["recall_sensitivity_tpr"],
        "specificity": published["specificity_tnr"],
        "f1": published["f1"], "brier": published["brier"], "ece": published["ece"],
        "note": "A second, disjoint 667-clip half of the Nexar test set — scored only "
                "for the Stage AA-H family and its two noise-floor controls, so this "
                "panel does not appear on every arm's page.",
    }


# ------------------------------------------------------------------------------- main
def main():
    pools = build_pools()
    group_by_vid = {r["video_id"]: r["group"] for r in load_jsonl(TEST_MANIFEST)}
    prompt_cache = {}

    arms = []
    for arm in ARMS:
        p = arm.get("prompt")
        if p and p not in prompt_cache:
            prompt_cache[p] = load_prompt(p)

        hy = arm.get("hyper")
        if isinstance(hy, Path):
            args = json.load(open(hy, encoding="utf-8"))["args"]
            hyper = {"source": f"{hy.parent.name}/train_metrics.json (recorded by the run)",
                     "rows": hyper_rows(args)}
        elif isinstance(hy, dict):
            hyper = {"source": "run_a1fail321_4arms.sh", "rows": hyper_rows(hy)}
        else:
            hyper = None

        rec = {
            "key": arm["key"], "label": arm["label"], "order": arm["order"],
            "family": arm["family"], "tagline": arm["tagline"],
            "description": {"hypothesis": arm["hypothesis"], "aim": arm["aim"],
                            "method": arm["method"]},
            "prompt": prompt_cache.get(p), "prompt_note": arm.get("prompt_note"),
            "pool": arm.get("pool"),
            "media": arm.get("media"),
            # the page appends its own full stop after the note
            "arch": {**arm["arch"], "note": (arm["arch"].get("note") or "").rstrip(".") or None},
            "hyper": hyper, "hyper_note": arm.get("hyper_note"),
            "train": train_block(arm),
            "test": test_block(arm, group_by_vid),
            "public": public_block(arm),
            "inference": {
                "available": False,
                "note": "No arm has yet been scored on a dataset that is neither its own "
                        "training pool nor the 677-clip test set, so there is no "
                        "third-set inference result to show.",
            },
        }
        arms.append(rec)
        t = rec["test"]
        pub = f" public=AP {rec['public']['ap']:.4f} FP {rec['public']['fp']}" if rec["public"] else ""
        print(f"[arm] {arm['key']:<5} train={'yes' if rec['train']['available'] else 'NO ':<3} "
              f"valROC={'yes' if rec['train'].get('val_eval', {}).get('available') else 'no ':<3} "
              f"test={'AP %.4f AUC %.4f' % (t['ap'], t['auc']) if t['available'] else 'NONE'}{pub}")

    floor_means = [a["test"].get("mean_ap_over_epochs", {}).get("mean_ap")
                   for a in arms if a["key"] in NOISE_FLOOR_ARMS]
    assert len(floor_means) == 2 and None not in floor_means, \
        f"noise floor needs mean_ap_over_epochs on both {NOISE_FLOOR_ARMS}, got {floor_means}"
    noise_floor = {
        "low": min(floor_means), "high": max(floor_means),
        "note": "the same metric for AA-ctrl-seed1/2 (A1-compress256's recipe and split, "
                "only the LoRA-init seed changed) - how far it moves from random init alone. "
                "Published-summary basis: 0.8910-0.8958.",
    }
    print(f"[noise floor] {noise_floor['low']:.4f}-{noise_floor['high']:.4f}")

    data = {
        "generated_from": "build_experiments_data.py",
        "threshold": THRESHOLD,
        "tte_order": TTE_ORDER,
        "pools": pools,
        "stage1_noise_floor": noise_floor,
        "arms": arms,
    }
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("window.EXPERIMENTS_DATA = ")
        json.dump(data, f, separators=(",", ":"))
        f.write(";\n")
    print(f"[wrote] {OUT}  ({OUT.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
