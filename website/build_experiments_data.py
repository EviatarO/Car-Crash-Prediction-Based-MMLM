"""
build_experiments_data.py
=========================
Generates website/experiments_data.js (window.EXPERIMENTS_DATA) - the per-arm payload
behind the Experiments page's detail view: description, prompt, dataset composition,
architecture configuration, hyperparameters, training curves and test results for
A0, A1, B-v1, B-v2, B-v3, P1, V10 and V12.

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
    selection criterion. For V10/V12 only, per-epoch val accuracy/AUC ARE derived here
    from their `val_scores_ep*.jsonl` dumps.
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
CAPS = MMLM_AI / "outputs" / "semantic_captions"
MANIFESTS = MMLM_AI / "dataset" / "manifests"
TEST_MANIFEST = MANIFESTS / "test_manifest_hires.jsonl"

THRESHOLD = 0.5
# metrics_core.py's own mapping - group is an int on the test manifest and every test dump.
GROUP_LABEL = {0: "tte_0.5s", 1: "tte_1.0s", 2: "tte_1.5s"}
TTE_ORDER = ["tte_0.5s", "tte_1.0s", "tte_1.5s"]


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
    "infonce_tau_init", "siglip_model", "captions_path", "bank_captions",
    "lr", "lr_schedule", "warmup_frac", "epochs", "grad_accum", "clip_grad_per_group",
    "unfreeze_head", "head_lr_mult", "head_lr_schedule",
    "early_stop_patience", "select_by", "keep_top_k", "val_frac", "seed",
]


def shorten(v):
    """Long absolute paths are noise in a table; the basename identifies the file."""
    if isinstance(v, str) and ("/" in v or "\\" in v) and not v.startswith("re:"):
        return v.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    return v


def hyper_rows(args):
    rows = []
    for k in HYPER_KEYS:
        if k in args and args[k] is not None:
            rows.append([k, str(shorten(args[k]))])
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

    A(key="V10", label="V10 · Failure recovery, GT captions", order=6, family="a1fail321",
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
      test=None,
      test_note="Never scored on the 677-clip test set. Producing it needs a GPU pass with "
                "score_checkpoints_on_test.py pointed at this arm's epoch-10 LoRA adapter; "
                "no number is estimated here in the meantime."),

    A(key="V12", label="V12 · Failure recovery, neutral captions", order=7, family="a1fail321",
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
            "note": "This run predates --dump-val-scores, so no per-clip validation "
                    "scores exist: no ROC curve and no confusion matrix can be drawn for "
                    "its training phase. Only the loss and val-AP curves above survive.",
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
    return out


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
            "arch": arm["arch"],
            "hyper": hyper, "hyper_note": arm.get("hyper_note"),
            "train": train_block(arm),
            "test": test_block(arm, group_by_vid),
            "inference": {
                "available": False,
                "note": "No arm has yet been scored on a dataset that is neither its own "
                        "training pool nor the 677-clip test set, so there is no "
                        "third-set inference result to show.",
            },
        }
        arms.append(rec)
        t = rec["test"]
        print(f"[arm] {arm['key']:<5} train={'yes' if rec['train']['available'] else 'NO ':<3} "
              f"valROC={'yes' if rec['train'].get('val_eval', {}).get('available') else 'no ':<3} "
              f"test={'AP %.4f AUC %.4f' % (t['ap'], t['auc']) if t['available'] else 'NONE'}")

    data = {
        "generated_from": "build_experiments_data.py",
        "threshold": THRESHOLD,
        "tte_order": TTE_ORDER,
        "pools": pools,
        "arms": arms,
    }
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("window.EXPERIMENTS_DATA = ")
        json.dump(data, f, separators=(",", ":"))
        f.write(";\n")
    print(f"[wrote] {OUT}  ({OUT.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
