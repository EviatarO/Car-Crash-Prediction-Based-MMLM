"""P3 diagnostic: does semantic-loss training (patches tap) actually move the vector the
crash classifier reads, or does it shape directions the pooler ignores?

Background (see docs_agents/ARCHITECTURE_BLOCKS.md sec 5b/7c): the pooled-tap B1 probe
showed caption info SURVIVES the pooler's 2560x compression (22x chance retrieval). That
answers "does the information exist in the pooled vector" - it does NOT answer whether the
gradient actually produced by training at the patch grid (the current default) lands in
directions the pooler keeps or directions it discards. This script answers that directly,
no training required - just two existing checkpoints and a handful of forward passes.

Method:
  1. Load BADAS with A1_1761's LoRA (crash-only control) and with B-v3's LoRA (crash +
     semantic, patches tap) - same base weights, same frozen pooler/classifier in both.
  2. Run the same held-out clips through both; capture patches (P,D) and pooled (D,) via
     the existing hooks in TrainableBadasWrapper.
  3. delta_patches = patches_B - patches_A ; delta_pooled = pooled_B - pooled_A.
  4. CONTROL, per clip: draw --n-noise (default 20) independent gaussian perturbations of
     A's patches, each with the SAME norm as that clip's real delta_patches, push each
     through the SAME frozen pooler, average the resulting pooled-space change. Averaging
     over multiple draws (not the 1-draw version of this script's first pass) is needed
     because a single noise draw is itself a noisy estimate of the "what does a random
     equal-sized change look like" question.
  5. Compare ratio = ||delta_pooled|| / ||delta_patches||, real vs the per-clip random-noise
     mean - PAIRED per clip (same clip contributes one real value and one random value),
     not compared as two independent means. Report a bootstrap 95% CI on the mean paired
     difference (real - random), resampling CLIPS with replacement.
     CI excludes zero, real > random -> gradient reaches the decision path (>= a random
       perturbation would).
     CI excludes zero, real < random -> gradient moving directions the pooler discards.
     CI includes zero -> not distinguishable from a random perturbation at this n.

  2026-08-16 correction: the first version of this script reported only the mean/median
  ratio (real=0.0034 vs random=0.0019, "1.8x") with NO per-clip values saved and a SINGLE
  noise draw per clip, so no CI could be computed - the "1.8x" claim was unquantified. This
  version fixes both: per-clip values are saved to the output JSON, and the random control
  is averaged over multiple draws per clip before the paired comparison.

Usage (on the pod):
  HF_HOME=/root/.cache/huggingface python3 p3_delta_patches_vs_pooled.py \
      --config ../configs/e4_stageA.yaml \
      --a-lora /workspace/semsup/a1_1761/epoch_04/lora_adapter \
      --b-lora /workspace/semsup/b_v3_1761/epoch_02/lora_adapter \
      --n-clips 40 --n-noise 20 --n-boot 5000 --seed 0 \
      --out /workspace/semsup/p3_delta_result.json
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models"))

from semsup_common import TrainableBadasWrapper, load_training_examples, clip_level_split  # noqa: E402


def load_lora_weights(badas, lora_dir):
    """Same helper semsup_train.py uses for --lora-init (peft state-dict load, not a
    fresh get_peft_model construction), so both checkpoints load onto an IDENTICAL LoRA
    topology (query,key,value legacy targets - both A1_1761 and B-v3 used this recipe)."""
    from safetensors.torch import load_file as load_sft
    from peft.utils import set_peft_model_state_dict as set_peft_sd
    sft = Path(lora_dir) / "adapter_model.safetensors"
    set_peft_sd(badas.nn_model, load_sft(str(sft)))
    print(f"[load] LoRA weights from {sft}")


def capture(badas, clip):
    """Run one preprocessed clip through the model, return (patches, pooled) detached."""
    with torch.no_grad():
        _, patches = badas.forward_clip(clip)
    pooled = badas._captured.get("pooled")
    if pooled is None:
        raise RuntimeError("pooled hook did not fire - is the post-hook registered in "
                            "TrainableBadasWrapper.__init__? (semsup_common.py)")
    return patches.detach().clone(), pooled.squeeze(0).detach().clone()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--a-lora", required=True, help="control checkpoint (crash-only)")
    ap.add_argument("--b-lora", required=True, help="treatment checkpoint (crash+semantic)")
    ap.add_argument("--n-clips", type=int, default=40)
    ap.add_argument("--n-noise", type=int, default=20,
                     help="independent random-perturbation draws PER CLIP for the control, "
                          "averaged before the paired comparison (was implicitly 1 in the "
                          "first version of this script)")
    ap.add_argument("--n-boot", type=int, default=5000,
                     help="bootstrap resamples (of CLIPS, paired) for the CI on the mean "
                          "real-minus-random ratio difference")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import yaml
    cfg = yaml.safe_load(open(args.config))

    print("[setup] building wrapper with LoRA topology (query,key,value legacy targets, "
          "matching both A1_1761 and B-v3's recipe)")
    # A bare string is treated as a REGEX by TrainableBadasWrapper/peft, not a
    # comma-list - semsup_train.py's CLI does this same split before passing in.
    target_modules = ["query", "key", "value"]
    badas = TrainableBadasWrapper(cfg, lora_target_modules=target_modules,
                                   lora_r=16, lora_alpha=32, lora_dropout=0.05)
    device = badas.device

    print("[data] sampling held-out clips (same clip_level_split(seed=0) as training)")
    examples = load_training_examples(
        limit=0,
        captions_path="/workspace/MMLM_AI/outputs/semantic_captions/"
                       "Caption_V12_Neutral_1761_fortrain.jsonl")
    _, val_ex = clip_level_split(examples, val_frac=0.2, seed=0)
    import random
    rng = random.Random(args.seed)
    sample = rng.sample(val_ex, min(args.n_clips, len(val_ex)))
    print(f"[data] using {len(sample)} held-out clips")

    # Pass 1: load A's weights, capture patches+pooled for every sampled clip.
    load_lora_weights(badas, args.a_lora)
    badas.nn_model.eval()
    A_patches, A_pooled, clips_prepped = [], [], []
    for ex in sample:
        clip = badas._preprocess_clip(badas.vjepa, ex["frame_paths"]).to(device)
        p, pooled = capture(badas, clip)
        A_patches.append(p)
        A_pooled.append(pooled)
        clips_prepped.append(clip)  # reuse the exact same preprocessed tensor for pass 2
    print(f"[pass A] captured {len(A_patches)} clips  patches={tuple(A_patches[0].shape)}  "
          f"pooled={tuple(A_pooled[0].shape)}")

    # Pass 2: swap in B's weights, capture the same clips (reusing the SAME preprocessed
    # tensors, so any difference is purely due to the LoRA weight change, not preprocessing).
    load_lora_weights(badas, args.b_lora)
    badas.nn_model.eval()
    B_patches, B_pooled = [], []
    for clip in clips_prepped:
        p, pooled = capture(badas, clip)
        B_patches.append(p)
        B_pooled.append(pooled)
    print(f"[pass B] captured {len(B_patches)} clips")

    probe = getattr(badas.nn_model, "temporal_processor", None) or \
        getattr(badas.nn_model, "pooler", None)

    clip_ids, dp_norms, dpooled_norms = [], [], []
    real_ratios, random_ratios_mean = [], []
    per_clip_noise_ratios = []  # kept for inspection, not just the mean
    g = torch.Generator(device=device).manual_seed(args.seed)

    for i, ex in enumerate(sample):
        dp = (B_patches[i] - A_patches[i]).flatten()
        dpo = (B_pooled[i] - A_pooled[i]).flatten()
        dp_norm = dp.norm().item()
        dpo_norm = dpo.norm().item()
        clip_ids.append(ex.get("video_id", f"clip_{i}"))
        dp_norms.append(dp_norm)
        dpooled_norms.append(dpo_norm)
        real_ratio = dpo_norm / dp_norm if dp_norm > 0 else float("nan")
        real_ratios.append(real_ratio)

        # Control: N independent random perturbations of A's patches, each with the SAME
        # norm as this clip's real delta_patches, each pushed through the SAME frozen
        # pooler. A single draw is itself a noisy estimate of "what a random equal-sized
        # change looks like" - average N=--n-noise draws per clip before comparing.
        noise_ratios_this_clip = []
        for _ in range(args.n_noise):
            noise = torch.randn(A_patches[i].shape, generator=g, device=device)
            noise = noise * (dp_norm / (noise.norm().item() + 1e-12))
            perturbed = (A_patches[i] + noise).unsqueeze(0)
            with torch.no_grad():
                probe(perturbed)
            pooled_rand = badas._captured["pooled"].squeeze(0).detach()
            d_rand_norm = (pooled_rand - A_pooled[i]).norm().item()
            noise_ratios_this_clip.append(d_rand_norm / dp_norm if dp_norm > 0 else float("nan"))
        per_clip_noise_ratios.append(noise_ratios_this_clip)
        random_ratios_mean.append(sum(noise_ratios_this_clip) / len(noise_ratios_this_clip))

    # Paired difference per clip: real - (that clip's own random-noise mean).
    import statistics as st
    diffs = [r - n for r, n in zip(real_ratios, random_ratios_mean)]
    mean_diff = st.mean(diffs)

    # Paired bootstrap CI: resample CLIPS with replacement, recompute mean(diffs).
    rng = torch.Generator().manual_seed(args.seed)
    n = len(diffs)
    boot_means = []
    diffs_t = torch.tensor(diffs)
    for _ in range(args.n_boot):
        idx = torch.randint(0, n, (n,), generator=rng)
        boot_means.append(diffs_t[idx].mean().item())
    boot_means.sort()
    lo = boot_means[int(0.025 * args.n_boot)]
    hi = boot_means[int(0.975 * args.n_boot) - 1]
    ci_excludes_zero = (lo > 0) or (hi < 0)

    if ci_excludes_zero and mean_diff > 0:
        interpretation = ("CI excludes zero, real > random: gradient reaches the decision "
                           "path at least as well as a random perturbation would")
    elif ci_excludes_zero and mean_diff < 0:
        interpretation = ("CI excludes zero, real < random: gradient moving directions the "
                           "pooler discards (bypass)")
    else:
        interpretation = ("CI includes zero: real and random-perturbation ratios are NOT "
                           "distinguishable at this sample size")

    result = {
        "n_clips": len(sample),
        "n_noise_per_clip": args.n_noise,
        "n_boot": args.n_boot,
        "a_checkpoint": args.a_lora,
        "b_checkpoint": args.b_lora,
        "clip_ids": clip_ids,
        # Per-clip arrays, so a future re-analysis doesn't require re-running the model.
        "per_clip": {
            "delta_patches_norm": dp_norms,
            "delta_pooled_norm": dpooled_norms,
            "real_ratio": real_ratios,
            "random_ratio_mean_over_noise_draws": random_ratios_mean,
            "noise_ratio_all_draws": per_clip_noise_ratios,
        },
        "mean_delta_patches_norm": st.mean(dp_norms),
        "mean_delta_pooled_norm": st.mean(dpooled_norms),
        "real_ratio_mean": st.mean(real_ratios),
        "real_ratio_median": st.median(real_ratios),
        "random_control_ratio_mean": st.mean(random_ratios_mean),
        "random_control_ratio_median": st.median(random_ratios_mean),
        "paired_diff_mean": mean_diff,
        "paired_diff_ci95": [lo, hi],
        "ci_excludes_zero": ci_excludes_zero,
        "interpretation": interpretation,
    }
    print(f"\n[result] mean ||delta_patches|| = {result['mean_delta_patches_norm']:.4f}")
    print(f"[result] mean ||delta_pooled||  = {result['mean_delta_pooled_norm']:.4f}")
    print(f"[result] REAL   ratio: mean={result['real_ratio_mean']:.4f}  "
          f"median={result['real_ratio_median']:.4f}")
    print(f"[result] RANDOM-CONTROL ratio ({args.n_noise} draws/clip, averaged): "
          f"mean={result['random_control_ratio_mean']:.4f}  "
          f"median={result['random_control_ratio_median']:.4f}")
    print(f"[result] PAIRED diff (real - random), mean={mean_diff:.5f}  "
          f"95% CI=[{lo:.5f}, {hi:.5f}]  excludes_zero={ci_excludes_zero}")
    print(f"[interpretation] {interpretation}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"[wrote] {out_path}")


if __name__ == "__main__":
    main()
