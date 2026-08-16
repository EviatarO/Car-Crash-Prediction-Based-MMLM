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
  4. CONTROL: perturb A's patches with random gaussian noise of the SAME norm as
     delta_patches, push it through the SAME frozen pooler forward, measure the resulting
     pooled-space change.
  5. Compare ratio = ||delta_pooled|| / ||delta_patches||, real vs random-control.
     ratio_real << ratio_random  -> training moved directions the pooler discards (bypass).
     ratio_real ~= ratio_random  -> training's gradient already reaches the decision path.

Usage (on the pod):
  HF_HOME=/root/.cache/huggingface python3 p3_delta_patches_vs_pooled.py \
      --config ../configs/e4_stageA.yaml \
      --a-lora /workspace/semsup/a1_1761/epoch_04/lora_adapter \
      --b-lora /workspace/semsup/b_v3_1761/epoch_02/lora_adapter \
      --n-clips 40 --seed 0 \
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

    real_ratios, random_ratios = [], []
    dp_norms, dpooled_norms = [], []
    torch.manual_seed(args.seed)
    for i in range(len(sample)):
        dp = (B_patches[i] - A_patches[i]).flatten()
        dpo = (B_pooled[i] - A_pooled[i]).flatten()
        dp_norm = dp.norm().item()
        dpo_norm = dpo.norm().item()
        dp_norms.append(dp_norm)
        dpooled_norms.append(dpo_norm)
        if dp_norm > 0:
            real_ratios.append(dpo_norm / dp_norm)

        # Control: random perturbation of A's patches, SAME norm as the real delta,
        # pushed through the SAME frozen pooler (still holding B's weights loaded, but
        # the pooler/classifier are frozen and identical across A and B - see
        # ARCHITECTURE_BLOCKS.md sec 2/3 - so which arm's weights are loaded doesn't
        # affect the pooler itself).
        noise = torch.randn_like(A_patches[i])
        noise = noise * (dp_norm / (noise.norm().item() + 1e-12))
        perturbed = (A_patches[i] + noise).unsqueeze(0)
        # Call the SAME probe module the wrapper's hooks are attached to, so the
        # existing post-hook fires and repopulates _captured["pooled"].
        probe = getattr(badas.nn_model, "temporal_processor", None) or \
            getattr(badas.nn_model, "pooler", None)
        with torch.no_grad():
            probe(perturbed)
        pooled_rand = badas._captured["pooled"].squeeze(0).detach()
        d_rand_norm = (pooled_rand - A_pooled[i]).norm().item()
        if dp_norm > 0:
            random_ratios.append(d_rand_norm / dp_norm)

    import statistics as st
    result = {
        "n_clips": len(sample),
        "a_checkpoint": args.a_lora,
        "b_checkpoint": args.b_lora,
        "mean_delta_patches_norm": st.mean(dp_norms),
        "mean_delta_pooled_norm": st.mean(dpooled_norms),
        "real_ratio_mean": st.mean(real_ratios),
        "real_ratio_median": st.median(real_ratios),
        "random_control_ratio_mean": st.mean(random_ratios),
        "random_control_ratio_median": st.median(random_ratios),
        "interpretation": (
            "real >> random: gradient already reaching the decision path"
            if st.mean(real_ratios) > 1.5 * st.mean(random_ratios) else
            "real << random: gradient moving directions the pooler discards (bypass)"
            if st.mean(real_ratios) < 0.67 * st.mean(random_ratios) else
            "real ~= random: inconclusive at this sample size"
        ),
    }
    print(f"\n[result] mean ||delta_patches|| = {result['mean_delta_patches_norm']:.4f}")
    print(f"[result] mean ||delta_pooled||  = {result['mean_delta_pooled_norm']:.4f}")
    print(f"[result] REAL   ratio (||d_pooled||/||d_patches||): "
          f"mean={result['real_ratio_mean']:.4f}  median={result['real_ratio_median']:.4f}")
    print(f"[result] RANDOM-CONTROL ratio (same-norm noise -> pooled change): "
          f"mean={result['random_control_ratio_mean']:.4f}  "
          f"median={result['random_control_ratio_median']:.4f}")
    print(f"[interpretation] {result['interpretation']}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"[wrote] {out_path}")


if __name__ == "__main__":
    main()
