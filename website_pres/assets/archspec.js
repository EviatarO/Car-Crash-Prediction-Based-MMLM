/* ================================================================= archspec.js
   The two model diagrams of the presentation Home page, as DATA (rendered by archview.js). Each block: name, short line, state, the tensor
   it hands to the next block (out), and `inner`: the sub-blocks shown in its pop-up, each with in -> out sizes.
   state: "io" input / output, "fixed" no weights, "frozen" weights not trained, "trained", "mixed" (frozen base + trained LoRA adapters).
   Sources of every size (checked 2026-10-20, never guessed):
     V-JEPA2 ViT-L config (patch 16, tubelet 2, 24 layers, 16 heads, MLP ratio 4) and the 256x256 processor output: docs_agents/ARCHITECTURE_BLOCKS.md;
     2048 real + 512 predicted tokens: docs_agents/ARCHITECTURE.md section 1; crash-model LoRA targets: outputs/a1_compress256/train/epoch_02/lora_adapter
     (r 16 on q, k, v of the 24 encoder layers and the 12-layer, 384-wide predictor); projector: student_training/models/r1_bridge.py (Merger);
     Qwen3-VL-4B text model: dataset/public_samples/qwen3vl_cfg/config.json (hidden 2560, 36 layers, 32 query / 8 key-value heads, head dim 128, MLP 9728,
     vocab 151,936, M-RoPE sections 24/20/20); prompt / answer token counts: PromptBuilder on the real tokenizer (637 input tokens = 512 video + 125 text). */
(function (global) {
  const VIDEO = {id: "video", name: "Dashcam video", line: "16 frames · 2 s at 7.5 fps", state: "io", out: "16 × 720 × 1280 × 3",
    inner: {title: "Input window", note: "Every 4th frame of a 30-fps clip: 16 frames covering the last 2 seconds before the decision time.",
      blocks: [{name: "Clip (30 fps)", out: "≈ 60 source frames"}, {name: "Take every 4th frame", out: "16 frames · 1280 × 720"}]}};
  const PRE = {id: "pre", name: "Full-frame resize", line: "no crop", state: "fixed", out: "16 × 3 × 256 × 256",
    inner: {title: "Pre-processing (no learned weights)", note: "The whole frame is kept (squashed to a square) instead of a centre crop, so road users at the edges stay visible.",
      blocks: [{name: "Resize the full frame", in: "1280 × 720", out: "256 × 256"}, {name: "Normalise (ImageNet mean / std)", out: "3 × 256 × 256"},
               {name: "Stack the 16 frames", out: "16 × 3 × 256 × 256"}]}};
  const encInner = (lora) => ({title: "V-JEPA2 ViT-L video encoder (BADAS-Open)",
    note: lora === "trained" ? "Base weights frozen; LoRA adapters (rank 16) on the attention q / k / v of all 24 layers are trained with the crash loss."
                             : "The crash model's encoder, used as is: its weights (including the LoRA learned for crashes) are frozen, and its tokens are computed once.",
    blocks: [{name: "Tubelet patch embedding", line: "Conv3d 3 → 1024, kernel 2 × 16 × 16", in: "16 × 3 × 256 × 256", out: "8 × 16 × 16 = 2048 tokens × 1024", state: "frozen"},
             {name: "24 × transformer block", line: "LayerNorm → self-attention, 16 heads, 3D RoPE" + (lora === "trained" ? " (+ LoRA on q, k, v)" : "") + " → LayerNorm → MLP 1024 → 4096 → 1024",
              in: "2048 × 1024", out: "2048 × 1024", state: lora === "trained" ? "mixed" : "frozen"},
             {name: "Final LayerNorm", out: "2048 × 1024", state: "frozen"}]});

  const REASON = {
    key: "reason", title: "BADAS-JEPA-Reason", tag: "Reasoning model",
    caption: "One pass over the video gives the verdict, its probability, the reasoning and the expected outcome.",
    chain: [VIDEO, PRE,
      {id: "enc", name: "V-JEPA2 ViT-L encoder", line: "BADAS-Open (SFT) weights", state: "frozen", out: "2048 × 1024 tokens", inner: encInner("frozen")},
      {id: "proj", name: "Projector", line: "≈ 27 M parameters", state: "trained", out: "512 × 2560 video tokens",
        inner: {title: "Projector (the bridge from video to language)", note: "Trained from scratch: phase 1 alone (language model frozen), phase 2 together with the language-model LoRA.",
          blocks: [{name: "LayerNorm per token", in: "2048 × 1024", out: "2048 × 1024", state: "trained"},
                   {name: "Merge 2 × 2 neighbouring tokens", line: "concatenate", in: "8 × 16 × 16 × 1024", out: "8 × 8 × 8 = 512 × 4096", state: "fixed"},
                   {name: "Linear", in: "4096", out: "4096", state: "trained"}, {name: "GELU", out: "4096", state: "fixed"},
                   {name: "Linear", in: "4096", out: "2560 (the language model's width)", state: "trained"}]}},
      {id: "lm", row: true, name: "Qwen3-VL-4B language model", line: "36 layers · LoRA rank 16", state: "mixed", out: "answer text",
        side: {name: "Prompt", line: "“Look at the road user that matters most … Collision / Reasoning / Outcome”", state: "io", out: "125 text tokens"},
        inner: {title: "Qwen3-VL-4B language model (its own image encoder is not used)",
          note: "Input = 637 tokens: 8 time steps × 64 video tokens = 512, plus 125 text tokens. Base weights frozen; LoRA adapters (rank 16) on all attention and MLP projections are trained.",
          blocks: [{name: "Embed the prompt text", in: "125 tokens", out: "125 × 2560", state: "frozen"},
                   {name: "Interleave video and text", line: "8 × [time, ⟨vision start⟩, 64 video tokens, ⟨vision end⟩] + question", out: "637 × 2560", state: "fixed"},
                   {name: "3D positions (M-RoPE)", line: "time / height / width", out: "637 × 2560", state: "fixed"},
                   {name: "36 × decoder layer", line: "RMSNorm → attention (32 query / 8 key-value heads): q 2560 → 4096, k, v 2560 → 1024, o 4096 → 2560 → RMSNorm → SwiGLU MLP 2560 → 9728 → 2560; LoRA on every projection",
                    in: "637 × 2560", out: "637 × 2560", state: "mixed"},
                   {name: "Final RMSNorm + LM head", in: "2560", out: "151,936 (vocabulary) → next word", state: "frozen"}]}},
      {id: "ans", name: "Answer", line: "Collision: yes · Reasoning: … · Outcome: …", state: "io", out: "≈ 25–60 tokens",
        inner: {title: "Answer format (verdict first)", note: "The verdict is the first word, so the decision is read before any reasoning is written; the reasoning and the outcome follow.",
          blocks: [{name: "Collision: yes / no."}, {name: "Reasoning:", line: "how the key road user moves and how the gap changes"}, {name: "Outcome:", line: "one short sentence"}]}},
      {id: "dec", name: "Decision read-out", line: "at the word after “Collision:”", state: "fixed", out: "P(collision)",
        inner: {title: "From the language model to a probability", note: "No extra head: the probability is read from the language model's own next-word distribution.",
          blocks: [{name: "Next-word scores after “Collision:”", out: "151,936"}, {name: "Keep “ yes” and “ no”", out: "2"},
                   {name: "log P(yes) − log P(no)", out: "1 number"}, {name: "Sigmoid", out: "P(collision) ∈ [0, 1]; yes if P ≥ 0.5"}]}}],
    loss: {name: "Training loss", line: "next-word cross-entropy on the answer tokens only (the prompt and video tokens are not scored)"}
  };

  const CRASH = {
    key: "crash", title: "BADAS-Open (SFT)", tag: "Crash model",
    caption: "BADAS-Open fine-tuned on Nexar with full-frame input: a probability of collision, and the crash head's attention shows where it looks.",
    chain: [VIDEO, PRE,
      {id: "enc", name: "V-JEPA2 ViT-L encoder", line: "+ LoRA rank 16 on q, k, v", state: "mixed", out: "2048 × 1024 tokens", inner: encInner("trained")},
      {id: "pred", name: "BADAS predictor", line: "+ 512 predicted future tokens", state: "mixed", out: "2560 × 1024 tokens",
        inner: {title: "Predictor (anticipates the next moments in feature space)", note: "Base weights frozen; LoRA adapters (rank 16) on q / k / v of its 12 layers are trained with the crash loss.",
          blocks: [{name: "Project down", in: "2048 × 1024", out: "2048 × 384", state: "frozen"}, {name: "Add 512 future-position tokens", out: "2560 × 384", state: "frozen"},
                   {name: "12 × transformer block (width 384)", line: "+ LoRA on q, k, v", out: "2560 × 384", state: "mixed"},
                   {name: "Project the 512 predicted tokens up", in: "512 × 384", out: "512 × 1024", state: "frozen"},
                   {name: "Concatenate with the encoder tokens", line: "2048 seen + 512 predicted", out: "2560 × 1024", state: "fixed"}]}},
      {id: "head", row: true, name: "Crash head", line: "attentive probe (frozen)", state: "frozen", out: "2 logits",
        inner: {title: "Crash head (BADAS-Open's own head, kept frozen)", note: "Its attention weights over the 2560 tokens show where the model looks: the attention heat-map.",
          blocks: [{name: "Learned queries attend over all tokens", in: "2560 × 1024", out: "pooled 1024", state: "frozen"},
                   {name: "MLP classifier", in: "1024", out: "2 logits", state: "frozen"}]}},
      {id: "p", name: "P(collision)", line: "softmax", state: "io", out: "1 number",
        inner: {title: "Output", blocks: [{name: "Softmax over the 2 logits", in: "2", out: "P(collision) ∈ [0, 1]; yes if P ≥ 0.5"}]}}],
    loss: {name: "Training loss", line: "cross-entropy against the crash label"}
  };

  global.ARCH_SPEC = {paths: [REASON, CRASH],
    legend: [["trained", "trained"], ["mixed", "frozen base + trained LoRA"], ["frozen", "frozen"], ["fixed", "no weights"], ["io", "input / output"], ["loss", "training loss"]]};
})(window);
