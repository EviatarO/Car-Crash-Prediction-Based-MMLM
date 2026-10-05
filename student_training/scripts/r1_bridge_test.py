"""
r1_bridge_test.py - local CPU unit tests for student_training/models/r1_bridge.py (no GPU, tiny random LM).

 1. regroup_2x2 maps V-JEPA index t*256+h*16+w to the right merged token / sub-patch slot.
 2. PromptBuilder token ids == Qwen3VLProcessor's ids for a real 16x256x256 dummy video at 7.5 fps
    (grid (8,16,16), 512 video tokens, mm_token_type_ids).
 3. Logit equivalence: our injection path (scatter visual tokens + get_rope_index + language_model)
    == the OFFICIAL Qwen3VLModel forward given the same visual embeddings (tiny random model).
 4. Only the merger gets gradients; the LM stays frozen. LoRA adds trainable params only in the LM.
 5. Loss is computed on answer tokens only; blank / wrong-video modes change the loss.
 6. Greedy generate (KV cache) reproduces the argmax of a full forward.

    python student_training/scripts/r1_bridge_test.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models"))
from r1_bridge import (GRID, N_T, N_VIS, IGNORE, Merger, PromptBuilder, R1Bridge, regroup_2x2)  # noqa: E402

torch.manual_seed(0)


def tiny_qwen(vocab_from_tokenizer):
    from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration
    cfg = Qwen3VLConfig(
        text_config=dict(hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
                         num_key_value_heads=2, head_dim=16, vocab_size=vocab_from_tokenizer,
                         rope_scaling=dict(rope_type="default", mrope_interleaved=True, mrope_section=[4, 2, 2]),
                         max_position_embeddings=4096, tie_word_embeddings=True),
        vision_config=dict(depth=2, hidden_size=32, intermediate_size=64, num_heads=2, in_channels=3,
                           patch_size=16, spatial_merge_size=2, temporal_patch_size=2, out_hidden_size=64,
                           num_position_embeddings=2304, deepstack_visual_indexes=[]),
        image_token_id=151655, video_token_id=151656, vision_start_token_id=151652,
        vision_end_token_id=151653, tie_word_embeddings=True)
    return Qwen3VLForConditionalGeneration(cfg).eval()


def main():
    from huggingface_hub import snapshot_download
    from transformers import AutoProcessor
    from transformers.video_utils import VideoMetadata

    # local_dir (not the HF cache): the cache needs symlinks, which Windows blocks without privileges
    d = snapshot_download("Qwen/Qwen3-VL-4B-Instruct",
                          allow_patterns=["*.json", "*.txt", "*.jinja", "tokenizer*", "merges.txt", "vocab.json"],
                          local_dir=str(Path(__file__).resolve().parents[2] / "dataset" / "public_samples" / "qwen3vl_cfg"))
    proc = AutoProcessor.from_pretrained(d)
    tok = proc.tokenizer

    # 1 ---- regroup order
    idx = torch.arange(N_T * GRID * GRID).float().view(1, -1, 1).expand(1, -1, 3).clone()
    r = regroup_2x2(idx)                                          # (1, 512, 12)
    assert r.shape == (1, N_VIS, 12)
    for t, gh, gw in [(0, 0, 0), (3, 5, 2), (7, 7, 7)]:
        m = t * 64 + gh * 8 + gw
        want = [t * 256 + (gh * 2 + a) * 16 + (gw * 2 + b) for a in (0, 1) for b in (0, 1)]
        got = [int(r[0, m, 3 * k]) for k in range(4)]
        assert got == want, (t, gh, gw, got, want)
    print("[1] regroup order OK")

    # 2 ---- prompt ids == processor ids
    pb = PromptBuilder(tok)
    vid = np.random.randint(0, 255, (16, 256, 256, 3), dtype=np.uint8)
    md = VideoMetadata(total_num_frames=16, fps=7.5, frames_indices=list(range(16)), duration=16 / 7.5)
    msgs = [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": "Describe the motion."}]}]
    text = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    ref = proc(text=[text], videos=[vid], video_metadata=[md], return_tensors="pt", do_resize=False,
               do_sample_frames=False, return_mm_token_type_ids=True)
    ids, _, mm = pb.encode("", "Describe the motion.")
    assert ids == ref["input_ids"][0].tolist(), "prompt ids differ from the official processor"
    assert mm == ref["mm_token_type_ids"][0].tolist()
    assert ref["video_grid_thw"].tolist() == [[8, 16, 16]]
    assert sum(1 for i in ids if i == pb.video_id) == N_VIS
    print(f"[2] prompt ids identical to Qwen3VLProcessor ({len(ids)} tokens, grid (8,16,16), {N_VIS} video tokens)")

    # 3 ---- logit equivalence with the official forward (tiny random LM, deepstack off)
    qwen = tiny_qwen(len(tok))
    with torch.no_grad():
        pix = ref["pixel_values_videos"].float()
        emb_off = qwen.model.get_video_features(pix, ref["video_grid_thw"], return_dict=True).pooler_output
        vis = torch.cat(list(emb_off), 0)                          # (512, 64)
        official = qwen(input_ids=ref["input_ids"], attention_mask=ref["attention_mask"], pixel_values_videos=pix,
                        video_grid_thw=ref["video_grid_thw"], mm_token_type_ids=ref["mm_token_type_ids"]).logits
    br = R1Bridge(qwen, tok)
    br.eval()
    with torch.no_grad():
        ours = br.forward_vis(vis[None], ref["input_ids"], ref["mm_token_type_ids"], ref["attention_mask"])
    err = (official - ours).abs().max().item()
    assert err < 1e-4, f"logits differ from the official path: {err}"
    print(f"[3] logits match the official Qwen3VLModel forward (max abs diff {err:.2e})")

    # 4 ---- gradients: merger only; LoRA only inside the LM
    feats = torch.randn(2, N_T * GRID * GRID, 1024)
    items = [pb.encode("[T]", "Describe the scene.", "A car ahead brakes."),
             pb.encode("[T]", "Is a collision coming?", "Collision: no.")]
    batch = pb.collate(items)
    loss = br.loss_per_sample(feats, batch).mean()
    loss.backward()
    assert all(p.grad is not None for p in br.merger.parameters())
    assert all((not p.requires_grad) for n, p in br.named_parameters() if not n.startswith("merger"))
    assert all(p.grad is None for n, p in br.named_parameters() if not n.startswith("merger"))
    n_tr = sum(p.numel() for p in br.trainable_params())
    print(f"[4] gradients reach only the merger ({n_tr/1e6:.1f}M params at this tiny size)")
    br.zero_grad()
    br.enable_lora(r=4, alpha=8, dropout=0.0)
    loss = br.loss_per_sample(feats, batch).mean()
    loss.backward()
    lora = [n for n, p in br.named_parameters() if p.requires_grad and "lora_" in n]
    assert lora and all("language_model" in n or "text" in n for n in lora)
    assert all(p.grad is not None for n, p in br.named_parameters() if p.requires_grad)
    print(f"[4b] LoRA adds {len(lora)} trainable tensors, all with gradients")

    # 5 ---- answer-only loss, and the control modes change the loss
    ids_, lab, mm_, att = batch
    n_ans0 = (lab[0] != IGNORE).sum().item()
    a0 = len(tok("A car ahead brakes.<|im_end|>", add_special_tokens=False)["input_ids"])
    assert n_ans0 == a0, (n_ans0, a0)
    assert (lab[:, : 580] == IGNORE).all(), "video/prompt tokens must be masked"
    with torch.no_grad():
        real = br.loss_per_sample(feats, batch)
        blank = br.loss_per_sample(feats, batch, mode="blank")
        wrong = br.loss_per_sample(feats, batch, mode="wrong", other=feats.flip(0))
    assert not torch.allclose(real, blank) and not torch.allclose(real, wrong)
    print(f"[5] labels cover answer tokens only; real/blank/wrong losses differ "
          f"({real.mean():.3f}/{blank.mean():.3f}/{wrong.mean():.3f})")

    # 6 ---- generation (KV cache) == argmax of a full forward, token by token
    br.eval()
    f1 = feats[:1]
    with torch.no_grad():
        txt = br.generate(f1, "[T]", "Describe the scene.", max_new_tokens=6)
        ids0, _, mm0 = pb.encode("[T]", "Describe the scene.")
        cur = list(ids0)
        gen = []
        for _ in range(6):
            t = torch.tensor([cur])
            m = torch.tensor([[2 if i == pb.video_id else 0 for i in cur]], dtype=torch.int32)
            lg = br.forward_vis(br.visual_tokens(f1), t, m, torch.ones_like(t))
            nxt = int(lg[0, -1].argmax())
            if nxt == pb.im_end_id:
                break
            gen.append(nxt)
            cur.append(nxt)
    assert txt == tok.decode(gen, skip_special_tokens=True), (txt, tok.decode(gen))
    print("[6] cached greedy generation == full-forward argmax")
    print("ALL r1_bridge tests passed")


if __name__ == "__main__":
    main()
