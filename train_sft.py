import os
from pathlib import Path
import json
import sys
import time

# #region agent log
def _debug_log(message, data, hypothesis_id):
    payload = {
        "sessionId": "debug-session",
        "runId": os.environ.get("SLURM_JOB_ID", "manual"),
        "hypothesisId": hypothesis_id,
        "location": "train_sft.py:debug",
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    try:
        with open("/home/eprojuser011/.cursor/debug.log", "a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")
    except Exception:
        pass
# #endregion

_debug_log(
    "python_startup",
    {
        "executable": sys.executable,
        "version": sys.version,
        "conda_prefix": os.environ.get("CONDA_PREFIX"),
        "path_head": os.environ.get("PATH", "")[:200],
    },
    "H1",
)

from torch.utils.data import DataLoader
import torch
try:
    import yaml
    _debug_log("import_ok", {"module": "yaml"}, "H2")
except Exception as exc:  # pragma: no cover
    _debug_log("import_failed", {"module": "yaml", "error": str(exc)}, "H2")
    raise

from data.dataset import MCASlidingWindowDataset
from models.factory import build_student_model
from prompts.templates import BASE_PROMPT


def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_prompt_batch(tokenizer, batch_size: int, prompt: str, score_token: str):
    prompt = prompt.replace(score_token, score_token)
    tokens = tokenizer(
        [prompt] * batch_size,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = tokens.input_ids
    attention_mask = tokens.attention_mask

    score_token_id = tokenizer.convert_tokens_to_ids(score_token)
    score_token_index = (input_ids == score_token_id).int().argmax(dim=1)
    return input_ids, attention_mask, score_token_index


def evaluate(model, loader, device, score_token):
    model.eval()
    scores = []
    targets = []
    before_alert = []
    after_alert = []

    with torch.no_grad():
        for batch in loader:
            clip = batch["clip"].to(device)
            target = batch["target"].to(device)
            t_seconds = batch["t_seconds"].to(device)
            time_of_alert = batch["time_of_alert"].to(device)

            input_ids, attention_mask, score_idx = build_prompt_batch(
                model.tokenizer, clip.size(0), BASE_PROMPT, score_token
            )
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            score_idx = score_idx.to(device)

            out = model(clip=clip, input_ids=input_ids, attention_mask=attention_mask, score_token_index=score_idx)
            prob = torch.sigmoid(out.score_logits)
            scores.extend(prob.detach().cpu().tolist())
            targets.extend(target.detach().cpu().tolist())

            valid_alert = torch.isfinite(time_of_alert)
            before_mask = valid_alert & (t_seconds < time_of_alert)
            after_mask = valid_alert & (t_seconds >= time_of_alert)
            if before_mask.any():
                before_alert.extend(prob[before_mask].detach().cpu().tolist())
            if after_mask.any():
                after_alert.extend(prob[after_mask].detach().cpu().tolist())

    metrics = {}
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score

        metrics["auc"] = roc_auc_score(targets, scores) if len(set(targets)) > 1 else float("nan")
        metrics["ap"] = average_precision_score(targets, scores) if len(set(targets)) > 1 else float("nan")
    except Exception:
        metrics["auc"] = float("nan")
        metrics["ap"] = float("nan")

    metrics["mean_pre_alert"] = float(sum(before_alert) / max(len(before_alert), 1))
    metrics["mean_post_alert"] = float(sum(after_alert) / max(len(after_alert), 1))
    model.train()
    return metrics


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = MCASlidingWindowDataset(
        csv_path=cfg["train_csv"],
        frames_root=cfg["frames_root"],
        videos_dir=cfg["train_videos_dir"],
        teacher_jsonl=cfg.get("teacher_jsonl"),
        window_size=cfg["window_size"],
        stride=cfg["stride"],
        frame_size=cfg["frame_size"],
        max_windows_per_video=cfg["max_windows_per_video"],
        window_step=cfg["window_step"],
        safe_risk=cfg["safe_risk"],
        ramp_low=cfg["ramp_low"],
        ramp_high=cfg["ramp_high"],
        ignore_after_event=cfg["ignore_after_event"],
        limit_noncollision_half=cfg["limit_noncollision_half"],
    )

    loader = DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        pin_memory=True,
    )
    _debug_log(
        "train_dataset_ready",
        {
            "dataset_len": len(dataset),
            "batch_size": cfg["batch_size"],
            "num_workers": cfg["num_workers"],
        },
        "H3",
    )

    val_loader = None
    if cfg.get("val_csv"):
        val_dataset = MCASlidingWindowDataset(
            csv_path=cfg["val_csv"],
            frames_root=cfg["frames_root"],
            videos_dir=cfg["train_videos_dir"],
            teacher_jsonl=cfg.get("teacher_jsonl"),
            window_size=cfg["window_size"],
            stride=cfg["stride"],
            frame_size=cfg["frame_size"],
            max_windows_per_video=cfg["max_windows_per_video"],
            window_step=cfg["window_step"],
            safe_risk=cfg["safe_risk"],
            ramp_low=cfg["ramp_low"],
            ramp_high=cfg["ramp_high"],
            ignore_after_event=cfg["ignore_after_event"],
            limit_noncollision_half=cfg["limit_noncollision_half"],
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.get("val_batch_size", cfg["batch_size"]),
            shuffle=False,
            num_workers=cfg["num_workers"],
            pin_memory=True,
        )

    model = build_student_model(
        vision_model_id=cfg["vision_model_id"],
        llm_model_id=cfg["llm_model_id"],
        score_token=cfg["score_token"],
        use_4bit=cfg.get("use_4bit", False),
        lora_r=cfg.get("lora_r", 0),
        lora_alpha=cfg.get("lora_alpha", 16),
        lora_dropout=cfg.get("lora_dropout", 0.05),
        lora_target_modules=cfg.get("lora_target_modules"),
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    loss_fn = torch.nn.BCEWithLogitsLoss()
    _debug_log(
        "model_ready",
        {
            "vision_model_id": cfg["vision_model_id"],
            "llm_model_id": cfg["llm_model_id"],
            "use_4bit": bool(cfg.get("use_4bit", False)),
            "lora_r": int(cfg.get("lora_r", 0)),
        },
        "H4",
    )

    model.train()
    for epoch in range(cfg["epochs"]):
        total_loss = 0.0
        for step, batch in enumerate(loader):
            if step == 0:
                _debug_log(
                    "first_batch_loaded",
                    {
                        "clip_shape": list(batch["clip"].shape),
                        "target_risk_min": float(batch["target_risk"].min().item()),
                        "target_risk_max": float(batch["target_risk"].max().item()),
                    },
                    "H5",
                )
            clip = batch["clip"].to(device)
            target = batch["target_risk"].to(device)

            input_ids, attention_mask, score_idx = build_prompt_batch(
                model.tokenizer, clip.size(0), BASE_PROMPT, cfg["score_token"]
            )
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            score_idx = score_idx.to(device)

            out = model(clip=clip, input_ids=input_ids, attention_mask=attention_mask, score_token_index=score_idx)
            loss = loss_fn(out.score_logits, target)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / max(len(loader), 1)
        log_line = f"Epoch {epoch+1}/{cfg['epochs']} - loss: {avg_loss:.4f}"
        if val_loader:
            metrics = evaluate(model, val_loader, device, cfg["score_token"])
            log_line += (
                f" | val_auc={metrics['auc']:.4f} val_ap={metrics['ap']:.4f} "
                f"pre_alert={metrics['mean_pre_alert']:.3f} post_alert={metrics['mean_post_alert']:.3f}"
            )
        print(log_line)

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "student_phase1.pt")


if __name__ == "__main__":
    main()
