"""APO metric: deterministic scoring for prompt optimization.

Three components:
  1. Verdict correctness (binary, weight 0.30)
  2. BERTScore F1 vs translated Hebrew GT reasoning (weight 0.45)
  3. Length compliance: ≤150 words (weight 0.25, soft penalty above)

No LLM-as-judge — fully deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

# bert-score is in requirements.txt. Lazy-load to avoid 1.4GB model download
# on import for callers that don't need scoring (e.g., signature inspection).
_bert_score_fn = None
_bert_model_loaded = False


def _get_bert_score_fn():
    global _bert_score_fn, _bert_model_loaded
    if _bert_score_fn is None:
        from bert_score import score as bert_score_fn
        _bert_score_fn = bert_score_fn
    return _bert_score_fn


def warmup_bertscore() -> None:
    """Force-load the BERTScore model (downloads ~1.4GB on first use)."""
    global _bert_model_loaded
    if _bert_model_loaded:
        return
    fn = _get_bert_score_fn()
    # Tiny dummy call to trigger model load + cache
    _, _, _ = fn(["a stationary vehicle ahead."], ["a parked car ahead."], lang="en", verbose=False)
    _bert_model_loaded = True


# ---------------------------------------------------------------------------
# Weights (sum to 1.0)
# ---------------------------------------------------------------------------
W_VERDICT = 0.30
W_ALIGNMENT = 0.45
W_LENGTH = 0.25
LENGTH_LIMIT = 150


@dataclass
class ScoreBreakdown:
    composite: float
    verdict: float
    alignment: float
    length: float
    word_count: int

    def to_dict(self) -> Dict:
        return {
            "composite": round(self.composite, 4),
            "verdict": round(self.verdict, 4),
            "alignment": round(self.alignment, 4),
            "length": round(self.length, 4),
            "word_count": self.word_count,
        }


def score_one(
    pred_verdict: Optional[str],
    pred_reasoning: Optional[str],
    gt_verdict: str,
    gt_reasoning_en: str,
) -> ScoreBreakdown:
    """Score a single prediction against GT.

    Args:
        pred_verdict: model's collision_verdict ("YES" / "NO" / None)
        pred_reasoning: model's verdict_reasoning string
        gt_verdict: ground-truth verdict ("YES" / "NO")
        gt_reasoning_en: ground-truth reasoning, translated English

    Returns:
        ScoreBreakdown with composite + per-component scores
    """
    # 1. Verdict component (binary)
    if pred_verdict is None or pred_reasoning is None:
        # Parse failure — score everything 0
        return ScoreBreakdown(
            composite=0.0, verdict=0.0, alignment=0.0, length=0.0, word_count=0,
        )

    verdict_score = 1.0 if pred_verdict.upper() == gt_verdict.upper() else 0.0

    # 2. Alignment component (BERTScore F1)
    if not gt_reasoning_en or not pred_reasoning.strip():
        alignment_score = 0.0
    else:
        fn = _get_bert_score_fn()
        # bert-score returns precision, recall, F1 as 1-D tensors
        _, _, f1 = fn([pred_reasoning], [gt_reasoning_en], lang="en", verbose=False)
        alignment_score = float(f1[0])

    # 3. Length component (soft penalty above LENGTH_LIMIT words)
    word_count = len(pred_reasoning.split())
    if word_count <= LENGTH_LIMIT:
        length_score = 1.0
    else:
        # Linear decay: 0.0 at word_count = 2*LENGTH_LIMIT
        length_score = max(0.0, 1.0 - (word_count - LENGTH_LIMIT) / LENGTH_LIMIT)

    # Composite
    composite = W_VERDICT * verdict_score + W_ALIGNMENT * alignment_score + W_LENGTH * length_score

    return ScoreBreakdown(
        composite=composite,
        verdict=verdict_score,
        alignment=alignment_score,
        length=length_score,
        word_count=word_count,
    )


def score_batch(
    predictions: list[Dict],
    ground_truths: list[Dict],
) -> list[ScoreBreakdown]:
    """Score a batch of predictions against GTs (1:1 alignment by index).

    Each prediction dict must have keys: 'collision_verdict', 'verdict_reasoning'
    Each ground_truth dict must have keys: 'gt_verdict', 'gt_reasoning_en'
    """
    if len(predictions) != len(ground_truths):
        raise ValueError(f"length mismatch: {len(predictions)} vs {len(ground_truths)}")
    return [
        score_one(
            pred.get("collision_verdict"),
            pred.get("verdict_reasoning"),
            gt["gt_verdict"],
            gt["gt_reasoning_en"],
        )
        for pred, gt in zip(predictions, ground_truths)
    ]


def mean_composite(scores: list[ScoreBreakdown]) -> float:
    """Mean composite score across a list of ScoreBreakdown."""
    if not scores:
        return 0.0
    return sum(s.composite for s in scores) / len(scores)


# ---------------------------------------------------------------------------
# Train-only metric (verdict + length, NO alignment)
# ---------------------------------------------------------------------------
# Used in --mode v11scale where train clips have only verdict labels (no GT reasoning)
W_TRAIN_VERDICT = 0.70
W_TRAIN_LENGTH = 0.30


def score_train_only(
    pred_verdict: Optional[str],
    pred_reasoning: Optional[str],
    gt_verdict: str,
) -> ScoreBreakdown:
    """Score for v11 train clips (no GT reasoning available).

    Returns ScoreBreakdown with composite = 0.70*verdict + 0.30*length.
    The 'alignment' field is set to 0.0 (not used in this metric).
    """
    if pred_verdict is None or pred_reasoning is None:
        return ScoreBreakdown(composite=0.0, verdict=0.0, alignment=0.0, length=0.0, word_count=0)

    verdict_score = 1.0 if pred_verdict.upper() == gt_verdict.upper() else 0.0

    word_count = len(pred_reasoning.split())
    if word_count <= LENGTH_LIMIT:
        length_score = 1.0
    else:
        length_score = max(0.0, 1.0 - (word_count - LENGTH_LIMIT) / LENGTH_LIMIT)

    composite = W_TRAIN_VERDICT * verdict_score + W_TRAIN_LENGTH * length_score
    return ScoreBreakdown(
        composite=composite,
        verdict=verdict_score,
        alignment=0.0,  # not used
        length=length_score,
        word_count=word_count,
    )


# ---------------------------------------------------------------------------
# Regression metric: pure binary verdict accuracy (mean across clips)
# ---------------------------------------------------------------------------

def verdict_accuracy(predictions: list, ground_truths: list) -> float:
    """Mean binary verdict accuracy across a batch.

    Each prediction must be a string ("YES"/"NO"/None).
    Each ground truth must be a string ("YES"/"NO").
    Returns 0.0 if predictions is empty.
    """
    if not predictions:
        return 0.0
    correct = 0
    for pv, gv in zip(predictions, ground_truths):
        if pv is None:
            continue  # parse failure counts as wrong
        if pv.upper() == gv.upper():
            correct += 1
    return correct / len(predictions)


# ---------------------------------------------------------------------------
# CLI sanity test (optional)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Loading BERTScore model (one-time download ~1.4GB if not cached)...")
    warmup_bertscore()
    print("Done. Running sanity test...\n")

    # Test 1: perfect match (verdict + reasoning aligned)
    s1 = score_one(
        pred_verdict="YES",
        pred_reasoning="The ego vehicle is approaching a stationary SUV at high speed with no evasive path.",
        gt_verdict="YES",
        gt_reasoning_en="EGO closing on stopped SUV at high velocity, no lateral escape route available.",
    )
    print(f"Test 1 (good match): composite={s1.composite:.3f}  verdict={s1.verdict}  alignment={s1.alignment:.3f}  length={s1.length}")

    # Test 2: wrong verdict
    s2 = score_one(
        pred_verdict="NO",
        pred_reasoning="The vehicle ahead is far away. Safe.",
        gt_verdict="YES",
        gt_reasoning_en="EGO closing on stopped SUV at high velocity, no lateral escape route available.",
    )
    print(f"Test 2 (wrong verdict): composite={s2.composite:.3f}  verdict={s2.verdict}  alignment={s2.alignment:.3f}  length={s2.length}")

    # Test 3: too long
    long_text = "word " * 200
    s3 = score_one(
        pred_verdict="YES",
        pred_reasoning=long_text,
        gt_verdict="YES",
        gt_reasoning_en="Stopped vehicle ahead.",
    )
    print(f"Test 3 (too long): composite={s3.composite:.3f}  verdict={s3.verdict}  alignment={s3.alignment:.3f}  length={s3.length}  words={s3.word_count}")
