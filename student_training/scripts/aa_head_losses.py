"""
aa_head_losses.py
==================
Stage AA-H (child plan 2026-09-24-AA-H-head-attention-supervision): pure, model-free
loss functions that supervise what BADAS's crash HEAD itself attends to / bases its
decision on, instead of a side probe on an intermediate ViT-L layer (the six prior
Stage AA runs - see 2026-09-19 child plan's status block - showed a side-probe loss at
layer 17 teaches something real there but it never reaches the head).

All three losses take token-level 0/1 SET MASKS (not the raw relevance label - see the
plan's "Are we measuring the right thing?" section for why R is used only as WHERE to
look, never as a crash signal), so they never depend on aa4_token_labels.py's exact
scoring formula. Kept deliberately free of any torch-model import so every function is
unit-testable on synthetic tensors with no GPU / BADAS load (Stage-0 verification).

Token layout convention throughout this file: index 0..2047 = the 2048 REAL tokens (8
tubelets x 16x16, same flatten order as aa1_token_probe.py / aa4_token_labels.py);
index 2048..2559 = the 512 PREDICTOR ("future") tokens, which have no label and are
never in P/V/B.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

N_REAL = 2048
N_TOTAL = 2560


def token_sets_from_labels(rel: torch.Tensor, occ: torch.Tensor,
                            positives_only: bool = False, has_positive: bool = True):
    """rel, occ: (2048,) float tensors (aa4_token_labels.py's `rel`/`occ` arrays, or the
    hindsight `partner` array used as `rel` under --aux-label partner_pos - see the plan).
    Returns boolean masks (P, V, B), each (2048,), or None if this window should be
    skipped (positives_only and not has_positive, or P is empty - an aux loss over an
    empty P is undefined and must not silently become a background-only loss).

    P = relevant-car tokens (rel > 0)
    V = other-vehicle tokens (occ == 1 and rel == 0)
    B = background tokens (occ == 0)
    """
    if positives_only and not has_positive:
        return None
    P = rel > 0
    if not bool(P.any()):
        return None
    V = (occ > 0) & (~P)
    B = occ <= 0
    return P, V, B


def _pad_real_to_total(mask_real: torch.Tensor, n_total: int = N_TOTAL) -> torch.Tensor:
    """(2048,) bool -> (n_total,) bool, False on the 512 predictor tokens."""
    pad = torch.zeros(n_total - mask_real.shape[0], dtype=torch.bool, device=mask_real.device)
    return torch.cat([mask_real, pad])


def mean_attention_received(attn: torch.Tensor) -> torch.Tensor:
    """attn: (1, N, N) or (N, N) row-normalized attention weights (row q sums to 1: how
    much query q takes from every key). Returns r: (N,) = attention RECEIVED per token,
    averaged over queries (column mean) - this is what the head's subsequent mean-pool
    over queries turns into "how much this token influences the final decision".
    r sums to 1 (mean of N rows that each sum to 1)."""
    if attn.dim() == 3:
        attn = attn[0]
    return attn.mean(dim=0)


def rank_loss(attn: torch.Tensor, P_real: torch.Tensor, V_real: torch.Tensor,
              B_real: torch.Tensor, margin: float, eps: float = 1e-8) -> torch.Tensor:
    """Variant A (RARE-style ranking). Relevant tokens (P) must receive at least e^margin
    times the attention of other-vehicle tokens (V) and of background tokens (B):
        L = max(0, margin - ln(rho_P/rho_V)) + max(0, margin - ln(rho_P/rho_B))
    where rho_S = mean attention received over token set S. A term is 0 (no gradient,
    cannot over-push once satisfied) whenever S is empty or the margin is already met.
    attn: (1,2560,2560) or (2560,2560). P_real/V_real/B_real: (2048,) bool."""
    r = mean_attention_received(attn)  # (2560,)
    P = _pad_real_to_total(P_real, r.shape[0])
    V = _pad_real_to_total(V_real, r.shape[0])
    B = _pad_real_to_total(B_real, r.shape[0])
    rho_P = r[P].mean()
    terms = []
    if bool(V.any()):
        rho_V = r[V].mean()
        terms.append(F.relu(margin - torch.log(rho_P + eps) + torch.log(rho_V + eps)))
    if bool(B.any()):
        rho_B = r[B].mean()
        terms.append(F.relu(margin - torch.log(rho_P + eps) + torch.log(rho_B + eps)))
    if not terms:
        return torch.zeros((), device=attn.device, dtype=r.dtype)
    return torch.stack(terms).sum()


def mass_loss(attn: torch.Tensor, rel_real: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Variant B (FAX-style attention mass). Raise the share of the head's attention
    that lands on relevant tokens, weighted by how relevant (soft `rel`, not the P/V/B
    sets): L = -ln( sum_k p_k * R_hat_k + eps ), where p = r renormalized over the 2048
    REAL tokens only (the 512 predictor tokens are excluded from the competition since
    they have no relevance label) and R_hat = rel / max(rel). Always positive-valued,
    minimized at 0 only when all mass sits on the single highest-relevance token - the
    on/off lambda schedule and attention-entropy logging (see semsup_train.py) exist to
    stop this from collapsing attention onto one patch."""
    r = mean_attention_received(attn)  # (2560,)
    r_real = r[:N_REAL]
    p = r_real / (r_real.sum() + eps)
    r_hat = rel_real / (rel_real.max() + eps)
    return -torch.log((p * r_hat).sum() + eps)


def attention_entropy(attn: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Shannon entropy (nats) of r = mean_attention_received(attn), a monitoring-only
    diagnostic (logged every epoch, not optimized) for mass_loss's collapse risk."""
    r = mean_attention_received(attn)
    p = r / (r.sum() + eps)
    return -(p * torch.log(p + eps)).sum()


def gradcam_loss(grad_x: torch.Tensor, x: torch.Tensor, P_real: torch.Tensor,
                  VB_real: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Variant C (GAIN/CAMAL-style decision-map alignment). grad_x = d(crash logit)/dX,
    x = the head's own input X (both (1,2560,1024) or (2560,1024) - the SAME tensor:
    BADAS's TrainableBadasWrapper already captures X as `_captured["patches"]`, the
    pre-hook on temporal_processor). c_k = ReLU(sum_d grad_x_kd * x_kd), a per-token
    Grad-CAM-style relevance-to-the-decision score, normalized to [0,1] by its own max.
    alpha = mean c over P (relevant), beta = mean c over V union B (everything else).
    L = beta - alpha, range [-1, 1], minimized by making the decision's own evidence
    concentrate on the relevant tokens. ALWAYS positives-only by construction (the
    caller must not call this on a window with no crash-partner labels) - see the plan's
    "R is used only as WHERE to look" note: this loss explicitly shapes what counts as
    evidence FOR a crash, so it must never see a normal (no-crash) window."""
    if grad_x.dim() == 3:
        grad_x = grad_x[0]
    if x.dim() == 3:
        x = x[0]
    c_full = F.relu((grad_x * x).sum(dim=-1))  # (2560,)
    c_real = c_full[:N_REAL]
    c_real = c_real / (c_real.max() + eps)
    VB = _pad_real_to_total(VB_real, N_REAL)[:N_REAL] if VB_real.shape[0] == N_REAL else VB_real
    alpha = c_real[P_real].mean() if bool(P_real.any()) else torch.zeros((), device=x.device)
    beta = c_real[VB].mean() if bool(VB.any()) else torch.zeros((), device=x.device)
    return beta - alpha


def lambda_schedule(epoch_frac: float, lam_max: float, warmup_frac: float = 0.5,
                     on_epochs: float | None = 3.0, total_epochs: float = 8.0) -> float:
    """epoch_frac: continuous progress through the run in EPOCHS (e.g. 1.3 = 30% through
    epoch 2). Ramps 0 -> lam_max linearly over the first `warmup_frac` of epoch 1 (the
    defect-localization paper's ramp, cheaper to reason about than a cosine), holds
    lam_max through `on_epochs`, then drops to exactly 0 for the rest of the run (HASTE /
    REPA's early-stop finding: the aux gradient helps early and turns into a brake once
    the crash-vs-aux cosine drifts toward 0 - see this project's own grad_trace.jsonl for
    AA-rel/AA-occ). on_epochs=None or >= total_epochs = always on after warmup (the old
    AA-rel/AA-occ behavior, for A/B comparison)."""
    if epoch_frac < warmup_frac:
        return lam_max * (epoch_frac / warmup_frac) if warmup_frac > 0 else lam_max
    if on_epochs is not None and epoch_frac >= on_epochs:
        return 0.0
    return lam_max
