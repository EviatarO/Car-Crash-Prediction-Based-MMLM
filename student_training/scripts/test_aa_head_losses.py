"""
test_aa_head_losses.py
=======================
Stage-0 local unit checks for aa_head_losses.py (child plan 2026-09-24-AA-H). No GPU,
no BADAS load - pure synthetic tensors. Run: python test_aa_head_losses.py
"""
import torch

from aa_head_losses import (
    N_REAL, N_TOTAL, token_sets_from_labels, mean_attention_received, rank_loss,
    mass_loss, attention_entropy, gradcam_loss, lambda_schedule,
)

PASS = []


def check(name, cond):
    PASS.append((name, bool(cond)))
    print(f"  [{'OK' if cond else 'FAIL'}] {name}")


def uniform_attn(n=N_TOTAL):
    return torch.full((1, n, n), 1.0 / n)


# ---------------------------------------------------------------------------
print("1) mean_attention_received: rows sum to 1 -> r sums to 1")
a = torch.rand(1, N_TOTAL, N_TOTAL)
a = a / a.sum(dim=-1, keepdim=True)
r = mean_attention_received(a)
check("attn shape (1,2560,2560) accepted, r shape (2560,)", tuple(r.shape) == (N_TOTAL,))
check("r sums to 1", abs(r.sum().item() - 1.0) < 1e-4)
check("2D attn also accepted", tuple(mean_attention_received(a[0]).shape) == (N_TOTAL,))

# ---------------------------------------------------------------------------
print("2) token_sets_from_labels")
rel = torch.zeros(N_REAL)
rel[10:15] = torch.tensor([0.2, 0.5, 1.0, 0.3, 0.1])
occ = torch.zeros(N_REAL)
occ[10:20] = 1.0  # tokens 15-19 are "other vehicle" (occ=1, rel=0)
sets = token_sets_from_labels(rel, occ)
check("token_sets_from_labels returns 3 masks", sets is not None and len(sets) == 3)
P, V, B = sets
check("P has exactly the 5 rel>0 tokens", P.sum().item() == 5)
check("V has exactly the 5 occ=1&rel=0 tokens", V.sum().item() == 5)
check("B has exactly 2048-10 background tokens", B.sum().item() == N_REAL - 10)
check("P/V/B are disjoint", not bool((P & V).any()) and not bool((P & B).any()))

empty_rel = torch.zeros(N_REAL)
check("empty P (all-background window) -> None",
      token_sets_from_labels(empty_rel, occ) is None)
check("positives_only + has_positive=False -> None",
      token_sets_from_labels(rel, occ, positives_only=True, has_positive=False) is None)
check("positives_only + has_positive=True -> normal result",
      token_sets_from_labels(rel, occ, positives_only=True, has_positive=True) is not None)

# ---------------------------------------------------------------------------
print("3) rank_loss: known-answer synthetic cases")
P_real = torch.zeros(N_REAL, dtype=torch.bool)
P_real[0:10] = True
V_real = torch.zeros(N_REAL, dtype=torch.bool)
V_real[10:20] = True
B_real = ~(P_real | V_real)

margin = 0.4
u = uniform_attn()
L_uniform = rank_loss(u, P_real, V_real, B_real, margin)
# uniform attention -> rho_P == rho_V == rho_B -> ln(ratio)=0 -> each term = margin
check(f"uniform attention -> L = 2*margin ({2*margin:.4f})",
      abs(L_uniform.item() - 2 * margin) < 1e-4)

# Attention concentrated entirely on P (well beyond the margin) -> loss should be 0
a2 = torch.zeros(1, N_TOTAL, N_TOTAL)
a2[0, :, 0:10] = 1.0 / 10  # every query takes all its mass from the 10 P tokens
L_concentrated = rank_loss(a2, P_real, V_real, B_real, margin)
check("attention fully on P (way past margin) -> loss == 0", L_concentrated.item() == 0.0)

# Attention concentrated entirely on V (worst case for P) -> should be large and positive
a3 = torch.zeros(1, N_TOTAL, N_TOTAL)
a3[0, :, 10:20] = 1.0 / 10
L_worst = rank_loss(a3, P_real, V_real, B_real, margin, eps=1e-8)
check("attention fully on V (P gets none) -> loss is large and finite",
      L_worst.item() > 5.0 and L_worst.item() == L_worst.item())

L_grad = rank_loss(u.clone().requires_grad_(True), P_real, V_real, B_real, margin)
L_grad.backward()
check("rank_loss is differentiable (uniform case)", True)  # no exception = pass

# ---------------------------------------------------------------------------
print("4) mass_loss: known-answer synthetic cases")
rel_full = torch.zeros(N_REAL)
rel_full[0:10] = 1.0  # 10 fully-relevant tokens, rest 0
L_mass_uniform = mass_loss(uniform_attn(), rel_full)
# p is uniform 1/2048 over real tokens, R_hat is 1 on 10 tokens else 0
# sum p*R_hat = 10/2048
import math
expected = -math.log(10 / N_REAL + 1e-6)
check(f"uniform attention -> L_B ~= -ln(10/2048) ({expected:.4f})",
      abs(L_mass_uniform.item() - expected) < 1e-3)

a4 = torch.zeros(1, N_TOTAL, N_TOTAL)
a4[0, :, 0:10] = 1.0 / 10
L_mass_concentrated = mass_loss(a4, rel_full)
check("attention fully on the 10 relevant tokens -> L_B ~= 0",
      L_mass_concentrated.item() < 1e-3)

ent = attention_entropy(uniform_attn())
check(f"uniform attention over 2560 tokens -> entropy ~= ln(2560) ({math.log(N_TOTAL):.3f})",
      abs(ent.item() - math.log(N_TOTAL)) < 1e-3)
a_onehot = torch.zeros(1, N_TOTAL, N_TOTAL)
a_onehot[0, :, 0] = 1.0
ent0 = attention_entropy(a_onehot)
check("all attention on 1 token -> entropy ~= 0", ent0.item() < 1e-4)

# ---------------------------------------------------------------------------
print("5) gradcam_loss: known-answer synthetic cases")
# Constant-valued tokens (x=1) -> every token that DOES get gradient scores an IDENTICAL
# dot product (1024), so after max-normalization it is exactly 1.0, not just "positive" -
# this gives exact, not approximate, expected alpha/beta values.
x = torch.ones(1, N_TOTAL, 1024)
# VB passed as exactly V (not V|B) to isolate beta's arithmetic from B's dilution - the
# production wiring passes the full V|B set, exercised separately in the pod smoke test.
grad_on_P = torch.zeros(1, N_TOTAL, 1024)
grad_on_P[0, 0:10] = 1.0
L_cam_best = gradcam_loss(grad_on_P, x, P_real, V_real)
check("gradient evidence entirely on P -> L_C == -1 (alpha=1, beta=0)",
      abs(L_cam_best.item() - (-1.0)) < 1e-6)

grad_on_V = torch.zeros(1, N_TOTAL, 1024)
grad_on_V[0, 10:20] = 1.0
L_cam_worst = gradcam_loss(grad_on_V, x, P_real, V_real)
check("gradient evidence entirely on V (none on P) -> L_C == +1 (alpha=0, beta=1)",
      abs(L_cam_worst.item() - 1.0) < 1e-6)

x2 = x.clone().requires_grad_(True)
Lg = gradcam_loss(grad_on_P, x2, P_real, V_real)
Lg.backward()
check("gradcam_loss is differentiable wrt x", x2.grad is not None)

# ---------------------------------------------------------------------------
print("6) lambda_schedule")
check("epoch_frac=0 -> lambda=0", lambda_schedule(0.0, 1.0, warmup_frac=0.5) == 0.0)
check("epoch_frac=0.25 (mid-warmup) -> lambda=0.5*max",
      abs(lambda_schedule(0.25, 1.0, warmup_frac=0.5) - 0.5) < 1e-6)
check("epoch_frac=0.5 (end of warmup) -> lambda=max",
      abs(lambda_schedule(0.5, 1.0, warmup_frac=0.5) - 1.0) < 1e-6)
check("epoch_frac=2.9 (within on_epochs=3) -> lambda=max",
      lambda_schedule(2.9, 1.0, warmup_frac=0.5, on_epochs=3.0) == 1.0)
check("epoch_frac=3.1 (past on_epochs=3) -> lambda=0",
      lambda_schedule(3.1, 1.0, warmup_frac=0.5, on_epochs=3.0) == 0.0)
check("on_epochs=None -> stays on for the whole run",
      lambda_schedule(7.9, 1.0, warmup_frac=0.5, on_epochs=None) == 1.0)

# ---------------------------------------------------------------------------
n_fail = sum(1 for _, ok in PASS if not ok)
print(f"\n{len(PASS)} checks, {n_fail} failed.")
if n_fail:
    raise SystemExit(1)
print("ALL PASS")
