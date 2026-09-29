"""
lookahead.py  (Stage 2 of the 1.5s plan)
=========================================
Look-Ahead head: from the pooled 1024-d vector z of the CURRENT window (the only thing the crash
head sees), predict the pooled vector of the window 0.5 s later in the same video, and let the
frozen crash classifier vote on that predicted future too.

    z_hat  = z + MLP(z)                      (last layer zero-init: z_hat == z at step 0)
    logits = logits(z) + g * classifier(z_hat)   (g learnable scalar, init 0: model == BADAS-Open at step 0)

Training-only supervision: MSE(z_hat, z_future) against a precomputed FROZEN BADAS-Open vector
(features.npz from lookahead_extract_features.py), for windows that have a 0.5 s-later partner.
Inference needs the current window only.
"""
import torch
import torch.nn as nn


class LookAheadHead(nn.Module):
    def __init__(self, dim: int = 1024, hidden: int = 256, gate_frozen: bool = False):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.GELU(),
                                 nn.Linear(hidden, dim))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.g = nn.Parameter(torch.zeros(1), requires_grad=not gate_frozen)

    def predict(self, z: torch.Tensor) -> torch.Tensor:
        z = z.float()
        return z + self.net(z)


# ----------------------------------------------------------------------------------------------
# Pair construction (pure functions, no GPU) - unit-testable locally.
# ----------------------------------------------------------------------------------------------
import random
import re
from collections import defaultdict

import numpy as np

_DIR = re.compile(r"^(?P<vid>.+)_hires_(?P<kind>tte|midtest)(?P<hz>05|10|15)$")
_NEXT = {"15": "10", "10": "05"}


def partner_name(frames_dir: str):
    """Directory of the window 0.5 s later in the same video (1.5->1.0, 1.0->0.5); None for 0.5 s."""
    m = _DIR.match(frames_dir)
    if not m or m["hz"] == "05":
        return None
    return f"{m['vid']}_hires_{m['kind']}{_NEXT[m['hz']]}"


def load_features(npz_path):
    f = np.load(npz_path, allow_pickle=True)
    return {str(n): z.astype(np.float32) for n, z in zip(f["names"], f["z"])}


def build_pair_targets(examples, feats, shuffle: bool = False, seed: int = 0):
    """{frames_dir: target vector}, copy_scale, n_no_features.

    target = frozen-BADAS-Open vector of the partner window. copy_scale = mean squared distance
    between a window's vector and its partner's (the "just copy it" error), so the training loss
    MSE/copy_scale reads like the Stage-0b rel_mse: 1.0 = no better than copying.
    shuffle=True (control): each window gets the partner-vector of a DIFFERENT video with the same
    class and horizon step, by a fixed seeded derangement.
    """
    pairs, miss = {}, 0
    for ex in examples:
        p = partner_name(ex["frames_dir"])
        if p is None:
            continue
        if p not in feats or ex["frames_dir"] not in feats:
            miss += 1
            continue
        pairs[ex["frames_dir"]] = (p, ex["label"])
    copy_scale = float(np.mean([np.mean((feats[p] - feats[fd]) ** 2) for fd, (p, _) in pairs.items()])) if pairs else 1.0
    targets = {fd: feats[p] for fd, (p, _) in pairs.items()}
    if shuffle and targets:
        groups = defaultdict(list)
        for fd, (_, lab) in pairs.items():
            groups[(lab, _DIR.match(fd)["hz"])].append(fd)
        rng = random.Random(seed)
        shuffled = {}
        for fds in groups.values():
            order = sorted(fds)
            rng.shuffle(order)
            for i, fd in enumerate(order):
                shuffled[fd] = targets[order[(i + 1) % len(order)]]
        targets = shuffled
    return targets, copy_scale, miss
