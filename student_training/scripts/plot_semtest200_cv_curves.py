"""
plot_semtest200_cv_curves.py
==============================
Loss-vs-epoch figure for the SemTest-200-v2 cross-validated runs (4 arms x up to 5
folds, <cv-root>/<arm>/fold_XX/epoch_metrics.jsonl).

WHY NOT OVERLAY ALL FOLD LINES
-------------------------------
Once multiple folds exist, plotting every fold's train/val crash_loss AND sem_loss
line on one panel is 4 series x 5 folds = 20 lines per arm panel - unreadable, and
the reader cannot tell "this arm is noisy across folds" from "this arm is
inconsistent within a fold" at a glance.

Fixed instead to MEAN +- 1 STD ACROSS FOLDS: one line per series (train crash_loss,
val crash_loss, train sem_loss, val sem_loss), computed epoch-by-epoch from however
many folds have completed so far, with a shaded +-1 std band. At fold 1 the band has
zero width (a single fold has no spread) and thickens as more folds land - so this
script is safe to re-run after every fold without producing a misleading band from
too little data; it also means the SAME script and SAME reading convention serves the
"stop after fold 1 to check status" checkpoint and the final 5-fold figure, no mode
switch needed.

Shares the shared-y-axis fix from plot_semtest200_curves.py (one global crash_loss
range and one global sem_loss range across all 4 arm panels, vision gets an empty
twin axis) for the same reason: panels must be visually comparable, not just
individually pretty.

Usage:
  python plot_semtest200_cv_curves.py --cv-root ../../outputs/semtest200_v2/results \
      --out-dir ../../outputs/semtest200_v2/figures
"""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]

DEFAULT_ARMS = ["vision", "v10", "v12", "v12shuf"]

# Display titles for arm dir names this script has seen before. An arm not listed here
# (e.g. a1cont) falls back to using its dir name as-is - not an error, just plainer.
TITLES = {
    "vision": "vision-only (crash-loss control)",
    "v10": "v10 - InfoNCE, V10 captions (leaky)",
    "v12": "v12 - InfoNCE, V12 captions (clean)",
    "v12shuf": "v12shuf - InfoNCE, V12 shuffled (content-destroyed)",
    "a1cont": "a1cont - crash-only, continued from A1 (control)",
}

BG = "#1C2340"
FG = "#FFFFFF"
MUTED_C = "#A0B4CC"
TRAIN_C = "#00BFFF"
VAL_C = "#FF6B6B"
SEM_TRAIN_C = "#7CFC98"
SEM_VAL_C = "#FFD166"
SEL_C = "#FFA726"          # selected-checkpoint marker (matches the E3a deck's accent)
INIT_C = "#FF6B6B"         # "A1 = init" reference annotation


def load_fold_epochs(cv_root, arm_dir):
    """One list of epoch-row-lists per completed fold (fold dirs without an
    epoch_metrics.jsonl yet - i.e. not started - are silently skipped, so this is
    safe to call after fold 1 alone)."""
    per_fold = []
    for fold_dir in sorted((cv_root / arm_dir).glob("fold_*")):
        p = fold_dir / "epoch_metrics.jsonl"
        if not p.exists():
            continue
        rows = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
        rows.sort(key=lambda r: r["epoch"])
        if rows:
            per_fold.append(rows)
    return per_fold


def mean_std_series(per_fold, key):
    """(epochs, mean, std) across whatever folds are present, truncated to the
    shortest fold's epoch count if they differ (e.g. a fold still mid-run)."""
    n_epochs = min(len(f) for f in per_fold)
    if n_epochs < len(max(per_fold, key=len)):
        print(f"    [note] folds have unequal length ({[len(f) for f in per_fold]}) - "
              f"truncating the mean/std curve to {n_epochs} epochs (shortest fold)")
    epochs = [per_fold[0][i]["epoch"] for i in range(n_epochs)]
    mat = np.array([[f[i][key] for i in range(n_epochs)] for f in per_fold])
    return epochs, mat.mean(axis=0), (mat.std(axis=0) if len(per_fold) > 1 else np.zeros(n_epochs))


def _padded_range(values, pad_frac=0.05):
    lo, hi = min(values), max(values)
    span = hi - lo
    if span == 0:
        span = abs(hi) if hi else 1.0
    return lo - pad_frac * span, hi + pad_frac * span


def plot_cv_grid(cv_root, out_dir, arm_names, title_prefix="SemTest-200-v2 (CV)",
                 mark_epochs=None, init_note=None):
    plt.rcParams.update({
        "text.color": FG, "axes.labelcolor": FG, "xtick.color": MUTED_C,
        "ytick.color": MUTED_C, "axes.edgecolor": MUTED_C,
    })
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.0))
    fig.patch.set_facecolor(BG)
    mark_epochs = mark_epochs or {}

    all_folds = {}
    for arm_dir in arm_names:
        pf = load_fold_epochs(cv_root, arm_dir)
        all_folds[arm_dir] = pf
        print(f"  {arm_dir}: {len(pf)} fold(s) with data "
              f"({[len(f) for f in pf]} epochs each)")

    n_folds_seen = {len(pf) for pf in all_folds.values() if pf}
    if not n_folds_seen:
        raise SystemExit(f"no epoch_metrics.jsonl found anywhere under {cv_root} - "
                          f"has any fold finished at least 1 epoch?")

    # has_sem is auto-detected from the data (any nonzero sem_loss across all loaded
    # epochs) rather than hardcoded per arm name - self-describing, so a new arm name
    # (e.g. a1cont) needs no entry anywhere to be classified correctly.
    def _has_sem(pf):
        return any(r.get("sem_loss", 0.0) != 0.0 for fold in pf for r in fold)

    ARMS = [(arm_dir, TITLES.get(arm_dir, arm_dir), _has_sem(all_folds[arm_dir]))
            for arm_dir in arm_names]

    # ---- one global y-range for crash_loss, one for sem_loss, shared by every panel ----
    crash_vals, sem_vals = [], []
    for arm_dir, _, has_sem in ARMS:
        pf = all_folds[arm_dir]
        if not pf:
            continue
        _, m_tr, s_tr = mean_std_series(pf, "crash_loss")
        _, m_va, s_va = mean_std_series(pf, "val_crash_loss")
        crash_vals += list(m_tr - s_tr) + list(m_tr + s_tr) + list(m_va - s_va) + list(m_va + s_va)
        if has_sem:
            _, m_tr2, s_tr2 = mean_std_series(pf, "sem_loss")
            _, m_va2, s_va2 = mean_std_series(pf, "val_sem_loss")
            sem_vals += list(m_tr2 - s_tr2) + list(m_tr2 + s_tr2) + list(m_va2 - s_va2) + list(m_va2 + s_va2)
    crash_ylim = _padded_range(crash_vals)
    sem_ylim = _padded_range(sem_vals) if sem_vals else (0.0, 1.0)

    for ax, (arm_dir, title, has_sem) in zip(axes.ravel(), ARMS):
        ax.set_facecolor(BG)
        for sp in ax.spines.values():
            sp.set_color(MUTED_C)
        pf = all_folds[arm_dir]
        n_folds = len(pf)

        if pf:
            ep, m, s = mean_std_series(pf, "crash_loss")
            ax.plot(ep, m, "o-", color=TRAIN_C, label="Train crash_loss (mean)",
                     markersize=4.5, linewidth=1.7)
            ax.fill_between(ep, m - s, m + s, color=TRAIN_C, alpha=0.18, linewidth=0)
            ep, m, s = mean_std_series(pf, "val_crash_loss")
            ax.plot(ep, m, "s-", color=VAL_C, label="Val crash_loss (mean)",
                     markersize=4.5, linewidth=1.7)
            ax.fill_between(ep, m - s, m + s, color=VAL_C, alpha=0.18, linewidth=0)
        ax.set_ylim(*crash_ylim)

        ax2 = ax.twinx()
        ax2.set_facecolor(BG)
        ax2.set_ylim(*sem_ylim)
        if has_sem and pf:
            ep, m, s = mean_std_series(pf, "sem_loss")
            ax2.plot(ep, m, "^--", color=SEM_TRAIN_C, label="Train sem_loss (mean)",
                      markersize=4, linewidth=1.3, alpha=0.9)
            ax2.fill_between(ep, m - s, m + s, color=SEM_TRAIN_C, alpha=0.15, linewidth=0)
            ep, m, s = mean_std_series(pf, "val_sem_loss")
            ax2.plot(ep, m, "v--", color=SEM_VAL_C, label="Val sem_loss (mean)",
                      markersize=4, linewidth=1.3, alpha=0.9)
            ax2.fill_between(ep, m - s, m + s, color=SEM_VAL_C, alpha=0.15, linewidth=0)
        # RIGHT-AXIS LEGIBILITY (2026-08-29): previously this axis was drawn in the same
        # muted grey as the left one with all spines hidden, so a reader could not tell
        # which curves belonged to which scale. Now the right spine is drawn in the
        # semantic-series colour and the ticks/label match it, so the axis visually keys
        # to the two dashed green/amber curves. Left axis is keyed to the crash colour
        # the same way. No data or limit logic changed - legibility only.
        # An arm with no semantic branch (the crash-only control) keeps the twin axis so
        # all panels share identical geometry, but its ticks are BLANKED and the label
        # says so - otherwise it shows a populated semantic scale with no curve on it,
        # which reads as a missing curve rather than an absent branch.
        for side, sp in ax2.spines.items():
            sp.set_visible(side == "right")
            if side == "right":
                sp.set_color(SEM_TRAIN_C if has_sem else MUTED_C)
                sp.set_linewidth(1.6 if has_sem else 0.8)
        if has_sem:
            ax2.set_ylabel("InfoNCE sem_loss  (RIGHT axis, dashed curves)", fontsize=8.5,
                           color=SEM_TRAIN_C, fontweight="bold")
            ax2.tick_params(labelsize=8.5, colors=SEM_TRAIN_C)
        else:
            ax2.set_ylabel("no semantic branch", fontsize=8.5, color=MUTED_C,
                           style="italic")
            ax2.set_yticklabels([])
            ax2.tick_params(axis="y", length=0)

        band_note = "band = +-1 std, single fold" if n_folds == 1 else f"band = +-1 std, {n_folds} folds"
        ax.set_title(f"{title}\n{band_note}", fontsize=9.5, fontweight="bold", color=TRAIN_C)
        ax.set_xlabel("Epoch", fontsize=10)
        ax.set_ylabel("crash_loss  (LEFT axis, solid curves)", fontsize=9,
                      color=TRAIN_C, fontweight="bold")
        ax.tick_params(axis="y", colors=TRAIN_C)
        ax.spines["left"].set_color(TRAIN_C)
        ax.spines["left"].set_linewidth(1.6)
        ax.grid(alpha=0.18, color=MUTED_C)

        # ---- working-point marker: the checkpoint that was actually test-scored ----
        if arm_dir in mark_epochs and pf:
            mep = mark_epochs[arm_dir]
            eps_all, m_all, _ = mean_std_series(pf, "val_crash_loss")
            if mep in eps_all:
                yv = m_all[eps_all.index(mep)]
                ax.plot([mep], [yv], "o", markersize=17, markerfacecolor="none",
                        markeredgecolor=SEL_C, markeredgewidth=2.4, zorder=10)
                ax.annotate(f"selected -> test set\n(epoch {mep})", xy=(mep, yv),
                            xytext=(-14, 26), textcoords="offset points",
                            fontsize=8, color=SEL_C, fontweight="bold",
                            ha="right", zorder=11)

        h1, l1 = ax.get_legend_handles_labels()
        if has_sem:
            h2, l2 = ax2.get_legend_handles_labels()
            h1, l1 = h1 + h2, l1 + l2
        ax.legend(h1, l1, fontsize=7.5, facecolor="#243060", edgecolor=MUTED_C,
                   labelcolor=FG, loc="upper right")
        ax.tick_params(labelsize=9)

    n_folds_str = "/".join(str(n) for n in sorted(n_folds_seen)) if len(n_folds_seen) > 1 else str(next(iter(n_folds_seen)))
    fig.suptitle(f"{title_prefix}: mean +-1 std across {n_folds_str} fold(s) per arm  -  "
                 "y-axes shared across all 4 panels",
                 fontsize=13, fontweight="bold", color=TRAIN_C, y=0.995)
    if init_note:
        # Stated on the figure because A1 is NOT a fifth arm and has no epoch on this
        # axis - it is the weights every panel starts from. Without saying so, a reader
        # looking for "the A1 curve" will not find one and may assume it was omitted.
        fig.text(0.5, 0.955, init_note, ha="center", va="top", fontsize=9.5,
                 color=INIT_C, style="italic")
    fig.tight_layout(rect=(0, 0, 1, 0.945 if init_note else 0.965))
    out = out_dir / "loss_curves_2x2_cv.png"
    fig.savefig(out, dpi=200, facecolor=BG, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cv-root", default=str(ROOT / "outputs" / "semtest200_v2" / "results"))
    ap.add_argument("--out-dir", default=str(ROOT / "outputs" / "semtest200_v2" / "figures"))
    ap.add_argument("--arms", nargs="+", default=DEFAULT_ARMS,
                     help="arm dir names under --cv-root, e.g. a1cont v10 v12 v12shuf. "
                          "has_sem is auto-detected from each arm's own data.")
    ap.add_argument("--title-prefix", default="SemTest-200-v2 (CV)",
                     help="figure suptitle prefix, e.g. 'A1-failure recovery (321 pool)'")
    ap.add_argument("--mark-epoch", nargs="+", default=None, metavar="ARM=EPOCH",
                     help="circle the checkpoint that was selected for downstream "
                          "evaluation, e.g. --mark-epoch v12=10")
    ap.add_argument("--init-note", default=None,
                     help="italic line under the suptitle, e.g. to say which weights "
                          "every arm was initialized from when that init is not itself "
                          "an epoch on the x-axis")
    args = ap.parse_args()

    mark_epochs = {}
    for spec in (args.mark_epoch or []):
        if "=" not in spec:
            raise SystemExit(f"--mark-epoch entry {spec!r} must be ARM=EPOCH")
        k, v = spec.split("=", 1)
        mark_epochs[k] = int(v)

    cv_root = Path(args.cv_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  wrote {plot_cv_grid(cv_root, out_dir, args.arms, args.title_prefix, mark_epochs, args.init_note)}")


if __name__ == "__main__":
    main()
