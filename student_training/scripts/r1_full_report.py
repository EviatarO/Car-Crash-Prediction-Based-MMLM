"""
r1_full_report.py - the full metrics report of P2V (verdict-first reasoning model) against the crash model it is built on (plan 2026-10-18, Part A).
Local, no GPU. Reads existing prediction files; whatever is missing (e.g. the train sample before the pod run) is reported as "pending", never invented.

Splits    train sample (600, pod)  |  val (524)  |  private test (677)  |  public test (667)  |  pooled test (1,344)
Scores    crash model: a1_p (probability, yes at P >= 0.5);  P2V: s_first = log P(" yes") - log P(" no") at "Collision:" (yes at score >= 0, i.e. P = 0.5)
TTE groups  a window set per TTE k = the crash windows of TTE k + the normal windows they are compared with. Test files: the normal windows carry a TTE
          (a fake-event horizon) and are paired by horizon. Val / train: normal windows carry no TTE, so a TTE group there is its crash windows vs ALL normal windows.
          Kaggle mAP = the mean of the three per-TTE APs (the BADAS papers' metric). Every thresholded number says its threshold.

Writes <out>/report.md, metrics.json, figures/*.png.     python r1_full_report.py [--out DIR] [--boot 1000]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import average_precision_score as AP, precision_recall_curve, roc_auc_score as AUC, roc_curve  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "r1_p2v_report_2026-10-18"
P2F = ROOT / "outputs" / "r1_phase2f_2026-10-15" / "phase2f_2026-10-15"
TEST_P = ROOT / "outputs" / "r1_test_eval_2026-10-17" / "test_eval_2026-10-17" / "P2V" / "results" / "predictions.jsonl"
PUB_P = ROOT / "outputs" / "r1_test_eval_public_2026-10-18" / "test_eval_public_2026-10-18" / "P2V" / "results" / "predictions.jsonl"
LOGS = {"P1aug": ROOT / "outputs" / "r1_phase1aug_2026-10-14" / "phase1aug_2026-10-14" / "P1aug" / "train_log.jsonl",
        "P1fixR": P2F / "P1fixR" / "train_log.jsonl", "P2V": P2F / "P2V" / "train_log.jsonl"}
TTES = (0.5, 1.0, 1.5)
C_CRASH, C_P2V = "#4C78A8", "#E45756"
SLOTS = ("road_user", "side", "motion", "gap")


def jl(p):
    p = Path(p)
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()] if p.exists() else None


class Split:
    """one evaluation split: arrays + the window masks of the TTE groups."""

    def __init__(self, name, rows):
        self.name, self.rows, self.n = name, rows, len(rows)
        self.y = np.array([r["label"] for r in rows])
        self.group = np.array([r["group"] for r in rows])
        self.tte = np.array([r["tte"] if r.get("tte") is not None else np.nan for r in rows], float)
        self.crash = np.array([r["a1_p"] for r in rows], float)
        self.p2v = np.array([r["s_first"] for r in rows], float)
        paired = bool(((self.y == 0) & ~np.isnan(self.tte)).any())
        self.paired = paired
        self.masks = {}
        for k in TTES:
            crash_k = (self.y == 1) & np.isclose(self.tte, k)
            normal = (self.y == 0) & (np.isclose(self.tte, k) if paired else True)
            self.masks[k] = crash_k | normal
        self.masks["all"] = np.ones(self.n, bool)

    def score(self, model):
        return self.crash if model == "crash model" else self.p2v

    def thr(self, model, p2v_thr=0.0):
        return 0.5 if model == "crash model" else p2v_thr


def pool(a, b, name):
    return Split(name, a.rows + b.rows)


def ap_auc(y, s):
    return (AP(y, s), AUC(y, s)) if 0 < y.sum() < len(y) else (float("nan"), float("nan"))


def metrics(sp, model):
    s, out = sp.score(model), {}
    for k in ("all",) + TTES:
        m = sp.masks[k]
        out[k] = dict(zip(("AP", "AUC"), ap_auc(sp.y[m], s[m])))
        out[k]["n_crash"], out[k]["n_normal"] = int(sp.y[m].sum()), int((1 - sp.y[m]).sum())
    out["kaggle_mAP"] = float(np.mean([out[k]["AP"] for k in TTES]))
    thr = np.quantile(s[sp.y == 1], 0.15), np.quantile(s[sp.y == 1], 0.10)
    out["FPR_at_85_recall"], out["FPR_at_90_recall"] = (float((s[sp.y == 0] >= thr[0]).mean()), float((s[sp.y == 0] >= thr[1]).mean()))
    return out


def conf(sp, model, thr, mask):
    s, y = sp.score(model)[mask], sp.y[mask]
    yes = s >= thr
    tp, fn, fp, tn = int((yes & (y == 1)).sum()), int((~yes & (y == 1)).sum()), int((yes & (y == 0)).sum()), int((~yes & (y == 0)).sum())
    rec, spec, prec = tp / max(1, tp + fn), tn / max(1, tn + fp), tp / max(1, tp + fp)
    return dict(TP=tp, FN=fn, FP=fp, TN=tn, recall=rec, specificity=spec, precision=prec, F1=2 * prec * rec / max(1e-9, prec + rec))


def boot_delta(sp, n=1000, seed=0):
    """paired bootstrap over windows of P2V - crash model: AP, AUC, Kaggle mAP (the same resampled windows for both models)."""
    rng = np.random.default_rng(seed)
    d = {"AP": [], "AUC": [], "kaggle_mAP": []}
    for _ in range(n):
        i = rng.integers(0, sp.n, sp.n)
        if len(set(sp.y[i])) < 2:
            continue
        ok, per = True, ([], [])
        for k in TTES:
            m = sp.masks[k][i]
            if len(set(sp.y[i][m])) < 2:
                ok = False
                break
            per[0].append(AP(sp.y[i][m], sp.p2v[i][m]))
            per[1].append(AP(sp.y[i][m], sp.crash[i][m]))
        if not ok:
            continue
        d["AP"].append(AP(sp.y[i], sp.p2v[i]) - AP(sp.y[i], sp.crash[i]))
        d["AUC"].append(AUC(sp.y[i], sp.p2v[i]) - AUC(sp.y[i], sp.crash[i]))
        d["kaggle_mAP"].append(float(np.mean(per[0]) - np.mean(per[1])))
    return {k: (float(np.mean(v)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in d.items()}


def train_threshold(tr):
    """log-odds threshold for P2V: max balanced accuracy on the train sample (a deployment number; P2V saw these windows in training)."""
    s, y = tr.p2v, tr.y
    cand = np.unique(s)
    bal = [0.5 * ((s[y == 1] >= t).mean() + (s[y == 0] < t).mean()) for t in cand]
    return float(cand[int(np.argmax(bal))]), float(max(bal))


# ----------------------------------------------------------------------------- text
def card_acc(model_cards, teacher_cards, rows=None):
    """fact accuracy per slot: card(model text) == card(teacher text), teacher slot 'not_stated' skipped; majority-guess baseline of the same pairs."""
    out = {}
    for s in SLOTS:
        pairs = [(teacher_cards[i][s], c[s]) for i, c in model_cards.items() if i in teacher_cards and teacher_cards[i][s] != "not_stated"
                 and (rows is None or i in rows)]
        if not pairs:
            continue
        maj = max(set(t for t, _ in pairs), key=[t for t, _ in pairs].count)
        out[s] = {"n": len(pairs), "acc": float(np.mean([t == m for t, m in pairs])), "baseline": float(np.mean([t == maj for t, _ in pairs]))}
    return out


def cards(path):
    r = jl(path)
    return None if r is None else {x["id"]: x for x in r}


def read_json(p):
    p = Path(p)
    return json.load(open(p, encoding="utf-8")) if p.exists() else None


def written(rows):
    ok = [(r["gen_collision"], r["label"], r["s_first"]) for r in rows if r.get("gen_collision") is not None]
    return {"verdict_acc_vs_label": float(np.mean([w == y for w, y, _ in ok])), "verdict_vs_score_agree": float(np.mean([w == int(s >= 0) for w, _, s in ok])),
            "distinct": len({r["gen_explanation"] for r in rows}) / len(rows), "n": len(rows)}


# ----------------------------------------------------------------------------- figures
def fig_loss(fig_dir):
    logs = {k: jl(p) for k, p in LOGS.items()}
    fig, ax = plt.subplots(2, 3, figsize=(13, 6.5))
    for j, (name, rows) in enumerate(logs.items()):
        if not rows:
            continue
        ep = [r["epoch"] for r in rows]
        v = [r["val_nexar_reasoning"]["loss_real"] for r in rows]
        ax[0, j].plot(ep, [r["train_loss"] for r in rows], "o-", color=C_CRASH, label="train loss (epoch mean)")
        ax[0, j].plot(ep, v, "s-", color=C_P2V, label="val loss (teacher text)")
        b = int(np.argmin(v))
        ax[0, j].axvline(ep[b], color="gray", ls=":", label=f"best val loss: epoch {ep[b]}")
        ax[0, j].set_title(f"{name}: loss vs epoch"), ax[0, j].set_xlabel("epoch"), ax[0, j].legend(fontsize=8)
        gap = [r["val_nexar_reasoning"]["gap_wrong"] for r in rows]
        a2 = ax[1, j]
        a2.plot(ep, gap, "o-", color="#54A24B", label="wrong-video gap (nats)")
        a2.set_xlabel("epoch")
        a3 = a2.twinx()
        auc = [next((d["auc"] for d in r["decision"].values() if isinstance(d, dict) and "auc" in d), None) for r in rows] if rows[0].get("decision") else []
        if auc and all(a is not None for a in auc):
            kind = next(iter(rows[0]["decision"]))
            a3.plot(ep, auc, "s--", color="#B279A2", label=f"val decision AUC ('{kind}' score)")
            a3.set_ylabel("AUC")
        a2.set_title(f"{name}: video dependence and decision AUC")
        h1, l1 = a2.get_legend_handles_labels(), a3.get_legend_handles_labels()
        a2.legend(h1[0] + l1[0], h1[1] + l1[1], fontsize=8, loc="best")
    fig.suptitle("Training chain P1aug -> P1fixR -> P2V (the chosen epoch is the one with the lowest validation loss)")
    fig.tight_layout()
    fig.savefig(fig_dir / "loss_curves.png", dpi=130)
    plt.close(fig)


def fig_roc_pr(splits, fig_dir):
    n = len(splits)
    fig, ax = plt.subplots(2, n, figsize=(3.6 * n, 7), squeeze=False)
    for j, sp in enumerate(splits):
        for model, col in (("crash model", C_CRASH), ("P2V", C_P2V)):
            s = sp.score(model)
            fpr, tpr, _ = roc_curve(sp.y, s)
            pr, rc, _ = precision_recall_curve(sp.y, s)
            ax[0, j].plot(fpr, tpr, color=col, label=f"{model} AUC {AUC(sp.y, s):.3f}")
            ax[1, j].plot(rc, pr, color=col, label=f"{model} AP {AP(sp.y, s):.3f}")
        ax[0, j].plot([0, 1], [0, 1], ":", color="gray")
        ax[0, j].set_title(f"{sp.name} (n={sp.n})"), ax[0, j].set_xlabel("false-positive rate"), ax[0, j].set_ylabel("recall")
        ax[1, j].set_xlabel("recall"), ax[1, j].set_ylabel("precision"), ax[1, j].set_ylim(0.4, 1.02)
        ax[0, j].legend(fontsize=8, loc="lower right"), ax[1, j].legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(fig_dir / "roc_pr.png", dpi=130)
    plt.close(fig)


def fig_ap_tte(splits, fig_dir):
    """top row: AUC per TTE for every split (does not depend on the crash : normal ratio); bottom row: AP per TTE for the test splits only
    (val / train-sample groups pair crash windows with ALL normals, about 1 : 10, so their AP is not comparable with the 1 : 1 test groups)."""
    n = len(splits)
    fig, ax = plt.subplots(2, n, figsize=(3.6 * n, 7), squeeze=False)
    for j, sp in enumerate(splits):
        for row, key, lim in ((0, "AUC", (0.85, 1.0)), (1, "AP", (0.8, 1.0))):
            a = ax[row, j]
            if key == "AP" and not sp.paired:
                a.axis("off")
                a.text(0.5, 0.5, "AP per TTE not shown:\ncrash : normal is about 1 : 10\nhere, 1 : 1 in the test", ha="center", va="center", fontsize=9)
                continue
            for q, (model, col) in enumerate((("crash model", C_CRASH), ("P2V", C_P2V))):
                v = [metrics(sp, model)[k][key] for k in TTES]
                a.bar(np.arange(3) + 0.38 * q, v, 0.36, color=col, label=model)
                for x, h in zip(np.arange(3) + 0.38 * q, v):
                    a.text(x, h + 0.003, f"{h:.3f}", ha="center", fontsize=7)
            a.set_xticks(np.arange(3) + 0.19), a.set_xticklabels([f"{k} s" for k in TTES]), a.set_ylim(*lim)
            a.set_title(f"{sp.name}: {key} per TTE", fontsize=10), a.set_xlabel("time to event")
    ax[0, 0].set_ylabel("AUC"), ax[1, 0].set_ylabel("AP"), ax[0, 0].legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(fig_dir / "ap_by_tte.png", dpi=130)
    plt.close(fig)


def fig_cm(sp, thr_p2v, fig_dir, tag):
    fig, ax = plt.subplots(2, 4, figsize=(12, 5.6))
    for i, model in enumerate(("crash model", "P2V")):
        for j, k in enumerate(("all",) + TTES):
            c = conf(sp, model, sp.thr(model, thr_p2v), sp.masks[k])
            M = np.array([[c["TP"], c["FN"]], [c["FP"], c["TN"]]])
            ax[i, j].imshow(M / M.sum(1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
            for a in range(2):
                for b in range(2):
                    ax[i, j].text(b, a, str(M[a, b]), ha="center", va="center", fontsize=12, color="black")
            ax[i, j].set_xticks([0, 1]), ax[i, j].set_xticklabels(["yes", "no"]), ax[i, j].set_yticks([0, 1]), ax[i, j].set_yticklabels(["crash", "normal"])
            ax[i, j].set_title(f"{model}, {'all TTE' if k == 'all' else f'TTE {k} s'}", fontsize=9)
            ax[i, j].set_xlabel("predicted"), ax[i, j].set_ylabel("true")
    fig.suptitle(f"Confusion matrices, {sp.name}, {tag} (cell colour = row share)")
    fig.tight_layout()
    fig.savefig(fig_dir / f"cm_{sp.name.split()[0]}_{tag.split()[0].replace('=', '')}.png", dpi=130)
    plt.close(fig)


def fig_reliab(sp, fig_dir):
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    bins = np.linspace(0, 1, 11)
    for model, col in (("crash model", C_CRASH), ("P2V", C_P2V)):
        p = sp.crash if model == "crash model" else 1 / (1 + np.exp(-np.clip(sp.p2v, -30, 30)))
        idx = np.clip(np.digitize(p, bins) - 1, 0, 9)
        xs = [p[idx == b].mean() for b in range(10) if (idx == b).sum() >= 5]
        ys = [sp.y[idx == b].mean() for b in range(10) if (idx == b).sum() >= 5]
        ax[0].plot(xs, ys, "o-", color=col, label=model)
        ax[1 if model == "crash model" else 2].hist(p, bins=20, color=col)
        ax[1 if model == "crash model" else 2].set_title(f"{model}: score histogram", fontsize=9)
        ax[1 if model == "crash model" else 2].set_xlabel(f"P(collision); {100 * np.mean((p < 0.02) | (p > 0.98)):.0f}% of windows < 0.02 or > 0.98")
    ax[0].plot([0, 1], [0, 1], ":", color="gray"), ax[0].set_xlabel("predicted P(collision)"), ax[0].set_ylabel("observed crash share"), ax[0].legend(fontsize=8)
    ax[0].set_title(f"reliability, {sp.name} (bins with >= 5 windows)", fontsize=9)
    fig.tight_layout()
    fig.savefig(fig_dir / "reliability.png", dpi=130)
    plt.close(fig)


# ----------------------------------------------------------------------------- main
def f3(x):
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--boot", type=int, default=1000)
    a = ap.parse_args()
    out = Path(a.out)
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    pod = out / "pod"
    raw = {"train sample": jl(pod / "train600" / "predictions.jsonl"), "val": jl(P2F / "P2V" / "results" / "predictions.jsonl"),
           "private test": jl(TEST_P), "public test": jl(PUB_P)}
    sp = {k: Split(k, v) for k, v in raw.items() if v}
    pending = [k for k, v in raw.items() if not v]
    if "private test" in sp and "public test" in sp:
        sp["pooled test"] = pool(sp["private test"], sp["public test"], "pooled test")
    thr_p2v, thr_info = 0.0, None
    if "train sample" in sp:
        thr_p2v, bal = train_threshold(sp["train sample"])
        thr_info = f"log-odds {thr_p2v:+.2f} (P = {1 / (1 + math.exp(-thr_p2v)):.2f}), balanced accuracy {bal:.3f} on the train sample"
    M, B = {}, {}
    for name, s in sp.items():
        M[name] = {m: metrics(s, m) for m in ("crash model", "P2V")}
        B[name] = boot_delta(s, a.boot)
    L = ["# P2V report: decision and text metrics, train sample / val / test (plan 2026-10-18, Part A)", "",
         "P2V = frozen V-JEPA2 encoder (the BADAS-based crash model) -> trained projector -> Qwen3-VL-4B + LoRA, verdict first. Its decision score is read at "
         "`Collision:` before any reasoning is written. Crash model = the encoder's own head. **AP, AUC and Kaggle mAP are threshold-free; confusion matrices and "
         "recall / specificity / precision / F1 are at P = 0.5 unless a train-sample threshold is named.**", ""]
    if pending:
        L += [f"> **Pending, not yet measured:** {', '.join(pending)} (the train sample comes from the pod run `r1_pod_trainsplit.sh`; its columns appear when "
              f"`pod/train600/predictions.jsonl` exists).", ""]
    L += ["## 1. Decision metrics", "", "TTE groups: test = crash windows of TTE k vs the normal windows of the same horizon; val / train sample = crash windows of TTE k "
          "vs all normal windows (their normal windows carry no TTE). Kaggle mAP = mean of the three per-TTE APs.", "",
          "| Split | n | Model | AP | AUC | Kaggle mAP | AP 0.5 / 1.0 / 1.5 s | AUC 0.5 / 1.0 / 1.5 s | False alarms at 85% / 90% recall |", "|---|---|---|---|---|---|---|---|---|"]
    for name, s in sp.items():
        dag = "" if s.paired else " †"
        for m in ("crash model", "P2V"):
            x = M[name][m]
            L.append(f"| {name} | {s.n} | {m} | {f3(x['all']['AP'])} | {f3(x['all']['AUC'])} | {f3(x['kaggle_mAP'])}{dag} | " + " / ".join(f3(x[k]["AP"]) for k in TTES) + dag
                     + " | " + " / ".join(f3(x[k]["AUC"]) for k in TTES) + f" | {x['FPR_at_85_recall']:.3f} / {x['FPR_at_90_recall']:.3f} |")
    L += ["", "† val / train sample: each TTE group is its crash windows against ALL normal windows (about 1 crash : 10 normal at TTE 1.5 s), while the test groups are 1 : 1. "
          "AP depends on that ratio, so **per-TTE AP and Kaggle mAP of val / train are not comparable with the test numbers** (use them only to compare the two models on the same split; "
          "AUC per TTE does not depend on the ratio)."]
    L += ["", "**P2V minus the crash model, paired bootstrap over windows (mean [95% interval]):**", ""]
    for name in sp:
        b = B[name]
        L.append(f"- {name}: " + "; ".join(f"{k} {v[0]:+.4f} [{v[1]:+.4f}, {v[2]:+.4f}]" for k, v in b.items()))
    L += ["", "Test intervals only: a single P2V training seed (seed noise on the crash path was about 0.008 AP sd). Window-level intervals are slightly narrow (the test "
          "clips come from fewer source videos).", ""]
    L += ["## 2. Confusion matrices", "", "Per TTE and overall. Crash model: yes at P >= 0.5. P2V: yes at score >= 0 (P = 0.5)"
          + (f", and a second block at the train-sample threshold: {thr_info}. That threshold is a deployment number: P2V saw those windows in training, so it can be optimistic." if thr_info else "."), "",
          "| Split | TTE | Model | threshold | TP | FN | FP | TN | recall | specificity | precision | F1 |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, s in sp.items():
        for k in ("all",) + TTES:
            for m in ("crash model", "P2V"):
                c = conf(s, m, s.thr(m), s.masks[k])
                L.append(f"| {name} | {'all' if k == 'all' else f'{k} s'} | {m} | P = 0.5 | {c['TP']} | {c['FN']} | {c['FP']} | {c['TN']} | {c['recall']:.2f} | {c['specificity']:.2f} | {c['precision']:.2f} | {c['F1']:.2f} |")
    if thr_info:
        L += ["", f"P2V at the train-sample threshold ({thr_info}):", "", "| Split | TTE | TP | FN | FP | TN | recall | specificity | precision | F1 |", "|---|---|---|---|---|---|---|---|---|---|"]
        for name, s in sp.items():
            for k in ("all",) + TTES:
                c = conf(s, "P2V", thr_p2v, s.masks[k])
                L.append(f"| {name} | {'all' if k == 'all' else f'{k} s'} | {c['TP']} | {c['FN']} | {c['FP']} | {c['TN']} | {c['recall']:.2f} | {c['specificity']:.2f} | {c['precision']:.2f} | {c['F1']:.2f} |")
    # figures
    fig_loss(fig_dir)
    show = [sp[k] for k in ("train sample", "val", "private test", "public test", "pooled test") if k in sp]
    fig_roc_pr(show, fig_dir)
    fig_ap_tte(show, fig_dir)
    for k in ("val", "pooled test"):
        if k in sp:
            fig_cm(sp[k], 0.0, fig_dir, "P=0.5")
    if "pooled test" in sp:
        fig_reliab(sp["pooled test"], fig_dir)
    L += ["", "Figures (`figures/`): `loss_curves.png` (P1aug -> P1fixR -> P2V), `roc_pr.png`, `ap_by_tte.png`, `cm_val_P0.5.png`, `cm_pooled_P0.5.png`, `reliability.png`.", ""]

    # ----- text
    T = {"train sample": (pod / "train600", raw["train sample"], cards(out / "cards_train600_model.jsonl")),
         "val": (P2F / "P2V" / "results", raw["val"], cards(ROOT / "outputs" / "r1_phase2f_2026-10-15" / "cards_P2V.jsonl")),
         "private test, agreed subset (431)": (pod / "test_agreed" if (pod / "test_agreed" / "text_metrics.json").exists() else out / "test_agreed",
                                               jl(out / "test_agreed" / "predictions.jsonl"), cards(out / "cards_test_model.jsonl"))}
    teach_train_val = cards(ROOT / "outputs" / "teacher_dataset_2026-10" / "fact_cards_teacher.jsonl")
    teach_test = cards(out / "cards_test_teacher.jsonl")
    cols = list(T)
    L += ["## 3. Text metrics: P2V's explanation vs the teacher's", "",
          "BERTScore (rescaled), ROUGE-L, BLEU-4 and token F1 are model-vs-teacher / floor (the same model texts against another window's teacher text). The private-test "
          "column uses only the 431 windows where the pass-1 teacher verdict equals the label: it keeps 86% of the normal windows but only 52% / 40% / 24% of the crash windows at "
          "TTE 0.5 / 1.0 / 1.5 s (19 windows at 1.5 s), so it is biased toward easier windows. Test texts exist only for this subset.", "",
          "| Metric | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    tm = {c: read_json(T[c][0] / "text_metrics.json") for c in cols}
    for key, lab in (("bertscore_f1_rescaled", "BERTScore F1 (rescaled), model / floor"), ("rouge_l_f1", "ROUGE-L F1, model / floor"), ("bleu4", "BLEU-4, model / floor"),
                     ("token_f1", "token F1, model / floor")):
        L.append(f"| {lab} | " + " | ".join((f"{tm[c][key]['model']:.3f} / {tm[c][key]['floor']:.3f}" if tm[c] and key in tm[c] else "pending") for c in cols) + " |")
    wr = {c: (written(T[c][1]) if T[c][1] else None) for c in cols}
    L.append("| written verdict vs label | " + " | ".join((f"{wr[c]['verdict_acc_vs_label']:.3f}" if wr[c] else "pending") for c in cols) + " |")
    L.append("| written verdict vs decision score (agree) | " + " | ".join((f"{wr[c]['verdict_vs_score_agree']:.3f}" if wr[c] else "pending") for c in cols) + " |")
    L.append("| distinct explanations (share of windows) | " + " | ".join((f"{wr[c]['distinct']:.3f}" if wr[c] else "pending") for c in cols) + " |")
    teachers = {"train sample": teach_train_val, "val": teach_train_val, "private test, agreed subset (431)": teach_test}
    for only_crash in (False, True):
        for s in SLOTS:
            row = []
            for c in cols:
                mc, tc = T[c][2], teachers[c]
                ids = {r["id"] for r in T[c][1] if str(r["group"]).startswith("tte")} if (only_crash and T[c][1]) else None
                acc = card_acc(mc, tc, ids) if mc and tc else {}
                row.append(f"{acc[s]['acc']:.2f} (guess {acc[s]['baseline']:.2f}, n={acc[s]['n']})" if s in acc else "pending")
            L.append(f"| fact card: {s.replace('_', ' ')} agrees with the teacher, {'crash windows only' if only_crash else 'all windows'} | " + " | ".join(row) + " |")
    for lab, fn in (("loss / perplexity on the teacher text", lambda m: f"{m['after']['loss_real']:.3f} / {m['after']['perplexity_real']:.2f}"),
                    ("wrong-video gap (nats) / blank-video gap", lambda m: f"{m['after']['gap_wrong']:+.3f} / {m['after']['gap_blank']:+.3f}"),
                    ("retrieval@1 of 10 (chance 0.10)", lambda m: f"{m['retrieval_at_10']['top1']:.2f}")):
        row = []
        for c in cols:
            m = read_json(T[c][0] / "metrics.json") if c != cols[2] else None
            row.append(fn(m) if m and m.get("after") else ("n/a (no model loss without a reference)" if c == cols[2] else "pending"))
        L.append(f"| {lab} | " + " | ".join(row) + " |")
    L += ["", "The train-vs-val difference in these rows is the overfitting check for the text. For the decision it is the first block of section 1.", ""]
    (out / "report.md").write_text("\n".join(L), encoding="utf-8")
    (out / "metrics.json").write_text(json.dumps({"decision": M, "bootstrap_p2v_minus_crash": B, "train_threshold": thr_info, "pending": pending}, indent=1, default=float), encoding="utf-8")
    print("\n".join(L[:30]))
    print(f"... -> {out / 'report.md'} ({len(L)} lines), {len(list(fig_dir.glob('*.png')))} figures")


if __name__ == "__main__":
    main()
