"""
r1_extra_gates.py - the automatic gates of the unattended teacher run on the extra TTE windows (plan 2026-10-19). Each prints one line and exits 0 (pass) or 2 (stop).
Defined BEFORE the run so nobody decides afterwards what a good result is; a failed gate stops the driver (r1_extra_pipeline.sh), nothing more is spent.

  boxes     >= 95% of the 874 new windows have a red box (the box stage found a road user); the rest are listed
  e3        <= 5% of the new windows have no parsed first-look answer (after the retry); prints the yes-rate (the rest go to the second look)
  flow      <= 15% of the new windows are flagged "hazard not visible even with hindsight" (they would be removed), <= 5% unparsed second looks
  rewords   >= 90% of the kept new windows have >= 1 accepted rewording
  budget X  the OpenRouter balance is at least X dollars
"""
from __future__ import annotations

import collections
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
W = Path(os.environ.get("R1_PILOT_OUT") or ROOT / "outputs" / "teacher_extra_2026-10")
NEW = (0.75, 1.25)


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()] if Path(p).exists() else []


def new_fds():
    return {w["frames_dir"] for w in jl(W / "windows.jsonl") if w["horizon"] in NEW and "split" in w}


def out(ok, msg):
    print(("PASS " if ok else "FAIL ") + msg)
    sys.exit(0 if ok else 2)


def main():
    g = sys.argv[1]
    fds = new_fds()
    n = len(fds)
    if g == "boxes":
        rows = {r["frames_dir"]: r for r in jl(W / "boxes.jsonl")}
        have = [rows[f] for f in fds if f in rows]
        flags = collections.Counter(r["flag"] for r in have)
        boxed = sum(1 for r in have if r["flag"] != "no_object" and r["frames_boxed"] > 0)
        out(len(have) == n and boxed >= 0.95 * n, f"boxes: {len(have)} of {n} new windows have a record, {boxed} boxed (need >= 95%); flags {dict(flags)}")
    if g == "e3":
        e3 = {r["frames_dir"]: r for r in jl(W / "runs" / "E3.jsonl") if r.get("parsed")}
        got = [e3[f] for f in fds if f in e3]
        yes = sum(r["parsed"].get("collision") == "yes" for r in got)
        out(len(got) >= 0.95 * n, f"E3: {len(got)} of {n} new windows parsed (need >= 95%); first look says yes on {yes} ({yes / max(1, len(got)):.0%}), the other {len(got) - yes} go to the second look")
    if g == "flow":
        flow = {r["frames_dir"]: r for r in jl(W / "flow_final.jsonl")}
        got = [flow[f] for f in fds if f in flow]
        flagged = [r for r in got if str(r.get("status", "ok")).startswith("flagged")]
        paths = collections.Counter(r.get("final_from") for r in got)
        out(len(got) == n and len(flagged) <= 0.15 * n, f"flow: {len(got)} of {n} routed, flagged {len(flagged)} (limit {int(0.15 * n)}); final_from {dict(paths)}")
    if g == "rewords":
        keep = {r["frames_dir"] for r in jl(W / "dataset_final.jsonl")}
        pj = collections.Counter(r["id"].replace("nexar_", "", 1) for r in jl(W / "paraphrase" / "paraphrases_judged.jsonl"))
        have = sum(1 for f in keep if pj.get(f, 0) >= 1)
        out(have >= 0.90 * len(keep), f"rewordings: {have} of {len(keep)} kept new windows have >= 1 accepted version (need >= 90%); versions {sum(pj.values())}, windows with 3: {sum(1 for f in keep if pj.get(f, 0) >= 3)}")
    if g == "budget":
        need = float(sys.argv[2])
        bal = float(subprocess.check_output([sys.executable, str(ROOT / "student_training" / "scripts" / "r1_or_balance.py")], text=True).strip().splitlines()[-1])
        out(bal >= need, f"budget: OpenRouter balance ${bal:.2f}, need ${need:.2f}")
    sys.exit(1)


if __name__ == "__main__":
    main()
