"""
r1_extra_audit_report.py - score the Opus audit of the rewordings of the extra windows (plan 2026-10-19), the same design as the original audit (2026-10-15):
300 versions shown BLIND (no judge decision): 250 the Sonnet judge accepted, 50 it rejected; two Opus auditors (audit_out_a / _b) mark each version ok / minor nuance / major error.

GATE, fixed before the audit: PASS if at most 1% of the judge-ACCEPTED versions have a MAJOR error (at most 2 of the 250). Minor nuances (strength / degree, a small added or dropped
detail) are reported, not gated (the original audit found 7.2% and the user kept the texts as the "strict set").

    python r1_extra_audit_report.py        -> prints the result, writes audit/audit_summary.json; exit code 0 = PASS, 2 = FAIL
"""
from __future__ import annotations

import collections
import json
import math
import sys
from pathlib import Path

P = Path(__file__).resolve().parents[2] / "outputs" / "teacher_extra_2026-10" / "paraphrase"


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def wilson_upper(k, n, z=1.96):
    if n == 0:
        return 1.0
    p = k / n
    d = 1 + z * z / n
    return (p + z * z / (2 * n) + z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d


def main():
    sample = {r["key"]: r for r in jl(P / "paraphrase_audit_sample.jsonl")}
    ans = {}
    for n in ("a", "b"):
        for r in jl(P / "audit" / f"audit_out_{n}.jsonl"):
            ans[r["key"]] = r
    missing = [k for k in sample if k not in ans]
    if missing:
        print(f"FAIL audit incomplete: {len(missing)} of {len(sample)} versions have no answer, e.g. {missing[:2]}")
        sys.exit(2)

    def verdict(k):
        a = ans[k]
        return "ok" if a.get("ok") else (a.get("severity") or "minor")

    acc = [k for k, r in sample.items() if r["judge_ok"]]
    rej = [k for k, r in sample.items() if not r["judge_ok"]]
    ca, cr = collections.Counter(verdict(k) for k in acc), collections.Counter(verdict(k) for k in rej)
    major, minor = ca["major"], ca["minor"]
    ok = major <= max(1, int(0.01 * len(acc)) + 1)                 # <= 1% rounded up: 2 of 250
    out = {"judge_accepted": len(acc), "accepted_ok": ca["ok"], "accepted_minor": minor, "accepted_major": major,
           "major_rate": major / len(acc), "major_rate_upper95": wilson_upper(major, len(acc)), "minor_rate": minor / len(acc),
           "judge_rejected": len(rej), "rejected_but_opus_ok": cr["ok"], "gate_pass": ok,
           "major_examples": [{"key": k, "problem": ans[k].get("problem"), "rule": ans[k].get("rule")} for k in acc if verdict(k) == "major"][:10]}
    (P / "audit" / "audit_summary.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(("PASS " if ok else "FAIL ") + f"audit: of {len(acc)} judge-accepted versions Opus finds {major} major errors ({major / len(acc):.1%}, 95% upper bound {wilson_upper(major, len(acc)):.1%}; "
          f"gate <= 1%), {minor} minor nuances ({minor / len(acc):.1%}); of {len(rej)} judge-rejected versions Opus would have accepted {cr['ok']}")
    for e in out["major_examples"]:
        print("   major:", e)
    sys.exit(0 if ok else 2)


if __name__ == "__main__":
    main()
