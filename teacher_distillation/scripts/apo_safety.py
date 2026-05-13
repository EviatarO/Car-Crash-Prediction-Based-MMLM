"""APO safety: billing & liveness self-checks.

Layers:
  1. Pre-run credit check (check_openrouter_credits)
  2. Per-iteration credit re-check (same function, called repeatedly)
  3. Per-call cost anomaly detection (CallCostMonitor class)
  4. Heartbeat / liveness logs (heartbeat_log)

NOT included here (handled in optimizer):
  5. Incremental trial-log writes (already in optimizer)
  6. Watchdog (skipped for v1 per plan)
"""

from __future__ import annotations

import os
import sys
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import requests


# ---------------------------------------------------------------------------
# Layer 1 / 2: OpenRouter credit check
# ---------------------------------------------------------------------------

@dataclass
class CreditStatus:
    total_credits: float
    total_usage: float
    remaining: float
    raw_response: dict
    fetched_ok: bool


def check_openrouter_credits(api_key: str, timeout: float = 10.0) -> Optional[CreditStatus]:
    """Query OpenRouter /api/v1/credits endpoint.

    Returns:
        CreditStatus on success.
        None if the endpoint is unreachable (caller should warn but may continue).
    """
    if not api_key:
        return None
    try:
        resp = requests.get(
            "https://openrouter.ai/api/v1/credits",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
        )
        resp.raise_for_status()
        body = resp.json()
        data = body.get("data", body)  # newer responses wrap in "data"
        total = float(data.get("total_credits", 0) or 0)
        used = float(data.get("total_usage", 0) or 0)
        return CreditStatus(
            total_credits=total,
            total_usage=used,
            remaining=total - used,
            raw_response=body,
            fetched_ok=True,
        )
    except Exception as exc:
        print(f"  [safety] OpenRouter /credits unreachable: {exc!r}", flush=True)
        return None


def assert_sufficient_credit(
    api_key: str,
    projected_cost: float,
    safety_margin: float = 1.5,
    label: str = "pre-run",
) -> CreditStatus:
    """Layer 1: abort if insufficient credit.

    Aborts (sys.exit) if remaining credit is below `projected_cost * safety_margin`.
    Returns the CreditStatus on success.
    """
    print(f"  [safety/{label}] Querying OpenRouter credits...", flush=True)
    status = check_openrouter_credits(api_key)
    if status is None:
        print(f"  [safety/{label}] WARNING: cannot read /credits. Continuing without check.",
              flush=True)
        return CreditStatus(0, 0, 0, {}, False)

    needed = projected_cost * safety_margin
    print(f"  [safety/{label}] total_credits=${status.total_credits:.2f}  "
          f"total_usage=${status.total_usage:.2f}  remaining=${status.remaining:.2f}",
          flush=True)
    print(f"  [safety/{label}] projected cost=${projected_cost:.2f}, "
          f"required (with {safety_margin}x margin)=${needed:.2f}",
          flush=True)

    if status.remaining < needed:
        print(f"\n  [safety/{label}] ABORT: insufficient credit.", flush=True)
        print(f"  [safety/{label}] You have ${status.remaining:.2f} but need at least ${needed:.2f}.",
              flush=True)
        print(f"  [safety/{label}] Top up at https://openrouter.ai/settings/credits and re-run.",
              flush=True)
        sys.exit(2)

    print(f"  [safety/{label}] OK: sufficient credit.", flush=True)
    return status


def soft_credit_check(
    api_key: str,
    minimum_remaining: float = 5.0,
    label: str = "iter",
) -> bool:
    """Layer 2: per-iteration check. Returns True if OK, False if should abort gracefully.

    Caller decides whether to abort or finalize current state.
    """
    status = check_openrouter_credits(api_key)
    if status is None:
        # Can't check — let caller continue; don't block on a network glitch.
        return True

    print(f"  [credit-check/{label}] remaining=${status.remaining:.2f}", flush=True)
    if status.remaining < minimum_remaining:
        print(f"  [credit-check/{label}] LOW BALANCE: ${status.remaining:.2f} < ${minimum_remaining:.2f}",
              flush=True)
        return False
    return True


# ---------------------------------------------------------------------------
# Layer 3: Per-call cost anomaly detection
# ---------------------------------------------------------------------------

class CallCostMonitor:
    """Tracks per-call costs and aborts after N consecutive anomalies.

    Usage:
        monitor = CallCostMonitor(expected_call_cost=0.06, anomaly_factor=3.0, max_consecutive=3)
        for call in calls:
            cost = run_call(...)
            if monitor.record_and_check(cost):
                continue  # OK
            else:
                # 3 consecutive anomalies -> abort
                sys.exit(3)
    """

    def __init__(
        self,
        expected_call_cost: float = 0.06,
        anomaly_factor: float = 3.0,
        max_consecutive: int = 3,
    ):
        self.expected_cost = expected_call_cost
        self.threshold = expected_call_cost * anomaly_factor
        self.max_consecutive = max_consecutive
        self.consecutive_anomalies = 0
        self.total_anomalies = 0
        self.history = deque(maxlen=20)

    def record_and_check(self, call_cost: float) -> bool:
        """Record a call cost. Return False if threshold breached on N consecutive calls."""
        self.history.append(call_cost)
        if call_cost > self.threshold:
            self.consecutive_anomalies += 1
            self.total_anomalies += 1
            print(
                f"  [cost-anomaly] call cost=${call_cost:.4f} "
                f"(expected ~${self.expected_cost:.4f}, threshold=${self.threshold:.4f}). "
                f"Consecutive anomalies: {self.consecutive_anomalies}/{self.max_consecutive}",
                flush=True,
            )
            if self.consecutive_anomalies >= self.max_consecutive:
                print(
                    f"  [cost-anomaly] ABORT: {self.consecutive_anomalies} consecutive anomalies. "
                    f"Likely a bug (token explosion, retry loop, etc.). Stopping run.",
                    flush=True,
                )
                return False
        else:
            # Reset counter if we get a normal call
            if self.consecutive_anomalies > 0:
                print(f"  [cost-anomaly] anomaly streak reset (current call=${call_cost:.4f})",
                      flush=True)
                self.consecutive_anomalies = 0
        return True


# ---------------------------------------------------------------------------
# Layer 4: Heartbeat / liveness logs
# ---------------------------------------------------------------------------

def heartbeat_log(message: str) -> None:
    """Emit a timestamped liveness log line. Use sparingly during long operations."""
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"  [HEARTBEAT {ts}] {message}", flush=True)


# ---------------------------------------------------------------------------
# CLI sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("OPENROUTER_API_KEY not set; cannot test")
        sys.exit(1)
    print("=== Test Layer 1: pre-run credit check ===")
    status = check_openrouter_credits(api_key)
    if status:
        print(f"  total=${status.total_credits:.2f}  used=${status.total_usage:.2f}  remaining=${status.remaining:.2f}")
    print()
    print("=== Test Layer 3: cost monitor ===")
    monitor = CallCostMonitor(expected_call_cost=0.06)
    for c in [0.05, 0.07, 0.30, 0.25, 0.05, 0.30, 0.30, 0.30]:
        ok = monitor.record_and_check(c)
        print(f"  call=${c:.4f}  ok={ok}")
        if not ok: break
    print()
    print("=== Test Layer 4: heartbeat ===")
    heartbeat_log("Phase C iter 1, candidate 1, clip 5/31")
    print("\nAll safety layers functional.")
