"""Tool-call correctness tests using a mock API.

The mock records every call the agent makes, so we can assert:
  - APIs are called at the right MOMENT (never before verification/confirmation)
  - payloads are correct and pre-validated (Luhn-valid card, amount within
    balance and 2dp, normalized account ID, integer expiry fields)
  - API error codes and transport failures are handled per spec

Runs fully offline and deterministically (LLM disabled).

Usage:
    python evals/test_tool_calls.py
"""

from __future__ import annotations

import os
import sys

os.environ["GEMINI_API_KEY"] = ""  # deterministic
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import Agent            # noqa: E402
from api_client import ApiResult   # noqa: E402
from validators import luhn_ok     # noqa: E402

ACCOUNTS = {
    "ACC1001": {"account_id": "ACC1001", "full_name": "Nithin Jain", "dob": "1990-05-14",
                "aadhaar_last4": "4321", "pincode": "400001", "balance": 1250.75},
}


class MockApi:
    """Mirrors ApiClient's interface; records calls; scriptable failures."""

    def __init__(self, payment_errors=None, lookup_transport_fails=0):
        self.lookups = []
        self.payments = []
        self.payment_errors = list(payment_errors or [])   # error codes to return, in order
        self.lookup_transport_fails = lookup_transport_fails

    def lookup_account(self, account_id):
        self.lookups.append(account_id)
        if self.lookup_transport_fails > 0:
            self.lookup_transport_fails -= 1
            return ApiResult(ok=False, transport_error=True)
        if account_id in ACCOUNTS:
            return ApiResult(ok=True, data=dict(ACCOUNTS[account_id]))
        return ApiResult(ok=False, error_code="account_not_found")

    def process_payment(self, **payload):
        self.payments.append(payload)
        if self.payment_errors:
            return ApiResult(ok=False, error_code=self.payment_errors.pop(0))
        return ApiResult(ok=True, data={"success": True, "transaction_id": "txn_mock_001"})


CHECKS = []


def check(name, cond):
    CHECKS.append((name, bool(cond)))
    print(("pass  " if cond else "FAIL  ") + name)


def drive(agent, turns):
    last = ""
    for t in turns:
        last = agent.next(t)["message"]
    return last


def main() -> int:
    # ── 1. Happy path: calls at the right time with correct payloads ──
    api = MockApi()
    a = Agent(api_client=api)
    a.next("Hi")
    a.next("acc 1001 is my account")
    check("lookup called exactly once", len(api.lookups) == 1)
    check("lookup payload normalized to ACC1001", api.lookups == ["ACC1001"])
    drive(a, ["Nithin Jain", "1990-05-14"])
    check("no payment call before amount/card collected", len(api.payments) == 0)
    drive(a, ["pay 500.50", "card 4532015112830366, Nithin Jain on the card", "12/27", "123"])
    check("no payment call before user confirmation", len(api.payments) == 0)
    last = drive(a, ["yes"])
    check("payment called exactly once after confirmation", len(api.payments) == 1)
    if api.payments:
        p = api.payments[0]
        check("payment account normalized", p["account_id"] == "ACC1001")
        check("amount is 2dp and within balance", p["amount"] == 500.50 <= 1250.75)
        check("card passes Luhn before API", luhn_ok(p["card_number"]))
        check("expiry sent as ints, year normalized", p["expiry_month"] == 12 and p["expiry_year"] == 2027)
        check("cvv correct length", len(p["cvv"]) == 3)
    check("success message carries transaction id", "txn_mock_001" in last)

    # ── 2. Failed verification: payment API must never be called ──
    api = MockApi()
    a = Agent(api_client=api)
    drive(a, ["Hi", "ACC1001", "Wrong Name", "Also Wrong", "Still Wrong",
              "pay 100", "card 4532015112830366", "yes"])
    check("no payment call after failed verification (locked)", len(api.payments) == 0)

    # ── 3. API error codes handled and retried correctly ──
    api = MockApi(payment_errors=["invalid_card"])
    a = Agent(api_client=api)
    drive(a, ["Hi", "ACC1001", "Nithin Jain", "4321", "pay 100",
              "card 4532015112830366, Nithin Jain on the card, 12/27, cvv 123"])
    msg = drive(a, ["yes"])
    check("invalid_card from API -> user-fixable retry message", "re-check" in msg.lower() or "invalid" in msg.lower())
    msg = drive(a, ["4111111111111111"])
    # CVV is deliberately wiped after every payment attempt (never retained
    # post-authorization), so the agent must re-collect it on retry.
    check("agent re-collects CVV after wipe (security)", "cvv" in msg.lower())
    msg = drive(a, ["cvv 123"])
    check("re-confirms before second attempt", "confirm" in msg.lower())
    msg = drive(a, ["yes"])
    check("second attempt succeeds after fix", "txn_mock_001" in msg)
    check("payment endpoint called twice total", len(api.payments) == 2)

    # ── 4. Transport failure on lookup: retry once, then close cleanly ──
    api = MockApi(lookup_transport_fails=2)
    a = Agent(api_client=api)
    a.next("Hi")
    msg = a.next("ACC1001")["message"]
    check("transport error -> polite retry ask", "temporary problem" in msg.lower())
    msg = a.next("ACC1001")["message"]
    check("second transport error -> clean close, blame on our side", "our side" in msg.lower())

    # ── summary ──
    passed = sum(1 for _, ok in CHECKS if ok)
    print(f"\n{passed}/{len(CHECKS)} tool-call checks passed")
    return 0 if passed == len(CHECKS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
