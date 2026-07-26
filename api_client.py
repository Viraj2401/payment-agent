"""HTTP client for the payment/verification API.

Every call returns an ApiResult so the agent never has to touch raw HTTP
concerns: success payload, a known error_code, or a transport failure are
all represented explicitly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import requests

DEFAULT_BASE_URL = (
    "https://se-payment-verification-api.service.external.usea2.aws.prodigaltech.com"
)
TIMEOUT_SECONDS = 10

# Error codes documented in the assignment spec.
KNOWN_ERROR_CODES = {
    "account_not_found",
    "invalid_amount",
    "insufficient_balance",
    "invalid_card",
    "invalid_cvv",
    "invalid_expiry",
}


@dataclass
class ApiResult:
    ok: bool
    data: Dict[str, Any] = field(default_factory=dict)
    error_code: Optional[str] = None   # a KNOWN_ERROR_CODES value when the API rejected us
    transport_error: bool = False      # network/timeout/5xx — not the user's fault


class ApiClient:
    def __init__(self, base_url: Optional[str] = None, session: Optional[requests.Session] = None):
        self.base_url = (base_url or os.getenv("API_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.session = session or requests.Session()

    # ── endpoints ──────────────────────────────────────────

    def lookup_account(self, account_id: str) -> ApiResult:
        return self._post("/api/lookup-account", {"account_id": account_id})

    def process_payment(
        self,
        account_id: str,
        amount: float,
        cardholder_name: str,
        card_number: str,
        cvv: str,
        expiry_month: int,
        expiry_year: int,
    ) -> ApiResult:
        payload = {
            "account_id": account_id,
            "amount": amount,
            "payment_method": {
                "type": "card",
                "card": {
                    "cardholder_name": cardholder_name,
                    "card_number": card_number,
                    "cvv": cvv,
                    "expiry_month": expiry_month,
                    "expiry_year": expiry_year,
                },
            },
        }
        return self._post("/api/process-payment", payload)

    # ── plumbing ───────────────────────────────────────────

    def _post(self, path: str, payload: Dict[str, Any]) -> ApiResult:
        try:
            resp = self.session.post(
                f"{self.base_url}{path}", json=payload, timeout=TIMEOUT_SECONDS
            )
        except requests.RequestException:
            return ApiResult(ok=False, transport_error=True)

        if resp.status_code >= 500:
            return ApiResult(ok=False, transport_error=True)

        try:
            body = resp.json()
        except ValueError:
            return ApiResult(ok=False, transport_error=True)

        if resp.status_code == 200 and body.get("success", True):
            return ApiResult(ok=True, data=body)

        code = body.get("error_code")
        if code in KNOWN_ERROR_CODES:
            return ApiResult(ok=False, data=body, error_code=code)
        # 4xx with an unknown shape — treat as transport-ish so the agent
        # apologises rather than blaming the user.
        return ApiResult(ok=False, data=body, transport_error=True)
