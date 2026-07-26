"""Strict identity verification with retry accounting.

Spec: full name must match EXACTLY, plus at least one of DOB /
Aadhaar-last-4 / pincode. No fuzzy matching. Account values are never
surfaced to the user - we only ever say whether *their* input matched.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

MAX_ATTEMPTS = 3  # total failed verification attempts before lockout


@dataclass
class VerificationSession:
    account: Dict          # full record from lookup (kept internal, never echoed)
    name_ok: bool = False
    factor_ok: bool = False
    failed_attempts: int = 0

    @property
    def verified(self) -> bool:
        return self.name_ok and self.factor_ok

    @property
    def locked(self) -> bool:
        return self.failed_attempts >= MAX_ATTEMPTS

    # ── checks (all strict equality) ───────────────────────

    def try_name(self, name: str) -> bool:
        """Exact match, case-sensitive, single-spaced. We normalize only
        whitespace - casing and spelling must match the account record."""
        ok = " ".join(name.split()) == self.account["full_name"]
        self._record(ok, is_name=True)
        return ok

    def try_dob(self, dob_iso: str) -> bool:
        ok = dob_iso == self.account["dob"]
        self._record(ok)
        return ok

    def try_aadhaar(self, last4: str) -> bool:
        ok = last4 == self.account["aadhaar_last4"]
        self._record(ok)
        return ok

    def try_pincode(self, pincode: str) -> bool:
        ok = pincode == self.account["pincode"]
        self._record(ok)
        return ok

    def _record(self, ok: bool, is_name: bool = False) -> None:
        if ok:
            if is_name:
                self.name_ok = True
            else:
                self.factor_ok = True
        else:
            self.failed_attempts += 1

    @property
    def attempts_left(self) -> int:
        return max(0, MAX_ATTEMPTS - self.failed_attempts)
