"""Deterministic validation for everything the agent sends to an API.

All correctness-critical checks live here, in plain Python, so the agent
never depends on an LLM to decide whether data is valid.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime
from typing import Optional


# ── Results ────────────────────────────────────────────────


@dataclass
class ValidationResult:
    ok: bool
    error: Optional[str] = None  # human-readable reason when not ok


# ── Account ID ─────────────────────────────────────────────

ACCOUNT_ID_RE = re.compile(r"^ACC\d{4,}$")


def normalize_account_id(raw: str) -> str:
    """'acc 1001' / 'ACC-1001' -> 'ACC1001'."""
    return re.sub(r"[\s\-]", "", raw).upper()


def validate_account_id(raw: str) -> ValidationResult:
    if ACCOUNT_ID_RE.match(normalize_account_id(raw)):
        return ValidationResult(True)
    return ValidationResult(False, "Account IDs look like ACC followed by digits, e.g. ACC1001.")


# ── Dates (DOB) ────────────────────────────────────────────


def validate_iso_date(value: str) -> ValidationResult:
    """Strict YYYY-MM-DD check. datetime.strptime handles leap years
    correctly, so 1988-02-29 is valid while 1989-02-29 is not."""
    try:
        datetime.strptime(value, "%Y-%m-%d")
        return ValidationResult(True)
    except ValueError:
        return ValidationResult(False, "That date doesn't exist on the calendar.")


# ── Aadhaar last-4 / pincode ───────────────────────────────


def validate_aadhaar_last4(value: str) -> ValidationResult:
    if re.fullmatch(r"\d{4}", value):
        return ValidationResult(True)
    return ValidationResult(False, "Aadhaar last-4 should be exactly 4 digits.")


def validate_pincode(value: str) -> ValidationResult:
    if re.fullmatch(r"\d{6}", value):
        return ValidationResult(True)
    return ValidationResult(False, "Pincode should be exactly 6 digits.")


# ── Amount ─────────────────────────────────────────────────


def validate_amount(amount: float, balance: float) -> ValidationResult:
    if amount <= 0:
        return ValidationResult(False, "The amount must be greater than zero.")
    # More than 2 decimal places is rejected by the API (invalid_amount).
    if round(amount, 2) != amount:
        return ValidationResult(False, "The amount can have at most 2 decimal places.")
    if amount > balance:
        return ValidationResult(
            False, "That amount is more than the outstanding balance."
        )
    return ValidationResult(True)


# ── Card ───────────────────────────────────────────────────


def luhn_ok(card_number: str) -> bool:
    digits = [int(d) for d in card_number]
    checksum = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        checksum += d
    return checksum % 10 == 0


def is_amex(card_number: str) -> bool:
    return card_number.startswith(("34", "37"))


def validate_card_number(card_number: str) -> ValidationResult:
    if not re.fullmatch(r"\d{13,19}", card_number):
        return ValidationResult(False, "Card numbers are 13-19 digits.")
    if not luhn_ok(card_number):
        return ValidationResult(False, "That card number doesn't pass the checksum - please re-check the digits.")
    return ValidationResult(True)


def validate_cvv(cvv: str, card_number: str) -> ValidationResult:
    expected = 4 if is_amex(card_number) else 3
    if re.fullmatch(rf"\d{{{expected}}}", cvv):
        return ValidationResult(True)
    return ValidationResult(False, f"The CVV should be {expected} digits for this card.")


def validate_expiry(month: int, year: int, today: Optional[date] = None) -> ValidationResult:
    if not 1 <= month <= 12:
        return ValidationResult(False, "The expiry month should be between 1 and 12.")
    today = today or date.today()
    if year < 100:  # '27' -> 2027
        year += 2000
    if (year, month) < (today.year, today.month):
        return ValidationResult(False, "That card appears to be expired.")
    if year > today.year + 30:
        return ValidationResult(False, "That expiry year looks too far in the future.")
    return ValidationResult(True)


def normalize_expiry_year(year: int) -> int:
    return year + 2000 if year < 100 else year


def mask_card(card_number: str) -> str:
    """For any user-facing text or logs: only ever show the last 4."""
    return f"****{card_number[-4:]}" if len(card_number) >= 4 else "****"
