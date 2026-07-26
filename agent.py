"""Payment-collection conversational agent.

Architecture: an explicit state machine owns the flow; the Extractor
(rules + optional LLM) only proposes structured field values from messy
input; validators.py gates everything before an API call. The LLM never
decides state transitions and never sees account data.

Interface (required by the assignment):

    agent = Agent()
    agent.next("Hi")  ->  {"message": "..."}
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, Optional

import validators as v
from api_client import ApiClient
from extractor import Extracted, Extractor
from verification import VerificationSession

MAX_LOOKUP_ATTEMPTS = 3
MAX_PAYMENT_ATTEMPTS = 3


class State(str, Enum):
    GREET = "GREET"
    ACCOUNT_ID = "ACCOUNT_ID"
    VERIFY_NAME = "VERIFY_NAME"
    VERIFY_FACTOR = "VERIFY_FACTOR"
    ASK_AMOUNT = "ASK_AMOUNT"
    COLLECT_CARD = "COLLECT_CARD"
    CONFIRM = "CONFIRM"
    DONE = "DONE"
    LOCKED = "LOCKED"


class Agent:
    def __init__(self, api_client: Optional[ApiClient] = None, extractor: Optional[Extractor] = None):
        self.api = api_client or ApiClient()
        self.extractor = extractor or Extractor()

        self.state = State.GREET
        self.slots: Dict = {          # everything the user has told us
            "account_id": None,
            "full_name": None,
            "dob": None,
            "aadhaar_last4": None,
            "pincode": None,
            "amount": None,
            "card_number": None,
            "cvv": None,
            "expiry_month": None,
            "expiry_year": None,
            "cardholder_name": None,
        }
        self.verification: Optional[VerificationSession] = None
        self.lookup_attempts = 0
        self.payment_attempts = 0
        self.transport_retry_used = False
        self.transaction_id: Optional[str] = None

    # ── required interface ─────────────────────────────────

    def next(self, user_input: str) -> dict:
        text = (user_input or "").strip()

        if self.state in (State.DONE, State.LOCKED):
            return self._say(
                "This session has ended. Please start a new conversation if you need anything else."
            )

        ex = self.extractor.extract(text, self.state.value)

        # user wants out, from any state
        if ex.wants_exit and self.state != State.GREET:
            self.state = State.DONE
            return self._say(
                "No problem, I've cancelled this session. Nothing was charged. "
                "Feel free to reach out whenever you're ready. Goodbye!"
            )

        self._stash(ex)

        handler = {
            State.GREET: self._h_greet,
            State.ACCOUNT_ID: self._h_account_id,
            State.VERIFY_NAME: self._h_verify_name,
            State.VERIFY_FACTOR: self._h_verify_factor,
            State.ASK_AMOUNT: self._h_amount,
            State.COLLECT_CARD: self._h_card,
            State.CONFIRM: self._h_confirm,
        }[self.state]
        return handler(ex)

    # ── slot memory (context management) ───────────────────

    def _stash(self, ex: Extracted) -> None:
        """Remember everything the user volunteers, in any order. Values
        are *used* only when their step arrives — steps are never skipped."""
        for f in self.slots:
            val = getattr(ex, f, None)
            if val is not None and self.slots[f] is None:
                self.slots[f] = val
        if ex.pay_full:
            self.slots["_pay_full"] = True

    # ── state handlers ─────────────────────────────────────

    def _h_greet(self, ex: Extracted) -> dict:
        self.state = State.ACCOUNT_ID
        if self.slots["account_id"]:
            return self._h_account_id(ex)
        return self._say(
            "Hi! I'm here to help you take care of a pending payment on your account. "
            "Could you share your account ID to get started? It looks like ACC1001."
        )

    def _h_account_id(self, ex: Extracted) -> dict:
        acc = self.slots["account_id"]
        if not acc:
            return self._say("I didn't catch an account ID. It looks like ACC followed by digits, e.g. ACC1001.")

        # Normalize defensively regardless of which extractor produced it.
        acc = v.normalize_account_id(acc)
        self.slots["account_id"] = acc

        check = v.validate_account_id(acc)
        if not check.ok:
            self.slots["account_id"] = None
            return self._say(check.error)

        result = self.api.lookup_account(acc)
        if result.transport_error:
            return self._transport_hiccup("looking up your account")
        if not result.ok:
            self.lookup_attempts += 1
            self.slots["account_id"] = None
            if self.lookup_attempts >= MAX_LOOKUP_ATTEMPTS:
                self.state = State.DONE
                return self._say(
                    "I couldn't find an account after several tries, so I'll stop here. "
                    "Please double-check your account ID and reach out again. Goodbye!"
                )
            return self._say(
                f"I couldn't find an account with ID {acc}. Could you re-check and share it again?"
            )

        self.verification = VerificationSession(account=result.data)
        self.state = State.VERIFY_NAME
        # If the name was volunteered earlier, consume it immediately.
        if self.slots["full_name"]:
            return self._h_verify_name(ex)
        return self._say(
            "Found your account. For security I need to verify your identity first. "
            "Could you tell me your full name as it appears on the account?"
        )

    def _h_verify_name(self, ex: Extracted) -> dict:
        name = self.slots["full_name"]
        if not name:
            return self._say("Sorry, I didn't catch your name. Could you share your full name?")

        if self.verification.try_name(name):
            self.state = State.VERIFY_FACTOR
            if self._pending_factor():
                return self._h_verify_factor(ex)
            return self._say(
                "Thanks. One more check: could you verify your date of birth, "
                "the last 4 digits of your Aadhaar, or your pincode?"
            )

        self.slots["full_name"] = None
        if self.verification.locked:
            return self._lockout()
        return self._say(
            "That name doesn't match our records. The match is exact, so spelling and "
            "capitalization both matter. Please share your full name exactly as it appears "
            f"on the account. ({self._attempts_text()})"
        )

    def _pending_factor(self) -> bool:
        return any(self.slots[k] for k in ("dob", "aadhaar_last4", "pincode"))

    def _h_verify_factor(self, ex: Extracted) -> dict:
        s = self.slots
        tried = False

        if s["dob"]:
            tried = True
            dob, s["dob"] = s["dob"], None
            if v.validate_iso_date(dob).ok and self.verification.try_dob(dob):
                return self._verified()
        elif s["aadhaar_last4"]:
            tried = True
            a4, s["aadhaar_last4"] = s["aadhaar_last4"], None
            if v.validate_aadhaar_last4(a4).ok and self.verification.try_aadhaar(a4):
                return self._verified()
        elif s["pincode"]:
            tried = True
            pc, s["pincode"] = s["pincode"], None
            if v.validate_pincode(pc).ok and self.verification.try_pincode(pc):
                return self._verified()

        if not tried:
            return self._say(
                "I need one of these to finish verification: your date of birth, "
                "the last 4 digits of your Aadhaar, or your 6-digit pincode."
            )
        if self.verification.locked:
            return self._lockout()
        return self._say(
            "That doesn't match our records. You can try your date of birth, Aadhaar last 4, "
            f"or pincode. ({self._attempts_text()})"
        )

    def _verified(self) -> dict:
        self.state = State.ASK_AMOUNT
        balance = self.verification.account["balance"]
        if balance <= 0:
            self.state = State.DONE
            return self._say(
                "You're verified. Good news — your outstanding balance is ₹0.00, so there's "
                "nothing to pay today. Have a great day!"
            )
        msg = (
            f"You're verified. Your outstanding balance is ₹{balance:,.2f}. "
            "How much would you like to pay today? You can pay the full amount or a part of it."
        )
        # amount volunteered earlier?
        if self.slots["amount"] is not None or self.slots.get("_pay_full"):
            follow = self._h_amount(Extracted())
            return follow if follow else self._say(msg)
        return self._say(msg)

    def _h_amount(self, ex: Extracted) -> dict:
        balance = self.verification.account["balance"]
        if self.slots.get("_pay_full"):
            self.slots["amount"] = balance
            self.slots.pop("_pay_full", None)

        amount = self.slots["amount"]
        if amount is None:
            return self._say(
                f"How much would you like to pay? Your outstanding balance is ₹{balance:,.2f}."
            )

        check = v.validate_amount(float(amount), balance)
        if not check.ok:
            self.slots["amount"] = None
            return self._say(f"{check.error} Your outstanding balance is ₹{balance:,.2f} — how much should I charge?")

        self.state = State.COLLECT_CARD
        return self._h_card(ex)

    CARD_PROMPTS = {
        "card_number": "your card number",
        "cardholder_name": "the name on the card",
        "expiry_month": "the expiry (MM/YY)",
        "cvv": "the CVV",
    }

    def _h_card(self, ex: Extracted) -> dict:
        s = self.slots

        # validate whatever we have; discard invalid entries with guidance
        if s["card_number"]:
            check = v.validate_card_number(s["card_number"])
            if not check.ok:
                s["card_number"] = None
                return self._payment_strike(check.error)
        if s["cvv"] and s["card_number"]:
            check = v.validate_cvv(s["cvv"], s["card_number"])
            if not check.ok:
                s["cvv"] = None
                return self._payment_strike(check.error)
        if s["expiry_month"] is not None and s["expiry_year"] is not None:
            check = v.validate_expiry(int(s["expiry_month"]), int(s["expiry_year"]))
            if not check.ok:
                s["expiry_month"] = s["expiry_year"] = None
                return self._payment_strike(check.error)

        missing = [label for f, label in self.CARD_PROMPTS.items()
                   if s[f] is None or (f == "expiry_month" and s["expiry_year"] is None)]
        if len(missing) == 1:
            return self._say(f"Almost there — I just need {missing[0]}.")
        if missing:
            return self._say(
                f"To take the payment of ₹{float(s['amount']):,.2f}, I'll need "
                + self._join_natural(missing)
                + ". Your card details are used only to process this payment."
            )

        self.state = State.CONFIRM
        return self._say(
            f"To confirm: I'll charge ₹{float(s['amount']):,.2f} to the card ending "
            f"{v.mask_card(s['card_number'])[-4:]} (expiring "
            f"{int(s['expiry_month']):02d}/{v.normalize_expiry_year(int(s['expiry_year']))}). "
            "Shall I go ahead? (yes/no)"
        )

    def _h_confirm(self, ex: Extracted) -> dict:
        if ex.no:
            self.slots["amount"] = None
            self.state = State.ASK_AMOUNT
            return self._say(
                "Okay, I won't charge the card. Would you like to change the amount, or type "
                "'cancel' to stop here?"
            )
        if not ex.yes:
            return self._say("Just to be safe — should I process the payment? Please reply yes or no.")

        s = self.slots
        result = self.api.process_payment(
            account_id=s["account_id"],
            amount=float(s["amount"]),
            cardholder_name=s["cardholder_name"],
            card_number=s["card_number"],
            cvv=s["cvv"],
            expiry_month=int(s["expiry_month"]),
            expiry_year=v.normalize_expiry_year(int(s["expiry_year"])),
        )
        self._wipe_card()

        if result.transport_error:
            self.state = State.DONE
            return self._say(
                "I'm having trouble reaching the payment service right now, so I haven't charged "
                "your card. Please try again in a little while. Sorry for the inconvenience!"
            )

        if result.ok:
            self.transaction_id = result.data.get("transaction_id", "")
            self.state = State.DONE
            balance = self.verification.account["balance"]
            remaining = max(0.0, balance - float(s["amount"]))
            return self._say(
                f"Payment successful! ₹{float(s['amount']):,.2f} has been received. "
                f"Your transaction ID is {self.transaction_id}. "
                f"Remaining balance: ₹{remaining:,.2f}. "
                "Thanks for taking care of this today — goodbye!"
            )

        return self._payment_api_error(result.error_code)

    # ── failure handling ───────────────────────────────────

    def _payment_api_error(self, code: Optional[str]) -> dict:
        s = self.slots
        fixable = {
            "invalid_card": ("card_number", "The card number was declined as invalid. Could you re-check and share it again?"),
            "invalid_cvv": ("cvv", "The CVV didn't go through. Could you share it again?"),
            "invalid_expiry": ("expiry_month", "The expiry date was rejected — the card may be expired. Could you share the expiry again, or use a different card?"),
        }
        if code in fixable:
            field_, msg = fixable[code]
            s[field_] = None
            if field_ == "expiry_month":
                s["expiry_year"] = None
            self.state = State.COLLECT_CARD
            return self._payment_strike(msg)
        if code == "insufficient_balance":
            s["amount"] = None
            self.state = State.ASK_AMOUNT
            balance = self.verification.account["balance"]
            return self._say(
                f"That amount exceeds your outstanding balance. You owe ₹{balance:,.2f} — "
                "how much would you like to pay?"
            )
        if code == "invalid_amount":
            s["amount"] = None
            self.state = State.ASK_AMOUNT
            return self._say("The payment service rejected that amount. Please give me a positive amount with at most 2 decimals.")
        # account_not_found or anything unexpected at this stage is terminal
        self.state = State.DONE
        return self._say(
            "Something unexpected went wrong on our side and I couldn't process the payment. "
            "Your card was not charged. Please contact support or try again later. Sorry about this!"
        )

    def _payment_strike(self, message: str) -> dict:
        self.payment_attempts += 1
        if self.payment_attempts >= MAX_PAYMENT_ATTEMPTS:
            self.state = State.DONE
            return self._say(
                "We've had a few unsuccessful attempts with these card details, so I'll stop here "
                "for security. Nothing has been charged. Please verify your card details and try "
                "again later. Goodbye!"
            )
        return self._say(message)

    def _lockout(self) -> dict:
        self.state = State.LOCKED
        return self._say(
            "I'm sorry, but I couldn't verify your identity after several attempts, so I can't "
            "proceed for security reasons. Please contact support for help with your account. Goodbye."
        )

    def _transport_hiccup(self, doing: str) -> dict:
        if not self.transport_retry_used:
            self.transport_retry_used = True
            return self._say(
                f"I hit a temporary problem while {doing}. Could you send that once more?"
            )
        self.state = State.DONE
        return self._say(
            f"I'm still having trouble {doing} — it's on our side, not yours. "
            "Please try again in a little while. Sorry for the inconvenience!"
        )

    # ── misc ───────────────────────────────────────────────

    def _wipe_card(self) -> None:
        """Card data is kept only long enough to build the API payload."""
        for f in ("card_number", "cvv"):
            self.slots[f] = None

    def _attempts_text(self) -> str:
        n = self.verification.attempts_left
        return f"{n} attempt{'s' if n != 1 else ''} left."

    @staticmethod
    def _join_natural(items) -> str:
        items = list(items)
        if len(items) <= 1:
            return items[0] if items else ""
        return ", ".join(items[:-1]) + ", and " + items[-1]

    @staticmethod
    def _say(message: str) -> dict:
        return {"message": message}
