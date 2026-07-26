"""Turns messy natural-language input into structured fields.

Two layers:

1. DeterministicExtractor - regex + date parsing + number words. Fully
   offline, fully reproducible. Handles every example in the assignment
   spec. This is also the safety net that keeps the agent functional
   (and tests deterministic) when no LLM key is configured.

2. LLMExtractor - optional Gemini Flash call (temperature 0) that fills
   the same schema for phrasings the rules miss. Output is merged with -
   and validated by - the deterministic layer; the LLM never controls
   flow, it only proposes field values.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Dict, Optional

from dateutil import parser as dateparser

# Load .env no matter which entry point imports us (CLI, evaluator, tests).
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # env vars can still be set by the shell


# ── The shared schema both layers fill ─────────────────────


@dataclass
class Extracted:
    account_id: Optional[str] = None
    full_name: Optional[str] = None
    dob: Optional[str] = None          # normalized YYYY-MM-DD
    aadhaar_last4: Optional[str] = None
    pincode: Optional[str] = None
    amount: Optional[float] = None
    pay_full: bool = False
    card_number: Optional[str] = None
    cvv: Optional[str] = None
    expiry_month: Optional[int] = None
    expiry_year: Optional[int] = None
    cardholder_name: Optional[str] = None
    yes: bool = False
    no: bool = False
    wants_exit: bool = False

    def merge_missing(self, other: "Extracted") -> None:
        """Fill any field this extraction is missing from `other`
        (deterministic values win; LLM only fills gaps)."""
        for f in self.__dataclass_fields__:
            mine = getattr(self, f)
            theirs = getattr(other, f)
            if mine in (None, False) and theirs not in (None, False):
                setattr(self, f, theirs)


# ── Number words (for "a thousand rupees", "one two three") ─

_UNITS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19,
}
_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}
_SCALES = {"hundred": 100, "thousand": 1000, "lakh": 100000}

_MONTHS = {
    m: i + 1
    for i, m in enumerate(
        ["january", "february", "march", "april", "may", "june", "july",
         "august", "september", "october", "november", "december"]
    )
}
_MONTHS.update({m[:3]: v for m, v in list(_MONTHS.items())})


def words_to_number(text: str) -> Optional[float]:
    """'a thousand' -> 1000, 'five hundred' -> 500, 'twenty five' -> 25."""
    tokens = re.findall(r"[a-z]+", text.lower())
    total, current, seen = 0, 0, False
    for tok in tokens:
        if tok in ("a", "an", "and"):
            continue
        if tok in _UNITS:
            current += _UNITS[tok]; seen = True
        elif tok in _TENS:
            current += _TENS[tok]; seen = True
        elif tok in _SCALES:
            current = max(current, 1) * _SCALES[tok]
            total += current
            current = 0
            seen = True
        else:
            continue
    return float(total + current) if seen else None


def spoken_digits(text: str) -> Optional[str]:
    """'one two three' -> '123', '4 0 0 0 0 1' -> '400001'."""
    out = []
    for tok in re.findall(r"[a-z]+|\d", text.lower()):
        if tok.isdigit():
            out.append(tok)
        elif tok in _UNITS and _UNITS[tok] <= 9:
            out.append(str(_UNITS[tok]))
        elif tok in ("oh", "o"):
            out.append("0")
        else:
            # a non-digit word breaks a digit run only if we haven't started
            if out:
                break
    s = "".join(out)
    return s if s else None


# ── Deterministic layer ────────────────────────────────────


class DeterministicExtractor:
    """State-aware rule extraction. `state` biases how bare values are
    interpreted (a lone 4-digit number means Aadhaar-last-4 during
    verification but CVV during card collection)."""

    def extract(self, text: str, state: str) -> Extracted:
        out = Extracted()
        t = text.strip()
        low = t.lower()

        # intents
        if re.search(r"\b(cancel|stop|quit|exit|bye|goodbye|not now|later)\b", low):
            out.wants_exit = True
        if re.fullmatch(r"(yes|yep|yeah|sure|ok|okay|confirm|proceed|go ahead|y)[.! ]*", low):
            out.yes = True
        if re.fullmatch(r"(no|nope|nah|don'?t|cancel that|n)[.! ]*", low):
            out.no = True

        # account id ("ACC 1001", "acc1001")
        m = re.search(r"\bacc[\s\-]*\d{4,}\b", low)
        if m:
            out.account_id = re.sub(r"[\s\-]", "", m.group(0)).upper()

        # amount
        if re.search(r"\b(full|entire|whole|complete)\b.*\b(amount|balance)\b|\bclear (it|the)\b|\bpay (it )?(all|off)\b", low):
            out.pay_full = True
        else:
            money = re.search(r"(?:rs\.?|inr|₹)?\s*(\d{1,7}(?:,\d{3})*(?:\.\d{1,4})?)\s*(?:rs|rupees|inr)?\b", low)
            if money and state in ("ASK_AMOUNT",):
                out.amount = float(money.group(1).replace(",", ""))
            elif state == "ASK_AMOUNT":
                w = words_to_number(low)
                if w:
                    out.amount = w

        # card number: 13-19 digits once separators removed
        for m in re.finditer(r"(?:\d[\s\-]?){13,19}", t):
            digits = re.sub(r"\D", "", m.group(0))
            if 13 <= len(digits) <= 19:
                out.card_number = digits
                break

        # expiry: 12/27, 12/2027, 12-2027, "December 2027"
        m = re.search(r"\b(0?[1-9]|1[0-2])\s*[/\-]\s*((?:20)?\d{2})\b", t)
        if m and not out.dob:
            out.expiry_month = int(m.group(1))
            out.expiry_year = int(m.group(2))
        else:
            m = re.search(r"\b([a-z]{3,9})\.?,?\s+((?:20)?\d{2})\b", low)
            if m and m.group(1) in _MONTHS and re.search(r"expir|valid|till|until|card", low):
                out.expiry_month = _MONTHS[m.group(1)]
                out.expiry_year = int(m.group(2))

        # dob - only if the text contains something actually date-shaped
        # (a bare number like "4321" must never be read as a date)
        out.dob = self._parse_dob(t, low)

        # aadhaar / pincode / cvv - labelled
        # ("last 4"/"last four" phrasing means we can't anchor on position;
        #  take the final standalone 4-digit group in an aadhaar-mentioning message)
        if re.search(r"aadhaa?r", low):
            groups = re.findall(r"\b\d{4}\b", low)
            if groups:
                out.aadhaar_last4 = groups[-1]
            else:
                sd = spoken_digits(low.split("aadhaar")[-1] if "aadhaar" in low else low)
                if sd and len(sd) == 4:
                    out.aadhaar_last4 = sd
        m = re.search(r"pin\s?code\D*((?:\d[\s]*){6})", low)
        if m:
            out.pincode = re.sub(r"\D", "", m.group(1))
        m = re.search(r"cvv\D*((?:[a-z]+[\s\-]*){3,4}|\d{3,4})", low)
        if m:
            raw = m.group(1)
            out.cvv = raw if raw.isdigit() else spoken_digits(raw)

        # bare values, interpreted by state
        bare = re.sub(r"\D", "", t)
        spoken = spoken_digits(low)
        if state == "VERIFY_FACTOR":
            cand = bare or (spoken or "")
            if len(cand) == 4 and not out.aadhaar_last4 and not out.dob:
                out.aadhaar_last4 = cand
            elif len(cand) == 6 and not out.pincode:
                out.pincode = cand
        elif state == "COLLECT_CARD":
            # A bare number is a CVV only when the message is *just* digits
            # (or spoken digits) - otherwise "expires December 2027" would
            # have its year misread as a CVV.
            if not out.cvv and not out.card_number and out.expiry_month is None:
                if re.fullmatch(r"[\d\s\-]+", t) and len(bare) in (3, 4):
                    out.cvv = bare
                elif spoken and len(spoken) in (3, 4) and not re.search(r"\d", t):
                    out.cvv = spoken

        # names - card-specific patterns first, then generic ones
        card_name = self._parse_card_name(t)
        if card_name:
            out.cardholder_name = card_name
        else:
            name = self._parse_name(t, low, state)
            if name:
                if state == "COLLECT_CARD":
                    out.cardholder_name = name
                else:
                    out.full_name = name

        return out

    # ── helpers ────────────────────────────────────────────

    # A DOB is only ever read from a date-SHAPED fragment. Bare numbers
    # ("4321", "400001") must fall through to Aadhaar/pincode handling -
    # a misparsed date would burn a strict-match verification attempt.
    DATE_PATTERNS = [
        r"\b\d{4}-\d{1,2}-\d{1,2}\b",                                        # 1990-05-14
        r"\b\d{1,2}[/.\-]\d{1,2}[/.\-]\d{2,4}\b",                            # 14-05-1990, 14/05/90
        r"\b\d{1,2}(?:st|nd|rd|th)?(?:\s+of)?\s+[a-z]{3,9}\.?,?\s+\d{2,4}\b",  # 14th May 1990
        r"\b[a-z]{3,9}\.?\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{2,4}\b",          # May 14, 90
    ]

    def _parse_dob(self, t: str, low: str) -> Optional[str]:
        month_words = set(_MONTHS)
        for pat in self.DATE_PATTERNS:
            for m in re.finditer(pat, low):
                frag = re.sub(r"(\d+)(st|nd|rd|th)\b", r"\1", m.group(0))
                words = re.findall(r"[a-z]{3,9}", frag)
                # if the fragment contains words, one must be a real month
                if words and not any(w in month_words or w[:3] in month_words for w in words):
                    continue
                try:
                    dt = dateparser.parse(frag, dayfirst=True)
                except (ValueError, OverflowError):
                    continue
                if dt and 1900 <= dt.year <= 2099:
                    return dt.strftime("%Y-%m-%d")
        return None

    CARD_NAME_PATTERNS = [
        r"name on (?:the )?card(?: is)?[:\s]+([a-zA-Z .'-]+)",
        r"cardholder(?:'s)?(?: name)?(?: is)?[:\s]+([a-zA-Z .'-]+)",
        r"card is under[:\s]+([a-zA-Z .'-]+)",
        r"([a-zA-Z][a-zA-Z .'-]+?)\s+(?:is )?on (?:the )?card\b",  # "Nithin Jain on the card"
    ]

    def _parse_card_name(self, t: str) -> Optional[str]:
        for p in self.CARD_NAME_PATTERNS:
            m = re.search(p, t, flags=re.IGNORECASE)
            if m:
                return self._clean_name(m.group(1))
        return None

    def _parse_name(self, t: str, low: str, state: str) -> Optional[str]:
        patterns = [
            r"full name is ([a-zA-Z .'-]+)",
            r"my name(?:'s| is)? ([a-zA-Z .'-]+)",
            r"name\s*[:\-]\s*([a-zA-Z .'-]+)",
            r"this is ([a-zA-Z .'-]+)",
            r"i am ([a-zA-Z .'-]+)",
            r"i'm ([a-zA-Z .'-]+)",
        ]
        for p in patterns:
            m = re.search(p, t, flags=re.IGNORECASE)
            if m:
                return self._clean_name(m.group(1))
        # "it's Nithin, Nithin Jain" -> take the part after the last comma
        m = re.search(r"it'?s\s+(.+)", t, flags=re.IGNORECASE)
        if m:
            cand = m.group(1).split(",")[-1]
            return self._clean_name(cand)
        # bare name typed exactly when we asked for a name
        if state in ("VERIFY_NAME", "COLLECT_CARD") and re.fullmatch(r"[a-zA-Z .'-]{2,60}", t):
            if not re.fullmatch(r"(yes|no|ok|okay|sure|hi|hello|hey)", low.strip(". !")):
                return self._clean_name(t)
        return None

    @staticmethod
    def _clean_name(raw: str) -> str:
        # strip trailing hedges but DO NOT change casing - matching is strict
        cleaned = re.sub(r"\b(i think|thanks|please|btw)\b.*$", "", raw, flags=re.IGNORECASE)
        return re.sub(r"\s+", " ", cleaned).strip(" .,!")


# ── Optional LLM layer ─────────────────────────────────────

_LLM_SCHEMA_HINT = """Extract any of these fields present in the user's message.
Return ONLY a JSON object; omit fields that are not present. Do not guess.
{
  "account_id": "string like ACC1001",
  "full_name": "person's full name EXACTLY as they stated it (preserve casing)",
  "dob": "date of birth normalized to YYYY-MM-DD",
  "aadhaar_last4": "4 digits",
  "pincode": "6 digits",
  "amount": number,
  "pay_full": true if they want to pay the full/entire balance,
  "card_number": "digits only",
  "cvv": "3-4 digits",
  "expiry_month": 1-12,
  "expiry_year": 4-digit year,
  "cardholder_name": "name on card",
  "yes": true if they are agreeing/confirming,
  "no": true if they are declining,
  "wants_exit": true if they want to stop/cancel the conversation
}"""


class LLMExtractor:
    """Gemini Flash structured extraction. Fails soft: any error returns
    an empty Extracted so the deterministic layer's result stands."""

    def __init__(self, model: Optional[str] = None):
        self.model = model or os.getenv("EXTRACTOR_MODEL", "gemini-2.5-flash")
        self._client = None
        key = os.getenv("GEMINI_API_KEY", "")
        if key and key != "your-key-here":
            try:
                from google import genai
                self._client = genai.Client(api_key=key)
            except Exception:
                self._client = None

    @property
    def available(self) -> bool:
        return self._client is not None

    def extract(self, text: str, state: str) -> Optional[Extracted]:
        """Returns None when the call itself failed (so the caller can fall
        back to rules entirely) vs an empty Extracted when the model
        confidently found nothing."""
        if not self._client:
            return None
        prompt = (
            f"Conversation stage: {state}\n"
            f"{_LLM_SCHEMA_HINT}\n\nUser message: {json.dumps(text)}"
        )
        try:
            resp = self._client.models.generate_content(
                model=self.model,
                contents=prompt,
                config={"temperature": 0, "response_mime_type": "application/json"},
            )
            data = json.loads(resp.text)
        except Exception:
            return None
        return self._from_json(data)

    @staticmethod
    def _from_json(data: Dict) -> Extracted:
        out = Extracted()
        if not isinstance(data, dict):
            return out
        for f in out.__dataclass_fields__:
            if f in data and data[f] not in (None, "", []):
                setattr(out, f, data[f])
        # normalize types defensively
        if out.expiry_month is not None:
            try: out.expiry_month = int(out.expiry_month)
            except (TypeError, ValueError): out.expiry_month = None
        if out.expiry_year is not None:
            try: out.expiry_year = int(out.expiry_year)
            except (TypeError, ValueError): out.expiry_year = None
        if out.amount is not None:
            try: out.amount = float(out.amount)
            except (TypeError, ValueError): out.amount = None
        for sfield in ("aadhaar_last4", "pincode", "cvv", "card_number"):
            v = getattr(out, sfield)
            if v is not None:
                setattr(out, sfield, re.sub(r"\D", "", str(v)))
        if out.account_id:
            out.account_id = re.sub(r"[\s\-]", "", str(out.account_id)).upper()
        return out


# ── Facade the agent uses ──────────────────────────────────


class Extractor:
    """Merge policy:

    - LLM unavailable or its call failed  -> rules result stands alone.
    - LLM succeeded -> LLM is PRIMARY. Rules only fill pattern-shaped
      fields the LLM omitted (IDs, digits, dates, amounts, intents).
      Name fields are LLM-only in this mode: the rule heuristics can
      capture whole sentences as "names", and a wrong name burns a
      strict-match verification attempt - a redundant re-ask is cheaper
      than a wrong value.
    """

    NAME_FIELDS = {"full_name", "cardholder_name"}

    def __init__(self):
        self.rules = DeterministicExtractor()
        self.llm = LLMExtractor()

    def extract(self, text: str, state: str) -> Extracted:
        rules = self.rules.extract(text, state)
        if not self.llm.available:
            return rules
        llm = self.llm.extract(text, state)
        if llm is None:  # transport/parse failure - fail soft to rules
            return rules
        for f in llm.__dataclass_fields__:
            if f in self.NAME_FIELDS:
                continue
            if getattr(llm, f) in (None, False) and getattr(rules, f) not in (None, False):
                setattr(llm, f, getattr(rules, f))
        return llm
