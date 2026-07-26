# Design Document - Payment Collection Agent

## Architecture overview

```
                 user text
                     │
        ┌────────────▼────────────┐
        │  Extractor (two layers) │   "what did they say?"
        │  1. rules: regex/dates  │
        │  2. LLM: Gemini Flash,  │   temp 0, JSON schema, optional
        │     fills same schema   │
        └────────────┬────────────┘
                     │  structured fields only
        ┌────────────▼────────────┐
        │  Agent state machine    │   "what happens next?"
        │  GREET → … → CONFIRM    │   slots memory, retry counters
        └──────┬───────────┬──────┘
               │           │
     ┌─────────▼───┐   ┌───▼─────────┐
     │ validators  │   │ verification│    all deterministic
     │ Luhn/expiry/│   │ strict match│
     │ amount/date │   │ + lockout   │
     └─────────┬───┘   └─────────────┘
               │  only validated payloads
        ┌──────▼──────┐
        │  api_client │ → lookup-account / process-payment
        └─────────────┘
```

**The core principle: the LLM interprets, deterministic code decides.**
The LLM's only job is turning "expires December twenty seven" into
`{expiry_month: 12, expiry_year: 2027}`. It never chooses state
transitions, never validates, never sees account data, and its output
passes through the same validators as the rules layer. The worst outcome
of a bad LLM parse is a redundant question - never a wrong payment.

## Key decisions and why

**1. Hybrid extraction, LLM-primary with a deterministic floor.**
Rules (regex, dateutil, number-words) cover every documented phrasing and
keep the agent fully functional and reproducible with no API key. The LLM
layer (Gemini Flash - extraction is a bounded task; a small fast model is
the right cost/latency choice) handles the unbounded long tail: negations
("not ACC1001, use ACC1002"), typos, casual confirmations ("sure, go
ahead"). When both produce values, the LLM wins on semantic fields and
name fields are LLM-only - testing showed rule heuristics can capture
whole sentences as "names", and a wrong name burns a strict-match attempt,
while a re-ask costs nothing. If the LLM call fails, rules stand alone
(fail-soft, no crash, no stall).

**2. Rule-based verification, not LLM-based.** Identity comparison is
`==` in Python. An LLM has no place in a security decision: it can be
persuaded, and it can't be audited. Matching is exact and case-sensitive
per the spec ("no case-insensitive workarounds") - the only liberty taken
is whitespace normalization (documented assumption: double spaces are a
typing artifact, not an identity difference). The failure message tells
users that spelling *and capitalization* matter, which gives honest users
a fair retry without weakening the rule.

**3. A date is only ever parsed from a date-shaped fragment.** Early
testing found the fuzzy parser reading a bare "4321" (an Aadhaar answer)
as the year 4321 - silently burning a verification attempt. Dates are now
recognized only in explicit shapes (ISO, 14-05-1990, "14th May 1990",
"May 14, 90"), which also makes 1988-02-29 (a real leap-year date) pass
and 1989-02-29 fail naturally via `datetime`.

**4. Slot memory: store early, use in order.** Volunteered info (e.g.
card details given before verification) is remembered but only *used*
when its step arrives - satisfying both "don't re-ask" and "don't skip
steps". Nothing collected pre-verification is acted on before
verification passes.

**5. Security posture.** Account data from the lookup is never echoed -
the agent only confirms whether the *user's* input matched. Card numbers
appear only masked; card number + CVV are wiped after every payment
attempt (a retry re-collects the CVV deliberately, mirroring
no-CVV-retention practice). Failed lookups, verification, and card
attempts each have a 3-strike limit with a clean, polite close.

**6. Failure taxonomy.** Every API outcome maps to one of: user-fixable
(invalid card/CVV/expiry → clear that field, guide a retry), flow-level
(insufficient balance → re-ask amount), or terminal/transport (our-side
apology, retry once, close cleanly; never blame the user for a 500).

## Tradeoffs accepted

- **Latency:** one LLM call per turn (~0.5–1s). Acceptable for chat; the
  deterministic mode exists where it isn't.
- **Vague references re-ask instead of guessing.** "Half of that" or
  "same name as the account" produce a clarifying question, not an
  inference. For money movement, a guess is worse than a question.
- **Strict name matching hurts honest-but-sloppy users** (lowercase
  typers). Spec-mandated; mitigated through explicit guidance in the
  retry message.
- **Session model is single-conversation.** No persistence between
  `Agent()` instances - matches the evaluation interface.

## What I'd improve with more time

- **Context-aware extraction:** pass safe context (e.g. outstanding
  balance) so "half of that" resolves - with deterministic arithmetic,
  not LLM math.
- **Conversation summarization** for very long sessions (token control).
- **Idempotency keys** on process-payment to make client-side retries
  double-charge-safe if the API supported them.
- **Structured logging/tracing** with PII redaction for production
  observability, and latency percentiles per turn.
- **Broader persona fleet** in evaluation (multilingual, adversarial
  prompt-injection personas attempting to extract account data).

## Evaluation approach

Three layers (details in README):
1. **Scripted scenarios** (13) - per-turn assertions + a privacy checker
   that greps agent output for account secrets. Run in rules-only mode
   (deterministic CI gate) and through the LLM path.
2. **Tool-call correctness** (19 checks) - a mock API records calls to
   assert timing (never before verification/confirmation), payload
   validity (Luhn, 2dp amounts, normalized IDs), and error-code handling.
3. **LLM persona simulation** - mirrors the grading method: an LLM plays
   a chatty payer, a wrong-identity fraudster, and a terse full-payer;
   outcomes are judged by code (transaction reached / locked out / no
   leaks). Current: 13/13, 19/19, 3/3.

**Correctness definition per step:** account resolved to a normalized ID;
verification passes only on exact matches; amount within balance and 2dp;
card fields valid before any API call; success surfaces the transaction
ID; every failure ends in either a guided retry or a clean close.

**Observed weaknesses:** vague quantity references re-ask rather than
resolve; rules-only mode can't handle negations or typos (by design -
that's the LLM layer's job); persona simulation is non-deterministic by
nature, so it's exploratory rather than a CI gate.
