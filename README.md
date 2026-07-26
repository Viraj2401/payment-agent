# Payment Collection AI Agent

A conversational agent that takes a user from account lookup through strict
identity verification to a confirmed card payment - built for messy,
real-world language, with every consequential decision made by deterministic
code rather than an LLM.

```
User:  yeah my account number is ACC 1001 I think
Agent: Found your account. For security I need to verify your identity first...
...
Agent: Payment successful! ₹500.00 has been received. Your transaction ID is txn_...
```

## Quick start

```bash
pip install -r requirements.txt

# optional but recommended - enables LLM extraction of messy input:
cp .env.example .env        # then put your Gemini API key in .env

python cli.py               # interactive chat
```

**No API key? It still works.** The agent degrades gracefully to a
deterministic parser (regex + date parsing + number words) that covers all
standard phrasings. The LLM layer widens coverage to arbitrary natural
language; it is never required for correctness.

> Windows note: if you see a `UnicodeEncodeError` about `₹`, run
> `set PYTHONIOENCODING=utf-8` (or `$env:PYTHONIOENCODING='utf-8'` in
> PowerShell) first. Purely a console encoding quirk.

## Required interface

```python
from agent import Agent

agent = Agent()
agent.next("Hi")
# -> {"message": "Hi! I'm here to help you take care of a pending payment..."}
```

State is held inside the `Agent` instance; each `next()` call is one turn; no
external setup between turns. With no `GEMINI_API_KEY` set, behavior is fully
deterministic across repeated runs; with a key, extraction runs at
temperature 0.

## Project structure

| File | Responsibility |
|---|---|
| `agent.py` | State machine + required `Agent.next()` interface. Owns all flow decisions. |
| `extractor.py` | Messy text → structured fields. Deterministic rules layer + optional Gemini layer (temp 0, JSON schema). LLM output never bypasses validation. |
| `validators.py` | Luhn, CVV (Amex-aware), expiry, amount (≤ 2dp, ≤ balance), leap-year-correct dates. Gate before every API call. |
| `verification.py` | Strict identity matching (exact name + one factor) with a 3-attempt lockout. |
| `api_client.py` | Both API endpoints; timeouts; the 6 documented error codes mapped; transport failures kept separate from user errors. |
| `cli.py` | Interactive chat loop. |
| `evals/` | Three-layer evaluation suite (below). |

## Conversation flow

```
GREET → ACCOUNT_ID → VERIFY_NAME → VERIFY_FACTOR → ASK_AMOUNT
      → COLLECT_CARD → CONFIRM → DONE
                                (LOCKED on verification lockout)
```

Users can volunteer information out of order at any point - it is stored in
slots and consumed when its step arrives, so steps are never skipped and
nothing is ever re-asked.

## Security behavior

- Account data (DOB, Aadhaar, pincode, balance-before-verification) is
  **never echoed** to the user - the agent only says whether *their* input matched.
- Verification is **strict**: exact name match (case-sensitive) plus one exact
  secondary factor. No fuzzy matching.
- Card numbers only ever appear masked (`ending 0366`); card number and CVV
  are **wiped from memory after every payment attempt** - a retry re-collects
  the CVV by design.
- Nothing invalid reaches the API: Luhn, CVV length, expiry, and amount are
  all validated client-side first.

## Evaluation

Three layers, from fully deterministic to adversarial:

```bash
python evals/run_scenarios.py      # 13 scripted conversations, per-turn assertions
                                   #   + privacy checks (no account data in output)
python evals/run_scenarios.py --llm  # same suite through the LLM extraction path
python evals/test_tool_calls.py    # 19 checks against a mock API: call timing,
                                   #   payload correctness, error-code handling
python evals/persona_sim.py        # LLM plays 3 user personas (chatty, fraudster,
                                   #   terse); outcomes judged deterministically
python evals/generate_transcripts.py  # regenerates sample_conversations.md
                                      #   from real runs
```

Current results: **13/13 scenarios (both modes), 19/19 tool-call checks,
3/3 personas** (including a wrong-identity persona that must get locked out
without ever reaching a payment).

See [DESIGN.md](DESIGN.md) for architecture rationale and
[sample_conversations.md](sample_conversations.md) for the four required
sample flows, generated from real runs.
