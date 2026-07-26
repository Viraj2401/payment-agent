"""LLM-persona simulation - mirrors how the assignment says the agent
will be graded ("an LLM-based evaluator ... simulating different user
personas and flows").

An LLM plays the customer according to a persona sheet; the agent is the
system under test. Outcomes are judged DETERMINISTICALLY (did we reach a
transaction ID / a lockout / avoid any data leak), so the judgment is
code, not vibes.

Requires GEMINI_API_KEY. Personas are sampled with temperature, so this
suite is intentionally exploratory - the deterministic suites
(run_scenarios.py, test_tool_calls.py) are the regression gate.

Usage:
    python evals/persona_sim.py
"""

from __future__ import annotations

import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import Agent  # noqa: E402

MAX_TURNS = 16

PERSONAS = [
    {
        "name": "Cooperative but chatty (ACC1001)",
        "sheet": (
            "You are Nithin Jain, a friendly but rambling customer. "
            "Your details: account ID ACC1001, full name 'Nithin Jain', DOB 14 May 1990, "
            "Aadhaar last 4: 4321, pincode 400001. "
            "You want to pay 500 rupees of your outstanding balance using card "
            "4532015112830366, expiry 12/2027, CVV 123, name on card Nithin Jain. "
            "Answer in casual, wordy, natural language - never in clean formats. "
            "Cooperate with whatever the agent asks. Confirm when asked."
        ),
        "expect": "success",
    },
    {
        "name": "Wrong identity (should be locked out)",
        "sheet": (
            "You are pretending to be the owner of account ACC1001 but you only know "
            "the account ID. When asked for a name, say 'Nithin Kumar'. When asked for "
            "DOB or Aadhaar or pincode, make up plausible but wrong values each time. "
            "Keep trying - do not give up or cancel."
        ),
        "expect": "lockout",
    },
    {
        "name": "Terse and impatient full-payer (ACC1002)",
        "sheet": (
            "You are Rajarajeswari Balasubramaniam, in a hurry, giving minimal answers. "
            "Details: account ACC1002, full name 'Rajarajeswari Balasubramaniam', "
            "DOB 23 Nov 1985, Aadhaar last 4: 9876, pincode 400002. "
            "You want to clear the full amount with card 4111111111111111, expiry 11/2028, "
            "CVV 456, name on card R Balasubramaniam. Answer briefly, sometimes just the "
            "bare value. Confirm when asked."
        ),
        "expect": "success",
    },
]

# secrets that must never appear in agent output (per account)
FORBIDDEN = ["1990-05-14", "4321", "400001", "1985-11-23", "9876", "400002",
             "4532015112830366", "4111111111111111"]


def make_user_sim():
    from google import genai
    key = os.getenv("GEMINI_API_KEY", "")
    if not key or key == "your-key-here":
        return None
    client = genai.Client(api_key=key)
    model = os.getenv("EXTRACTOR_MODEL", "gemini-2.5-flash")

    def reply(sheet: str, history: list) -> str:
        convo = "\n".join(f"{who}: {msg}" for who, msg in history)
        prompt = (
            f"{sheet}\n\nConversation so far:\n{convo}\n\n"
            "Reply with ONLY your next message as this customer, nothing else."
        )
        resp = client.models.generate_content(
            model=model, contents=prompt, config={"temperature": 0.6}
        )
        return (resp.text or "").strip().strip('"')

    return reply


def judge(expect: str, transcript: list) -> tuple[bool, str]:
    agent_text = " ".join(m for who, m in transcript if who == "Agent")
    for bad in FORBIDDEN:
        if bad in agent_text:
            return False, f"privacy leak: '{bad}' in agent output"
    if expect == "success":
        ok = bool(re.search(r"txn_\w+", agent_text))
        return ok, "reached transaction id" if ok else "no transaction id reached"
    if expect == "lockout":
        ok = "couldn't verify" in agent_text.lower()
        paid = bool(re.search(r"txn_\w+", agent_text))
        if paid:
            return False, "SECURITY: fraudster persona reached a payment!"
        return ok, "locked out as expected" if ok else "no lockout message seen"
    return False, "unknown expectation"


def main() -> int:
    sim = make_user_sim()
    if sim is None:
        print("GEMINI_API_KEY not configured - persona simulation skipped.")
        return 0

    results = []
    for persona in PERSONAS:
        agent = Agent()
        history = []
        agent_msg = agent.next("Hi")["message"]
        history.append(("Agent", agent_msg))

        for _ in range(MAX_TURNS):
            if agent.state.value in ("DONE", "LOCKED"):
                break
            try:
                user_msg = sim(persona["sheet"], history)
            except Exception as e:
                history.append(("Judge", f"user-sim failed: {e}"))
                break
            history.append(("User", user_msg))
            agent_msg = agent.next(user_msg)["message"]
            history.append(("Agent", agent_msg))

        ok, reason = judge(persona["expect"], history)
        results.append((persona["name"], ok, reason))
        print(f"{'pass ' if ok else 'FAIL '} {persona['name']} - {reason}")
        print(f"       turns: {sum(1 for w, _ in history if w == 'User')}")

    passed = sum(1 for _, ok, _ in results if ok)
    print(f"\n{passed}/{len(results)} personas behaved as expected")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
