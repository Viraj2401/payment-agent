"""Scripted scenario runner for the payment agent.

Runs deterministic conversation scripts against a fresh Agent per
scenario and checks per-turn expectations plus global privacy rules.

By default runs in RULES-ONLY mode (LLM disabled) so results are fully
deterministic; pass --llm to exercise the LLM extraction layer.

Usage:
    python evals/run_scenarios.py          # deterministic
    python evals/run_scenarios.py --llm    # with LLM extraction
"""

from __future__ import annotations

import os
import sys

if "--llm" not in sys.argv:
    os.environ["GEMINI_API_KEY"] = ""  # force deterministic fallback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import Agent  # noqa: E402


# Each scenario: name, turns as (user_input, [expected substrings in reply]),
# and forbidden strings that must NEVER appear in any agent message.
SCENARIOS = [
    {
        "name": "A. Happy path (messy inputs)",
        "turns": [
            ("Hi", ["account id"]),
            ("yeah my account number is ACC 1001 I think", ["full name"]),
            ("my name is Nithin Jain", ["date of birth"]),
            ("I was born on 14th May 1990", ["verified", "1,250.75"]),
            ("can I do 500 for now?", ["card"]),
            ("the card number is 4532 0151 1283 0366, name on card is Nithin Jain", ["expiry"]),
            ("expires December 2027", ["cvv"]),
            ("CVV is 123", ["confirm", "0366"]),
            ("yes", ["successful", "txn_"]),
        ],
        "forbidden": ["4321", "400001", "4532015112830366"],
    },
    {
        "name": "B. Verification lockout after 3 failures",
        "turns": [
            ("Hi", ["account id"]),
            ("ACC1001", ["full name"]),
            ("my name is Nithin Jane", ["doesn't match"]),
            ("Nithin Jain", ["date of birth"]),
            ("DOB is 1991-01-01", ["doesn't match"]),
            ("Aadhaar last 4 is 9999", ["couldn't verify", "support"]),
            ("hello?", ["session has ended"]),
        ],
        "forbidden": ["1990-05-14", "4321", "400001", "1,250.75"],
    },
    {
        "name": "C. Zero balance (ACC1003)",
        "turns": [
            ("Hi", ["account id"]),
            ("ACC1003", ["full name"]),
            ("Priya Agarwal", ["date of birth"]),
            ("pincode 400003", ["0.00", "nothing to pay"]),
            ("ok", ["session has ended"]),
        ],
        "forbidden": ["1992-08-10", "2468"],
    },
    {
        "name": "D1. Leap-year DOB accepted (ACC1004)",
        "turns": [
            ("Hi", ["account id"]),
            ("ACC1004", ["full name"]),
            ("Rahul Mehta", ["date of birth"]),
            ("I was born on 29 Feb 1988", ["verified", "3,200.50"]),
        ],
        "forbidden": ["1357", "400004"],
    },
    {
        "name": "D2. Near-miss date fails, correct date then passes",
        "turns": [
            ("Hi", ["account id"]),
            ("ACC1004", ["full name"]),
            ("Rahul Mehta", ["date of birth"]),
            ("28 Feb 1988", ["doesn't match"]),
            ("sorry, 29 Feb 1988", ["verified"]),
        ],
        "forbidden": ["1357", "400004"],
    },
    {
        "name": "E. Invalid card (Luhn) x3 -> polite close",
        "turns": [
            ("Hi", ["account id"]),
            ("ACC1001", ["full name"]),
            ("Nithin Jain", ["date of birth"]),
            ("4321", ["verified"]),
            ("pay 100", ["card"]),
            ("4532015112830367", ["checksum"]),
            ("4532015112830367", ["checksum"]),
            ("4532015112830367", ["unsuccessful attempts"]),
        ],
        "forbidden": [],
    },
    {
        "name": "F. Expired card message",
        "turns": [
            ("Hi", ["account id"]),
            ("ACC1001", ["full name"]),
            ("Nithin Jain", ["date of birth"]),
            ("1990-05-14", ["verified"]),
            ("pay 100", ["card"]),
            ("card 4532015112830366, Nithin Jain on the card, cvv 123", ["expiry"]),
            ("01/2020", ["expired"]),
        ],
        "forbidden": [],
    },
    {
        "name": "G. Overpayment re-asks (ACC1002)",
        "turns": [
            ("Hi", ["account id"]),
            ("ACC1002", ["full name"]),
            ("Rajarajeswari Balasubramaniam", ["date of birth"]),
            ("pincode is 400002", ["verified", "540.00"]),
            ("I want to pay 5000", ["more than the outstanding balance"]),
        ],
        "forbidden": ["1985-11-23", "9876"],
    },
    {
        "name": "H. 'Clear the full amount' (ACC1002)",
        "turns": [
            ("Hi", ["account id"]),
            ("ACC1002", ["full name"]),
            ("Rajarajeswari Balasubramaniam", ["date of birth"]),
            ("9876", ["verified"]),
            ("just clear the full amount", ["540.00", "card"]),
        ],
        "forbidden": [],
    },
    {
        "name": "I. Out-of-order info in one message",
        "turns": [
            ("Hi, I'm Nithin Jain, account ACC1001, DOB 1990-05-14", ["verified", "1,250.75"]),
        ],
        "forbidden": ["4321", "400001"],
    },
    {
        "name": "J. Cancel mid-flow",
        "turns": [
            ("Hi", ["account id"]),
            ("ACC1001", ["full name"]),
            ("cancel", ["cancelled", "nothing was charged"]),
            ("hello", ["session has ended"]),
        ],
        "forbidden": [],
    },
    {
        "name": "K. Unknown account x3 -> close",
        "turns": [
            ("Hi", ["account id"]),
            ("ACC9999", ["couldn't find"]),
            ("ACC8888", ["couldn't find"]),
            ("ACC7777", ["couldn't find an account after several tries"]),
            ("ACC1001", ["session has ended"]),
        ],
        "forbidden": [],
    },
    {
        "name": "L. Card volunteered before verification is not used early",
        "turns": [
            ("Hi", ["account id"]),
            ("ACC1001 and my card number is 4532015112830366", ["full name"]),
            ("Nithin Jain", ["date of birth"]),
            ("400001", ["verified"]),
        ],
        "forbidden": ["4532015112830366"],
    },
]


def run() -> int:
    passed = failed = 0
    failures = []

    for sc in SCENARIOS:
        agent = Agent()
        transcript = []
        sc_failures = []

        for i, (user, expected) in enumerate(sc["turns"]):
            reply = agent.next(user)["message"]
            transcript.append((user, reply))
            low = reply.lower()
            for exp in expected:
                if exp.lower() not in low:
                    sc_failures.append(
                        f"  turn {i+1}: expected '{exp}' in reply\n    user : {user}\n    agent: {reply}"
                    )

        joined = " ".join(r for _, r in transcript)
        for bad in sc["forbidden"]:
            if bad in joined:
                sc_failures.append(f"  PRIVACY: forbidden string '{bad}' appeared in agent output")

        if sc_failures:
            failed += 1
            failures.append((sc["name"], sc_failures, transcript))
            print(f"FAIL  {sc['name']}")
        else:
            passed += 1
            print(f"pass  {sc['name']}")

    print(f"\n{passed} passed, {failed} failed out of {len(SCENARIOS)}")
    for name, errs, transcript in failures:
        print(f"\n=== {name} ===")
        for e in errs:
            print(e)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(run())
