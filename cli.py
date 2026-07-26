"""Interactive chat with the payment agent.

Usage:  python cli.py
"""

from dotenv import load_dotenv

load_dotenv()

from agent import Agent  # noqa: E402  (after load_dotenv on purpose)


def main() -> None:
    agent = Agent()
    print("Payment Agent CLI — type 'quit' to leave.\n")
    print(f"Agent: {agent.next('Hi')['message']}")
    while True:
        try:
            user = input("You:   ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        if user.lower() in {"quit", "q"}:
            break
        reply = agent.next(user)
        print(f"Agent: {reply['message']}")


if __name__ == "__main__":
    main()
