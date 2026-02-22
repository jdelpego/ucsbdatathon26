import json
import os

_DIR = os.path.dirname(os.path.abspath(__file__))
_JSON_PATH = os.path.join(_DIR, "state_judges.json")

with open(_JSON_PATH) as f:
    _STATE_JUDGES: dict[str, list[str]] = json.load(f)


def get_state_judges(state_code: str) -> list[str]:
    """Return the list of judges for a two-letter US state code.

    Args:
        state_code: Two-letter state abbreviation (e.g. "TX", "CA").

    Returns:
        List of judge names, or an empty list if the code is not found.
    """
    return _STATE_JUDGES.get(state_code.upper(), [])


if __name__ == "__main__":
    code = input("Enter a two-letter state code: ").strip()
    judges = get_state_judges(code)
    if judges:
        print(f"\nJudges for {code.upper()} ({len(judges)}):")
        for name in judges:
            print(f"  - {name}")
    else:
        print(f"No judges found for '{code}'.")
