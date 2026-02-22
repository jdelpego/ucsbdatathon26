import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def get_district_judges(location_description: str, judges_list: list[str]) -> list[str]:
    """
    Given a location description and a list of judge names, uses OpenRouter
    LLM (with web search) to determine which judges are currently active
    for the federal district / division covering that location.

    Args:
        location_description: A sentence describing a location,
            e.g. "I need help on a case in San Diego".
        judges_list: A list of judge names to filter from.

    Returns:
        A list of judge names from judges_list that are active in the
        relevant federal district court / division.
    """
    judges_str = "\n".join(f"  - {name}" for name in judges_list)

    prompt = (
        f"Given the location below, determine the US federal district court "
        f"and division that covers it. From the judges list, return ONLY the "
        f"names of currently active judges (including senior status) in that "
        f"district as a JSON array. If none match, return [].\n\n"
        f"Location: \"{location_description}\"\n\n"
        f"Judges:\n{judges_str}\n\n"
        f"JSON array only:"
    )

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/cold-cases",
        "X-Title": "Cold Cases Judge Lookup",
    }

    payload = {
        "model": "google/gemini-2.0-flash-lite-001",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.0,
    }

    try:
        response = requests.post(
            OPENROUTER_URL, headers=headers, json=payload, timeout=20
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"].strip()

        # Strip markdown code fences if present
        if content.startswith("```"):
            content = content.split("\n", 1)[1] if "\n" in content else content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

        result = json.loads(content)

        # Validate that returned names are actually in the input list
        valid = [name for name in result if name in judges_list]
        return valid

    except requests.ConnectionError:
        print("Error: Cannot connect to OpenRouter API.")
        return []
    except requests.Timeout:
        print("Error: OpenRouter request timed out.")
        return []
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        print(f"Error parsing response: {e}")
        print(f"Raw content: {content if 'content' in dir() else 'N/A'}")
        return []
    except Exception as e:
        print(f"Error: {e}")
        return []


if __name__ == "__main__":
    from get_state_judges import get_state_judges

    # Example: look up judges for a location in California
    location = "I need help on a case in San Diego"
    state_judges = get_state_judges("CA")

    print(f"Location: {location}")
    print(f"Total CA judges in list: {len(state_judges)}")
    print(f"\nQuerying OpenRouter for active judges in that district...\n")

    active = get_district_judges(location, state_judges)

    if active:
        print(f"Active judges for this district ({len(active)}):")
        for name in active:
            print(f"  - {name}")
    else:
        print("No matching active judges found.")
