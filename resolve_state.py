import requests
import json
import re
import os
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def resolve_state_code(city_input: str) -> str:
    """
    Takes a city name or a sentence containing a city name and returns
    the two-letter US state code using OpenRouter (Gemini Flash).

    Args:
        city_input: A city name (e.g. "Austin") or a sentence
                    containing a city (e.g. "The victim was found in Austin").

    Returns:
        A two-letter state code (e.g. "TX"), or an error message if not resolved.
    """
    prompt = (
        f"Given the following input, identify the US city mentioned and respond "
        f"with ONLY the two-letter US state abbreviation for that city. "
        f"Do not include any other text, explanation, or punctuation. "
        f"Just the two capital letters.\n\n"
        f"Input: \"{city_input}\"\n\n"
        f"State abbreviation:"
    )

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/cold-cases",
        "X-Title": "Cold Cases State Lookup",
    }

    payload = {
        "model": "google/gemini-2.0-flash-lite-001",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 4,
    }

    try:
        response = requests.post(
            OPENROUTER_URL, headers=headers, json=payload, timeout=15
        )
        response.raise_for_status()
        result = response.json()["choices"][0]["message"]["content"].strip()

        # Extract just the two-letter code from the response
        match = re.search(r"\b([A-Z]{2})\b", result)
        if match:
            return match.group(1)
        return f"Could not parse state code from response: {result}"

    except requests.ConnectionError:
        return "Error: Cannot connect to OpenRouter API."
    except requests.Timeout:
        return "Error: OpenRouter request timed out."
    except Exception as e:
        return f"Error: {e}"


if __name__ == "__main__":
    code = resolve_state_code("miami")
    print(f"{'miami'!r:50s} -> {code}")
