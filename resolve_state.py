import requests
import json
import re


OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3"


def resolve_state_code(city_input: str) -> str:
    """
    Takes a city name or a sentence containing a city name and returns
    the two-letter US state code using a local Ollama llama3 model.

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

    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=30)
        response.raise_for_status()
        result = response.json().get("response", "").strip()

        # Extract just the two-letter code from the response
        match = re.search(r"\b([A-Z]{2})\b", result)
        if match:
            return match.group(1)
        return f"Could not parse state code from response: {result}"

    except requests.ConnectionError:
        return "Error: Cannot connect to Ollama. Is it running on localhost:11434?"
    except requests.Timeout:
        return "Error: Ollama request timed out."
    except Exception as e:
        return f"Error: {e}"


if __name__ == "__main__":
    code = resolve_state_code("miami")
    print(f"{"miami"!r:50s} -> {code}")
