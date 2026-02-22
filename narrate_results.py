import os
import subprocess
import sys
import requests
from dotenv import load_dotenv

load_dotenv()

ELEVENLABS_API_KEY = os.environ["ELEVENLABS_API_KEY"]

# "Daniel" — warm, storytelling male voice
DEFAULT_VOICE_ID = "onwK4e9ZLuTAKqWW03F9"  # Daniel
TTS_URL = f"https://api.elevenlabs.io/v1/text-to-speech/{DEFAULT_VOICE_ID}/stream"

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "narration_output.mp3")


def build_narration_script(
    prompt: str,
    district_judges: list[str],
    judge_pmf: dict[str, float],
    judge_attorney_rankings: dict[str, dict[str, int]],
    attorney_rankings: dict[str, float],
) -> str:
    """
    Build an engaging, Disney-storyteller-style narration from the pipeline results.
    """
    # Top 3 attorneys
    sorted_attorneys = list(attorney_rankings.items())
    best = sorted_attorneys[0] if len(sorted_attorneys) > 0 else None
    second = sorted_attorneys[1] if len(sorted_attorneys) > 1 else None
    third = sorted_attorneys[2] if len(sorted_attorneys) > 2 else None

    # Find the judge most associated with the top attorney
    top_judge_for_best = None
    top_score_for_best = -1
    if best:
        for judge, ranking in judge_attorney_rankings.items():
            score = ranking.get(best[0], 0)
            if score > top_score_for_best:
                top_score_for_best = score
                top_judge_for_best = judge

    judges_list = ", ".join(district_judges[:-1])
    if len(district_judges) > 1:
        judges_list += f", and {district_judges[-1]}"
    elif district_judges:
        judges_list = district_judges[0]

    script = (
        f"Alright, here's what I found. "
        f"You've got {len(district_judges)} federal district judges "
        f"who could be assigned your case. "
    )

    if top_judge_for_best and best:
        script += (
            f"Now here's the interesting part — Judge {top_judge_for_best} "
            f"has a strong track record with {best[0]}, "
            f"scoring {top_score_for_best} points. That connection matters. "
        )

    if best:
        script += (
            f"After weighing all the judges and probabilities, "
            f"your top attorney is {best[0]} with a score of {best[1]:.0f}. "
        )

    if second:
        script += (
            f"Your second best option is {second[0]} at {second[1]:.0f}. "
        )

    if third:
        script += (
            f"And third, {third[0]} at {third[1]:.0f}. "
        )

    script += "That's your game plan. Now go make it happen!"

    return script


def narrate(text: str, output_path: str = OUTPUT_PATH) -> str:
    """
    Convert text to speech using ElevenLabs streaming TTS, save to file,
    and start playback as soon as the first bytes arrive.

    Args:
        text: The narration script to speak.
        output_path: Where to save the MP3 file.

    Returns:
        Path to the saved audio file, or empty string on error.
    """
    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json",
    }

    payload = {
        "text": text,
        "model_id": "eleven_turbo_v2_5",
        "voice_settings": {
            "stability": 0.35,
            "similarity_boost": 0.80,
            "style": 0.65,
            "use_speaker_boost": False,
        },
    }

    try:
        response = requests.post(
            TTS_URL, headers=headers, json=payload, timeout=120, stream=True
        )
        response.raise_for_status()

        # Stream chunks to file as they arrive
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=4096):
                if chunk:
                    f.write(chunk)

        print(f"🔊  Narration saved to {output_path}")

        # Play the saved file
        if sys.platform == "darwin":
            subprocess.run(["afplay", output_path])
        elif sys.platform == "linux":
            subprocess.run(["mpg123", output_path])
        else:
            subprocess.run(["start", output_path], shell=True)

        return output_path

    except requests.ConnectionError:
        print("Error: Cannot connect to ElevenLabs API.")
        return ""
    except requests.Timeout:
        print("Error: ElevenLabs request timed out.")
        return ""
    except Exception as e:
        print(f"Error: {e}")
        return ""


if __name__ == "__main__":
    sample_text = (
        "Well, well, well... let me take a look at what we've got here. "
        "This is just a test of the narration system."
    )
    narrate(sample_text)
