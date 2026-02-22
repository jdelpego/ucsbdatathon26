"""
Flask backend for the Cold Cases visual storytelling app.

Endpoints:
    POST /api/transcribe   — upload a WAV blob, get back transcribed text
    POST /api/pipeline      — run the full pipeline from a text prompt
    GET  /                  — serve the frontend
"""

import base64
import json
import os
import re
import requests

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

# Import existing pipeline modules
from resolve_state import resolve_state_code
from get_state_judges import get_state_judges
from get_district_judges import get_district_judges
from judge_probability import get_judge_pmf
from attorney_ranking import get_judge_attorney_rankings, get_weighted_attorney_rankings

ELEVENLABS_API_KEY = os.environ["ELEVENLABS_API_KEY"]
OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]

VOICE_ID = "pNInz6obpgDQGcFmaJgB"  # Adam — deep American male
ELEVENLABS_TTS_TIMESTAMPS_URL = (
    f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}/with-timestamps"
)
ELEVENLABS_STT_URL = "https://api.elevenlabs.io/v1/speech-to-text"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

app = Flask(__name__, static_folder="frontend", static_url_path="")
CORS(app)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _get_judge_image_urls(judges: list[str]) -> dict[str, str]:
    """Use an LLM with web search to find real judge headshot URLs from Google."""
    judges_str = ", ".join(judges)
    prompt = (
        f"Search Google Images for official headshot photos of each of these "
        f"US federal judges. Find the real photo URL (the actual image file URL "
        f"ending in .jpg, .png, or .webp) from official court websites, Wikipedia, "
        f"or Ballotpedia. Do NOT use placeholder or generic URLs.\n\n"
        f"Judges: {judges_str}\n\n"
        f"Return ONLY a JSON object mapping each judge's full name to the "
        f"direct image URL. If you truly cannot find a photo, use null. JSON only:"
    )
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "google/gemini-2.0-flash-001",
        "plugins": [{"id": "web"}],
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
    }
    try:
        resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"].strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[1] if "\n" in content else content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
        return json.loads(content)
    except Exception as e:
        print(f"Image URL fetch error: {e}")
        return {}


def _build_narration_and_events(
    prompt: str,
    district_judges: list[str],
    judge_pmf: dict[str, float],
    judge_attorney_rankings: dict[str, dict[str, int]],
    attorney_rankings: dict[str, float],
    judge_images: dict[str, str | None],
) -> tuple[str, list[dict]]:
    """
    Build narration script and a list of visual events keyed by sentence.
    Each event has: text, type (intro|judge|attorney|closing), entity name, etc.
    """
    sorted_attorneys = list(attorney_rankings.items())
    best = sorted_attorneys[0] if len(sorted_attorneys) > 0 else None
    second = sorted_attorneys[1] if len(sorted_attorneys) > 1 else None
    third = sorted_attorneys[2] if len(sorted_attorneys) > 2 else None

    # Judge most associated with top attorney
    top_judge_for_best = None
    top_score_for_best = -1
    if best:
        for judge, ranking in judge_attorney_rankings.items():
            score = ranking.get(best[0], 0)
            if score > top_score_for_best:
                top_score_for_best = score
                top_judge_for_best = judge

    segments: list[dict] = []

    # Intro
    intro = (
        f"What we uncovered may surprise you. "
        f"Behind the scenes, {len(district_judges)} federal district judges "
        f"hold the power to decide your fate."
    )
    segments.append({
        "text": intro,
        "type": "intro",
        "data": {"judge_count": len(district_judges)},
    })

    # Judge highlight
    if top_judge_for_best and best:
        judge_text = (
            f"But here's what the data revealed — a striking connection "
            f"between Judge {top_judge_for_best} and {best[0]}. "
            f"A score of {top_score_for_best}. That number tells a story."
        )
        segments.append({
            "text": judge_text,
            "type": "judge_highlight",
            "data": {
                "judge": top_judge_for_best,
                "attorney": best[0],
                "score": top_score_for_best,
                "judge_image": judge_images.get(top_judge_for_best),
            },
        })

    # Top attorney
    if best:
        best_text = (
            f"When every probability was accounted for, "
            f"one name rose to the top — {best[0]}, scoring {best[1]:.0f}."
        )
        segments.append({
            "text": best_text,
            "type": "attorney",
            "data": {"attorney": best[0], "score": round(best[1]), "rank": 1},
        })

    # Second
    if second:
        second_text = (
            f"Close behind, {second[0]} at {second[1]:.0f}."
        )
        segments.append({
            "text": second_text,
            "type": "attorney",
            "data": {"attorney": second[0], "score": round(second[1]), "rank": 2},
        })

    # Third
    if third:
        third_text = (
            f"And a third contender — {third[0]} at {third[1]:.0f}."
        )
        segments.append({
            "text": third_text,
            "type": "attorney",
            "data": {"attorney": third[0], "score": round(third[1]), "rank": 3},
        })

    # Closing
    closing = "The evidence is clear. Now it's your move."
    segments.append({"text": closing, "type": "closing", "data": {}})

    full_script = " ".join(seg["text"] for seg in segments)
    return full_script, segments


def _tts_with_timestamps(text: str) -> dict:
    """
    Call ElevenLabs TTS with timestamps. Returns dict with:
        audio_base64: str (mp3)
        characters: list of {character, character_start_times_seconds, character_end_times_seconds}
    """
    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json",
    }
    payload = {
        "text": text,
        "model_id": "eleven_turbo_v2_5",
        "voice_settings": {
            "stability": 0.55,
            "similarity_boost": 0.75,
            "style": 0.45,
            "use_speaker_boost": True,
        },
    }
    resp = requests.post(
        ELEVENLABS_TTS_TIMESTAMPS_URL, headers=headers, json=payload, timeout=120
    )
    resp.raise_for_status()
    return resp.json()


def _compute_segment_timings(
    full_script: str,
    segments: list[dict],
    tts_data: dict,
) -> list[dict]:
    """
    Map segments to start/end times using character-level timestamps from ElevenLabs.
    """
    char_starts = tts_data.get("alignment", {}).get("character_start_times_seconds", [])
    char_ends = tts_data.get("alignment", {}).get("character_end_times_seconds", [])

    offset = 0
    timed_segments = []

    for seg in segments:
        seg_text = seg["text"]
        # Find where this segment starts in the full script
        idx = full_script.find(seg_text, offset)
        if idx == -1:
            idx = offset  # fallback

        start_char = idx
        end_char = idx + len(seg_text) - 1

        # Get timing from character arrays
        start_time = char_starts[start_char] if start_char < len(char_starts) else 0
        end_time = char_ends[end_char] if end_char < len(char_ends) else start_time + 3

        timed_segments.append({
            **seg,
            "start_time": start_time,
            "end_time": end_time,
        })

        offset = idx + len(seg_text)

    return timed_segments


# ── Routes ───────────────────────────────────────────────────────────────────


@app.route("/")
def index():
    return send_from_directory("frontend", "index.html")


@app.route("/api/transcribe", methods=["POST"])
def transcribe():
    """Accept an audio file upload and return transcription."""
    if "audio" not in request.files:
        return jsonify({"error": "No audio file"}), 400

    audio_file = request.files["audio"]
    headers = {"xi-api-key": ELEVENLABS_API_KEY}

    # Determine mime type from filename
    filename = audio_file.filename or "recording.webm"
    if filename.endswith(".webm"):
        mime = "audio/webm"
    elif filename.endswith(".wav"):
        mime = "audio/wav"
    else:
        mime = "audio/webm"

    files = {"file": (filename, audio_file, mime)}
    data = {"model_id": "scribe_v1"}

    resp = requests.post(ELEVENLABS_STT_URL, headers=headers, files=files, data=data, timeout=30)
    resp.raise_for_status()
    text = resp.json().get("text", "").strip()
    return jsonify({"text": text})


@app.route("/api/pipeline", methods=["POST"])
def pipeline():
    """
    Run the full pipeline from a text prompt.
    Returns: narration audio (base64), timed visual segments, all data.
    """
    body = request.get_json()
    prompt = body.get("prompt", "")

    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400

    # Run pipeline
    state_code = resolve_state_code(prompt)
    state_judges = get_state_judges(state_code)
    district_judges = get_district_judges(
        location_description=prompt, judges_list=state_judges
    )
    judge_pmf = get_judge_pmf(district_judges)
    judge_attorney_rankings = get_judge_attorney_rankings(district_judges)
    attorney_rankings = get_weighted_attorney_rankings(judge_pmf, judge_attorney_rankings)

    # Get judge images
    judge_images = _get_judge_image_urls(district_judges) if district_judges else {}

    # Build narration + visual events
    full_script, segments = _build_narration_and_events(
        prompt, district_judges, judge_pmf,
        judge_attorney_rankings, attorney_rankings, judge_images,
    )

    # Get TTS with timestamps
    tts_data = _tts_with_timestamps(full_script)
    audio_b64 = tts_data.get("audio_base64", "")

    # Compute timings for each visual segment
    timed_segments = _compute_segment_timings(full_script, segments, tts_data)

    return jsonify({
        "script": full_script,
        "audio_base64": audio_b64,
        "segments": timed_segments,
        "judge_images": judge_images,
        "district_judges": district_judges,
        "attorney_rankings": dict(list(attorney_rankings.items())[:5]),
    })


if __name__ == "__main__":
    app.run(debug=True, port=5001)
