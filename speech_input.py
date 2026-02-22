import io
import os
import wave
import time

import numpy as np
import requests
import sounddevice as sd
from dotenv import load_dotenv

load_dotenv()

ELEVENLABS_API_KEY = os.environ["ELEVENLABS_API_KEY"]
ELEVENLABS_STT_URL = "https://api.elevenlabs.io/v1/speech-to-text"

SAMPLE_RATE = 16000  # 16 kHz — standard for speech
CHANNELS = 1
LAST_RECORDING_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "last_recording.wav")

# Silence detection settings
SILENCE_DURATION = 1.5     # seconds of silence before auto-stop
CHUNK_DURATION = 0.1       # seconds per analysis chunk
MAX_DURATION = 30          # hard cap in seconds
CALIBRATION_SECONDS = 0.5  # how long to sample ambient noise
THRESHOLD_MULTIPLIER = 3   # speech must be this many times louder than ambient


def _calibrate_threshold() -> float:
    """Record a short ambient sample and return an RMS-based speech threshold."""
    chunk_samples = int(CALIBRATION_SECONDS * SAMPLE_RATE)
    print("🔇  Calibrating mic (stay quiet)...")
    ambient = sd.rec(chunk_samples, samplerate=SAMPLE_RATE, channels=CHANNELS, dtype="int16")
    sd.wait()
    ambient_rms = np.sqrt(np.mean(ambient.astype(np.float32) ** 2))
    threshold = max(ambient_rms * THRESHOLD_MULTIPLIER, 50)  # floor of 50 to avoid zero
    print(f"    ambient RMS={ambient_rms:.0f}, threshold={threshold:.0f}")
    return threshold


def record_until_silence(max_duration: int = MAX_DURATION) -> np.ndarray:
    """
    Record from the mic and automatically stop after detecting silence.

    First calibrates against ambient noise, then listens in small chunks.
    Once speech is detected, keeps recording until SILENCE_DURATION seconds
    of consecutive silence, then stops immediately.

    Returns:
        Numpy array of int16 PCM samples.
    """
    threshold = _calibrate_threshold()

    chunk_samples = int(CHUNK_DURATION * SAMPLE_RATE)
    silence_chunks_needed = int(SILENCE_DURATION / CHUNK_DURATION)
    max_chunks = int(max_duration / CHUNK_DURATION)

    print("🎙  Listening... speak now (I'll stop when you're done)")

    chunks: list[np.ndarray] = []
    silence_count = 0
    speech_started = False

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE, channels=CHANNELS, dtype="int16",
        blocksize=chunk_samples,
    )
    stream.start()

    try:
        for _ in range(max_chunks):
            data, _ = stream.read(chunk_samples)
            chunks.append(data.copy())

            rms = np.sqrt(np.mean(data.astype(np.float32) ** 2))

            if rms > threshold:
                speech_started = True
                silence_count = 0
            elif speech_started:
                silence_count += 1
                if silence_count >= silence_chunks_needed:
                    break
    finally:
        stream.stop()
        stream.close()

    print("✅  Got it!")
    return np.concatenate(chunks)


def save_wav(audio: np.ndarray, path: str) -> None:
    """Save a numpy int16 audio array to a WAV file."""
    with wave.open(path, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio.tobytes())


def _transcribe_wav(wav_path: str) -> str:
    """
    Transcribe a WAV file using ElevenLabs Speech-to-Text API.
    """
    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
    }

    with open(wav_path, "rb") as f:
        files = {
            "file": ("recording.wav", f, "audio/wav"),
        }
        data = {
            "model_id": "scribe_v1",
        }

        try:
            response = requests.post(
                ELEVENLABS_STT_URL, headers=headers, files=files, data=data, timeout=30
            )
            response.raise_for_status()
            return response.json().get("text", "").strip()

        except requests.ConnectionError:
            print("Error: Cannot connect to ElevenLabs API.")
            return ""
        except requests.Timeout:
            print("Error: ElevenLabs STT request timed out.")
            return ""
        except Exception as e:
            print(f"Error: {e}")
            return ""


def capture_prompt() -> str:
    """
    Record speech from the mic (auto-stops on silence) and transcribe
    using ElevenLabs STT.

    Returns:
        Transcribed text string.
    """
    audio = record_until_silence()
    save_wav(audio, LAST_RECORDING_PATH)
    return _transcribe_wav(LAST_RECORDING_PATH)


def transcribe_from_file(path: str = LAST_RECORDING_PATH) -> str:
    """
    Transcribe a previously saved WAV file using ElevenLabs STT.

    Args:
        path: Path to a WAV file. Defaults to the last recording.

    Returns:
        Transcribed text string.
    """
    return _transcribe_wav(path)


if __name__ == "__main__":
    text = capture_prompt()
    print(f"Transcribed: {text}")
