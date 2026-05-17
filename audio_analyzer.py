import subprocess
import os
import wave
import struct
import math


def extract_audio_wav(video_path: str, output_path: str) -> str:
    """Extract audio as 16kHz mono WAV for analysis."""
    command = [
        "ffmpeg", "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        "-y", output_path
    ]
    subprocess.run(command, capture_output=True, timeout=120)
    return output_path


def compute_energy_timeline(wav_path: str) -> list:
    """
    Computes RMS energy for every second of audio.
    Returns list of { second, energy, normalized }
    """
    try:
        with wave.open(wav_path, "rb") as wav:
            framerate  = wav.getframerate()
            n_frames   = wav.getnframes()
            total_secs = int(n_frames / framerate)

            energy_per_second = []

            for second in range(total_secs):
                wav.setpos(second * framerate)
                frames = wav.readframes(framerate)

                if len(frames) < 2:
                    energy_per_second.append({"second": second, "energy": 0})
                    continue

                # Parse 16-bit signed samples
                sample_count = len(frames) // 2
                samples = struct.unpack(f"{sample_count}h", frames[:sample_count * 2])

                # RMS energy
                rms = math.sqrt(sum(s * s for s in samples) / len(samples))
                energy_per_second.append({"second": second, "energy": rms})

        # Normalize 0-1
        max_e = max((e["energy"] for e in energy_per_second), default=1)
        max_e = max(max_e, 1)
        for e in energy_per_second:
            e["normalized"] = round(e["energy"] / max_e, 3)

        return energy_per_second

    except Exception as ex:
        print(f"Energy timeline error: {ex}")
        return []


def score_window_energy(energy_timeline: list, start: int, end: int) -> float:
    """Average normalized energy for a time window."""
    window = [e["normalized"] for e in energy_timeline if start <= e["second"] < end]
    return round(sum(window) / len(window), 3) if window else 0.0


def compute_speech_rate_windows(segments: list, window_size: int = 10) -> list:
    """
    Computes words-per-second in sliding windows.
    High WPS = excited, fast-talking = potentially viral.
    """
    if not segments:
        return []

    total_duration = segments[-1]["end"]
    windows = []

    step = window_size // 2
    for start in range(0, int(total_duration) - window_size + 1, step):
        end = start + window_size
        word_count = 0
        for seg in segments:
            if seg["start"] >= start and seg["end"] <= end:
                word_count += len(seg["text"].split())
        windows.append({
            "start":      start,
            "end":        end,
            "wps":        round(word_count / window_size, 2),
            "word_count": word_count,
        })

    # Normalize WPS
    max_wps = max((w["wps"] for w in windows), default=1)
    max_wps = max(max_wps, 0.1)
    for w in windows:
        w["normalized_wps"] = round(w["wps"] / max_wps, 3)

    return windows


def score_window_speech_rate(speech_windows: list, start: int, end: int) -> float:
    """Average normalized speech rate for a time window."""
    matching = [w["normalized_wps"] for w in speech_windows
                if w["start"] >= start and w["end"] <= end]
    return round(sum(matching) / len(matching), 3) if matching else 0.0


# ─────────────────────────────────────────────
# KEYWORD SCORING
# ─────────────────────────────────────────────

# Weighted viral trigger words/patterns
VIRAL_KEYWORDS = {
    # Very high weight — these almost always signal viral moments
    "nobody talks about":   10,
    "the truth is":         10,
    "they don't want you":  10,
    "secret":               8,
    "i was wrong":          9,
    "biggest mistake":      9,
    "changed my life":      8,
    "can't believe":        8,
    "blew my mind":         8,
    "what nobody tells":    9,
    "honest truth":         8,

    # High weight
    "actually":             4,
    "honestly":             5,
    "the problem is":       5,
    "here's the thing":     5,
    "let me tell you":      4,
    "real reason":          6,
    "dark side":            6,
    "shocking":             6,
    "unbelievable":         6,
    "insane":               5,
    "you need to know":     6,
    "most people don't":    7,
    "stop doing":           5,
    "don't make this":      5,

    # Medium weight — stats and numbers signal credibility
    "percent":              3,
    "million":              3,
    "billion":              4,
    "study shows":          4,
    "research":             3,
    "scientists":           3,
    "proven":               3,

    # Story markers
    "and then":             2,
    "suddenly":             4,
    "i never":              3,
    "that's when":          4,
    "everything changed":   5,
    "turning point":        4,

    # Question hooks
    "did you know":         5,
    "what if i told you":   6,
    "have you ever":        4,
    "why does":             3,
    "how is it possible":   4,
}


def score_window_keywords(segments: list, start: float, end: float) -> tuple[float, list]:
    """
    Scores a time window based on viral keyword presence.
    Returns (score, matched_keywords)
    """
    # Collect all text in this window
    window_text = " ".join(
        seg["text"].lower() for seg in segments
        if seg["start"] >= start and seg["end"] <= end
    )

    total_score   = 0
    matched       = []

    for keyword, weight in VIRAL_KEYWORDS.items():
        if keyword in window_text:
            total_score += weight
            matched.append(keyword)

    # Bonus: detect numbers/statistics
    import re
    number_matches = re.findall(r'\b\d+(?:\.\d+)?(?:%|x|times|million|billion|thousand)?\b', window_text)
    if number_matches:
        total_score += min(len(number_matches) * 2, 8)

    # Normalize to 0-10
    normalized = min(total_score / 2, 10)
    return round(normalized, 2), matched


def detect_pauses(segments: list) -> list:
    """
    Detects significant pauses (>0.8s) between segments.
    A pause before a statement signals importance.
    Returns list of { pause_after_second, duration }
    """
    pauses = []
    for i in range(1, len(segments)):
        gap = segments[i]["start"] - segments[i - 1]["end"]
        if gap >= 0.8:
            pauses.append({
                "at_second":  segments[i]["start"],
                "duration":   round(gap, 2),
                "next_text":  segments[i]["text"][:80],
            })
    return pauses


def pause_score_for_window(pauses: list, start: float, end: float) -> float:
    """Bonus score for pauses within a window (signals important statements)."""
    window_pauses = [p for p in pauses if start <= p["at_second"] <= end]
    if not window_pauses:
        return 0.0
    # Longer pauses = higher score, max 3 bonus points
    return min(sum(min(p["duration"], 2) for p in window_pauses), 3.0)


def analyze_audio(video_path: str, segments: list) -> dict:
    """
    Full audio analysis pipeline.
    Returns energy timeline, speech rate windows, and pauses.
    """
    audio_path = video_path.replace(".mp4", "_analysis.wav")

    try:
        extract_audio_wav(video_path, audio_path)

        if not os.path.exists(audio_path) or os.path.getsize(audio_path) < 100:
            return {"energy_timeline": [], "speech_windows": [], "pauses": [], "available": False}

        energy_timeline  = compute_energy_timeline(audio_path)
        speech_windows   = compute_speech_rate_windows(segments, window_size=10)
        pauses           = detect_pauses(segments)

        print(f"Audio analysis: {len(energy_timeline)}s analyzed, {len(pauses)} pauses detected")

        return {
            "energy_timeline": energy_timeline,
            "speech_windows":  speech_windows,
            "pauses":          pauses,
            "available":       True,
        }

    except Exception as e:
        print(f"Audio analysis failed: {e}")
        return {"energy_timeline": [], "speech_windows": [], "pauses": [], "available": False}

    finally:
        if os.path.exists(audio_path):
            try:
                os.remove(audio_path)
            except Exception:
                pass