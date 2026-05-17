import os
import asyncio
import edge_tts

VOICES = {
    "christopher": {"id": "en-US-ChristopherNeural", "label": "Christopher (Deep Male, US)", "gender": "male"},
    "guy":         {"id": "en-US-GuyNeural",         "label": "Guy (Casual Male, US)",      "gender": "male"},
    "ryan":        {"id": "en-GB-RyanNeural",         "label": "Ryan (British Male)",         "gender": "male"},
    "jenny":       {"id": "en-US-JennyNeural",        "label": "Jenny (Female, US)",          "gender": "female"},
    "aria":        {"id": "en-US-AriaNeural",         "label": "Aria (Expressive Female, US)","gender": "female"},
    "sonia":       {"id": "en-GB-SoniaNeural",        "label": "Sonia (British Female)",      "gender": "female"},
}

DEFAULT_VOICE = "christopher"


async def _generate_audio_async(text: str, voice_id: str, output_path: str):
    """Generate audio only — no word timestamps from EdgeTTS."""
    communicate = edge_tts.Communicate(text, voice_id, rate="+0%", volume="+0%")
    await communicate.save(output_path)


def generate_voice(text: str, voice_key: str, output_path: str) -> list:
    """
    Generates MP3 audio using EdgeTTS.
    Then transcribes it with Groq Whisper to get accurate word timestamps.
    Returns list of { word, start, end } dicts.
    """
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    voice_id = VOICES.get(voice_key, VOICES[DEFAULT_VOICE])["id"]

    # Step 1: Generate audio
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(_generate_audio_async(text, voice_id, output_path))
    finally:
        loop.close()

    if not os.path.exists(output_path) or os.path.getsize(output_path) < 100:
        raise RuntimeError(f"Voice generation failed — audio file empty: {output_path}")

    # Step 2: Transcribe with Groq Whisper to get word timestamps
    word_timestamps = _transcribe_for_timestamps(output_path)
    print(f"Got {len(word_timestamps)} word timestamps from Whisper")

    return word_timestamps


def _transcribe_for_timestamps(audio_path: str) -> list:
    """
    Transcribes the generated audio using Groq Whisper.
    Returns word-level timestamps.
    """
    import os
    from groq import Groq
    from dotenv import load_dotenv
    load_dotenv()

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    try:
        with open(audio_path, "rb") as f:
            transcription = client.audio.transcriptions.create(
                file=(os.path.basename(audio_path), f.read()),
                model="whisper-large-v3-turbo",
                response_format="verbose_json",
                timestamp_granularities=["word"],
            )

        word_timestamps = []

        # Groq returns words directly when timestamp_granularities=["word"]
        if hasattr(transcription, "words") and transcription.words:
            for w in transcription.words:
                if isinstance(w, dict):
                    word_timestamps.append({
                        "word":  w.get("word", "").strip(),
                        "start": round(float(w.get("start", 0)), 3),
                        "end":   round(float(w.get("end", 0)), 3),
                    })
                else:
                    word_timestamps.append({
                        "word":  w.word.strip(),
                        "start": round(float(w.start), 3),
                        "end":   round(float(w.end), 3),
                    })

        # Fallback: if words not available, use segments
        if not word_timestamps and hasattr(transcription, "segments") and transcription.segments:
            for seg in transcription.segments:
                if isinstance(seg, dict):
                    word_timestamps.append({
                        "word":  seg.get("text", "").strip(),
                        "start": round(float(seg.get("start", 0)), 3),
                        "end":   round(float(seg.get("end", 0)), 3),
                    })
                else:
                    word_timestamps.append({
                        "word":  seg.text.strip(),
                        "start": round(float(seg.start), 3),
                        "end":   round(float(seg.end), 3),
                    })

        return word_timestamps

    except Exception as e:
        print(f"Whisper transcription failed: {e}")
        return []


def get_audio_duration(word_timestamps: list) -> float:
    if not word_timestamps:
        return 0.0
    return word_timestamps[-1]["end"] + 0.5


def get_voices_list() -> list:
    return [
        {"key": key, "label": v["label"], "gender": v["gender"]}
        for key, v in VOICES.items()
    ]
# import os
# import asyncio
# import edge_tts

# # ─────────────────────────────────────────────────────────────
# # AVAILABLE VOICES
# # All free via Microsoft EdgeTTS — no API key needed
# # ─────────────────────────────────────────────────────────────

# VOICES = {
#     "christopher": {
#         "id": "en-US-ChristopherNeural",
#         "label": "Christopher (Deep Male, US)",
#         "gender": "male",
#     },
#     "guy": {
#         "id": "en-US-GuyNeural",
#         "label": "Guy (Casual Male, US)",
#         "gender": "male",
#     },
#     "ryan": {
#         "id": "en-GB-RyanNeural",
#         "label": "Ryan (British Male)",
#         "gender": "male",
#     },
#     "jenny": {
#         "id": "en-US-JennyNeural",
#         "label": "Jenny (Female, US)",
#         "gender": "female",
#     },
#     "aria": {
#         "id": "en-US-AriaNeural",
#         "label": "Aria (Expressive Female, US)",
#         "gender": "female",
#     },
#     "sonia": {
#         "id": "en-GB-SoniaNeural",
#         "label": "Sonia (British Female)",
#         "gender": "female",
#     },
# }

# DEFAULT_VOICE = "christopher"


# async def _generate_voice_async(text: str, voice_id: str, output_path: str) -> list:
#     """
#     Async core of voice generation.
#     Streams audio to file and collects word-level timestamps.
#     Returns list of { word, start, end } dicts.
#     """
#     os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

#     communicate = edge_tts.Communicate(text, voice_id, rate="+0%", volume="+0%")

#     word_timestamps = []

#     with open(output_path, "wb") as audio_file:
#         async for chunk in communicate.stream():
#             if chunk["type"] == "audio":
#                 audio_file.write(chunk["data"])
#             elif chunk["type"] == "WordBoundary":
#                 # EdgeTTS returns offset/duration in 100-nanosecond units
#                 start_sec = chunk["offset"] / 10_000_000
#                 duration_sec = chunk["duration"] / 10_000_000
#                 word_timestamps.append({
#                     "word": chunk["text"],
#                     "start": round(start_sec, 3),
#                     "end": round(start_sec + duration_sec, 3),
#                 })

#     return word_timestamps


# def generate_voice(text: str, voice_key: str, output_path: str) -> list:
#     """
#     Synchronous wrapper for voice generation.
#     Generates MP3 audio file and returns word timestamps.
#     """
#     voice_id = VOICES.get(voice_key, VOICES[DEFAULT_VOICE])["id"]

#     # Run async function in new event loop (safe for use in FastAPI background tasks)
#     loop = asyncio.new_event_loop()
#     asyncio.set_event_loop(loop)
#     try:
#         word_timestamps = loop.run_until_complete(
#             _generate_voice_async(text, voice_id, output_path)
#         )
#     finally:
#         loop.close()

#     if not os.path.exists(output_path) or os.path.getsize(output_path) < 100:
#         raise RuntimeError(f"Voice generation failed — audio file empty or missing: {output_path}")

#     return word_timestamps


# def get_audio_duration(word_timestamps: list) -> float:
#     """Get total audio duration from word timestamps."""
#     if not word_timestamps:
#         return 0.0
#     return word_timestamps[-1]["end"] + 0.5  # Add small buffer at end


# def get_voices_list() -> list:
#     """Return voice options for the frontend."""
#     return [
#         {"key": key, "label": v["label"], "gender": v["gender"]}
#         for key, v in VOICES.items()
#     ]