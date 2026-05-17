import os
import json
import re
from google import genai
from google.genai import types
from dotenv import load_dotenv
from audio_analyzer import (
    analyze_audio,
    score_window_energy,
    score_window_speech_rate,
    score_window_keywords,
    pause_score_for_window,
)

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


# ─────────────────────────────────────────────
# CANDIDATE WINDOW SCORING
# ─────────────────────────────────────────────

def score_all_windows(
    segments:       list,
    audio_data:     dict,
    video_duration: int,
    window_size:    int = 60,
    step:           int = 15,
) -> list:
    """
    Slides a window across the video and scores every possible clip.
    Combines 4 signals: audio energy, speech rate, keywords, pauses.
    Returns top candidates sorted by combined score.
    """
    energy_timeline = audio_data.get("energy_timeline", [])
    speech_windows  = audio_data.get("speech_windows",  [])
    pauses          = audio_data.get("pauses",           [])
    audio_available = audio_data.get("available",        False)

    candidates = []

    for start in range(0, max(1, int(video_duration) - window_size + 1), step):
        end = min(start + window_size, video_duration)
        if end - start < 25:
            continue

        # Get transcript text for this window
        window_segments = [s for s in segments if s["start"] >= start and s["end"] <= end]
        if not window_segments:
            continue

        window_text = " ".join(s["text"] for s in window_segments)
        if len(window_text.split()) < 20:
            continue

        # ── Score each signal ──────────────────────────────────
        energy_score  = score_window_energy(energy_timeline, start, end) * 10 if audio_available else 5.0
        speech_score  = score_window_speech_rate(speech_windows, start, end) * 10 if audio_available else 5.0
        keyword_score, matched_keywords = score_window_keywords(segments, start, end)
        pause_bonus   = pause_score_for_window(pauses, start, end)

        # ── Combined score (weighted) ──────────────────────────
        # Keywords are most predictive, followed by energy, then speech rate, then pauses
        combined = (
            keyword_score * 0.40 +
            energy_score  * 0.25 +
            speech_score  * 0.20 +
            pause_bonus   * 0.15
        )

        candidates.append({
            "start":           start,
            "end":             end,
            "duration":        end - start,
            "text":            window_text,
            "combined_score":  round(combined, 3),
            "energy_score":    round(energy_score, 2),
            "speech_score":    round(speech_score, 2),
            "keyword_score":   round(keyword_score, 2),
            "pause_bonus":     round(pause_bonus, 2),
            "keywords_found":  matched_keywords,
        })

    # Sort by combined score, return top 12 candidates for Gemini to pick from
    candidates.sort(key=lambda x: x["combined_score"], reverse=True)
    return candidates[:12]


# ─────────────────────────────────────────────
# GEMINI FINAL SELECTION PROMPT
# ─────────────────────────────────────────────

SELECTION_PROMPT = """You are a world-class viral content strategist who has helped channels grow from 0 to millions of subscribers.

I have pre-analyzed a YouTube video and identified {count} high-potential clip candidates using audio energy, speech rate, keyword analysis, and pause detection. Each candidate already has a combined signal score.

Your job: From these candidates, select the BEST 5 clips that will actually go viral on YouTube Shorts, Instagram Reels, and TikTok.

WHAT MAKES A CLIP ACTUALLY GO VIRAL:
1. Hook in first 3 seconds — viewer must be unable to scroll past
2. Emotional trigger — surprise, anger, inspiration, curiosity, humor, or shock
3. Self-contained — makes complete sense without watching the full video  
4. Rewatchable — people watch it twice or share it
5. Comment-worthy — provokes strong reactions or opinions
6. Relatable OR aspirational — viewer sees themselves in it OR wants to be that
7. Quotable ending — last line people want to screenshot

AVOID selecting clips that:
- Start mid-sentence with no context
- Are just technical explanations with no emotion
- Require background knowledge to understand
- End weakly or trail off

CANDIDATES:
{candidates}

Respond ONLY with a valid JSON array. No explanation. No markdown. Raw JSON only.

Format:
[
  {{
    "clip_number": 1,
    "start_time": 45.0,
    "end_time": 98.0,
    "title": "Punchy 6-word title",
    "hook": "Exact opening line that stops the scroll",
    "why_viral": "One specific sentence — what emotion does this trigger and why will people share it",
    "viral_score": 9,
    "content_type": "story|fact|opinion|revelation|humor|motivation"
  }}
]

viral_score: 1-10. Only include clips you genuinely believe score 6+.
Select clips with diverse content_types when possible — variety keeps viewers watching more.
"""


def format_candidates_for_prompt(candidates: list) -> str:
    """Format top candidates into readable text for Gemini."""
    lines = []
    for i, c in enumerate(candidates, 1):
        keywords_str = ", ".join(c["keywords_found"]) if c["keywords_found"] else "none"
        lines.append(
            f"CANDIDATE {i}:\n"
            f"  Time: {c['start']:.0f}s - {c['end']:.0f}s ({c['duration']:.0f}s)\n"
            f"  Combined Score: {c['combined_score']:.2f} | "
            f"Energy: {c['energy_score']:.1f}/10 | "
            f"Speech Rate: {c['speech_score']:.1f}/10 | "
            f"Keywords: {c['keyword_score']:.1f}/10\n"
            f"  Viral Keywords Found: {keywords_str}\n"
            f"  Transcript: {c['text'][:300]}{'...' if len(c['text']) > 300 else ''}\n"
        )
    return "\n".join(lines)


def find_viral_clips(segments: list, video_duration: int, video_path: str = None) -> list:
    """
    Main function — multi-signal viral clip detection.
    
    Pipeline:
    1. Analyze audio (energy, speech rate, pauses)
    2. Score every possible clip window with combined signals
    3. Send top candidates to Gemini for final intelligent selection
    4. Return validated clips with rich metadata
    """

    # Step 1: Audio analysis
    print("Running audio analysis...")
    if video_path and os.path.exists(video_path):
        audio_data = analyze_audio(video_path, segments)
    else:
        audio_data = {"energy_timeline": [], "speech_windows": [], "pauses": [], "available": False}

    # Step 2: Score all windows
    print("Scoring candidate windows...")
    candidates = score_all_windows(segments, audio_data, video_duration)
    print(f"Top {len(candidates)} candidates identified for Gemini review")

    if not candidates:
        raise ValueError("No viable clip candidates found in this video.")

    # Step 3: Gemini final selection
    candidates_text = format_candidates_for_prompt(candidates)
    prompt = SELECTION_PROMPT.format(
        count=len(candidates),
        candidates=candidates_text,
    )

    import time
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model="gemini-3.1-flash-lite-preview",
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.2,   # Low temperature = consistent, reliable picks
                    max_output_tokens=2000,
                ),
            )
            break
        except Exception as e:
            if "503" in str(e) or "UNAVAILABLE" in str(e):
                if attempt < 2:
                    print(f"Gemini overloaded, retrying in 5s...")
                    time.sleep(5)
                    continue
            raise

    raw = response.text.strip()
    raw = re.sub(r"^```json\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    raw = raw.strip()

    clips = json.loads(raw)

    # Step 4: Validate and enrich clips
    validated = []
    for clip in clips:
        start    = float(clip.get("start_time", 0))
        end      = float(clip.get("end_time", 0))
        duration = end - start

        if start < 0 or end <= start:
            continue
        if duration < 20 or duration > 120:
            continue
        if end > video_duration + 5:
            continue

        # Find matching candidate to get signal scores
        matching = next(
            (c for c in candidates if abs(c["start"] - start) < 10),
            None
        )

        validated.append({
            "clip_number":    clip.get("clip_number", len(validated) + 1),
            "start_time":     start,
            "end_time":       end,
            "duration":       round(duration, 1),
            "title":          clip.get("title", f"Clip {len(validated) + 1}"),
            "hook":           clip.get("hook", ""),
            "why_viral":      clip.get("why_viral", ""),
            "viral_score":    clip.get("viral_score", 7),
            "content_type":   clip.get("content_type", "story"),
            "energy_score":   matching["energy_score"]  if matching else 5.0,
            "keyword_score":  matching["keyword_score"] if matching else 5.0,
            "keywords_found": matching["keywords_found"] if matching else [],
        })

    validated.sort(key=lambda x: x["viral_score"], reverse=True)
    print(f"Final: {len(validated)} viral clips selected")
    return validated[:5]
# import os
# import json
# import re
# from google import genai
# from google.genai import types
# from dotenv import load_dotenv

# load_dotenv()

# client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# VIRAL_PROMPT = """You are an expert viral content analyst for YouTube Shorts, Instagram Reels, and TikTok.

# You will receive a transcript of a YouTube video with timestamps. Your job is to identify the TOP 5 most viral-worthy clips.

# A viral clip has ONE OR MORE of these qualities:
# - A strong HOOK in the first 3 seconds (surprising, bold, or controversial statement)
# - A shocking or counterintuitive fact
# - A strong opinion or hot take
# - An emotional or funny moment
# - A clear beginning, middle, and end (a mini story)
# - Something people would screenshot or share
# - A quotable, memorable line

# RULES:
# - Each clip must be between 30 seconds and 90 seconds long
# - Clips should NOT overlap each other
# - Pick the BEST moments, not just the first ones
# - Each clip must make sense on its own without watching the full video

# Transcript:
# {transcript}

# Respond ONLY with a valid JSON array. No explanation, no markdown, no backticks. Just raw JSON.

# Format:
# [
#   {{
#     "clip_number": 1,
#     "start_time": 45.2,
#     "end_time": 98.7,
#     "title": "Short punchy title for this clip (max 8 words)",
#     "hook": "The opening line that makes this viral",
#     "why_viral": "One sentence explaining why this will perform well",
#     "viral_score": 8
#   }}
# ]

# viral_score is from 1-10. Only include clips scoring 6 or above.
# """


# def find_viral_clips(segments: list, video_duration: int) -> list:
#     transcript_lines = []
#     for seg in segments:
#         timestamp = f"[{seg['start']:.1f}s]"
#         transcript_lines.append(f"{timestamp} {seg['text']}")
#     transcript = "\n".join(transcript_lines)

#     if len(transcript) > 50000:
#         transcript = transcript[:50000] + "\n... (transcript truncated)"

#     prompt = VIRAL_PROMPT.format(transcript=transcript)

#     response = client.models.generate_content(
#         model="gemini-3.1-flash-lite-preview",
#         contents=prompt,
#         config=types.GenerateContentConfig(
#             temperature=0.3,
#             max_output_tokens=2000,
#         ),
#     )

#     raw = response.text.strip()
#     raw = re.sub(r"^```json\s*", "", raw)
#     raw = re.sub(r"\s*```$", "", raw)
#     raw = raw.strip()

#     clips = json.loads(raw)

#     validated = []
#     for clip in clips:
#         start = float(clip.get("start_time", 0))
#         end = float(clip.get("end_time", 0))
#         duration = end - start

#         if start < 0 or end <= start:
#             continue
#         if duration < 20 or duration > 120:
#             continue
#         if end > video_duration + 5:
#             continue

#         validated.append({
#             "clip_number": clip.get("clip_number", len(validated) + 1),
#             "start_time": start,
#             "end_time": end,
#             "duration": round(duration, 1),
#             "title": clip.get("title", f"Clip {len(validated) + 1}"),
#             "hook": clip.get("hook", ""),
#             "why_viral": clip.get("why_viral", ""),
#             "viral_score": clip.get("viral_score", 7),
#         })

#     validated.sort(key=lambda x: x["viral_score"], reverse=True)
#     return validated[:5]
# import os
# import json
# import re
# import google.generativeai as genai
# from dotenv import load_dotenv

# load_dotenv()

# genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


# VIRAL_PROMPT = """You are an expert viral content analyst for YouTube Shorts, Instagram Reels, and TikTok.

# You will receive a transcript of a YouTube video with timestamps. Your job is to identify the TOP 5 most viral-worthy clips.

# A viral clip has ONE OR MORE of these qualities:
# - A strong HOOK in the first 3 seconds (surprising, bold, or controversial statement)
# - A shocking or counterintuitive fact
# - A strong opinion or hot take
# - An emotional or funny moment
# - A clear beginning, middle, and end (a mini story)
# - Something people would screenshot or share
# - A quotable, memorable line

# RULES:
# - Each clip must be between 30 seconds and 90 seconds long
# - Clips should NOT overlap each other
# - Pick the BEST moments, not just the first ones
# - Each clip must make sense on its own without watching the full video

# Transcript:
# {transcript}

# Respond ONLY with a valid JSON array. No explanation, no markdown, no backticks. Just raw JSON.

# Format:
# [
#   {{
#     "clip_number": 1,
#     "start_time": 45.2,
#     "end_time": 98.7,
#     "title": "Short punchy title for this clip (max 8 words)",
#     "hook": "The opening line that makes this viral",
#     "why_viral": "One sentence explaining why this will perform well",
#     "viral_score": 8
#   }}
# ]

# viral_score is from 1-10. Only include clips scoring 6 or above.
# """


# def find_viral_clips(segments: list, video_duration: int) -> list:
#     transcript_lines = []
#     for seg in segments:
#         timestamp = f"[{seg['start']:.1f}s]"
#         transcript_lines.append(f"{timestamp} {seg['text']}")
#     transcript = "\n".join(transcript_lines)

#     if len(transcript) > 50000:
#         transcript = transcript[:50000] + "\n... (transcript truncated)"

#     model = genai.GenerativeModel("gemini-1.5-flash")

#     prompt = VIRAL_PROMPT.format(transcript=transcript)

#     response = model.generate_content(
#         prompt,
#         generation_config=genai.types.GenerationConfig(
#             temperature=0.3,
#             max_output_tokens=2000,
#         ),
#     )

#     raw = response.text.strip()
#     raw = re.sub(r"^```json\s*", "", raw)
#     raw = re.sub(r"\s*```$", "", raw)
#     raw = raw.strip()

#     clips = json.loads(raw)

#     validated = []
#     for clip in clips:
#         start = float(clip.get("start_time", 0))
#         end = float(clip.get("end_time", 0))
#         duration = end - start

#         if start < 0 or end <= start:
#             continue
#         if duration < 20 or duration > 120:
#             continue
#         if end > video_duration + 5:
#             continue

#         validated.append({
#             "clip_number": clip.get("clip_number", len(validated) + 1),
#             "start_time": start,
#             "end_time": end,
#             "duration": round(duration, 1),
#             "title": clip.get("title", f"Clip {len(validated) + 1}"),
#             "hook": clip.get("hook", ""),
#             "why_viral": clip.get("why_viral", ""),
#             "viral_score": clip.get("viral_score", 7),
#         })

#     validated.sort(key=lambda x: x["viral_score"], reverse=True)
#     return validated[:5]