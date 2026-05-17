import os
import json
import re
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

VIRAL_PROMPT = """You are an expert viral content analyst for YouTube Shorts, Instagram Reels, and TikTok.

You will receive a transcript of a YouTube video with timestamps. Your job is to identify the TOP 5 most viral-worthy clips.

A viral clip has ONE OR MORE of these qualities:
- A strong HOOK in the first 3 seconds (surprising, bold, or controversial statement)
- A shocking or counterintuitive fact
- A strong opinion or hot take
- An emotional or funny moment
- A clear beginning, middle, and end (a mini story)
- Something people would screenshot or share
- A quotable, memorable line

RULES:
- Each clip must be between 30 seconds and 90 seconds long
- Clips should NOT overlap each other
- Pick the BEST moments, not just the first ones
- Each clip must make sense on its own without watching the full video

Transcript:
{transcript}

Respond ONLY with a valid JSON array. No explanation, no markdown, no backticks. Just raw JSON.

Format:
[
  {{
    "clip_number": 1,
    "start_time": 45.2,
    "end_time": 98.7,
    "title": "Short punchy title for this clip (max 8 words)",
    "hook": "The opening line that makes this viral",
    "why_viral": "One sentence explaining why this will perform well",
    "viral_score": 8
  }}
]

viral_score is from 1-10. Only include clips scoring 6 or above.
"""


def find_viral_clips(segments: list, video_duration: int) -> list:
    transcript_lines = []
    for seg in segments:
        timestamp = f"[{seg['start']:.1f}s]"
        transcript_lines.append(f"{timestamp} {seg['text']}")
    transcript = "\n".join(transcript_lines)

    if len(transcript) > 50000:
        transcript = transcript[:50000] + "\n... (transcript truncated)"

    prompt = VIRAL_PROMPT.format(transcript=transcript)

    response = client.models.generate_content(
        model="gemini-3.1-flash-lite-preview",
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.3,
            max_output_tokens=2000,
        ),
    )

    raw = response.text.strip()
    raw = re.sub(r"^```json\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    raw = raw.strip()

    clips = json.loads(raw)

    validated = []
    for clip in clips:
        start = float(clip.get("start_time", 0))
        end = float(clip.get("end_time", 0))
        duration = end - start

        if start < 0 or end <= start:
            continue
        if duration < 20 or duration > 120:
            continue
        if end > video_duration + 5:
            continue

        validated.append({
            "clip_number": clip.get("clip_number", len(validated) + 1),
            "start_time": start,
            "end_time": end,
            "duration": round(duration, 1),
            "title": clip.get("title", f"Clip {len(validated) + 1}"),
            "hook": clip.get("hook", ""),
            "why_viral": clip.get("why_viral", ""),
            "viral_score": clip.get("viral_score", 7),
        })

    validated.sort(key=lambda x: x["viral_score"], reverse=True)
    return validated[:5]
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