import os
import subprocess
from groq import Groq
from dotenv import load_dotenv

load_dotenv()


def extract_audio(video_path: str) -> str:
    audio_path = video_path.replace(".mp4", ".mp3")
    command = [
        "ffmpeg", "-i", video_path,
        "-vn",
        "-acodec", "libmp3lame",
        "-ar", "16000",
        "-ac", "1",
        "-q:a", "5",
        "-y",
        audio_path
    ]
    subprocess.run(command, check=True, capture_output=True)
    return audio_path


def transcribe_audio(video_path: str) -> list:
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    audio_path = extract_audio(video_path)

    try:
        with open(audio_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                file=(os.path.basename(audio_path), audio_file.read()),
                model="whisper-large-v3-turbo",
                response_format="verbose_json",
                timestamp_granularities=["segment"],
            )

        segments = []
        for seg in transcription.segments:
            # Handle both dict and object style responses from Groq
            if isinstance(seg, dict):
                segments.append({
                    "start": round(float(seg.get("start", 0)), 2),
                    "end": round(float(seg.get("end", 0)), 2),
                    "text": seg.get("text", "").strip(),
                })
            else:
                segments.append({
                    "start": round(float(seg.start), 2),
                    "end": round(float(seg.end), 2),
                    "text": seg.text.strip(),
                })

        # segments = []
        # for seg in transcription.segments:
        #     segments.append({
        #         "start": round(seg.start, 2),
        #         "end": round(seg.end, 2),
        #         "text": seg.text.strip(),
        #     })

        return segments

    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)