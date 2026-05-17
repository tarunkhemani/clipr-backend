import os
import subprocess


def validate_file(path: str, min_bytes: int = 100) -> bool:
    try:
        return os.path.exists(path) and os.path.getsize(path) > min_bytes
    except Exception:
        return False


def get_video_duration(video_path: str) -> float:
    try:
        result = subprocess.run([
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path
        ], capture_output=True, text=True, timeout=30)
        return float(result.stdout.strip())
    except Exception:
        return 60.0


def prepare_background_video(bg_video_path: str, target_duration: float, output_path: str) -> str:
    bg_duration = get_video_duration(bg_video_path)
    loops_needed = max(1, int(target_duration / bg_duration) + 2)

    command = [
        "ffmpeg",
        "-stream_loop", str(loops_needed),
        "-i", bg_video_path,
        "-t", str(target_duration + 1),
        "-vf", "scale=-2:1920,crop=1080:1920",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-an",
        "-pix_fmt", "yuv420p",
        "-avoid_negative_ts", "make_zero",
        "-y",
        output_path,
    ]

    result = subprocess.run(command, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        raise RuntimeError(f"Background preparation failed:\n{result.stderr[-400:]}")
    if not validate_file(output_path):
        raise RuntimeError("Background video loop produced empty file.")
    return output_path

def build_drawtext_filter(caption_chunks: list) -> str:
    print(f"Building captions from {len(caption_chunks)} chunks")
    
    if not caption_chunks:
        return "null"

    filters = []

    for chunk in caption_chunks:
        start = chunk["start"]
        end   = chunk["end"]
        text  = chunk["text"]

        # Clean special characters
        text = (
            text
            .replace("\\", "")
            .replace("'",  "")
            .replace('"',  "")
            .replace(":",  " ")
            .replace(",",  " ")
            .replace("[",  "")
            .replace("]",  "")
            .replace("%",  "")
        )

        if not text.strip():
            continue

        # Max 5 words per line, 2 lines max
        words = text.split()
        lines = []
        line  = []
        for word in words:
            line.append(word)
            if len(line) >= 5:
                lines.append(" ".join(line))
                line = []
        if line:
            lines.append(" ".join(line))
        lines   = lines[:2]
        wrapped = "\\n".join(lines)

        # Bold white text + black border + semi-transparent box behind text
        filters.append(
            f"drawtext="
            f"text='{wrapped}':"
            f"fontsize=54:"
            f"fontcolor=white:"
            f"bordercolor=black:"
            f"borderw=4:"
            f"x=(w-text_w)/2:"
            f"y=h-text_h-160:"
            f"boxcolor=black@0.4:"
            f"box=1:"
            f"boxborderw=12:"
            f"enable='between(t,{start:.3f},{end:.3f})'"
        )

    return ",".join(filters) if filters else "null"
# def build_drawtext_filter(caption_chunks: list) -> str:
#     print(f"Building captions from {len(caption_chunks)} chunks")  # ADD THIS
#     """
#     Builds FFmpeg drawtext filter chain from caption chunks.
#     Each chunk: { start, end, text }
#     Avoids SRT/libass entirely — works reliably on Windows.
#     """
#     filters = []

#     for chunk in caption_chunks:
#         start = chunk["start"]
#         end   = chunk["end"]
#         text  = chunk["text"]

#         # Escape characters that break FFmpeg filter syntax
#         text = (
#             text
#             .replace("\\", "")
#             .replace("'",  "")
#             .replace('"',  "")
#             .replace(":",  " ")
#             .replace(",",  " ")
#             .replace("[",  "")
#             .replace("]",  "")
#             .replace("%",  "")
#         )

#         if not text.strip():
#             continue

#         # Word wrap — max 6 words per line
#         words = text.split()
#         lines = []
#         line  = []
#         for word in words:
#             line.append(word)
#             if len(line) >= 6:
#                 lines.append(" ".join(line))
#                 line = []
#         if line:
#             lines.append(" ".join(line))
#         wrapped = "\\n".join(lines)

#         filters.append(
#             f"drawtext="
#             f"text='{wrapped}':"
#             f"fontsize=54:"
#             f"fontcolor=white:"
#             f"bordercolor=black:"
#             f"borderw=3:"
#             f"x=(w-text_w)/2:"
#             f"y=h-text_h-140:"
#             f"enable='between(t,{start:.3f},{end:.3f})'"
#         )

#     return ",".join(filters) if filters else "null"


def compose_short(
    bg_video_path: str,
    audio_path: str,
    caption_chunks: list,
    output_path: str,
    audio_duration: float,
) -> str:
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    output_abs = os.path.abspath(output_path)
    audio_abs  = os.path.abspath(audio_path)
    bg_abs     = os.path.abspath(bg_video_path)

    # Always use ffprobe for actual audio duration — never trust word timestamps
    actual_duration = get_video_duration(audio_abs)
    print(f"Actual audio duration from ffprobe: {actual_duration:.2f}s")
    
    if actual_duration < 2.0:
        raise RuntimeError(f"Audio file too short ({actual_duration:.2f}s) — voice generation may have failed.")

    # Step 1: Prepare looped background using REAL duration
    looped_bg = output_abs.replace(".mp4", "_bg_prepared.mp4")
    prepare_background_video(bg_abs, actual_duration, looped_bg)

    # Step 2: Build drawtext filter inline
    vf_filter = build_drawtext_filter(caption_chunks)

    # Step 3: Compose final short
    command = [
        "ffmpeg",
        "-i", looped_bg,
        "-i", audio_abs,
        "-map", "0:v",
        "-map", "1:a",
        "-vf", vf_filter,
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "128k",
        "-shortest",
        "-movflags", "+faststart",
        "-avoid_negative_ts", "make_zero",
        "-y",
        output_abs,
    ]

    result = subprocess.run(command, capture_output=True, text=True, timeout=600)

    # Cleanup
    if os.path.exists(looped_bg):
        try:
            os.remove(looped_bg)
        except Exception:
            pass

    if result.returncode != 0:
        raise RuntimeError(f"Final composition failed:\n{result.stderr[-500:]}")

    if not validate_file(output_abs, min_bytes=10000):
        raise RuntimeError("Final short is empty or corrupt.")

    return output_abs