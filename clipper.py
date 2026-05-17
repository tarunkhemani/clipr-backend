import os
import subprocess
import shutil


# ─────────────────────────────────────────────
# SAFE VIDEO PROBING (using ffprobe, NOT moviepy reader)
# Avoids the "Duration: N/A" crash from Bug 1 in ShortGPT
# ─────────────────────────────────────────────

def get_video_duration(video_path: str) -> float:
    """
    Uses ffprobe directly to get video duration.
    Never uses MoviePy's reader — avoids the FFmpeg metadata parsing bug.
    Falls back to 60.0 seconds if probing fails.
    """
    try:
        command = [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path,
        ]
        result = subprocess.run(command, capture_output=True, text=True, timeout=30)
        duration = float(result.stdout.strip())
        return duration
    except Exception:
        print("Warning: ffprobe could not read duration. Using 60.0s fallback.")
        return 60.0


def validate_file(file_path: str, min_size_bytes: int = 44) -> bool:
    """
    Validates a file exists and is not a ghost/empty file.
    min_size_bytes=44 matches the ShortGPT patch (blank WAV header size).
    Protects against Windows pipe interruption bug.
    """
    try:
        return os.path.exists(file_path) and os.path.getsize(file_path) > min_size_bytes
    except Exception:
        return False


def extract_clip(
    video_path: str,
    start_time: float,
    end_time: float,
    output_path: str,
    vertical: bool = True,
) -> str:
    """
    Extracts a clip from a video between start_time and end_time.
    Crops to 9:16 vertical format if vertical=True.
    Uses subprocess directly — never MoviePy reader.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Safety clamp duration always positive
    duration = max(0.1, end_time - start_time)

    if vertical:
        vf_filter = "scale=-2:1920,crop=1080:1920"
    else:
        vf_filter = "scale=1920:-2"

    command = [
        "ffmpeg",
        "-ss", str(start_time),
        "-i", video_path,
        "-t", str(duration),
        "-vf", vf_filter,
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-c:a", "aac",
        "-b:a", "128k",
        "-movflags", "+faststart",
        "-avoid_negative_ts", "make_zero",
        "-y",
        output_path,
    ]

    result = subprocess.run(command, capture_output=True, text=True, timeout=300)

    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg clip extraction failed:\n{result.stderr[-500:]}")

    if not validate_file(output_path, min_size_bytes=1000):
        raise RuntimeError(f"Clip extraction produced an empty or corrupt file: {output_path}")

    return output_path
def split_segments_into_chunks(segments: list, words_per_chunk: int = 4) -> list:
    """
    Breaks sentence-level segments into smaller word chunks.
    Distributes timing evenly so captions change more frequently.
    """
    chunks = []

    for seg in segments:
        words = seg["text"].strip().split()
        if not words:
            continue

        seg_duration = seg["end"] - seg["start"]
        time_per_word = seg_duration / len(words)

        # Split into groups of words_per_chunk
        i = 0
        while i < len(words):
            chunk_words = words[i:i + words_per_chunk]
            chunk_start = seg["start"] + (i * time_per_word)
            chunk_end   = seg["start"] + ((i + len(chunk_words)) * time_per_word)

            chunks.append({
                "start": round(chunk_start, 3),
                "end":   round(chunk_end, 3),
                "text":  " ".join(chunk_words),
            })
            i += words_per_chunk

    return chunks
def add_captions_to_clip(
    clip_path: str,
    segments: list[dict],
    start_time: float,
    end_time: float,
    output_path: str,
) -> str:
    # Filter segments that fall within this clip
    clip_segments = []
    for seg in segments:
        if seg["end"] > start_time and seg["start"] < end_time:
            clip_segments.append({
                "start": round(max(0.0, seg["start"] - start_time), 3),
                "end":   round(min(end_time - start_time, seg["end"] - start_time), 3),
                "text":  seg["text"],
            })

    if not clip_segments:
        shutil.copy2(clip_path, output_path)
        return output_path

    # Break into smaller chunks so captions change frequently
    chunks = split_segments_into_chunks(clip_segments, words_per_chunk=4)

    drawtext_filters = []
    for seg in chunks:        # ← now iterating chunks, not clip_segments
        if seg["end"] <= seg["start"]:
            continue

        text = (
            seg["text"]
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

        # Already 4 words max so no need to wrap into multiple lines
        drawtext_filters.append(
            f"drawtext="
            f"text='{text}':"
            f"fontsize=38:"
            f"fontcolor=white:"
            f"bordercolor=black:"
            f"borderw=3:"
            f"x=(w-text_w)/2:"
            f"y=h-text_h-160:"
            f"fix_bounds=1:"
            f"enable='between(t,{seg['start']},{seg['end']})'"
        )

    if not drawtext_filters:
        shutil.copy2(clip_path, output_path)
        return output_path

    command = [
        "ffmpeg",
        "-i", clip_path,
        "-vf", ",".join(drawtext_filters),
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-c:a", "copy",
        "-y",
        output_path,
    ]

    result = subprocess.run(command, capture_output=True, text=True, timeout=300)

    if result.returncode != 0 or not validate_file(output_path, min_size_bytes=1000):
        print(f"Warning: Caption burn failed. Returning clip without captions.")
        shutil.copy2(clip_path, output_path)

    return output_path
# def add_captions_to_clip(
#     clip_path: str,
#     segments: list[dict],
#     start_time: float,
#     end_time: float,
#     output_path: str,
# ) -> str:
#     clip_segments = []
#     for seg in segments:
#         if seg["end"] > start_time and seg["start"] < end_time:
#             clip_segments.append({
#                 "start": round(max(0.0, seg["start"] - start_time), 3),
#                 "end":   round(min(end_time - start_time, seg["end"] - start_time), 3),
#                 "text":  seg["text"],
#             })

#     if not clip_segments:
#         shutil.copy2(clip_path, output_path)
#         return output_path

#     drawtext_filters = []
#     for seg in clip_segments:
#         if seg["end"] <= seg["start"]:
#             continue

#         text = (
#             seg["text"]
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

#         # Max 5 words per line, 2 lines max — fits 1080px width at fontsize 44
#         words = text.split()
#         lines = []
#         line  = []
#         for word in words:
#             line.append(word)
#             if len(line) >= 4:
#                 lines.append(" ".join(line))
#                 line = []
#         if line:
#             lines.append(" ".join(line))
#         # Cap at 2 lines so text never overflows vertically
#         lines = lines[:2]
#         wrapped = "\\n".join(lines)

#         drawtext_filters.append(
#             f"drawtext="
#             f"text='{wrapped}':"
#             f"fontsize=38:"
#             f"fontcolor=white:"
#             f"bordercolor=black:"
#             f"borderw=3:"
#             f"x=(w-text_w)/2:"
#             f"y=h-text_h-160:"
#             f"fix_bounds=1:"
#             f"enable='between(t,{seg['start']},{seg['end']})'"
#         )

#     if not drawtext_filters:
#         shutil.copy2(clip_path, output_path)
#         return output_path

#     command = [
#         "ffmpeg",
#         "-i", clip_path,
#         "-vf", ",".join(drawtext_filters),
#         "-c:v", "libx264",
#         "-preset", "fast",
#         "-crf", "23",
#         "-c:a", "copy",
#         "-y",
#         output_path,
#     ]

#     result = subprocess.run(command, capture_output=True, text=True, timeout=300)

#     if result.returncode != 0 or not validate_file(output_path, min_size_bytes=1000):
#         print(f"Warning: Caption burn failed. Returning clip without captions.")
#         shutil.copy2(clip_path, output_path)

#     return output_path
# def add_captions_to_clip(
#     clip_path: str,
#     segments: list[dict],
#     start_time: float,
#     end_time: float,
#     output_path: str,
# ) -> str:
#     """
#     Burns captions into a clip using FFmpeg drawtext.
#     Falls back to returning the uncaptioned clip if anything goes wrong.
#     """
#     clip_segments = []
#     for seg in segments:
#         seg_start = seg["start"]
#         seg_end = seg["end"]

#         if seg_end > start_time and seg_start < end_time:
#             clip_segments.append({
#                 "start": round(max(0.0, seg_start - start_time), 3),
#                 "end": round(min(end_time - start_time, seg_end - start_time), 3),
#                 "text": seg["text"],
#             })

#     if not clip_segments:
#         shutil.copy2(clip_path, output_path)
#         return output_path

#     drawtext_filters = []
#     for seg in clip_segments:
#         if seg["end"] <= seg["start"]:
#             continue

#         text = (
#             seg["text"]
#             .replace("\\", "\\\\")
#             .replace("'", "\u2019")
#             .replace(":", "\\:")
#             .replace(",", "\\,")
#             .replace("[", "\\[")
#             .replace("]", "\\]")
#         )

#         words = text.split()
#         lines = []
#         current_line = []
#         for word in words:
#             current_line.append(word)
#             if len(current_line) >= 6:
#                 lines.append(" ".join(current_line))
#                 current_line = []
#         if current_line:
#             lines.append(" ".join(current_line))
#         wrapped = "\\n".join(lines)

#         drawtext_filters.append(
#             f"drawtext="
#             f"text='{wrapped}':"
#             f"fontsize=52:"
#             f"fontcolor=white:"
#             f"bordercolor=black:"
#             f"borderw=3:"
#             f"x=(w-text_w)/2:"
#             f"y=h-text_h-150:"
#             f"enable='between(t,{seg['start']},{seg['end']})'"
#         )

#     if not drawtext_filters:
#         shutil.copy2(clip_path, output_path)
#         return output_path

#     vf = ",".join(drawtext_filters)

#     command = [
#         "ffmpeg",
#         "-i", clip_path,
#         "-vf", vf,
#         "-c:v", "libx264",
#         "-preset", "fast",
#         "-crf", "23",
#         "-c:a", "copy",
#         "-y",
#         output_path,
#     ]

#     result = subprocess.run(command, capture_output=True, text=True, timeout=300)

#     if result.returncode != 0 or not validate_file(output_path, min_size_bytes=1000):
#         print(f"Warning: Caption burn failed. Returning clip without captions.")
#         shutil.copy2(clip_path, output_path)

#     return output_path


def process_all_clips(
    video_path: str,
    clips_data: list[dict],
    segments: list[dict],
    output_dir: str,
    unique_id: str,
    add_captions: bool = True,
) -> list[dict]:
    """
    Processes all viral clips: extracts, crops to 9:16, burns captions.
    Validates every file before and after processing (Windows pipe bug protection).
    """
    os.makedirs(output_dir, exist_ok=True)

    if not validate_file(video_path, min_size_bytes=10000):
        raise RuntimeError(f"Source video is missing or corrupt: {video_path}")

    processed_clips = []

    for clip in clips_data:
        clip_num = clip["clip_number"]
        start = max(0.0, float(clip["start_time"]))
        end = max(start + 1.0, float(clip["end_time"]))

        raw_clip_path = os.path.join(output_dir, f"{unique_id}_clip{clip_num}_raw.mp4")
        final_clip_path = os.path.join(output_dir, f"{unique_id}_clip{clip_num}.mp4")

        try:
            print(f"Processing clip {clip_num}: {start:.1f}s to {end:.1f}s")

            extract_clip(video_path, start, end, raw_clip_path, vertical=True)

            if not validate_file(raw_clip_path, min_size_bytes=1000):
                raise RuntimeError("Raw clip extraction produced empty file.")

            if add_captions:
                add_captions_to_clip(raw_clip_path, segments, start, end, final_clip_path)
                if os.path.exists(raw_clip_path) and raw_clip_path != final_clip_path:
                    os.remove(raw_clip_path)
            else:
                os.rename(raw_clip_path, final_clip_path)

            if not validate_file(final_clip_path, min_size_bytes=1000):
                raise RuntimeError("Final clip is empty after processing.")

            print(f"Clip {clip_num} done.")

            processed_clips.append({
                **clip,
                "output_path": final_clip_path,
                "filename": os.path.basename(final_clip_path),
                "status": "success",
            })

        except Exception as e:
            print(f"Clip {clip_num} failed: {e}")

            for path in [raw_clip_path, final_clip_path]:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except Exception:
                        pass

            processed_clips.append({
                **clip,
                "output_path": None,
                "filename": None,
                "status": "error",
                "error": str(e),
            })

    return processed_clips