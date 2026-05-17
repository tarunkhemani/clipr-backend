import cv2
import subprocess
import os
import numpy as np


# ─────────────────────────────────────────────
# FACE DETECTION USING OPENCV
# Uses Haar Cascade — free, no API needed
# ─────────────────────────────────────────────

# OpenCV's built-in face detector
FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def extract_frame(video_path: str, timestamp: float, output_path: str) -> bool:
    """Extract a single frame from video at given timestamp."""
    command = [
        "ffmpeg",
        "-ss", str(timestamp),
        "-i", video_path,
        "-vframes", "1",
        "-q:v", "2",
        "-y",
        output_path,
    ]
    result = subprocess.run(command, capture_output=True, timeout=30)
    return os.path.exists(output_path) and os.path.getsize(output_path) > 100


def detect_face_position(frame_path: str) -> dict | None:
    """
    Detects the largest face in a frame.
    Returns { cx, cy, x, y, w, h } — center x/y and bounding box.
    Returns None if no face found.
    """
    img = cv2.imread(frame_path)
    if img is None:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = img.shape[:2]

    faces = FACE_CASCADE.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
    )

    if len(faces) == 0:
        return None

    # Pick the largest face (most likely the main subject)
    largest = max(faces, key=lambda f: f[2] * f[3])
    x, y, w, h = largest

    return {
        "x":  int(x),
        "y":  int(y),
        "w":  int(w),
        "h":  int(h),
        "cx": int(x + w // 2),   # Face center x
        "cy": int(y + h // 2),   # Face center y
        "frame_width":  width,
        "frame_height": height,
    }


def analyze_face_position(video_path: str, clip_duration: float) -> dict:
    """
    Samples multiple frames across the clip to find the average face position.
    Sampling multiple frames handles cases where the speaker moves.
    Returns crop position info.
    """
    # Sample at 4 points: 20%, 40%, 60%, 80% through the clip
    sample_points = [0.2, 0.4, 0.6, 0.8]
    face_cx_values = []

    temp_dir = os.path.dirname(video_path)

    for i, pct in enumerate(sample_points):
        timestamp = clip_duration * pct
        frame_path = os.path.join(temp_dir, f"_sample_frame_{i}.jpg")

        try:
            if extract_frame(video_path, timestamp, frame_path):
                face = detect_face_position(frame_path)
                if face:
                    face_cx_values.append({
                        "cx": face["cx"],
                        "frame_width": face["frame_width"],
                        "frame_height": face["frame_height"],
                    })
        except Exception as e:
            print(f"Frame sampling error at {pct}: {e}")
        finally:
            if os.path.exists(frame_path):
                os.remove(frame_path)

    if not face_cx_values:
        print("No face detected in any sample frame — using center crop")
        return {"mode": "center"}

    # Average face center x across all samples
    avg_cx = sum(f["cx"] for f in face_cx_values) / len(face_cx_values)
    frame_width = face_cx_values[0]["frame_width"]
    frame_height = face_cx_values[0]["frame_height"]

    print(f"Face detected — average center x: {avg_cx:.0f} / {frame_width}")

    return {
        "mode":         "face",
        "avg_cx":       avg_cx,
        "frame_width":  frame_width,
        "frame_height": frame_height,
    }


def calculate_crop_x(face_data: dict, input_width: int, output_width: int) -> int:
    """
    Calculates the x offset for cropping to keep the face centered.
    Clamps to valid range so crop never goes out of frame.
    """
    if face_data["mode"] == "center":
        # Fallback: center crop
        return (input_width - output_width) // 2

    # Scale face position to the actual video dimensions
    scale = input_width / face_data["frame_width"]
    face_x_scaled = face_data["avg_cx"] * scale

    # Center the crop window on the face
    crop_x = int(face_x_scaled - output_width // 2)

    # Clamp so crop stays within frame
    crop_x = max(0, min(crop_x, input_width - output_width))

    return crop_x


def get_video_dimensions(video_path: str) -> tuple[int, int]:
    """Returns (width, height) of video using ffprobe."""
    try:
        result = subprocess.run([
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "csv=p=0",
            video_path,
        ], capture_output=True, text=True, timeout=30)

        parts = result.stdout.strip().split(",")
        return int(parts[0]), int(parts[1])
    except Exception:
        return 1920, 1080  # Safe fallback


def smart_crop_to_vertical(
    video_path: str,
    output_path: str,
    start_time: float,
    duration: float,
) -> str:
    """
    Main function — extracts a clip with smart face-centered vertical crop.
    Falls back to center crop if no face detected.

    Steps:
    1. Extract raw clip first (no crop, fast)
    2. Analyze face position in the raw clip
    3. Apply smart crop based on face position
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Step 1: Extract raw clip without crop for analysis
    raw_path = output_path.replace(".mp4", "_raw_nocrop.mp4")

    extract_command = [
        "ffmpeg",
        "-ss", str(start_time),
        "-i", video_path,
        "-t", str(duration),
        "-c:v", "libx264",
        "-preset", "ultrafast",   # Fast — this is just for analysis
        "-crf", "28",
         "-pix_fmt", "yuv420p", 
        "-c:a", "aac",
        "-avoid_negative_ts", "make_zero",
        "-y",
        raw_path,
    ]

    result = subprocess.run(extract_command, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(f"Raw clip extraction failed:\n{result.stderr[-300:]}")

    # Step 2: Analyze face position
    face_data = analyze_face_position(raw_path, duration)

    # Step 3: Get video dimensions
    input_width, input_height = get_video_dimensions(raw_path)

    # Target: 9:16 vertical (1080x1920)
    target_width  = 1080
    target_height = 1920

    # Calculate scale factor to make height = 1920
    scale_factor = target_height / input_height
    scaled_width = int(input_width * scale_factor)

    # Calculate face-aware crop x on the scaled video
    # Scale face position to match scaled dimensions
    if face_data["mode"] == "face":
        face_data_scaled = {
            "mode":        "face",
            "avg_cx":      face_data["avg_cx"] * (scaled_width / face_data["frame_width"]),
            "frame_width": scaled_width,
        }
    else:
        face_data_scaled = {"mode": "center"}

    crop_x = calculate_crop_x(face_data_scaled, scaled_width, target_width)

    print(f"Smart crop — mode: {face_data['mode']}, crop_x: {crop_x}")

    # Step 4: Apply scale + smart crop in one FFmpeg pass
    vf_filter = f"scale={scaled_width}:{target_height},crop={target_width}:{target_height}:{crop_x}:0"

    crop_command = [
        "ffmpeg",
        "-i", raw_path,
        "-vf", vf_filter,
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "128k",
        "-movflags", "+faststart",
        "-avoid_negative_ts", "make_zero",
        "-y",
        output_path,
    ]

    result = subprocess.run(crop_command, capture_output=True, text=True, timeout=300)

    # Clean up raw clip
    if os.path.exists(raw_path):
        os.remove(raw_path)

    if result.returncode != 0:
        raise RuntimeError(f"Smart crop failed:\n{result.stderr[-300:]}")

    return output_path