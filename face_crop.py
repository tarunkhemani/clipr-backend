import cv2
import subprocess
import os
import statistics


# ─────────────────────────────────────────────
# MULTIPLE CASCADE CLASSIFIERS
# Frontal default + alt2 catches more angles
# Profile catches side-facing speakers
# Upper body as last resort fallback
# ─────────────────────────────────────────────

CASCADE_FRONTAL_DEFAULT = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
CASCADE_FRONTAL_ALT2 = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml"
)
CASCADE_PROFILE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_profileface.xml"
)
CASCADE_UPPERBODY = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_upperbody.xml"
)


def get_video_dimensions(video_path: str) -> tuple:
    """Returns (width, height) using ffprobe."""
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
        return 1920, 1080


def extract_frame(video_path: str, timestamp: float, output_path: str) -> bool:
    """Extract a single frame at given timestamp."""
    command = [
        "ffmpeg", "-ss", str(timestamp),
        "-i", video_path,
        "-vframes", "1",
        "-q:v", "2",
        "-y", output_path,
    ]
    result = subprocess.run(command, capture_output=True, timeout=30)
    return os.path.exists(output_path) and os.path.getsize(output_path) > 100


def detect_subject_center(frame_path: str) -> dict | None:
    """
    Detects face or upper body in frame using multiple cascades.
    Returns center x as fraction of frame width (0.0 to 1.0).
    This makes it resolution-independent.

    Priority: frontal face → alt2 face → profile face → upper body → None
    """
    img = cv2.imread(frame_path)
    if img is None:
        return None

    height, width = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Enhance contrast to help detection in different lighting
    gray = cv2.equalizeHist(gray)

    def find_largest(cascade, scale=1.1, neighbors=4, min_size_ratio=0.05):
        min_dim = int(min(width, height) * min_size_ratio)
        detections = cascade.detectMultiScale(
            gray,
            scaleFactor=scale,
            minNeighbors=neighbors,
            minSize=(min_dim, min_dim),
        )
        if len(detections) == 0:
            return None
        largest = max(detections, key=lambda d: d[2] * d[3])
        x, y, w, h = largest
        return {
            "cx_fraction": (x + w / 2) / width,
            "cy_fraction": (y + h / 2) / height,
            "size_fraction": (w * h) / (width * height),
        }

    # Try each cascade in priority order
    result = (
        find_largest(CASCADE_FRONTAL_ALT2, scale=1.1, neighbors=5) or
        find_largest(CASCADE_FRONTAL_DEFAULT, scale=1.1, neighbors=4) or
        find_largest(CASCADE_PROFILE, scale=1.1, neighbors=4) or
        # Flip image for other profile direction
        _detect_flipped_profile(gray, width, height) or
        find_largest(CASCADE_UPPERBODY, scale=1.05, neighbors=3, min_size_ratio=0.1)
    )

    return result


def _detect_flipped_profile(gray, width, height):
    """Detect profile faces looking the other direction by flipping the image."""
    flipped = cv2.flip(gray, 1)
    detections = CASCADE_PROFILE.detectMultiScale(
        flipped, scaleFactor=1.1, minNeighbors=4,
        minSize=(int(min(width, height) * 0.05),) * 2,
    )
    if len(detections) == 0:
        return None
    largest = max(detections, key=lambda d: d[2] * d[3])
    x, y, w, h = largest
    # Un-flip the x coordinate
    cx_flipped = (x + w / 2) / width
    return {
        "cx_fraction":   1.0 - cx_flipped,
        "cy_fraction":   (y + h / 2) / height,
        "size_fraction": (w * h) / (width * height),
    }


def remove_outliers_median(values: list) -> float:
    """
    Returns median of values after removing top/bottom 15% outliers.
    Much more robust than mean for noisy face detection.
    """
    if not values:
        return 0.5
    if len(values) <= 2:
        return statistics.median(values)

    values_sorted = sorted(values)
    trim = max(1, int(len(values_sorted) * 0.15))
    trimmed = values_sorted[trim:-trim] if trim > 0 else values_sorted

    return statistics.median(trimmed)


def analyze_face_positions(video_path: str, clip_duration: float) -> dict:
    """
    Samples frames every 2 seconds across the clip.
    Returns face position data with confidence score.
    """
    temp_dir = os.path.dirname(os.path.abspath(video_path))

    # Sample every 2 seconds — much more reliable than 4 fixed points
    sample_interval = 2.0
    sample_times = []
    t = 1.0
    while t < clip_duration - 0.5:
        sample_times.append(t)
        t += sample_interval

    if not sample_times:
        sample_times = [clip_duration * 0.5]

    print(f"Face detection: sampling {len(sample_times)} frames...")

    # Collect detections with timestamps
    detections = []

    for i, timestamp in enumerate(sample_times):
        frame_path = os.path.join(temp_dir, f"_face_frame_{i}.jpg")
        try:
            if extract_frame(video_path, timestamp, frame_path):
                result = detect_subject_center(frame_path)
                if result:
                    detections.append({
                        "time":        timestamp,
                        "cx_fraction": result["cx_fraction"],
                        "size":        result["size_fraction"],
                    })
        except Exception as e:
            print(f"Frame {i} detection error: {e}")
        finally:
            if os.path.exists(frame_path):
                try:
                    os.remove(frame_path)
                except Exception:
                    pass

    detection_rate = len(detections) / len(sample_times) if sample_times else 0
    print(f"Face detection rate: {len(detections)}/{len(sample_times)} ({detection_rate:.0%})")

    if not detections or detection_rate < 0.25:
        print("Low detection rate — using center crop")
        return {"mode": "center", "detection_rate": detection_rate}

    # Get all cx fractions
    cx_values = [d["cx_fraction"] for d in detections]

    # Robust median position
    median_cx = remove_outliers_median(cx_values)

    # Check how much the face moves across the clip
    cx_range = max(cx_values) - min(cx_values)

    print(f"Face position: median={median_cx:.2f}, range={cx_range:.2f}, detections={len(detections)}")

    # If face moves a lot, use dynamic crop
    if cx_range > 0.20 and len(detections) >= 4:
        return {
            "mode":           "dynamic",
            "detections":     detections,
            "median_cx":      median_cx,
            "cx_range":       cx_range,
            "detection_rate": detection_rate,
        }

    return {
        "mode":           "static",
        "median_cx":      median_cx,
        "detection_rate": detection_rate,
    }


def cx_to_crop_x(cx_fraction: float, scaled_width: int, target_width: int) -> int:
    """
    Convert face center fraction to FFmpeg crop x offset.
    Adds safety margin toward center so face doesn't get clipped on movement.
    """
    face_x = cx_fraction * scaled_width

    # Bias toward center by 15% — safety margin for movement
    center_x    = scaled_width / 2
    biased_x    = face_x * 0.85 + center_x * 0.15

    crop_x = int(biased_x - target_width / 2)
    crop_x = max(0, min(crop_x, scaled_width - target_width))
    return crop_x


def build_dynamic_crop_expression(detections: list, scaled_width: int,
                                   target_width: int, scale_factor: float) -> str:
    """
    Builds a smooth FFmpeg crop x expression that follows face movement over time.
    Uses linear interpolation between detected positions.
    """
    # Convert to crop_x positions
    keyframes = []
    for d in sorted(detections, key=lambda x: x["time"]):
        cx_scaled = d["cx_fraction"] * scaled_width
        # Bias toward center (safety margin)
        center    = scaled_width / 2
        cx_biased = cx_scaled * 0.85 + center * 0.15
        crop_x    = max(0, min(int(cx_biased - target_width / 2),
                               scaled_width - target_width))
        keyframes.append({"t": d["time"], "x": crop_x})

    if len(keyframes) == 1:
        return str(keyframes[0]["x"])

    # Build piecewise linear interpolation expression
    # FFmpeg: if(lt(t, T1), lerp(x0, x1, (t-t0)/(t1-t0)), ...)
    def lerp_segment(t0, x0, t1, x1):
        dt = t1 - t0
        if dt <= 0:
            return str(x0)
        return f"({x0}+({x1}-{x0})*(t-{t0})/{dt})"

    # Build nested if expression
    expr = str(keyframes[-1]["x"])  # Default to last position
    for i in range(len(keyframes) - 2, -1, -1):
        k0 = keyframes[i]
        k1 = keyframes[i + 1]
        segment = lerp_segment(k0["t"], k0["x"], k1["t"], k1["x"])
        expr = f"if(between(t\\,{k0['t']}\\,{k1['t']})\\,{segment}\\,{expr})"

    return expr


def smart_crop_to_vertical(
    video_path: str,
    output_path: str,
    start_time: float,
    duration: float,
) -> str:
    """
    Main function — smart face-aware vertical crop.

    Modes:
    - static:  face barely moves → single optimal crop position
    - dynamic: face moves a lot  → smooth following crop expression
    - center:  no face detected  → safe center crop
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # Step 1: Extract raw clip for analysis
    raw_path = output_path.replace(".mp4", "_raw_nocrop.mp4")
    extract_command = [
        "ffmpeg",
        "-ss", str(start_time),
        "-i", video_path,
        "-t", str(duration),
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-crf", "28",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-avoid_negative_ts", "make_zero",
        "-y", raw_path,
    ]
    result = subprocess.run(extract_command, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(f"Raw extraction failed:\n{result.stderr[-300:]}")

    # Step 2: Analyze face positions
    face_data = analyze_face_positions(raw_path, duration)

    # Step 3: Get dimensions and calculate scale
    input_width, input_height = get_video_dimensions(raw_path)
    target_width  = 1080
    target_height = 1920
    scale_factor  = target_height / input_height
    scaled_width  = int(input_width * scale_factor)
    scaled_width  = scaled_width + (scaled_width % 2)  # Ensure even number

    # Step 4: Build crop filter based on mode
    mode = face_data.get("mode", "center")

    if mode == "dynamic":
        crop_x_expr = build_dynamic_crop_expression(
            face_data["detections"], scaled_width, target_width, scale_factor
        )
        # Clamp expression to valid range
        crop_x_expr = f"max(0\\,min({scaled_width - target_width}\\,{crop_x_expr}))"
        vf_filter = (
            f"scale={scaled_width}:{target_height},"
            f"crop={target_width}:{target_height}:{crop_x_expr}:0"
        )
        print(f"Dynamic crop — face follows {len(face_data['detections'])} keyframes")

    elif mode == "static":
        crop_x = cx_to_crop_x(face_data["median_cx"], scaled_width, target_width)
        vf_filter = (
            f"scale={scaled_width}:{target_height},"
            f"crop={target_width}:{target_height}:{crop_x}:0"
        )
        print(f"Static smart crop — face at {face_data['median_cx']:.2f}, crop_x={crop_x}")

    else:
        # Center crop fallback
        crop_x = (scaled_width - target_width) // 2
        vf_filter = (
            f"scale={scaled_width}:{target_height},"
            f"crop={target_width}:{target_height}:{crop_x}:0"
        )
        print(f"Center crop fallback — crop_x={crop_x}")

    # Step 5: Apply crop
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
        "-y", output_path,
    ]
    result = subprocess.run(crop_command, capture_output=True, text=True, timeout=300)

    # Cleanup
    if os.path.exists(raw_path):
        try:
            os.remove(raw_path)
        except Exception:
            pass

    if result.returncode != 0:
        raise RuntimeError(f"Crop failed:\n{result.stderr[-300:]}")

    return output_path
# import cv2
# import subprocess
# import os
# import numpy as np


# # ─────────────────────────────────────────────
# # FACE DETECTION USING OPENCV
# # Uses Haar Cascade — free, no API needed
# # ─────────────────────────────────────────────

# # OpenCV's built-in face detector
# FACE_CASCADE = cv2.CascadeClassifier(
#     cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
# )


# def extract_frame(video_path: str, timestamp: float, output_path: str) -> bool:
#     """Extract a single frame from video at given timestamp."""
#     command = [
#         "ffmpeg",
#         "-ss", str(timestamp),
#         "-i", video_path,
#         "-vframes", "1",
#         "-q:v", "2",
#         "-y",
#         output_path,
#     ]
#     result = subprocess.run(command, capture_output=True, timeout=30)
#     return os.path.exists(output_path) and os.path.getsize(output_path) > 100


# def detect_face_position(frame_path: str) -> dict | None:
#     """
#     Detects the largest face in a frame.
#     Returns { cx, cy, x, y, w, h } — center x/y and bounding box.
#     Returns None if no face found.
#     """
#     img = cv2.imread(frame_path)
#     if img is None:
#         return None

#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     height, width = img.shape[:2]

#     faces = FACE_CASCADE.detectMultiScale(
#         gray,
#         scaleFactor=1.1,
#         minNeighbors=5,
#         minSize=(30, 30),
#     )

#     if len(faces) == 0:
#         return None

#     # Pick the largest face (most likely the main subject)
#     largest = max(faces, key=lambda f: f[2] * f[3])
#     x, y, w, h = largest

#     return {
#         "x":  int(x),
#         "y":  int(y),
#         "w":  int(w),
#         "h":  int(h),
#         "cx": int(x + w // 2),   # Face center x
#         "cy": int(y + h // 2),   # Face center y
#         "frame_width":  width,
#         "frame_height": height,
#     }


# def analyze_face_position(video_path: str, clip_duration: float) -> dict:
#     """
#     Samples multiple frames across the clip to find the average face position.
#     Sampling multiple frames handles cases where the speaker moves.
#     Returns crop position info.
#     """
#     # Sample at 4 points: 20%, 40%, 60%, 80% through the clip
#     sample_points = [0.2, 0.4, 0.6, 0.8]
#     face_cx_values = []

#     temp_dir = os.path.dirname(video_path)

#     for i, pct in enumerate(sample_points):
#         timestamp = clip_duration * pct
#         frame_path = os.path.join(temp_dir, f"_sample_frame_{i}.jpg")

#         try:
#             if extract_frame(video_path, timestamp, frame_path):
#                 face = detect_face_position(frame_path)
#                 if face:
#                     face_cx_values.append({
#                         "cx": face["cx"],
#                         "frame_width": face["frame_width"],
#                         "frame_height": face["frame_height"],
#                     })
#         except Exception as e:
#             print(f"Frame sampling error at {pct}: {e}")
#         finally:
#             if os.path.exists(frame_path):
#                 os.remove(frame_path)

#     if not face_cx_values:
#         print("No face detected in any sample frame — using center crop")
#         return {"mode": "center"}

#     # Average face center x across all samples
#     avg_cx = sum(f["cx"] for f in face_cx_values) / len(face_cx_values)
#     frame_width = face_cx_values[0]["frame_width"]
#     frame_height = face_cx_values[0]["frame_height"]

#     print(f"Face detected — average center x: {avg_cx:.0f} / {frame_width}")

#     return {
#         "mode":         "face",
#         "avg_cx":       avg_cx,
#         "frame_width":  frame_width,
#         "frame_height": frame_height,
#     }


# def calculate_crop_x(face_data: dict, input_width: int, output_width: int) -> int:
#     """
#     Calculates the x offset for cropping to keep the face centered.
#     Clamps to valid range so crop never goes out of frame.
#     """
#     if face_data["mode"] == "center":
#         # Fallback: center crop
#         return (input_width - output_width) // 2

#     # Scale face position to the actual video dimensions
#     scale = input_width / face_data["frame_width"]
#     face_x_scaled = face_data["avg_cx"] * scale

#     # Center the crop window on the face
#     crop_x = int(face_x_scaled - output_width // 2)

#     # Clamp so crop stays within frame
#     crop_x = max(0, min(crop_x, input_width - output_width))

#     return crop_x


# def get_video_dimensions(video_path: str) -> tuple[int, int]:
#     """Returns (width, height) of video using ffprobe."""
#     try:
#         result = subprocess.run([
#             "ffprobe", "-v", "error",
#             "-select_streams", "v:0",
#             "-show_entries", "stream=width,height",
#             "-of", "csv=p=0",
#             video_path,
#         ], capture_output=True, text=True, timeout=30)

#         parts = result.stdout.strip().split(",")
#         return int(parts[0]), int(parts[1])
#     except Exception:
#         return 1920, 1080  # Safe fallback


# def smart_crop_to_vertical(
#     video_path: str,
#     output_path: str,
#     start_time: float,
#     duration: float,
# ) -> str:
#     """
#     Main function — extracts a clip with smart face-centered vertical crop.
#     Falls back to center crop if no face detected.

#     Steps:
#     1. Extract raw clip first (no crop, fast)
#     2. Analyze face position in the raw clip
#     3. Apply smart crop based on face position
#     """
#     os.makedirs(os.path.dirname(output_path), exist_ok=True)

#     # Step 1: Extract raw clip without crop for analysis
#     raw_path = output_path.replace(".mp4", "_raw_nocrop.mp4")

#     extract_command = [
#         "ffmpeg",
#         "-ss", str(start_time),
#         "-i", video_path,
#         "-t", str(duration),
#         "-c:v", "libx264",
#         "-preset", "ultrafast",   # Fast — this is just for analysis
#         "-crf", "28",
#          "-pix_fmt", "yuv420p", 
#         "-c:a", "aac",
#         "-avoid_negative_ts", "make_zero",
#         "-y",
#         raw_path,
#     ]

#     result = subprocess.run(extract_command, capture_output=True, text=True, timeout=120)
#     if result.returncode != 0:
#         raise RuntimeError(f"Raw clip extraction failed:\n{result.stderr[-300:]}")

#     # Step 2: Analyze face position
#     face_data = analyze_face_position(raw_path, duration)

#     # Step 3: Get video dimensions
#     input_width, input_height = get_video_dimensions(raw_path)

#     # Target: 9:16 vertical (1080x1920)
#     target_width  = 1080
#     target_height = 1920

#     # Calculate scale factor to make height = 1920
#     scale_factor = target_height / input_height
#     scaled_width = int(input_width * scale_factor)

#     # Calculate face-aware crop x on the scaled video
#     # Scale face position to match scaled dimensions
#     if face_data["mode"] == "face":
#         face_data_scaled = {
#             "mode":        "face",
#             "avg_cx":      face_data["avg_cx"] * (scaled_width / face_data["frame_width"]),
#             "frame_width": scaled_width,
#         }
#     else:
#         face_data_scaled = {"mode": "center"}

#     crop_x = calculate_crop_x(face_data_scaled, scaled_width, target_width)

#     print(f"Smart crop — mode: {face_data['mode']}, crop_x: {crop_x}")

#     # Step 4: Apply scale + smart crop in one FFmpeg pass
#     vf_filter = f"scale={scaled_width}:{target_height},crop={target_width}:{target_height}:{crop_x}:0"

#     crop_command = [
#         "ffmpeg",
#         "-i", raw_path,
#         "-vf", vf_filter,
#         "-c:v", "libx264",
#         "-preset", "fast",
#         "-crf", "23",
#         "-pix_fmt", "yuv420p",
#         "-c:a", "aac",
#         "-b:a", "128k",
#         "-movflags", "+faststart",
#         "-avoid_negative_ts", "make_zero",
#         "-y",
#         output_path,
#     ]

#     result = subprocess.run(crop_command, capture_output=True, text=True, timeout=300)

#     # Clean up raw clip
#     if os.path.exists(raw_path):
#         os.remove(raw_path)

#     if result.returncode != 0:
#         raise RuntimeError(f"Smart crop failed:\n{result.stderr[-300:]}")

#     return output_path