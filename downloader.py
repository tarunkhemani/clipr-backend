import yt_dlp
import os
import uuid


def download_video(youtube_url: str, output_dir: str = "downloads") -> dict:
    os.makedirs(output_dir, exist_ok=True)

    unique_id = str(uuid.uuid4())[:8]
    output_template = os.path.join(output_dir, f"{unique_id}.%(ext)s")

    ydl_opts = {
        "format": "bestvideo[ext=mp4][height<=1080]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "outtmpl": output_template,
        "merge_output_format": "mp4",
        "quiet": True,
        "no_warnings": True,
        "cookiefile": "cookies.txt",
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=True)
        title = info.get("title", "video")
        duration = info.get("duration", 0)

        video_path = os.path.join(output_dir, f"{unique_id}.mp4")
        if not os.path.exists(video_path):
            for f in os.listdir(output_dir):
                if f.startswith(unique_id):
                    video_path = os.path.join(output_dir, f)
                    break

    return {
        "video_path": video_path,
        "title": title,
        "duration": duration,
        "unique_id": unique_id,
    }


def cleanup_file(file_path: str):
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except Exception:
        pass