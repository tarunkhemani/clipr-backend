import os
import shutil
import asyncio
import uuid
from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from downloader import download_video, cleanup_file
from transcriber import transcribe_audio
from analyzer import find_viral_clips
from clipper import process_all_clips
from script_generator import generate_script
from voice_generator import generate_voice, get_audio_duration, get_voices_list
from caption_builder import build_srt_from_words
from short_composer import compose_short
from collections import defaultdict
from datetime import datetime, date
from fastapi import Request

# ─────────────────────────────────────────────
# RATE LIMITING — max 5 requests per IP per day
# ─────────────────────────────────────────────

MAX_REQUESTS_PER_DAY = 5

# Format: { "ip_address": { "date": date, "count": int } }
ip_request_log: dict = defaultdict(lambda: {"date": date.today(), "count": 0})


def check_rate_limit(ip: str) -> bool:
    """
    Returns True if request is allowed, False if limit exceeded.
    Resets count automatically on a new day.
    """
    record = ip_request_log[ip]

    # Reset counter if it's a new day
    if record["date"] != date.today():
        record["date"]  = date.today()
        record["count"] = 0

    if record["count"] >= MAX_REQUESTS_PER_DAY:
        return False

    record["count"] += 1
    return True

load_dotenv()

app = FastAPI(title="Clipr API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

jobs: dict = {}

DOWNLOADS_DIR = "downloads"
CLIPS_DIR = "clips"
SHORTS_DIR = "shorts"
UPLOADS_DIR = "uploads"

for d in [DOWNLOADS_DIR, CLIPS_DIR, SHORTS_DIR, UPLOADS_DIR]:
    os.makedirs(d, exist_ok=True)


# ─────────────────────────────────────────────
# SHARED HELPERS
# ─────────────────────────────────────────────

def make_job(job_id: str) -> dict:
    return {
        "job_id": job_id,
        "status": "pending",
        "progress": 0,
        "message": "Starting...",
        "clips": [],
        "download_url": "",
        "script": "",
        "error": "",
    }


# ─────────────────────────────────────────────
# MODE 1: CLIP EXTRACTOR PIPELINE
# ─────────────────────────────────────────────

class ClipRequest(BaseModel):
    youtube_url: str


async def run_clip_extractor(job_id: str, youtube_url: str):
    video_path = None
    try:
        jobs[job_id].update({"status": "processing", "progress": 10, "message": "Downloading video from YouTube..."})

        loop = asyncio.get_event_loop()
        video_info = await loop.run_in_executor(None, download_video, youtube_url, DOWNLOADS_DIR)
        video_path = video_info["video_path"]
        unique_id = video_info["unique_id"]
        duration = video_info["duration"]

        jobs[job_id].update({"progress": 30, "message": "Transcribing audio..."})
        segments = await loop.run_in_executor(None, transcribe_audio, video_path)

        jobs[job_id].update({"progress": 55, "message": "AI finding viral moments..."})
        viral_clips = await loop.run_in_executor(None, find_viral_clips, segments, duration, video_path)

        if not viral_clips:
            raise ValueError("No viral clips found. Try a different video.")

        jobs[job_id].update({"progress": 70, "message": f"Found {len(viral_clips)} clips! Processing..."})

        clip_output_dir = os.path.join(CLIPS_DIR, unique_id)
        processed = await loop.run_in_executor(
            None, process_all_clips, video_path, viral_clips, segments, clip_output_dir, unique_id, True
        )

        result_clips = []
        for clip in processed:
            if clip["status"] == "success":
                result_clips.append({
                    "clip_number": clip["clip_number"],
                    "title": clip["title"],
                    "hook": clip["hook"],
                    "why_viral": clip["why_viral"],
                    "viral_score": clip["viral_score"],
                    "duration": clip["duration"],
                    "download_url": f"/download/clip/{unique_id}/{clip['clip_number']}",
                })

        jobs[job_id].update({
            "status": "done", "progress": 100,
            "message": f"Done! {len(result_clips)} clips ready.",
            "clips": result_clips,
        })

    except Exception as e:
        jobs[job_id].update({"status": "error", "progress": 0, "message": "Something went wrong.", "error": str(e)})
    finally:
        if video_path and os.path.exists(video_path):
            cleanup_file(video_path)


@app.post("/process")
async def start_clip_extraction(request: ClipRequest, background_tasks: BackgroundTasks, req: Request):
    ip = req.client.host
    if not check_rate_limit(ip):
        raise HTTPException(status_code=429, detail=f"Daily limit reached. You can process {MAX_REQUESTS_PER_DAY} videos per day for free. Come back tomorrow!")

    url = request.youtube_url.strip()
    if not url:
        raise HTTPException(status_code=400, detail="YouTube URL is required")
    if "youtube.com" not in url and "youtu.be" not in url:
        raise HTTPException(status_code=400, detail="Please provide a valid YouTube URL")

    job_id = str(uuid.uuid4())[:12]
    jobs[job_id] = make_job(job_id)
    background_tasks.add_task(run_clip_extractor, job_id, url)
    return jobs[job_id]


@app.get("/download/clip/{unique_id}/{clip_number}")
def download_clip(unique_id: str, clip_number: int):
    clip_path = os.path.join(CLIPS_DIR, unique_id, f"{unique_id}_clip{clip_number}.mp4")
    if not os.path.exists(clip_path):
        raise HTTPException(status_code=404, detail="Clip not found or expired")
    return FileResponse(path=clip_path, media_type="video/mp4", filename=f"viral_clip_{clip_number}.mp4")


# ─────────────────────────────────────────────
# MODE 2: AI SHORTS CREATOR PIPELINE
# ─────────────────────────────────────────────

async def run_short_creator(job_id: str, template: str, topic: str, voice: str, bg_video_path: str):
    try:
        loop = asyncio.get_event_loop()
        short_id = job_id

        short_dir = os.path.join(SHORTS_DIR, short_id)
        os.makedirs(short_dir, exist_ok=True)

        # Step 1: Generate script
        jobs[job_id].update({"status": "processing", "progress": 15, "message": "AI is writing your script..."})
        script = await loop.run_in_executor(None, generate_script, template, topic)
        jobs[job_id].update({"progress": 30, "message": "Script ready! Generating voice...", "script": script})

        # Step 2: Generate voice
        audio_path = os.path.join(short_dir, f"{short_id}_voice.mp3")
        word_timestamps = await loop.run_in_executor(None, generate_voice, script, voice, audio_path)
        audio_duration = get_audio_duration(word_timestamps)

        jobs[job_id].update({"progress": 55, "message": f"Voice done ({audio_duration:.1f}s). Building captions..."})

        # Step 3: Build caption chunks
        # Step 3: Build caption chunks
        from caption_builder import build_caption_chunks
        caption_chunks = build_caption_chunks(word_timestamps, words_per_chunk=4)
        print(f"Word timestamps received: {len(word_timestamps)}")
        print(f"Caption chunks built: {len(caption_chunks)}")
        if caption_chunks:
            print(f"First chunk: {caption_chunks[0]}")
        # from caption_builder import build_caption_chunks
        # caption_chunks = build_caption_chunks(word_timestamps, words_per_chunk=4)

        # jobs[job_id].update({"progress": 70, "message": "Captions ready! Composing final short..."})

        # Step 4: Compose final short
        output_path = os.path.join(short_dir, f"{short_id}_short.mp4")
        await loop.run_in_executor(
            None, compose_short, bg_video_path, audio_path, caption_chunks, output_path, audio_duration
        )
        # # Step 3: Build SRT captions
        # srt_path = os.path.join(short_dir, f"{short_id}_captions.srt")
        # await loop.run_in_executor(None, build_srt_from_words, word_timestamps, srt_path, 4)

        # jobs[job_id].update({"progress": 70, "message": "Captions ready! Composing final short..."})

        # # Step 4: Compose final short
        # output_path = os.path.join(short_dir, f"{short_id}_short.mp4")
        # await loop.run_in_executor(
        #     None, compose_short, bg_video_path, audio_path, srt_path, output_path, audio_duration
        # )

        jobs[job_id].update({
            "status": "done",
            "progress": 100,
            "message": "Your short is ready!",
            "download_url": f"/download/short/{short_id}",
        })

    except Exception as e:
        jobs[job_id].update({"status": "error", "progress": 0, "message": "Something went wrong.", "error": str(e)})
    finally:
        # Clean up uploaded background video
        if bg_video_path and os.path.exists(bg_video_path):
            try:
                os.remove(bg_video_path)
            except Exception:
                pass


@app.post("/create-short")
async def start_short_creation(
    background_tasks: BackgroundTasks,
    req: Request,
    template: str = Form(...),
    topic: str = Form(...),
    voice: str = Form("christopher"),
    background_video: UploadFile = File(...),
):
    ip = req.client.host
    if not check_rate_limit(ip):
        raise HTTPException(status_code=429, detail=f"Daily limit reached. You can create {MAX_REQUESTS_PER_DAY} shorts per day for free. Come back tomorrow!")

    if not topic.strip():
        raise HTTPException(status_code=400, detail="Topic is required")

    if not background_video.filename.lower().endswith((".mp4", ".mov", ".avi", ".mkv", ".webm")):
        raise HTTPException(status_code=400, detail="Please upload a valid video file (MP4, MOV, AVI)")

    job_id = str(uuid.uuid4())[:12]

    # Save uploaded background video
    bg_path = os.path.join(UPLOADS_DIR, f"{job_id}_bg{os.path.splitext(background_video.filename)[1]}")
    with open(bg_path, "wb") as f:
        shutil.copyfileobj(background_video.file, f)

    jobs[job_id] = make_job(job_id)
    background_tasks.add_task(run_short_creator, job_id, template, topic, voice, bg_path)
    return jobs[job_id]


@app.get("/download/short/{short_id}")
def download_short(short_id: str):
    short_path = os.path.join(SHORTS_DIR, short_id, f"{short_id}_short.mp4")
    if not os.path.exists(short_path):
        raise HTTPException(status_code=404, detail="Short not found or expired")
    return FileResponse(path=short_path, media_type="video/mp4", filename="clipr_short.mp4")


# ─────────────────────────────────────────────
# SHARED ROUTES
# ─────────────────────────────────────────────

@app.get("/")
def root():
    return {"message": "Clipr API v2 is running!"}


@app.get("/status/{job_id}")
def get_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[job_id]


@app.get("/voices")
def list_voices():
    return get_voices_list()
# ─────────────────────────────────────────────
# AUTO CLEANUP — delete files older than 1 hour
# ─────────────────────────────────────────────

import threading
import time as time_module


def cleanup_old_files():
    """Runs in background thread, deletes output files older than 1 hour."""
    while True:
        time_module.sleep(1800)  # Run every 30 minutes
        cutoff = time_module.time() - 3600  # 1 hour ago

        for folder in [CLIPS_DIR, SHORTS_DIR]:
            if not os.path.exists(folder):
                continue
            for subfolder in os.listdir(folder):
                subfolder_path = os.path.join(folder, subfolder)
                try:
                    if os.path.isdir(subfolder_path):
                        if os.path.getmtime(subfolder_path) < cutoff:
                            shutil.rmtree(subfolder_path)
                            print(f"Cleaned up: {subfolder_path}")
                except Exception as e:
                    print(f"Cleanup error: {e}")


# Start cleanup thread when server starts
cleanup_thread = threading.Thread(target=cleanup_old_files, daemon=True)
cleanup_thread.start()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
# import os
# import shutil
# import asyncio
# import uuid
# from fastapi import FastAPI, HTTPException, BackgroundTasks
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import FileResponse
# from pydantic import BaseModel
# from dotenv import load_dotenv

# from downloader import download_video, cleanup_file
# from transcriber import transcribe_audio
# from analyzer import find_viral_clips
# from clipper import process_all_clips

# load_dotenv()

# app = FastAPI(title="Clipr API", version="1.0.0")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# jobs: dict = {}

# DOWNLOADS_DIR = "downloads"
# CLIPS_DIR = "clips"
# os.makedirs(DOWNLOADS_DIR, exist_ok=True)
# os.makedirs(CLIPS_DIR, exist_ok=True)


# class ProcessRequest(BaseModel):
#     youtube_url: str


# async def process_video_job(job_id: str, youtube_url: str):
#     video_path = None

#     try:
#         jobs[job_id].update({
#             "status": "processing",
#             "progress": 10,
#             "message": "Downloading video from YouTube...",
#         })

#         loop = asyncio.get_event_loop()
#         video_info = await loop.run_in_executor(
#             None, download_video, youtube_url, DOWNLOADS_DIR
#         )
#         video_path = video_info["video_path"]
#         unique_id = video_info["unique_id"]
#         duration = video_info["duration"]

#         jobs[job_id].update({
#             "progress": 30,
#             "message": f"Downloaded. Now transcribing audio...",
#         })

#         segments = await loop.run_in_executor(
#             None, transcribe_audio, video_path
#         )

#         jobs[job_id].update({
#             "progress": 55,
#             "message": f"Transcribed. Finding viral moments with AI...",
#         })

#         viral_clips = await loop.run_in_executor(
#             None, find_viral_clips, segments, duration
#         )

#         if not viral_clips:
#             raise ValueError("No viral clips found in this video. Try a different video.")

#         jobs[job_id].update({
#             "progress": 70,
#             "message": f"Found {len(viral_clips)} viral clips! Processing video...",
#         })

#         clip_output_dir = os.path.join(CLIPS_DIR, unique_id)
#         processed = await loop.run_in_executor(
#             None,
#             process_all_clips,
#             video_path,
#             viral_clips,
#             segments,
#             clip_output_dir,
#             unique_id,
#             True,
#         )

#         result_clips = []
#         for clip in processed:
#             if clip["status"] == "success":
#                 result_clips.append({
#                     "clip_number": clip["clip_number"],
#                     "title": clip["title"],
#                     "hook": clip["hook"],
#                     "why_viral": clip["why_viral"],
#                     "viral_score": clip["viral_score"],
#                     "duration": clip["duration"],
#                     "download_url": f"/download/{unique_id}/{clip['clip_number']}",
#                 })

#         jobs[job_id].update({
#             "status": "done",
#             "progress": 100,
#             "message": f"Done! {len(result_clips)} clips ready.",
#             "clips": result_clips,
#         })

#     except Exception as e:
#         jobs[job_id].update({
#             "status": "error",
#             "progress": 0,
#             "message": "Something went wrong.",
#             "error": str(e),
#         })

#     finally:
#         if video_path and os.path.exists(video_path):
#             cleanup_file(video_path)


# @app.get("/")
# def root():
#     return {"message": "Clipr API is running!"}


# @app.post("/process")
# async def start_processing(request: ProcessRequest, background_tasks: BackgroundTasks):
#     url = request.youtube_url.strip()
#     if not url:
#         raise HTTPException(status_code=400, detail="YouTube URL is required")
#     if "youtube.com" not in url and "youtu.be" not in url:
#         raise HTTPException(status_code=400, detail="Please provide a valid YouTube URL")

#     job_id = str(uuid.uuid4())[:12]
#     jobs[job_id] = {
#         "job_id": job_id,
#         "status": "pending",
#         "progress": 0,
#         "message": "Starting...",
#         "clips": [],
#         "error": "",
#     }

#     background_tasks.add_task(process_video_job, job_id, url)
#     return jobs[job_id]


# @app.get("/status/{job_id}")
# def get_status(job_id: str):
#     if job_id not in jobs:
#         raise HTTPException(status_code=404, detail="Job not found")
#     return jobs[job_id]


# @app.get("/download/{unique_id}/{clip_number}")
# def download_clip(unique_id: str, clip_number: int):
#     clip_path = os.path.join(CLIPS_DIR, unique_id, f"{unique_id}_clip{clip_number}.mp4")
#     if not os.path.exists(clip_path):
#         raise HTTPException(status_code=404, detail="Clip not found or expired")
#     return FileResponse(
#         path=clip_path,
#         media_type="video/mp4",
#         filename=f"viral_clip_{clip_number}.mp4",
#     )


# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)