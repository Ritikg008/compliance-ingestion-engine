import shutil
import uuid
import cv2
import os
from pathlib import Path
from typing import List, Tuple

import yt_dlp

from app.core.config import settings


def _node_js_runtime() -> dict:
    """Enable Node for yt-dlp YouTube JS challenges when available."""
    path = settings.node_path or shutil.which("node")
    return {"node": {"path": path} if path else {}}


def download_video(url: str) -> Tuple[Path, dict]:
    """
    Downloads a YouTube video to temp storage using yt-dlp.
    Returns the local file path and basic metadata.
    """
    video_id = str(uuid.uuid4())[:8]
    output_template = str(settings.temp_video_dir / f"{video_id}.%(ext)s")

    ydl_opts = {
        "format": "18/best[height<=720]/best",
        "merge_output_format": "mp4",
        "outtmpl": output_template,
        "quiet": True,
        "no_warnings": True,
        "js_runtimes": _node_js_runtime(),
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filepath = Path(ydl.prepare_filename(info))

    metadata = {
        "title": info.get("title"),
        "duration": info.get("duration"),
        "uploader": info.get("uploader"),
        "video_id": video_id,
    }

    return filepath, metadata


def extract_frames(video_path: Path, interval_seconds: int = 5) -> List[Path]:
    """
    Extracts one frame every `interval_seconds` from the video.
    Frames are saved as JPGs in temp_frames_dir.
    """
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frame_interval = int(fps * interval_seconds)

    frame_paths = []
    frame_count = 0
    video_id = video_path.stem

    while True:
        success, frame = cap.read()
        if not success:
            break

        if frame_count % frame_interval == 0:
            frame_filename = settings.temp_frames_dir / f"{video_id}_frame{frame_count}.jpg"
            cv2.imwrite(str(frame_filename), frame)
            frame_paths.append(frame_filename)

        frame_count += 1

    cap.release()
    return frame_paths


def cleanup_temp_files(video_path: Path, frame_paths: List[Path]):
    """Deletes temp video and frame files after processing to save disk space."""
    if video_path.exists():
        os.remove(video_path)
    for fp in frame_paths:
        if fp.exists():
            os.remove(fp)
