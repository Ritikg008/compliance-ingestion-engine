from faster_whisper import WhisperModel
from pathlib import Path
from typing import List, Dict
from app.core.config import settings
import threading
import gc

_model = None
_model_lock = threading.Lock()


def get_whisper_model() -> WhisperModel:

    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                print(f"Loading Whisper model: {settings.whisper_model} ...")
                _model = WhisperModel(
                    settings.whisper_model,
                    device=settings.whisper_device,
                    compute_type=settings.whisper_compute_type,
                )
                print("Whisper model loaded.")
    return _model


def transcribe_video(video_path: Path) -> Dict:

    model = get_whisper_model()

    segments, info = model.transcribe(
        str(video_path),
        beam_size=5,          
        vad_filter=True,      
        vad_parameters=dict(
            min_silence_duration_ms=500
        ),
    )

    # segments is a generator — consume it fully into a list
    segment_list = []
    full_text_parts = []

    for segment in segments:
        segment_list.append({
            "start": round(segment.start, 2),
            "end": round(segment.end, 2),
            "text": segment.text.strip(),
        })
        full_text_parts.append(segment.text.strip())

    return {
        "language": info.language,
        "duration": round(info.duration, 2),
        "full_text": " ".join(full_text_parts),
        "segments": segment_list,
    }


def unload_whisper_model():

    global _model
    if _model is not None:
        del _model
        _model = None
        gc.collect()
        print("Whisper model unloaded from memory.")