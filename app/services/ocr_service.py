import pytesseract
from PIL import Image
from pathlib import Path
from typing import List, Dict
import platform
if platform.system() == "Windows":
    default_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if Path(default_path).exists():
        pytesseract.pytesseract.tesseract_cmd = default_path
def extract_text_from_frame(frame_path: Path) -> str:
    """Runs OCR on a single frame and returns cleaned text."""
    try:
        image = Image.open(frame_path)
        text = pytesseract.image_to_string(image)
        return text.strip()
    except Exception as e:
        print(f"OCR failed on {frame_path}: {e}")
        return ""
def extract_text_from_frames(frame_paths: List[Path]) -> List[Dict]:
    results = []
    for frame_path in frame_paths:
        text = extract_text_from_frame(frame_path)
        if text:  # only keep frames where OCR actually found something
            results.append({
                "frame": str(frame_path),
                "text": text,
            })
    return results