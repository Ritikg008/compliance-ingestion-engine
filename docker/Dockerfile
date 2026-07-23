FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    ffmpeg \
    tesseract-ocr \
    libglib2.0-0 \
    libgl1 \
    nodejs \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# Install CPU-only PyTorch first to avoid CUDA downloads (~700MB saved)
RUN pip install --no-cache-dir torch==2.4.1+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY .env .env

RUN mkdir -p data/temp_video data/temp_frames

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]