# coad-mini demo — FastAPI inference server (CPU)
# Build:  docker build -t coad-mini .
# Run:    docker run -p 7860:7860 coad-mini   ->  http://localhost:7860
# Also deploy-ready for Hugging Face Spaces (Docker SDK, app_port 7860).
FROM python:3.11-slim

# ffmpeg/ffprobe: needed by the /predict video-upload path
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only torch keeps the image small, then the rest of the deps
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-cache CLIP ViT-B/32 so the first request isn't slow (bakes weights into image)
RUN python -c "from transformers import CLIPModel, CLIPProcessor; \
    CLIPModel.from_pretrained('openai/clip-vit-base-patch32'); \
    CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')"

# App code + trained checkpoints (everything the server needs to run)
COPY app.py config.py ./
COPY static ./static
COPY src ./src
COPY checkpoints ./checkpoints

ENV PORT=7860
EXPOSE 7860
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-7860}"]
