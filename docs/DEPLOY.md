# Deploying the live demo

The interactive demo (realtime captioning, webcam, few-shot enrollment, anomaly
badge) is the FastAPI app in `app.py`. The GitHub Pages site is only a static
landing page — to make the demo publicly usable you need to host the app.

## Run locally (Docker)

```bash
docker build -t coad-mini .
docker run -p 7860:7860 coad-mini
# open http://localhost:7860
```

Or without Docker:

```bash
pip install -r requirements.txt   # needs ffmpeg on PATH
uvicorn app:app --port 8000
```

## Hugging Face Spaces (free hosting)

The `Dockerfile` is Spaces-ready (Docker SDK, port 7860, CLIP weights pre-cached).

1. Create a new Space → **SDK: Docker** → Blank.
2. Add this front-matter to the Space's `README.md` (Spaces reads it for config):

   ```yaml
   ---
   title: coad-mini
   emoji: ⚡
   colorFrom: blue
   colorTo: indigo
   sdk: docker
   app_port: 7860
   ---
   ```

3. Push this repo (it already contains `Dockerfile`, `app.py`, `src/`,
   `checkpoints/`) to the Space's git remote:

   ```bash
   git remote add space https://huggingface.co/spaces/<user>/coad-mini
   git push space main
   ```

4. The Space builds the image and serves the demo. Link it from the GitHub
   Pages landing (`docs/index.html`).

### Notes
- **Webcam** requires HTTPS (Spaces provides it; on localhost it also works).
- **Enrolled actions** (`/enroll`) are written to `checkpoints/agem_enrolled.pt`
  inside the container — they persist while the container runs but reset on a
  Space rebuild/restart unless you attach persistent storage.
- The image is CPU-only; inference is light (CLIP B/32 + a small GRU).
