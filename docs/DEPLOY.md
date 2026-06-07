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
2. The Space's `README.md` must carry the Spaces config (YAML front-matter). A
   ready-made one is in this repo at [`deploy/space-README.md`](../deploy/space-README.md) —
   use it as the Space's `README.md`.
3. Push this repo (it already contains `Dockerfile`, `app.py`, `src/`,
   `checkpoints/`) to the Space's git remote, with the Space README in place:

   ```bash
   git remote add space https://huggingface.co/spaces/<user>/coad-mini
   cp deploy/space-README.md README.space.md   # then commit it AS README.md on the space branch
   # simplest: clone the empty Space, copy repo files in, copy deploy/space-README.md -> README.md, push
   git push space main
   ```

   (The Space needs its `README.md` to be the front-matter version; the GitHub
   project keeps its own `README.md`, so swap the file when pushing to the Space.)
4. The Space builds the image and serves the demo. Link it from the GitHub
   Pages landing (`docs/index.html`).

### Large files (checkpoints) must use Git LFS

HF rejects raw binaries (`Your push was rejected because it contains binary files`).
The `.pt` checkpoints must be tracked with Git LFS **before** pushing to the Space:

```bash
git lfs install
git lfs track "*.pt"          # writes .gitattributes
git add .gitattributes checkpoints/*.pt
git commit -m "track checkpoints with LFS"
git push
```

If they were already committed as plain blobs, rewrite first:
`git lfs migrate import --include="*.pt"` then `git lfs ls-files` to confirm, then push.
(Alternatively, skip git entirely: `hf upload <user>/coad-mini . --repo-type=space`.)

### Updating a deployed Space

The Space is a separate git repo — pushing to GitHub does **not** update it.
Re-copy the changed files into the Space checkout (or re-run `hf upload`) and push again.

### Notes
- **Webcam** requires HTTPS (Spaces provides it; on localhost it also works).
- **Enrolled actions** (`/enroll`) are written to `checkpoints/agem_enrolled.pt`
  inside the container — they persist while the container runs but reset on a
  Space rebuild/restart unless you attach persistent storage.
- The image is CPU-only; inference is light (CLIP B/32 + a small GRU).
