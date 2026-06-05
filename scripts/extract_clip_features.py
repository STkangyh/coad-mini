"""
CLIP feature extraction pipeline

video (.webm)
↓
16 frame uniform sampling
↓
CLIP ViT-B/32  (openai/clip-vit-base-patch32)
↓
(16, 512) float32
↓
data/features/{split}/{id}.npy
"""

import json
import sys
import numpy as np
from pathlib import Path

import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import av  # PyAV — webm 디코딩

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import FEATURES_DIR, SUBSET_DIR, VIDEO_DIR


# ── 설정 ────────────────────────────────────────────────────────────────────
MODEL_NAME   = "openai/clip-vit-base-patch32"
NUM_FRAMES   = 16
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"


# ── CLIP 로드 ────────────────────────────────────────────────────────────────
print(f"Loading {MODEL_NAME} on {DEVICE}...")
model     = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE)
processor = CLIPProcessor.from_pretrained(MODEL_NAME)
model.eval()


# ── 유틸 ────────────────────────────────────────────────────────────────────
def sample_frames(video_path: Path, n: int = NUM_FRAMES) -> list[Image.Image]:
    """webm에서 균등 간격으로 n 프레임 추출 → PIL Image 리스트"""
    container = av.open(str(video_path))
    stream = container.streams.video[0]

    # 전체 프레임 수 추정
    total = stream.frames or 0
    frames_raw = []
    for frame in container.decode(stream):
        frames_raw.append(frame.to_image())
        if total and len(frames_raw) >= total:
            break
    container.close()

    if len(frames_raw) == 0:
        raise ValueError(f"No frames decoded: {video_path}")

    # 균등 샘플링
    indices = np.linspace(0, len(frames_raw) - 1, n, dtype=int)
    return [frames_raw[i] for i in indices]


@torch.no_grad()
def extract(frames: list[Image.Image]) -> np.ndarray:
    """PIL 프레임 리스트 → (n, 512) float32"""
    inputs = processor(images=frames, return_tensors="pt").to(DEVICE)
    vision_out = model.vision_model(**inputs)        # BaseModelOutputWithPooling
    pooled     = vision_out.pooler_output            # (n, hidden_dim)
    feats      = model.visual_projection(pooled)     # (n, 512)
    feats      = feats / feats.norm(dim=-1, keepdim=True)   # L2 정규화
    return feats.cpu().float().numpy()


# ── 메인 ────────────────────────────────────────────────────────────────────
def process_split(split: str):
    if VIDEO_DIR is None:
        raise RuntimeError(
            "COAD_VIDEO_DIR is not set. Point it to the Something-Something V2 video directory."
        )
    src = SUBSET_DIR / f"{split}_mini.json"
    if not src.exists():
        print(f"[SKIP] {src} not found. Run create_mini_subset.py first.")
        return

    with open(src, encoding="utf-8") as f:
        samples = json.load(f)

    out_dir = FEATURES_DIR / split
    out_dir.mkdir(parents=True, exist_ok=True)

    ok, skip, fail = 0, 0, 0

    for i, sample in enumerate(samples):
        vid_id    = sample["id"]
        out_path  = out_dir / f"{vid_id}.npy"

        if out_path.exists():
            skip += 1
            continue

        video_path = VIDEO_DIR / f"{vid_id}.webm"
        if not video_path.exists():
            print(f"  [MISSING] {video_path}")
            fail += 1
            continue

        try:
            frames = sample_frames(video_path, NUM_FRAMES)
            feats  = extract(frames)             # (16, 512)
            np.save(out_path, feats)
            ok += 1
        except Exception as e:
            print(f"  [ERROR] {vid_id}: {e}")
            fail += 1

        if (i + 1) % 50 == 0:
            print(f"  [{split}] {i+1}/{len(samples)}  ok={ok} skip={skip} fail={fail}")

    print(f"=== {split} done: ok={ok} skip={skip} fail={fail} ===\n")


if __name__ == "__main__":
    for split in ["train", "val"]:
        process_split(split)
