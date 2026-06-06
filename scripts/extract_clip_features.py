"""
Vision feature extraction pipeline

video (.webm)
↓
uniform frame sampling
↓
CLIP / OpenCLIP / SigLIP image encoder
↓
(num_frames, feature_dim) float32
↓
data/features_<backbone>/{split}/{id}.npy

Examples:
  python scripts/extract_clip_features.py --backbone clip_b32
  python scripts/extract_clip_features.py --backbone openclip_l14
  python scripts/extract_clip_features.py --backbone siglip_l16_384
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import av
import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import DATA_DIR, FEATURES_DIR, SUBSET_DIR, VIDEO_DIR


@dataclass(frozen=True)
class BackboneConfig:
    key: str
    model_name: str
    output_dir: Path
    note: str


BACKBONES = {
    "clip_b32": BackboneConfig(
        key="clip_b32",
        model_name="openai/clip-vit-base-patch32",
        output_dir=FEATURES_DIR,
        note="Original CLIP ViT-B/32 baseline.",
    ),
    "openclip_l14": BackboneConfig(
        key="openclip_l14",
        model_name="laion/CLIP-ViT-L-14-laion2B-s32B-b82K",
        output_dir=DATA_DIR / "features_openclip_l14",
        note="OpenCLIP ViT-L/14 candidate backbone.",
    ),
    "siglip_l16_384": BackboneConfig(
        key="siglip_l16_384",
        model_name="google/siglip-large-patch16-384",
        output_dir=DATA_DIR / "features_siglip_l16_384",
        note="SigLIP large patch16 384 candidate backbone.",
    ),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", choices=BACKBONES, default="clip_b32")
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_config(args) -> BackboneConfig:
    base = BACKBONES[args.backbone]
    return BackboneConfig(
        key=base.key,
        model_name=args.model_name or base.model_name,
        output_dir=args.output_dir or base.output_dir,
        note=base.note,
    )


def sample_frames(video_path: Path, n: int) -> list[Image.Image]:
    """Decode a webm video and return n uniformly sampled PIL frames."""
    container = av.open(str(video_path))
    stream = container.streams.video[0]

    frames_raw = [frame.to_image() for frame in container.decode(stream)]
    container.close()

    if not frames_raw:
        raise ValueError(f"No frames decoded: {video_path}")

    indices = np.linspace(0, len(frames_raw) - 1, n, dtype=int)
    return [frames_raw[i] for i in indices]


@torch.no_grad()
def extract(frames: list[Image.Image], model, processor, device: str) -> np.ndarray:
    """PIL frames -> normalized image features.

    Always uses the vision encoder path (pixel_values only) to avoid
    text-encoder side-effects from **inputs unpacking on CLIPModel.
    """
    inputs = processor(images=frames, return_tensors="pt").to(device)
    pixel_values = inputs["pixel_values"]

    if hasattr(model, "vision_model"):
        # CLIP / OpenCLIP / SigLIP — use vision encoder + optional projection
        vision_out = model.vision_model(pixel_values=pixel_values)
        feats = vision_out.pooler_output          # (N, hidden_dim)
        if hasattr(model, "visual_projection"):
            feats = model.visual_projection(feats)  # (N, proj_dim)
    elif hasattr(model, "get_image_features"):
        # Fallback: models that only expose get_image_features
        feats = model.get_image_features(pixel_values=pixel_values)
    else:
        raise RuntimeError(f"Cannot extract image features from {type(model).__name__}")

    feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return feats.cpu().float().numpy()


def write_metadata(config: BackboneConfig, output_dir: Path, num_frames: int, feature_dim: int):
    output_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "backbone": config.key,
        "model_name": config.model_name,
        "num_frames": num_frames,
        "feature_dim": feature_dim,
        "note": config.note,
    }
    with open(output_dir / "feature_config.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def process_split(
    split: str,
    config: BackboneConfig,
    model,
    processor,
    device: str,
    num_frames: int,
    overwrite: bool,
) -> int | None:
    if VIDEO_DIR is None:
        raise RuntimeError(
            "COAD_VIDEO_DIR is not set. Point it to the Something-Something V2 video directory."
        )

    src = SUBSET_DIR / f"{split}_mini.json"
    if not src.exists():
        print(f"[SKIP] {src} not found. Run create_mini_subset.py first.")
        return None

    with open(src, encoding="utf-8") as f:
        samples = json.load(f)

    out_dir = config.output_dir / split
    out_dir.mkdir(parents=True, exist_ok=True)

    ok, skip, fail = 0, 0, 0
    feature_dim = None

    for i, sample in enumerate(samples):
        vid_id = sample["id"]
        out_path = out_dir / f"{vid_id}.npy"

        if out_path.exists() and not overwrite:
            skip += 1
            if feature_dim is None:
                feature_dim = int(np.load(out_path, mmap_mode="r").shape[-1])
            continue

        video_path = VIDEO_DIR / f"{vid_id}.webm"
        if not video_path.exists():
            print(f"  [MISSING] {video_path}")
            fail += 1
            continue

        try:
            frames = sample_frames(video_path, num_frames)
            feats = extract(frames, model, processor, device)
            np.save(out_path, feats)
            feature_dim = int(feats.shape[-1])
            ok += 1
        except Exception as e:
            print(f"  [ERROR] {vid_id}: {e}")
            fail += 1

        if (i + 1) % 50 == 0:
            print(f"  [{split}] {i+1}/{len(samples)}  ok={ok} skip={skip} fail={fail}")

    print(f"=== {split} done: ok={ok} skip={skip} fail={fail} ===")
    return feature_dim


def select_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"          # Apple Silicon GPU — much faster than CPU
    return "cpu"


def main():
    args = parse_args()
    config = resolve_config(args)
    device = select_device()

    print(f"Loading {config.model_name} on {device}...")
    model = AutoModel.from_pretrained(config.model_name).to(device)
    processor = AutoProcessor.from_pretrained(config.model_name)
    model.eval()

    feature_dim = None
    for split in args.splits:
        split_dim = process_split(
            split=split,
            config=config,
            model=model,
            processor=processor,
            device=device,
            num_frames=args.num_frames,
            overwrite=args.overwrite,
        )
        feature_dim = feature_dim or split_dim

    if feature_dim is not None:
        write_metadata(config, config.output_dir, args.num_frames, feature_dim)
        print(f"feature_dim={feature_dim}")
        print(f"metadata: {config.output_dir / 'feature_config.json'}")


if __name__ == "__main__":
    main()
