import os
from pathlib import Path

FEATURE_DIM = 512   # CLIP ViT-B/32
NUM_CLASSES = 16    # 4 stages × 4 classes
WINDOW_SIZE = 16    # 영상당 샘플링 프레임 수

REPO_ROOT = Path(__file__).resolve().parent
DATA_DIR = REPO_ROOT / "data"
SUBSET_DIR = DATA_DIR / "subset"
FEATURES_DIR = DATA_DIR / "features"
CHECKPOINT_DIR = REPO_ROOT / "checkpoints"

VIDEO_DIR = Path(os.environ["COAD_VIDEO_DIR"]).expanduser() if os.environ.get("COAD_VIDEO_DIR") else None
LABELS_DIR = Path(os.environ["COAD_LABELS_DIR"]).expanduser() if os.environ.get("COAD_LABELS_DIR") else None