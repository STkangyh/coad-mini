"""
Efficiency measurement for the meeting brief (CPU).
Reports: param counts, per-stage train wall-clock, inference latency/throughput
(CLIP feature extraction + GRU head), and peak RAM. All on CPU.

Run: python3 dev/measure_efficiency.py
"""
import os, sys, time, resource, statistics
import numpy as np
import torch
sys.path.insert(0, ".")
from src.models.gru_detector import GRUDetector
from src.models.gru_attention import GRUAttentionDetector
from src.models.ssm_detector import SSMDetector
from src.trainer import load_samples, train_epoch
import torch.nn as nn

torch.set_num_threads(os.cpu_count() or 4)
device = "cpu"
FEAT_DIR = "data/features/train"     # CLIP B/32 features (512-d)
def peak_ram_mb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / (1024*1024) if sys.platform == "darwin" else r / 1024  # mac=bytes, linux=KB


def nparams(m):
    return sum(p.numel() for p in m.parameters())


print("== 1) Model size (params) ==")
gru  = GRUDetector(feature_dim=512, hidden_dim=256, num_classes=48)
attn = GRUAttentionDetector(feature_dim=512, hidden_dim=256, num_classes=48)
ssm  = SSMDetector(feature_dim=512, hidden_dim=256, num_classes=48, num_layers=2)
for name, m in [("GRU (main)", gru), ("GRU+Attention", attn), ("diagonal-SSM", ssm)]:
    print(f"  {name:16s}: {nparams(m):,} params ({nparams(m)/1e3:.1f}K)")
print("  (context: VideoMAE-V2 ViT-g ~1,013,000K; InternVideo2-6B ~6,000,000K)")

print("\n== 2) Inference latency (temporal head, precomputed 16x512 feature) ==")
gru.eval()
x = torch.randn(1, 16, 512)
with torch.no_grad():
    for _ in range(5): gru(x, None)          # warmup
    t = []
    for _ in range(200):
        s = time.perf_counter(); gru(x, None); t.append((time.perf_counter()-s)*1000)
print(f"  GRU head: {statistics.mean(t):.3f} ms/window  ->  {1000/statistics.mean(t):.0f} windows/s")

print("\n== 3) CLIP ViT-B/32 feature extraction (16 frames, the realtime bottleneck) ==")
try:
    from transformers import CLIPModel, CLIPProcessor
    from PIL import Image
    clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval()
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    frames = [Image.fromarray(np.random.randint(0,255,(224,224,3),dtype=np.uint8)) for _ in range(16)]
    px = proc(images=frames, return_tensors="pt")["pixel_values"]
    with torch.no_grad():
        for _ in range(2): clip.vision_model(pixel_values=px)   # warmup
        t = []
        for _ in range(10):
            s = time.perf_counter(); clip.vision_model(pixel_values=px); t.append((time.perf_counter()-s)*1000)
    ms = statistics.mean(t)
    print(f"  CLIP 16-frame batch: {ms:.0f} ms  ->  end-to-end (CLIP+GRU) ~{ms:.0f} ms/window ~ {1000/ms:.1f} windows/s")
except Exception as e:
    print("  [skip CLIP]", e)

print("\n== 4) Training wall-clock (one stage = 6 classes x100 samples, B/32) ==")
try:
    train = load_samples("data/subset/train_mini.json")
    stage1 = [s for s in train if s["class_id"] < 6][:600]
    have = sum(os.path.exists(f"{FEAT_DIR}/{s['id']}.npy") for s in stage1)
    if have < 50:
        print(f"  [skip] only {have} B/32 features present")
    else:
        m = GRUDetector(feature_dim=512, hidden_dim=256, num_classes=48)
        crit = nn.BCEWithLogitsLoss(); opt = torch.optim.Adam(m.parameters(), lr=1e-4)
        from pathlib import Path
        fd = Path(FEAT_DIR)
        s = time.perf_counter(); train_epoch(m, stage1, fd, crit, opt, device); dt = time.perf_counter()-s
        print(f"  1 epoch on {len(stage1)} samples: {dt:.1f}s")
        print(f"  -> per stage (15 epochs): ~{dt*15:.0f}s ; full 8-stage run: ~{dt*15*8/60:.1f} min (single seed, CPU)")
except Exception as e:
    print("  [skip train]", e)

print(f"\n== Peak RAM: {peak_ram_mb():.0f} MB ==")
