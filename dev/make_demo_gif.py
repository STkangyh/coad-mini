"""
데모 GIF 생성기 — 실제 SS-V2 영상 + 실제 A-GEM/Baseline 모델 예측을
웹 데모(app.py)와 동일한 자막 오버레이 스타일로 합성한다.

출력: docs/assets/demo.gif

사용:
  COAD_VIDEO_DIR=/path/to/20bn-something-something-v2 python3 dev/make_demo_gif.py
"""
import os, sys, json
from pathlib import Path

import av
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.models.gru_detector import GRUDetector
from src.anomaly.detector import energy_score

VIDEO_DIR = Path(os.environ.get(
    "COAD_VIDEO_DIR",
    "/Users/younghoon-kang/something_v2_videos/20bn-something-something-v2"))
N_FRAMES = 16
W = 600                       # GIF 가로폭
FONT = "/System/Library/Fonts/AppleSDGothicNeo.ttc"
OUT = ROOT / "docs/assets/demo.gif"

device = "cpu"


def font(sz):
    return ImageFont.truetype(FONT, sz)


# ── 모델/라벨 로드 ────────────────────────────────────────────────────────────
def load_model(p):
    ck = torch.load(p, map_location="cpu", weights_only=False)
    m = GRUDetector(feature_dim=ck.get("feature_dim", 512),
                    hidden_dim=ck.get("hidden_dim", 256),
                    num_classes=ck.get("n_classes", 48),
                    num_layers=ck.get("num_layers", 1),
                    dropout=ck.get("dropout", 0.0),
                    bidirectional=ck.get("bidirectional", False))
    m.load_state_dict(ck["state_dict"]); m.eval()
    return m


print("loading models + CLIP ...")
agem = load_model(ROOT / "checkpoints/agem_48cls.pt")
base = load_model(ROOT / "checkpoints/baseline_48cls.pt")
labels = json.load(open(ROOT / "checkpoints/class_labels.json", encoding="utf-8"))
def lab(cid):
    e = labels.get(str(cid), {})
    return e.get("label") if isinstance(e, dict) else str(e)

from transformers import CLIPModel, CLIPProcessor
clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval()
proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


# ── 영상 → 프레임 ─────────────────────────────────────────────────────────────
def decode_frames(vid_id):
    path = VIDEO_DIR / f"{vid_id}.webm"
    if not path.exists():
        return None
    cont = av.open(str(path))
    frames = [f.to_image() for f in cont.decode(cont.streams.video[0])]
    cont.close()
    return frames or None


@torch.no_grad()
def features(pil_frames):
    idx = np.linspace(0, len(pil_frames) - 1, N_FRAMES, dtype=int)
    sel = [pil_frames[i] for i in idx]
    px = proc(images=sel, return_tensors="pt")["pixel_values"]
    vo = clip.vision_model(pixel_values=px)
    feat = clip.visual_projection(vo.pooler_output)
    feat = feat / feat.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return feat.unsqueeze(0)            # (1,16,512)


@torch.no_grad()
def predict(feat):
    la, _ = agem(feat, None); lb, _ = base(feat, None)
    pa = torch.softmax(la, -1)[0]; pb = torch.softmax(lb, -1)[0]
    ca = int(pa.argmax()); cb = int(pb.argmax())
    ood = float(energy_score(la)[0])    # 높을수록 OOD
    return {"agem": (ca, float(pa[ca]), lab(ca)),
            "base": (cb, float(pb[cb]), lab(cb)),
            "ood": ood}


# ── 후보 선택: agem 이 맞춘 서로 다른 stage 클립 3개 ──────────────────────────
def pick_clips(n=3):
    val = json.load(open(ROOT / "data/subset/val_mini.json", encoding="utf-8"))
    chosen, seen_stage = [], set()
    for s in val:
        cid = s["class_id"]
        if cid >= 48:
            continue
        stage = cid // 6
        if stage in seen_stage:
            continue
        fr = decode_frames(s["id"])
        if not fr:
            continue
        pred = predict(features(fr))
        if pred["agem"][0] == cid:           # 맞춘 클립만 (데모용)
            chosen.append((s, fr, pred))
            seen_stage.add(stage)
            print(f"  pick id={s['id']} stage{stage+1} '{pred['agem'][2]}' "
                  f"p={pred['agem'][1]:.2f} ood={pred['ood']:.2f}")
        if len(chosen) >= n:
            break
    return chosen


# ── 오버레이 합성 ─────────────────────────────────────────────────────────────
def rounded(draw, box, r, fill):
    draw.rounded_rectangle(box, radius=r, fill=fill)


def compose(frame_img, pred, w=W):
    # 비디오 프레임을 w 폭으로 letterbox
    vw, vh = frame_img.size
    scale = w / vw
    fh = int(vh * scale)
    vid = frame_img.resize((w, fh), Image.LANCZOS).convert("RGB")
    HEAD = 46
    H = HEAD + fh
    canvas = Image.new("RGB", (w, H), (15, 17, 23))
    canvas.paste(vid, (0, HEAD))
    ov = Image.new("RGBA", (w, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    # 헤더
    d.rectangle([0, 0, w, HEAD], fill=(22, 33, 62, 255))
    d.text((w//2, HEAD//2), "A-GEM Continual Learning Demo  ·  Baseline vs A-GEM",
           font=font(15), fill=(88, 166, 255, 255), anchor="mm")
    # OOD 칩 (중앙 상단, 영상 위)
    chip = f"OOD score {pred['ood']:.2f}"
    cb = d.textbbox((0, 0), chip, font=font(13))
    cw = cb[2] - cb[0]
    cx0 = w//2 - cw//2 - 10
    rounded(d, [cx0, HEAD + fh - 96, cx0 + cw + 20, HEAD + fh - 72], 6,
            (40, 50, 70, 220))
    d.text((w//2, HEAD + fh - 84), chip, font=font(13),
           fill=(201, 209, 217, 255), anchor="mm")
    # 자막 박스 2개
    def sub(x0, x1, tag, tagcol, boxcol, text, prob):
        y0 = HEAD + fh - 66
        y1 = HEAD + fh - 10
        rounded(d, [x0, y0, x1, y1], 8, boxcol)
        cx = (x0 + x1) // 2
        d.text((cx, y0 + 11), tag, font=font(11), fill=tagcol, anchor="mm")
        d.text((cx, y0 + 30), text, font=font(15), fill=(255, 255, 255, 255), anchor="mm")
        d.text((cx, y0 + 47), f"{prob*100:.0f}%", font=font(11),
               fill=(230, 230, 230, 230), anchor="mm")
    gap = 8
    half = (w - gap - 16) // 2
    sub(8, 8 + half, "BASELINE", (255, 200, 200, 255), (200, 60, 50, 225),
        pred["base"][2], pred["base"][1])
    sub(w - 8 - half, w - 8, "A-GEM", (200, 255, 210, 255), (40, 160, 60, 225),
        pred["agem"][2], pred["agem"][1])
    out = Image.alpha_composite(canvas.convert("RGBA"), ov).convert("RGB")
    return out


def main():
    clips = pick_clips(3)
    if not clips:
        print("ERROR: 적합한 클립을 못 찾음"); sys.exit(1)
    frames_out = []
    target_h = None
    for s, fr, pred in clips:
        # 클립당 균등 10프레임
        idx = np.linspace(0, len(fr) - 1, 10, dtype=int)
        for i in idx:
            img = compose(fr[i], pred)
            if target_h is None:
                target_h = img.height
            if img.height != target_h:       # 높이 통일(letterbox 편차 보정)
                img = img.resize((W, target_h), Image.LANCZOS)
            frames_out.append(img)
        # 클립 사이 짧은 정지(마지막 프레임 2회 추가)
        frames_out += [frames_out[-1]] * 3

    OUT.parent.mkdir(parents=True, exist_ok=True)
    # 팔레트 최적화하여 GIF 저장
    pal = [f.convert("P", palette=Image.ADAPTIVE, colors=128) for f in frames_out]
    pal[0].save(OUT, save_all=True, append_images=pal[1:],
                duration=140, loop=0, optimize=True, disposal=2)
    print(f"saved {OUT}  ({OUT.stat().st_size/1024:.0f} KB, {len(pal)} frames)")


if __name__ == "__main__":
    main()
