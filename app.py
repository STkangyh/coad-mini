"""
FastAPI Inference Server
========================
엔드포인트:
  GET  /              → 웹 UI (Baseline vs A-GEM 비교 데모)
  POST /predict       → 영상 파일 업로드 → {baseline, agem} 예측 결과 반환
  GET  /forgetting    → 학습 과정 forgetting 곡선 데이터 (JSON)
  GET  /health        → 서버 상태

실행:
  pip install fastapi uvicorn python-multipart
  uvicorn app:app --host 0.0.0.0 --port 8000

핸드폰에서 접속:
  http://<맥북 IP>:8000
"""

import io
import json
import tempfile
import subprocess
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import HTMLResponse, JSONResponse

from src.models.gru_detector import GRUDetector
from src.enroll.few_shot import expand_classifier, FewShotEnroller
from src.anomaly.detector import AnomalyScorer

# ── 설정 ─────────────────────────────────────────────────────────────────────
CKPT_DIR    = Path("checkpoints")
ENROLLED_CKPT   = CKPT_DIR / "agem_enrolled.pt"       # few-shot 으로 확장된 모델 (persist)
ENROLLED_LABELS = CKPT_DIR / "class_labels_enrolled.json"
FEATURE_DIM = 512
HIDDEN_DIM  = 256
N_CLASSES   = 48
N_FRAMES    = 16
device      = "cpu"   # 추론만이므로 CPU로 충분

# ── 모델 & 레이블 로드 ────────────────────────────────────────────────────────
def load_model(ckpt_path: Path) -> GRUDetector:
    ckpt  = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = GRUDetector(
        feature_dim=ckpt.get("feature_dim", FEATURE_DIM),
        hidden_dim=ckpt.get("hidden_dim",  HIDDEN_DIM),
        num_classes=ckpt.get("n_classes",  N_CLASSES),
        num_layers=ckpt.get("num_layers", 1),
        dropout=ckpt.get("dropout", 0.0),
        bidirectional=ckpt.get("bidirectional", False),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


def load_labels(path: Path = None) -> dict:
    p = path or (CKPT_DIR / "class_labels.json")
    if not p.exists():
        return {str(i): {"template": f"class_{i}", "label": f"class_{i}", "examples": []}
                for i in range(N_CLASSES)}
    with open(p, encoding="utf-8") as f:
        raw = json.load(f)
    # 구 포맷(string) → 신 포맷(dict) 자동 변환
    result = {}
    for k, v in raw.items():
        if isinstance(v, str):
            result[k] = {"template": v, "label": v.replace("[something]", "…").replace("[somewhere]", "…"), "examples": []}
        else:
            result[k] = v
    return result


# ── CLIP feature 추출 ─────────────────────────────────────────────────────────
_clip_model = None
_clip_preprocess = None

def get_clip():
    global _clip_model, _clip_preprocess
    if _clip_model is None:
        try:
            from transformers import CLIPProcessor, CLIPModel
            _clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            _clip_preprocess = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            _clip_model.eval()
        except Exception as e:
            raise HTTPException(500, f"CLIP load failed: {e}")
    return _clip_model, _clip_preprocess


def extract_frames_from_video(video_bytes: bytes, n_frames: int = N_FRAMES) -> np.ndarray:
    """ffmpeg로 영상에서 균등 간격 n_frames 추출 → (n_frames, H, W, 3) numpy"""
    try:
        from PIL import Image
    except ImportError:
        raise HTTPException(500, "Pillow is required (pip install Pillow)")

    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name

    # ffmpeg로 총 프레임 수 파악
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-count_frames", "-show_entries", "stream=nb_read_frames",
         "-of", "csv=p=0", tmp_path],
        capture_output=True, text=True,
    )
    try:
        total_frames = int(probe.stdout.strip())
    except Exception:
        total_frames = 30   # fallback

    # 균등 간격 timestamp 계산
    indices = np.linspace(0, total_frames - 1, n_frames, dtype=int)

    frames = []
    for idx in indices:
        result = subprocess.run(
            ["ffmpeg", "-i", tmp_path,
             "-vf", f"select=eq(n\\,{idx})",
             "-vframes", "1",
             "-f", "image2pipe", "-vcodec", "png", "-"],
            capture_output=True,
        )
        if result.returncode == 0 and result.stdout:
            img = Image.open(io.BytesIO(result.stdout)).convert("RGB")
            frames.append(img)

    Path(tmp_path).unlink(missing_ok=True)

    if len(frames) < 2:
        raise HTTPException(400, "Failed to extract frames from the video. Check that it is webm/mp4.")

    # 부족한 경우 마지막 프레임 반복
    while len(frames) < n_frames:
        frames.append(frames[-1])

    return frames[:n_frames]


@torch.no_grad()
def video_to_feature(video_bytes: bytes) -> torch.Tensor:
    """영상 bytes → CLIP feature (1, N_FRAMES, 512)"""
    clip_model, processor = get_clip()
    frames = extract_frames_from_video(video_bytes)

    inputs = processor(images=frames, return_tensors="pt", padding=True)
    vision_out = clip_model.vision_model(pixel_values=inputs["pixel_values"])
    pooled     = vision_out.pooler_output                        # (N_FRAMES, 768)
    feats      = clip_model.visual_projection(pooled)            # (N_FRAMES, 512)
    feats      = F.normalize(feats, dim=-1)
    return feats.unsqueeze(0)                                    # (1, N_FRAMES, 512)


@torch.no_grad()
def predict(model: GRUDetector, feat: torch.Tensor, labels: dict, top_k: int = 5):
    """logits → top-k 결과 반환"""
    logits, _ = model(feat, None)
    probs      = torch.softmax(logits[0], dim=-1).cpu().numpy()
    top_ids    = probs.argsort()[::-1][:top_k]
    results    = []
    for i in top_ids:
        entry = labels.get(str(i), {})
        if isinstance(entry, dict):
            template = entry.get("template", f"class_{i}")
            label    = entry.get("label",    template)
            examples = entry.get("examples", [])
        else:
            template = label = str(entry)
            examples = []
        results.append({
            "class_id": int(i),
            "template": template,
            "label":    label,
            "examples": examples,
            "prob":     float(probs[i]),
        })
    return results


# ── 실시간 추론: base64 프레임 리스트 → 예측 ──────────────────────────────────
@torch.no_grad()
def frames_to_feature(frames_b64: list[str]) -> torch.Tensor:
    """base64 PNG 리스트 (16장) → CLIP feature (1, N_FRAMES, 512)"""
    try:
        from PIL import Image
    except ImportError:
        raise HTTPException(500, "Pillow is required (pip install Pillow)")

    clip_model, processor = get_clip()
    imgs = []
    for b64 in frames_b64:
        import base64
        raw  = base64.b64decode(b64.split(",")[-1])  # data:image/png;base64,... 제거
        img  = Image.open(io.BytesIO(raw)).convert("RGB")
        imgs.append(img)

    # 16장에 맞게 맞추기 (부족하면 마지막 반복, 초과하면 균등 서브샘플)
    if len(imgs) < N_FRAMES:
        while len(imgs) < N_FRAMES:
            imgs.append(imgs[-1])
    elif len(imgs) > N_FRAMES:
        idx  = np.linspace(0, len(imgs) - 1, N_FRAMES, dtype=int)
        imgs = [imgs[i] for i in idx]

    inputs     = processor(images=imgs, return_tensors="pt", padding=True)
    vision_out = clip_model.vision_model(pixel_values=inputs["pixel_values"])
    pooled     = vision_out.pooler_output
    feats      = clip_model.visual_projection(pooled)
    feats      = F.normalize(feats, dim=-1)
    return feats.unsqueeze(0)   # (1, N_FRAMES, 512)


# ── 앱 초기화 ─────────────────────────────────────────────────────────────────
app = FastAPI(title="A-GEM Continual Learning Demo", version="1.0")

# 체크포인트 존재 확인
_ckpt_ready = (CKPT_DIR / "baseline_48cls.pt").exists() and \
              (CKPT_DIR / "agem_48cls.pt").exists()

def _load_active_gem():
    """enrolled 체크포인트가 있으면 그걸(=가르친 동작 유지), 없으면 base A-GEM 을 서빙."""
    if ENROLLED_CKPT.exists() and ENROLLED_LABELS.exists():
        m, ck = load_model(ENROLLED_CKPT)
        return m, ck, load_labels(ENROLLED_LABELS), True
    m, ck = load_model(CKPT_DIR / "agem_48cls.pt")
    return m, ck, load_labels(), False


if _ckpt_ready:
    _bl_model,  _bl_ckpt  = load_model(CKPT_DIR / "baseline_48cls.pt")
    _gem_model, _gem_ckpt, _labels, _enrolled_active = _load_active_gem()
    print(f"✓ Checkpoints loaded (enrolled_active={_enrolled_active}, "
          f"n_classes={_gem_model.num_classes})")
else:
    _bl_model = _gem_model = _bl_ckpt = _gem_ckpt = None
    _labels = load_labels()
    print("⚠ Checkpoints not found. Run save_checkpoints.py first.")

# ── Anomaly / OOD detector ────────────────────────────────────────────────────
# Higher anomaly score == more out-of-distribution (see src/anomaly/detector.py).
# ANOMALY_THRESHOLD env var, if set, overrides everything below.
#
# BUGFIX: previously, with no env var set (the default `uvicorn app:app` /
# Docker path), `threshold` stayed None forever and `/predict_anomaly` /
# `is_anomaly` in the UI badge NEVER fired — the feature silently did nothing
# out of the box. We now auto-calibrate at startup: real val features if
# present (local dev), else synthetic in-distribution windows (Docker/Spaces,
# where data/ isn't shipped) so the badge is functional either way.
import os as _os
_ANOMALY_METRIC = _os.environ.get("ANOMALY_METRIC", "energy")
_anomaly_thr_env = _os.environ.get("ANOMALY_THRESHOLD")
_anomaly_scorer = AnomalyScorer(
    metric=_ANOMALY_METRIC,
    threshold=float(_anomaly_thr_env) if _anomaly_thr_env else None,
    smoothing="ema",
    ema_alpha=0.5,
)


def _auto_calibrate_anomaly(scorer: AnomalyScorer, model, target_fpr: float = 0.05) -> str:
    """Set scorer.threshold from real val features if available, else synthetic
    in-distribution windows. Returns a short string describing the source."""
    real_dir = Path("data/features/val")
    real_files = sorted(real_dir.glob("*.npy"))[:300] if real_dir.exists() else []
    scores = []
    if real_files:
        with torch.no_grad():
            for p in real_files:
                x = torch.tensor(np.load(p), dtype=torch.float32).unsqueeze(0)
                logits, _ = model(x, None)
                scores.append(scorer.score_logits(logits))
        source = f"real val features (n={len(scores)})"
    else:
        from experiments.calibrate_anomaly import synthetic_in_dist_windows
        rng = np.random.default_rng(0)
        windows = synthetic_in_dist_windows(300, model.feature_dim, rng)
        with torch.no_grad():
            for w in windows:
                x = torch.tensor(w, dtype=torch.float32).unsqueeze(0)
                logits, _ = model(x, None)
                scores.append(scorer.score_logits(logits))
        source = f"synthetic in-distribution windows (n={len(scores)}, no data/ shipped)"
    scorer.calibrate(scores, target_fpr=target_fpr)
    return source


if _anomaly_thr_env:
    print(f"✓ Anomaly threshold from ANOMALY_THRESHOLD env: {_anomaly_scorer.threshold}")
elif _gem_model is not None:
    try:
        _src = _auto_calibrate_anomaly(_anomaly_scorer, _gem_model)
        print(f"✓ Anomaly threshold auto-calibrated: {_anomaly_scorer.threshold:.3f} "
              f"(metric={_ANOMALY_METRIC}, target_fpr=0.05, source={_src})")
    except Exception as e:
        print(f"⚠ Anomaly auto-calibration failed ({e}); OOD badge will not fire until "
              f"ANOMALY_THRESHOLD is set or experiments/calibrate_anomaly.py is run.")


# ── 라우트 ─────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "checkpoints_ready": _ckpt_ready,
        "n_classes": N_CLASSES,
        "device": device,
    }


@app.get("/forgetting")
def forgetting_data():
    """학습 중 acc_table → forgetting 곡선 데이터 반환"""
    if not _ckpt_ready:
        raise HTTPException(503, "checkpoints not ready")

    def extract_curve(ckpt: dict, method: str):
        """acc_table[after_stage][eval_stage] → S1 곡선"""
        acc_table = ckpt.get("acc_table", {})
        stages    = sorted(acc_table.keys())
        return {
            "method": method,
            "stages": stages,
            "s1_acc": [acc_table[s].get(1, 0.0) for s in stages],
            "avg_acc": [
                float(np.mean(list(acc_table[s].values()))) for s in stages
            ],
        }

    return JSONResponse({
        "baseline": extract_curve(_bl_ckpt,  "Baseline"),
        "agem":     extract_curve(_gem_ckpt, "A-GEM"),
    })


@app.post("/predict")
async def predict_action(file: UploadFile = File(...)):
    """영상 업로드 → Baseline / A-GEM 예측 결과"""
    if not _ckpt_ready:
        raise HTTPException(503, "checkpoints not ready — run save_checkpoints.py first")

    video_bytes = await file.read()
    if len(video_bytes) == 0:
        raise HTTPException(400, "Empty file")

    try:
        feat = video_to_feature(video_bytes)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Feature extraction failed: {e}")

    return JSONResponse({
        "filename": file.filename,
        "baseline": predict(_bl_model,  feat, _labels),
        "agem":     predict(_gem_model, feat, _labels),
    })


class RTRequest(dict):
    pass

from pydantic import BaseModel

class RTPayload(BaseModel):
    frames: list[str]   # base64 PNG strings (최근 N_FRAMES장)

@app.post("/predict_rt")
async def predict_realtime(payload: RTPayload):
    """실시간 추론: 클라이언트 canvas 캡처 프레임 → 예측"""
    if not _ckpt_ready:
        raise HTTPException(503, "checkpoints not ready")
    if len(payload.frames) == 0:
        raise HTTPException(400, "No frames")

    try:
        feat = frames_to_feature(payload.frames)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Feature extraction failed: {e}")

    # 실시간이므로 top-1만 반환 (속도 우선)
    def top1(model):
        r = predict(model, feat, _labels, top_k=3)
        return r

    return JSONResponse({
        "baseline": top1(_bl_model),
        "agem":     top1(_gem_model),
    })


# ── Few-shot enrollment: 새 행동 클래스 등록 ──────────────────────────────────
@app.post("/enroll")
async def enroll_new_class(
    label: str = Form(...),
    files: list[UploadFile] = File(...),
):
    """
    새 행동 클래스를 few-shot 으로 등록하고, 서빙 모델에 즉시 반영(hot-swap)한다.

    multipart 입력:
      - label : 새 클래스 이름 (문자열)
      - files : 새 클래스 예시 영상 (1개 이상)

    동작:
      1. 각 영상 → CLIP feature window (1, 16, 512) 추출 (기존 헬퍼 재사용)
      2. A-GEM 모델을 N → N+1 클래스로 확장
      3. 기존 클래스 exemplar 를 replay 하며 새 클래스 학습 (FewShotEnroller)
      4. 체크포인트/레이블 저장(persist) + 서빙 모델 즉시 교체 → /predict_rt 에 바로 등장

    NOTE: feature 추출에는 ffmpeg 와 실제 영상이 필요하다. 모델/엔드포인트
          로직 자체는 합성 입력으로도 검증 가능하다 (tests 참고).
    """
    global _gem_model, _gem_ckpt, _labels
    if not _ckpt_ready:
        raise HTTPException(503, "checkpoints not ready — run save_checkpoints.py first")
    if not label or not label.strip():
        raise HTTPException(400, "Label is empty")
    if not files:
        raise HTTPException(400, "At least one video is required")

    label = label.strip()

    # 1. feature 추출
    new_windows = []
    for f in files:
        video_bytes = await f.read()
        if len(video_bytes) == 0:
            continue
        try:
            feat = video_to_feature(video_bytes)   # (1, 16, 512)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(500, f"Feature extraction failed ({f.filename}): {e}")
        new_windows.append(feat.squeeze(0))        # (16, 512)

    if not new_windows:
        raise HTTPException(400, "No valid videos")

    # 2. 모델 확장 (현재 A-GEM 모델의 deepcopy 로 작업)
    import copy
    model = copy.deepcopy(_gem_model)
    old_n = model.num_classes
    new_class_id = old_n
    model = expand_classifier(model, n_new=1)

    # 3. 기존 클래스 exemplar replay 재료 만들기
    #    학습 중 모델이 잘 분류하는 합성 exemplar 를 기존 클래스마다 만들어
    #    replay buffer 로 사용한다 (실제 feature 파일은 worktree 에 없음).
    enroller = FewShotEnroller(model, device=device, use_agem=True, seed=0)
    exemplars, ex_labels = _build_replay_exemplars(_gem_model, old_n, n_per_class=1)
    enroller.add_exemplars(exemplars, ex_labels)

    # 4. 학습
    model = enroller.enroll(
        new_windows, [new_class_id] * len(new_windows),
        epochs=20, lr=1e-3, replay_per_step=8,
    )

    # 5. 저장
    new_state = model.state_dict()
    out_ckpt = {
        "state_dict": new_state,
        "n_classes": model.num_classes,
        "feature_dim": model.feature_dim,
        "hidden_dim": model.hidden_dim,
        "num_layers": model.num_layers,
        "dropout": model.dropout,
        "bidirectional": model.bidirectional,
        "method": "A-GEM + few-shot enrollment",
        "enrolled_from": "agem_48cls.pt",
    }
    torch.save(out_ckpt, ENROLLED_CKPT)

    # 레이블 파일 갱신
    new_labels = dict(_labels)
    new_labels[str(new_class_id)] = {
        "template": label,
        "label": label,
        "examples": [],
    }
    with open(ENROLLED_LABELS, "w", encoding="utf-8") as fp:
        json.dump(new_labels, fp, ensure_ascii=False, indent=2)

    # 6. ★ 서빙 모델 즉시 교체 (hot-swap) — /predict_rt 가 새 동작을 바로 인식
    _gem_model = model
    _gem_ckpt  = out_ckpt
    _labels    = new_labels

    return JSONResponse({
        "status": "ok",
        "new_class_id": new_class_id,
        "label": label,
        "n_classes": model.num_classes,
        "n_examples": len(new_windows),
        "checkpoint": str(ENROLLED_CKPT),
        "labels_file": str(ENROLLED_LABELS),
        "live": True,
    })


@torch.no_grad()
def _build_replay_exemplars(model: GRUDetector, n_classes: int, n_per_class: int = 1):
    """
    기존 클래스 replay 용 합성 exemplar 생성.
    모델이 클래스 c 로 예측하는 합성 window 를 찾아 (window, c) 로 모은다.
    실제 feature 파일이 없는 환경에서도 동작하도록 한 폴백이다.
    """
    rng = np.random.default_rng(0)
    fdim = model.feature_dim
    found: dict[int, list] = {c: [] for c in range(n_classes)}
    remaining = n_classes * n_per_class
    # 무작위 window 를 흘려보내 클래스별로 채운다 (상한 트라이).
    for _ in range(n_classes * n_per_class * 40):
        if remaining <= 0:
            break
        w = rng.standard_normal((N_FRAMES, fdim)).astype(np.float32)
        logits, _ = model(torch.from_numpy(w).unsqueeze(0), None)
        c = int(logits.argmax(-1).item())
        if c < n_classes and len(found[c]) < n_per_class:
            found[c].append(w)
            remaining -= 1
    windows, labels = [], []
    for c, ws in found.items():
        for w in ws:
            windows.append(w)
            labels.append(c)
    return windows, labels


# ── 등록된 동작 관리 ──────────────────────────────────────────────────────────
@app.get("/classes")
def list_classes():
    """현재 서빙 모델의 클래스 목록 (기본 48 + 사용자가 등록한 동작)."""
    if _gem_model is None:
        raise HTTPException(503, "model not ready")
    items = []
    for cid in range(_gem_model.num_classes):
        e = _labels.get(str(cid), {})
        lbl = e.get("label") if isinstance(e, dict) else str(e)
        items.append({"id": cid, "label": lbl or f"class_{cid}", "enrolled": cid >= N_CLASSES})
    return {
        "n_classes": _gem_model.num_classes,
        "base_count": N_CLASSES,
        "enrolled": [x for x in items if x["enrolled"]],
    }


@app.post("/reset_classes")
def reset_classes():
    """등록한 동작을 모두 제거하고 기본 48-class A-GEM 모델로 되돌린다."""
    global _gem_model, _gem_ckpt, _labels
    if not _ckpt_ready:
        raise HTTPException(503, "checkpoints not ready")
    _gem_model, _gem_ckpt = load_model(CKPT_DIR / "agem_48cls.pt")
    _labels = load_labels()
    for p in (ENROLLED_CKPT, ENROLLED_LABELS):
        try:
            p.unlink()
        except FileNotFoundError:
            pass
    return {"status": "reset", "n_classes": _gem_model.num_classes}


# ── 이상행동 / OOD 탐지 ────────────────────────────────────────────────────────
@app.post("/predict_anomaly")
async def predict_anomaly(payload: RTPayload):
    """이상/신규(OOD) 행동 탐지.

    입력: base64 PNG 프레임 리스트 (/predict_rt 와 동일).
    출력: {top1, top1_prob, anomaly_score, is_anomaly}

    anomaly_score 가 높을수록 OOD(학습되지 않은/저신뢰) 행동.
    임계값은 ANOMALY_THRESHOLD 환경변수 또는 experiments/calibrate_anomaly.py
    로 교정하며, 미설정 시 is_anomaly 는 항상 False(점수만 반환).
    A-GEM 모델 logits 기준으로 평가한다.
    """
    if not _ckpt_ready:
        raise HTTPException(503, "checkpoints not ready")
    if len(payload.frames) == 0:
        raise HTTPException(400, "No frames")

    try:
        feat = frames_to_feature(payload.frames)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Feature extraction failed: {e}")

    # 단발 추론이므로 temporal smoothing 미사용(smooth=False)
    result = _anomaly_scorer.score_window(_gem_model, feat, smooth=False)
    entry = _labels.get(str(result["top1"]), {})
    label = entry.get("label") if isinstance(entry, dict) else str(entry)

    return JSONResponse({
        "top1": label or f"class_{result['top1']}",
        "top1_class_id": result["top1"],
        "top1_prob": result["top1_prob"],
        "anomaly_score": result["raw_score"],
        "is_anomaly": result["is_anomaly"],
        "metric": _ANOMALY_METRIC,
        "threshold": result["threshold"],
    })


# ── 웹 UI ─────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
def index():
    return HTML_PAGE


HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>A-GEM Continual Learning Demo</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: #0f1117; color: #e1e4e8; min-height: 100vh; }

  .header { background: linear-gradient(135deg, #1a1d2e 0%, #16213e 100%);
             padding: 24px 20px; text-align: center; border-bottom: 1px solid #30363d; }
  .header h1 { font-size: 1.6rem; font-weight: 700; color: #58a6ff; }
  .header p  { color: #8b949e; margin-top: 6px; font-size: 0.9rem; }

  .container { max-width: 760px; margin: 0 auto; padding: 24px 16px; }

  .card { background: #161b22; border: 1px solid #30363d;
          border-radius: 12px; padding: 20px; margin-bottom: 20px; }
  .card h2 { font-size: 1rem; font-weight: 600; color: #58a6ff;
             margin-bottom: 14px; display: flex; align-items: center; gap: 8px; }

  /* Upload */
  .upload-area { border: 2px dashed #30363d; border-radius: 8px;
                 padding: 32px 16px; text-align: center; cursor: pointer;
                 transition: border-color 0.2s; }
  .upload-area:hover { border-color: #58a6ff; }
  .upload-area input { display: none; }
  .upload-area .icon { font-size: 2.5rem; }
  .upload-area .hint { color: #8b949e; font-size: 0.85rem; margin-top: 8px; }
  .upload-area .filename { color: #3fb950; margin-top: 8px; font-size: 0.9rem; }

  .btn { width: 100%; padding: 12px; border: none; border-radius: 8px;
         background: #238636; color: #fff; font-size: 1rem; font-weight: 600;
         cursor: pointer; transition: background 0.2s; margin-top: 14px; }
  .btn:hover:not(:disabled) { background: #2ea043; }
  .btn:disabled { background: #21262d; color: #484f58; cursor: not-allowed; }

  /* Results */
  .result-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
  @media (max-width: 480px) { .result-grid { grid-template-columns: 1fr; } }

  .method-box { background: #0d1117; border-radius: 8px; padding: 14px; }
  .method-box.baseline { border-left: 3px solid #f85149; }
  .method-box.agem     { border-left: 3px solid #3fb950; }
  .method-title { font-weight: 700; font-size: 0.95rem; margin-bottom: 10px; }
  .method-box.baseline .method-title { color: #f85149; }
  .method-box.agem     .method-title { color: #3fb950; }

  .pred-item { display: flex; align-items: center; gap: 8px; margin-bottom: 8px; }
  .pred-rank { color: #8b949e; font-size: 0.75rem; width: 20px; flex-shrink: 0; }
  .pred-bar-wrap { flex: 1; background: #21262d; border-radius: 4px; height: 6px; overflow: hidden; }
  .pred-bar { height: 100%; border-radius: 4px; transition: width 0.5s; }
  .method-box.baseline .pred-bar { background: #f85149; }
  .method-box.agem     .pred-bar { background: #3fb950; }
  .pred-label { font-size: 0.78rem; color: #c9d1d9; }
  .pred-prob  { font-size: 0.78rem; color: #8b949e; width: 40px; text-align: right; flex-shrink: 0; }
  .top1-label { font-size: 1rem; font-weight: 700; color: #e6edf3; margin-bottom: 10px; }

  /* Forgetting chart */
  canvas { width: 100% !important; }

  .spinner { text-align: center; padding: 20px; color: #58a6ff; font-size: 0.9rem; }
  .error   { color: #f85149; font-size: 0.85rem; margin-top: 10px; }

  .badge { display: inline-block; padding: 2px 8px; border-radius: 12px;
           font-size: 0.72rem; font-weight: 600; }
  .badge.green { background: #0d3321; color: #3fb950; }
  .badge.red   { background: #2d1217; color: #f85149; }

  .stats-row { display: flex; gap: 16px; flex-wrap: wrap; margin-top: 4px; }
  .stat { font-size: 0.82rem; color: #8b949e; }
  .stat span { color: #e6edf3; font-weight: 600; }

  @keyframes fadeIn {
    from { opacity: 0; transform: translateY(4px); }
    to   { opacity: 1; transform: translateY(0); }
  }
</style>
</head>
<body>

<div class="header">
  <h1>⚡ A-GEM Continual Learning Demo</h1>
  <p>Something-Something V2 · 48 classes · 8 stages · GRU + CLIP ViT-B/32</p>
</div>

<div class="container">

  <!-- 업로드 -->
  <div class="card">
    <h2>📹 Upload video</h2>
    <!-- before upload: dropzone -->
    <div class="upload-area" id="dropZone" onclick="document.getElementById('fileInput').click()">
      <input type="file" id="fileInput" accept="video/*,.webm,.mp4,.avi">
      <div class="icon">🎬</div>
      <div class="hint">Click or drag &amp; drop a video<br>webm · mp4 · avi</div>
      <div class="filename" id="filenameLabel"></div>
    </div>
    <!-- live webcam -->
    <button class="btn" id="webcamBtn" onclick="startWebcam()"
            style="background:#1f6feb;margin-top:10px">📷 Live webcam demo</button>
    <!-- 업로드 후: 인라인 플레이어 + 자막 오버레이 -->
    <div id="videoWrap" style="display:none;margin-top:12px;border-radius:8px;
         overflow:hidden;background:#000;position:relative;user-select:none">
      <video id="videoPreview" controls playsinline
             style="width:100%;max-height:340px;display:block;object-fit:contain"></video>

      <!-- ① 현재 행동 자막 (영상 위) -->
      <div id="subtitleOverlay" style="display:none;position:absolute;bottom:52px;
           left:0;right:0;pointer-events:none;padding:0 10px">
        <!-- 이상/신규(OOD) 동작 배지 -->
        <div id="anomalyBadge" style="display:none;text-align:center;margin-bottom:5px">
          <span id="anomalyChip" style="background:rgba(40,50,70,0.85);color:#c9d1d9;
                border-radius:6px;padding:3px 10px;font-size:0.7rem;font-weight:600;
                backdrop-filter:blur(6px)">OOD <span id="anomalyScoreText">—</span></span>
        </div>
        <div style="display:flex;gap:6px;justify-content:center">
          <!-- Baseline -->
          <div id="subBaseline" style="background:rgba(200,60,50,0.88);color:#fff;
               border-radius:6px;padding:4px 10px;font-size:0.78rem;font-weight:600;
               backdrop-filter:blur(6px);max-width:46%;text-align:center;
               transition:opacity 0.3s;line-height:1.4">
            <div style="font-size:0.62rem;opacity:0.75;letter-spacing:.5px">BASELINE</div>
            <div id="subBaselineText">—</div>
            <div id="subBaselineProb" style="font-size:0.65rem;opacity:0.7"></div>
          </div>
          <!-- A-GEM -->
          <div id="subAgem" style="background:rgba(40,160,60,0.88);color:#fff;
               border-radius:6px;padding:4px 10px;font-size:0.78rem;font-weight:600;
               backdrop-filter:blur(6px);max-width:46%;text-align:center;
               transition:opacity 0.3s;line-height:1.4">
            <div style="font-size:0.62rem;opacity:0.75;letter-spacing:.5px">A-GEM</div>
            <div id="subAgemText">—</div>
            <div id="subAgemProb" style="font-size:0.65rem;opacity:0.7"></div>
          </div>
        </div>
      </div>

      <!-- 버튼 -->
      <div style="position:absolute;top:8px;right:8px;display:flex;gap:6px">
        <button id="rtBtn" onclick="toggleRT()"
                style="background:rgba(88,166,255,0.85);color:#fff;border:none;
                       border-radius:6px;padding:4px 10px;font-size:0.8rem;cursor:pointer">
          ▶ Live captions
        </button>
        <button onclick="resetUpload()"
                style="background:rgba(0,0,0,0.6);color:#fff;border:none;
                       border-radius:6px;padding:4px 10px;font-size:0.8rem;cursor:pointer">
          ✕ Reset
        </button>
      </div>
      <canvas id="captureCanvas" style="display:none"></canvas>
    </div>

    <!-- ② 행동 변화 타임라인 (영상 아래) -->
    <div id="timelineWrap" style="display:none;margin-top:10px">
      <div style="font-size:0.75rem;color:#8b949e;margin-bottom:6px;
                  display:flex;align-items:center;gap:6px">
        ⏱ Action timeline
        <span style="font-size:0.68rem;color:#484f58">
          (logged when the A-GEM prediction changes)
        </span>
      </div>
      <div id="timelineTrack" style="display:flex;gap:4px;flex-wrap:wrap;
           max-height:90px;overflow-y:auto;align-content:flex-start"></div>
    </div>
    <button class="btn" id="predictBtn" disabled onclick="runPredict()">
      🔍 Predict (Baseline vs A-GEM)
    </button>
    <div class="error" id="errorMsg"></div>
  </div>

  <!-- 예측 결과 -->
  <div class="card" id="resultCard" style="display:none">
    <h2>🎯 Prediction <span id="resultFilename" style="font-weight:400;color:#8b949e;font-size:0.8rem"></span></h2>
    <div class="result-grid" id="resultGrid"></div>
  </div>

  <!-- 새 동작 등록 (few-shot) -->
  <div class="card">
    <h2>➕ Enroll a new action <span style="font-weight:400;color:#8b949e;font-size:0.8rem">(few-shot · A-GEM)</span></h2>
    <p style="font-size:0.82rem;color:#8b949e;margin-bottom:12px">
      Teach a brand-new action class from a few example clips.
      A-GEM replay adds the (49th+) class without forgetting the existing 48.
    </p>
    <input id="enrollLabel" type="text" placeholder="New action name (e.g. waving)"
           style="width:100%;padding:11px 12px;border:1px solid #30363d;border-radius:8px;
                  background:#0d1117;color:#e6edf3;font-size:0.95rem;margin-bottom:12px">
    <div class="upload-area" id="enrollDrop" onclick="document.getElementById('enrollFiles').click()">
      <input type="file" id="enrollFiles" accept="video/*,.webm,.mp4,.avi" multiple>
      <div class="icon">📥</div>
      <div class="hint">Select one or more example videos</div>
      <div class="filename" id="enrollFileLabel"></div>
    </div>
    <button class="btn" id="enrollBtn" onclick="runEnroll()">➕ Enroll</button>
    <div class="error" id="enrollError"></div>
    <div id="enrollResult" style="display:none;margin-top:12px;padding:12px;
         background:#0d3321;border-left:3px solid #3fb950;border-radius:8px;
         font-size:0.9rem;color:#e6edf3;line-height:1.5"></div>

    <!-- 등록된 동작 목록 -->
    <div id="enrolledWrap" style="display:none;margin-top:16px">
      <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:8px">
        <span style="font-size:0.8rem;color:#8b949e">🧩 My enrolled actions</span>
        <button onclick="resetClasses()" style="background:#21262d;color:#f85149;border:1px solid #30363d;
                border-radius:6px;padding:4px 10px;font-size:0.75rem;cursor:pointer">Reset all</button>
      </div>
      <div id="enrolledList" style="display:flex;gap:6px;flex-wrap:wrap"></div>
    </div>
  </div>

  <!-- Forgetting 곡선 -->
  <div class="card">
    <h2>📉 Forgetting curve <span style="font-weight:400;color:#8b949e;font-size:0.8rem">(Stage 1 accuracy over stages)</span></h2>
    <div id="chartContainer">
      <div class="spinner">Loading…</div>
    </div>
    <div class="stats-row" id="statsRow"></div>
  </div>

</div>

<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<script>
// ── 파일 선택 ─────────────────────────────────────────────────────────────────
const fileInput     = document.getElementById('fileInput');
const filenameLabel = document.getElementById('filenameLabel');
const predictBtn    = document.getElementById('predictBtn');
const dropZone      = document.getElementById('dropZone');
const videoWrap     = document.getElementById('videoWrap');
const videoPreview  = document.getElementById('videoPreview');

function showVideoPreview(file) {
  stopWebcam();
  videoPreview.setAttribute('controls', '');
  const url = URL.createObjectURL(file);
  videoPreview.src = url;
  dropZone.style.display = 'none';
  videoWrap.style.display = 'block';
  predictBtn.disabled = false;
}

// ── 웹캠 라이브 ───────────────────────────────────────────────────────────────
let webcamStream = null;
async function startWebcam() {
  try {
    webcamStream = await navigator.mediaDevices.getUserMedia({
      video: { width: 480, height: 480 }, audio: false });
  } catch (e) {
    document.getElementById('errorMsg').textContent = 'Webcam access failed: ' + e.message;
    return;
  }
  stopRT();
  videoPreview.srcObject = webcamStream;
  videoPreview.removeAttribute('controls');
  dropZone.style.display = 'none';
  videoWrap.style.display = 'block';
  document.getElementById('resultCard').style.display = 'none';
  predictBtn.disabled = true;          // 라이브 모드엔 파일 업로드 예측 비활성
  await videoPreview.play().catch(() => {});
  startRT();
}

function stopWebcam() {
  if (webcamStream) { webcamStream.getTracks().forEach(t => t.stop()); webcamStream = null; }
  if (videoPreview.srcObject) videoPreview.srcObject = null;
}

function resetUpload() {
  stopRT();
  stopWebcam();
  videoPreview.setAttribute('controls', '');
  videoPreview.src = '';
  videoWrap.style.display = 'none';
  dropZone.style.display = 'block';
  filenameLabel.textContent = '';
  predictBtn.disabled = true;
  fileInput.value = '';
  document.getElementById('resultCard').style.display = 'none';
  document.getElementById('subtitleOverlay').style.display = 'none';
  document.getElementById('timelineWrap').style.display = 'none';
  document.getElementById('anomalyBadge').style.display = 'none';
  document.getElementById('timelineTrack').innerHTML = '';
}

fileInput.addEventListener('change', () => {
  const f = fileInput.files[0];
  if (f) {
    filenameLabel.textContent = '✓ ' + f.name;
    showVideoPreview(f);
    stopRT();
    document.getElementById('timelineTrack').innerHTML = '';
    document.getElementById('timelineWrap').style.display = 'none';
  }
});

// drag & drop
const uploadArea = document.querySelector('.upload-area');
uploadArea.addEventListener('dragover', e => { e.preventDefault(); uploadArea.style.borderColor = '#58a6ff'; });
uploadArea.addEventListener('dragleave', () => { uploadArea.style.borderColor = '#30363d'; });
uploadArea.addEventListener('drop', e => {
  e.preventDefault();
  uploadArea.style.borderColor = '#30363d';
  const dt = new DataTransfer();
  dt.items.add(e.dataTransfer.files[0]);
  fileInput.files = dt.files;
  fileInput.dispatchEvent(new Event('change'));
});

// ── 실시간 자막 ───────────────────────────────────────────────────────────────
const FRAME_BUFFER_SIZE = 16;  // GRU 슬라이딩 윈도우
const CAPTURE_INTERVAL  = 200; // 200ms마다 캡처 (≈5fps)
const INFER_EVERY       = 2;   // 2 캡처마다 추론 (≈400ms)
const STABLE_WINDOW     = 3;   // 최근 N번 예측 중 최빈값으로 안정화 (떨림 방지)

let rtActive    = false;
let rtTimer     = null;
let frameBuffer = [];
let captureCount = 0;
let inferPending = false;

// 안정화용 최근 예측 버퍼
let predHistory = { baseline: [], agem: [] };  // 최근 STABLE_WINDOW개

// 타임라인: A-GEM 예측이 바뀔 때만 기록
let lastGemLabel = null;
let timelineEntries = [];

// 자막 색상 팔레트 (클래스 stage별)
const STAGE_COLORS = [
  '#c0392b','#e67e22','#f39c12','#27ae60',
  '#16a085','#2980b9','#8e44ad','#2c3e50',
];

function stageColor(label) {
  // label로 stage 색 추정 (간단히 해시)
  let h = 0;
  for (const c of label) h = (h * 31 + c.charCodeAt(0)) & 0xfffffff;
  return STAGE_COLORS[h % STAGE_COLORS.length];
}

const captureCanvas = document.getElementById('captureCanvas');
const captureCtx    = captureCanvas.getContext('2d');

function captureFrame() {
  if (videoPreview.paused || videoPreview.ended) return;
  const W = 224, H = 224;
  captureCanvas.width  = W;
  captureCanvas.height = H;
  captureCtx.drawImage(videoPreview, 0, 0, W, H);
  const b64 = captureCanvas.toDataURL('image/jpeg', 0.65);
  frameBuffer.push(b64);
  if (frameBuffer.length > FRAME_BUFFER_SIZE) frameBuffer.shift();
  captureCount++;
  if (captureCount % INFER_EVERY === 0 &&
      frameBuffer.length === FRAME_BUFFER_SIZE &&
      !inferPending) {
    runRT([...frameBuffer]);
  }
}

async function runRT(frames) {
  inferPending = true;
  try {
    const res = await fetch('/predict_rt', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ frames }),
    });
    if (res.ok) storeAndRender(await res.json());
    // 이상/신규(OOD) 점수 — 같은 프레임으로 병렬 호출 (비차단)
    fetch('/predict_anomaly', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ frames }),
    }).then(r => r.ok ? r.json() : null)
      .then(d => { if (d) updateAnomaly(d); })
      .catch(() => {});
  } catch(_) {}
  finally { inferPending = false; }
}

// ── 이상/신규(OOD) 배지 업데이트 ───────────────────────────────────────────────
function updateAnomaly(d) {
  const badge = document.getElementById('anomalyBadge');
  const chip  = document.getElementById('anomalyChip');
  badge.style.display = 'block';
  const score = (d.anomaly_score != null) ? d.anomaly_score.toFixed(2) : '—';
  if (d.is_anomaly) {
    chip.style.background = 'rgba(248,81,73,0.92)';
    chip.style.color = '#fff';
    chip.innerHTML = '⚠ Unknown action? <span id="anomalyScoreText">' + score + '</span>';
  } else {
    chip.style.background = 'rgba(40,50,70,0.85)';
    chip.style.color = '#c9d1d9';
    chip.innerHTML = 'OOD score <span id="anomalyScoreText">' + score + '</span>';
  }
}

// ── 안정화: 최근 N번 중 최빈값 선택 ──────────────────────────────────────────
function stablePred(history, newPred) {
  history.push(newPred);
  if (history.length > STABLE_WINDOW) history.shift();
  const freq = {};
  let best = null, bestN = 0;
  for (const p of history) {
    freq[p.label] = (freq[p.label] || 0) + 1;
    if (freq[p.label] > bestN) { best = p; bestN = freq[p.label]; }
  }
  return best;
}

function storeAndRender(data) {
  const rawBl  = data.baseline[0];
  const rawGem = data.agem[0];

  const stableBl  = stablePred(predHistory.baseline, rawBl);
  const stableGem = stablePred(predHistory.agem,     rawGem);

  updateSubtitle(stableBl, stableGem);

  // 타임라인: A-GEM 예측이 바뀔 때만 항목 추가
  if (stableGem.label !== lastGemLabel) {
    lastGemLabel = stableGem.label;
    const t = videoPreview.currentTime.toFixed(1);
    addTimelineEntry(t, stableBl, stableGem);
  }
}

// ── 자막 업데이트 (페이드 효과) ───────────────────────────────────────────────
function updateSubtitle(bl, gem) {
  document.getElementById('subtitleOverlay').style.display = 'block';

  const blText  = document.getElementById('subBaselineText');
  const blProb  = document.getElementById('subBaselineProb');
  const gemText = document.getElementById('subAgemText');
  const gemProb = document.getElementById('subAgemProb');

  // 바뀐 경우에만 페이드 처리
  if (blText.textContent !== bl.label) {
    flashElement(document.getElementById('subBaseline'));
    blText.textContent = bl.label;
  }
  if (gemText.textContent !== gem.label) {
    flashElement(document.getElementById('subAgem'));
    gemText.textContent = gem.label;
  }
  blProb.textContent  = (bl.prob  * 100).toFixed(0) + '%';
  gemProb.textContent = (gem.prob * 100).toFixed(0) + '%';
}

function flashElement(el) {
  el.style.opacity = '0.3';
  el.style.transform = 'scale(0.97)';
  el.style.transition = 'opacity 0.25s, transform 0.25s';
  requestAnimationFrame(() => requestAnimationFrame(() => {
    el.style.opacity = '1';
    el.style.transform = 'scale(1)';
  }));
}

// ── 타임라인 항목 추가 ─────────────────────────────────────────────────────────
function addTimelineEntry(t, bl, gem) {
  const track = document.getElementById('timelineTrack');
  const div   = document.createElement('div');
  div.style.cssText = `
    background: #161b22; border: 1px solid #30363d; border-radius: 6px;
    padding: 4px 8px; font-size: 0.72rem; min-width: 90px;
    border-left: 3px solid ${stageColor(gem.label)};
    animation: fadeIn 0.3s ease;
  `;
  div.innerHTML = `
    <div style="color:#8b949e;font-size:0.62rem">⏱ ${t}s</div>
    <div style="color:#3fb950;font-weight:600;line-height:1.3;margin-top:1px">
      ${gem.label}
    </div>
    <div style="color:#f85149;font-size:0.65rem;margin-top:1px">
      BL: ${bl.label}
    </div>
  `;
  track.appendChild(div);
  track.scrollTop = track.scrollHeight;
  // 최대 50개 유지
  while (track.children.length > 50) track.removeChild(track.firstChild);
}

function toggleRT() {
  rtActive ? stopRT() : startRT();
}

function startRT() {
  if (rtActive) return;
  rtActive       = true;
  frameBuffer    = [];
  captureCount   = 0;
  inferPending   = false;
  predHistory    = { baseline: [], agem: [] };
  lastGemLabel   = null;

  document.getElementById('rtBtn').textContent = '⏹ Stop captions';
  document.getElementById('rtBtn').style.background = 'rgba(248,81,73,0.85)';
  document.getElementById('subtitleOverlay').style.display = 'block';
  document.getElementById('timelineWrap').style.display = 'block';
  document.getElementById('anomalyBadge').style.display = 'block';

  videoPreview.play().catch(()=>{});
  rtTimer = setInterval(captureFrame, CAPTURE_INTERVAL);
}

function stopRT() {
  rtActive = false;
  clearInterval(rtTimer);
  rtTimer  = null;
  const btn = document.getElementById('rtBtn');
  if (btn) {
    btn.textContent = '▶ Live captions';
    btn.style.background = 'rgba(88,166,255,0.85)';
  }
}

videoPreview.addEventListener('ended', stopRT);
videoPreview.addEventListener('pause', () => { if (rtActive) stopRT(); });

// ── 예측 (파일 전체 업로드) ────────────────────────────────────────────────────
async function runPredict() {
  const f = fileInput.files[0];
  if (!f) return;

  predictBtn.disabled = true;
  predictBtn.textContent = '⏳ Analyzing…';
  document.getElementById('errorMsg').textContent = '';
  document.getElementById('resultCard').style.display = 'none';
  document.getElementById('resultFilename').textContent = f.name;

  const fd = new FormData();
  fd.append('file', f);

  try {
    const res = await fetch('/predict', { method: 'POST', body: fd });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || res.statusText);
    }
    const data = await res.json();
    renderResults(data);
  } catch (e) {
    document.getElementById('errorMsg').textContent = 'Error: ' + e.message;
  } finally {
    predictBtn.disabled = false;
    predictBtn.textContent = '🔍 Predict (Baseline vs A-GEM)';
  }
}

function renderResults(data) {
  const grid = document.getElementById('resultGrid');
  grid.innerHTML = '';

  for (const [key, label, cls] of [['baseline','Baseline','baseline'], ['agem','A-GEM (best)','agem']]) {
    const preds = data[key];
    const top1  = preds[0];
    const box   = document.createElement('div');
    box.className = `method-box ${cls}`;

    let html = `<div class="method-title">${label}</div>`;
    html += `<div class="top1-label">🏆 ${top1.label}</div>`;
    // 예시 뱃지
    if (top1.examples && top1.examples.length > 0) {
      html += `<div style="margin-bottom:10px;display:flex;flex-wrap:wrap;gap:4px">`;
      for (const ex of top1.examples.slice(0,4)) {
        html += `<span style="background:#21262d;color:#8b949e;font-size:0.72rem;padding:2px 7px;border-radius:10px">${ex}</span>`;
      }
      html += `</div>`;
    }
    // template (흐리게)
    html += `<div style="font-size:0.75rem;color:#484f58;margin-bottom:10px">${top1.template}</div>`;

    for (const p of preds) {
      const pct = (p.prob * 100).toFixed(1);
      html += `<div class="pred-item">
        <div class="pred-bar-wrap"><div class="pred-bar" style="width:${pct}%"></div></div>
        <div class="pred-label">${p.label}</div>
        <div class="pred-prob">${pct}%</div>
      </div>`;
    }
    box.innerHTML = html;
    grid.appendChild(box);
  }

  document.getElementById('resultCard').style.display = 'block';
}

// ── Forgetting 곡선 ───────────────────────────────────────────────────────────
async function loadForgetting() {
  const container = document.getElementById('chartContainer');
  try {
    const res = await fetch('/forgetting');
    if (!res.ok) {
      container.innerHTML = '<div class="error">No checkpoints — run save_checkpoints.py</div>';
      return;
    }
    const data = await res.json();
    const bl  = data.baseline;
    const gem = data.agem;

    container.innerHTML = '<canvas id="forgettingChart"></canvas>';
    const ctx = document.getElementById('forgettingChart').getContext('2d');
    new Chart(ctx, {
      type: 'line',
      data: {
        labels: bl.stages.map(s => `Stage ${s}`),
        datasets: [
          {
            label: 'Baseline — S1 Acc',
            data: bl.s1_acc,
            borderColor: '#f85149', backgroundColor: '#f8514920',
            tension: 0.3, pointRadius: 5, fill: false,
          },
          {
            label: 'A-GEM — S1 Acc',
            data: gem.s1_acc,
            borderColor: '#3fb950', backgroundColor: '#3fb95020',
            tension: 0.3, pointRadius: 5, fill: false,
          },
          {
            label: 'Baseline — Avg Acc',
            data: bl.avg_acc,
            borderColor: '#f85149', backgroundColor: 'transparent',
            borderDash: [5, 4], tension: 0.3, pointRadius: 3,
          },
          {
            label: 'A-GEM — Avg Acc',
            data: gem.avg_acc,
            borderColor: '#3fb950', backgroundColor: 'transparent',
            borderDash: [5, 4], tension: 0.3, pointRadius: 3,
          },
        ],
      },
      options: {
        responsive: true,
        plugins: {
          legend: { labels: { color: '#c9d1d9', font: { size: 12 } } },
          tooltip: { mode: 'index', intersect: false },
        },
        scales: {
          x: { ticks: { color: '#8b949e' }, grid: { color: '#21262d' } },
          y: {
            min: 0, max: 1,
            ticks: { color: '#8b949e', format: { style: 'percent' } },
            grid: { color: '#21262d' },
          },
        },
      },
    });

    // 통계 요약
    const bl_final  = bl.avg_acc[bl.avg_acc.length - 1];
    const gem_final = gem.avg_acc[gem.avg_acc.length - 1];
    const bl_s1drop  = bl.s1_acc[0]  - bl.s1_acc[bl.s1_acc.length - 1];
    const gem_s1drop = gem.s1_acc[0] - gem.s1_acc[gem.s1_acc.length - 1];
    document.getElementById('statsRow').innerHTML = `
      <div class="stat">BL Avg Acc <span>${(bl_final*100).toFixed(1)}%</span></div>
      <div class="stat">A-GEM Avg Acc <span>${(gem_final*100).toFixed(1)}%</span></div>
      <div class="stat">BL S1 Forgetting <span class="badge red">−${(bl_s1drop*100).toFixed(1)}%</span></div>
      <div class="stat">A-GEM S1 Forgetting <span class="badge green">${gem_s1drop > 0 ? '−' : '+'}${(Math.abs(gem_s1drop)*100).toFixed(1)}%</span></div>
      <div class="stat">Δ Avg <span class="badge green">+${((gem_final-bl_final)*100).toFixed(1)}%</span></div>
    `;

  } catch (e) {
    container.innerHTML = `<div class="error">Load failed: ${e.message}</div>`;
  }
}

// ── 새 동작 등록 (few-shot) ────────────────────────────────────────────────────
document.getElementById('enrollFiles').addEventListener('change', () => {
  const fs = document.getElementById('enrollFiles').files;
  document.getElementById('enrollFileLabel').textContent =
    fs.length ? `✓ ${fs.length} selected` : '';
});

async function runEnroll() {
  const label = document.getElementById('enrollLabel').value.trim();
  const files = document.getElementById('enrollFiles').files;
  const err   = document.getElementById('enrollError');
  const out   = document.getElementById('enrollResult');
  const btn   = document.getElementById('enrollBtn');
  err.textContent = ''; out.style.display = 'none';

  if (!label)        { err.textContent = 'Enter an action name'; return; }
  if (!files.length) { err.textContent = 'Select at least one example video'; return; }

  btn.disabled = true; btn.textContent = '⏳ Enrolling…';
  const fd = new FormData();
  fd.append('label', label);
  for (const f of files) fd.append('files', f);

  try {
    const res  = await fetch('/enroll', { method: 'POST', body: fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || res.statusText);
    out.style.display = 'block';
    out.innerHTML =
      `✅ Enrolled '<b>${data.label}</b>' — class #${data.new_class_id}, ${data.n_classes} classes total ` +
      `(${data.n_examples} examples). It now appears in the live caption 🎉<br>` +
      `<span style="color:#8b949e;font-size:0.78rem">checkpoint: ${data.checkpoint}</span>`;
    document.getElementById('enrollLabel').value = '';
    document.getElementById('enrollFiles').value = '';
    document.getElementById('enrollFileLabel').textContent = '';
    refreshClasses();
  } catch (e) {
    err.textContent = 'Error: ' + e.message;
  } finally {
    btn.disabled = false; btn.textContent = '➕ Enroll';
  }
}

// ── 등록된 동작 목록/초기화 ────────────────────────────────────────────────────
async function refreshClasses() {
  try {
    const res = await fetch('/classes');
    if (!res.ok) return;
    const data = await res.json();
    const wrap = document.getElementById('enrolledWrap');
    const list = document.getElementById('enrolledList');
    if (!data.enrolled.length) { wrap.style.display = 'none'; return; }
    wrap.style.display = 'block';
    list.innerHTML = '';
    for (const c of data.enrolled) {
      const chip = document.createElement('span');
      chip.style.cssText = 'background:#0d3321;color:#3fb950;border:1px solid #238636;' +
        'border-radius:14px;padding:3px 11px;font-size:0.78rem;font-weight:600';
      chip.textContent = `#${c.id} ${c.label}`;
      list.appendChild(chip);
    }
  } catch (_) {}
}

async function resetClasses() {
  if (!confirm('Remove all enrolled actions and revert to the base 48?')) return;
  try {
    await fetch('/reset_classes', { method: 'POST' });
    await refreshClasses();
  } catch (_) {}
}

refreshClasses();
loadForgetting();
</script>
</body>
</html>
"""
