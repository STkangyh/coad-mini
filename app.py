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
from src.models.fecam_head import FeCAMHead

# ── 설정 ─────────────────────────────────────────────────────────────────────
CKPT_DIR    = Path("checkpoints")
ENROLLED_CKPT   = CKPT_DIR / "agem_enrolled.pt"       # few-shot 으로 확장된 모델 (persist)
ENROLLED_LABELS = CKPT_DIR / "class_labels_enrolled.json"
FECAM_CKPT          = CKPT_DIR / "fecam_head.npz"            # base 48-class FeCAM head
FECAM_ENROLLED_CKPT = CKPT_DIR / "fecam_head_enrolled.npz"   # + user-enrolled classes
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

# ── FeCAM head (backprop-free classifier — beats the GRU head on all metrics;
#    see reports/cpu_friendly_methods_result.md). Enrollment = one mean vector,
#    so registering a new action is instant and cannot forget existing classes.
def _load_fecam():
    if FECAM_ENROLLED_CKPT.exists():
        return FeCAMHead.load(FECAM_ENROLLED_CKPT)
    if FECAM_CKPT.exists():
        return FeCAMHead.load(FECAM_CKPT)
    return None


_fecam_head = _load_fecam()
if _fecam_head is not None:
    print(f"✓ FeCAM head loaded (n_classes={_fecam_head.n_classes}, "
          f"pooling={_fecam_head.pooling}, dim={_fecam_head.feature_dim})")
else:
    print("⚠ FeCAM head not found — run dev/build_fecam_head.py (demo works without it)")


def fecam_predict(feat: torch.Tensor, top_k: int = 5):
    """(1, T, D) feature window -> top-k results in the same format as predict()."""
    emb = _fecam_head.window_to_embedding(feat.cpu().numpy())
    s = _fecam_head.scores(emb[None, :])[0]
    active = _fecam_head.counts > 0
    e = np.exp((s - s[active].max()) * 0.5)
    e[~active] = 0.0
    probs = e / (e.sum() + 1e-12)
    top_ids = s.argsort()[::-1][:top_k]
    out = []
    for i in top_ids:
        if not active[i]:
            continue
        entry = _labels.get(str(int(i)), {})
        if isinstance(entry, dict):
            template = entry.get("template", f"class_{i}")
            label = entry.get("label", template)
            examples = entry.get("examples", [])
        else:
            template = label = str(entry)
            examples = []
        out.append({"class_id": int(i), "template": template, "label": label,
                    "examples": examples, "prob": float(probs[i])})
    return out


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
        "fecam_ready": _fecam_head is not None,
        "fecam_n_classes": _fecam_head.n_classes if _fecam_head else 0,
        "fecam_pooling": _fecam_head.pooling if _fecam_head else None,
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

    resp = {
        "filename": file.filename,
        "baseline": predict(_bl_model,  feat, _labels),
        "agem":     predict(_gem_model, feat, _labels),
    }
    if _fecam_head is not None:
        resp["fecam"] = fecam_predict(feat)
    return JSONResponse(resp)


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

    resp = {
        "baseline": top1(_bl_model),
        "agem":     top1(_gem_model),
    }
    if _fecam_head is not None:
        resp["fecam"] = fecam_predict(feat, top_k=3)
    return JSONResponse(resp)


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
    global _gem_model, _gem_ckpt, _labels, _fecam_head
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

    # 7. FeCAM 프로토타입 등록 — 평균 벡터 1개 계산이라 밀리초 단위이며,
    #    기존 클래스 통계를 전혀 건드리지 않아 망각이 구조적으로 불가능.
    fecam_ms = None
    if _fecam_head is not None:
        import time as _time
        t0 = _time.perf_counter()
        _fecam_head.enroll_class(new_class_id, [w.numpy() for w in new_windows])
        _fecam_head.save(FECAM_ENROLLED_CKPT)
        fecam_ms = (_time.perf_counter() - t0) * 1000

    return JSONResponse({
        "status": "ok",
        "new_class_id": new_class_id,
        "label": label,
        "n_classes": model.num_classes,
        "n_examples": len(new_windows),
        "checkpoint": str(ENROLLED_CKPT),
        "labels_file": str(ENROLLED_LABELS),
        "live": True,
        "fecam_ms": fecam_ms,
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
    """등록한 동작을 모두 제거하고 기본 48-class 모델(GRU + FeCAM)로 되돌린다."""
    global _gem_model, _gem_ckpt, _labels, _fecam_head
    if not _ckpt_ready:
        raise HTTPException(503, "checkpoints not ready")
    _gem_model, _gem_ckpt = load_model(CKPT_DIR / "agem_48cls.pt")
    _labels = load_labels()
    for p in (ENROLLED_CKPT, ENROLLED_LABELS, FECAM_ENROLLED_CKPT):
        try:
            p.unlink()
        except FileNotFoundError:
            pass
    _fecam_head = _load_fecam()
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
# The page lives in static/index.html rather than as a 781-line string literal
# here: it is HTML/CSS/JS and belongs where an editor and a linter can see it.
# Read once at import so a missing file fails at startup, loudly, instead of
# turning every request for "/" into a 500.
STATIC_DIR = Path(__file__).parent / "static"
try:
    HTML_PAGE = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
except FileNotFoundError as e:                      # e.g. a partial deploy
    raise RuntimeError(
        f"UI template missing: {STATIC_DIR / 'index.html'}. It must ship with "
        f"app.py -- check the Dockerfile COPY list and any manual upload."
    ) from e


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML_PAGE
