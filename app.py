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

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse

from src.models.gru_detector import GRUDetector

# ── 설정 ─────────────────────────────────────────────────────────────────────
CKPT_DIR    = Path("checkpoints")
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


def load_labels() -> dict:
    p = CKPT_DIR / "class_labels.json"
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
            raise HTTPException(500, f"CLIP 로드 실패: {e}")
    return _clip_model, _clip_preprocess


def extract_frames_from_video(video_bytes: bytes, n_frames: int = N_FRAMES) -> np.ndarray:
    """ffmpeg로 영상에서 균등 간격 n_frames 추출 → (n_frames, H, W, 3) numpy"""
    try:
        from PIL import Image
    except ImportError:
        raise HTTPException(500, "pip install Pillow 필요")

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
        raise HTTPException(400, "영상에서 프레임 추출 실패. webm/mp4 형식인지 확인하세요.")

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
        raise HTTPException(500, "pip install Pillow 필요")

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

if _ckpt_ready:
    _bl_model,  _bl_ckpt  = load_model(CKPT_DIR / "baseline_48cls.pt")
    _gem_model, _gem_ckpt = load_model(CKPT_DIR / "agem_48cls.pt")
    _labels = load_labels()
    print("✓ Checkpoints loaded")
else:
    _bl_model = _gem_model = _bl_ckpt = _gem_ckpt = None
    _labels = load_labels()
    print("⚠ Checkpoints not found. Run save_checkpoints.py first.")


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
        raise HTTPException(400, "빈 파일")

    try:
        feat = video_to_feature(video_bytes)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"feature 추출 실패: {e}")

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
        raise HTTPException(400, "frames 없음")

    try:
        feat = frames_to_feature(payload.frames)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"feature 추출 실패: {e}")

    # 실시간이므로 top-1만 반환 (속도 우선)
    def top1(model):
        r = predict(model, feat, _labels, top_k=3)
        return r

    return JSONResponse({
        "baseline": top1(_bl_model),
        "agem":     top1(_gem_model),
    })


# ── 웹 UI ─────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
def index():
    return HTML_PAGE


HTML_PAGE = """<!DOCTYPE html>
<html lang="ko">
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
    <h2>📹 영상 업로드</h2>
    <!-- 업로드 전: 드롭존 -->
    <div class="upload-area" id="dropZone" onclick="document.getElementById('fileInput').click()">
      <input type="file" id="fileInput" accept="video/*,.webm,.mp4,.avi">
      <div class="icon">🎬</div>
      <div class="hint">클릭하거나 영상을 끌어다 놓으세요<br>webm · mp4 · avi 지원</div>
      <div class="filename" id="filenameLabel"></div>
    </div>
    <!-- 업로드 후: 인라인 플레이어 + 자막 오버레이 -->
    <div id="videoWrap" style="display:none;margin-top:12px;border-radius:8px;
         overflow:hidden;background:#000;position:relative;user-select:none">
      <video id="videoPreview" controls playsinline
             style="width:100%;max-height:340px;display:block;object-fit:contain"></video>

      <!-- ① 현재 행동 자막 (영상 위) -->
      <div id="subtitleOverlay" style="display:none;position:absolute;bottom:52px;
           left:0;right:0;pointer-events:none;padding:0 10px">
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
          ▶ 실시간 자막
        </button>
        <button onclick="resetUpload()"
                style="background:rgba(0,0,0,0.6);color:#fff;border:none;
                       border-radius:6px;padding:4px 10px;font-size:0.8rem;cursor:pointer">
          ✕ 다시 선택
        </button>
      </div>
      <canvas id="captureCanvas" style="display:none"></canvas>
    </div>

    <!-- ② 행동 변화 타임라인 (영상 아래) -->
    <div id="timelineWrap" style="display:none;margin-top:10px">
      <div style="font-size:0.75rem;color:#8b949e;margin-bottom:6px;
                  display:flex;align-items:center;gap:6px">
        ⏱ 행동 변화 타임라인
        <span style="font-size:0.68rem;color:#484f58">
          (A-GEM 예측이 바뀔 때마다 기록)
        </span>
      </div>
      <div id="timelineTrack" style="display:flex;gap:4px;flex-wrap:wrap;
           max-height:90px;overflow-y:auto;align-content:flex-start"></div>
    </div>
    <button class="btn" id="predictBtn" disabled onclick="runPredict()">
      🔍 Baseline vs A-GEM 예측
    </button>
    <div class="error" id="errorMsg"></div>
  </div>

  <!-- 예측 결과 -->
  <div class="card" id="resultCard" style="display:none">
    <h2>🎯 예측 결과 <span id="resultFilename" style="font-weight:400;color:#8b949e;font-size:0.8rem"></span></h2>
    <div class="result-grid" id="resultGrid"></div>
  </div>

  <!-- Forgetting 곡선 -->
  <div class="card">
    <h2>📉 Forgetting 곡선 <span style="font-weight:400;color:#8b949e;font-size:0.8rem">(Stage 1 accuracy over stages)</span></h2>
    <div id="chartContainer">
      <div class="spinner">데이터 로딩 중...</div>
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
  const url = URL.createObjectURL(file);
  videoPreview.src = url;
  dropZone.style.display = 'none';
  videoWrap.style.display = 'block';
  predictBtn.disabled = false;
}

function resetUpload() {
  stopRT();
  videoPreview.src = '';
  videoWrap.style.display = 'none';
  dropZone.style.display = 'block';
  filenameLabel.textContent = '';
  predictBtn.disabled = true;
  fileInput.value = '';
  document.getElementById('resultCard').style.display = 'none';
  document.getElementById('subtitleOverlay').style.display = 'none';
  document.getElementById('timelineWrap').style.display = 'none';
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
    if (!res.ok) return;
    const data = await res.json();
    storeAndRender(data);
  } catch(_) {}
  finally { inferPending = false; }
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

  document.getElementById('rtBtn').textContent = '⏹ 자막 중지';
  document.getElementById('rtBtn').style.background = 'rgba(248,81,73,0.85)';
  document.getElementById('subtitleOverlay').style.display = 'block';
  document.getElementById('timelineWrap').style.display = 'block';

  videoPreview.play().catch(()=>{});
  rtTimer = setInterval(captureFrame, CAPTURE_INTERVAL);
}

function stopRT() {
  rtActive = false;
  clearInterval(rtTimer);
  rtTimer  = null;
  const btn = document.getElementById('rtBtn');
  if (btn) {
    btn.textContent = '▶ 실시간 자막';
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
  predictBtn.textContent = '⏳ 분석 중...';
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
    document.getElementById('errorMsg').textContent = '오류: ' + e.message;
  } finally {
    predictBtn.disabled = false;
    predictBtn.textContent = '🔍 Baseline vs A-GEM 예측';
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
      container.innerHTML = '<div class="error">체크포인트 없음 — save_checkpoints.py 실행 필요</div>';
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
    container.innerHTML = `<div class="error">로드 실패: ${e.message}</div>`;
  }
}

loadForgetting();
</script>
</body>
</html>
"""
