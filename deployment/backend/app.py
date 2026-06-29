from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
import subprocess
from typing import Annotated, Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from inference.preprocess import load_rgb_image
from inference.predictor import OnnxImageClassifier


APP_DIR = Path(__file__).resolve().parent
MODELS_DIR = APP_DIR / "models"
FRONTEND_DIST = Path(__file__).resolve().parents[1] / "frontend" / "dist"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024

predictor: OnnxImageClassifier | None = None
gpu_status: dict[str, Any] = {"available": False, "message": "not checked"}


@asynccontextmanager
async def lifespan(_: FastAPI):
    global predictor, gpu_status
    gpu_status = detect_nvidia_gpu()
    predictor = OnnxImageClassifier(MODELS_DIR, prefer_cuda=bool(gpu_status["available"]))
    yield


app = FastAPI(title="CSC3109 Aerial Image Classifier", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "model_loaded": predictor is not None,
        "gpu_status": gpu_status,
        "execution_providers": predictor.requested_providers if predictor else [],
    }


@app.get("/models")
def models() -> dict[str, object]:
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model registry is not loaded.")
    return predictor.list_models()


@app.post("/predict")
async def predict(
    file: Annotated[UploadFile, File(description="Aerial image to classify")],
    model_id: Annotated[str | None, Form(description="Optional model ID from /models")] = None,
) -> dict[str, object]:
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Upload must be an image file.")

    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="Uploaded image is empty.")
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Uploaded image is larger than 10 MiB.")

    try:
        image = load_rgb_image(payload)
        return predictor.predict(image, model_id=model_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


if (FRONTEND_DIST / "assets").exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="assets")


@app.get("/")
def index() -> FileResponse:
    return frontend_file("index.html")


@app.get("/{path:path}")
def frontend_fallback(path: str) -> FileResponse:
    candidate = FRONTEND_DIST / path
    if path and candidate.exists() and candidate.is_file():
        return FileResponse(candidate)
    return frontend_file("index.html")


def frontend_file(name: str) -> FileResponse:
    path = FRONTEND_DIST / name
    if not path.exists():
        raise HTTPException(status_code=404, detail="Frontend build not found.")
    return FileResponse(path)


def detect_nvidia_gpu() -> dict[str, object]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return {"available": False, "message": "nvidia-smi unavailable; CPU fallback active"}

    output = (result.stdout or result.stderr).strip()
    if result.returncode == 0 and output:
        return {"available": True, "message": output}
    return {"available": False, "message": "no NVIDIA GPU visible; CPU fallback active"}
