# Deployment

The current submission deployment is a single Docker container that serves the React frontend and exposes backend inference endpoints.

```text
browser
  -> React/Vite frontend served by FastAPI
  -> GET /models for the model registry
  -> POST /predict with uploaded image + model_id
  -> ONNX Runtime inference on the backend
  -> JSON response: predicted_label, confidence, class_scores
```

This is intentionally scope-first for the CSC3109 marking requirements: the container proves that a final model can be served through an inference endpoint, accepts aerial image input, and returns a prediction with confidence scores.

## Runtime path

```text
repo root Dockerfile
  frontend-build stage: Bun + Vite builds deployment/frontend
  runtime stage: Python + FastAPI serves deployment/backend and frontend/dist

/app/backend/models/models.json
/app/backend/models/class_labels.json
/app/backend/models/resnet18_finetuned.onnx
/app/backend/models/custom_cnn_small_int8_qdq.onnx
```

The packaged deployment models are the ResNet18 fine-tuned transfer-learning classifier and the Custom CNN Small INT8 classifier. ResNet18 remains the active default; Custom CNN is included as a lightweight comparison model with the same 4-class output contract. The backend model registry can list additional showcase candidates later, but only entries whose ONNX file exists under `deployment/backend/models/` are runnable in the Docker build.

## Model selection

The frontend model picker is driven by:

```text
GET /models
```

Each registry entry reports whether it is packaged:

```json
{
  "id": "resnet18_finetuned",
  "display_name": "ResNet18 fine-tuned",
  "available": true,
  "role": "Packaged final baseline"
}
```

Prediction accepts a selected model ID:

```text
POST /predict
multipart form fields:
  file: image file
  model_id: model ID from /models
```

The current registry intentionally lists only the two packaged models. If another model is added later, copy its ONNX file to `deployment/backend/models/` and make sure its `models.json` entry has the correct tensor names and preprocessing settings; otherwise the frontend will mark it as `Not packaged` and the backend will reject direct calls with HTTP 404.

## GPU, CPU, and Mac behavior

The default Docker build uses CPU ONNX Runtime. That is intentional: it is the safest submission path for assessment machines, Windows laptops, Linux machines, and MacBooks where NVIDIA CUDA may not exist.

The container checks `nvidia-smi` on startup and reports the result from `/health`. If an NVIDIA GPU is visible and the installed ONNX Runtime package exposes `CUDAExecutionProvider`, the backend asks ONNX Runtime to use CUDA first and CPU second. Otherwise it uses `CPUExecutionProvider`. Every prediction response includes the actual `execution_provider`, so the runtime path is visible during grading.

CUDA is optional and NVIDIA-specific. A future GPU variant would need to run with Docker GPU support, for example `docker run --gpus all ...`, and use a CUDA-compatible image with `onnxruntime-gpu` plus matching NVIDIA runtime libraries. That is useful for performance testing, but it should not be required for the final submission because it would make the assessor environment more fragile.

MacBooks should be expected to run this container through `CPUExecutionProvider`. CUDA does not apply to Apple Silicon or Intel MacBooks without NVIDIA hardware. A native Apple acceleration path using Metal/CoreML would be a separate deployment route, not part of this single Docker container.

## Build and run

From the repository root:

```powershell
docker build -t csc3109-aerial-classifier .
docker run --rm -p 8080:8080 csc3109-aerial-classifier
```

Open the frontend at:

```text
http://localhost:8080
```

Health check:

```powershell
Invoke-RestMethod http://localhost:8080/health
```

List models:

```powershell
Invoke-RestMethod http://localhost:8080/models
```

Prediction endpoint with curl:

```powershell
curl.exe -F "file=@data\set 12\bridge\bridge001.jpg" -F "model_id=resnet18_finetuned" http://localhost:8080/predict
```

The prediction response has this shape:

```json
{
  "model_id": "resnet18_finetuned",
  "display_name": "ResNet18 fine-tuned",
  "predicted_label": "bridge",
  "confidence": 0.998,
  "class_scores": {
    "bridge": 0.998,
    "freeway": 0.001,
    "overpass": 0.001,
    "railway": 0.0
  },
  "inference_ms": 12.3,
  "execution_provider": "CPUExecutionProvider"
}
```

## Exporting the ONNX model

If `deployment/backend/models/resnet18_finetuned.onnx` is missing, export it from the local ResNet checkpoint before building the final image:

```powershell
.\.venv\Scripts\python.exe -m src.quantization.export_onnx_classifier `
  --checkpoint model/resnet18_finetune_last_block.pt `
  --output-dir deployment\backend\models `
  --onnx-fp32-output resnet18_finetuned.onnx `
  --exporter legacy_tracer `
  --device cpu
```

Do not copy the raw dataset or old training checkpoints into the Docker image. The image only needs the backend code, the frontend build, the two selected packaged ONNX model files, and the class-label/registry JSON files.

## What this replaces

Earlier planning discussed static-only hosting, Caddy, CDN delivery, Brotli, and browser-side ONNX. Those may be useful optimisations later, but they do not by themselves prove a containerised inference endpoint. For this submission path, the old Caddy/static Docker route is deprecated in favour of the root `Dockerfile`.
