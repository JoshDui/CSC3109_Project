# Deployment

The current deployment is a local Docker demo that serves the React frontend,
exposes backend inference through FastAPI, and can optionally sit behind Caddy
for same-host ONNX model delivery.

```text
browser
  -> React/Vite frontend served by FastAPI or Caddy reverse proxy
  -> GET /models for the model registry
  -> POST /predict with uploaded image + model_id
  -> ONNX Runtime inference on the backend
  -> JSON response: predicted_label, confidence, class_scores

browser local mode
  -> downloads HETMCL/CG-AF ONNX from /edge-models/models or a configured CDN
  -> caches the model in the browser
  -> runs ONNX Runtime Web locally
```

This is intentionally scope-first for the CSC3109 marking requirements: the
container proves that selected final models can be served through an inference
endpoint, accepts aerial image input, and returns predictions with confidence
scores. The Caddy/edge path is primarily for local smoke testing of browser ONNX
delivery; VPS/TLS/QUIC or Cloudflare R2/CDN are optional extensions when a real
model-host URL is provided.

## Runtime path

```text
repo root Dockerfile
  frontend-build stage: Bun + Vite builds deployment/frontend
  runtime stage: Python + FastAPI serves deployment/backend and frontend/dist

/app/backend/models/models.json
/app/backend/models/class_labels.json
/app/backend/models/hetmcl_lite_best_stop_int8_qdq.onnx
/app/backend/models/semantic_guided_cgaf_fft_int8_qdq_fullcalib_minmax.onnx
```

The packaged deployment models are HETMCL-lite INT8 and Semantic-Guided CG-AF
INT8. HETMCL remains the active default for backend `/predict`; CG-AF uses its
`scene_logits` output for the same 4-class response contract. The same two ONNX
artifacts are also the browser-side Local mode models.

## Model selection

The frontend model picker is driven by:

```text
GET /models
```

Each registry entry reports whether it is packaged:

```json
{
  "id": "hetmcl_lite_int8",
  "display_name": "HETMCL-lite ResNet18 Hybrid (INT8)",
  "available": true,
  "role": "Packaged backend and edge classifier"
}
```

Prediction accepts a selected model ID:

```text
POST /predict
multipart form fields:
  file: image file
  model_id: model ID from /models
```

The current registry intentionally lists only HETMCL-lite INT8 and
Semantic-Guided CG-AF INT8. If another model is added later, copy its ONNX file
to `deployment/backend/models/` and make sure its `models.json` entry has the
correct tensor names and preprocessing settings; otherwise the frontend will
mark it as `Not packaged` and the backend will reject direct calls with HTTP
404.

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

For a Cloudflare/R2 model host, pass the ONNX base URL at image build time:

```powershell
docker build --build-arg VITE_ONNX_MODEL_BASE_URL=https://models.example.com/models -t csc3109-aerial-classifier .
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
curl.exe -F "file=@data\raw\val\bridge\bridge722.jpg" -F "model_id=hetmcl_lite_int8" http://localhost:8080/predict
```

The prediction response has this shape:

```json
{
  "model_id": "hetmcl_lite_int8",
  "display_name": "HETMCL-lite ResNet18 Hybrid (INT8)",
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

## Local Caddy + browser ONNX smoke path

For the full local demo, run the FastAPI/frontend container behind Caddy and
serve the same HETMCL/CG-AF ONNX artifacts from `/edge-models/models`:

```bash
docker compose -f deployment/edge_onnx/compose.local.yaml up --build
```

Open `http://127.0.0.1:8090`. Web mode uses backend `/predict`; Local mode
downloads ONNX from Caddy, caches it in the browser, and runs ONNX Runtime Web.

Use:

```bash
docker compose -f deployment/edge_onnx/compose.local.yaml down
```

to stop the stack.

Do not copy the raw dataset or old training checkpoints into the Docker image.
The image only needs the backend code, frontend build, the two selected ONNX
model files, and the class-label/registry JSON files.

## Optional VPS/CDN path

The local demo is the primary target. If a VPS/domain is available, put Caddy in
front of the same app container and use the same `/edge-models/models` route for
ONNX delivery. Caddy can obtain ACME/Let's Encrypt certificates and serve HTTP/3
when TCP/443 and UDP/443 are open.

Cloudflare is optional: upload raw ONNX files to R2 with a custom CDN domain and
build the frontend with `VITE_ONNX_MODEL_BASE_URL=https://models.example.com/models`.
Cloudflare handles CDN/protocol/compression behavior for that path.
