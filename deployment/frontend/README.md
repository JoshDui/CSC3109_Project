# Frontend

This React/Vite app is the user interface for the CSC3109 deployment container.

The current deployment flow is backend inference:

```text
GET /models -> choose packaged model -> POST /predict -> FastAPI backend -> ONNX Runtime -> JSON result
```

The frontend is built into static files by the root Dockerfile and served by the FastAPI runtime image. It displays the two packaged deployment models, predicted class, confidence, per-class scores, preprocessing note, and the ONNX execution provider reported by the backend. The top-right endpoint badge is informational; the `Run prediction` button is what sends the image to `POST /predict`.

## Commands

```bash
bun install
bun run dev
bun run check
bun run build
```

`bun run sync:assets` remains available only for the older browser-side ONNX experiment. It is not part of the final Docker deployment path.

## Model selector

The model cards come from the backend `/models` endpoint, not from `public/models.json`. The current deployment intentionally ships two runnable models: `resnet18_finetuned` and `custom_cnn_small_int8`. Additional cards should only be added when their ONNX artifacts are copied into `deployment/backend/models/` and verified against the backend response format.
