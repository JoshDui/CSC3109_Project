# Frontend

This React/Vite app is the user interface for the CSC3109 deployment container.

The app has two inference modes:

```text
Web   -> GET /models -> choose packaged model -> POST /predict -> FastAPI backend -> ONNX Runtime -> JSON result
Local -> fetch ONNX from VITE_ONNX_MODEL_BASE_URL -> browser cache -> ONNX Runtime Web -> local result
```

The frontend is built into static files by the root Dockerfile and served by the
FastAPI runtime image. In the local smoke stack, Caddy sits in front of the same
container and serves ONNX artifacts from `/edge-models/models` for Local mode.
The deployment model is intentionally limited to Semantic-Guided CG-AF INT8.

## Commands

```bash
bun install
bun run dev
bun run check
bun run sync:assets
bun run build
```

For Local mode, choose the ONNX host at build/deploy time:

```bash
# Local Caddy smoke stack / same host
VITE_ONNX_MODEL_BASE_URL=/edge-models/models bun run build

# Optional Cloudflare R2/CDN custom domain
VITE_ONNX_MODEL_BASE_URL=https://models.example.com/models bun run build
```

## Model selector

Web mode model cards come from the backend `/models` endpoint. Local mode model
cards come from `public/models.json`; both expose only
`semantic_guided_cgaf_int8`. Additional cards should only be added when the
assignment scope, report, backend registry, edge ONNX catalog, and smoke tests
are updated together.

Run `bun run sync:assets` before a standalone frontend build that packages local
ONNX files. The command removes stale generated ONNX copies from
`public/models/` before copying CG-AF.
