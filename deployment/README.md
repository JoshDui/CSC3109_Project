# Deployment

The Streamlit deployment path is superseded by a browser-side edge inference
deployment scaffold:

```text
deployment/
  frontend/   React + Vite + ONNX Runtime Web scaffold
  cloudflare/ Workers Static Assets + R2 model-hosting notes
  backend/    optional authenticated model-manifest contract
  docker/     fallback static frontend container and CDN-origin notes
```

## Architecture

The backend is optional and does **not** perform inference. The simplest public
demo can use a static model manifest. If access control is required, one small
authenticated endpoint can return an authorized model manifest. In both cases,
the frontend downloads the versioned ONNX artifact from the CDN and performs
inference locally with ONNX Runtime Web.

```text
browser client
  -> static manifest or optional /api/model-manifest
  -> versioned CDN/R2 model URL
  -> browser cache / IndexedDB
  -> ONNX Runtime Web inference on edge device
```

This framing keeps RF airtime and backend compute low for constrained aerial
deployment scenarios such as drones, aircraft, satellites, or field terminals.

## Local frontend scaffold

```bash
cd deployment/frontend
bun install
bun run dev
```

Build and precompress static assets:

```bash
bun run build:cdn
```

## Cloudflare-first deployment

Use Workers Static Assets for the React/Vite app and Cloudflare R2 for ONNX
model artifacts:

```text
app.example.com     Workers Static Assets: HTML/CSS/JS/ORT wasm
models.example.com  R2 custom domain: versioned .onnx files
```

Cloudflare Workers Static Assets currently have a 25 MiB individual file limit,
so the model files should not be uploaded as frontend static assets. Store ONNX
artifacts in R2, connect a custom domain, enable HTTP/3, configure CORS for the
frontend origin, and use cache rules for long-lived immutable model caching.

See `deployment/cloudflare/` for a `wrangler.jsonc` scaffold and R2/CORS notes.

## Docker static frontend fallback

From the repository root:

```bash
docker build -f deployment/docker/frontend.Dockerfile -t csc3109-edge-frontend deployment
docker run --rm -p 8080:8080 csc3109-edge-frontend
```

The Docker container serves the static frontend only. Use it for local smoke
tests or as a non-Cloudflare origin fallback. Production model delivery should
still come from CDN/R2 rather than through the container.
