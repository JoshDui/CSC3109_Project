# Deployment

The Streamlit deployment path is superseded by a browser-side edge inference
demo:

```text
deployment/
  frontend/   React + Vite + ONNX Runtime Web app
  docker/     optional static frontend container for local smoke tests
```

## Architecture

The demo is a static frontend: it fetches a lightweight `models.json` catalog,
downloads a selected ONNX artifact, and performs inference locally with ONNX
Runtime Web. Images stay in the browser; there is no server-side inference path.

```text
browser client
  -> static models.json catalog
  -> local static model or external model URL
  -> browser cache
  -> ONNX Runtime Web inference on edge device
```

This framing keeps RF airtime and server compute low for constrained aerial
deployment scenarios such as drones, aircraft, satellites, or field terminals.

See `edge-inference-benchmarks.md` for CPU ONNX latency vs delivery-size
measurements across all exported models and which are practical for edge.

## Local frontend scaffold

```bash
cd deployment/frontend
bun install
bun run dev
```

Build the static React app:

```bash
bun run build
```

## Docker static frontend fallback

From the repository root:

```bash
docker build -f deployment/docker/frontend.Dockerfile -t csc3109-edge-frontend deployment
docker run --rm -p 8080:8080 csc3109-edge-frontend
```

The Docker container serves the static frontend only. Use it for local smoke
tests or as a simple static origin. Large ONNX artifacts can be served from a
separate object-storage/CDN host by pointing `public/models.json` URLs there.
