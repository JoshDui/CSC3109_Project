# Docker fallback

Docker is scoped to static frontend delivery for local smoke tests or a simple
static origin.

Build the React/Vite frontend image from the repository root with the deployment
folder as the build context:

```bash
docker build -f deployment/docker/frontend.Dockerfile -t csc3109-edge-frontend deployment
docker run --rm -p 8080:8080 csc3109-edge-frontend
```

The runtime image serves the built React app with Caddy. It is suitable as an
origin behind a CDN. For production-scale serving, large ONNX models can be
served from separate object storage/CDN URLs referenced by `models.json`.

Recommended CDN headers:

```text
/assets/*  Cache-Control: public, max-age=31536000, immutable
/models/*  Cache-Control: public, max-age=31536000, immutable
/ort/*     Cache-Control: public, max-age=31536000, immutable
/index.html Cache-Control: no-cache
```
