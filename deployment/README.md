# Deployment

The current deployment direction is a static React web app served from Docker.
The browser should run inference with ONNX Runtime Web, so the container only
needs to serve static files, model assets, and metadata. CUDA is useful for
training, but it should not be required for the submitted web demo.

Target structure:

```text
deployment/
  frontend/
    public/
      models/
        models.json
    src/
  docker/
    Dockerfile
    Caddyfile
```

Submission goal:

- The assessor opens one webpage.
- The page accepts an aerial image upload.
- The page loads a selected ONNX model from the static model registry.
- The page returns predicted class, top confidence, and per-class confidence.
- The Docker image serves the built frontend with Caddy.

Out of scope for the primary submission path:

- Flask or Streamlit as the main app.
- Server-side CUDA inference.
- Cloud hosting as a required grading dependency.

A backend can still be added later if the project scope changes, but the safer
submission target is a reproducible local Docker container that serves the static
web app.
