# Docker

This folder is for the static web deployment container.

Planned approach:

- Build the React frontend.
- Copy the production build into a small Caddy image.
- Serve the site and model registry over HTTP.
- Keep inference in the browser through ONNX Runtime Web.

Expected command shape once the Dockerfile exists:

```powershell
docker build -t csc3109-aerial-classifier:latest -f deployment/docker/Dockerfile .
docker run --rm -p 8080:80 csc3109-aerial-classifier:latest
```

Then open:

```text
http://localhost:8080
```

The container should not assume the assessor has an NVIDIA GPU. Training scripts
may use CUDA locally, but the deployment image should work as a CPU/browser-side
static app.
