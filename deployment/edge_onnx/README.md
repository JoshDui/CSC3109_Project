# Edge ONNX model serving path

This path sits **next to** the containerized FastAPI deployment. Its primary use
is a fast local Docker/Caddy smoke demo for browser-side ONNX inference.

Use it when the React Local mode needs model artifacts from Caddy or an optional
CDN/static model host. The backend container also packages the same HETMCL and
CG-AF ONNX files for Web mode `/predict`.

## Models in scope

```text
HETMCL INT8 QDQ
model/hetmcl_lite/onnx/hetmcl_lite_best_stop_int8_qdq.onnx

Semantic-Guided CG-AF INT8 QDQ
model/semantic_guided_cgaf_onnx_int8_fullcalib_minmax_20260616/semantic_guided_cgaf_fft_int8_qdq_fullcalib_minmax.onnx
```

## Self-hosted Caddy model CDN

Stage the ONNX files under a model root such as:

```text
/srv/edge-models/models/hetmcl_lite_best_stop_int8_qdq.onnx
/srv/edge-models/models/semantic_guided_cgaf_fft_int8_qdq_fullcalib_minmax.onnx
```

For Brotli on the self-hosted path, place precompressed sidecars beside the raw
ONNX files:

```text
/srv/edge-models/models/hetmcl_lite_best_stop_int8_qdq.onnx.br
/srv/edge-models/models/semantic_guided_cgaf_fft_int8_qdq_fullcalib_minmax.onnx.br
```

Caddy serves those sidecars with:

```caddyfile
file_server {
	precompressed br gzip
}
```

Caddy does not create `.br` files; it selects them when they already exist and
the browser sends `Accept-Encoding: br`.

Validate the model host with:

```bash
curl -I -H 'Accept-Encoding: br' \
  https://models.example.com/models/hetmcl_lite_best_stop_int8_qdq.onnx
```

Expected headers include:

```text
Content-Encoding: br
Access-Control-Allow-Origin: *
Cross-Origin-Resource-Policy: cross-origin
Cache-Control: public, max-age=31536000, immutable
```

## Cloudflare R2 + CDN

For Cloudflare, upload the raw `.onnx` objects to R2 and expose them through a
custom domain such as `https://models.example.com/models/...`.

Do not require Brotli sidecars for this path. Let Cloudflare handle CDN,
protocol, and compression behavior.

Minimal upload flow using Wrangler:

```bash
cd deployment/edge_onnx
cp .env.cloudflare.example .env.cloudflare.local
# edit R2_BUCKET / R2_PREFIX / VITE_ONNX_MODEL_BASE_URL as needed
set -a && . ./.env.cloudflare.local && set +a
sh upload-r2.sh
cd ../frontend
VITE_ONNX_MODEL_BASE_URL=https://models.example.com/models bun run build
```

Expose the bucket through a Cloudflare custom domain matching
`VITE_ONNX_MODEL_BASE_URL`. The app does not need Workers for this path.

## Linking to the existing app

The containerized deployment remains the backend/frontend path:

```text
browser -> Caddy/app host -> root Dockerfile container -> POST /predict
```

The ONNX edge path is same-host in the local smoke stack:

```text
browser -> Caddy -> /edge-models/models/*.onnx
```

The React frontend includes the ONNX Runtime Web scaffold. Its **Local**
inference mode fetches ONNX files from a CDN/static model host, caches the bytes
in the browser Cache API where available, then runs inference in the browser.
The **Web** mode keeps using the backend `POST /predict` flow.

Choose the model host at build/deploy time:

```bash
# Same-host Caddy model CDN
VITE_ONNX_MODEL_BASE_URL=/edge-models/models bun run build

# Cloudflare R2/CDN custom domain
VITE_ONNX_MODEL_BASE_URL=https://models.example.com/models bun run build
```

`models.edge.example.json` shows the same artifact names without baking in a
specific host. The backend `/predict` flow does not need these edge model URLs.

## Local smoke stack

Run the FastAPI/frontend container behind Caddy and serve ONNX edge models from
the same local host:

```bash
docker compose -f deployment/edge_onnx/compose.local.yaml up --build
```

Open:

```text
http://127.0.0.1:8090
```

Smoke checks:

```bash
curl -I http://127.0.0.1:8090/
curl http://127.0.0.1:8090/health
curl http://127.0.0.1:8090/models
curl -I http://127.0.0.1:8090/edge-models/models/hetmcl_lite_best_stop_int8_qdq.onnx
curl -F "file=@data/raw/val/bridge/bridge722.jpg" \
  -F "model_id=hetmcl_lite_int8" \
  http://127.0.0.1:8090/predict
```

Stop it with:

```bash
docker compose -f deployment/edge_onnx/compose.local.yaml down
```

The named Docker volume keeps staged model files between runs. Use `down -v` to
remove it.
