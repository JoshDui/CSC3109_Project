# Cloudflare deployment playbook

Primary target for the edge-inference demo:

```text
Workers Static Assets  -> React/Vite app + ONNX Runtime Web wasm
Cloudflare R2          -> versioned quantized ONNX model artifacts
Cloudflare CDN edge    -> HTTP/3, cache, compression, CORS
```

## Why split frontend and models?

Workers Static Assets are convenient for the SPA, but individual static asset
files are limited to 25 MiB. The FocalNet INT8 QDQ ONNX artifact is slightly
larger than that and the FP32 artifact is much larger, so model files belong in
R2 behind a custom domain.

## Frontend hosting

From `deployment/frontend`:

```bash
bun install
bun run build:cdn
wrangler deploy --config ../cloudflare/wrangler.jsonc
```

The scaffold config uses `assets.not_found_handling = "single-page-application"`
so Vite SPA routes fall back to `index.html`.

## Model hosting

Recommended layout in R2:

```text
models/
  focalnet_tiny_srf/
    20260624/
      focalnet_tiny_srf_int8_qdq.onnx
      focalnet_tiny_srf_fp32.onnx
      model-manifest.json
```

Use an R2 custom domain such as `models.example.com`, not the `r2.dev` URL for
production. Configure CORS for the frontend origin and a cache rule that caches
model files aggressively.

## Headers and cache rules

For versioned model paths:

```text
Cache-Control: public, max-age=31536000, immutable
Content-Type: application/octet-stream
Accept-Ranges: bytes
```

For manifests that can change:

```text
Cache-Control: no-cache
Content-Type: application/json
```

Cloudflare caches by default only for known file extensions. Create a Cache Rule
for the model hostname/path to cache everything under `/models/*`, or serve model
artifacts with origin cache headers and an extension/path rule that Cloudflare
matches.

## Compression

Cloudflare can deliver gzip, Brotli, or Zstandard to visitors depending on plan
and rules. Quantization is the biggest win; compression is a second pass. Add a
Compression Rule for model paths if `.onnx` responses are not compressed by
default.

Do not shard models into MTU-sized chunks manually. Let HTTP/3/QUIC and the CDN
handle transport segmentation.

## Optional auth

If model URLs should not be public, add one Bun/TypeScript Worker route:

```text
GET /api/model-manifest
```

It should authenticate the caller and return a short-lived signed R2/CDN URL or
authorized manifest. It should not proxy the full ONNX response unless there is a
specific access-control reason.
