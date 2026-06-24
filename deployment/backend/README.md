# Backend contract

Backend implementation language is intentionally open. The backend is optional;
if used, it is an authentication and model-manifest service only and must not run
model inference.

For a public demo, skip the backend and serve a static manifest next to the
frontend. Add a backend only when access control, signed model URLs, model rollout
rules, or audit logging are required.

## Runtime options

- **Bun/TypeScript Worker**: best fit for Cloudflare-native deployment. Use this
  for `/api/model-manifest` if deploying the frontend with Workers Static Assets.
- **FastAPI**: fine for a Docker/container origin or conventional server. Keep it
  behind the CDN and use it only for auth/manifest responses.
- **Rust Axum or Go**: also fine for a container backend, but more ceremony than
  needed for the initial manifest endpoint.

Do not add custom Diffie-Hellman/session crypto. TLS 1.3 already handles key
agreement. Use short-lived sessions or signed URLs if authorization is needed.

## Optional authenticated endpoint

```http
GET /api/model-manifest?model=focalnet_tiny_srf&variant=int8_qdq
Authorization: Bearer <token>
```

Responsibilities:

- authenticate the caller;
- choose the approved model/version for that caller/device;
- return a short-lived signed CDN URL or otherwise authorized immutable model URL;
- include the expected model hash, byte size, preprocessing config, and labels;
- never accept raw imagery or perform inference.

The frontend should treat this as the single authenticated backend endpoint.
Model bytes are delivered by the CDN, then cached on the client for browser-side
ONNX Runtime inference.

## Non-goals for the backend

- no image upload path;
- no CPU/GPU/NPU inference;
- no model training or quantization;
- no frontend-specific session state beyond authentication/authorization.

## Contract files

- `openapi.yaml` documents the initial HTTP shape.
- `model-manifest.schema.json` documents the manifest payload.
