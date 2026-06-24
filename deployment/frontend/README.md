# Frontend scaffold

React + Vite scaffold for browser-side ONNX Runtime inference.

## Commands

```bash
bun install
bun run dev
bun run check
bun run build:cdn
```

`build:cdn` emits the production Vite bundle and creates `.br` / `.gz`
sidecars for CDN or static-server delivery.

To deploy the static SPA to Cloudflare Workers Static Assets after building:

```bash
wrangler deploy --config ../cloudflare/wrangler.jsonc
```

Do not place large ONNX files in the Vite `public/` directory. Host models from
R2/CDN and point the manifest `model.url` at that location.

## Runtime flow

1. Fetch the authenticated model manifest from `VITE_MODEL_MANIFEST_URL`.
2. Configure ONNX Runtime Web from manifest runtime settings.
3. Load the CDN-served ONNX model into a browser-side inference session.
4. Cache model bytes in a later phase with Cache API or IndexedDB.

The checked-in `public/model-manifest.example.json` is a local development
placeholder. Production should return the manifest from the backend contract in
`deployment/backend/`.
