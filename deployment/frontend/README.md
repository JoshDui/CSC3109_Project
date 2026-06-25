# Frontend — browser-side ONNX inference

React 19 + Vite 8 + TypeScript SPA that runs quantized aerial-image classifiers
and a semantic overlay model fully in the browser with ONNX Runtime Web. Images
never leave the client; only the model is transferred over the network, cached
on device, and executed locally. This keeps RF airtime and server compute low
for constrained aerial deployments (drones, aircraft, satellites, field
terminals).

## Commands

```bash
bun install
bun run dev          # syncs assets, starts dev server on a random free port
bun run check        # tsc type check
bun run build        # production Vite build
bun run sync:assets  # copy models into public/ (auto-run by dev/build)
```

`bun run dev` prints the chosen port when it starts. If you want a fixed port
for one-off use, run:

```bash
bunx vite --host 0.0.0.0 --port 5173
```

## Model catalog (`public/models.json`)

The SPA is driven by a lightweight static catalog — no server API or manifest
versioning required for the static dev demo. `public/models.json` is deliberately
small: it contains shared labels plus each model's ID, display name,
description, and artifact URL. ONNX runtime details that are tied to the
frontend implementation — input/output tensor names, preprocessing, payload
size, accuracy, and execution-provider preference — live in
`src/onnx/modelRegistry.ts`.

| Model | Payload (INT8) | Val acc | Interp | Notes |
| --- | --- | --- | --- | --- |
| `custom_cnn_small_int8` | 1.3 MB | 96.25% | bilinear | Smallest payload; best for scarce bandwidth |
| `vit_dinov2_lora_int8` | 23 MB | 98.25% | bicubic | Smallest transformer payload |
| `focalnet_tiny_srf_int8` | 28 MB | 99.50% | bicubic | Best accuracy-per-byte; recommended default |
| `swin_tiny_lora_int8` | 32 MB | 99.25% | bicubic | LoRA Swin; high accuracy |
| `clip_fft_int8` | 329 MB | 98.00% | bicubic | Fine-tuned CLIP image classifier; large payload comparison model |
| `semantic_guided_cgaf_int8` | 28 MB | n/a | bilinear | Hidden overlay model launched by the Overlay toggle |

Most classifiers stretch-resize to 224×224, `/255`, ImageNet mean/std, NCHW
`[1,3,224,224]`, input `images` → output `logits[1,4]`, labels
`bridge, freeway, overpass, railway`. CLIP uses its own `pixel_values` input,
shortest-edge center crop, and CLIP mean/std. The Semantic-Guided CG-AF model
uses 512×512 image input and returns `segmentation_logits` plus scene logits.
The transformer INT8 artifacts (swin, vit) were produced by the same QDQ static
quantization used elsewhere — see `../edge-inference-benchmarks.md`.

A **Benchmark latency (×30)** button runs repeated inference on the loaded image
and reports the real in-browser median latency / fps for the active execution
provider. The Semantic-Guided CG-AF model also returns `segmentation_logits`,
which the UI renders as a toggleable overlay. It is intentionally hidden from
the classifier picker: clicking **Overlay** loads/runs it lazily for the current
image, while **Raw** hides the already-computed overlay.

### Preprocessing parity

Browser preprocessing mirrors the Python eval pipeline
(`src/data/image_classification.py::build_eval_transform`): a direct stretch
resize to 224×224 (no center crop) plus per-model interpolation and ImageNet
normalization. Canvas resampling cannot bit-match PIL's bilinear/bicubic kernel,
but the sub-pixel difference does not change argmax for this 4-class task
(verified: custom_cnn 39/40, focalnet 40/40 on a val sample using the mirrored
pipeline).

> CLIP (`pixel_values` input, CLIP normalization, shortest-edge + center crop)
> is included as a fine-tuned image classifier. It is not prompt-based CLIPSeg;
> the 344 MB payload is useful for comparison but poor for RF-constrained edge
> delivery.

## Execution providers (incl. NPU)

`src/onnx/executionProvider.ts` feature-detects providers and the session loader
attempts each in the model's `preferredEP` order, falling back to `wasm`:

- `wasm` — CPU, threaded when cross-origin isolated (COOP/COEP set in Vite).
  Reliable default for INT8 QDQ graphs.
- `webgpu` — requires `navigator.gpu`.
- `webnn` — requires `navigator.ml`; configured with `deviceType: "npu"` to
  target an on-device NPU where supported (experimental, browser/hardware-gated).

## Assets (static dev)

`bun run sync:assets` copies quantized models from the repo `model/` directories
into `public/models/`. These frontend copies are git-ignored and reproduced on
demand, so large binaries are not duplicated. It runs automatically before `dev`
and `build`. For local static serving, model URLs are fixed under
`/models/<artifact>.onnx`; for production, change the `url` values in
`models.json` to external model-host URLs if needed.

## Production hosting

The build output is a static Vite app. Serve `dist/` from any static host. Large
ONNX artifacts do not need to be bundled with the app; point each catalog `url`
at object storage or a CDN-backed model host when the model is too large for the
static frontend host. Re-introduce integrity (`sha256`) / signed-URL fields only
if access control is needed.
