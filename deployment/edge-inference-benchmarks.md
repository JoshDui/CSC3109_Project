# Edge inference benchmarks (CPU ONNX)

CPU-only ONNX Runtime latency for every exported model in the repo, used to
decide which artifacts are practical for browser-side / edge delivery.

## Method

- Runtime: `onnxruntime` CPUExecutionProvider, sequential exec mode.
- Input: single image (batch = 1), random tensor at each model's native input
  shape; 5 warmup + 30 timed runs (15 for the 512² model); median reported.
- Threads: benchmarked at **1 thread** (single-core lower bound) and **4
  threads** (typical small edge SoC).
- Host: 24-core desktop CPU. **Not edge hardware** — treat these as optimistic.
  A drone/satellite ARM core is typically 3–10× slower single-threaded, and the
  in-browser WASM (SIMD) runtime the SPA uses is roughly **2–3× slower than
  native** CPU ORT. Multiply accordingly for real deployment.
- Delivery size = `.onnx` + any `.onnx.data` external-weights sidecar.

## Results

| Model | Task | Input | Delivery | 1-thr ms | 4-thr ms | fps@4 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| custom_cnn_small INT8 | classify | 224² | **1.3 MB** | 64.3 | 21.3 | 46.8 |
| custom_cnn_small FP32 | classify | 224² | 4.6 MB | 46.6 | 13.1 | 76.4 |
| focalnet_tiny_srf INT8 | classify | 224² | 27.6 MB | 47.2 | 17.2 | 58.0 |
| focalnet_tiny_srf FP32 | classify | 224² | 105.7 MB | 90.3 | 27.7 | 36.1 |
| swin_tiny_lora FP32 | classify | 224² | 108.7 MB | 98.9 | 29.6 | 33.8 |
| swin_tiny_lora INT8 † | classify | 224² | **31.8 MB** | 72.4 | 33.9 | 29.5 |
| vit_dinov2_lora FP32 | classify | 224² | 83.6 MB | 112.7 | 31.9 | 31.3 |
| vit_dinov2_lora INT8 † | classify | 224² | **23.1 MB** | 64.5 | 27.3 | 36.6 |
| semantic_guided_cgaf INT8 | seg + scene | 512² | 28.0 MB | 504.0 | 215.0 | 4.7 |
| semantic_guided_cgaf FP32 | seg + scene | 512² | 108.7 MB | 573.5 | 157.3 | 6.4 |
| clip_fft INT8 | classify | 224² | 328.6 MB | 76.4 | 21.8 | 45.8 |
| clip_fft FP32 | classify | 224² | ~703 MB | 77.3 | 23.3 | 43.0 |

## Verdicts

**Practical for edge delivery (small payload + fast):**
- **custom_cnn_small** (1.3 MB INT8 / 4.6 MB FP32) — trivially deliverable over
  scarce RF; 96.25% acc. Best when bandwidth dominates.
- **focalnet_tiny_srf INT8** (27.6 MB, 99.5% acc, fast) — the **sweet spot**:
  highest accuracy, fast, and payload still feasible on a decent link. The right
  default for accuracy-led edge deployment.

**Now practical after INT8 export († quantized in this probe):**
- **swin_tiny_lora INT8** (113.9 → **31.8 MB**, 3.6×, **99.25%** val acc) and
  **vit_dinov2_lora INT8** (87.6 → **23.1 MB**, 3.8×, **98.25%** val acc) both
  drop straight into the focalnet-INT8 tier with negligible accuracy loss. INT8
  export converts the heavy FP32 transformers into edge-deliverable artifacts.
  Quantization config matches the project default (QDQ, QInt8 act+weight,
  per-channel, MinMax; calibrated on 128 train images).

**Borderline — latency fine, payload heavy (FP32 only; prefer the INT8 export):**
- **focalnet FP32 (106 MB)**, **swin_tiny_lora FP32 (109 MB)**,
  **vit_dinov2_lora FP32 (84 MB)**. FP32 is wasteful over RF when INT8 holds
  accuracy — ship the INT8 variant instead.

**Included for comparison, but impractical for RF/edge delivery:**
- **CLIP (329 MB INT8 / ~703 MB FP32)** — latency is fine (~22 ms), but the
  payload is the killer: it ships a full CLIP ViT-B/32 backbone for a 4-class
  head. ~250× the custom_cnn payload for marginal accuracy. Quantization barely
  helps the delivery story here. The SPA can run the INT8 classifier, but it is
  a CDN-hosted demonstration model rather than a good scarce-link edge payload.

**Different problem, heavier compute:**
- **semantic_guided_cgaf (512², 157–215 ms/4-thr, ~5–6 fps)** — a
  segmentation + scene multitask model, ~10× the per-inference cost of the
  classifiers and a 512² input. 28 MB INT8 payload is fine, but in-browser WASM
  (×2–3) would put it near ~0.5–1 s/frame. Use it only when you need
  segmentation output, not for fast scene tagging.

## Notable observations

- **INT8 isn't always faster on CPU.** For `custom_cnn_small`, INT8 (64 ms) was
  *slower* than FP32 (47 ms) single-threaded: QDQ quantize/dequantize overhead
  outweighs the tiny compute saving on a small model. INT8's win there is
  **payload** (3.6×), not speed. For the larger focalnet, INT8 helps both
  (47 ms vs 90 ms, 27.6 MB vs 106 MB).
- **Quantization's main edge benefit is delivery size**, not latency — which is
  exactly the RF-airtime argument. The clearest win is focalnet INT8.
- **Payload, not latency, is the binding constraint** for edge delivery. Every
  classifier runs well under ~120 ms single-thread; the deciding factor is how
  many MB cross the link.
