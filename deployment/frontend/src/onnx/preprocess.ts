import * as ort from "onnxruntime-web/wasm";

import type { ModelPreprocessing } from "./modelRegistry";

/**
 * Convert an image element into a normalized NCHW float32 tensor.
 *
 * Browser canvas resampling cannot bit-match PIL bilinear/bicubic, but the
 * sub-pixel difference does not change argmax for this 4-class task. The
 * geometry (stretch vs shortest-edge + center crop) and per-model
 * mean/std/scale are matched to the Python eval pipeline.
 */
export function imageToTensor(
  image: HTMLImageElement | HTMLCanvasElement | ImageBitmap,
  cfg: ModelPreprocessing,
): ort.Tensor {
  const size = cfg.imageSize;
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  if (!ctx) {
    throw new Error("Could not acquire 2D canvas context for preprocessing");
  }

  const { width, height } = sourceDimensions(image);

  if (cfg.resize === "shortest-centercrop") {
    // Resize so the shortest edge is `size`, then center-crop a size x size box.
    const scale = size / Math.min(width, height);
    const scaledW = width * scale;
    const scaledH = height * scale;
    const dx = (size - scaledW) / 2;
    const dy = (size - scaledH) / 2;
    ctx.drawImage(image, dx, dy, scaledW, scaledH);
  } else {
    // Direct stretch to size x size.
    ctx.drawImage(image, 0, 0, size, size);
  }

  const { data } = ctx.getImageData(0, 0, size, size);
  const plane = size * size;
  const float = new Float32Array(3 * plane);
  const [mr, mg, mb] = cfg.mean;
  const [sr, sg, sb] = cfg.std;

  for (let i = 0; i < plane; i += 1) {
    const r = data[i * 4] / 255;
    const g = data[i * 4 + 1] / 255;
    const b = data[i * 4 + 2] / 255;
    float[i] = (r - mr) / sr; // R plane
    float[plane + i] = (g - mg) / sg; // G plane
    float[2 * plane + i] = (b - mb) / sb; // B plane
  }

  return new ort.Tensor("float32", float, [1, 3, size, size]);
}

function sourceDimensions(
  image: HTMLImageElement | HTMLCanvasElement | ImageBitmap,
): { width: number; height: number } {
  if (image instanceof HTMLImageElement) {
    return { width: image.naturalWidth, height: image.naturalHeight };
  }
  return { width: image.width, height: image.height };
}
