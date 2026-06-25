import type * as ort from "onnxruntime-web/wasm";

export const SEGMENTATION_COLORS: [number, number, number][] = [
  [28, 31, 35],
  [239, 71, 111],
  [17, 138, 178],
  [255, 209, 102],
  [6, 214, 160],
];

export interface SegmentationOverlay {
  dataUrl: string;
  width: number;
  height: number;
  labels: string[];
  presentClassIds: number[];
}

export function segmentationOverlayFromLogits(
  tensor: ort.Tensor,
  labels: string[],
  alpha = 120,
): SegmentationOverlay {
  const dims = tensor.dims.map(Number);
  if (dims.length !== 4) {
    throw new Error(`Expected segmentation logits [1,C,H,W], got [${dims.join(",")}]`);
  }

  const [, classes, height, width] = dims;
  const data = tensor.data as Float32Array;
  const pixels = width * height;
  const rgba = new Uint8ClampedArray(pixels * 4);
  const present = new Set<number>();

  for (let pixel = 0; pixel < pixels; pixel += 1) {
    let bestClass = 0;
    let bestLogit = data[pixel];
    for (let classId = 1; classId < classes; classId += 1) {
      const logit = data[classId * pixels + pixel];
      if (logit > bestLogit) {
        bestLogit = logit;
        bestClass = classId;
      }
    }

    if (bestClass === 0) continue;

    present.add(bestClass);
    const [r, g, b] = SEGMENTATION_COLORS[bestClass] ?? colorForClass(bestClass);
    const out = pixel * 4;
    rgba[out] = r;
    rgba[out + 1] = g;
    rgba[out + 2] = b;
    rgba[out + 3] = alpha;
  }

  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  if (!ctx) {
    throw new Error("Could not create segmentation overlay canvas");
  }
  ctx.putImageData(new ImageData(rgba, width, height), 0, 0);

  return {
    dataUrl: canvas.toDataURL("image/png"),
    width,
    height,
    labels,
    presentClassIds: [...present].sort((a, b) => a - b),
  };
}

export function colorForClass(classId: number): [number, number, number] {
  if (classId < SEGMENTATION_COLORS.length) {
    return SEGMENTATION_COLORS[classId];
  }
  const value = Math.imul(classId + 17, 2654435761) >>> 0;
  return [80 + (value & 175), 80 + ((value >> 8) & 175), 80 + ((value >> 16) & 175)];
}
