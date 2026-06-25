// Copies quantized model artifacts into public/ so the dev server and build can
// serve them locally. Binaries are not committed; run `bun run sync:assets`
// (also run automatically before dev/build).
import { cp, mkdir, access, stat } from "node:fs/promises";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const frontendDir = resolve(here, "..");
const repoRoot = resolve(frontendDir, "..", "..");

const modelsOut = join(frontendDir, "public", "models");

// Model artifacts: [source relative to repo root, destination filename].
const MODELS = [
  ["model/custom_cnn_small_onnx/custom_cnn_small_int8_qdq.onnx", "custom_cnn_small_int8_qdq.onnx"],
  ["model/focalnet_tiny_srf_onnx/focalnet_tiny_srf_int8_qdq.onnx", "focalnet_tiny_srf_int8_qdq.onnx"],
  ["model/swin_tiny_lora_onnx/swin_tiny_lora_int8_qdq.onnx", "swin_tiny_lora_int8_qdq.onnx"],
  ["reports/clip_training/clip_onnx_int8_qdq/clip_fft_int8_qdq.onnx", "clip_fft_int8_qdq.onnx"],
  [
    "model/vit_small_patch14_dinov2_lvd142m_lora_onnx/vit_small_patch14_dinov2_lvd142m_lora_int8_qdq.onnx",
    "vit_dinov2_lora_int8_qdq.onnx",
  ],
  [
    "model/semantic_guided_cgaf_onnx_int8_fullcalib_minmax_20260616/semantic_guided_cgaf_fft_int8_qdq_fullcalib_minmax.onnx",
    "semantic_guided_cgaf_fft_int8_qdq_fullcalib_minmax.onnx",
  ],
];

async function exists(path) {
  try {
    await access(path);
    return true;
  } catch {
    return false;
  }
}

async function syncModels() {
  await mkdir(modelsOut, { recursive: true });
  let copied = 0;
  let skipped = 0;
  for (const [rel, dest] of MODELS) {
    const src = join(repoRoot, rel);
    if (!(await exists(src))) {
      console.warn(`WARN: model not found, skipping: ${rel}`);
      continue;
    }
    const out = join(modelsOut, dest);
    if (await sameSize(src, out)) {
      skipped += 1;
      continue;
    }
    await cp(src, out);
    copied += 1;
  }
  console.log(`synced ${copied}/${MODELS.length} model artifacts -> public/models/ (${skipped} unchanged)`);
}

async function sameSize(left, right) {
  try {
    const [leftStat, rightStat] = await Promise.all([stat(left), stat(right)]);
    return leftStat.size === rightStat.size;
  } catch {
    return false;
  }
}

await syncModels();
