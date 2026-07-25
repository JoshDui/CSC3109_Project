// Legacy/dev helper for copying the browser ONNX artifact into public/models/.
// The current Docker+Caddy smoke path serves the same artifact
// from /edge-models/models instead.
import { access, cp, mkdir, readdir, rm, stat } from "node:fs/promises";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const frontendDir = resolve(here, "..");
const repoRoot = resolve(frontendDir, "..", "..");

const modelsOut = join(frontendDir, "public", "models");

// Model artifacts: [source relative to repo root, destination filename].
const MODELS = [
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
  const expectedArtifacts = new Set(MODELS.flatMap(([, dest]) => [dest, `${dest}.br`]));
  let removed = 0;
  for (const entry of await readdir(modelsOut, { withFileTypes: true })) {
    const isModelArtifact = entry.isFile() && (entry.name.endsWith(".onnx") || entry.name.endsWith(".onnx.br"));
    if (isModelArtifact && !expectedArtifacts.has(entry.name)) {
      await rm(join(modelsOut, entry.name));
      removed += 1;
    }
  }

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
  console.log(
    `synced ${copied}/${MODELS.length} model artifact${MODELS.length === 1 ? "" : "s"} -> public/models/ (${skipped} unchanged, ${removed} stale removed)`,
  );
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
