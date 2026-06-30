import * as ort from "onnxruntime-web/wasm";

import { classificationOutputName, type ModelConfig } from "./modelRegistry";
import { imageToTensor } from "./preprocess";
import { resolveProviderOrder, toOrtProvider } from "./executionProvider";
import { segmentationOverlayFromLogits, type SegmentationOverlay } from "./segmentation";

export interface LoadedModel {
  config: ModelConfig;
  session: ort.InferenceSession;
  provider: string;
  loadMs: number;
}

export interface Prediction {
  label: string;
  prob: number;
}

export interface InferenceResult {
  predictions: Prediction[];
  top: Prediction;
  inferenceMs: number;
  provider: string;
  segmentation?: SegmentationOverlay;
}

const wasmConfigured = (() => {
  let done = false;
  return () => {
    if (done) return;
    if (import.meta.env.VITE_ORT_WASM_PATHS) {
      ort.env.wasm.wasmPaths = import.meta.env.VITE_ORT_WASM_PATHS;
    }
    // Single-thread WASM avoids ORT worker-loading issues in the local Caddy demo.
    ort.env.wasm.proxy = false;
    ort.env.wasm.numThreads = 1;
    done = true;
  };
})();

const sessionCache = new Map<string, LoadedModel>();
const modelBytesCache = new Map<string, Uint8Array>();
const BROWSER_MODEL_CACHE = "csc3109-onnx-models-v1";

/**
 * Create (or reuse) an inference session for a model, attempting each resolved
 * execution provider in order and falling back on failure.
 */
export async function loadModel(config: ModelConfig): Promise<LoadedModel> {
  const cached = sessionCache.get(config.id);
  if (cached) return cached;

  wasmConfigured();

  const providers = resolveProviderOrder(config.preferredEP);
  const errors: string[] = [];

  for (const provider of providers) {
    try {
      const start = performance.now();
      const modelSource = await loadModelSource(config.url);
      const options = {
        executionProviders: [toOrtProvider(provider) as never],
        graphOptimizationLevel: "all",
      } satisfies ort.InferenceSession.SessionOptions;
      const session = typeof modelSource === "string"
        ? await ort.InferenceSession.create(modelSource, options)
        : await ort.InferenceSession.create(modelSource, options);
      const loadMs = performance.now() - start;
      const loaded: LoadedModel = { config, session, provider, loadMs };
      sessionCache.set(config.id, loaded);
      return loaded;
    } catch (cause) {
      errors.push(`${provider}: ${cause instanceof Error ? cause.message : String(cause)}`);
    }
  }

  throw new Error(`Failed to create session for ${config.id}. Tried -> ${errors.join(" | ")}`);
}

async function loadModelSource(url: string): Promise<string | Uint8Array> {
  const cachedBytes = modelBytesCache.get(url);
  if (cachedBytes) return cachedBytes;

  if (!("caches" in globalThis)) return url;

  let cache: Cache;
  try {
    cache = await caches.open(BROWSER_MODEL_CACHE);
  } catch (cause) {
    console.warn(`Model Cache API unavailable for ${url}; falling back to direct URL session`, cause);
    return url;
  }

  const cached = await cache.match(url);
  if (cached) {
    try {
      const bytes = await responseToModelBytes(cached, url);
      modelBytesCache.set(url, bytes);
      return bytes;
    } catch (cause) {
      await cache.delete(url);
      console.warn(`Discarded invalid cached model response for ${url}`, cause);
    }
  }

  const response = await fetch(url, { cache: "force-cache" });
  const clone = response.clone();
  const bytes = await responseToModelBytes(response, url);
  await cache.put(url, clone);
  modelBytesCache.set(url, bytes);
  return bytes;
}

async function responseToModelBytes(response: Response, url: string): Promise<Uint8Array> {
  if (!response.ok) {
    throw new Error(`Failed to fetch ONNX model ${url}: HTTP ${response.status}`);
  }
  const contentType = response.headers.get("content-type") ?? "";
  if (contentType.includes("text/html") || contentType.includes("application/json")) {
    throw new Error(`Model URL ${url} returned '${contentType || "unknown"}' instead of ONNX bytes`);
  }
  const bytes = new Uint8Array(await response.arrayBuffer());
  if (bytes.byteLength < 16 || bytes[0] === 0x3c) {
    throw new Error(`Model URL ${url} did not return a plausible ONNX payload`);
  }
  return bytes;
}

export async function classifyImage(
  loaded: LoadedModel,
  image: HTMLImageElement | HTMLCanvasElement | ImageBitmap,
): Promise<InferenceResult> {
  const { config, session, provider } = loaded;
  const tensor = imageToTensor(image, config.preprocessing);

  const start = performance.now();
  const outputs = await session.run({ [config.inputName]: tensor });
  const inferenceMs = performance.now() - start;

  const classOutputName = classificationOutputName(config);
  const output = outputs[classOutputName] ?? outputs[session.outputNames[0]];
  if (!output) {
    throw new Error(`Model output '${classOutputName}' not found`);
  }

  const logits = Array.from(output.data as Float32Array);
  const probs = softmax(logits);
  const predictions: Prediction[] = probs
    .map((prob, index) => ({ label: config.labels[index] ?? `class_${index}`, prob }))
    .sort((a, b) => b.prob - a.prob);

  let segmentation;
  if (config.segmentationOutputName) {
    const segmentationOutput = outputs[config.segmentationOutputName];
    if (!segmentationOutput) {
      throw new Error(`Model output '${config.segmentationOutputName}' not found`);
    }
    segmentation = segmentationOverlayFromLogits(segmentationOutput, config.segmentationLabels ?? []);
  }

  return { predictions, top: predictions[0], inferenceMs, provider, segmentation };
}

export interface BenchmarkResult {
  iters: number;
  medianMs: number;
  minMs: number;
  fps: number;
  provider: string;
}

/**
 * Run repeated inference on a single image to report real in-browser (WASM/
 * WebGPU/WebNN) latency. Preprocesses once, warms up, then times `iters` runs.
 */
export async function benchmark(
  loaded: LoadedModel,
  image: HTMLImageElement | HTMLCanvasElement | ImageBitmap,
  iters = 30,
  warmup = 5,
): Promise<BenchmarkResult> {
  const { config, session, provider } = loaded;
  const feeds = { [config.inputName]: imageToTensor(image, config.preprocessing) };

  for (let i = 0; i < warmup; i += 1) {
    await session.run(feeds);
  }

  const samples: number[] = [];
  for (let i = 0; i < iters; i += 1) {
    const start = performance.now();
    await session.run(feeds);
    samples.push(performance.now() - start);
  }
  samples.sort((a, b) => a - b);
  const medianMs = samples[Math.floor(samples.length / 2)];

  return { iters, medianMs, minMs: samples[0], fps: 1000 / medianMs, provider };
}

export function clearSessionCache(): void {
  sessionCache.clear();
}

function softmax(logits: number[]): number[] {
  const max = Math.max(...logits);
  const exps = logits.map((value) => Math.exp(value - max));
  const sum = exps.reduce((acc, value) => acc + value, 0);
  return exps.map((value) => value / sum);
}
