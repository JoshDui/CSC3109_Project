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
    ort.env.wasm.numThreads = Math.max(1, Math.min(4, navigator.hardwareConcurrency || 1));
    done = true;
  };
})();

const sessionCache = new Map<string, LoadedModel>();

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
      const session = await ort.InferenceSession.create(config.url, {
        executionProviders: [toOrtProvider(provider) as never],
        graphOptimizationLevel: "all",
      });
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
