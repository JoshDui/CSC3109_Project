import * as ort from "onnxruntime-web/wasm";

import type { EdgeModelManifest } from "./modelManifest";

export interface SessionWarmupResult {
  modelId: string;
  provider: string;
  loadMs: number;
  inputNames: readonly string[];
  outputNames: readonly string[];
}

export async function createSessionFromManifest(
  manifest: EdgeModelManifest,
): Promise<SessionWarmupResult> {
  const wasmPaths = import.meta.env.VITE_ORT_WASM_PATHS ?? manifest.runtime.wasmPaths;
  ort.env.wasm.wasmPaths = wasmPaths;
  ort.env.wasm.numThreads = Math.max(1, Math.min(4, navigator.hardwareConcurrency || 1));

  const start = performance.now();
  const session = await ort.InferenceSession.create(manifest.model.url, {
    executionProviders: [manifest.runtime.executionProvider],
    graphOptimizationLevel: "all",
  });
  const loadMs = performance.now() - start;

  return {
    modelId: manifest.model.id,
    provider: manifest.runtime.executionProvider,
    loadMs,
    inputNames: session.inputNames,
    outputNames: session.outputNames,
  };
}
