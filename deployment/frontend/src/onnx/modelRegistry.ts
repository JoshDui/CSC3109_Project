export type ExecutionProvider = "wasm" | "webgpu" | "webnn";

export type ResizeMode = "stretch" | "shortest-centercrop";

export type Interpolation = "bilinear" | "bicubic" | "nearest";

export type ModelTask = "classification" | "segmentation_scene";

export interface ModelPreprocessing {
  imageSize: number;
  resize: ResizeMode;
  interpolation: Interpolation;
  mean: [number, number, number];
  std: [number, number, number];
}

export interface ModelConfig {
  id: string;
  displayName: string;
  description: string;
  task: ModelTask;
  url: string;
  sizeBytes: number;
  accuracy?: number;
  inputName: string;
  outputName?: string;
  classificationOutputName?: string;
  segmentationOutputName?: string;
  preprocessing: ModelPreprocessing;
  labels: string[];
  segmentationLabels?: string[];
  preferredEP: ExecutionProvider[];
}

interface ModelCatalogEntry {
  id: string;
  displayName: string;
  description: string;
  url: string;
}

interface ModelRuntimeSpec {
  task: ModelTask;
  sizeBytes: number;
  accuracy?: number;
  inputName: string;
  outputName?: string;
  classificationOutputName?: string;
  segmentationOutputName?: string;
  preprocessing: ModelPreprocessing;
  preferredEP: ExecutionProvider[];
}

export function classificationOutputName(model: ModelConfig): string {
  return model.classificationOutputName ?? model.outputName ?? "logits";
}

export interface ModelCatalog {
  labels?: string[];
  segmentationLabels?: string[];
  models: ModelCatalogEntry[];
}

export const catalogUrl = import.meta.env.VITE_MODEL_CATALOG_URL ?? "/models.json";

export async function fetchModelCatalog(url = catalogUrl): Promise<ModelConfig[]> {
  const response = await fetch(url, {
    headers: { Accept: "application/json" },
  });

  if (!response.ok) {
    throw new Error(`Failed to load model catalog (${response.status})`);
  }

  const catalog = (await response.json()) as ModelCatalog;
  if (!catalog.models?.length) {
    throw new Error("Model catalog is empty");
  }
  return catalog.models.map((entry) => normalizeModelEntry(entry, catalog));
}

const CLASS_LABELS = ["bridge", "freeway", "overpass", "railway"];
const SEGMENTATION_LABELS = ["background", ...CLASS_LABELS];
const IMAGENET_PREPROCESSING: ModelPreprocessing = {
  imageSize: 224,
  resize: "stretch",
  interpolation: "bicubic",
  mean: [0.485, 0.456, 0.406],
  std: [0.229, 0.224, 0.225],
};

const MODEL_SPECS: Record<string, ModelRuntimeSpec> = {
  custom_cnn_small_int8: {
    task: "classification",
    sizeBytes: 1_318_351,
    accuracy: 0.9625,
    inputName: "images",
    classificationOutputName: "logits",
    preprocessing: { ...IMAGENET_PREPROCESSING, interpolation: "bilinear" },
    preferredEP: ["wasm", "webgpu"],
  },
  focalnet_tiny_srf_int8: {
    task: "classification",
    sizeBytes: 28_954_694,
    accuracy: 0.995,
    inputName: "images",
    classificationOutputName: "logits",
    preprocessing: IMAGENET_PREPROCESSING,
    preferredEP: ["wasm", "webgpu"],
  },
  vit_dinov2_lora_int8: {
    task: "classification",
    sizeBytes: 23_102_850,
    accuracy: 0.9825,
    inputName: "images",
    classificationOutputName: "logits",
    preprocessing: IMAGENET_PREPROCESSING,
    preferredEP: ["wasm", "webgpu"],
  },
  swin_tiny_lora_int8: {
    task: "classification",
    sizeBytes: 31_842_653,
    accuracy: 0.9925,
    inputName: "images",
    classificationOutputName: "logits",
    preprocessing: IMAGENET_PREPROCESSING,
    preferredEP: ["wasm", "webgpu"],
  },
  clip_fft_int8: {
    task: "classification",
    sizeBytes: 344_587_839,
    accuracy: 0.98,
    inputName: "pixel_values",
    classificationOutputName: "logits",
    preprocessing: {
      imageSize: 224,
      resize: "shortest-centercrop",
      interpolation: "bicubic",
      mean: [0.48145466, 0.4578275, 0.40821073],
      std: [0.26862954, 0.26130258, 0.27577711],
    },
    preferredEP: ["wasm", "webgpu"],
  },
  semantic_guided_cgaf_int8: {
    task: "segmentation_scene",
    sizeBytes: 29_321_225,
    inputName: "images",
    classificationOutputName: "scene_logits",
    segmentationOutputName: "segmentation_logits",
    preprocessing: {
      imageSize: 512,
      resize: "stretch",
      interpolation: "bilinear",
      mean: [0.485, 0.456, 0.406],
      std: [0.229, 0.224, 0.225],
    },
    preferredEP: ["wasm"],
  },
};

function normalizeModelEntry(entry: ModelCatalogEntry, catalog: ModelCatalog): ModelConfig {
  const spec = MODEL_SPECS[entry.id];
  if (!spec) {
    throw new Error(`Model '${entry.id}' is missing a runtime spec in modelRegistry.ts`);
  }

  return {
    ...entry,
    ...spec,
    labels: catalog.labels ?? CLASS_LABELS,
    segmentationLabels: spec.task === "segmentation_scene" ? catalog.segmentationLabels ?? SEGMENTATION_LABELS : undefined,
  };
}

export function selectModel(models: ModelConfig[], id: string): ModelConfig {
  const model = models.find((entry) => entry.id === id);
  if (!model) {
    throw new Error(`Model not found in catalog: ${id}`);
  }
  return model;
}

export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  const units = ["KiB", "MiB", "GiB"];
  let value = bytes / 1024;
  let unit = units[0];
  for (let index = 1; index < units.length && value >= 1024; index += 1) {
    value /= 1024;
    unit = units[index];
  }
  return `${value.toFixed(2)} ${unit}`;
}
