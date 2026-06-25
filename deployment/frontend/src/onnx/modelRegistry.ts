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
  runtimeProfile: RuntimeProfileId;
  task: ModelTask;
  url: string;
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
  runtimeProfile: RuntimeProfileId;
}

interface RuntimeProfileSpec {
  task: ModelTask;
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

const CLASS_LABELS = ["bridge", "freeway", "overpass", "railway"];
const SEGMENTATION_LABELS = ["background", ...CLASS_LABELS];
const IMAGENET_PREPROCESSING: ModelPreprocessing = {
  imageSize: 224,
  resize: "stretch",
  interpolation: "bicubic",
  mean: [0.485, 0.456, 0.406],
  std: [0.229, 0.224, 0.225],
};

const RUNTIME_PROFILES = {
  "imagenet-224-bilinear-classifier": {
    task: "classification",
    inputName: "images",
    classificationOutputName: "logits",
    preprocessing: { ...IMAGENET_PREPROCESSING, interpolation: "bilinear" },
    preferredEP: ["wasm", "webgpu"],
  },
  "imagenet-224-bicubic-classifier": {
    task: "classification",
    inputName: "images",
    classificationOutputName: "logits",
    preprocessing: IMAGENET_PREPROCESSING,
    preferredEP: ["wasm", "webgpu"],
  },
  "clip-224-classifier": {
    task: "classification",
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
  "semantic-cgaf-512-overlay": {
    task: "segmentation_scene",
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
} satisfies Record<string, RuntimeProfileSpec>;

export type RuntimeProfileId = keyof typeof RUNTIME_PROFILES;

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

function normalizeModelEntry(entry: ModelCatalogEntry, catalog: ModelCatalog): ModelConfig {
  const profile = RUNTIME_PROFILES[entry.runtimeProfile];
  if (!profile) {
    throw new Error(`Model '${entry.id}' references unknown runtime profile '${entry.runtimeProfile}'`);
  }

  return {
    ...entry,
    ...profile,
    labels: catalog.labels ?? CLASS_LABELS,
    segmentationLabels: profile.task === "segmentation_scene" ? catalog.segmentationLabels ?? SEGMENTATION_LABELS : undefined,
  };
}

export function selectModel(models: ModelConfig[], id: string): ModelConfig {
  const model = models.find((entry) => entry.id === id);
  if (!model) {
    throw new Error(`Model not found in catalog: ${id}`);
  }
  return model;
}
