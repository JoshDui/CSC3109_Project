export type ExecutionProvider = "wasm" | "webgpu" | "webnn";

export interface EdgeModelManifest {
  schemaVersion: "1";
  model: {
    id: string;
    family: string;
    variant: string;
    format: "onnx";
    url: string;
    sha256: string;
    bytes: number;
    expiresAt?: string;
  };
  runtime: {
    executionProvider: ExecutionProvider;
    wasmPaths: string;
    ortVersion?: string;
  };
  labels: string[];
  preprocessing: {
    imageSize: number;
    mean: [number, number, number];
    std: [number, number, number];
    colorSpace?: "RGB";
  };
}

export const defaultManifestUrl =
  import.meta.env.VITE_MODEL_MANIFEST_URL ?? "/model-manifest.example.json";

export async function fetchModelManifest(url = defaultManifestUrl): Promise<EdgeModelManifest> {
  const response = await fetch(url, {
    credentials: "include",
    headers: {
      Accept: "application/json",
    },
  });

  if (!response.ok) {
    throw new Error(`Failed to load model manifest (${response.status})`);
  }

  return (await response.json()) as EdgeModelManifest;
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
