export interface ModelSummary {
  id: string;
  name: string;
  display_name: string;
  description: string;
  family: string;
  role: string;
  active: boolean;
  available: boolean;
  onnx_path: string;
  input_size: [number, number];
  preprocessing: string;
}

export interface ModelsResponse {
  active_model: string;
  models: ModelSummary[];
}

export interface PredictResponse {
  model_id: string;
  model_name: string;
  display_name: string;
  predicted_label: string;
  confidence: number;
  class_scores: Record<string, number>;
  inference_ms: number;
  execution_provider: string;
  execution_providers: string[];
  preprocessing: string;
}

export async function fetchModels(): Promise<ModelsResponse> {
  const response = await fetch("/models");
  if (!response.ok) {
    const message = await readError(response);
    throw new Error(message);
  }
  return (await response.json()) as ModelsResponse;
}

export async function predictImage(file: File, modelId: string): Promise<PredictResponse> {
  const form = new FormData();
  form.append("file", file);
  form.append("model_id", modelId);

  const response = await fetch("/predict", {
    method: "POST",
    body: form,
  });

  if (!response.ok) {
    const message = await readError(response);
    throw new Error(message);
  }

  return (await response.json()) as PredictResponse;
}

async function readError(response: Response): Promise<string> {
  try {
    const payload = (await response.json()) as { detail?: string };
    return payload.detail || `Request failed with HTTP ${response.status}`;
  } catch {
    return `Request failed with HTTP ${response.status}`;
  }
}
