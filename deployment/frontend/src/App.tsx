import { useEffect, useMemo, useState } from "react";

import { fetchModels, predictImage, type ModelSummary, type PredictResponse } from "./api/predict";
import { fetchModelCatalog, type ModelConfig } from "./onnx/modelRegistry";
import { classifyImage, loadModel, type InferenceResult } from "./onnx/ortInference";
import "./styles.css";

type Status = "loading-models" | "idle" | "predicting" | "error" | "complete";
type InferenceMode = "web" | "local";

const CLASS_ORDER = ["bridge", "freeway", "overpass", "railway"];
const DEPLOYMENT_MODEL_ID = "semantic_guided_cgaf_int8";
const LOCAL_ONNX_MODEL_IDS = new Set([DEPLOYMENT_MODEL_ID]);

export function App() {
  const [models, setModels] = useState<ModelSummary[]>([]);
  const [localModels, setLocalModels] = useState<ModelConfig[]>([]);
  const [selectedModelId, setSelectedModelId] = useState<string>("");
  const [selectedLocalModelId, setSelectedLocalModelId] = useState<string>("");
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [localResult, setLocalResult] = useState<InferenceResult | null>(null);
  const [inferenceMode, setInferenceMode] = useState<InferenceMode>("web");
  const [status, setStatus] = useState<Status>("loading-models");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([fetchModels(), fetchModelCatalog()])
      .then(([catalog, localCatalog]) => {
        setModels(catalog.models);
        setLocalModels(
          localCatalog.filter((model) => LOCAL_ONNX_MODEL_IDS.has(model.id)),
        );
        const firstRunnable = catalog.models.find((model) => model.id === catalog.active_model && model.available)
          ?? catalog.models.find((model) => model.available)
          ?? catalog.models[0];
        setSelectedModelId(firstRunnable?.id ?? "");
        setSelectedLocalModelId(localCatalog.find((model) => model.id === DEPLOYMENT_MODEL_ID)?.id ?? "");
        setStatus("idle");
      })
      .catch((cause) => {
        setStatus("error");
        setError(cause instanceof Error ? cause.message : String(cause));
      });
  }, []);

  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  const selectedModel = useMemo(
    () => models.find((model) => model.id === selectedModelId) ?? null,
    [models, selectedModelId],
  );
  const selectedLocalModel = useMemo(
    () => localModels.find((model) => model.id === selectedLocalModelId) ?? null,
    [localModels, selectedLocalModelId],
  );

  const scores = useMemo(() => {
    if (inferenceMode === "local") {
      if (!localResult) return [];
      return CLASS_ORDER.map((label) => ({
        label,
        value: localResult.predictions.find((prediction) => prediction.label === label)?.prob ?? 0,
      }));
    }
    if (!result) return [];
    return CLASS_ORDER.map((label) => ({
      label,
      value: result.class_scores[label] ?? 0,
    }));
  }, [inferenceMode, localResult, result]);

  const activeResult = inferenceMode === "local" ? localResult : result;
  const activeModel = inferenceMode === "local" ? selectedLocalModel : selectedModel;
  const activeModelAvailable = inferenceMode === "local" ? Boolean(selectedLocalModel) : Boolean(selectedModel?.available);

  function selectFile(nextFile: File | undefined) {
    if (!nextFile) return;
    if (!nextFile.type.startsWith("image/")) {
      setStatus("error");
      setError("Please choose an image file.");
      return;
    }
    setFile(nextFile);
    setResult(null);
    setLocalResult(null);
    setError(null);
    setStatus("idle");
    setPreviewUrl((current) => {
      if (current) URL.revokeObjectURL(current);
      return URL.createObjectURL(nextFile);
    });
  }

  function chooseModel(modelId: string) {
    if (inferenceMode === "local") {
      setSelectedLocalModelId(modelId);
    } else {
      setSelectedModelId(modelId);
    }
    setResult(null);
    setLocalResult(null);
    setError(null);
    setStatus("idle");
  }

  function chooseMode(mode: InferenceMode) {
    setInferenceMode(mode);
    setResult(null);
    setLocalResult(null);
    setError(null);
    setStatus("idle");
  }

  async function runPrediction() {
    if (!file || !activeModel) return;
    if (!activeModelAvailable) {
      setStatus("error");
      setError(
        inferenceMode === "local"
          ? "No local/browser ONNX model is configured. Check models.json and VITE_ONNX_MODEL_BASE_URL."
          : `${selectedModel?.display_name ?? "Selected model"} is listed for showcase, but its ONNX file is not packaged in this Docker build.`,
      );
      return;
    }

    setStatus("predicting");
    setError(null);
    try {
      if (inferenceMode === "local") {
        if (!selectedLocalModel) throw new Error("No local/browser ONNX model is selected.");
        const image = await createImageBitmap(file);
        try {
          const loaded = await loadModel(selectedLocalModel);
          setLocalResult(await classifyImage(loaded, image));
        } finally {
          image.close();
        }
      } else {
        if (!selectedModel) throw new Error("No web inference model is selected.");
        const prediction = await predictImage(file, selectedModel.id);
        setResult(prediction);
      }
      setStatus("complete");
    } catch (cause) {
      setStatus("error");
      setError(cause instanceof Error ? cause.message : String(cause));
    }
  }

  const runnableCount = models.filter((model) => model.available).length;
  const displayedModels = inferenceMode === "local" ? localModels : models;
  const selectedDisplayId = inferenceMode === "local" ? selectedLocalModelId : selectedModelId;
  const endpointLabel = inferenceMode === "local" ? "Browser ONNX" : "POST /predict";

  return (
    <main className="app-shell">
      <header className="topbar">
        <div>
          <p className="eyebrow">CSC3109 deployment | Group 12</p>
          <h1>Aerial image classifier</h1>
        </div>
        <div className="endpoint-chip" aria-label="Backend prediction endpoint">
          <span>{inferenceMode === "local" ? "Local inference" : "Web inference"}</span>
          <strong>{endpointLabel}</strong>
        </div>
      </header>

      <section className="workspace">
        <section className="panel image-panel">
          <div className="panel-header">
            <div>
              <p className="eyebrow">Input</p>
              <h2>Upload aerial image</h2>
            </div>
            {file ? <span className="file-name">{file.name}</span> : null}
          </div>

          <div
            className="drop-target"
            onDragOver={(event) => event.preventDefault()}
            onDrop={(event) => {
              event.preventDefault();
              selectFile(event.dataTransfer.files?.[0]);
            }}
          >
            {previewUrl ? (
              <img src={previewUrl} alt="Uploaded aerial scene preview" className="preview-image" />
            ) : (
              <label className="empty-state">
                <input type="file" accept="image/*" onChange={(event) => selectFile(event.target.files?.[0])} />
                <strong>Select image</strong>
                <span>Drop an aerial image here or choose one from disk.</span>
              </label>
            )}
          </div>

          <div className="actions">
            <label className="ghost-button">
              {file ? "Replace image" : "Choose image"}
              <input type="file" accept="image/*" onChange={(event) => selectFile(event.target.files?.[0])} />
            </label>
            <button
              className="primary-button"
              onClick={runPrediction}
              disabled={!file || !activeModelAvailable || status === "predicting"}
            >
              {status === "predicting" ? "Predicting..." : inferenceMode === "local" ? "Run local ONNX" : "Run prediction"}
            </button>
          </div>
        </section>

        <aside className="side-panel">
          <section className="panel model-panel">
            <div className="model-panel-header">
              <div>
                <p className="eyebrow">Model showcase</p>
                <h2>{inferenceMode === "local" ? `${localModels.length} CDN ONNX ${localModels.length === 1 ? "model" : "models"}` : `${runnableCount} packaged / ${models.length} listed`}</h2>
              </div>
            </div>
            <div className="mode-switch" aria-label="Inference mode">
              <button className={inferenceMode === "web" ? "mode-option active" : "mode-option"} onClick={() => chooseMode("web")} type="button">
                <strong>Web</strong>
                <span>backend /predict</span>
              </button>
              <button className={inferenceMode === "local" ? "mode-option active" : "mode-option"} onClick={() => chooseMode("local")} type="button">
                <strong>Local</strong>
                <span>CDN cached model</span>
              </button>
            </div>
            <div className="model-list">
              {displayedModels.map((model) => (
                <button
                  key={model.id}
                  className={`model-option ${model.id === selectedDisplayId ? "selected" : ""}`}
                  onClick={() => chooseModel(model.id)}
                  type="button"
                >
                  <span className="model-copy">
                    <strong>{"display_name" in model ? model.display_name : model.displayName}</strong>
                    <small>{model.description}</small>
                  </span>
                  <span className={inferenceMode === "local" || ("available" in model && model.available) ? "status-chip ready" : "status-chip pending"}>
                    {inferenceMode === "local" ? "CDN" : "available" in model && model.available ? "Packaged" : "Not packaged"}
                  </span>
                </button>
              ))}
            </div>
          </section>

          <section className="panel result-panel">
            <p className="eyebrow">Prediction</p>
            {activeResult ? (
              <>
                <div className="prediction-header">
                  <span>{inferenceMode === "local" ? localResult?.top.label : result?.predicted_label}</span>
                  <strong>{formatPercent(inferenceMode === "local" ? localResult?.top.prob ?? 0 : result?.confidence ?? 0)}</strong>
                </div>
                <div className="score-list">
                  {scores.map((score) => (
                    <div className="score-row" key={score.label}>
                      <span className="score-label">{score.label}</span>
                      <span className="score-track">
                        <span className="score-fill" style={{ width: formatPercent(score.value) }} />
                      </span>
                      <span className="score-value">{formatPercent(score.value)}</span>
                    </div>
                  ))}
                </div>
              </>
            ) : (
              <p className="muted">Choose Web for backend inference or Local to fetch/cache an ONNX model from the CDN and run it in this browser.</p>
            )}
          </section>

          <section className="panel detail-panel">
            <p className="eyebrow">Runtime</p>
            <dl>
              <dt>Endpoint</dt>
              <dd>{endpointLabel}</dd>
              <dt>Selected</dt>
              <dd>{inferenceMode === "local" ? selectedLocalModel?.displayName ?? "Loading local models" : selectedModel?.display_name ?? "Loading models"}</dd>
              <dt>Role</dt>
              <dd>{inferenceMode === "local" ? "CDN/browser ONNX" : selectedModel?.role ?? "-"}</dd>
              <dt>Model</dt>
              <dd>{inferenceMode === "local" ? selectedLocalModel?.id ?? "-" : result?.display_name ?? selectedModel?.family ?? "-"}</dd>
              <dt>Inference</dt>
              <dd>{inferenceMode === "local" ? localResult ? `${localResult.inferenceMs.toFixed(1)} ms` : "Waiting for image" : result ? `${result.inference_ms.toFixed(1)} ms` : "Waiting for image"}</dd>
              <dt>Provider</dt>
              <dd>{inferenceMode === "local" ? localResult?.provider ?? "Browser ORT" : result?.execution_provider ?? "Detected by backend"}</dd>
              <dt>Preprocess</dt>
              <dd>{inferenceMode === "local" ? `${selectedLocalModel?.preprocessing.imageSize ?? "-"}×${selectedLocalModel?.preprocessing.imageSize ?? "-"}` : result?.preprocessing ?? selectedModel?.preprocessing ?? "-"}</dd>
            </dl>
          </section>
        </aside>
      </section>

      {error ? <section className="error-panel">{error}</section> : null}
    </main>
  );
}

function formatPercent(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}
