import { useEffect, useMemo, useState } from "react";

import { fetchModels, predictImage, type ModelSummary, type PredictResponse } from "./api/predict";
import "./styles.css";

type Status = "loading-models" | "idle" | "predicting" | "error" | "complete";

const CLASS_ORDER = ["bridge", "freeway", "overpass", "railway"];

export function App() {
  const [models, setModels] = useState<ModelSummary[]>([]);
  const [selectedModelId, setSelectedModelId] = useState<string>("");
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [status, setStatus] = useState<Status>("loading-models");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchModels()
      .then((catalog) => {
        setModels(catalog.models);
        const firstRunnable = catalog.models.find((model) => model.id === catalog.active_model && model.available)
          ?? catalog.models.find((model) => model.available)
          ?? catalog.models[0];
        setSelectedModelId(firstRunnable?.id ?? "");
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

  const scores = useMemo(() => {
    if (!result) return [];
    return CLASS_ORDER.map((label) => ({
      label,
      value: result.class_scores[label] ?? 0,
    }));
  }, [result]);

  function selectFile(nextFile: File | undefined) {
    if (!nextFile) return;
    if (!nextFile.type.startsWith("image/")) {
      setStatus("error");
      setError("Please choose an image file.");
      return;
    }
    setFile(nextFile);
    setResult(null);
    setError(null);
    setStatus("idle");
    setPreviewUrl((current) => {
      if (current) URL.revokeObjectURL(current);
      return URL.createObjectURL(nextFile);
    });
  }

  function chooseModel(modelId: string) {
    setSelectedModelId(modelId);
    setResult(null);
    setError(null);
    setStatus("idle");
  }

  async function runPrediction() {
    if (!file || !selectedModel) return;
    if (!selectedModel.available) {
      setStatus("error");
      setError(`${selectedModel.display_name} is listed for showcase, but its ONNX file is not packaged in this Docker build.`);
      return;
    }

    setStatus("predicting");
    setError(null);
    try {
      const prediction = await predictImage(file, selectedModel.id);
      setResult(prediction);
      setStatus("complete");
    } catch (cause) {
      setStatus("error");
      setError(cause instanceof Error ? cause.message : String(cause));
    }
  }

  const runnableCount = models.filter((model) => model.available).length;

  return (
    <main className="app-shell">
      <header className="topbar">
        <div>
          <p className="eyebrow">CSC3109 deployment | Group 12</p>
          <h1>Aerial image classifier</h1>
        </div>
        <div className="endpoint-chip" aria-label="Backend prediction endpoint">
          <span>API endpoint</span>
          <strong>POST /predict</strong>
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
              disabled={!file || !selectedModel?.available || status === "predicting"}
            >
              {status === "predicting" ? "Predicting..." : "Run prediction"}
            </button>
          </div>
        </section>

        <aside className="side-panel">
          <section className="panel model-panel">
            <div className="model-panel-header">
              <div>
                <p className="eyebrow">Model showcase</p>
                <h2>{runnableCount} packaged / {models.length} listed</h2>
              </div>
            </div>
            <div className="model-list">
              {models.map((model) => (
                <button
                  key={model.id}
                  className={`model-option ${model.id === selectedModelId ? "selected" : ""}`}
                  onClick={() => chooseModel(model.id)}
                  type="button"
                >
                  <span className="model-copy">
                    <strong>{model.display_name}</strong>
                    <small>{model.description}</small>
                  </span>
                  <span className={model.available ? "status-chip ready" : "status-chip pending"}>
                    {model.available ? "Packaged" : "Not packaged"}
                  </span>
                </button>
              ))}
            </div>
          </section>

          <section className="panel result-panel">
            <p className="eyebrow">Prediction</p>
            {result ? (
              <>
                <div className="prediction-header">
                  <span>{result.predicted_label}</span>
                  <strong>{formatPercent(result.confidence)}</strong>
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
              <p className="muted">Choose a packaged model, upload an image, and run the backend ONNX endpoint.</p>
            )}
          </section>

          <section className="panel detail-panel">
            <p className="eyebrow">Runtime</p>
            <dl>
              <dt>Endpoint</dt>
              <dd>POST /predict</dd>
              <dt>Selected</dt>
              <dd>{selectedModel?.display_name ?? "Loading models"}</dd>
              <dt>Role</dt>
              <dd>{selectedModel?.role ?? "-"}</dd>
              <dt>Model</dt>
              <dd>{result?.display_name ?? selectedModel?.family ?? "-"}</dd>
              <dt>Inference</dt>
              <dd>{result ? `${result.inference_ms.toFixed(1)} ms` : "Waiting for image"}</dd>
              <dt>Provider</dt>
              <dd>{result?.execution_provider ?? "Detected by backend"}</dd>
              <dt>Preprocess</dt>
              <dd>{result?.preprocessing ?? selectedModel?.preprocessing ?? "-"}</dd>
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
