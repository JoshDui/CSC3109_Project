import { useCallback, useEffect, useRef, useState } from "react";

import "./styles.css";
import {
  fetchModelCatalog,
  formatBytes,
  type ModelConfig,
} from "./onnx/modelRegistry";
import {
  benchmark,
  classifyImage,
  loadModel,
  type BenchmarkResult,
  type InferenceResult,
  type LoadedModel,
} from "./onnx/ortInference";
import { colorForClass, type SegmentationOverlay } from "./onnx/segmentation";

type Status =
  | "idle"
  | "loading-catalog"
  | "ready"
  | "loading-model"
  | "loading-overlay-model"
  | "classifying"
  | "segmenting-overlay"
  | "benchmarking"
  | "error";

interface OverlayRun {
  inferenceMs: number;
  provider: string;
}

export function App() {
  const [models, setModels] = useState<ModelConfig[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [loadedClassifier, setLoadedClassifier] = useState<LoadedModel | null>(null);
  const [loadedOverlay, setLoadedOverlay] = useState<LoadedModel | null>(null);
  const [classificationResult, setClassificationResult] = useState<InferenceResult | null>(null);
  const [segmentationOverlay, setSegmentationOverlay] = useState<SegmentationOverlay | null>(null);
  const [overlayRun, setOverlayRun] = useState<OverlayRun | null>(null);
  const [bench, setBench] = useState<BenchmarkResult | null>(null);
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [showOverlay, setShowOverlay] = useState(false);
  const [status, setStatus] = useState<Status>("idle");
  const [error, setError] = useState<string | null>(null);

  const imgRef = useRef<HTMLImageElement | null>(null);

  useEffect(() => {
    setStatus("loading-catalog");
    fetchModelCatalog()
      .then((catalog) => {
        const classifiers = catalog.filter((model) => model.task === "classification");
        setModels(catalog);
        setSelectedId(classifiers[0]?.id ?? null);
        setStatus("ready");
      })
      .catch((cause) => {
        setStatus("error");
        setError(cause instanceof Error ? cause.message : String(cause));
      });
  }, []);

  useEffect(() => {
    return () => {
      if (imageUrl) URL.revokeObjectURL(imageUrl);
    };
  }, [imageUrl]);

  const classifierModels = models.filter((model) => model.task === "classification");
  const overlayModel = models.find((model) => model.task === "segmentation_scene") ?? null;
  const selected = classifierModels.find((model) => model.id === selectedId) ?? null;
  const busy =
    status === "loading-model" ||
    status === "loading-overlay-model" ||
    status === "classifying" ||
    status === "segmenting-overlay" ||
    status === "benchmarking";
  const overlayAvailable = Boolean(segmentationOverlay);
  const overlayVisible = overlayAvailable && showOverlay;
  const overlayBusy = status === "loading-overlay-model" || status === "segmenting-overlay";

  const onSelectModel = useCallback((id: string) => {
    setSelectedId(id);
    setLoadedClassifier(null);
    setClassificationResult(null);
    setBench(null);
    setError(null);
  }, []);

  const onFile = useCallback((file: File | undefined) => {
    if (!file) return;
    setClassificationResult(null);
    setSegmentationOverlay(null);
    setOverlayRun(null);
    setBench(null);
    setError(null);
    setShowOverlay(false);
    setImageUrl((prev) => {
      if (prev) URL.revokeObjectURL(prev);
      return URL.createObjectURL(file);
    });
  }, []);

  const ensureClassifierLoaded = useCallback(async () => {
    if (!selected) return null;
    if (loadedClassifier?.config.id === selected.id) return loadedClassifier;
    setStatus("loading-model");
    const next = await loadModel(selected);
    setLoadedClassifier(next);
    return next;
  }, [selected, loadedClassifier]);

  const ensureOverlayLoaded = useCallback(async () => {
    if (!overlayModel) return null;
    if (loadedOverlay?.config.id === overlayModel.id) return loadedOverlay;
    setStatus("loading-overlay-model");
    const next = await loadModel(overlayModel);
    setLoadedOverlay(next);
    return next;
  }, [overlayModel, loadedOverlay]);

  const onClassify = useCallback(async () => {
    if (!imgRef.current) return;
    setError(null);
    try {
      const model = await ensureClassifierLoaded();
      if (!model) return;
      setStatus("classifying");
      const inference = await classifyImage(model, imgRef.current);
      setClassificationResult(inference);
      setStatus("ready");
    } catch (cause) {
      setStatus("error");
      setError(cause instanceof Error ? cause.message : String(cause));
    }
  }, [ensureClassifierLoaded]);

  const onShowOverlay = useCallback(async () => {
    if (!imgRef.current) return;
    setError(null);
    if (segmentationOverlay) {
      setShowOverlay(true);
      return;
    }

    if (!overlayModel) {
      setStatus("error");
      setError("Semantic overlay model is not available in the catalog.");
      return;
    }

    try {
      const model = await ensureOverlayLoaded();
      if (!model) return;
      setStatus("segmenting-overlay");
      const inference = await classifyImage(model, imgRef.current);
      if (!inference.segmentation) {
        throw new Error("Semantic overlay model did not return segmentation logits.");
      }
      setSegmentationOverlay(inference.segmentation);
      setOverlayRun({ inferenceMs: inference.inferenceMs, provider: inference.provider });
      setShowOverlay(true);
      setStatus("ready");
    } catch (cause) {
      setStatus("error");
      setError(cause instanceof Error ? cause.message : String(cause));
    }
  }, [ensureOverlayLoaded, overlayModel, segmentationOverlay]);

  const onBenchmark = useCallback(async () => {
    if (!imgRef.current) return;
    setError(null);
    try {
      const model = await ensureClassifierLoaded();
      if (!model) return;
      setStatus("benchmarking");
      const measured = await benchmark(model, imgRef.current, 30);
      setBench(measured);
      setStatus("ready");
    } catch (cause) {
      setStatus("error");
      setError(cause instanceof Error ? cause.message : String(cause));
    }
  }, [ensureClassifierLoaded]);

  const overlayTitle = !imageUrl
    ? "Upload an image first"
    : overlayAvailable
      ? "Show semantic segmentation overlay"
      : "Run the semantic-guided model and show overlay";

  const overlayLabel = overlayBusy
    ? status === "loading-overlay-model"
      ? "Loading…"
      : "Segmenting…"
    : "Overlay";

  return (
    <main className="app-shell">
      <header className="topbar">
        <div>
          <p className="eyebrow">CSC3109 deployment</p>
          <h1>Aerial Edge Classifier</h1>
        </div>
        <p className="topbar-note">Browser-side ONNX inference · images stay local</p>
      </header>

      <section className="workspace">
        <div className="image-panel panel">
          <div className="image-toolbar">
            <span>Image</span>
            <div className="toggle-group" aria-label="Image view toggles">
              <button
                className={!overlayVisible ? "toggle active" : "toggle"}
                onClick={() => setShowOverlay(false)}
                disabled={!imageUrl || overlayBusy}
              >
                Raw
              </button>
              <button
                className={overlayVisible ? "toggle active" : "toggle"}
                onClick={onShowOverlay}
                disabled={!imageUrl || !overlayModel || busy}
                title={overlayTitle}
              >
                {overlayLabel}
              </button>
            </div>
          </div>

          <div
            className="drop-target"
            onDragOver={(event) => event.preventDefault()}
            onDrop={(event) => {
              event.preventDefault();
              onFile(event.dataTransfer.files?.[0]);
            }}
          >
            {imageUrl ? (
              <div className="image-frame">
                <img ref={imgRef} src={imageUrl} alt="Uploaded aerial scene" className="base-image" />
                {overlayVisible && segmentationOverlay ? (
                  <img
                    className="overlay-image"
                    src={segmentationOverlay.dataUrl}
                    alt="Semantic segmentation overlay"
                  />
                ) : null}
              </div>
            ) : (
              <label className="empty-state">
                <input type="file" accept="image/*" onChange={(e) => onFile(e.target.files?.[0])} />
                <strong>Upload aerial image</strong>
                <span>Drop here or click to choose an image. It stays local.</span>
              </label>
            )}
          </div>

          {imageUrl ? (
            <label className="replace-link">
              Replace image
              <input type="file" accept="image/*" onChange={(e) => onFile(e.target.files?.[0])} />
            </label>
          ) : null}
        </div>

        <aside className="side-panel">
          <section className="panel result-card">
            <p className="eyebrow">Classifier results</p>
            {classificationResult ? (
              <>
                <div className="prediction-header">
                  <strong>{classificationResult.top.label}</strong>
                  <span>{(classificationResult.top.prob * 100).toFixed(1)}%</span>
                </div>
                <div className="bars">
                  {classificationResult.predictions.map((prediction) => (
                    <div className="bar-row" key={prediction.label}>
                      <span className="bar-label">{prediction.label}</span>
                      <span className="bar-track">
                        <span
                          className="bar-fill"
                          style={{ width: `${(prediction.prob * 100).toFixed(1)}%` }}
                        />
                      </span>
                      <span className="bar-value">{(prediction.prob * 100).toFixed(1)}%</span>
                    </div>
                  ))}
                </div>

              </>
            ) : (
              <p className="muted">Run a model to see final class probabilities and raw output summary.</p>
            )}

            {segmentationOverlay ? (
              <div className="legend">
                <p className="muted">Segmentation overlay classes</p>
                <div className="legend-grid">
                  {segmentationOverlay.labels.map((label, classId) => (
                    <span key={label} className="legend-item">
                      <span className="swatch" style={{ backgroundColor: rgb(colorForClass(classId)) }} />
                      {label}
                    </span>
                  ))}
                </div>
              </div>
            ) : null}
          </section>

          <section className="panel controls-card">
            <p className="eyebrow">Model + runtime</p>
            <div className="model-list">
              {classifierModels.map((model) => (
                <button
                  key={model.id}
                  className={`model-option ${model.id === selectedId ? "selected" : ""}`}
                  onClick={() => onSelectModel(model.id)}
                  disabled={busy}
                >
                  <span>
                    <strong>{model.displayName}</strong>
                    <small>{model.id === "clip_fft_int8" ? "CLIP classifier" : "classifier"}</small>
                  </span>
                  <span className="model-meta">
                    {formatBytes(model.sizeBytes)}
                    {model.accuracy ? ` · ${(model.accuracy * 100).toFixed(2)}%` : ""}
                  </span>
                </button>
              ))}
            </div>

            {selected ? (
              <dl className="telemetry">
                <dt>Selected</dt>
                <dd>{selected.displayName}</dd>
                <dt>Input</dt>
                <dd>{selected.preprocessing.imageSize}×{selected.preprocessing.imageSize}</dd>
                <dt>Providers</dt>
                <dd>{selected.preferredEP.join(" → ")}</dd>
                <dt>Overlay</dt>
                <dd>{overlayModel ? "toggle runs Semantic-Guided CG-AF" : "not available"}</dd>
              </dl>
            ) : null}

            <div className="actions">
              <button className="primary" onClick={onClassify} disabled={!selected || !imageUrl || busy}>
                {status === "classifying" ? "Classifying…" : "Classify"}
              </button>
              <button className="ghost" onClick={onBenchmark} disabled={!selected || !imageUrl || busy}>
                {status === "benchmarking" ? "Benchmarking…" : "Benchmark ×30"}
              </button>
            </div>

            {loadedClassifier || classificationResult || loadedOverlay || overlayRun || bench ? (
              <dl className="telemetry runtime">
                {loadedClassifier ? (
                  <>
                    <dt>Classifier load</dt>
                    <dd>{loadedClassifier.loadMs.toFixed(1)} ms</dd>
                  </>
                ) : null}
                {classificationResult ? (
                  <>
                    <dt>Last inference</dt>
                    <dd>{classificationResult.inferenceMs.toFixed(1)} ms · {classificationResult.provider}</dd>
                  </>
                ) : null}
                {loadedOverlay ? (
                  <>
                    <dt>Overlay load</dt>
                    <dd>{loadedOverlay.loadMs.toFixed(1)} ms</dd>
                  </>
                ) : null}
                {overlayRun ? (
                  <>
                    <dt>Overlay run</dt>
                    <dd>{overlayRun.inferenceMs.toFixed(1)} ms · {overlayRun.provider}</dd>
                  </>
                ) : null}
                {bench ? (
                  <>
                    <dt>Benchmark</dt>
                    <dd>{bench.medianMs.toFixed(1)} ms median · {bench.fps.toFixed(1)} fps</dd>
                  </>
                ) : null}
              </dl>
            ) : null}
          </section>
        </aside>
      </section>

      {error ? <section className="error-panel">{error}</section> : null}
    </main>
  );
}

function rgb(color: [number, number, number]): string {
  return `rgb(${color[0]} ${color[1]} ${color[2]})`;
}
