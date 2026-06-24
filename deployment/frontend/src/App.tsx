import { useCallback, useState } from "react";

import "./styles.css";
import { createSessionFromManifest, type SessionWarmupResult } from "./onnx/ortInference";
import {
  defaultManifestUrl,
  fetchModelManifest,
  formatBytes,
  type EdgeModelManifest,
} from "./onnx/modelManifest";

type LoadState = "idle" | "loading" | "ready" | "error";

export function App() {
  const [manifest, setManifest] = useState<EdgeModelManifest | null>(null);
  const [session, setSession] = useState<SessionWarmupResult | null>(null);
  const [state, setState] = useState<LoadState>("idle");
  const [error, setError] = useState<string | null>(null);

  const loadManifest = useCallback(async () => {
    setState("loading");
    setError(null);
    setSession(null);
    try {
      const nextManifest = await fetchModelManifest();
      setManifest(nextManifest);
      setState("ready");
    } catch (cause) {
      setState("error");
      setError(cause instanceof Error ? cause.message : String(cause));
    }
  }, []);

  const warmSession = useCallback(async () => {
    if (!manifest) return;
    setState("loading");
    setError(null);
    try {
      const nextSession = await createSessionFromManifest(manifest);
      setSession(nextSession);
      setState("ready");
    } catch (cause) {
      setState("error");
      setError(cause instanceof Error ? cause.message : String(cause));
    }
  }, [manifest]);

  return (
    <main className="shell">
      <section className="hero">
        <p className="eyebrow">CSC3109 deployment scaffold</p>
        <h1>CDN-served ONNX, browser-side edge inference</h1>
        <p>
          The backend authenticates and returns a model manifest. The browser downloads the
          quantized ONNX model from a CDN, caches it, and runs inference locally with ONNX
          Runtime Web.
        </p>
      </section>

      <section className="panel">
        <div>
          <h2>Model manifest</h2>
          <p className="muted">Default manifest URL: {defaultManifestUrl}</p>
        </div>
        <button onClick={loadManifest} disabled={state === "loading"}>
          Load manifest
        </button>
      </section>

      {manifest ? (
        <section className="grid">
          <article className="card">
            <h3>{manifest.model.id}</h3>
            <dl>
              <dt>Family</dt>
              <dd>{manifest.model.family}</dd>
              <dt>Variant</dt>
              <dd>{manifest.model.variant}</dd>
              <dt>Model size</dt>
              <dd>{formatBytes(manifest.model.bytes)}</dd>
              <dt>Runtime</dt>
              <dd>{manifest.runtime.executionProvider}</dd>
            </dl>
          </article>
          <article className="card">
            <h3>Preprocessing</h3>
            <dl>
              <dt>Image size</dt>
              <dd>{manifest.preprocessing.imageSize}px</dd>
              <dt>Labels</dt>
              <dd>{manifest.labels.join(", ")}</dd>
              <dt>Model URL</dt>
              <dd className="break-word">{manifest.model.url}</dd>
            </dl>
          </article>
        </section>
      ) : null}

      <section className="panel">
        <div>
          <h2>ONNX Runtime warmup</h2>
          <p className="muted">Creates an inference session from the manifest model URL.</p>
        </div>
        <button onClick={warmSession} disabled={!manifest || state === "loading"}>
          Warm session
        </button>
      </section>

      {session ? (
        <section className="card success">
          <h3>Session ready</h3>
          <p>Loaded {session.modelId} in {session.loadMs.toFixed(1)} ms.</p>
          <p>Inputs: {session.inputNames.join(", ") || "n/a"}</p>
          <p>Outputs: {session.outputNames.join(", ") || "n/a"}</p>
        </section>
      ) : null}

      {error ? <section className="card error">{error}</section> : null}
    </main>
  );
}
