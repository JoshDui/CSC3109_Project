/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_MODEL_MANIFEST_URL?: string;
  readonly VITE_ORT_WASM_PATHS?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
