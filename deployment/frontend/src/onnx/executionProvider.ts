import type { ExecutionProvider } from "./modelRegistry";

/**
 * Feature-detect which execution providers the current browser can attempt.
 *
 * - `wasm`  : always available (CPU, threaded when cross-origin isolated).
 * - `webgpu`: requires navigator.gpu.
 * - `webnn` : requires navigator.ml; can target an NPU on supported devices.
 *
 * Detection only tells us what is plausible. Session creation still validates
 * the provider, so callers should attempt providers in order and fall back.
 */
export function isProviderAvailable(provider: ExecutionProvider): boolean {
  switch (provider) {
    case "wasm":
      return true;
    case "webgpu":
      return typeof navigator !== "undefined" && "gpu" in navigator;
    case "webnn":
      return typeof navigator !== "undefined" && "ml" in navigator;
    default:
      return false;
  }
}

/**
 * Resolve the ordered list of providers to attempt, keeping the model's
 * preferred order but dropping ones the browser cannot provide. `wasm` is
 * always appended as a final fallback.
 */
export function resolveProviderOrder(preferred: ExecutionProvider[]): ExecutionProvider[] {
  const order: ExecutionProvider[] = [];
  for (const provider of preferred) {
    if (isProviderAvailable(provider) && !order.includes(provider)) {
      order.push(provider);
    }
  }
  if (!order.includes("wasm")) {
    order.push("wasm");
  }
  return order;
}

/**
 * Build the ONNX Runtime executionProviders entry for a given provider.
 * WebNN is configured to prefer an NPU device when available.
 */
export function toOrtProvider(provider: ExecutionProvider): unknown {
  if (provider === "webnn") {
    return { name: "webnn", deviceType: "npu", powerPreference: "low-power" };
  }
  return provider;
}
