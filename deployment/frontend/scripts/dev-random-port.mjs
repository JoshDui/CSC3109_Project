import net from "node:net";
import { spawn } from "node:child_process";

const MIN_PORT = 20000;
const MAX_PORT = 60000;
const MAX_ATTEMPTS = 25;

function randomPort() {
  return Math.floor(Math.random() * (MAX_PORT - MIN_PORT + 1)) + MIN_PORT;
}

function isPortFree(port) {
  return new Promise((resolve) => {
    const server = net.createServer();
    server.unref();
    server.on("error", () => resolve(false));
    server.listen({ port, host: "0.0.0.0" }, () => {
      server.close(() => resolve(true));
    });
  });
}

async function choosePort() {
  for (let attempt = 0; attempt < MAX_ATTEMPTS; attempt += 1) {
    const port = randomPort();
    if (await isPortFree(port)) {
      return port;
    }
  }
  throw new Error(`Could not find a free random port after ${MAX_ATTEMPTS} attempts`);
}

const port = await choosePort();
console.log(`Starting Vite on random port ${port}`);

const child = spawn(
  "bunx",
  ["vite", "--host", "0.0.0.0", "--port", String(port), ...process.argv.slice(2)],
  {
    stdio: "inherit",
    env: {
      ...process.env,
      PORT: String(port),
      VITE_PORT: String(port),
    },
  },
);

child.on("exit", (code, signal) => {
  if (signal) {
    process.kill(process.pid, signal);
    return;
  }
  process.exit(code ?? 0);
});
