import { createReadStream, createWriteStream } from "node:fs";
import { readdir, stat } from "node:fs/promises";
import { join } from "node:path";
import { pipeline } from "node:stream/promises";
import { createBrotliCompress, createGzip, constants } from "node:zlib";

const root = process.argv[2] ?? "dist";
const compressible = /\.(html|js|mjs|css|json|svg|wasm|onnx|ort)$/i;

async function* walk(directory) {
  for (const entry of await readdir(directory, { withFileTypes: true })) {
    const path = join(directory, entry.name);
    if (entry.isDirectory()) {
      yield* walk(path);
    } else if (entry.isFile() && compressible.test(entry.name)) {
      yield path;
    }
  }
}

async function compressBrotli(path) {
  await pipeline(
    createReadStream(path),
    createBrotliCompress({
      params: {
        [constants.BROTLI_PARAM_QUALITY]: 11,
      },
    }),
    createWriteStream(`${path}.br`),
  );
}

async function compressGzip(path) {
  await pipeline(
    createReadStream(path),
    createGzip({ level: 9 }),
    createWriteStream(`${path}.gz`),
  );
}

let files = 0;
let originalBytes = 0;
let compressedBytes = 0;

for await (const path of walk(root)) {
  const before = await stat(path);
  await Promise.all([compressBrotli(path), compressGzip(path)]);
  const [br, gz] = await Promise.all([stat(`${path}.br`), stat(`${path}.gz`)]);
  files += 1;
  originalBytes += before.size;
  compressedBytes += br.size + gz.size;
}

console.log(
  `precompressed ${files} files from ${(originalBytes / 1024).toFixed(1)} KiB ` +
    `to ${(compressedBytes / 1024).toFixed(1)} KiB of br+gz sidecars`,
);
