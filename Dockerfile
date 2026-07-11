# syntax=docker/dockerfile:1

FROM oven/bun:1.3.14-alpine AS frontend-build

WORKDIR /frontend
ARG VITE_ONNX_MODEL_BASE_URL=/edge-models/models
ENV VITE_ONNX_MODEL_BASE_URL=$VITE_ONNX_MODEL_BASE_URL
COPY deployment/frontend/package.json deployment/frontend/bun.lock ./
RUN bun install --frozen-lockfile

COPY deployment/frontend/index.html ./index.html
COPY deployment/frontend/tsconfig.json ./tsconfig.json
COPY deployment/frontend/vite.config.ts ./vite.config.ts
COPY deployment/frontend/public ./public
COPY deployment/frontend/src ./src

RUN bun run build

FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY deployment/backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r ./backend/requirements.txt

COPY deployment/backend ./backend
COPY model/hetmcl_lite/onnx/hetmcl_lite_best_stop_int8_qdq.onnx ./backend/models/hetmcl_lite_best_stop_int8_qdq.onnx
COPY model/semantic_guided_cgaf_onnx_int8_fullcalib_minmax_20260616/semantic_guided_cgaf_fft_int8_qdq_fullcalib_minmax.onnx ./backend/models/semantic_guided_cgaf_fft_int8_qdq_fullcalib_minmax.onnx
COPY --from=frontend-build /frontend/dist ./frontend/dist

WORKDIR /app/backend
EXPOSE 8080

CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
