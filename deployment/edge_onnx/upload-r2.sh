#!/usr/bin/env sh
set -eu

: "${R2_BUCKET:?Set R2_BUCKET, e.g. csc3109-edge-models}"
: "${R2_PREFIX:=models}"

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)

put_model() {
  src=$1
  dest=$2
  wrangler r2 object put "${R2_BUCKET}/${R2_PREFIX}/${dest}" \
    --file "${ROOT_DIR}/${src}" \
    --content-type application/octet-stream \
    --cache-control "public, max-age=31536000, immutable"
}

put_model \
  "model/hetmcl_lite/onnx/hetmcl_lite_best_stop_int8_qdq.onnx" \
  "hetmcl_lite_best_stop_int8_qdq.onnx"

put_model \
  "model/semantic_guided_cgaf_onnx_int8_fullcalib_minmax_20260616/semantic_guided_cgaf_fft_int8_qdq_fullcalib_minmax.onnx" \
  "semantic_guided_cgaf_fft_int8_qdq_fullcalib_minmax.onnx"

echo "Uploaded ONNX edge models to r2://${R2_BUCKET}/${R2_PREFIX}/"
