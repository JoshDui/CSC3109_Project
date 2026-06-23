# CLIP evaluation

CLIP-owned evaluation helpers live here during the model-family cleanup.

- `export_onnx_int8_qdq.py` exports the CLIP FFT classifier to ONNX, produces
  an ONNX Runtime static INT8 QDQ artifact, and writes FP32/INT8 evaluation
  summaries under `reports/clip_training/clip_onnx_int8_qdq/`.

The old module path `src.evaluation.export_clip_onnx_int8_qdq` remains as a
compatibility wrapper.
