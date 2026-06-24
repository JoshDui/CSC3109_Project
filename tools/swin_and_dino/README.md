# Swin and DINO tools

This folder owns Swin and ViT/DINO command-line tools, benchmark summaries, and
ONNX export helpers for the PEFT/LoRA deployment path.

## ONNX FP32 export and evaluation

Export merged PEFT/LoRA checkpoints to ONNX:

```bash
uv run python -m tools.swin_and_dino.export_peft_lora_onnx \
  --run-dir model/swin_and_dino/dino/vit_small_patch14_dinov2_lvd142m_lora \
  --output-dir reports/onnx/vit_small_patch14_dinov2_lvd142m_lora \
  --batch-size 1 \
  --opset 18

uv run python -m tools.swin_and_dino.export_peft_lora_onnx \
  --run-dir model/swin_and_dino/swin/swin_tiny_lora \
  --output-dir reports/onnx/swin_tiny_lora \
  --batch-size 1 \
  --opset 18
```

Evaluate the ONNX Runtime artifacts on `data/raw/val`:

```bash
uv run python -m tools.swin_and_dino.evaluate_peft_lora_onnx \
  --export-manifest reports/onnx/vit_small_patch14_dinov2_lvd142m_lora/export_manifest.json \
  --output-dir reports/onnx/vit_small_patch14_dinov2_lvd142m_lora/eval \
  --batch-size 32 \
  --warmup-batches 1

uv run python -m tools.swin_and_dino.evaluate_peft_lora_onnx \
  --export-manifest reports/onnx/swin_tiny_lora/export_manifest.json \
  --output-dir reports/onnx/swin_tiny_lora/eval \
  --batch-size 32 \
  --warmup-batches 1
```

The export manifest stays under `reports/onnx/<run>/` with the evaluation
outputs, while the deployable ONNX files are written under each model run's
`onnx/` folder. The PyTorch dynamo ONNX exporter writes external-data artifacts
for these transformers. Keep the `.onnx` graph and matching `.onnx.data` tensor
file together for deployment.

## Result collation

Collate legacy baselines, Torch PEFT runs, ONNX Runtime runs, per-class metrics,
artifact availability, and the report figure:

```bash
uv run python -m tools.swin_and_dino.collate_peft_lora_results
```

Generated summary artifacts:

```text
reports/tables/swin_dino_peft_lora_summary.csv
reports/tables/swin_dino_peft_lora_summary.json
reports/tables/swin_dino_peft_lora_per_class.csv
reports/tables/swin_dino_peft_lora_artifact_manifest.csv
reports/figures/swin_dino_peft_lora_macro_f1.png
```

Current ONNX Runtime results:

| Run | Accuracy | Macro-F1 | Total ONNX size | Avg batch latency |
| --- | ---: | ---: | ---: | ---: |
| DINOv2 ViT-S/14 LoRA ONNX | 0.9950 | 0.9950 | 87,609,868 bytes | 0.6136 s |
| Swin-Tiny LoRA ONNX | 0.9900 | 0.9900 | 113,949,899 bytes | 0.5974 s |
