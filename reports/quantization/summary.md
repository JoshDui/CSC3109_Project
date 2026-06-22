# Quantization and Efficiency Summary

This folder contains post-training benchmark outputs and exported lower-precision checkpoints.

## Notes

- **ResNet18 INT8** numbers are **CPU** results and are directly comparable against the FP32 CPU baseline.
- **FocalNet, Swin, and DINOv2 FP16** numbers are **CUDA** results and are directly comparable against their FP32 CUDA baselines.
- Transformer-family INT8 was intentionally left out of the main deliverables; this summary follows the safe-first plan.
- A small **dynamic INT8 linear-layer** experiment was run on Swin and DINOv2 as exploratory CPU-only work.

## Main deliverables

| Model | Precision / Method | Accuracy | Macro F1 | Avg batch latency | Size | Artifact |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| ResNet18 frozen | FP32 CPU baseline | 0.9825 | 0.9826 | 0.1613 s | 44.8 MB | `model/resnet18_frozen.pt` |
| ResNet18 frozen | Static INT8 PTQ (CPU) | 0.9825 | 0.9826 | 0.0254 s | 11.3 MB | `reports/quantization/resnet18_int8/resnet18_frozen_int8.pt` |
| FocalNet-Tiny SRF | FP32 CUDA baseline | 0.9950 | 0.9950 | 0.0227 s | 332.2 MB | `model/focalnet_tiny_srf_notebook/final/best_model.pt` |
| FocalNet-Tiny SRF | FP16 CUDA export | 0.9950 | 0.9950 | 0.0131 s | 276.9 MB | `reports/quantization/focalnet_fp16/focalnet_tiny_srf_fp16.pt` |
| Swin-Tiny | FP32 CUDA baseline | 0.9975 | 0.9975 | 0.0279 s | 330.5 MB | `model/swin_tiny/best_model.pt` |
| Swin-Tiny | FP16 CUDA export | 0.9975 | 0.9975 | 0.0109 s | 275.4 MB | `reports/quantization/swin_tiny_fp16/swin_tiny_fp16.pt` |
| DINOv2-S/14 linear probe | FP32 CUDA baseline | 0.9850 | 0.9850 | 0.0221 s | 86.6 MB | `model/vit_small_patch14_dinov2_lvd142m_linear_probe/best_model.pt` |
| DINOv2-S/14 linear probe | FP16 CUDA export | 0.9850 | 0.9850 | 0.0076 s | 43.3 MB | `reports/quantization/dinov2_linear_probe_fp16/dinov2_linear_probe_fp16.pt` |
| DINOv2-S/14 fine-tune | FP32 CUDA baseline | 0.9975 | 0.9975 | 0.0236 s | 259.8 MB | `model/vit_small_patch14_dinov2_lvd142m_finetune/best_model.pt` |
| DINOv2-S/14 fine-tune | FP16 CUDA export | 0.9975 | 0.9975 | 0.0085 s | 216.5 MB | `reports/quantization/dinov2_finetune_fp16/dinov2_finetune_fp16.pt` |

## Exploratory transformer INT8 results

| Model | Method | Accuracy | Macro F1 | Avg batch latency | Size | Verdict |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Swin-Tiny | FP32 CPU baseline | 0.9975 | 0.9975 | 0.5549 s | 330.5 MB | baseline |
| Swin-Tiny | Dynamic INT8 linear layers (CPU) | 0.9950 | 0.9950 | 0.3499 s | 29.1 MB | **viable exploratory result** |
| DINOv2-S/14 fine-tune | FP32 CPU baseline | 0.9975 | 0.9975 | 0.4805 s | 259.8 MB | baseline |
| DINOv2-S/14 fine-tune | Dynamic INT8 linear layers (CPU) | 0.9975 | 0.9975 | 0.6089 s | 23.0 MB | **not recommended: latency regressed** |

Artifacts:

- `reports/quantization/swin_tiny_dynamic_int8/`
- `reports/quantization/dinov2_finetune_dynamic_int8/`