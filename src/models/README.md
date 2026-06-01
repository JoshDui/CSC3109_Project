# Models

Place model definitions here.

Current model files:

- `swin_transformer.py` - reusable Swin-Tiny/Swin-Small classifier factories for the aerial scene classifier.
- `timm_classifier.py` - generic `timm` classifier factory with DINOv2 aliases.
- `resnet18_frozen.py` - frozen ResNet18 transfer-learning baseline.

Suggested files for later phases:

- `baseline_cnn.py`
- `transfer_resnet.py`
- `transfer_efficientnet.py`
- `transfer_mobilenet.py`

Keep the final model architecture reusable by both training code and the Streamlit app.

Note: Swin and generic timm factories require the `timm` package when instantiated.

## Current DINOv2 Model Options

`timm_classifier.py` exposes these DINOv2 aliases:

- `dinov2-small` -> `vit_small_patch14_dinov2.lvd142m`
- `dinov2-small-reg` -> `vit_small_patch14_reg4_dinov2.lvd142m`
- `dinov2-base` -> `vit_base_patch14_dinov2.lvd142m`
- `dinov2-base-reg` -> `vit_base_patch14_reg4_dinov2.lvd142m`

Use `dinov2-small` first for a deployable self-supervised transformer baseline.

## Current ResNet Model

`resnet18_frozen.py` builds the first transfer-learning baseline:

- Loads pretrained ResNet18.
- Freezes the feature extractor.
- Replaces the final layer with a 4-class classifier.

This is the first no-augmentation baseline before later augmentation or fine-tuning experiments.
