# Models

Place model definitions here.

Current model files:

- `swin_transformer.py` - reusable Swin-Tiny/Swin-Small classifier factories for the aerial scene classifier.
- `timm_classifier.py` - generic `timm` classifier factory with DINOv2 aliases.
- `resnet/frozen.py` - frozen ResNet18 transfer-learning baseline.
- `resnet/finetune.py` - ResNet18 last-block fine-tuning helper.
- `resnet18_scratch.py` - ResNet18 with random initialization for from-scratch comparison runs.
- `convnext_scratch.py` - ConvNeXtV2 Tiny with random initialization for from-scratch comparison runs.

Root-level `resnet18_frozen.py` and `resnet18_finetune.py` remain compatibility
wrappers for older imports.

Suggested files for later phases:

- `baseline_cnn.py`
- `transfer_resnet.py`
- `transfer_efficientnet.py`
- `transfer_mobilenet.py`

Keep the final model architecture reusable by both training code and the deployment app.

Note: Swin and generic timm factories require the `timm` package when instantiated.

## Current DINOv2 Model Options

`timm_classifier.py` exposes these DINOv2 aliases:

- `dinov2-small` -> `vit_small_patch14_dinov2.lvd142m`
- `dinov2-small-reg` -> `vit_small_patch14_reg4_dinov2.lvd142m`
- `dinov2-base` -> `vit_base_patch14_dinov2.lvd142m`
- `dinov2-base-reg` -> `vit_base_patch14_reg4_dinov2.lvd142m`

Use `dinov2-small` first for a deployable self-supervised transformer baseline.

## Current ResNet Model

`resnet/frozen.py` builds the first transfer-learning baseline:

- Loads pretrained ResNet18.
- Freezes the feature extractor.
- Replaces the final layer with a 4-class classifier.

This is the first no-augmentation baseline before later augmentation or fine-tuning experiments.

`resnet/finetune.py` builds the controlled fine-tuning comparison:

- Loads pretrained ResNet18.
- Replaces the final layer with a 4-class classifier.
- Keeps early ResNet layers frozen.
- Unfreezes only `layer4` and `fc`.
- Provides separate optimizer parameter groups for conservative backbone tuning and faster classifier learning.

`resnet18_scratch.py` builds the diagnostic non-pretrained comparison:

- Creates ResNet18 with `weights=None`.
- Replaces the final layer with a 4-class classifier.
- Keeps all layers trainable because the starting features are random.
- Should be compared against the pretrained strict-split ResNet18 runs, not treated as the main deployment model unless it is competitive.

`convnext_scratch.py` builds the non-pretrained ConvNeXtV2 comparison:

- Creates ConvNeXtV2 Tiny through `timm` with `pretrained=False`.
- Replaces the classifier with a 4-class output head.
- Keeps all layers trainable because the starting weights are random.
- Should be compared against local pretrained ConvNeXtV2 artifacts, not treated as the main deployment model unless it is competitive.
