# Semantic-Guided CG-AF Best Recipe

This is the current best combined recipe for the BF16 W8A8 QAT pipeline. It was selected from prior remote runs on `vaporeon`, where the best LoveDA pretraining run reached about `0.5550` mIoU and the best transfer runs reached `1.0000` macro-F1 on the internal tune split.

## Run identity

Current corrected full-pipeline run:

```text
run_id = semantic_guided_bf16_qat_best_recipe_20260615
screen = semantic_bf16_qat_best_recipe
log    = logs/semantic_guided_bf16_qat_best_recipe_20260615.log
```

## LoveDA pretraining

```text
--amp --amp-dtype bf16
--qat-mode w8a8
--qat-observer-warmup-epochs 1
--qat-freeze-observer-epoch 0

--epochs 30
--batch-size 8
--image-size 512
--lr 1e-4
--weight-decay 0.05

--scheduler cosine
--warmup-epochs 3
--min-lr 0
--encoder-lr-mult 0.3

--early-stopping-patience 8
--early-stopping-min-delta 0

--class-weight-mode inverse_sqrt
--focal-gamma 1.0
```

Rationale:

- Cosine LR and differential encoder LR prevent the pretrained ConvNeXt backbone from being over-updated early.
- `inverse_sqrt` class weights and focal CE improve minority/hard-class segmentation behavior.
- BF16 keeps AMP stable on RTX 3090 without GradScaler.
- W8A8 QAT noise makes the pretrained checkpoint more robust for later quantization evaluation.

## FFT transfer

```text
--fine-tuning-mode fft
--amp --amp-dtype bf16
--qat-mode w8a8

--epochs 30
--batch-size 8
--image-size 512
--lr 1e-4
--weight-decay 0.05

--scheduler cosine
--warmup-epochs 1
--min-lr 0
--encoder-lr-mult 0.25

--early-stopping-patience 8
--early-stopping-min-delta 0
--monitor macro_f1

--freeze-backbone-epochs 3
--focal-gamma 0
```

## PEFT transfer

```text
--fine-tuning-mode peft
--amp --amp-dtype bf16
--qat-mode w8a8

--epochs 30
--batch-size 8
--image-size 512
--lr 1e-4
--weight-decay 0.05

--scheduler cosine
--warmup-epochs 1
--min-lr 0
--encoder-lr-mult 0.25

--early-stopping-patience 8
--early-stopping-min-delta 0
--monitor macro_f1

--freeze-backbone
--focal-gamma 0
```

## Quantization and review artifacts

```text
--quant-modes fp32,awq_w8a8
--mask-export-quant-mode awq_w8a8
--mask-export-max-examples 0
```

Expected exported artifacts:

```text
raw FFT checkpoint
raw PEFT checkpoint
AWQ-style W8A8 FFT checkpoint
AWQ-style W8A8 PEFT checkpoint
AWQ-style W8A8 mask exports and review panels
Jupyter artifact manifest and summary
```

The internal `internal_tune` split has `560` SAM3 pseudo-mask examples (`140` per class). Use it for segmentation agreement, Dice/mIoU against SAM3 pseudo-masks, FFT-vs-PEFT selection, and AWQ-style robustness probes. Do **not** present its `140/class` scene confusion matrices as unseen-set results.

Final-facing confusion matrices should come from the held-out ImageFolder split:

```text
data/raw/val
400 images total
100 images per scene class
scene labels only; no segmentation ground truth and no raw-val mIoU
```

The reproducible final artifact lane is:

```text
FFT BF16/QAT checkpoint
→ FP32 ONNX export
→ ONNX Runtime static INT8 QDQ/PTQ with MinMax train-split calibration
→ canonical raw-validation Torch BF16 / ONNX FP32 / ONNX INT8 QDQ / AWQ-style W8A8 scene-confusion evaluation
→ ONNX-only qualitative panels, Johor OOD case study, AWQ-style vs ONNX INT8 case study
```

The AWQ-style raw-validation row is a PyTorch emulation/proxy for research comparison only; ONNX INT8 QDQ remains the deployment artifact.

## Dropout status

Current explicit dropout is limited to the scene classification head:

```text
scene_dropout = 0.1
```

The LoveDA segmentation pretraining path does not currently apply decoder dropout. Regularization there comes mainly from data augmentation, class weighting/focal loss, AdamW weight decay, differential LR, BF16/QAT noise, and early stopping.

If overfitting remains a problem, test dropout as an ablation rather than changing the best recipe immediately. Suggested first sweep:

```text
scene_dropout: 0.1, 0.2, 0.3
decoder/spatial dropout if implemented: 0.05, 0.1, 0.2
```

Track not only best macro-F1, but also pseudo-mask mIoU/Dice, confidence, calibration, and epochs-to-1.000 macro-F1.
