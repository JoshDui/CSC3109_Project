# Swin and DINO evaluation

This folder owns Swin and ViT/DINO evaluation code that is specific to the
parameter-efficient pipeline.

## PEFT-LoRA held-out evaluation

Evaluate a saved LoRA adapter run on the official held-out split:

```bash
uv run python -m src.evaluation.swin_and_dino.evaluate_peft_lora \
  --run-dir model/vit_small_patch14_dinov2_lvd142m_lora \
  --output-dir reports/vit_small_patch14_dinov2_lvd142m_lora_eval

uv run python -m src.evaluation.swin_and_dino.evaluate_peft_lora \
  --run-dir model/swin_tiny_lora \
  --output-dir reports/swin_tiny_lora_eval
```

The evaluator loads `run_manifest.json`, restores the PEFT adapter, applies the
recorded preprocessing, and writes:

```text
metrics.json
confusion_matrix.png
predictions.csv
```

Current held-out PEFT results on `data/raw/val`:

| Run | Accuracy | Macro-F1 | Errors |
| --- | ---: | ---: | ---: |
| DINOv2 ViT-S/14 LoRA | 0.9950 | 0.9950 | 2 / 400 |
| Swin-Tiny LoRA | 0.9900 | 0.9900 | 4 / 400 |

Legacy Swin/DINO evaluators remain available at their existing paths:

- `src/evaluation/evaluate_swin.py`
- `src/evaluation/evaluate_timm_classifier.py`
