# Swin and DINO models

This folder owns Swin and ViT/DINO model helpers that are not shared with other
model families.

## PEFT-LoRA helpers

`peft_lora.py` builds `timm` Swin/DINO classifiers wrapped with Hugging Face
PEFT LoRA adapters. The legacy full fine-tuning builders remain available in
the shared modules, but new parameter-efficient experiments should use these
owner-scoped helpers.

Main public helpers:

- `build_peft_lora_classifier(...)` - constructs a pretrained Swin or DINOv2
  classifier with LoRA adapters and returns model metadata.
- `load_peft_lora_from_run(...)` - reloads a saved adapter run from
  `run_manifest.json`.
- `merge_lora_model(...)` and `save_merged_checkpoint_from_run(...)` - merge the
  adapter into the base FP32 model for self-contained evaluation/export.
- `default_lora_output_dir(...)` - resolves the default `model/*_lora/` output
  path.

Default LoRA targets:

| Family | Target regex | Modules saved normally |
| --- | --- | --- |
| DINOv2 | `.*(attn\.(qkv|proj)|mlp\.fc[12])$` | `head` |
| Swin | `.*(attn\.(qkv|proj)|mlp\.fc[12])$` | `head.fc` |

Current report runs:

- DINOv2 LoRA: `591,364 / 22,221,704` trainable parameters (`2.66%`).
- Swin-Tiny LoRA: `568,324 / 28,090,754` trainable parameters (`2.02%`).
