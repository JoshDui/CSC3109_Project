# Swin and DINO training

Owner-scoped home for Swin and ViT/DINO training entrypoints.

Current canonical owner entrypoint:

- `src.training.swin_and_dino.train_peft_lora` - PEFT/LoRA adapter training for
  DINOv2 and Swin.

Example commands:

```bash
python -m src.training.swin_and_dino.train_peft_lora --family dinov2 --device auto
python -m src.training.swin_and_dino.train_peft_lora --family swin --variant tiny --device auto
```

`--device auto` prefers CUDA when available, then Apple MPS, then CPU. Use
`--device cuda` or `--device mps` to force a specific accelerator.

Legacy full fine-tuning remains at the original paths for now because the timm
helpers are still shared with other model families:

- `src.training.train_swin`
- `src.training.train_timm_classifier`
