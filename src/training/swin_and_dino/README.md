# Swin and DINO training

Owner-scoped home for Swin and ViT/DINO training entrypoints.

Current canonical owner entrypoint:

- `src.training.swin_and_dino.train_peft_lora` - PEFT/LoRA adapter training for
  DINOv2 and Swin.

Example commands:

```bash
python -m src.training.swin_and_dino.train_peft_lora --family dinov2 --device mps
python -m src.training.swin_and_dino.train_peft_lora --family swin --variant tiny --device mps
```

Legacy full fine-tuning remains at the original paths for now because the timm
helpers are still shared with other model families:

- `src.training.train_swin`
- `src.training.train_timm_classifier`
