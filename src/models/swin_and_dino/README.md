# Swin and DINO models

Target home for Swin and ViT/DINO model definitions that currently share generic
`timm` classifier infrastructure.

Current candidates to migrate gradually:

- `src/models/swin_transformer.py`
- DINO aliases and shared timm model specs currently in `src/models/timm_classifier.py`

The generic timm helper may stay shared until Swin/DINO/FocalNet ownership is
split cleanly.
