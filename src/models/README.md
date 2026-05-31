# Models

Place model definitions here.

Current model files:

- `swin_transformer.py` - reusable Swin-Tiny/Swin-Small classifier factories for the aerial scene classifier.
- `resnet18_frozen.py` - frozen ResNet18 transfer-learning baseline.

Suggested files for later phases:

- `baseline_cnn.py`
- `transfer_resnet.py`
- `transfer_efficientnet.py`
- `transfer_mobilenet.py`

Keep the final model architecture reusable by both training code and the Streamlit app.

Note: Swin factories require the optional `timm` package when instantiated.

## Current ResNet Model

`resnet18_frozen.py` builds the first transfer-learning baseline:

- Loads pretrained ResNet18.
- Freezes the feature extractor.
- Replaces the final layer with a 4-class classifier.

This is the first no-augmentation baseline before later augmentation or fine-tuning experiments.
