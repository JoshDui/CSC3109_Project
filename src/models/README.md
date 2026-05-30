# Models

Place model definitions here.

Current model files:

- `swin_transformer.py` — reusable Swin-Tiny/Swin-Small classifier factories for the aerial scene classifier.

Suggested files for later phases:

- `baseline_cnn.py`
- `transfer_resnet.py`
- `transfer_efficientnet.py`
- `transfer_mobilenet.py`

Keep the final model architecture reusable by both training code and the Streamlit app.

Note: Swin factories require the optional `timm` package when instantiated.
