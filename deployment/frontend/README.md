# Frontend

This folder is reserved for the static React application.

Planned frontend responsibilities:

- Let the user upload an aerial image.
- Read the available models from `public/models/models.json`.
- Run inference locally in the browser with ONNX Runtime Web.
- Show the predicted class and per-class confidence scores.
- Keep the UI focused on the ML result rather than cloud infrastructure.

Large ONNX model binaries should be placed under `public/models/` locally for
build/testing, but they are ignored by Git. The lightweight `models.json` registry
is tracked so the app knows which model files to expect.
