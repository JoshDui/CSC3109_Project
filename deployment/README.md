# Deployment

Use this folder for the Streamlit app and Docker deployment files.

Suggested final structure:

```text
deployment/
  streamlit_app/
    app.py
  docker/
    Dockerfile
```

The final app should accept an aerial image and return:

- Predicted class label.
- Confidence score.
- Confidence scores for all classes.

