# Deprecated Docker Path

The old Caddy/static-frontend Docker path has been retired for the CSC3109 submission deployment.

Use the root `Dockerfile` instead. It builds the React frontend, starts a FastAPI backend, serves the frontend, and exposes `POST /predict` for model inference.
