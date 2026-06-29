FROM python:3.12-slim

WORKDIR /app

# requirements.txt is generated in CI (see .github/workflows/build-and-deploy.yml).
# It lists third-party deps and includes `-e .`, which installs this repo as a package.
# For `-e .` to succeed, frontend/, backend/, and db/ must already be copied into the image.
# pyproject.toml is required by `-e .` (defines packages: frontend, backend, db).
COPY requirements.txt pyproject.toml ./
COPY frontend/ frontend/
COPY backend/ backend/
COPY db/ db/
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 8080 for Cloud Run
EXPOSE 8080

# Serve the Dash app via gunicorn. `frontend.dash_app.main:server` is the WSGI
# callable (Flask) exposed by the Dash app. Requests are I/O-bound on the
# retrieval server and model providers, so a single worker with several threads
# handles concurrency well; scale workers/threads to the host's CPU and memory.
CMD ["gunicorn", "frontend.dash_app.main:server", \
     "--bind=0.0.0.0:8080", \
     "--workers=1", \
     "--threads=8", \
     "--timeout=120"]
