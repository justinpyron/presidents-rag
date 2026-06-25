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

# Run Streamlit with Cloud Run-compatible settings
CMD ["streamlit", "run", "frontend/app.py", \
     "--server.port=8080", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
