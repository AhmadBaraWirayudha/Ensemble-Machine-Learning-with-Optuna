# CNC surface-roughness prediction API.
#
# Trains offline, serves an immutable artifact: this image ships with
# whatever's already in models/saved_models/ at build time (see
# scripts/train_model.py) rather than training on container start. That
# keeps boot fast and deterministic. Mount a volume over /app/models to
# swap in a newly-retrained bundle without rebuilding the image - see
# docker-compose.yml.

FROM python:3.12-slim

WORKDIR /app

# Layer-cache dependency installs separately from app code.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY app/ ./app/
COPY monitoring/ ./monitoring/
COPY scripts/ ./scripts/
COPY data/ ./data/
COPY models/ ./models/

# logs/ and reports/ are created on demand by src/config.py at import
# time, but declaring them here means they exist (and are volume-mountable)
# even before the app has written anything.
RUN mkdir -p logs reports/drift reports/metrics reports/figures

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health').read()" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
